# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.
It generates the OHLCV data, executes the trades in a loop but delegates the trade strategy to *TradingStrategy*.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV
using EnvConfig, Ohlcv, TradingStrategy, CryptoXch, Classify, Assets, Features

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

"""
*TradeCache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *TradeCache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct TradeCache
    xc::CryptoXch.XchCache  # required to connect to exchange
    tc::TradingStrategy.TradeConfig # maintains the bases to trade and their classifiers
    tradegapminutes::Integer  # required to buy/sell in Minute(tradegapminutes) time gaps  #* DEPRECATED - should stay at 1 will be extended as required
    topx::Integer  # defines how many best candidates are considered for trading
    tradeassetfraction # defines the default trade fraction of total assets  #* DEPRECATED
    maxassetfraction # defines the maximum ration of (a specific asset) / ( total assets) - sell only if this is exceeded
    lastbuy::Dict  # Dict(base, DateTime) required to buy in = Minute(tradegapminutes) time gaps
    lastsell::Dict  # Dict(base, DateTime) required to sell in = Minute(tradegapminutes) time gaps
    function TradeCache(; tradegapminutes=1, topx=50, tradeassetfraction=1/2000, maxassetfraction=0.05, xc=CryptoXch.XchCache(true), tc = TradingStrategy.TradeConfig(xc))
        new(xc, tc, tradegapminutes, topx, tradeassetfraction, maxassetfraction, Dict(), Dict())
    end
end

function Base.show(io::IO, cache::TradeCache)
    print(io::IO, "TradeCache: startdt=$(cache.xc.startdt) currentdt=$(cache.xc.currentdt) enddt=$(cache.xc.enddt)")
end

ohlcvdf(cache, base) = Ohlcv.dataframe(cache.bd[base].ohlcv)
ohlcv(cache, base) = cache.bd[base].ohlcv
classifier(cache, base) = cache.bd[base].classifier
backtest(cache) = cache.backtestperiod >= Dates.Minute(1)
dummytime() = DateTime("2000-01-01T00:00:00")


significantsellpricechange(tc, orderprice) = abs(tc.sellprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2
significantbuypricechange(tc, orderprice) = abs(tc.buyprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2


MAKER_CORRECTION = 0.0005
makerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? MAKER_CORRECTION : -MAKER_CORRECTION))
TAKER_CORRECTION = 0.0001
takerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? -TAKER_CORRECTION : TAKER_CORRECTION))
currentprice(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix]

"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, basecfg::DataFrameRow, tp::Classify.InvestProposal, assets::AbstractDataFrame)
    base = basecfg.basecoin
    sym = CryptoXch.symboltoken(base)
    totalusdt = sum(assets.usdtvalue)
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])
    freebase = sum(assets[assets[!, :coin] .== base, :free])
    # quotequantity = totalusdt * cache.tradeassetfraction  # target amount to buy or sell - that will be slightly corrected below due to constraints
    quotequantity = cache.maxassetfraction * totalusdt / basecfg.medianccbuycnt
    syminfo = CryptoXch.minimumqty(cache.xc, sym)
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    dtnow = Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)]
    price = currentprice(ohlcv)
    minimumbasequantity = 1.01 * max(syminfo.minbaseqty, syminfo.minquoteqty/price) # 1% more to avoid issues by rounding errors
    minqteratio = round(Int, (minimumbasequantity * price) / quotequantity)  # if quotequantity target exceeds minimum quote constraints then extend gaps because spending budget is low
    tradegapminutes = minqteratio > 1 ? cache.tradegapminutes * minqteratio : cache.tradegapminutes
    basedominatesassets = (sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt) > cache.maxassetfraction
    if tp == sell
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = freebase <= 2 * minimumbasequantity ? (freebase >= minimumbasequantity ? freebase : minimumbasequantity) : minimumbasequantity
        basequantity = min(max(quotequantity/price, minimumbasequantity), freebase)
        sufficientsellbalance = (basequantity <= freebase) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        tradegapminutes = basedominatesassets ? 1 : tradegapminutes  # accelerate selloff if basedominatesassets
        if sufficientsellbalance && exceedsminimumbasequantity && (!(base in keys(cache.lastsell)) || (cache.lastsell[base] + Minute(tradegapminutes) <= dtnow))
            oid = CryptoXch.createsellorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true)
            if !isnothing(oid)
                cache.lastsell[base] = dtnow
                println("\r$(tradetime(cache)) created $base sell order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, $(basecfg.sellonly ? "sell only" : "")")
            else
                println("\r$(tradetime(cache)) failed to create $base maker sell order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, $(basecfg.sellonly ? "sell only" : "")")
            end
        end
    elseif (tp == buy) && !basecfg.sellonly
        basequantity = min(max(quotequantity/price, minimumbasequantity) * price, freeusdt - 0.01 * totalusdt) / price # keep 1% * totalusdt as head room
        sufficientbuybalance = (basequantity * price < freeusdt) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if basedominatesassets
            # println("\r$(tradetime(cache)) skip $base buy due to basefraction=$basefraction > maxassetfraction=$(cache.maxassetfraction)")
            return
        end
        if sufficientbuybalance && exceedsminimumbasequantity && (!(base in keys(cache.lastbuy)) || (cache.lastbuy[base] + Minute(tradegapminutes) <= dtnow))
            oid = CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true)
            if !isnothing(oid)
                cache.lastbuy[base] = dtnow
                println("\r$(tradetime(cache)) created $base buy order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
            else
                println("\r$(tradetime(cache)) failed to create $base maker buy order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        end
    end
end

tradetime(cache::TradeCache) = CryptoXch.ttstr(cache.xc)
USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), locked=$(round(Int, sum(assets.locked .* assets.usdtprice))), free=$(round(Int, sum(assets.free .* assets.usdtprice)))")

"""
**`tradeloop`** has to
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)
"""
function tradeloop(cache::TradeCache)
    # TODO add hooks to enable coupling to the cockpit visualization
    # @info "$(EnvConfig.now()): trading bases=$(CryptoXch.bases(cache.xc)) period=$(isnothing(cache.xc.startdt) ? "start canned" : cache.xc.startdt) enddt=$(isnothing(cache.xc.enddt) ? "start canned" : cache.xc.enddt)"
    # @info "$(EnvConfig.now()): trading with open orders $(CryptoXch.getopenorders(cache.xc))"
    # @info "$(EnvConfig.now()): trading with assets $(CryptoXch.balances(cache.xc))"
    # for base in keys(cache.xc.bases)
    #     syminfo = CryptoXch.minimumqty(cache.xc, CryptoXch.symboltoken(base))
    #     @info "$syminfo"
    # end
    try
        assets = CryptoXch.portfolio!(cache.xc)
        TradingStrategy.train!(cache.tc, assets[!, :coin]; enddt=cache.xc.startdt)
        TradingStrategy.timerangecut!(cache.tc, cache.xc.startdt, isnothing(cache.xc.enddt) ? cache.xc.currentdt : cache.xc.enddt)
        for c in cache.xc
            (verbosity == 3) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")
            oo = CryptoXch.getopenorders(cache.xc)
            for ooe in eachrow(oo)  # all orders to be cancelled because amending maker orders will lead to rejections and "Order does not exist" returns
                if CryptoXch.openstatus(ooe.status)
                    CryptoXch.cancelorder(cache.xc, CryptoXch.basequote(ooe.symbol).basecoin, ooe.orderid)
                end
            end
            assets = CryptoXch.portfolio!(cache.xc)
            for basecfg in eachrow(cache.tc.cfg)
                Classify.supplement!(basecfg.classifier)
                ohlcvix = Ohlcv.rowix(basecfg.classifier.ohlcv.df[!, :opentime], cache.xc.currentdt)
                tp = basecfg.classifier.ohlcv.df[ohlcvix, :opentime] == cache.xc.currentdt ? Classify.advice(basecfg.classifier, ohlcvix, basecfg.regrwindow, basecfg.gainthreshold, basecfg.model) : Classify.noop
                print("\r$(tradetime(cache)): $(USDTmsg(assets))")  # trading=$([k for k in advice]) ")
                trade!(cache, basecfg, tp, assets)
            end
            if (Time(cache.xc.currentdt) == Time("04:00:00"))
                @info "$(tradetime(cache)): daily loading" cache.tc.cfg
                TradingStrategy.train!(cache.tc, assets[!, :coin]; enddt=cache.xc.currentdt)
                TradingStrategy.timerangecut!(cache.tc, cache.xc.startdt, isnothing(cache.xc.enddt) ? cache.xc.currentdt : cache.xc.enddt)
            end
            #TODO low prio: for closed orders check fees
            #TODO low prio: aggregate orders and transactions in bookkeeping

        end
    catch ex
        if isa(ex, InterruptException)
            println("Ctrl+C pressed by tradeloop")
        end
        rethrow()
    finally
        println("$(tradetime(cache)): finished trading core loop")
        @info "$(EnvConfig.now()): closed orders log" cache.xc.closedorders
        @info "$(EnvConfig.now()): open orders log" cache.xc.orders
        assets = CryptoXch.portfolio!(cache.xc)
        @info "assets = $assets"
        totalusdt = sum(assets.usdtvalue)
        @info "total USDT = $totalusdt"
        #TODO save investlog

        # sf = EnvConfig.logsubfolder()
        # EnvConfig.setlogpath(nothing)
        # of = EnvConfig.logpath("orderlog.csv")
        # EnvConfig.checkbackup(of)
        # println(cache.orderlogdf[1:5,:])
        # @info "$(EnvConfig.now()): orderlog saved in  $of" cache.orderlogdf
        # cache.orderlogdf.asset = string.(cache.orderlogdf.asset)
        # cache.orderlogdf.oid = string.(cache.orderlogdf.oid)
        # cache.orderlogdf.action = string.(cache.orderlogdf.action)
        # cache.orderlogdf.base = string.(cache.orderlogdf.base)
        # CSV.write(cache.orderlogdf, of)  # , decimal='.', delim=';')
        # EnvConfig.setlogpath(sf)

        # Profile.print()
    end
end

function tradelooptest(cache::TradeCache)
    for c in cache
        println("$(Dates.now(UTC)) $c  $(CryptoXch.ohlcv(c.xc, "BTC"))")
        # reportliquidity(cache, nothing)
        # total, free, locked = totalusdtliquidity(cache)
        # println("liquidity portfolio total free: $(round(free;digits=3)) USDT, locked: $(round(locked;digits=3)) USDT, total: $(round(total;digits=3)) USDT")
    end
end


end  # module
