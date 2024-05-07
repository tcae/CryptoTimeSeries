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
- 2: essential status messages, e.g. load and save messages, are reported
- 3: print debug info
"""
verbosity = 2

"""
*TradeCache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *TradeCache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct TradeCache
    xc::CryptoXch.XchCache  # required to connect to exchange
    tc::TradingStrategy.TradeConfig # maintains the bases to trade and their classifiers
    tradegapminutes::Integer  # required to buy/sell in Minute(tradegapminutes) time gaps - will be extended on low budget to satisfy minimum spending constraints per trade
    topx::Integer  # defines how many best candidates are considered for trading
    tradeassetfraction # defines the default trade fraction of an assets versus total assets value as a function(maxassetfraction, count buy sell coins)
    maxassetfraction # defines the maximum ration of (a specific asset) / ( total assets) - sell only if this is exceeded
    lastbuy::Dict  # Dict(base, DateTime) required to buy in = Minute(tradegapminutes) time gaps
    lastsell::Dict  # Dict(base, DateTime) required to sell in = Minute(tradegapminutes) time gaps
    reloadtimes::Vector{Time}  # provides time info when the portfolio of coin candidates shall be reassessed
    function TradeCache(; tradegapminutes=1, topx=50, maxassetfraction=0.10, xc=CryptoXch.XchCache(true), tc = TradingStrategy.TradeConfig(xc), reloadtimes=[])
        new(xc, tc, tradegapminutes, topx, 1f0, maxassetfraction, Dict(), Dict(), reloadtimes)
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
maxconcurrentbuycount(regrwindow) = regrwindow / 2.0  #* heuristic

"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, basecfg::DataFrameRow, tp::Classify.InvestProposal, assets::AbstractDataFrame)
    stopbuying = true
    executed = noop
    base = basecfg.basecoin
    totalusdt = sum(assets.usdtvalue)
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])
    freebase = sum(assets[assets[!, :coin] .== base, :free])
    # 1800 = likely maxccbuycount
    quotequantity = cache.tradeassetfraction * totalusdt / maxconcurrentbuycount(basecfg.regrwindow)
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    dtnow = Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)]
    price = currentprice(ohlcv)
    minimumbasequantity = CryptoXch.minimumbasequantity(cache.xc, base, price)
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
                executed = sell
                (verbosity > 2) && println("\r$(tradetime(cache)) created $base sell order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, $(basecfg.sellonly ? "sell only" : "")")
            else
                (verbosity > 2) && println("\r$(tradetime(cache)) failed to create $base maker sell order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, $(basecfg.sellonly ? "sell only" : "")")
            end
        end
    elseif !stopbuying && (tp == buy) && !basecfg.sellonly
        basequantity = min(max(quotequantity/price, minimumbasequantity) * price, freeusdt - 0.01 * totalusdt) / price #* keep 1% * totalusdt as head room
        sufficientbuybalance = (basequantity * price < freeusdt) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if basedominatesassets
            (verbosity > 2) && println("\r$(tradetime(cache)) skip $base buy due to basefraction=$(sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt) > maxassetfraction=$(cache.maxassetfraction)")
            return
        end
        if sufficientbuybalance && exceedsminimumbasequantity && (!(base in keys(cache.lastbuy)) || (cache.lastbuy[base] + Minute(tradegapminutes) <= dtnow))
            oid = CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true)
            if !isnothing(oid)
                cache.lastbuy[base] = dtnow
                executed = buy
                (verbosity > 2) && println("\r$(tradetime(cache)) created $base buy order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
            else
                (verbosity > 2) && println("\r$(tradetime(cache)) failed to create $base maker buy order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        end
    end
    return executed
end

tradetime(cache::TradeCache) = CryptoXch.ttstr(cache.xc)
# USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), locked=$(round(Int, sum(assets.locked .* assets.usdtprice))), free=$(round(Int, sum(assets.free .* assets.usdtprice)))")
USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), free=$(round(Int, sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])/sum(assets.usdtvalue)*100))%")

"""
**`tradeloop`** has to
+ get initial TradinStrategy config (if not present at entry) and refresh daily according to `reloadtimes`
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
    if size(cache.tc.cfg, 1) == 0
        assets = CryptoXch.balances(cache.xc)
        (verbosity >= 2) && print("\r$(tradetime(cache)): start loading trading strategy")
        if isnothing(TradingStrategy.read!(cache.tc, cache.xc.startdt))
            (verbosity >= 2) && print("\r$(tradetime(cache)): start reassessing trading strategy")
            TradingStrategy.train!(cache.tc, assets[!, :coin]; datetime=cache.xc.startdt)
        end
        (verbosity > 2) && @info "$(tradetime(cache)) initial trading strategy: $(cache.tc.cfg)"
        # else TradingStrategy was created outside tradeloop - take it as is
    end
    cache.tradeassetfraction = min(cache.maxassetfraction, 1/(count(cache.tc.cfg[!, :buysell]) > 0 ? count(cache.tc.cfg[!, :buysell]) : 1))
    TradingStrategy.timerangecut!(cache.tc, cache.xc.startdt, isnothing(cache.xc.enddt) ? cache.xc.currentdt : cache.xc.enddt)
    try
        for c in cache.xc
            (verbosity > 2) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")
            oo = CryptoXch.getopenorders(cache.xc)
            for ooe in eachrow(oo)  # all orders to be cancelled because amending maker orders will lead to rejections and "Order does not exist" returns
                if CryptoXch.openstatus(ooe.status)
                    CryptoXch.cancelorder(cache.xc, CryptoXch.basequote(ooe.symbol).basecoin, ooe.orderid)
                end
            end
            assets = CryptoXch.portfolio!(cache.xc)
            sellbases = []
            buybases = []
            for basecfg in eachrow(cache.tc.cfg)
                Classify.supplement!(basecfg.classifier)
                ohlcvix = Ohlcv.rowix(basecfg.classifier.ohlcv.df[!, :opentime], cache.xc.currentdt)
                tp = basecfg.classifier.ohlcv.df[ohlcvix, :opentime] == cache.xc.currentdt ? Classify.advice(basecfg.classifier, ohlcvix, basecfg.regrwindow, basecfg.gainthreshold, basecfg.headwindow, basecfg.trendwindow) : Classify.noop
                # print("\r$(tradetime(cache)): $(USDTmsg(assets))")
                executed = trade!(cache, basecfg, tp, assets)
                if executed == buy
                    push!(buybases, basecfg.basecoin)
                elseif executed == sell
                    push!(sellbases, basecfg.basecoin)
                end
            end
            (verbosity >= 2) && print("\r$(tradetime(cache)): $(USDTmsg(assets)), bought: $(buybases), sold: $(sellbases)                                                                    ")
            if Time(cache.xc.currentdt) in cache.reloadtimes  # e.g. [Time("04:00:00"))]
                assets = CryptoXch.portfolio!(cache.xc)
                (verbosity >= 2) && println("\n$(tradetime(cache)): start reassessing trading strategy")
                TradingStrategy.train!(cache.tc, assets[!, :coin]; datetime=cache.xc.currentdt)
                (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.tc.cfg)"
                TradingStrategy.timerangecut!(cache.tc, cache.xc.startdt, isnothing(cache.xc.enddt) ? cache.xc.currentdt : cache.xc.enddt)
                cache.tradeassetfraction = min(cache.maxassetfraction, 1/(count(cache.tc.cfg[!, :buysell]) > 0 ? count(cache.tc.cfg[!, :buysell]) : 1))
            end
            #TODO low prio: for closed orders check fees
            #TODO low prio: aggregate orders and transactions in bookkeeping
        end
    catch ex
        if isa(ex, InterruptException)
            (verbosity >= 0) && println("\nCtrl+C pressed within tradeloop")
        else
            (verbosity >= 2) && println("tradeloop retry")
            (verbosity >= 0) && @error "exception=$ex"
        end
        bt = catch_backtrace()
        for ptr in bt
            frame = StackTraces.lookup(ptr)
            for fr in frame
                if occursin("CryptoTimeSeries", string(fr.file))
                    (verbosity >= 1) && println("fr.func=$(fr.func) fr.linfo=$(fr.linfo) fr.file=$(fr.file) fr.line=$(fr.line) fr.from_c=$(fr.from_c) fr.inlined=$(fr.inlined) fr.pointer=$(fr.pointer)")
                end
            end
        end
    end
    (verbosity >= 2) && println("$(tradetime(cache)): finished trading core loop")
    (verbosity >= 2) && @info "$(EnvConfig.now()): closed orders log" cache.xc.closedorders
    (verbosity >= 2) && @info "$(EnvConfig.now()): open orders log" cache.xc.orders
    assets = CryptoXch.portfolio!(cache.xc)
    (verbosity >= 2) && @info "assets = $assets"
    totalusdt = sum(assets.usdtvalue)
    (verbosity >= 2) && @info "total USDT = $totalusdt"
    #TODO save investlog
end

function tradelooptest(cache::TradeCache)
    for c in cache
        println("$(Dates.now(UTC)) $c  $(CryptoXch.ohlcv(c.xc, "BTC"))")
    end
end


end  # module
