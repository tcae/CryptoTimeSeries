# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.
It generates the OHLCV data, executes the trades in a loop but delegates the trade strategy to *TradingStrategy*.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV
using EnvConfig, Ohlcv, TradingStrategy, CryptoXch, Classify, Assets

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
    cls # required to get trade signal
    startdt::Union{Nothing, Dates.DateTime}  # start time back testing; nothing == start of canned data
    currentdt::Union{Nothing, Dates.DateTime}  # current back testing time
    enddt::Union{Nothing, Dates.DateTime}  # end time back testing; nothing == request life data without defined termination
    tradegapminutes::Integer  # required to buy/sell in Minute(tradegapminutes) time gaps  #* DEPRECATED - should stay at 1 will be extended as required
    topx::Integer  # defines how many best candidates are considered for trading
    tradeassetfraction # defines the default trade fraction of total assets  #* DEPRECATED
    maxassetfraction # defines the maximum ration of (a specific asset) / ( total assets) - sell only if this is exceeded
    lastbuy::Dict  # Dict(base, DateTime) required to buy in = Minute(tradegapminutes) time gaps
    lastsell::Dict  # Dict(base, DateTime) required to sell in = Minute(tradegapminutes) time gaps
    function TradeCache(; bases=[], startdt=nothing, enddt=nothing, classifier=Classify.ClassifierSet001(), tradegapminutes=1, topx=50, tradeassetfraction=1/2000, maxassetfraction=0.05)
        startdt = isnothing(startdt) ? nothing : floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        xc = CryptoXch.XchCache(bases, isnothing(startdt) ? nothing : startdt - Minute(Classify.requiredminutes(classifier)), enddt, 100000)
        new(xc, classifier, startdt, nothing, enddt, tradegapminutes, topx, tradeassetfraction, maxassetfraction, Dict(), Dict())
    end
end

function Base.show(io::IO, cache::TradeCache)
    print(io::IO, "TradeCache: startdt=$(cache.startdt) currentdt=$(cache.currentdt) enddt=$(cache.enddt)")
end

ohlcvdf(cache, base) = Ohlcv.dataframe(cache.bd[base].ohlcv)
ohlcv(cache, base) = cache.bd[base].ohlcv
classifier(cache, base) = cache.bd[base].classifier
backtest(cache) = cache.backtestperiod >= Dates.Minute(1)
dummytime() = DateTime("2000-01-01T00:00:00")


significantsellpricechange(tc, orderprice) = abs(tc.sellprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2
significantbuypricechange(tc, orderprice) = abs(tc.buyprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2

"""
looks for the base configuration in the last configuration dataframe to activate it in the current one if those have minimal free quantity
"""
function activatesellonlyconfigs!(cache::TradeCache, bases, prevconfig, assets)
    added = []
    notadded = []
    for base in bases
        if !CryptoXch.validbase(cache.xc, base)
            @info "skipping sell only activation of invalid base $base"
            push!(notadded, base)
            continue
        end
        # ix = findfirst(row -> row.basecoin == base, eachrow(cache.cls.cfg))
        oldbasecfg = Classify.baseclassifieractiveconfigs(prevconfig, base)
        if isnothing(oldbasecfg)
            @warn "unexpected: missing base $base in previous config"
            ix = findfirst(row -> (row.basecoin == base), eachrow(cache.cls.cfg))
            if isnothing(ix)
                @warn "unexpected: missing base $base in current config"
                cfg = Classify.addreplaceconfig!(cache.cls, base, Classify.STDREGRWINDOW, Classify.STDGAINTHRSHLD, true, true)
                if isnothing(cfg)
                    @error "unexpected: failed to add base $base in current config df, current cfg=$cfg"
                end
            else
                aix = findfirst(row -> (row.coin == base), eachrow(assets))
                if !isnothing(aix)
                    price = assets[aix, :usdtprice]
                    syminfo = CryptoXch.minimumqty(cache.xc, CryptoXch.symboltoken(base))
                    minimumbasequantity = 1.01 * max(syminfo.minbaseqty, syminfo.minquoteqty/price) # 1% more to avoid issues by rounding errors
                    if assets[aix, :free] > minimumbasequantity
                        cache.cls.cfg[ix, :active] = true
                        cache.cls.cfg[ix, :sellonly] = true
                        push!(added, base)
                        continue
                    else
                        println("asset coin $base not added due to low asset value $(assets[aix, :free]) versus minimum base quatity $(minimumbasequantity) ")
                    end
                else
                    @error "missing $base in $assets"
                end
            end
        else
            cfg = Classify.addreplaceconfig!(cache.cls, base, oldbasecfg.regrwindow, oldbasecfg.gainthreshold, true, true)
            if isnothing(cfg)
                @error "unexpected: failed to add base $base in current config df, lastconfig=$oldbasecfg, current cfg=$cfg"
            end
        end
        push!(notadded, base)
    end
    return added, notadded
end

# "Returns a Dict(base, InvestProposal)"
# function tradingadvice(cache::TradeCache)
#     ad = Dict()
#     for (base, ohlcv) in CryptoXch.baseohlcvdict(cache.xc)
#         ad[base] = Classify.advice(cache.cls, ohlcv)
#     end
#     return ad
# end

MAKER_CORRECTION = 0.0005
makerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? MAKER_CORRECTION : -MAKER_CORRECTION))
TAKER_CORRECTION = 0.0001
takerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? -TAKER_CORRECTION : TAKER_CORRECTION))
currentprice(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix]

"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, base, tp::Classify.InvestProposal, openorders::AbstractDataFrame, assets::AbstractDataFrame, sellonly)
    sym = CryptoXch.symboltoken(base)
    totalusdt = sum(assets.usdtvalue)
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])
    freebase = sum(assets[assets[!, :coin] .== base, :free])

    cfg = Classify.baseclassifieractiveconfigs(cache.cls, base)
    if isnothing(cfg)
        @error "Classify.baseclassifieractiveconfigs(cache.cls, $base) returned nothing, $(cache.cls)"
    end
    # quotequantity = totalusdt * cache.tradeassetfraction  # target amount to buy or sell - that will be slightly corrected below due to constraints
    quotequantity = cache.maxassetfraction * totalusdt / cfg.medianccbuycnt
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
        sufficientsellbalance = basequantity <= freebase
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        tradegapminutes = basedominatesassets ? 1 : tradegapminutes  # accelerate selloff if basedominatesassets
        if sufficientsellbalance && exceedsminimumbasequantity && (!(base in keys(cache.lastsell)) || (cache.lastsell[base] + Minute(tradegapminutes) <= dtnow))
            oid = CryptoXch.createsellorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true)
            if !isnothing(oid)
                cache.lastsell[base] = dtnow
                println("\r$(tradetime(cache)) created $base sell order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), total USDT=$(totalusdt), tgm=$tradegapminutes, $(sellonly ? "sell only" : "")")
            else
                println("\r$(tradetime(cache)) failed to create $base maker sell order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), total USDT=$(totalusdt), tgm=$tradegapminutes, $(sellonly ? "sell only" : "")")
            end
        end
    elseif (tp == buy) && !sellonly
        basequantity = min(max(quotequantity/price, minimumbasequantity) * price, freeusdt - 0.01 * totalusdt) / price # keep 1% * totalusdt as head room
        sufficientbuybalance = basequantity * price < freeusdt
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if basedominatesassets
            # println("\r$(tradetime(cache)) skip $base buy due to basefraction=$basefraction > maxassetfraction=$(cache.maxassetfraction)")
            return
        end
        if sufficientbuybalance && exceedsminimumbasequantity && (!(base in keys(cache.lastbuy)) || (cache.lastbuy[base] + Minute(tradegapminutes) <= dtnow))
            oid = CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true)
            if !isnothing(oid)
                cache.lastbuy[base] = dtnow
                println("\r$(tradetime(cache)) created $base buy order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), total USDT=$(totalusdt), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
            else
                println("\r$(tradetime(cache)) failed to create $base maker buy order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), total USDT=$(totalusdt), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        end
    end
end

function Base.iterate(cache::TradeCache, currentdt=nothing)
    if isnothing(currentdt)
        ov = CryptoXch.ohlcv(cache.xc)
        currentdt = isnothing(cache.startdt) ? minimum([Ohlcv.dataframe(o).opentime[begin] for o in ov]) + Minute(Classify.requiredminutes(cache.cls) - Minute(1)) : cache.startdt
    else
        currentdt += Minute(1)
    end
    # println("\rcurrentdt=$(string(currentdt)) cache.enddt=$(string(cache.enddt)) ")
    if !isnothing(cache.enddt) && (currentdt > cache.enddt)
        return nothing
    end
    CryptoXch.setcurrenttime!(cache.xc, currentdt)  # also updates bases if current time is > last time of cache
    cache.currentdt = currentdt
    return cache, currentdt
end

tradetime(cache::TradeCache) = "TT" * Dates.format((isnothing(cache.currentdt) ? Dates.now(UTC) : cache.currentdt), EnvConfig.datetimeformat)

function _refreshtradecoins!(cache::TradeCache)
    if isnothing(cache.currentdt) || (size(cache.cls.cfg, 1) == 0)
        if isnothing(cache.enddt) # request for most current data
            Classify.read!(cache.cls, nothing)
        else
            Classify.read!(cache.cls, cache.startdt)
        end
        if (size(cache.cls.cfg, 1) > 0)
            println("assets: $(CryptoXch.portfolio!(cache.xc))")
            CryptoXch.removeallbases(cache.xc)
            CryptoXch.addbases!(cache.xc, cache.cls.cfg[cache.cls.cfg[!, :active], :basecoin], (isnothing(cache.currentdt) ? cache.startdt : cache.currentdt) - Minute(Classify.requiredminutes(cache.cls)), cache.enddt)
            return
        else
            println("no read of config")
        end
    end
    if isnothing(cache.currentdt) || (Time(cache.currentdt) == Time("04:00:00"))
        oo = CryptoXch.getopenorders(cache.xc)
        assets = CryptoXch.portfolio!(cache.xc)
        println("assets: $assets")
        buyorders = oo[oo[!, :side] .== "Buy", :]
        for order in eachrow(buyorders)
            oid = CryptoXch.cancelorder(cache.xc, CryptoXch.basequote(order.symbol).basecoin, order.orderid)
            (order.orderid != oid) && @warn "cancelorder of order $order failed"
        end
        assetbases = setdiff(assets[!, :coin], CryptoXch.basestablecoin)
        oldcfg = cache.cls.cfg
        # enforce to add existing asset coins to evalation
        topxdf = Classify.train!(cache.cls, cache.xc, cache.topx, Day(10), (isnothing(cache.currentdt) ? cache.startdt : cache.currentdt), true, assetbases)
        # now identify all assetbases that are not in topX, register them as sell only and activate their config
        sellonlybases = setdiff(assetbases, topxdf[!, :basecoin])
        added, notadded = activatesellonlyconfigs!(cache, sellonlybases, oldcfg, assets) # but only those with minimal free assets

        if isnothing(cache.enddt) # nothing = for most current data
            Classify.write(cache.cls, nothing)
        else
            println("cache.startdt=$(cache.startdt), cache.currentdt=$(cache.currentdt)")
            Classify.write(cache.cls, (isnothing(cache.currentdt) ? cache.startdt : cache.currentdt))
        end

        CryptoXch.removeallbases(cache.xc)
        CryptoXch.addbases!(cache.xc, cache.cls.cfg[cache.cls.cfg[!, :active], :basecoin], (isnothing(cache.currentdt) ? cache.startdt : cache.currentdt) - Minute(Classify.requiredminutes(cache.cls)), cache.enddt)

        println("$(tradetime(cache))  sell only=$added (excluding=$notadded), assetbases=$assetbases, active coin trading config: $(cache.cls.cfg[cache.cls.cfg[!, :active] .== true, :])")
    end
end

"""
**`tradeloop`** has to
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)

"""
function tradeloop(cache::TradeCache)
    # TODO add hooks to enable coupling to the cockpit visualization
    # @info "$(EnvConfig.now()): trading bases=$(CryptoXch.bases(cache.xc)) period=$(isnothing(cache.startdt) ? "start canned" : cache.startdt) enddt=$(isnothing(cache.enddt) ? "start canned" : cache.enddt)"
    # @info "$(EnvConfig.now()): trading with open orders $(CryptoXch.getopenorders(cache.xc))"
    # @info "$(EnvConfig.now()): trading with assets $(CryptoXch.balances(cache.xc))"
    # for base in keys(cache.xc.bases)
    #     syminfo = CryptoXch.minimumqty(cache.xc, CryptoXch.symboltoken(base))
    #     @info "$syminfo"
    # end
    try
        _refreshtradecoins!(cache::TradeCache)
        for c in cache
            activebases = cache.cls.cfg[cache.cls.cfg[!, :active], :basecoin]
            oo = CryptoXch.getopenorders(cache.xc)
            for ooe in eachrow(oo)  # all orders to be cancelled because amending maker orders will lead to rejections and "Order does not exist" returns
                if CryptoXch.openstatus(ooe.status)
                    CryptoXch.cancelorder(cache.xc, CryptoXch.basequote(ooe.symbol).basecoin, ooe.orderid)
                end
            end
            assets = CryptoXch.portfolio!(cache.xc)
            sellonlybases = cache.cls.cfg[cache.cls.cfg[!, :sellonly], :basecoin]
            for base in activebases
                ohlcv = CryptoXch.ohlcv(cache.xc, String(base))
                if (size(ohlcv.df, 1) == 0) || (cache.currentdt != ohlcv.df[ohlcv.ix, :opentime])
                    if isnothing(cache.enddt)
                        @warn "unexpected missing ohlcv data $ohlcv at $(cache.currentdt)"
                    else
                        continue  # can happen in backtest because coin candidates are collected from current data
                    end
                end
                tp = Classify.advice(cache.cls, ohlcv)  # Returns a Dict(base, InvestProposal)
                print("\r$(tradetime(cache)): total USDT=$(sum(assets.usdtvalue))")  # trading=$([k for k in advice]) ")
                sellonly = ohlcv.base in sellonlybases
                trade!(cache, ohlcv.base, tp, oo, assets, sellonly)
            end
            _refreshtradecoins!(cache::TradeCache)
            #TODO low prio: for closed orders check fees
            #TODO low prio: aggregate orders and transactions in bookkeeping

        end
    catch ex
        if isa(ex, InterruptException)
            println("Ctrl+C pressed by tradeloop")
        end
        rethrow()
    finally
        println("finished trading core loop")
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
