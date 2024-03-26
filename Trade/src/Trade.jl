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

minimumdayusdtvolume = 10000000  # per day = 6944 per minute

"""
*TradeCache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *TradeCache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct TradeCache
    xc::CryptoXch.XchCache  # required to connect to exchange
    cls::Classify.AbstractClassifier  # required to get trade signal
    startdt::Union{Nothing, Dates.DateTime}  # start time back testing; nothing == start of canned data
    currentdt::Union{Nothing, Dates.DateTime}  # current back testing time
    enddt::Union{Nothing, Dates.DateTime}  # end time back testing; nothing == request life data without defined termination
    tradegapminutes::Integer  # required to buy/sell in > Minute(tradegapminutes) time gaps
    topx::Integer  # defines how many best candidates are considered for trading
    tradeassetfraction # defines the default trade fraction of total assets
    maxassetfraction # defines the maximum ration of (a specific asset) / ( total assets) - sell only if this is exceeded
    lastbuy::Dict  # Dict(base, DateTime) required to buy in > Minute(tradegapminutes) time gaps
    lastsell::Dict  # Dict(base, DateTime) required to sell in > Minute(tradegapminutes) time gaps
    function TradeCache(; bases=[], startdt=nothing, enddt=nothing, classifier=Classify.Classifier001(), tradegapminutes=5, topx=20, tradeassetfraction=1/2000, maxassetfraction=0.1)
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


"DEPRECATED: check all orders have a loaded base - if not add or warning"
function ensureohlcvorderbase!(cache::TradeCache, oo::AbstractDataFrame)
    if size(oo, 1) == 0
        return
    end
    bases = unique(DataFrame(CryptoXch.basequote.(oo.symbol))[!, :basecoin]) # basequote returns a named tuple to define the DataFrame columns
    missingbases = setdiff(bases, keys(cache.xc.bases))
    for base in missingbases
        CryptoXch.addbase!(cache.xc, base, cache.currentdt, cache.enddt)
    end
end

"""
looks for the base configuration in the last configuration dataframe to activate it in the current one if those have minimal free quantity
"""
function activatesellonlyconfigs!(cache::TradeCache, bases, prevconfig, assets)
    added = []
    notadded = []
    for base in bases
        # ix = findfirst(row -> row.basecoin == base, eachrow(cache.cls.cfg))
        oldbasecfg = Classify.baseclassifieractiveconfigs(prevconfig, base)
        if isnothing(oldbasecfg)
            @warn "unexpected: missing base $base in previous config"
            ix = findfirst(row -> (row.basecoin == base), eachrow(cache.cls.cfg))
            if isnothing(ix)
                @warn "unexpected: missing base $base in current config"
            else
                aix = findfirst(row -> (row.basecoin == base), eachrow(assets))
                if !isnothing(aix)
                    price = assets[aix, :usdtprice]
                    syminfo = CryptoXch.minimumqty(cache.xc, CryptoXch.symboltoken(base))
                    minimumbasequantity = 1.01 * max(syminfo.minbaseqty, syminfo.minquoteqty/price) # 1% more to avoid issues by rounding errors
                    if assets[aix, :free] > minimumbasequantity
                        cache.cls.cfg[ix, :active] = true
                        cache.cls.cfg[ix, :sellonly] = true
                        push!(added, base)
                        continue
                    end
                else
                    @error "missing $base in $assets"
                end
            end
        else
            cfg = Classify.addreplaceconfig!(cache.cls, base, oldbasecfg.regrwindow, oldbasecfg.gainthreshold, true, true)
            if isnothing(cfg) || (cfg.basecoin != oldbasecfg.basecoin) || (cfg.regrwindow != oldbasecfg.regrwindow) || (cfg.gainthreshold != oldbasecfg.gainthreshold)
                @warn "unexpected: missing base $base in last config, lastconfig=$oldbasecfg, current cfg=$cfg"
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

"iterate through all orders and adjust or create new order"
function trade!(cache::TradeCache, base, tp::Classify.InvestProposal, openorders::AbstractDataFrame, assets::AbstractDataFrame, sellonly)
    sym = CryptoXch.symboltoken(base)
    oo = openorders[sym .== openorders.symbol, :]
    if tp == hold
        for ix in eachindex(oo.symbol)
            CryptoXch.cancelorder(cache.xc, base, oo[ix, :orderid])
        end
        return
    end
    if size(oo, 1) > 1
        @error "more than 1 open order for base $base"
    end
    totalusdt = sum(assets.usdtvalue)
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])
    freebase = sum(assets[assets[!, :coin] .== base, :free])
    quotequantity = totalusdt * cache.tradeassetfraction  # target amount to buy or sell - that will be slightly corrected below due to constraints
    syminfo = CryptoXch.minimumqty(cache.xc, sym)
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    dtnow = Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)]
    price = currentprice(ohlcv)
    minimumbasequantity = 1.01 * max(syminfo.minbaseqty, syminfo.minquoteqty/price) # 1% more to avoid issues by rounding errors
    # limitprice = makerfeeprice(ohlcv, tp)  # preliminary check shows worse number for makerfeeprice approach -> stick to takerfee
    limitprice = takerfeeprice(ohlcv, tp)
    basequantity = max(quotequantity/price, minimumbasequantity)

    # basefree = tp == sell ? sum(assets[assets[!, :coin] .== base, :free]) : freeusdt / limitprice
    # basequantity = (basefree - basequantity) < minimumbasequantity ? basefree : basequantity  # if remaining free is below minimum then add it to quantity
    # basequantity = basequantity > basefree ? basefree : basequantity
    minqteratio = round(Int, (minimumbasequantity * price) / quotequantity)  # if quotequantity target exceeds minimum quote constraints then extend gaps because spending budget is low
    tradegapminutes = minqteratio > 1 ? cache.tradegapminutes * minqteratio : cache.tradegapminutes
    sufficientbuybalance = ((basequantity * limitprice * (1 + cache.xc.feerate + 0.001)) < freeusdt) && (0.01 * totalusdt <= freeusdt)
    sufficientsellbalance = basequantity < freebase
    exceedsminimumbasequantity = basequantity > minimumbasequantity
    basedominatesassets = (sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt) > cache.maxassetfraction
    if tp == sell
        tradegapminutes = basedominatesassets ? 1 : tradegapminutes  # accelerate selloff if basedominatesassets
        if size(oo, 1) == 1
            if oo[1, :side] == "Buy"
                oid = CryptoXch.cancelorder(cache.xc, base, oo[1, :orderid])
                println("\r$(tradetime(cache)) cancel $base buy order with oid $oid, total USDT=$(totalusdt)")
            else  # open order on Sell side
                if freebase < minimumbasequantity
                    basequantity = freebase + oo[1, :baseqty]
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice, basequantity=basequantity)
                    println("\r$(tradetime(cache)) change $base sell order with oid $oid to limitprice=$limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*limitprice), total USDT=$(totalusdt)")
                else
                    basequantity = oo[1, :baseqty]
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice)
                    println("\r$(tradetime(cache)) change $base sell order with oid $oid to limitprice=$limitprice * unchanged basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*limitprice), total USDT=$(totalusdt)")
                end
            end
        elseif sufficientsellbalance && exceedsminimumbasequantity && (!(base in keys(cache.lastsell)) || (cache.lastsell[base] + Minute(tradegapminutes) <= dtnow))
            cache.lastsell[base] = dtnow
            oid = CryptoXch.createsellorder(cache.xc, base; limitprice=limitprice, basequantity=basequantity, maker=true)
            println("\r$(tradetime(cache)) created $base sell order with oid $oid, limitprice=$limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*limitprice), total USDT=$(totalusdt), tgm=$tradegapminutes, $(sellonly ? ", sell only" : "")")
        end
    elseif (tp == buy) && !sellonly
        if basedominatesassets
            # println("\r$(tradetime(cache)) skip $base buy due to basefraction=$basefraction > maxassetfraction=$(cache.maxassetfraction)")
            return
        end
        if size(oo, 1) == 1
            if oo[1, :side] == "Sell"
                oid = CryptoXch.cancelorder(cache.xc, base, oo[1, :orderid])
                println("\r$(tradetime(cache)) cancel $base sell order with oid $oid, total USDT=$(totalusdt)")
            else  # open order on Buy side -> adapt limitprice
                basequantity = oo[1, :baseqty]
                oid = (CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice))
                println("\r$(tradetime(cache)) change $base buy order with oid $oid to limitprice=$limitprice * unchanged basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*limitprice), total USDT=$(totalusdt)")
            end
        elseif sufficientbuybalance && exceedsminimumbasequantity && (!(base in keys(cache.lastbuy)) || (cache.lastbuy[base] + Minute(tradegapminutes) <= dtnow))
            cache.lastbuy[base] = dtnow
            oid = CryptoXch.createbuyorder(cache.xc, base; limitprice=limitprice, basequantity=basequantity, maker=true)
            println("\r$(tradetime(cache)) created $base buy order with oid $oid, limitprice=$limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*limitprice), total USDT=$(totalusdt), tgm=$tradegapminutes")
            if isnothing(oid)
                println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $limitprice * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
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
        topxdf = Classify.best!(cache.cls, cache.xc, cache.topx, Day(10), (isnothing(cache.currentdt) ? cache.startdt : cache.currentdt), true, assetbases)
        # now identify all assetbases that are not in topX, register them as sell only and activate their config
        if isnothing(cache.enddt) # nothing = for most current data
            Classify.write(cache.cls, nothing)
        else
            println("cache.startdt=$(cache.startdt), cache.currentdt=$(cache.currentdt)")
            Classify.write(cache.cls, (isnothing(cache.currentdt) ? cache.startdt : cache.currentdt))
        end
        metadata!(cache.cls.cfg, "tradegapminutes", "$(cache.tradegapminutes)", style=:note)
        sellonlybases = setdiff(assetbases, topxdf[!, :basecoin])
        added, notadded = activatesellonlyconfigs!(cache, sellonlybases, oldcfg, assets) # but only those with minimal free assets

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
    #     showerror(stdout, ex, backtrace())
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
