# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.
It generates the OHLCV data, executes the trades in a loop but delegates the trade strategy to *TradingStrategy*.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV
using EnvConfig, Ohlcv, TradingStrategy, CryptoXch, Classify

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
    messagelog  # fileid   # only bookkeeping for debugging
    lastbuy::Dict  # Dict(base, DateTime) required to buy in > Minute(tradegapminutes) time gaps
    lastsell::Dict  # Dict(base, DateTime) required to sell in > Minute(tradegapminutes) time gaps
    function TradeCache(; bases=[], startdt=nothing, enddt=nothing, classifier=Classify.Classifier001(), tradegapminutes=5, messagelog=nothing)
        startdt = isnothing(startdt) ? nothing : floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        xc = CryptoXch.XchCache(bases, isnothing(startdt) ? nothing : startdt - Minute(Classify.requiredminutes(classifier)), enddt)
        Classify.preparebacktest!(classifier, CryptoXch.ohlcv(xc))
        new(xc, classifier, startdt, nothing, enddt, tradegapminutes, messagelog, Dict(), Dict())
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


"check all orders have a loaded base - if not add or warning"
function ensureorderbase!(cache::TradeCache, oo::AbstractDataFrame)
    if size(oo, 1) == 0
        return
    end
    bases = unique(DataFrame(CryptoXch.basequote.(oo.symbol))[!, :basecoin])
    missingbases = setdiff(bases, keys(cache.xc.bases))
    for base in missingbases
        CryptoXch.addbase!(cache.xc, base, cache.currentdt, cache.enddt)
    end
end

"Returns a Dict(base, InvestProposal)"
function tradingadvice(cache::TradeCache)
    ad = Dict()
    for (base, ohlcv) in CryptoXch.baseohlcvdict(cache.xc)
        ad[base] = Classify.advice(cache.cls, ohlcv, ohlcv.ix)
    end
    return ad
end

MAKER_CORRECTION = 0.0005
makerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? MAKER_CORRECTION : -MAKER_CORRECTION))
TAKER_CORRECTION = 0.0001
takerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? -TAKER_CORRECTION : TAKER_CORRECTION))
currentprice(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix]

"iterate through all orders and adjust or create new order"
function trade!(cache::TradeCache, base, tp::Classify.InvestProposal, openorders::AbstractDataFrame, assets::AbstractDataFrame)
    sym = CryptoXch.symbolusdt(base)
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
    asset = tp == sell ? base : EnvConfig.cryptoquote
    totalusdt = sum(assets.usdtvalue)
    assets = assets[assets.coin .== asset, :]
    assetfree = sum(assets.free)
    syminfo = CryptoXch.minimumqty(cache.xc, sym)
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    dtnow = Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)]
    price = currentprice(ohlcv)
    minimumbasequantity = max(syminfo.minbaseqty, syminfo.minquoteqty/price)
    # limitprice = makerfeeprice(ohlcv, tp)  # preliminary check shows worse number for makerfeeprice approach -> stick to takerfee
    limitprice = takerfeeprice(ohlcv, tp)
    basequantity = 2 * minimumbasequantity  # have a quantity strategy: for now 2*minimum, later a percentage but above minimum
    basefree = tp == sell ? assetfree : assetfree / limitprice
    basequantity = basefree - basequantity < minimumbasequantity ? basefree : basequantity  # if remaining free is below minimum then add it to quantity
    basequantity = basequantity > basefree ? basefree : basequantity
    if tp == sell
        if size(oo, 1) == 1
            if oo[1, :side] == "Buy"
                oid = CryptoXch.cancelorder(cache.xc, base, oo[1, :orderid])
                println("$(tradetime(cache)) cancel $base buy order with oid $oid, total USDT=$(totalusdt)")
            else  # open order on Sell side
                if basefree < syminfo.minbaseqty
                    basequantity = basefree + oo[1, :baseqty]
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice, basequantity=basequantity)
                    println("$(tradetime(cache)) change $base sell order with oid $oid to limitprice=$limitprice and basequantity=$basequantity, total USDT=$(totalusdt)")
                else
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice)
                    println("$(tradetime(cache)) change $base sell order with oid $oid to limitprice=$limitprice, total USDT=$(totalusdt)")
                end
            end
        elseif (basequantity >= minimumbasequantity) && (!(base in keys(cache.lastsell)) || (cache.lastsell[base] + Minute(cache.tradegapminutes) <= dtnow))
            cache.lastsell[base] = dtnow
            oid = CryptoXch.createsellorder(cache.xc, base; limitprice=limitprice, basequantity=basequantity)
            println("$(tradetime(cache)) created $base sell order with oid $oid, limitprice=$limitprice and basequantity=$basequantity, total USDT=$(totalusdt)")
        end
    elseif tp == buy
        if size(oo, 1) == 1
            if oo[1, :side] == "Sell"
                oid = CryptoXch.cancelorder(cache.xc, base, oo[1, :orderid])
                println("$(tradetime(cache)) cancel $base sell order with oid $oid, total USDT=$(totalusdt)")
            else  # open order on Buy side
                if basefree < syminfo.minquoteqty
                    basequantity = basefree + oo[1, :baseqty]
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice, basequantity=basequantity)
                    println("$(tradetime(cache)) change $base buy order with oid $oid to limitprice=$limitprice and basequantity=$basequantity, total USDT=$(totalusdt)")
                else
                    oid = (CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice))
                    println("$(tradetime(cache)) change $base buy order with oid $oid to limitprice=$limitprice , total USDT=$(totalusdt)")
                end
            end
        elseif (basequantity >= minimumbasequantity) && (!(base in keys(cache.lastbuy)) || (cache.lastbuy[base] + Minute(cache.tradegapminutes) <= dtnow))
            cache.lastbuy[base] = dtnow
            oid = CryptoXch.createbuyorder(cache.xc, base; limitprice=limitprice, basequantity=basequantity)
            println("$(tradetime(cache)) created $base buy order with oid $oid, limitprice=$limitprice and basequantity=$basequantity, total USDT=$(totalusdt)")
        end
    end
end

function Base.iterate(cache::TradeCache, currentdt=nothing)
    if isnothing(currentdt)
        ov = CryptoXch.ohlcv(cache.xc)
        currentdt = isnothing(cache.startdt) ? minimum([Ohlcv.dataframe(o).opentime[begin] for o in ov]) +  - Minute(Classify.requiredminutes(cache.cls)) : cache.startdt
    end
    # println("\rcurrentdt=$(string(currentdt)) cache.enddt=$(string(cache.enddt)) ")
    if !isnothing(cache.enddt) && (currentdt > cache.enddt)
        return nothing
    end
    CryptoXch.setcurrenttime!(cache.xc, currentdt)
    cache.currentdt = currentdt
    return cache, currentdt + Minute(1)
end

tradetime(cache::TradeCache) = Dates.format(cache.currentdt, EnvConfig.datetimeformat)

"""
**`tradeloop`** has to
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)

"""
function tradeloop(cache::TradeCache)
    # TODO add hooks to enable coupling to the cockpit visualization
    @info "$(EnvConfig.now()): trading bases=$(CryptoXch.bases(cache.xc)) period=$(isnothing(cache.startdt) ? "start canned" : cache.startdt) enddt=$(isnothing(cache.enddt) ? "start canned" : cache.enddt)"
    @info "$(EnvConfig.now()): trading with open orders $(CryptoXch.getopenorders(cache.xc))"
    @info "$(EnvConfig.now()): trading with assets $(CryptoXch.balances(cache.xc))"
    for base in keys(cache.xc.bases)
        syminfo = CryptoXch.minimumqty(cache.xc, CryptoXch.symbolusdt(base))
        @info "$syminfo"
    end
    try
        assets = CryptoXch.portfolio!(cache.xc)
        println("$(EnvConfig.now()) assets: $assets")
        oo = CryptoXch.getopenorders(cache.xc)
        println("$(EnvConfig.now()) open orders: $oo)")
        for c in cache
            oo = CryptoXch.getopenorders(cache.xc)
            assets = CryptoXch.portfolio!(cache.xc)
            ensureorderbase!(cache, oo)
            advice = tradingadvice(cache)
            print("\r$(tradetime(cache)): total USDT=$(sum(assets.usdtvalue)) trading=$([k for k in keys(advice)]) ")
            for (base, tp) in advice
                trade!(cache, base, tp, oo, assets)
            end
            #TODO low prio: for closed orders check fees
            #TODO low prio: aggregate orders and transactions in bookkeeping

        end
    # catch ex
    #     showerror(stdout, ex, backtrace())
    #     if isa(ex, InterruptException)
    #         println("Ctrl+C pressed by tradeloop")
    #     end
    finally
        println("finished trading core loop")
        @info "$(EnvConfig.now()): closed orders log" cache.xc.closedorders
        @info "$(EnvConfig.now()): open orders log" cache.xc.orders
        assets = CryptoXch.portfolio!(cache.xc)
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
