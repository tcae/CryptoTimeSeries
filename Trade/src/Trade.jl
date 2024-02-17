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
    orderlogdf::DataFrame  # only bookkeeping for debugging
    lastbuy::Dict  # Dict(base, DateTime) required to buy in > Minute(tradegapminutes) time gaps
    lastsell::Dict  # Dict(base, DateTime) required to sell in > Minute(tradegapminutes) time gaps
    investopendf::DataFrame  # only bookkeeping about percentage gain without performance impact
    investcloseddf::DataFrame  # only bookkeeping archive about percentage gain without performance impact
    function TradeCache(; bases=[], startdt=nothing, enddt=nothing, classifier=Classify.Classifier001(), tradegapminutes=5, messagelog=nothing)
        startdt = isnothing(startdt) ? nothing : floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        xc = CryptoXch.XchCache(bases, isnothing(startdt) ? nothing : startdt - Minute(Classify.requiredminutes(classifier)), enddt)
        Classify.preparebacktest!(classifier, CryptoXch.ohlcv(xc))
        # oldf = DataFrame(usdttotal=[], asset=[], afree=[], alocked=[], oid=[], action=[], base=[], baseqty=[], limitprice=[], orderUSDTvalue=[])
        oldf = DataFrame(usdttotal=Float32[], asset=String[], afree=Float32[], alocked=Float32[], oid=String[], action=String[], base=String[], baseqty=Float32[], limitprice=Float32[], orderUSDTvalue=Float32[])
        investopendf = DataFrame(buyoid=String[], buyprice=Float32[], base=String[], baseqty=Float32[], selloid=String[], sellprice=Float32[])
        investcloseddf = DataFrame(buyoid=String[], buyprice=Float32[], base=String[], baseqty=Float32[], selloid=String[], sellprice=Float32[])
        new(xc, classifier, startdt, nothing, enddt, tradegapminutes, messagelog, oldf, Dict(), Dict(), investopendf, investcloseddf)
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


function sleepuntilnextminute(lastdt)
    enddt = floor(Dates.now(Dates.UTC), Dates.Minute)
    if lastdt == enddt
        nowdt = Dates.now(Dates.UTC)
        nextdt = lastdt + Dates.Minute(1)
        period = Dates.Millisecond(nextdt - floor(nowdt, Dates.Millisecond))
        sleepseconds = floor(period, Dates.Second)
        sleepseconds = Dates.value(sleepseconds) + 1
        @info "trade loop sleep seconds: $sleepseconds"
        sleep(sleepseconds)
        enddt = floor(Dates.now(Dates.UTC), Dates.Minute)
    end
    return enddt
end



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

MISSINGROW = (buyoid="?", buyprice=0.0f0, base="?", baseqty=0.0f0, selloid="?", sellprice=0.0f0)

function nextsellinvest(cache::TradeCache, base)
    for iodix in eachindex(cache.investopendf.base)
        if (cache.investopendf[iodix, :base] == base) && (cache.investopendf[iodix, :buyprice] > 0.0f0)  # buyprice > 0 if Filled
            return (ix=iodix, row=NamedTuple(cache.investopendf[iodix, :]))
        end
    end
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    dtnow = Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)]
    @warn "missing investment row with base=$base and buyprice > 0, i.e. buy order is filled"
    return (ix=nothing, row=MISSINGROW)
end

function oidinvest(cache::TradeCache, oid, oidcol)
    iodix = findfirst(x -> x == oid, cache.investopendf[!, oidcol])
    return isnothing(iodix) ? (ix=nothing, row=MISSINGROW) : (ix=iodix, row=NamedTuple(cache.investopendf[iodix, :]))
end

selloidinvest(cache::TradeCache, selloid) = oidinvest(cache, selloid, "selloid")
buyoidinvest(cache::TradeCache, buyoid) = oidinvest(cache, buyoid, "buyoid")

function updateinvest!(cache::TradeCache, iodix, iodrow)
    # iodrow = (buyoid=buyoid, buyprice=buyprice, base=base, baseqty=baseqty, selloid=selloid, sellprice=sellprice)
    if isnothing(iodix) && !isnothing(iodrow)
        push!(cache.investopendf, iodrow)
    elseif isnothing(iodrow)
        deleteat!(cache.investopendf, iodix)
    else
        cache.investopendf[iodix, :] = iodrow
    end
end

function cleanupinvest!(cache::TradeCache, oo::AbstractDataFrame, buyselect::Bool)
    # println("cleanupinvest! full buyselect=$buyselect investopendf=$(cache.investopendf[!, buyselect ? :buyoid : :selloid])")
    filledoid = setdiff(cache.investopendf[!, buyselect ? :buyoid : :selloid], ["?"])
    filledoid = setdiff(filledoid, oo[!, :orderid])
    if length(filledoid) > 0
        for oid in filledoid
            order = CryptoXch.getorder(cache.xc, oid)
            if isnothing(order)
                @warn "cleanupinvest! error: no order with id $oid in XchCache"
            else
                iodix, iodrow = oidinvest(cache, oid, buyselect ? :buyoid : :selloid)
                if isnothing(iodix)
                    @warn "cleanupinvest! error: no order with id $oid in investopendf[!, $(buyselect ? "buyoid" : "selloid")])"
                else
                    if buyselect
                        cache.investopendf[iodix, :] = (iodrow..., buyprice=order.avgprice)
                    else
                        cache.investopendf[iodix, :] = (iodrow..., sellprice=order.avgprice)
                        push!(cache.investcloseddf, cache.investopendf[iodix, :])
                        deleteat!(cache.investopendf, iodix)
                    end
                end
            end
        end
    end
end

function cleanupinvest!(cache::TradeCache, oo::AbstractDataFrame)
    if size(cache.investopendf, 1) == 0
        return
    end
    oo = oo[CryptoXch.openstatus(oo[!, :status]), :]
    cleanupinvest!(cache, oo, true)
    cleanupinvest!(cache, oo, false)
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
            @info "cancelorder($base, $(oo[ix, :side]), $(oo[ix, :orderid]))"
        end
        return
    end
    if size(oo, 1) > 1
        @error "more than 1 open order for base $base"
    end
    totalusdt = sum(assets.usdtvalue)
    asset = tp == sell ? base : EnvConfig.cryptoquote
    assets = assets[assets.coin .== asset, :]
    assetfree = sum(assets.free)
    locked = sum(assets.locked)
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
                iodix = buyoidinvest(cache, oo[1, :orderid])[1]
                oid = CryptoXch.cancelorder(cache.xc, base, oo[1, :orderid])
                basequantity = oo[1, :baseqty]
                push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="cancelbuy", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
                updateinvest!(cache, iodix, nothing)
                # @info "Total USDT=$(totalusdt) $((isnothing(oid) ? "nothing" : oid)) = cancelorder($base, $(oo[1, :side]), $(oo[1, :orderid]))"
            else  # open order on Sell side
                iodix, iodrow = selloidinvest(cache, oo[1, :orderid])
                if basefree < syminfo.minbaseqty
                    basequantity = basefree + oo[1, :baseqty]
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice, basequantity=basequantity)
                    push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="changesell(l,q)", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
                    # @info "Total USDT=$(totalusdt) $(assets[1, :coin]) free=$assetfree locked=$locked $(isnothing(oid) ? "nothing" : oid)=changselleorder($base, $(oo[1, :orderid]); limitprice=$limitprice, basequantity=$basequantity) USDT value=$(limitprice*basequantity)"
                    updateinvest!(cache, iodix, (iodrow..., base=base,  baseqty=basequantity, selloid=oid))
                else
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice)
                    push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="changesell(l)", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
                    # @info "Total USDT=$(totalusdt) $(assets[1, :coin]) free=$assetfree locked=$locked $(isnothing(oid) ? "nothing" : oid)=changesellorder($base, $(oo[1, :orderid]); limitprice=$limitprice)"
                    updateinvest!(cache, iodix, (iodrow..., base=base, selloid=oid))
                end
            end
        elseif (basequantity >= minimumbasequantity) && (!(base in keys(cache.lastsell)) || (cache.lastsell[base] + Minute(cache.tradegapminutes) <= dtnow))
            iodix, iodrow = nextsellinvest(cache, base)
            #TODO take baseqty from buyinvest counterpart
            cache.lastsell[base] = dtnow
            oid = CryptoXch.createsellorder(cache.xc, base; limitprice=limitprice, basequantity=basequantity)
            push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="sell", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
            # @info "Total USDT=$(totalusdt) $(assets[1, :coin]) free=$assetfree locked=$locked $(isnothing(oid) ? "nothing" : oid)=createsellorder($base, limitprice=$limitprice, basequantity=$basequantity)"
            updateinvest!(cache, iodix, (iodrow..., base=base,  baseqty=basequantity, selloid=oid))
        end
    elseif tp == buy
        if size(oo, 1) == 1
            if oo[1, :side] == "Sell"
                iodix = selloidinvest(cache, oo[1, :orderid])[1]
                oid = CryptoXch.cancelorder(cache.xc, base, oo[1, :orderid])
                basequantity = oo[1, :baseqty]
                push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="cancelsell", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
                updateinvest!(cache, iodix, nothing)
                # @info "Total USDT=$(totalusdt) $((isnothing(oid) ? "nothing" : oid)) = cancelorder($base, $(oo[1, :side]), $(oo[1, :orderid]))"
            else  # open order on Buy side
                if basefree < syminfo.minquoteqty
                    iodix, iodrow = buyoidinvest(cache, oo[1, :orderid])
                    basequantity = basefree + oo[1, :baseqty]
                    oid = CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice, basequantity=basequantity)
                    push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="changebuy(l,q)", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
                    # @info "Total USDT=$(totalusdt) $(assets[1, :coin]) free=$assetfree locked=$locked $(isnothing(oid) ? "nothing" : oid)=changebuyorder($base, $(oo[1, :orderid]); limitprice=$limitprice, basequantity=$basequantity)"
                    updateinvest!(cache, iodix, (iodrow..., buyoid=oid, base=base, baseqty=basequantity))
                else
                    oid = (CryptoXch.changeorder(cache.xc, oo[1, :orderid]; limitprice=limitprice))
                    push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="changebuy(l)", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
                    # @info "Total USDT=$(totalusdt) $(assets[1, :coin]) free=$assetfree locked=$locked $(isnothing(oid) ? "nothing" : oid)=changebuyorder($base, $(oo[1, :orderid]); limitprice=$limitprice)"
                end
            end
        elseif (basequantity >= minimumbasequantity) && (!(base in keys(cache.lastbuy)) || (cache.lastbuy[base] + Minute(cache.tradegapminutes) <= dtnow))
            cache.lastbuy[base] = dtnow
            oid = CryptoXch.createbuyorder(cache.xc, base; limitprice=limitprice, basequantity=basequantity)
            push!(cache.orderlogdf, (usdttotal=totalusdt, asset=asset, afree=assetfree, alocked=locked, oid=string(oid), action="buy", base=base, baseqty=basequantity, limitprice=limitprice, orderUSDTvalue=basequantity*limitprice))
            # @info "Total USDT=$(totalusdt) $(assets[1, :coin]) free=$assetfree locked=$locked $(isnothing(oid) ? "nothing" : oid)=createbuyorder($base, limitprice=$limitprice, basequantity=$basequantity)"
            updateinvest!(cache, nothing, (MISSINGROW..., buyoid=oid, base=base,  baseqty=basequantity))
        end
    end
end

function Base.iterate(cache::TradeCache, currentdt=nothing)
    if isnothing(currentdt)
        ov = CryptoXch.ohlcv(cache.xc)
        currentdt = isnothing(cache.startdt) ? minimum([Ohlcv.dataframe(o).opentime[begin] for o in ov]) +  - Minute(Classify.requiredminutes(cache.cls)) : cache.startdt
    end
    if !isnothing(cache.enddt) && (currentdt > cache.enddt)
        return nothing
    end
    CryptoXch.setcurrenttime!(cache.xc, currentdt)
    cache.currentdt = currentdt
    return cache, currentdt + Minute(1)
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
    @info "$(EnvConfig.now()): trading bases=$(CryptoXch.bases(cache.xc)) period=$(isnothing(cache.startdt) ? "start canned" : cache.startdt) enddt=$(isnothing(cache.enddt) ? "start canned" : cache.enddt)"
    @info "$(EnvConfig.now()): trading with open orders $(CryptoXch.getopenorders(cache.xc))"
    @info "$(EnvConfig.now()): trading with assets $(CryptoXch.balances(cache.xc))"
    for base in keys(cache.xc.bases)
        syminfo = CryptoXch.minimumqty(cache.xc, CryptoXch.symbolusdt(base))
        @info "$syminfo"
    end
    for c in cache
        oo = CryptoXch.getopenorders(cache.xc)
        assets = CryptoXch.portfolio!(cache.xc)
        ensureorderbase!(cache, oo)
        cleanupinvest!(cache, oo)
        advice = tradingadvice(cache)
        for (base, tp) in advice
            trade!(cache, base, tp, oo, assets)
        end
        #TODO low prio: for closed orders check fees
        #TODO low prio: aggregate orders and transactions in bookkeeping

        # reportliquidity(cache, nothing)
        # total, free, locked = totalusdtliquidity(cache)
        # println("liquidity portfolio total free: $(round(free;digits=3)) USDT, locked: $(round(locked;digits=3)) USDT, total: $(round(total;digits=3)) USDT")
    end
    println("finished trading core loop")
    @info "$(EnvConfig.now()): orderlog" cache.orderlogdf  # cache.xc.orders
    @info "$(EnvConfig.now()): closed invest log" cache.investcloseddf
    @info "$(EnvConfig.now()): open invest log" cache.investopendf
    assets = CryptoXch.portfolio!(cache.xc)
    totalusdt = sum(assets.usdtvalue)
    println("total USDT = $totalusdt")
    # @info "$(EnvConfig.now()): closed invest log" cache.investcloseddf
    # @info "$(EnvConfig.now()): open invest log" cache.investopendf
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

function tradelooptest(cache::TradeCache)
    for c in cache
        println("$(Dates.now(UTC)) $c  $(CryptoXch.ohlcv(c.xc, "BTC"))")
        # reportliquidity(cache, nothing)
        # total, free, locked = totalusdtliquidity(cache)
        # println("liquidity portfolio total free: $(round(free;digits=3)) USDT, locked: $(round(locked;digits=3)) USDT, total: $(round(total;digits=3)) USDT")
    end
end


end  # module
