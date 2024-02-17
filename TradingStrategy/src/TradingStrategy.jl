"""
*TradingStrategy* assesses trade chances of a crypto base.
Trade chances are characterized by a buy and sell limit price as well as a stop loss price.
A probabiity is also provided but the current approach considers it in scope of TradeStrategy to recommend a buy by identifying a buy chance.
As soon as *Trade* assigns an order id a sell price and stop loss sell price is assigned.

- TradingStrategy.requiredminutes
- TradingStrategy.registerbuy!(...)
- TradingStrategy.deletetradechanceoforder!(...)
- TradingStrategy.tradechanceoforder(...)
- TradingStrategy.registerbuyorder!(..)
- TradingStrategy.registersellorder!(...)
- TradingStrategy.deletenewbuychanceofbase!(...)
- TradingStrategy.assesstrades!(...)

"""
module TradingStrategy

using DataFrames, Logging
using Dates, DataFrames
using EnvConfig, Ohlcv, Features
export requiredminutes

combinedbuysellfee = 0.002

"Interpolate a 2% gain for 24h and 0.1% for 5min as straight line gx+b = deltapercentage/deltaminutes * requestedminutes + percentage at start"
interpolateminpercentage(minutes) = (0.02 - 0.001) / (24*60 - 5) * (minutes - 5) + 0.001

function mingainminuterange(minutesarray)
    mingains = Dict()
    for minutes in minutesarray
        mingains[minutes] = interpolateminpercentage(minutes)
    end
    return mingains
end

mutable struct TradeChance
    base::String
    buydt  # buy datetime remains "missing" until completely bought
    buyprice
    buyorderid  # remains "missing" until order issued
    selldt  # sell datetime remains "missing" until completely bought
    sellprice
    sellorderid  # remains "missing" until order issued
    regrminutes
    stoplossprice # not used by TradeChances004
end

mutable struct TradingStrategyCache
    features
    targets
    classifier
    currentohlcvix
end

mutable struct TradeChances
    basedict  # Dict of new buy orders
    orderdict  # Dict of open orders
    holdlist  # array of TradeChance with closed buy but without sell order
    #TODO closebuy order should place trade in holdlist and assess shall take it from there to issue a sell order
    #TODO in trade remove automatic sell after buy. feedback to trade via basedict value
    function TradeChances()
        new(Dict(), Dict())
    end
end

upperbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] + stdfactor * fr.std[ix]
lowerbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] - stdfactor * fr.std[ix]
stoplossprice(fr::Features.Features002Regr, ix, stdfactor) = isnothing(stdfactor) ? 0.0 : fr.regry[ix] - stdfactor * fr.std[ix]
banddeltaprice(fr::Features.Features002Regr, ix, stdfactor) = 2 * stdfactor * fr.std[ix]

requiredtradehistory = Features.requiredminutes()
requiredminutes = requiredtradehistory + Features.requiredminutes()

"""
- grad = deltaprice / deltaminutes
- deltaprice per day = grad * 24 * 60
- (deltaprice / reference_price) per day = grad * 24 * 60 / reference_price
"""
minutesgain(gradient, refprice, minutes) = minutes * gradient / refprice
daygain(gradient, refprice) = minutesgain(gradient, refprice, 24 * 60)

function Base.show(io::IO, tc::TradeChance)
    sl = isnothing(tc.stoplossprice) ? "n/a" : round(tc.stoplossprice; digits=2)
    bo = isnothing(tc.breakoutstd) ? "n/a" : round(tc.breakoutstd; digits=2)
    buydt = isa(tc.buydt, Dates.DateTime) ? EnvConfig.timestr(tc.buydt) : string(tc.buydt)
    selldt = isa(tc.selldt, Dates.DateTime) ? EnvConfig.timestr(tc.selldt) : string(tc.selldt)
    print(io::IO, "tc: base=$(tc.base), buydt=$buydt, buy=$(round(tc.buyprice; digits=2)), buyid=$(tc.buyorderid), selldt=$selldt, sell=$(round(tc.sellprice; digits=2)), sellid=$(tc.sellorderid), window=$(tc.regrminutes), stop loss sell=$sl, breakoutstd=$bo")
end

function Base.show(io::IO, tcs::TradeChances)
    println("tradechances: new buy chances")
    for (ix, tc) in enumerate(values(tcs.basedict))
        if isa(tc[2], AbstractVector)
            for (vix, tce) in tc[2]
                println("$ix: basecoin: $(tc[1]), $vix: $tce")
            end
        else
            println("$ix: basecoin: $(tc[1]), tradechance: $(tc[2])")
        end
    end
    println("tradechances: open order chances")
    for (ix, tc) in enumerate(values(tcs.orderdict))
        if isa(tc[2], AbstractVector)
            for (vix, tce) in tc[2]
                println("$ix: order: $(tc[1]), $vix: $tce")
            end
        else
            println("$ix: order: $(tc[1]), tradechance: $(tc[2])")
        end
    end
end

Base.length(tcs::TradeChances) = length(keys(tcs.basedict)) + length(keys(tcs.orderdict))

function init(ohlcv::Ohlcv.OhlcvData, classifier)::Union{Nothing, TradingStrategyCache}
    println("$(EnvConfig.now()) start generating features002")
    f2 = Features.Features002(ohlcv)
end

"Used by Trade.closeorder! if buy order is executed"
function registerbuy!(tradechances::TradeChances, buyix, buyprice, buyorderid, f2::Features.Features002, stoplossstd)
    @assert buyix > 0
    @assert buyprice > 0
    tc = tradechanceoforder(tradechances, buyorderid)
    if isnothing(tc)
        @warn "missing order #$buyorderid in tradechances"
    else
        df = Ohlcv.dataframe(f2.ohlcv)
        opentime = df[!, :opentime]
        afr = f2.regr[tc.regrminutes]
        fix = Features.featureix(f2, buyix)
        tc.buydt = opentime[buyix]
        tc.buyprice = buyprice
        tc.buyorderid = buyorderid
        tc.stoplossprice = isnothing(stoplossstd) ? 0.0 : stoplossprice(afr, fix, stoplossstd)
        if !isnothing(tc.breakoutstd)
            spread = banddeltaprice(afr, fix, tc.breakoutstd)
            tc.sellprice = upperbandprice(afr, fix, tc.breakoutstd)
        end
        # tc.sellprice = afr.regry[fix] + halfband
    end
    return tc
end

"Delivers the tradechance based on orderid. E.g. used in TradingStrategy.registerbuy!, Trade.trade!, Trade.neorder"
function tradechanceoforder(tradechances::TradeChances, orderid)
    tc = nothing
    if orderid in keys(tradechances.orderdict)
        tc = tradechances.orderdict[orderid]
    end
    return tc
end

"Used by Trade.trade!"
function tradechanceofbase(tradechances::TradeChances, base)
    tc = nothing
    if base in keys(tradechances.basedict)
        tc = tradechances.basedict[base]
    end
    return tc
end

"Used by Trade.trade!"
baseswithnewtradechances(tradechances::TradeChances) = keys(tradechances.basedict)

"Used by Trade.closeorder!"
function deletetradechanceoforder!(tradechances::TradeChances, orderid)
    if orderid in keys(tradechances.orderdict)
        delete!(tradechances.orderdict, orderid)
    end
end

"Used by Trade.trade!"
function deletenewbuychanceofbase!(tradechances::TradeChances, base)
    if base in keys(tradechances.basedict)
        delete!(tradechances.basedict, base)
    end
end

"Used by Trade.neworder!"
function registerbuyorder!(tradechances::TradeChances, orderid, tc::TradeChance)
    tc.buyorderid = orderid
    tradechances.orderdict[orderid] = tc
end

"Used by Trade.neworder!"
function registersellorder!(tradechances::TradeChances, orderid, tc::TradeChance)
    tc.sellorderid = orderid
    tradechances.orderdict[orderid] = tc
end

"Remove all new buy chances. Those that were issued as orders are moved to the open order dict, Used by TradingStrategy.assesstrades!"
function cleanupnewbuychance!(tradechances::TradeChances, base)
    if (base in keys(tradechances.basedict))
        tc = tradechances.basedict[base]
        delete!(tradechances.basedict, base)
        if (tc.buyorderid > 0)
            tradechances.orderdict[tc.buyorderid] = tc
        end
        if (tc.sellorderid > 0)
            # it is possible that a buy order is partially executed and buy and sell orders are open
            tradechances.orderdict[tc.sellorderid] = tc
        end
    end
    return tradechances
end



"""
*TradeChances004* uses a given regression line to buy in regular intervals with low volume below std if std/regry is > gainthreshold and to sell in regular intervals above std if std/regry is > gainthreshold.
"""
mutable struct TradeChances004
    baseconfig
    tcs::TradeChances
    function TradeChances004()
        new(
            Dict("BTC" => (regrwindow=720, gain=0.01, gap=2)),
            # (regr=720, gain=1.0) = use regression window 720 minutes and trade when std/regry >= gain of 1% with a minimum gap of 2 minutes
            TradeChances()
            )
    end
end

function Base.show(io::IO, tcs::TradeChances004)
    cfg = ["$(p[1]): $(p[2])  " for p in tcs.baseconfig]
    show("io::IO, TradeChances004 - $cfg\n")
    show(tcs.tcs)
end


"""
Trading strategy:
- buy if current price is below regression line - regression std and regression std / regression regry > gainthreshold of the resp. base
- sell if current price is above regression line + regression std and regression std / regression regry > gainthreshold of the resp. base

if tradechances === nothing then an empty TradeChance array is created and with results returned
"""
function assesstrades!(tradechances::TradeChances004, f2::Features.Features002, ohlcvix)::TradeChances004
    @assert !isnothing(f2)
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(Features.ohlcv(f2))
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basecoin(Features.ohlcv(f2))
    if !(base in keys(tradechances.baseconfig))
        return nothing
    end
    @info "$(@doc TradeChances004)" tradechances.baseconfig[base] maxlog=1
    cleanupnewbuychance!(tradechances.tcs, base)
    fix = Features.featureix(f2, ohlcvix)
    rw = tradechances.baseconfig[base].regrwindow
    afr = f2.regr[rw]
    newtc = nothing
    if (pivot[ohlcvix] > f2.regr[rw].regry[fix] + f2.regr[rw].std[fix]) && ((f2.regr[rw].std[fix] / f2.regr[rw].regry[fix]) > tradechances.baseconfig[base].gain)
        buyprice = pivot[ohlcvix]
        sellprice = buyprice * (1 + 2 * tradechances.baseconfig[base].gain)
        newtc = TradeChance(base, nothing, buyprice, 0, sellprice, 0, rw, nothing, nothing) # stoplossprice = breakoutstd = nothing because both are disabled
    end
#TODO here to continue
    for tc in values(tradechances.tcs.orderdict)
        if tc.base == Ohlcv.basecoin(Features.ohlcv(f2))
            if isnothing(tc.buydt)
                if !isnothing(newtc) && (tc.regrminutes == newtc.regrminutes) # && (tc.breakoutstd == newtc.breakoutstd)
                    # not yet bought -> adapt with latest insights
                    tc.buyprice = newtc.buyprice
                    tc.stoplossprice = newtc.stoplossprice
                    tc.sellprice = newtc.sellprice
                    newtc = nothing
                end
            end
            afr = f2.regr[tc.regrminutes]
            if (afr.grad[featureix-1] >= 0) && (afr.grad[featureix] < 0)
                    tc.sellprice = df.pivot[ohlcvix]
            end
        end
    end
    if !isnothing(newtc)
        tradechances.tcs.basedict[base] = newtc
    end
    return tradechances
end

registerbuy!(tradechances::TradeChances004, buyix, buyprice, buyorderid, f2::Features.Features002) = registerbuy!(tradechances.tcs, buyix, buyprice, buyorderid, f2, nothing)
tradechanceoforder(tradechances::TradeChances004, orderid) = tradechanceoforder(tradechances.tcs, orderid)
deletetradechanceoforder!(tradechances::TradeChances004, orderid) = deletetradechanceoforder!(tradechances.tcs, orderid)
registerbuyorder!(tradechances::TradeChances004, orderid, tc::TradeChance) = registerbuyorder!(tradechances.tcs, orderid, tc)
registersellorder!(tradechances::TradeChances004, orderid, tc::TradeChance) = registersellorder!(tradechances.tcs, orderid, tc)
deletenewbuychanceofbase!(tradechances::TradeChances004, base) = deletenewbuychanceofbase!(tradechances.tcs, base)

end # module
