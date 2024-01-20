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
    buydt  # buy datetime remains nothing until completely bought
    buyprice
    buyorderid  # remains 0 until order issued
    sellprice
    sellorderid  # remains 0 until order issued
    regrminutes
    stoplossprice # not used by TradeChances004
    breakoutstd  # only relevant for TradeChances001; indices of extremes (<0 if min, >0 if max) that exceed std distance from regry
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
    print(io::IO, "tc: base=$(tc.base), buydt=$(EnvConfig.timestr(tc.buydt)), buy=$(round(tc.buyprice; digits=2)), buyid=$(tc.buyorderid) sell=$(round(tc.sellprice; digits=2)), sellid=$(tc.sellorderid), window=$(tc.regrminutes), stop loss sell=$sl, breakoutstd=$bo")
end

function Base.show(io::IO, tcs::TradeChances)
    println("tradechances: $(values(tcs.basedict)) new buy chances")
    for (ix, tc) in enumerate(values(tcs.basedict))
        println("$ix: $tc")
    end
    println("tradechances: $(values(tcs.orderdict)) open order chances")
    for (ix, tc) in enumerate(values(tcs.orderdict))
        println("$ix: $tc")
    end
end

Base.length(tcs::TradeChances) = length(keys(tcs.basedict)) + length(keys(tcs.orderdict))

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

"not yet used"
function tradechanceofbase(tradechances::TradeChances, base)
    tc = nothing
    if base in keys(tradechances.basedict)
        tc = tradechances.basedict[base]
    end
    return tc
end

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


# tr001 = TradeRules001(0.02, 0.0, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
# tr001 = TradeRules001(0.015, 0.0001, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])  # for test purposes
# tr001 = TradeRules001(0.05, 0.02, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])  #! topper
# tr001 = TradeRules001(0.005, 0.0, 3.0, [0.5])  # for test purposes

mutable struct TradeRules001  # only relevant for traderules001!
    minimumgain  # minimum spread around regression to consider buy
    minimumgradientdaygain  # 1% gain per day aspre-requisite to buy
    stoplossstd  # factor to multiply with std to dtermine stop loss price (only in combi with negative regr gradient)
    breakoutstdset  # set of breakoutstd factors to test when determining the best combi or regregression window and spread
end

"""
*TradeChances001* trading strategy:
- buy if price is
    - below normal deviation range of spread regression window
    - spread gradient is OK
    - spread satisfies minimum profit requirements
- sell if price is above normal deviation range of spread regression window
- stop loss sell: if regression price < buy regression price - stoplossstd * std
- spread gradient is OK = spread gradient > `minimumgradientdaygain`
- normal deviation range = regry +- breakoutstd * std = band around regry of std * 2 * breakoutstd
- spread satisfies minimum profit requirements = normal deviation range >= minimumgain

"""
mutable struct TradeChances001
    tr::TradeRules001
    tcs::TradeChances
    function TradeChances001()
        new(
            TradeRules001(0.05, 0.02, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),  # topper
            TradeChances()
            )
    end
end

function Base.show(io::IO, tcs::TradeChances001)
    println("TradeChances001: minimumgain=$(tcs.tr.minimumgain) minimumgradientdaygain=$(tcs.tr.minimumgradientdaygain) stoplossstd=$(tcs.tr.stoplossstd) breakoutstdset=$(tcs.tr.breakoutstdset)")
    println(tcs.tcs)
end

function buycompliant001(f2, window, breakoutstd, ix, tr001)
    df = Ohlcv.dataframe(f2.ohlcv)
    afr = f2.regr[window]
    fix = Features.featureix(f2, ix)

    pastix = ix - Int64(ceil(window / 4))
    pastfix = Features.featureix(f2, pastix)
    # pastmaximumgradientdaygain = daygain(afr.grad[fix], df.close[ix]) / 2

    spreadpercent = banddeltaprice(afr, fix, breakoutstd) / afr.regry[fix]
    lowerprice = lowerbandprice(afr, fix, breakoutstd)
    ok =  ((df.low[ix] < lowerprice) &&
        (spreadpercent >= tr001.minimumgain) &&
        (daygain(afr.grad[fix], df.close[ix]) > tr001.minimumgradientdaygain) &&
        (daygain(afr.grad[fix], df.close[ix]) > daygain(afr.grad[pastfix], df.close[pastix])) &&
        (daygain(f2.regr[1*60].grad[fix], df.close[ix]) > tr001.minimumgradientdaygain) &&
        (daygain(f2.regr[24*60].grad[fix], df.close[ix]) > tr001.minimumgradientdaygain))
    return ok
end

"""
Returns the best performing combination of spread window and breakoutstd factor.
In case that minimumgain requirements are not met, `bestwindow` returns 0 and `breakoutstd` returns 0.0.
"""
function bestspreadwindow001(f2::Features.Features002, ohlcvix, tr001)
    @assert length(tr001.breakoutstdset) > 0
    maxtrades = 0
    maxgain = tr001.minimumgain
    bestwindow = 0
    bestbreakoutstd = 0.0
    for breakoutstd in tr001.breakoutstdset
        for window in keys(f2.regr)
                trades, gain = calcspread001(f2, window, ohlcvix, breakoutstd, tr001)
            if gain > maxgain  # (trades > maxtrades) && (gain > 0)
                maxgain = gain
                maxtrades = trades
                bestwindow = window
                bestbreakoutstd = breakoutstd
            end
        end
    end
return bestwindow, bestbreakoutstd
end

"""
Returns the number of trades within the last `requiredminutes` and the gain achived.
In case that minimumgain requirements are not met by fr.std, `trades` and `gain` return 0.
"""
function calcspread001(f2::Features.Features002, window, ohlcvix, breakoutstd, tr001)
    fr = f2.regr[window]
    fix = Features.featureix(f2, ohlcvix)
    gain = 0.0
    trades = 0
    if fix >= (requiredminutes)
        startix = max(1, ohlcvix - requiredtradehistory)
        breakoutix = breakoutextremesix001!(f2, window, breakoutstd, startix, tr001)
        xix = [ix for ix in breakoutix if startix <= abs(ix)  <= ohlcvix]
        # println("breakoutextremesix @ window $window breakoutstd $breakoutstd : $xix")
        buyix = sellix = 0
        for ix in xix
            # first minimum as buy if sell is closed
            buyix = (buyix == 0) && (sellix == 0) && (ix < 0) ? ix : buyix
            # first maximum as sell if buy was done
            sellix = (buyix < 0) && (sellix == 0) && (ix > 0) ? ix : sellix
            if buyix < 0 < sellix
                trades += 1
                thisgain = (upperbandprice(fr, Features.featureix(f2, sellix), breakoutstd) - lowerbandprice(fr, Features.featureix(f2, abs(buyix)), breakoutstd)) / lowerbandprice(fr, Features.featureix(f2, abs(buyix)), breakoutstd)
                gain += thisgain
                buyix = sellix = 0
            end
        end
    end
    return trades, gain
end

function breakoutextremesix001!(f2::Features.Features002, window, breakoutstd, startindex, tr001)
    @assert f2.lastix >= startindex >= f2.firstix "$(f2.lastix) >= $startindex >= $(f2.firstix)"
    afr = f2.regr[window]
    df = Ohlcv.dataframe(Features.ohlcv(f2))
    extremeix = Int32[]
    breakoutix = 0  # negative index for minima, positive index for maxima, else 0
    for ix in startindex:f2.lastix
        fix = Features.featureix(f2, ix)
        if breakoutix <= 0
            if df[ix, :high] > afr.regry[fix] + breakoutstd * afr.std[fix]
                push!(extremeix, ix)
            end
        end
        if breakoutix >= 0
            if buycompliant001(f2, window, breakoutstd, ix, tr001)
            # if df[ix, :low] < afr.regry[fix] - breakoutstd * afr.std[fix]
                push!(extremeix, -ix)
            end
        end
        # the dealyed breakout assignment is required if spread band is between low and high
        # then the min and max ix shall be alternating every minute
        if (length(extremeix) > 0) && (sign(breakoutix) != sign(extremeix[end]))
            breakoutix = extremeix[end]
        end
    end
    return extremeix
end

function newbuychance001(f2::Features.Features002, ohlcvix, tr001)
    base = Ohlcv.basesymbol(f2.ohlcv)
    tc = nothing
    fix = Features.featureix(f2, ohlcvix)
    regrminutes, breakoutstd = bestspreadwindow001(f2, ohlcvix, tr001)
    if regrminutes > 0 # best window found
        if buycompliant001(f2, regrminutes, breakoutstd, ohlcvix, tr001)
            # @info "buy signal $base price=$(round(df.low[ohlcvix];digits=3)) window=$regrminutes ix=$ohlcvix time=$(opentime[ohlcvix])"
            afr = f2.regr[regrminutes]
            # spread = banddeltaprice(afr, fix, breakoutstd)
            buyprice = lowerbandprice(afr, fix, breakoutstd)
            sellprice = upperbandprice(afr, fix, breakoutstd)
            tcstoplossprice = stoplossprice(afr, fix, tr001.stoplossstd)
            tc = TradeChance(base, nothing, buyprice, 0, sellprice, 0, regrminutes, tcstoplossprice, breakoutstd)
        end
    end
    return tc
end

"""
traderules001!(tradechances, f2::Features.Features002, ohlcvix)

Returns the assessed tradechances to be executed by the trade loop
if tradechances === nothing then an empty TradeChance array is created and with results returned

"""
function assesstrades!(tradechances::TradeChances001, f2::Features.Features002, ohlcvix)::TradeChances001
    @info "$(@doc TradeChances001)" tradechances.tr.minimumgain tradechances.tr.minimumgradientdaygain tradechances.tr.stoplossstd tradechances.tr.breakoutstdset maxlog=1
    if isnothing(f2); return tradechances; end
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(f2.ohlcv)
    cleanupnewbuychance!(tradechances.tcs, base)
    newtc = newbuychance001(f2, ohlcvix, tradechances.tr)
    for tc in values(tradechances.tcs.orderdict)
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(f2))
            if isnothing(tc.buydt)
                if !isnothing(newtc) && (tc.regrminutes == newtc.regrminutes) && (tc.breakoutstd == newtc.breakoutstd)
                    # not yet bought -> adapt with latest insights
                    tc.buyprice = newtc.buyprice
                    tc.stoplossprice = newtc.stoplossprice
                    tc.sellprice = newtc.sellprice
                    newtc = nothing
                end
            end
            afr = f2.regr[tc.regrminutes]
            fix = Features.featureix(f2, ohlcvix)
            spread = banddeltaprice(afr, fix, tc.breakoutstd)
            if (pivot[ohlcvix] < tc.stoplossprice) && (afr.grad[fix] < 0)
                # stop loss exit due to plunge of price and negative regression line
                tc.sellprice = df.low[ohlcvix]
                @info "stop loss sell for $base due to plunge out of spread ix=$ohlcvix time=$(opentime[ohlcvix]) at regression price of $(afr.regry[fix]) tc: $tc"
            elseif buycompliant001(f2, tc.regrminutes, tc.breakoutstd, ohlcvix, tradechances.tr)
                tc.sellprice = upperbandprice(afr, fix, tc.breakoutstd)
            elseif afr.grad[fix] < 0
                tc.sellprice = max(lowerbandprice(afr, fix, tc.breakoutstd), tc.buyprice * (1 + combinedbuysellfee))
            else
                tc.sellprice = afr.regry[fix]
            end
            # @info "sell signal $(base) regrminutes=$(tc.regrminutes) breakoutstd=$(tc.breakoutstd) at price=$(round(tc.sellprice;digits=3)) ix=$ohlcvix  time=$(opentime[ohlcvix])"
        end
    end
    if !isnothing(newtc)
        tradechances.tcs.basedict[base] = newtc
    end

    # TODO use case of breakout rise following with tracker window is not yet covered - implemented in traderules002!
    # TODO use case of breakout rise after sell above deviation range is not yet covered
    return tradechances
end

registerbuy!(tradechances::TradeChances001, buyix, buyprice, buyorderid, f2::Features.Features002) = registerbuy!(tradechances.tcs, buyix, buyprice, buyorderid, f2, tradechances.tr.stoplossstd)
tradechanceoforder(tradechances::TradeChances001, orderid) = tradechanceoforder(tradechances.tcs, orderid)
deletetradechanceoforder!(tradechances::TradeChances001, orderid) = deletetradechanceoforder!(tradechances.tcs, orderid)
registerbuyorder!(tradechances::TradeChances001, orderid, tc::TradeChance) = registerbuyorder!(tradechances.tcs, orderid, tc)
registersellorder!(tradechances::TradeChances001, orderid, tc::TradeChance) = registersellorder!(tradechances.tcs, orderid, tc)
deletenewbuychanceofbase!(tradechances::TradeChances001, base) = deletenewbuychanceofbase!(tradechances.tcs, base)


mutable struct TradeRules002  # only relevant for traderules002!
    minimumbacktestgain  # minimum gain in backtest check to consider buy
    minimumbuygradientdict  # regression time window specific gradient dict creating a hysteresis to avoid frequent buy/sell around a minimum/maximum
    shorterwindowimprovement  # gain factor to exceed by shorter backtest window in order to be considered
    stoplossstd  # not used in TradingStrategy002 - factor to multiply with std to dtermine stop loss price (only in combi with negative regr gradient)
end

"""
*TradeChances002* trading strategy:
- buy if price
    - exceeds a minimum gradient after reaching a minimum
    - use regression window that performd best in the backtest time window and that satisfies minimum profit requirement
    - use different backtest time windows but shorter time windows have to outperform the longer ones by >20% to be considered
    - only regression time windows that are <= backtest time window are considered
- sell if price is at maximum == gradient of regression is zero

"""
mutable struct TradeChances002
    tr::TradeRules002
    tcs::TradeChances
    function TradeChances002()
        new(
            TradeRules002(0.1, mingainminuterange(Features.regressionwindows002), 0.2, 3.0),
            TradeChances()
            )
    end
end

function Base.show(io::IO, tcs::TradeChances002)
    println("TradeChances002: minimumbacktestgain=$(tcs.tr.minimumbacktestgain) minimumbuygradientdict=$(tcs.tr.minimumbuygradientdict) shorterwindowimprovement=$(tcs.tr.shorterwindowimprovement)")
    show(tcs.tcs)
end

function alllargerwindowgradimprove(f2, window, featureix)
    ok = true
    count = ok = 0
    for w in Features.regressionwindows002
        if w >= window
            pastfeatureix = featureix - Int64(ceil(w / 4))
            ok = f2.regr[w].grad[featureix] > f2.regr[w].grad[pastfeatureix] ? ok + 1 : ok
            count += 1
        end
    end
    return (count / ok) < 2
end

function buycompliant002(f2, window, ohlcvix, tr002)
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    afr = f2.regr[window]
    fix = Features.featureix(f2, ohlcvix)

    pastix = ohlcvix - Int64(ceil(window / 4))
    pastfix = Features.featureix(f2, pastix)
    ok =  (
        (minutesgain(afr.grad[fix], df.close[ohlcvix], window) > tr002.minimumbuygradientdict[window]) &&
        alllargerwindowgradimprove(f2, window, fix) &&
        # (afr.grad[fix] > afr.grad[pastfix]) &&
        (minutesgain(f2.regr[1*60].grad[fix], df.close[ohlcvix], 1*60) > tr002.minimumbuygradientdict[1*60]) &&
        (minutesgain(f2.regr[24*60].grad[fix], df.close[ohlcvix], 24*60) > tr002.minimumbuygradientdict[24*60]))

    return ok
end

"""
Returns the best performing regression window within the last backtestminutes period before .
In case that minimumgain requirements are not met, `bestwindow` returns 0 and `maxgain` returns 0.0.
"""
function bestregrwindow002(f2::Features.Features002, ohlcvix, backtestminutes, tr002)
    maxtrades = 0
    maxgain = tr002.minimumbacktestgain
    bestwindow = 0
    fix = Features.featureix(f2, ohlcvix)
    startfix = fix - backtestminutes
    @assert startfix > 0
    for window in keys(f2.regr)
        extremeix = Features.regressionextremesix!(nothing, f2.regr[window].grad, startfix)
        gain = 0.0
        lastix = 0
        trades = 0
        for eix in extremeix
            if (lastix < 0) && (eix > 0)
                lastix = abs(lastix)
                while  (lastix < eix) && !buycompliant002(f2, window, Features.ohlcvix(f2, lastix), tr002)
                    lastix += 1
                end
                if lastix < eix
                    maxix = Features.ohlcvix(f2, eix)
                    minix = Features.ohlcvix(f2, lastix)
                    thisgain = Ohlcv.relativegain(f2.ohlcv.df[!, :pivot], maxix, minix; relativefee=combinedbuysellfee)
                    trades += 1
                    gain += thisgain
                end
            end
            lastix = eix
        end
        if gain > maxgain  # (trades > maxtrades) && (gain > 0)
            maxgain = gain
            maxtrades = trades
            bestwindow = window
        end
    end
    maxgain = bestwindow > 0 ? maxgain : 0.0
    return bestwindow, maxgain, maxtrades
end

"""
Returns the best performing combined regression window and backtestminutes period.
In case that minimumgain requirements are not met, `bestwindow` returns 0 and `bestbacktestminutes` returns 0.
"""
function bestbacktestwindow002(f2::Features.Features002, ohlcvix, tr002)
    maxgain = tr002.minimumbacktestgain
    bestwindow = 0
    bestbacktestminutes = 0
    for backtestminutes in reverse(Features.regressionwindows002)
        window, gain, trades = bestregrwindow002(f2, ohlcvix, backtestminutes, tr002)
        if (gain > maxgain) && (window > 0)
            if bestbacktestminutes > 0
                if gain > (1 + tr002.shorterwindowimprovement) * maxgain
                    maxgain = gain
                    bestwindow = window
                    bestbacktestminutes = backtestminutes
                end
            else
                maxgain = gain
                bestwindow = window
                bestbacktestminutes = backtestminutes
            end
        end
    end
    return bestwindow, bestbacktestminutes
end

function newbuychance002(f2::Features.Features002, ohlcvix, tr002)
    df = Ohlcv.dataframe(f2.ohlcv)
    # opentime = df[!, :opentime]
    base = Ohlcv.basesymbol(f2.ohlcv)
    tc = nothing
    fix = Features.featureix(f2, ohlcvix)
    regrminutes, bestbacktestminutes = bestbacktestwindow002(f2, ohlcvix, tr002)
    if regrminutes > 0 # best window found
        if buycompliant002(f2, regrminutes, ohlcvix, tr002)
            # @info "buy signal $base price=$(round(df.low[ohlcvix];digits=3)) window=$regrminutes ix=$ohlcvix time=$(opentime[ohlcvix])"
            afr = f2.regr[regrminutes]
            # spread = banddeltaprice(afr, fix, breakoutstd)
            buyprice = df.pivot[ohlcvix]
            sellprice = buyprice * 1.02
            tcstoplossprice = stoplossprice(afr, fix, 3.0)
            tc = TradeChance(base, nothing, buyprice, 0, sellprice, 0, regrminutes, tcstoplossprice, 1.0)
        end
    end
    return tc
end

"""
    traderules002!(tradechances, f2::Features.Features002, ohlcvix)

if tradechances === nothing then an empty TradeChance array is created and with results returned
"""
function assesstrades!(tradechances::TradeChances002, f2::Features.Features002, ohlcvix)::TradeChances002
    @info "$(@doc TradeChances002)" tradechances.tr.minimumbacktestgain tradechances.tr.minimumbuygradientdict tradechances.tr.shorterwindowimprovement tradechances.tr.stoplossstd maxlog=1
    if isnothing(f2); return tradechances; end
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(f2.ohlcv)
    cleanupnewbuychance!(tradechances.tcs, base)
    newtc = newbuychance002(f2, ohlcvix, tradechances.tr)
    for tc in values(tradechances.tcs.orderdict)
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(f2))
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
            fix = Features.featureix(f2, ohlcvix)
            if afr.grad[fix] < 0
                if pivot[ohlcvix] > lowerbandprice(afr, fix, 1.0)
                    tc.sellprice = df.pivot[ohlcvix]
                else
                    tc.sellprice = df.low[ohlcvix]
                end
            else  # if pivot[ohlcvix] > afr.regry[ohlcvix]  # above normal deviations
                tc.sellprice = pivot[ohlcvix] * 1.02  # normally not reachable within a minute or we catch a peak
                # @info "sell signal $(base) regrminutes=$(tc.regrminutes) breakoutstd=$(tc.breakoutstd) at price=$(round(tc.sellprice;digits=3)) ix=$ohlcvix  time=$(opentime[ohlcvix])"
            end
        end
    end
    if !isnothing(newtc)
        tradechances.tcs.basedict[base] = newtc
    end
    return tradechances
end

registerbuy!(tradechances::TradeChances002, buyix, buyprice, buyorderid, f2::Features.Features002) = registerbuy!(tradechances.tcs, buyix, buyprice, buyorderid, f2, tradechances.tr.stoplossstd)
tradechanceoforder(tradechances::TradeChances002, orderid) = tradechanceoforder(tradechances.tcs, orderid)
deletetradechanceoforder!(tradechances::TradeChances002, orderid) = deletetradechanceoforder!(tradechances.tcs, orderid)
registerbuyorder!(tradechances::TradeChances002, orderid, tc::TradeChance) = registerbuyorder!(tradechances.tcs, orderid, tc)
registersellorder!(tradechances::TradeChances002, orderid, tc::TradeChance) = registersellorder!(tradechances.tcs, orderid, tc)
deletenewbuychanceofbase!(tradechances::TradeChances002, base) = deletenewbuychanceofbase!(tradechances.tcs, base)

"""
*TradeChances000* uses a 24h minute regression line to buy on minimum and sell at maximum.
"""
mutable struct TradeChances000
    stoplossstd # not used in TradingStrategy000
    tcs::TradeChances
    function TradeChances000()
        new(
            3.0,
            TradeChances()
            )
    end
end

function Base.show(io::IO, tcs::TradeChances000)
    show("TradeChances000 - test print \n \n")
    show(tcs.tcs)
end


"""
    traderules000!(tradechances, f2::Features.Features002, ohlcvix)

if tradechances === nothing then an empty TradeChance array is created and with results returned

Trading strategy:
- buy if regression line is at minimum
- sell if regression line is at maximum == gradient of regression is zero

"""
function assesstrades!(tradechances::TradeChances000, f2::Features.Features002, ohlcvix)::TradeChances000
    regressionminutes = 24*60
    @info "$(@doc TradeChances000)" regressionminutes maxlog=1
    if isnothing(f2); return tradechances; end
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(f2.ohlcv)
    cleanupnewbuychance!(tradechances.tcs, base)
    featureix = Features.featureix(f2, ohlcvix)
    afr = f2.regr[regressionminutes]
    newtc = nothing
    if featureix > 1
        if (afr.grad[featureix-1] <= 0) && (afr.grad[featureix] > 0)
            buyprice = df.pivot[ohlcvix]
            sellprice = buyprice * 1.1
            tcstoplossprice = buyprice * 0.8
            newtc = TradeChance(base, nothing, buyprice, 0, sellprice, 0, regressionminutes, tcstoplossprice, 1.0)
        end
    end

    for tc in values(tradechances.tcs.orderdict)
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(f2))
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

registerbuy!(tradechances::TradeChances000, buyix, buyprice, buyorderid, f2::Features.Features002) = registerbuy!(tradechances.tcs, buyix, buyprice, buyorderid, f2, tradechances.stoplossstd)
tradechanceoforder(tradechances::TradeChances000, orderid) = tradechanceoforder(tradechances.tcs, orderid)
deletetradechanceoforder!(tradechances::TradeChances000, orderid) = deletetradechanceoforder!(tradechances.tcs, orderid)
registerbuyorder!(tradechances::TradeChances000, orderid, tc::TradeChance) = registerbuyorder!(tradechances.tcs, orderid, tc)
registersellorder!(tradechances::TradeChances000, orderid, tc::TradeChance) = registersellorder!(tradechances.tcs, orderid, tc)
deletenewbuychanceofbase!(tradechances::TradeChances000, base) = deletenewbuychanceofbase!(tradechances.tcs, base)

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
    cfg = ["$(p[1]): $(p[2])  " for p in tcs.baseconfig]...
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
    base = Ohlcv.basesymbol(Features.ohlcv(f2))
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
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(f2))
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
