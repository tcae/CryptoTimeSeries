"""
Train and evaluate the trading signal classifiers
"""
module Classify

using DataFrames, Logging  # , MLJ
using MLJ, MLJBase, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics
using PlotlyJS, WebIO, Dates, DataFrames
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv
export traderules001!, tradeperformance

mutable struct TradeRules001  # only relevant for traderules001!
    minimumgain  # minimum spread around regression to consider buy
    minimumgradientdaygain  # 1% gain per day aspre-requisite to buy
    stoplossstd  # factor to multiply with std to dtermine stop loss price (only in combi with negative regr gradient)
    breakoutstdset  # set of breakoutstd factors to test when determining the best combi or regregression window and spread
end

# tr001default = TradeRules001(0.02, 0.0, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
# tr001default = TradeRules001(0.015, 0.0001, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])  # for test purposes
const tr001default = TradeRules001(0.015, 0.02, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])  # topper
# tr001default = TradeRules001(0.005, 0.0, 3.0, [0.5])  # for test purposes
combinedbuysellfee = 0.002

mutable struct TradeRules002  # only relevant for traderules002!
    minimumbacktestgain  # minimum gain in backtest check to consider buy
    minimumbuygradientdict  # regression time window specific gradient dict creating a hysteresis to avoid frequent buy/sell around a minimum/maximum
    shorterwindowimprovement  # gain factor to exceed by shorter backtest window in order to be considered
end

"Interpolate a 2% gain for 24h and 0.1% for 5min as straight line gx+b = deltapercentage/deltaminutes * requestedminutes + percentage at start"
interpolateminpercentage(minutes) = (0.02 - 0.001) / (24*60 - 5) * (minutes - 5) + 0.001

function mingainminuterange(minutesarray)
    mingains = Dict()
    for minutes in minutesarray
        mingains[minutes] = interpolateminpercentage(minutes)
    end
    return mingains
end

const tr002default = TradeRules002(0.05, mingainminuterange(Features.regressionwindows002), 0.2)

mutable struct TradeChance001
    base::String
    buydt  # buy datetime remains nothing until completely bought
    buyprice
    buyorderid  # remains 0 until order issued
    sellprice
    sellorderid  # remains 0 until order issued
    probability  # probability to reach sell price once bought
    regrminutes
    stoplossprice
    breakoutstd  # only relevant for traderules001! (not traderules002!)
end

mutable struct TradeChances001
    basedict  # Dict of new buy orders
    orderdict  # Dict of open orders
end

upperbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] + stdfactor * fr.medianstd[ix]
lowerbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] - stdfactor * fr.medianstd[ix]
stoplossprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] - stdfactor * fr.medianstd[ix]
banddeltaprice(fr::Features.Features002Regr, ix, stdfactor) = 2 * stdfactor * fr.medianstd[ix]

requiredtradehistory = Features.requiredminutes
requiredminutes = requiredtradehistory + Features.requiredminutes

"""
- grad = deltaprice / deltaminutes
- deltaprice per day = grad * 24 * 60
- (deltaprice / reference_price) per day = grad * 24 * 60 / reference_price
"""
minutesgain(gradient, refprice, minutes) = minutes * gradient / refprice
daygain(gradient, refprice) = minutesgain(gradient, refprice, 24 * 60)

function Base.show(io::IO, tc::TradeChance001)
    print(io::IO, "tc: base=$(tc.base), buydt=$(EnvConfig.timestr(tc.buydt)), buy=$(round(tc.buyprice; digits=2)), buyid=$(tc.buyorderid) sell=$(round(tc.sellprice; digits=2)), sellid=$(tc.sellorderid), prob=$(round(tc.probability*100; digits=2))%, window=$(tc.regrminutes), stop loss sell=$(round(tc.stoplossprice; digits=2)), breakoutstd=$(tc.breakoutstd)")
end

function Base.show(io::IO, tcs::TradeChances001)
    println("tradechances: $(values(tcs.basedict)) new buy chances")
    for (ix, tc) in enumerate(values(tcs.basedict))
        println("$ix: $tc")
    end
    println("tradechances: $(values(tcs.orderdict)) open order chances")
    for (ix, tc) in enumerate(values(tcs.orderdict))
        println("$ix: $tc")
    end
end

Base.length(tcs::TradeChances001) = length(keys(tcs.basedict)) + length(keys(tcs.orderdict))

function buycompliant001(f2, window, breakoutstd, ix)
    df = Ohlcv.dataframe(f2.ohlcv)
    afr = f2.regr[window]
    fix = Features.featureix(f2, ix)

    pastix = ix - Int64(ceil(window / 4))
    pastfix = Features.featureix(f2, pastix)
    # pastmaximumgradientdaygain = daygain(afr.grad[fix], df.close[ix]) / 2

    spreadpercent = banddeltaprice(afr, fix, breakoutstd) / afr.regry[fix]
    lowerprice = lowerbandprice(afr, fix, breakoutstd)
    ok =  ((df.low[ix] < lowerprice) &&
        (spreadpercent >= tr001default.minimumgain) &&
        (daygain(afr.grad[fix], df.close[ix]) > tr001default.minimumgradientdaygain) &&
        (daygain(afr.grad[fix], df.close[ix]) > daygain(afr.grad[pastfix], df.close[pastix])) &&
        (daygain(f2.regr[1*60].grad[fix], df.close[ix]) > tr001default.minimumgradientdaygain) &&
        (daygain(f2.regr[24*60].grad[fix], df.close[ix]) > tr001default.minimumgradientdaygain))
    return ok
end

"""
Returns the best performing combination of spread window and breakoutstd factor.
In case that minimumgain requirements are not met, `bestwindow` returns 0 and `breakoutstd` returns 0.0.
"""
function bestspreadwindow001(f2::Features.Features002, ohlcvix, minimumgain, breakoutstdset)
    @assert length(breakoutstdset) > 0
    maxtrades = 0
    maxgain = minimumgain
    bestwindow = 0
    bestbreakoutstd = 0.0
    for breakoutstd in breakoutstdset
        for window in keys(f2.regr)
                trades, gain = calcspread001(f2, window, ohlcvix, breakoutstd)
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
In case that minimumgain requirements are not met by fr.medianstd, `trades` and `gain` return 0.
"""
function calcspread001(f2::Features.Features002, window, ohlcvix, breakoutstd)
    fr = f2.regr[window]
    fix = Features.featureix(f2, ohlcvix)
    gain = 0.0
    trades = 0
    if fix >= (requiredminutes)
        startix = max(1, ohlcvix - requiredtradehistory)
        breakoutix = breakoutextremesix001!(f2, window, breakoutstd, startix)
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

function breakoutextremesix001!(f2::Features.Features002, window, breakoutstd, startindex)
    @assert f2.lastix >= startindex >= f2.firstix "$(f2.lastix) >= $startindex >= $(f2.firstix)"
    afr = f2.regr[window]
    df = Ohlcv.dataframe(Features.ohlcv(f2))
    extremeix = Int32[]
    breakoutix = 0  # negative index for minima, positive index for maxima, else 0
    for ix in startindex:f2.lastix
        fix = Features.featureix(f2, ix)
        if breakoutix <= 0
            if df[ix, :high] > afr.regry[fix] + breakoutstd * afr.medianstd[fix]
                push!(extremeix, ix)
            end
        end
        if breakoutix >= 0
            if buycompliant001(f2, window, breakoutstd, ix)
            # if df[ix, :low] < afr.regry[fix] - breakoutstd * afr.medianstd[fix]
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

function registerbuy!(tradechances::TradeChances001, buyix, buyprice, buyorderid, f2::Features.Features002)
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
        tc.stoplossprice = stoplossprice(afr, fix, tr001default.stoplossstd)
        spread = banddeltaprice(afr, fix, tc.breakoutstd)
        tc.sellprice = upperbandprice(afr, fix, tc.breakoutstd)
        # tc.sellprice = afr.regry[fix] + halfband
        # probability to reach sell price
        tc.probability = max(min((1.5 * spread - (tc.sellprice - buyprice)) / spread, 1.0), 0.0)
    end
    return tc
end

function tradechanceoforder(tradechances::TradeChances001, orderid)
    tc = nothing
    if orderid in keys(tradechances.orderdict)
        tc = tradechances.orderdict[orderid]
    end
    return tc
end

function tradechanceofbase(tradechances::TradeChances001, base)
    tc = nothing
    if base in keys(tradechances.basedict)
        tc = tradechances.basedict[base]
    end
    return tc
end

function deletetradechanceoforder!(tradechances::TradeChances001, orderid)
    if orderid in keys(tradechances.orderdict)
        delete!(tradechances.orderdict, orderid)
    end
end

function deletenewbuychanceofbase!(tradechances::TradeChances001, base)
    if base in keys(tradechances.basedict)
        delete!(tradechances.basedict, base)
    end
end

function registerbuyorder!(tradechances::TradeChances001, orderid, tc::TradeChance001)
    tc.buyorderid = orderid
    tradechances.orderdict[orderid] = tc
end

function registersellorder!(tradechances::TradeChances001, orderid, tc::TradeChance001)
    tc.sellorderid = orderid
    tradechances.orderdict[orderid] = tc
end

"Remove all new buy chances. Those that were issued as orders are moved to the open order dict"
function cleanupnewbuychance!(tradechances::TradeChances001, base)
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

function newbuychance001(f2::Features.Features002, ohlcvix)
    base = Ohlcv.basesymbol(f2.ohlcv)
    tc = nothing
    fix = Features.featureix(f2, ohlcvix)
    regrminutes, breakoutstd = bestspreadwindow001(f2, ohlcvix, tr001default.minimumgain, tr001default.breakoutstdset)
    if regrminutes > 0 # best window found
        if buycompliant001(f2, regrminutes, breakoutstd, ohlcvix)
            # @info "buy signal $base price=$(round(df.low[ohlcvix];digits=3)) window=$regrminutes ix=$ohlcvix time=$(opentime[ohlcvix])"
            afr = f2.regr[regrminutes]
            # spread = banddeltaprice(afr, fix, breakoutstd)
            buyprice = lowerbandprice(afr, fix, breakoutstd)
            sellprice = upperbandprice(afr, fix, breakoutstd)
            # probability to reach buy price
            # probability = max(min((spread - (df.low[ohlcvix] - buyprice)) / spread, 1.0), 0.0)
            probability = 0.8
            tcstoplossprice = stoplossprice(afr, fix, tr001default.stoplossstd)
            tc = TradeChance001(base, nothing, buyprice, 0, sellprice, 0, probability, regrminutes, tcstoplossprice, breakoutstd)
        end
    end
    return tc
end

"""
    traderules001!(tradechances, f2::Features.Features002, ohlcvix)

returns the chance expressed in gain between currentprice and future sellprice * probability
if tradechances === nothing then an empty TradeChance001 array is created and with results returned

Trading strategy:
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
function traderules001!(tradechances, f2::Features.Features002, ohlcvix)
    @info "$(@doc traderules001!)" maxlog=1
    @info "tr001default: $(tr001default)" maxlog=1
    tradechances = isnothing(tradechances) ? TradeChances001(Dict(), Dict()) : tradechances
    if isnothing(f2); return tradechances; end
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(f2.ohlcv)
    cleanupnewbuychance!(tradechances, base)
    newtc = newbuychance001(f2, ohlcvix)
    for tc in values(tradechances.orderdict)
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(f2))
            if isnothing(tc.buydt)
                if !isnothing(newtc) && (tc.regrminutes == newtc.regrminutes) && (tc.breakoutstd == newtc.breakoutstd)
                    # not yet bought -> adapt with latest insights
                    tc.buyprice = newtc.buyprice
                    tc.probability = newtc.probability
                    tc.stoplossprice = newtc.stoplossprice
                    tc.sellprice = newtc.sellprice
                    newtc = nothing
                else
                    # outdated buy chance
                    tc.probability = 0.1
                end
            end
            afr = f2.regr[tc.regrminutes]
            fix = Features.featureix(f2, ohlcvix)
            spread = banddeltaprice(afr, fix, tc.breakoutstd)
            if (pivot[ohlcvix] < tc.stoplossprice) && (afr.grad[fix] < 0)
                # stop loss exit due to plunge of price and negative regression line
                tc.sellprice = df.low[ohlcvix]
                tc.probability = 1.0
                @info "stop loss sell for $base due to plunge out of spread ix=$ohlcvix time=$(opentime[ohlcvix]) at regression price of $(afr.regry[fix]) tc: $tc"
            elseif buycompliant001(f2, tc.regrminutes, tc.breakoutstd, ohlcvix)
                tc.sellprice = upperbandprice(afr, fix, tc.breakoutstd)
            elseif afr.grad[fix] < 0
                tc.sellprice = max(lowerbandprice(afr, fix, tc.breakoutstd), tc.buyprice * (1 + combinedbuysellfee))
            else
                tc.sellprice = afr.regry[fix]
            end
            tc.probability = 0.8 # * (1 - max((tc.buyprice - pivot[ohlcvix])/(tc.buyprice - tc.stoplossprice), 0.0))  # probability to reach sell price
            # @info "sell signal $(base) regrminutes=$(tc.regrminutes) breakoutstd=$(tc.breakoutstd) at price=$(round(tc.sellprice;digits=3)) ix=$ohlcvix  time=$(opentime[ohlcvix])"
        end
    end
    if !isnothing(newtc)
        tradechances.basedict[base] = newtc
    end

    # TODO use case of breakout rise following with tracker window is not yet covered - implemented in traderules002!
    # TODO use case of breakout rise after sell above deviation range is not yet covered
    return tradechances
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

function buycompliant002(f2, window, ohlcvix)
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    afr = f2.regr[window]
    fix = Features.featureix(f2, ohlcvix)

    pastix = ohlcvix - Int64(ceil(window / 4))
    pastfix = Features.featureix(f2, pastix)
    ok =  (
        (minutesgain(afr.grad[fix], df.close[ohlcvix], window) > tr002default.minimumbuygradientdict[window]) &&
        alllargerwindowgradimprove(f2, window, fix) &&
        # (afr.grad[fix] > afr.grad[pastfix]) &&
        (minutesgain(f2.regr[1*60].grad[fix], df.close[ohlcvix], 1*60) > tr002default.minimumbuygradientdict[1*60]) &&
        (minutesgain(f2.regr[24*60].grad[fix], df.close[ohlcvix], 24*60) > tr002default.minimumbuygradientdict[24*60]))

    return ok
end

"""
Returns the best performing regression window within the last backtestminutes period before .
In case that minimumgain requirements are not met, `bestwindow` returns 0 and `maxgain` returns 0.0.
"""
function bestregrwindow002(f2::Features.Features002, ohlcvix, backtestminutes)
    maxtrades = 0
    maxgain = tr002default.minimumbacktestgain
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
                while  (lastix < eix) && !buycompliant002(f2, window, Features.ohlcvix(f2, lastix))
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
function bestbacktestwindow002(f2::Features.Features002, ohlcvix)
    maxgain = tr002default.minimumbacktestgain
    bestwindow = 0
    bestbacktestminutes = 0
    for backtestminutes in reverse(Features.regressionwindows002)
        window, gain, trades = bestregrwindow002(f2, ohlcvix, backtestminutes)
        if (gain > maxgain) && (window > 0)
            if bestbacktestminutes > 0
                if gain > (1 + tr002default.shorterwindowimprovement) * maxgain
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

function newbuychance002(f2::Features.Features002, ohlcvix)
    df = Ohlcv.dataframe(f2.ohlcv)
    # opentime = df[!, :opentime]
    base = Ohlcv.basesymbol(f2.ohlcv)
    tc = nothing
    fix = Features.featureix(f2, ohlcvix)
    regrminutes, bestbacktestminutes = bestbacktestwindow002(f2, ohlcvix)
    if regrminutes > 0 # best window found
        if buycompliant002(f2, regrminutes, ohlcvix)
            # @info "buy signal $base price=$(round(df.low[ohlcvix];digits=3)) window=$regrminutes ix=$ohlcvix time=$(opentime[ohlcvix])"
            afr = f2.regr[regrminutes]
            # spread = banddeltaprice(afr, fix, breakoutstd)
            buyprice = df.pivot[ohlcvix]
            sellprice = buyprice * 1.02
            # probability to reach buy price
            # probability = max(min((spread - (df.low[ohlcvix] - buyprice)) / spread, 1.0), 0.0)
            probability = 0.8
            tcstoplossprice = stoplossprice(afr, fix, 3.0)
            tc = TradeChance001(base, nothing, buyprice, 0, sellprice, 0, probability, regrminutes, tcstoplossprice, 1.0)
        end
    end
    return tc
end

"""
    traderules002!(tradechances, f2::Features.Features002, ohlcvix)

returns the chance expressed in gain between currentprice and future sellprice * probability
if tradechances === nothing then an empty TradeChance001 array is created and with results returned

Trading strategy:
- buy if price is
    - exceeds a minimum gradient after reaching a minimum
    - use regression window that performd best in the backtest time window and that satisfies minimum profit requirement
    - use different backtest time windows but shorter time windows have to outperform the longer ones by >20% to be considered
    - only regression time windows that are <= backtest time window are considered
- sell if price is at maximum == gradient of regression is zero

"""
function traderules002!(tradechances, f2::Features.Features002, ohlcvix)
    @info "$(@doc traderules002!)" maxlog=1
    @info "tr002default: $(tr002default)" maxlog=1
    tradechances = isnothing(tradechances) ? TradeChances001(Dict(), Dict()) : tradechances
    if isnothing(f2); return tradechances; end
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(f2.ohlcv)
    cleanupnewbuychance!(tradechances, base)
    newtc = newbuychance002(f2, ohlcvix)
    for tc in values(tradechances.orderdict)
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(f2))
            if isnothing(tc.buydt)
                if !isnothing(newtc) && (tc.regrminutes == newtc.regrminutes) # && (tc.breakoutstd == newtc.breakoutstd)
                    # not yet bought -> adapt with latest insights
                    tc.buyprice = newtc.buyprice
                    tc.probability = newtc.probability
                    tc.stoplossprice = newtc.stoplossprice
                    tc.sellprice = newtc.sellprice
                    newtc = nothing
                else
                    # outdated buy chance
                    tc.probability = 0.1
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
                tc.probability = 1.0
            else  # if pivot[ohlcvix] > afr.regry[ohlcvix]  # above normal deviations
                tc.sellprice = pivot[ohlcvix] * 1.02  # normally not reachable within a minute or we catch a peak
                # probability to reach sell price
                tc.probability = 0.8 # * (1 - max((tc.buyprice - pivot[ohlcvix])/(tc.buyprice - lowerbandprice(afr, fix, 2.0)), 0.0))  # disputable
                # @info "sell signal $(base) regrminutes=$(tc.regrminutes) breakoutstd=$(tc.breakoutstd) at price=$(round(tc.sellprice;digits=3)) ix=$ohlcvix  time=$(opentime[ohlcvix])"
            end
        end
    end
    if !isnothing(newtc)
        tradechances.basedict[base] = newtc
    end
    return tradechances
end

"""
    traderules000!(tradechances, f2::Features.Features002, ohlcvix)

returns the chance expressed in gain between currentprice and future sellprice * probability
if tradechances === nothing then an empty TradeChance001 array is created and with results returned

Trading strategy:
- buy if regression line is at minimum
- sell if regression line is at maximum == gradient of regression is zero

"""
function traderules000!(tradechances, f2::Features.Features002, ohlcvix)
    regressionminutes = 24*60
    @info "$(@doc traderules000!)" regressionminutes maxlog=1
    tradechances = isnothing(tradechances) ? TradeChances001(Dict(), Dict()) : tradechances
    if isnothing(f2); return tradechances; end
    @assert f2.firstix < ohlcvix <= f2.lastix "$(f2.firstix) < $ohlcvix <= $(f2.lastix)"
    df = Ohlcv.dataframe(f2.ohlcv)
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(f2.ohlcv)
    cleanupnewbuychance!(tradechances, base)
    featureix = Features.featureix(f2, ohlcvix)
    afr = f2.regr[regressionminutes]
    newtc = nothing
    if featureix > 1
        if (afr.grad[featureix-1] <= 0) && (afr.grad[featureix] > 0)
            buyprice = df.pivot[ohlcvix]
            sellprice = buyprice * 1.1
            probability = 0.8
            tcstoplossprice = buyprice * 0.8
            newtc = TradeChance001(base, nothing, buyprice, 0, sellprice, 0, probability, regressionminutes, tcstoplossprice, 1.0)
        end
    end

    for tc in values(tradechances.orderdict)
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(f2))
            if isnothing(tc.buydt)
                if !isnothing(newtc) && (tc.regrminutes == newtc.regrminutes) # && (tc.breakoutstd == newtc.breakoutstd)
                    # not yet bought -> adapt with latest insights
                    tc.buyprice = newtc.buyprice
                    tc.probability = newtc.probability
                    tc.stoplossprice = newtc.stoplossprice
                    tc.sellprice = newtc.sellprice
                    newtc = nothing
                else
                    # outdated buy chance
                    tc.probability = 0.1
                end
            end
            afr = f2.regr[tc.regrminutes]
            if (afr.grad[featureix-1] >= 0) && (afr.grad[featureix] < 0)
                    tc.sellprice = df.pivot[ohlcvix]
                tc.probability = 1.0
            end
        end
    end
    if !isnothing(newtc)
        tradechances.basedict[base] = newtc
    end
    return tradechances
end


"""
Returns the trade performance percentage of trade sigals given in `signals` applied to `prices`.
"""
function tradeperformance(prices, signals)
    fee = 0.002  # 0.2% fee for each trade
    initialcash = cash = 100.0
    startprice = 1.0
    asset = 0.0

    for ix in 1:size(prices, 1)
        if (signals[ix] == "long") && (cash > 0)
                asset = cash / prices[ix] * (1 - fee)
                cash = 0.0
        elseif (asset > 0) && ((signals[ix] == "close") || (signals[ix] == "short"))
                cash = asset * prices[ix] * (1 - fee)
                asset = 0.0
        # elseif enableshort && (signals[ix] == "short") && (cash > 0)
            # to be done
        end
    end
    if asset > 0
        cash = asset * prices[end] * (1 - fee)
    end
    return (cash - initialcash) / initialcash * 100
end

function researchmodels()
    # filter(model) = model.is_supervised && model.target_scitype >: AbstractVector{<:Continuous}
    # models(filter)[4]

    filter(model) = model.is_supervised && model.is_pure_julia && model.target_scitype >: AbstractVector{<:Continuous}
    models(filter)[4]

    # models("regressor")
end

function get_probs(y::AbstractArray)
    counts     = Dict{eltype(y), Float64}()
    n_elements = length(y)

    for y_k in y
        if  haskey(counts, y_k)
            counts[y_k] +=1
        else
            counts[y_k] = 1
        end
    end

    for k in keys(counts)
        counts[k] = counts[k]/n_elements
    end
    return counts
end

function prepare(labelthresholds)
    if EnvConfig.configmode == test
        x, y = TestOhlcv.sinesamples(20*24*60, 2, [(150, 0, 0.5)])
        fdf, featuremask = Features.getfeatures(y)
        _, grad = Features.rollingregression(y, 50)
    else
        ohlcv = Ohlcv.defaultohlcv("btc")
        Ohlcv.setinterval!(ohlcv, "1m")
        Ohlcv.read!(ohlcv)
        y = Ohlcv.pivot!(ohlcv)
        println("pivot: $(typeof(y)) $(length(y))")
        fdf, featuremask = Features.getfeatures(ohlcv.df)
        _, grad = Features.rollingregression(y, 12*60)
    end
    fdf = Features.mlfeatures(fdf, featuremask)
    fdf = Features.polynomialfeatures!(fdf, 2)
    # fdf = Features.polynomialfeatures!(fdf, 3)

    labels, relativedist, distances, regressionix, priceix = Targets.continuousdistancelabels(y, grad, labelthresholds)
    # labels, relativedist, distances, priceix = Targets.continuousdistancelabels(y)
    # df = DataFrames.DataFrame()
    # df.x = x
    # df.y = y
    # df.grad = grad
    # df.dist = relativedist
    # df.pp = priceix
    # df.rp = regressionix
    println("size(features): $(size(fdf)) size(relativedist): $(size(relativedist))")
    # println(features[1:3,:])
    # println(relativedist[1:3])
    labels = CategoricalArray(labels, ordered=true)
    println(get_probs(labels))
    levels!(labels, Targets.labellevels)
    # println(levels(labels))
    return labels, relativedist, fdf, y
end

function pls1(relativedist, features, train, test)
    featuressrc = source(features)
    stdfeaturesnode = MLJ.transform(machine(Standardizer(), featuressrc), featuressrc)
    fit!(stdfeaturesnode, rows=train)
    ftest = features[test, :]
    # println(ftest[1:10, :])
    stdftest = stdfeaturesnode(rows=test)
    # println(stdftest[1:10, :])

    relativedistsrc = source(relativedist)
    stdlabelsmachine = machine(Standardizer(), relativedistsrc)
    stdlabelsnode = MLJBase.transform(stdlabelsmachine, relativedistsrc)
    # fit!(stdlabelsnode, rows=train)

    plsnode =  predict(machine(PartialLeastSquaresRegressor.PLSRegressor(n_factors=20), stdfeaturesnode, stdlabelsnode), stdfeaturesnode)
    yhat = inverse_transform(stdlabelsmachine, plsnode)
    fit!(yhat, rows=train)
    return yhat(rows=test), stdftest
end

function pls2(relativedist, features, train, test)
    featuressrc = source(features)
    stdfeaturesnode = MLJ.transform(machine(Standardizer(), featuressrc), featuressrc)
    fit!(stdfeaturesnode, rows=train)
    ftest = features[test, :]
    # println(ftest[1:10, :])
    stdftest = stdfeaturesnode(rows=test)
    # println(stdftest[1:10, :])

    relativedistsrc = source(relativedist)
    stdlabelsmachine = machine(Standardizer(), relativedistsrc)
    stdlabelsnode = MLJBase.transform(stdlabelsmachine, relativedistsrc)
    fit!(stdlabelsnode, rows=train)

    # plsnode =  predict(machine(PartialLeastSquaresRegressor.PLSRegressor(n_factors=20), stdfeaturesnode, stdlabelsnode), stdfeaturesnode)
    plsmodel = TunedModel(models=[PartialLeastSquaresRegressor.PLSRegressor(n_factors=20)], resampling=CV(nfolds=3), measure=rms)
    plsnode =  predict(machine(plsmodel, stdfeaturesnode, stdlabelsnode), stdfeaturesnode)
    yhat = inverse_transform(stdlabelsmachine, plsnode)
    fit!(yhat, rows=train)
    return yhat(rows=test), stdftest
end

function pls3(relativedist, features, train, test)
    featuressrc = source(features)
    stdfeaturesnode = MLJ.transform(machine(Standardizer(), featuressrc), featuressrc)
    fit!(stdfeaturesnode, rows=train)
    ftest = features[test, :]
    # println(ftest[1:10, :])
    stdftest = stdfeaturesnode(rows=test)
    # println(stdftest[1:10, :])

    relativedistsrc = source(relativedist)
    stdlabelsmachine = machine(Standardizer(), relativedistsrc)
    stdlabelsnode = MLJBase.transform(stdlabelsmachine, relativedistsrc)
    fit!(stdlabelsnode, rows=train)

    # plsnode =  predict(machine(PartialLeastSquaresRegressor.PLSRegressor(n_factors=20), stdfeaturesnode, stdlabelsnode), stdfeaturesnode)
    plsmodel = PartialLeastSquaresRegressor.PLSRegressor(n_factors=20)
    plsmachine =  machine(plsmodel, stdfeaturesnode, stdlabelsnode)
    e = evaluate!(plsmachine, resampling=CV(nfolds=3), measure=[rms, mae], verbosity=1)
    println(e)
    plsnode =  predict(plsmachine, stdfeaturesnode)
    yhat = inverse_transform(stdlabelsmachine, plsnode)
    return yhat(rows=test), stdftest
end

# function plssimple(relativedist, features, train, test)
#     pls_model = @pipeline Standardizer PartialLeastSquaresRegressor.PLSRegressor(n_factors=8) target=Standardizer
#     pls_machine = machine(pls_model, features, relativedist)
#     fit!(pls_machine, rows=train)
#     yhat = predict(pls_machine, rows=test)
#     return yhat
# end

function printresult(target, yhat)
    df = DataFrame()
    # println("target: $(size(target)) $(typeof(target)), yhat: $(size(yhat)) $(typeof(yhat))")
    # println("target: $(typeof(target)), yhat: $(typeof(yhat))")
    df[:, :target] = target
    df[:, :predict] = yhat
    df[:, :mae] = abs.(df[!, :target] - df[!, :predict])
    println(df[1:3,:])
    predictmae = mae(yhat, target) #|> mean
    # println("mean(df.mae)=$(sum(df.mae)/size(df,1))  vs. predictmae=$predictmae")
    println("predictmae=$predictmae")
    return df
end

function regression1()
    lt = Targets.defaultlabelthresholds
    labels, relativedist, features, y = prepare(lt)
    train, test = partition(eachindex(relativedist), 0.8) # 70:30 split
    # train, test = partition(eachindex(relativedist), 0.7, stratify=labels) # 70:30 split
    println("training: $(size(train,1))  test: $(size(test,1))")

    # models(matching(features, relativedist))
    # Regr = @load BayesianRidgeRegressor pkg=ScikitLearn  #load model class
    # regr = Regr()  #instatiate model

    # building a pipeline with scaling on data
    println("hello")
    # println("typeof(regressor) $(typeof(regressor))")
    yhat1, stdfeatures = pls1(relativedist, features, train, test)
    predictlabels = Targets.getlabels(yhat1, lt)
    predictlabels = CategoricalArray(predictlabels, ordered=true)
    levels!(predictlabels, Targets.labellevels)

    # confusion_matrix(predictlabels, labels[test])
    printresult(relativedist[test], yhat1)
    # yhat2 = plssimple(relativedist, features, train, test)
    # printresult(relativedist[test], yhat2)
    println("label performance $(round(tradeperformance(y[test], labels[test]); digits=1))%")
    println("trade performance $(round(tradeperformance(y[test], predictlabels); digits=1))%")
    # tay = ["$((labels[test])[ix])" for ix in 1:size(y, 1)]
    # tayhat = ["$(predictlabels[ix])" for ix in 1:size(y, 1)]
    # traces = [
    #     scatter(y=y[test], x=x[test], mode="lines", name="input"),
    #     # scatter(y=stdfeatures, x=x[test], mode="lines", name="std input"),
    #     scatter(y=relativedist[test], x=x[test], text=labels[test], mode="lines", name="target"),
    #     scatter(y=yhat1, x=x[test], text=predictlabels, mode="lines", name="predict")
    # ]
    # plot(traces)
end

# EnvConfig.init(production)
# EnvConfig.init(test)

end  # module
