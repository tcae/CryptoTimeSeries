"""
Train and evaluate the trading signal classifiers
"""
module Classify

using DataFrames, Logging  # , MLJ
using MLJ, MLJBase, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics
using PlotlyJS, WebIO, Dates, DataFrames
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv
export traderules001!, tradeperformance

mutable struct TradeRules001
    minimumgain
    breakoutstd
    minimumgradient
    emergencystd
    breakoutstdset
end

shortestwindow = minimum(Features.regressionwindows002)
# tr001default = TradeRules001(0.02, 1.0, 0.0, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
tr001default = TradeRules001(0.01, 1.0, 0.0001, 3.0, [1.0, 1.1])  # for test purposes

mutable struct TradeChance001
    base::String
    buyix  # remains 0 until actually bought
    buyprice
    buytime  # to be updated by
    buyregrprice
    buyorderid  # remains 0 until order issued
    sellprice
    sellix  # remains 0 until actually sold
    sellorderid  # remains 0 until order issued
    probability
    regrminutes
    emergencysellprice
    breakoutstd
end

mutable struct TradeChances001
    basedict::Dict  # hierachy of dict() tradechances[base][regrminutes][breakoutstd]
end
mutable struct TradeChance002
    base::String
    pricecurrent
    pricetarget
    probability
    chanceix
    regrminutes
    trackerminutes
end

function Base.show(io::IO, tc::TradeChance001)
    print(io::IO, "tc: base=$(tc.base), buyix=$(tc.buyix), buytime=$(tc.buytime), buy=$(round(tc.buyprice; digits=2)), buyregr=$(round(tc.buyregrprice; digits=2)), buyid=$(tc.buyorderid) sell=$(round(tc.sellprice; digits=2)), sellid=$(tc.sellorderid), prob=$(round(tc.probability*100; digits=2)), window=$(tc.regrminutes), emergencysell=$(round(tc.emergencysellprice; digits=2)), breakoutstd=$(tc.breakoutstd)")
end

function Base.show(io::IO, tcs::TradeChances001)
    for tc in tcs
        println(tc)
    end
end

function Base.show(io::IO, tc::TradeChance002)
    print(io::IO, "trade chance: base=$(tc.base) current=$(tc.pricecurrent) target=$(tc.pricetarget) probability=$(tc.probability)")
end

function Base.iterate(tradechances::TradeChances001)
    for (base, rmdict) in tradechances.basedict
        for (rm, bodict) in rmdict
            for (bo, tc) in bodict
                println("initial: $e / $e1")
                return (tc, (base, rm, bo))
            end
        end
    end
    return nothing
end

function Base.iterate(tradechances::TradeChances001, state)
    found = false
    for (base, rmdict) in tradechances.basedict
        for (rm, bodict) in rmdict
            for (bo, tc) in bodict
                if found
                    return (tc, (base, rm, bo))
                end
                if state == (base, rm, bo)
                    found = true
                end
            end
        end
    end
    return nothing
end

function buycompliant(f2, window, breakoutstd, ix, approx)
    df = Ohlcv.dataframe(f2.ohlcv)
    afr = f2.regr[window]
    spread = banddeltaprice(afr, ix, breakoutstd)
    lowerprice = lowerbandprice(afr, ix, breakoutstd * approx)
    ok =  ((df.low[ix] < lowerprice) &&
        (spread >= tr001default.minimumgain) &&
        (afr.grad[ix] > tr001default.minimumgradient))
    return ok
end

"""
Returns the best performing combination of spread window and breakoutstd factor.
In case that minimumgain requirements are not met, `bestwindow` returns 0 and `breakoutstd` returns 0.0.
"""
function bestspreadwindow(f2::Features.Features002, currentix, minimumgain, breakoutstdset)
    @assert length(breakoutstdset) > 0
    maxtrades = 0
    maxgain = minimumgain
    bestwindow = 0
    bestbreakoutstd = 0.0
    for breakoutstd in breakoutstdset
        for window in keys(f2.regr)
                trades, gain = calcspread(f2, window, currentix, breakoutstd)
            if gain > maxgain  # trades > maxtrades
                maxgain = gain
                maxtrades = trades
                bestwindow = window
                bestbreakoutstd = breakoutstd
            end
        end
    end
    return bestwindow, bestbreakoutstd
end

upperbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] + stdfactor * fr.medianstd[ix]
lowerbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] - stdfactor * fr.medianstd[ix]
banddeltaprice(fr::Features.Features002Regr, ix, stdfactor) = 2 * stdfactor * fr.medianstd[ix]

"""
Returns the number of trades within the last `requiredminutes` and the gain achived.
In case that minimumgain requirements are not met by fr.medianstd, `trades` and `gain` return 0.
"""
function calcspread(f2::Features.Features002, window, currentix, breakoutstd)
    fr = f2.regr[window]
    gain = 0.0
    trades = 0
    @assert 1 <= currentix <= length(fr.medianstd)
    medianstd = fr.medianstd[currentix]
    if buycompliant(f2, window, breakoutstd, currentix, 1.0)
        startix = max(1, currentix - Features.requiredminutes + 1)
        breakoutix = breakoutextremesix!(nothing, f2.ohlcv, fr.medianstd, fr.regry, breakoutstd, startix)
        xix = [ix for ix in breakoutix if startix <= abs(ix)  <= currentix]
        # println("breakoutextremesix @ window $window breakoutstd $breakoutstd : $xix")
        buyix = sellix = 0
        contributors = []
        for ix in xix
            # first minimum as buy if sell is closed
            buyix = (buyix == 0) && (sellix == 0) && (ix < 0) && buycompliant(f2, window, breakoutstd, abs(ix), 1.0) ? ix : buyix
            # first maximum as sell if buy was done
            sellix = (buyix < 0) && (sellix == 0) && (ix > 0) ? ix : sellix
            if buyix < 0 < sellix
                trades += 1
                thisgain = (upperbandprice(fr, sellix, breakoutstd) - lowerbandprice(fr, abs(buyix), breakoutstd)) / lowerbandprice(fr, abs(buyix), breakoutstd)
                gain += thisgain
                push!(contributors, (buyix, sellix, thisgain))
                buyix = sellix = 0
            end
        end
        # println("contributors @ window $window breakoutstd $breakoutstd gain $gain: $contributors")
    end
    return trades, gain
end

function breakoutextremesix!(extremeix, ohlcv, medianstd, regry, breakoutstd, startindex)
    @assert startindex > 0
    @assert !isnothing(ohlcv)
    df = ohlcv.df
    if startindex > size(df, 1)
        @warn "unepected startindex beyond length of search vector *ohlcv*" startindex size(df, 1)
        return extremeix
    end
    if isnothing(extremeix)
        extremeix = Int32[]
    end
    breakoutix = 0  # negative index for minima, positive index for maxima, else 0
    for ix in startindex:size(df, 1)
        if breakoutix <= 0
            if df[ix, :high] > regry[ix] + breakoutstd * medianstd[ix]
                push!(extremeix, ix)
            end
        end
        if breakoutix >= 0
            if df[ix, :low] < regry[ix] - breakoutstd * medianstd[ix]
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

function registerbuy!(tc::TradeChance001, buyix, buyprice, buyorderid, features::Features.Features002)
    @assert buyix > 0
    @assert buyprice > 0
    afr = features.regr[tc.regrminutes]
    tc.buyix = buyix
    opentime = Ohlcv.dataframe(features.ohlcv)[!, :opentime]
    if 0 < buyix <= size(opentime, 1)
        tc.buytime = opentime[buyix]
    else
        @warn "cannot find buyix $buyix in opentime of len=$(size(opentime, 1))"
    end
    tc.buyprice = buyprice
    tc.buyregrprice = afr.regry[buyix]
    tc.buyorderid = buyorderid
    tc.emergencysellprice = lowerbandprice(afr, buyix, tr001default.emergencystd)
    # tc.emergencysellprice = afr.regry[buyix] - tr001default.emergencystd * afr.medianstd[buyix]
    spread = banddeltaprice(afr, buyix, tc.breakoutstd)
    tc.sellprice = upperbandprice(afr, buyix, tc.breakoutstd)
    # tc.sellprice = afr.regry[buyix] + halfband
    # probability to reach sell price
    tc.probability = max(min((1.5 * spread - (tc.sellprice - buyprice)) / spread, 1.0), 0.0)
    return tc
end

function delete!(tradechances::TradeChances001, tc::TradeChance001)
    if tc.base in keys(tradechances.basedict)
        rmdict = tradechances.basedict[tc.base]
        if tc.regrminutes in keys(rmdict)
            bodict = rmdict[tc.regrminutes]
            if tc.breakoutstd in keys(bodict)
                delete!(bodict, tc.breakoutstd)
            else
                @warn "cannot find breakout $(tc.breakoutstd) in tradechances[$(tc.base)][$(tc.regrminutes)]"
            end
        else
            @warn "cannot find regrminutes $(tc.regrminutes) in tradechances[$(tc.base)]"
        end
    else
        @warn "cannot find base $(tc.base) in tradechances"
    end
    return tradechances
end

function tradechance(tradechances, orderid)
    for tc in tradechances
        if (tc.buyorderid == orderid) || (tc.sellorderid == orderid)
            return tc
        end
    end
    return nothing
end

"""
returns the chance expressed in gain between currentprice and future sellprice * probability
if tradechances === nothing then an empty TradeChance001 array is created and with results returned

Trading strategy:
- buy if price is
    - below normal deviation range of spread regression window
    - spread gradient is OK
    - spread satisfies minimum profit requirements
- sell if price is above normal deviation range of spread regression window
- emergency sell: if regression price < buy regression price - emergencystd * std
- spread gradient is OK = spread gradient > `minimumgradient`
- normal deviation range = regry +- breakoutstd * std = band around regry of std * 2 * breakoutstd
- spread satisfies minimum profit requirements = normal deviation range >= minimumgain

"""
function traderules001!(tradechances, features::Features.Features002, currentix)
    tradechances = isnothing(tradechances) ? TradeChances001(Dict()) : tradechances
    # hierachy of dict() tradechances[base][regrminutes][breakoutstd]
    if isnothing(features); return tradechances; end
    df = Ohlcv.dataframe(features.ohlcv)
    opentime = Ohlcv.dataframe(features.ohlcv)[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(features.ohlcv)
    temptc = nothing
    regrminutes, breakoutstd = bestspreadwindow(features, currentix, tr001default.minimumgain, tr001default.breakoutstd)
    if (regrminutes > 0) && buycompliant(features, regrminutes, breakoutstd, currentix, 0.75)
        # with 0.75*breakoutstd df.low is close enough to lower buy price to issue buy order
        @info "buy signal $base price=$(df.low[currentix]) window=$regrminutes ix=$currentix time=$(opentime[currentix])"
        afr = features.regr[regrminutes]
        spread = banddeltaprice(afr, currentix, breakoutstd)
        buyprice = lowerbandprice(afr, currentix, breakoutstd)
        sellprice = upperbandprice(afr, currentix, breakoutstd)
        # probability to reach buy price
        probability = max(min((spread - (df.low[currentix] - buyprice)) / spread, 1.0), 0.0)
        emergencysellprice = afr.regry[currentix] - tr001default.emergencystd * afr.medianstd[currentix]
        temptc = TradeChance001(base, 0, buyprice, opentime[currentix], afr.regry[currentix], 0, sellprice, 0, probability, regrminutes, emergencysellprice, breakoutstd)
    end
    checked = Set()
    newtradechance = !isnothing(temptc)
    if base in keys(tradechances.basedict)
        rmdict = tradechances.basedict[base]
        for (regrminutes, bodict) in rmdict
            for (breakoutstd, tc) in bodict
                if tc.buyix == 0
                    if newtradechance && (regrminutes == temptc.regrminutes) && (breakoutstd == temptc.breakoutstd)
                        # not yet bought -> adapt with latest insights
                        tc.buyprice = temptc.buyprice
                        tc.buyregrprice = temptc.buyregrprice
                        tc.probability = temptc.probability
                        tc.emergencysellprice = temptc.emergencysellprice
                        tc.sellprice = temptc.sellprice
                        newtradechance = false
                    elseif tc.buyorderid == 0
                        # outdated buy chance and no buy order issued
                        delete!(bodict, breakoutstd)
                    end
                else
                    if tc.sellorderid == 0
                        @warn "tradechance bought without sell order" tc
                    end
                    afr = features.regr[tc.regrminutes]
                    spread = banddeltaprice(afr, currentix, tc.breakoutstd)
                    if afr.regry[currentix] < tc.emergencysellprice  # (afr.regry[currentix] - tr001default.emergencystd * afr.medianstd[currentix])
                        # emergency exit due to plunge of regression line
                        println(tc)
                        tc.sellprice = df.low[currentix]
                        tc.probability = 1.0
                        @info "emergency sell for $base due to plunge out of spread regrminutes=$(tc.regrminutes) ix=$currentix time=$(opentime[currentix]) at regression price of $(afr.regry[currentix]) and sell price of $(tc.sellprice)"
                    else  # if pivot[currentix] > afr.regry[currentix]  # above normal deviations
                        tc.sellprice = upperbandprice(afr, currentix, tc.breakoutstd)
                        # probability to reach sell price
                        tc.probability = max(min((spread - (tc.sellprice - pivot[currentix])) / spread, 1.0), 0.0)
                        @info "sell signal for $(base) regrminutes=$(tc.regrminutes) breakoutstd=$(tc.breakoutstd) at price=$(tc.sellprice) ix=$currentix  time=$(opentime[currentix])"
                    end
                end
            end

        end
    elseif newtradechance
        tradechances.basedict[base] = Dict(regrminutes => Dict(breakoutstd => temptc))
    end
    # TODO use case of breakout rise following with tracker window is not yet covered - implemented in traderules002!
    # TODO use case of breakout rise after sell above deviation range is not yet covered
    return tradechances
end

"""
Returns the regression window with the least number of regressionextremes
between the last (opposite of deviation range) breakout and the current index.
"""
function besttrackerwindow(f2::Features.Features002, regrminutes, currentix)
    #! requires redesign because afr.xtrmix is removed
    pivot = Ohlcv.dataframe(f2.ohlcv)[!, :pivot]
    minextremes = currentix  # init with impossible high nbr of extremes
    bestwindow = 1
    boix = max(currentix - regrminutes, 1)  # in case there is no opposite spread extreme

    afr = f2.regr[regrminutes]
    upwards = pivot[currentix] > afr.regry[currentix]
    for xix in length(afr.xtrmix):-1:1
        ix = afr.xtrmix[xix]
        if (abs(ix) < currentix) && (upwards ? ix < 0 : ix > 0)
            boix = ix  # == index of opposite spread window extreme
            break
        end
    end

    for win in Features.regressionwindows002
        tfr = f2.regr[win]
        if (win < regrminutes) && (upwards ? tfr.regry[currentix] > afr.regry[currentix] : tfr.regry[currentix] < afr.regry[currentix])
            xix = [abs(ix) for ix in tfr.xtrmix if boix <= abs(ix) <= currentix]
            if (length(xix) == 0) ||
               ((length(xix) > 0) &&
                (upwards ? tfr.regry[xix[1]] < afr.regry[currentix] : tfr.regry[xix[1]] > afr.regry[currentix]) &&
                (length(xix) < minextremes))
                minextremes = length(xix)
                bestwindow = win
            end
        end
    end
    return bestwindow
end

"""
returns the chance expressed in gain between currentprice and future sellprice * probability

Trading strategy:
- buy if price is below normal deviation range of spread regression window, spread gradient is OK and tracker gradient becomes positive
- sell if price is above normal deviation range of spread regression window and tracker gradient becomes negative
- emergency sell after buy if price plunges below extended deviation range of spread regression window
- spread regression window = std * 2 * breakoutstd > minimumgain
- spread gradient is OK = spread gradient > `minimumgradient`
- normal deviation range = regry +- breakoutstd * std
- extended deviation range  = regry +- emergencystd * std

"""
function traderules002!(tradechances::Vector{TradeChance002}, features::Features.Features002, currentix)
    pivot = Ohlcv.dataframe(features.ohlcv)[!, :pivot]
    base = Ohlcv.basesymbol(features.ohlcv)
    checked = Set()
    cachetc = TradeChance002[]
    for tc in tradechances
        if tc.base == base
            # only check exit criteria for existing orders
            if !(tc.regrminutes in checked)
                union!(checked, tc.regrminutes)
                afr = features.regr[tc.regrminutes]
                if pivot[currentix] > (afr.regry[currentix] + tr001default.breakoutstd * afr.medianstd[currentix])  # above normal deviations
                    if tc.trackerminutes == 0
                        tc.trackerminutes = besttrackerwindow(features, tc.regrminutes, currentix)
                    end
                end
                if (((tc.trackerminutes == 1) && (pivot[currentix] < pivot[currentix-1])) ||
                    ((tc.trackerminutes > 1) && (features.regr[tc.trackerminutes].grad[currentix] < 0)))
                    @info "sell signal tracker window $(base) $(pivot[currentix]) $(tc.regrminutes) $(tc.trackerminutes) $currentix"
                    pricetarget = afr.regry[currentix] - tr001default.breakoutstd * afr.medianstd[currentix]
                    prob = 0.8
                    push!(cachetc, TradeChance002(base, pivot[currentix], pricetarget, prob, currentix, tc.regrminutes, tc.trackerminutes))
                end
                if pivot[currentix] < (afr.regry[currentix] - tr001default.emergencystd * afr.medianstd[currentix])
                    # emergency exit due to surprising plunge
                    @info "emergency sell signal due toplunge out of spread window" base tc.regrminutes currentix
                    pricetarget = afr.regry[currentix] - 2 * tr001default.emergencystd * afr.medianstd[currentix]
                    prob = 0.9
                    push!(cachetc, TradeChance002(base, pivot[currentix], pricetarget, prob, currentix, tc.regrminutes, 1))
                end
            end
        end
    end
    append!(tradechances, cachetc)

    regrminutes = bestspreadwindow(features, currentix, tr001default.minimumgain, tr001default.breakoutstd)
    afr = features.regr[regrminutes]
    if ((pivot[currentix] < (afr.regry[currentix] - tr001default.breakoutstd * afr.medianstd[currentix])) &&
        (afr.grad[currentix] > tr001default.minimumgradient))
        trackerminutes = besttrackerwindow(features, regrminutes, currentix)
        if (((trackerminutes == 1) && (pivot[currentix] > pivot[currentix-1])) ||
            ((trackerminutes > 1) && (features.regr[trackerminutes].grad[currentix] > 0)))
            @info "buy signal tracker window $base $(pivot[currentix]) $regrminutes $trackerminutes $currentix"
            pricetarget = afr.regry[currentix] + tr001default.breakoutstd * afr.medianstd[currentix]
            prob = 0.8
            push!(tradechances, TradeChance002(base, pivot[currentix], pricetarget, prob, currentix, regrminutes, trackerminutes))
        end
    end
    # TODO use case of breakout rise after sell above deviation range is not yet covered
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
    df.target = target
    df.predict = yhat
    df.mae = abs.(df.target - df.predict)
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

function loadclassifierhardwired(base, features::Features.Features002)
    return
end

function loadclassifier(base, features::Features.Features001)
    return loadclassifierhardwired(base, features)
end

# EnvConfig.init(production)
# EnvConfig.init(test)
# regression1()

end  # module
