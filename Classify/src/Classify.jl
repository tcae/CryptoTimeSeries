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
    minimumgain  # minimum spread around regression to consider buy
    minimumgradient  # pre-requisite to buy
    stoplossstd  # factor to multiply with std to dtermine stop loss price (only in combi with negative regr gradient)
    breakoutstdset  # set of breakoutstd factors to test when determining the best combi or regregression window and spread
end

shortestwindow = minimum(Features.regressionwindows002)
# tr001default = TradeRules001(0.02, 0.0, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
tr001default = TradeRules001(0.01, 0.0001, 3.0, [0.75, 1.0, 1.25, 1.5, 1.75, 2.0])  # for test purposes
clreported = false

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
    breakoutstd
end

mutable struct TradeChances001
    basedict  # Dict of new buy orders
    orderdict  # Dict of open orders
end

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

function buycompliant(f2, window, breakoutstd, ix)
    df = Ohlcv.dataframe(f2.ohlcv)
    afr = f2.regr[window]
    spreadpercent = banddeltaprice(afr, ix, breakoutstd) / afr.regry[ix]
    lowerprice = lowerbandprice(afr, ix, breakoutstd)
    ok =  ((df.low[ix] < lowerprice) &&
        (spreadpercent >= tr001default.minimumgain) &&
        (afr.grad[ix] > tr001default.minimumgradient))
    return ok
end

"""
Returns the best performing combination of spread window and breakoutstd factor.
In case that minimumgain requirements are not met, `bestwindow` returns 0 and `breakoutstd` returns 0.0.
"""
function bestspreadwindow(f2::Features.Features002, currentix, minimumgain, breakoutstdset)
    global clreported
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
            clreported = currentix < 2 * Features.requiredminutes ? false :  clreported
            if !clreported && (currentix > 2 * Features.requiredminutes)
                # println("currentix=$currentix window=$window breakoutstd=$breakoutstd trades=$trades gain=$gain")
            end
        end
    end
    if (currentix > 2 * Features.requiredminutes) && (bestwindow > 0)
        # println("currentix=$currentix bestwindow=$bestwindow bestbreakoutstd=$bestbreakoutstd maxtrades=$maxtrades maxgain=$maxgain clreported=$(clreported)")
        clreported = true
    end
return bestwindow, bestbreakoutstd
end

upperbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] + stdfactor * fr.medianstd[ix]
lowerbandprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] - stdfactor * fr.medianstd[ix]
stoplossprice(fr::Features.Features002Regr, ix, stdfactor) = fr.regry[ix] - stdfactor * fr.medianstd[ix]
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
    if currentix > (2 * Features.requiredminutes)
        # (currentix > (2 * Features.requiredminutes)) check required for 1 * requiredminutes to calc features and another 1 * requiredminutes to build trade history
        startix = max(1, currentix - Features.requiredminutes + 1)
        breakoutix = breakoutextremesix!(nothing, f2.ohlcv, fr.medianstd, fr.regry, breakoutstd, startix)
        xix = [ix for ix in breakoutix if startix <= abs(ix)  <= currentix]
        # println("breakoutextremesix @ window $window breakoutstd $breakoutstd : $xix")
        buyix = sellix = 0
        for ix in xix
            # first minimum as buy if sell is closed
            buyix = (buyix == 0) && (sellix == 0) && (ix < 0) && buycompliant(f2, window, breakoutstd, abs(ix)) ? ix : buyix
            # first maximum as sell if buy was done
            sellix = (buyix < 0) && (sellix == 0) && (ix > 0) ? ix : sellix
            if buyix < 0 < sellix
                trades += 1
                thisgain = (upperbandprice(fr, sellix, breakoutstd) - lowerbandprice(fr, abs(buyix), breakoutstd)) / lowerbandprice(fr, abs(buyix), breakoutstd)
                gain += thisgain
                buyix = sellix = 0
            end
        end
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

function registerbuy!(tradechances::TradeChances001, buyix, buyprice, buyorderid, features::Features.Features002)
    @assert buyix > 0
    @assert buyprice > 0
    tc = tradechanceoforder(tradechances, buyorderid)
    if isnothing(tc)
        @warn "missing order #$buyorderid in tradechances"
    else
        df = Ohlcv.dataframe(features.ohlcv)
        opentime = df[!, :opentime]
        afr = features.regr[tc.regrminutes]
        tc.buydt = opentime[buyix]
        tc.buyprice = buyprice
        tc.buyorderid = buyorderid
        tc.stoplossprice = stoplossprice(afr, buyix, tr001default.stoplossstd)
        spread = banddeltaprice(afr, buyix, tc.breakoutstd)
        tc.sellprice = upperbandprice(afr, buyix, tc.breakoutstd)
        # tc.sellprice = afr.regry[buyix] + halfband
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

function newbuychance(tradechances::TradeChances001, features::Features.Features002, currentix)
    df = Ohlcv.dataframe(features.ohlcv)
    opentime = df[!, :opentime]
    base = Ohlcv.basesymbol(features.ohlcv)
    tc = nothing
    regrminutes, breakoutstd = bestspreadwindow(features, currentix, tr001default.minimumgain, tr001default.breakoutstdset)
    if regrminutes > 0 # best window found
        if buycompliant(features, regrminutes, breakoutstd, currentix)
            # @info "buy signal $base price=$(round(df.low[currentix];digits=3)) window=$regrminutes ix=$currentix time=$(opentime[currentix])"
            afr = features.regr[regrminutes]
            # spread = banddeltaprice(afr, currentix, breakoutstd)
            buyprice = lowerbandprice(afr, currentix, breakoutstd)
            sellprice = upperbandprice(afr, currentix, breakoutstd)
            # probability to reach buy price
            # probability = max(min((spread - (df.low[currentix] - buyprice)) / spread, 1.0), 0.0)
            probability = 0.8
            tcstoplossprice = stoplossprice(afr, currentix, tr001default.stoplossstd)
            tc = TradeChance001(base, nothing, buyprice, 0, sellprice, 0, probability, regrminutes, tcstoplossprice, breakoutstd)
        else
            # best window found but buy conditions currently not met
        end
    end
    return tc
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
- stop loss sell: if regression price < buy regression price - stoplossstd * std
- spread gradient is OK = spread gradient > `minimumgradient`
- normal deviation range = regry +- breakoutstd * std = band around regry of std * 2 * breakoutstd
- spread satisfies minimum profit requirements = normal deviation range >= minimumgain

"""
function traderules001!(tradechances, features::Features.Features002, currentix)
    tradechances = isnothing(tradechances) ? TradeChances001(Dict(), Dict()) : tradechances
    # hierachy of dict() tradechances[base][regrminutes][breakoutstd]
    if isnothing(features); return tradechances; end
    df = Ohlcv.dataframe(features.ohlcv)
    opentime = df[!, :opentime]
    pivot = df[!, :pivot]
    base = Ohlcv.basesymbol(features.ohlcv)
    cleanupnewbuychance!(tradechances, base)
    newtc = newbuychance(tradechances, features, currentix)
    for tc in values(tradechances.orderdict)
        if tc.base == Ohlcv.basesymbol(Features.ohlcv(features))
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
            afr = features.regr[tc.regrminutes]
            spread = banddeltaprice(afr, currentix, tc.breakoutstd)
            if (pivot[currentix] < tc.stoplossprice) && (afr.grad[currentix] < 0)
                # stop loss exit due to plunge of price and negative regression line
                tc.sellprice = df.low[currentix]
                tc.probability = 1.0
                @info "stop loss sell for $base due to plunge out of spread ix=$currentix time=$(opentime[currentix]) at regression price of $(afr.regry[currentix]) tc: $tc"
            else  # if pivot[currentix] > afr.regry[currentix]  # above normal deviations
                tc.sellprice = upperbandprice(afr, currentix, tc.breakoutstd)
                # probability to reach sell price
                tc.probability = 0.8 * (1 - min((tc.buyprice - pivot[currentix])/(tc.buyprice - tc.stoplossprice), 0.0))
                # @info "sell signal $(base) regrminutes=$(tc.regrminutes) breakoutstd=$(tc.breakoutstd) at price=$(round(tc.sellprice;digits=3)) ix=$currentix  time=$(opentime[currentix])"
            end
        end
    end
    if !isnothing(newtc)
        tradechances.basedict[base] = newtc
    end

    # TODO use case of breakout rise following with tracker window is not yet covered - implemented in traderules002!
    # TODO use case of breakout rise after sell above deviation range is not yet covered
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
