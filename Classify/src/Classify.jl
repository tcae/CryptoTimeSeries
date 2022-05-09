"""
Train and evaluate the trading signal classifiers
"""
module Classify

using DataFrames, Logging  # , MLJ
using MLJ, MLJBase, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics
using PlotlyJS, WebIO, Dates, DataFrames
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv
export TradeChance, traderules001!, tradeperformance

mutable struct TradeRules001
    minimumprofit
    anchorbreakoutsigma
    anchorminimumgradient
    anchorsurprisesigma
end

shortestwindow = minimum(Features.regressionwindows002)
# tr001default = TradeRules001(0.02, 1.0, 0.0, 3.0)
tr001default = TradeRules001(0.01, 1.0, -0.9, 4.0)  # for test purposes

mutable struct TradeChance
    base::String
    pricecurrent
    pricetarget
    probability
    chanceix
    anchorminutes
    trackerminutes
end

function emptytradechance()
    println("this is emptytradechance")
    TradeChance("", 0.0, 0.0, 0.0, 0, 0, 0)
end

function Base.show(io::IO, tc::TradeChance)
    print(io::IO, "trade chance: base=$(tc.base) current=$(tc.pricecurrent) target=$(tc.pricetarget) probability=$(tc.probability)")
end


"""
returns the chance expressed in gain between currentprice and future targetprice * probability

Trading strategy:
- buy if price is below normal deviation range of anchor regression window, anchor gradient is OK and tracker gradient becomes positive
- sell if price is above normal deviation range of anchor regression window and tracker gradient becomes negative
- emergency sell after buy if price plunges below extended deviation range of anchor regression window
- anchor regression window = std * 2 * anchorbreakoutsigma > minimumprofit
- anchor gradient is OK = anchor gradient > `anchorminimumgradient`
- normal deviation range = regry +- anchorbreakoutsigma * std
- extended deviation range  = regry +- anchorsurprisesigma * std

"""
function traderules001!(tradechances::Vector{TradeChance}, features::Features.Features002, currentix)
    pivot = Ohlcv.dataframe(features.ohlcv)[!, :pivot]
    base = Ohlcv.basesymbol(features.ohlcv)
    checked = Set()
    cachetc = TradeChance[]
    for tc in tradechances
        if tc.base == base
            # only check exit criteria for existing orders
            if !(tc.anchorminutes in checked)
                union!(checked, tc.anchorminutes)
                afr = features.regr[tc.anchorminutes]
                if pivot[currentix] > (afr.regry[currentix] + tr001default.anchorbreakoutsigma * afr.medianstd[currentix])  # above normal deviations
                    if tc.trackerminutes == 0
                        tc.trackerminutes = Features.besttrackerwindow(features, tc.anchorminutes, currentix, true)
                    end
                end
                if (((tc.trackerminutes == 1) && (pivot[currentix] < pivot[currentix-1])) ||
                    ((tc.trackerminutes > 1) && (features.regr[tc.trackerminutes].grad[currentix] < 0)))
                    @info "sell signal tracker window $(base) $(pivot[currentix]) $(tc.anchorminutes) $(tc.trackerminutes) $currentix"
                    pricetarget = afr.regry[currentix] - tr001default.anchorbreakoutsigma * afr.medianstd[currentix]
                    prob = 0.8
                    push!(cachetc, TradeChance(base, pivot[currentix], pricetarget, prob, currentix, tc.anchorminutes, tc.trackerminutes))
                end
                if pivot[currentix] < (afr.regry[currentix] - tr001default.anchorsurprisesigma * afr.medianstd[currentix])
                    # emrgency exit due to surprising plunge
                    @info "emergency sell signal due toplunge out of anchor window" base tc.anchorminutes currentix
                    pricetarget = afr.regry[currentix] - 2 * tr001default.anchorsurprisesigma * afr.medianstd[currentix]
                    prob = 0.9
                    push!(cachetc, TradeChance(base, pivot[currentix], pricetarget, prob, currentix, tc.anchorminutes, 1))
                end
            end
        end
    end
    append!(tradechances, cachetc)

    anchorminutes = Features.bestanchorwindow(features, currentix, tr001default.minimumprofit, tr001default.anchorbreakoutsigma)
    afr = features.regr[anchorminutes]
    if ((pivot[currentix] < (afr.regry[currentix] - tr001default.anchorbreakoutsigma * afr.medianstd[currentix])) &&
        (afr.grad[currentix] > tr001default.anchorminimumgradient))
        trackerminutes = Features.besttrackerwindow(features, anchorminutes, currentix, false)
        if (((trackerminutes == 1) && (pivot[currentix] > pivot[currentix-1])) ||
            ((trackerminutes > 1) && (features.regr[trackerminutes].grad[currentix] > 0)))
            @info "buy signal tracker window $base $(pivot[currentix]) $anchorminutes $trackerminutes $currentix"
            pricetarget = afr.regry[currentix] + tr001default.anchorbreakoutsigma * afr.medianstd[currentix]
            prob = 0.8
            push!(tradechances, TradeChance(base, pivot[currentix], pricetarget, prob, currentix, anchorminutes, trackerminutes))
        end
    end
    # TODO use case of breakout rise after sell above deviation range is not yet covered
end

# """
# returns the chance expressed in gain between currentprice and future targetprice * probability
# """
# function tradechance(tradechances, features, currentix)::TradeChance
#     pricecurrent = Ohlcv.dataframe(features.ohlcv)[currentix, :pivot]
#     pricetarget = pricecurrent * 1.01
#     prob = 0.5  #! TODO  implementation
#     tc = TradeChance(features.base, pricecurrent, pricetarget, prob)
#     return tc
# end

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
