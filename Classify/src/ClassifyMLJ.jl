"""
Train and evaluate the trading signal classifiers
"""
module ClassifyMLJ

using DataFrames, Logging, Dates, PrettyPrinting
using MLJ, MLJBase, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics, JLSO, MLJFlux, Flux, Statistics, ProgressMeter # , StatisticalMeasures
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv
using Plots

"""
Returns the trade performance percentage of trade sigals given in `signals` applied to `prices`.
"""
function tradeperformance(prices, signals)
    fee = 0.002  # 0.2% fee for each trade
    initialcash = cash = 100.0
    startprice = 1.0
    asset = 0.0

    for ix in eachindex(prices)  # 1:size(prices, 1)
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

    @error "requires revision of deprecated continuousdistancelabels usage - new version with f2 features"
    # labels, relativedist, distances, regressionix, priceix = Targets.continuousdistancelabels(y, grad, labelthresholds)

    # println("size(features): $(size(fdf)) size(relativedist): $(size(relativedist))")
    # # println(features[1:3,:])
    # # println(relativedist[1:3])
    # labels = CategoricalArray(labels, ordered=true)
    # println(get_probs(labels))
    # levels!(labels, levels(Targets.all_labels))
    # # println(levels(labels))
    # return labels, relativedist, fdf, y
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
    levels!(predictlabels, levels(Targets.all_labels))

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


#region DataPrep
# folds(data, nfolds) = partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...)

"""
    - Returns a Dict of setname => vector of row ranges
    - input
        - `rowrange` is the range of rows that are split in subranges, e.g. 2000:5000
        - `samplesets` is a Dict of setname => relative setsize with sum of all setsizes == 1
        - `gapsize` is the number of rows between partitions of different sets that are not included in any partition
        - `relativesubrangesize` is the subrange size relative to `rowrange` of a consecutive range assigned to a setinterval
        - gaps will be removed from a subrange
        - relativesubrangesize * length(rowrange) > 2 * gapsize
        - any relative setsize > 2 * relativesubrangesize
        - a mixture of ranges and individual indices in a vector can be unpacked into a index vector via `[ix for r in rr for ix in r]`
"""
function setpartitions(rowrange, samplesets::Dict, gapsize, relativesubrangesize)
    rowstart = rowrange[1]
    rowend = rowrange[end]
    rows = rowend - rowstart + 1
    @assert relativesubrangesize * rows > 2 * gapsize
    @assert max(collect(values(samplesets))...) > 2 * relativesubrangesize "max(collect(values(samplesets))...)=$(max(collect(values(samplesets))...)) <= 2 * relativesubrangesize = $(2 * relativesubrangesize)"
    minrelsetsize = min(collect(values(samplesets))...)
    @assert minrelsetsize > 0.0
    sn = [setname for setname in keys(samplesets)]
    snl = length(sn)
    ix = rowstart
    aix = 1
    arr =[[] for _ in sn]
    while ix <= rowend
        subrangesize = round(Int, rows * relativesubrangesize * samplesets[sn[aix]] / minrelsetsize)
        push!(arr[aix], ((ix == rowstart ? ix : ix + gapsize), min(rowend, ix+subrangesize-1)))
        ix = ix + subrangesize
        aix = aix % snl + 1
    end
    res = Dict(sn[aix] => [t[1]:t[2]; for t in arr[aix]] for aix in eachindex(arr))
    return res
end

function test_setpartitions()
    # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 0, 1/9)
    # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 1, 1/9)
    res = setpartitions(1:49, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 1, 3/50)
    # 3-element Vector{Vector{Any}}:
    #     [UnitRange{Int64}[1:3, 10:12, 19:21]]
    #     [UnitRange{Int64}[4:6, 13:15, 22:24]]
    #     [UnitRange{Int64}[7:9, 16:18, 25:26]]

    for (k, vec) in res
        # for rangeset in vec  # rangeset not necessary
            for range in vec  # rangeset
                for ix in range
                    # println("$k: $ix")
                end
            end
        # end
    end
    println(res)
end

function basecombitestpartitions(rowcount, micropartions)
    micropartionsize = Int(ceil(rowcount/micropartions, digits=0))
    ix = 1
    aix = 1
    arr =[[],[],[]]
    while ix <= rowcount
        push!(arr[aix], (ix, min(rowcount, ix+micropartionsize-1)))
        ix = ix + micropartionsize
        aix = aix % 3 + 1
    end
    res =[[],[],[]]
    for aix in 1:3
        res[aix] = [[t[1]:t[2]; for t in arr[aix]]] # with rangeset
        res[aix] = [t[1]:t[2]; for t in arr[aix]] # without rangeset
    end
    return res
end

function test_basecombitestpartitions()
    res = basecombitestpartitions(26, 9)
    # 3-element Vector{Vector{Any}}:
    #     [UnitRange{Int64}[1:3, 10:12, 19:21]]
    #     [UnitRange{Int64}[4:6, 13:15, 22:24]]
    #     [UnitRange{Int64}[7:9, 16:18, 25:26]]

    for vec in res
        # for rangeset in vec  # rangeset not necessary
            for range in vec  # rangeset
                for ix in range
                    println(ix)
                end
            end
        # end
    end
    println(res)
end

"returns features as dataframe and targets as CategorialArray (which is why this function resides in Classify and not in Targets)."
function featurestargets(regrwindow, f3::Features.Features003, pe)
    @assert regrwindow in keys(Features.featureslookback01)
    @assert regrwindow in keys(pe)
    f12x, _ = Features.featureslookback01[regrwindow](f3)
    labels, _, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe["combi"])
    f12x.labels = labels[Features.ohlcvix(f3, 1):end]
    f12x = coerce(f12x, :labels=>OrderedFactor)  #  Multiclass
    targets, features = unpack(f12x, ==(:labels))
    levels!(targets, Targets.all_labels)
    @assert size(targets, 1) == size(features, 1)
    return features, targets
end


# apply out of sample stacking: https://burakhimmetoglu.com/2016/12/01/stacking-models-for-improved-predictions/

#endregion DataPrep
#region LearningNetwork

mylosscount=1

function myloss(yestimated, ylabel)
    global mylosscount
    eloss = Flux.crossentropy(yestimated, ylabel)
    if mylosscount < 0 #switched off
        println("yestimated=$yestimated, ylabel=$ylabel, loss=$eloss")
    end
    mylosscount += 1
    return eloss
end

hidden_size1 = 64
hidden_size2 = 32

tsbuilder = MLJFlux.@builder begin
                init=Flux.glorot_uniform(rng)
                hidden1 = Dense(n_in, hidden_size1, relu)
                hidden2 = Dense(hidden_size1, hidden_size2, relu)
                outputlayer = Dense(hidden_size2, n_out)
                Chain(hidden1, hidden2, outputlayer)
            end

nnc = @load NeuralNetworkClassifier pkg=MLJFlux

clf = nnc( builder = tsbuilder,
        finaliser = NNlib.softmax,
        optimiser = Adam(0.001, (0.9, 0.999)),
        loss =  myloss, # Flux.crossentropy,
        epochs = 2,
        batch_size = 2,
        lambda = 0.0,
        alpha = 0.0,
        optimiser_changes_trigger_retraining = false)

function newmacheval(clf, f3, pe, regrwindow)
    features, targets = featurestargets(regrwindow, f3, pe)
    mach = machine(clf, features, targets)
    result = evaluate!(
                mach,
                resampling=CV(),  # Holdout(fraction_train=0.7),
                measure=[cross_entropy],
                verbosity=1)
    # println("size(result.per_observation)=$(size(result.per_observation))")
    # println("size(result.per_observation[1])=$(size(result.per_observation[1]))")
    # println("size(result.per_observation[1][1])=$(size(result.per_observation[1][1]))")
    # pprint(result.fitted_params_per_fold)
    # println()
    # println("result.report_per_fold=$((result.report_per_fold))")
    # println("result.operation=$((result.operation))")
    # println("result.resampling=$((result.resampling))")
    # println("result.repeats=$((result.repeats))")
    # println("size(result.train_test_rows)=$(size(result.train_test_rows))")
    # println("size(result.train_test_rows[1][1])=$(size(result.train_test_rows[1][1]))")
    # println("size(result.train_test_rows[1][2])=$(size(result.train_test_rows[1][2]))")
    println()
    machpred = predict(mach, features)
    # println("machpred size: $(size(machpred)) \n $(machpred[1:5])")
    return mach, result, machpred, targets
end


function savemach(mach, filename)
    smach = serializable(mach)
    JLSO.save(filename, :machine => smach)
end

function loadmach(filename)
    loadedmach = JLSO.load(filename)[:machine]
    # Deserialize and restore learned parameters to useable form:
    restore!(loadedmach)
    return loadedmach
end

#endregion LearningNetwork

#region Evaluation

"returns a (scores, targets) tuple of binary predictions of class `label`"
function binarypredictions(multiclasspred, multiclasstargets, label)
    class_events = MLJ.recode(multiclasstargets, "other", label=>label)
    class_scores = pdf(multiclasspred, [label])[:,1]
    class_scores = UnivariateFinite(["other", label], class_scores, augment=true, pool=class_events)  #* label position within support is important
    # df = DataFrame(String(classes(class_scores)[1]) => pdf(class_scores, classes(class_scores))[:, 1], String(classes(class_scores)[2]) => pdf(class_scores, classes(class_scores))[:, 2], "truth" => class_events)
    # println(df[1:10,:])
    return class_scores, class_events
end

aucscores(pred, targets) = Dict(String(class_label) => auc(binarypredictions(pred, targets, class_label)...) for class_label in unique(targets))
    # auc_scores = []
    # for class_label in unique(targets)
    #     class_scores, class_events = binarypredictions(pred, targets, class_label)
    #     auc_score = auc(class_scores, class_events)
    #     push!(auc_scores, auc_score)
    # end
    # return auc_scores
# end

"Returns a Dict of class => roc tuple of vectors for false_positive_rates, true_positive_rates, thresholds"
roccurves(pred, targets) = Dict(class_label => roc(binarypredictions(pred, targets, class_label)...) for class_label in unique(targets))

function plotroccurves(rc::Dict)
    plotlyjs()
    default(legend=true)
    plt = plot()
    for (k, v) in rc
        # println("$k = $(length(v[1]))  xxx $(length(v[2]))")
        plot!(v[1], v[2], label=k)
    end
    # plot!(xlab="false positive rate", ylab="true positive rate")
    plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black, label="chance")
    xlabel!("false positive rate")
    ylabel!("true positive rate")
    title!("receiver operator characteristic")
end

function maxscoreclass(pred)
    classlabels = classes(pred)
    p = pdf(pred, classlabels)
    maxindex = mapslices(argmax, p, dims=2)
    labelvec = [classlabels[ix] for ix in maxindex]
    return labelvec[:,1]
end

#endregion Evaluation

# EnvConfig.init(production)
# EnvConfig.init(test)

end  # module
