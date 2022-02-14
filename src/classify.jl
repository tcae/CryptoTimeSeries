include("../src/targets.jl")
include("../src/testohlcv.jl")

module Classify

using DataFrames, Logging  # , MLJ
using MLJ, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics
using PlotlyJS, WebIO, Dates, DataFrames
using ..Features, ..Targets, ..TestOhlcv


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

"""
Returns a dataframe of features used for machien learning (ml)

- fullfeatures is a superset of features used for machine learning
- mlfeaturenames is a string array with column names of fullfeatures that shall be used for mlfeaturenames
- polynomialconnect is teh polynomial degree of feature combination via multiplication (usually 1 or 2)

polynomialconnect example 3rd degree 4 features
a b c d
ab ac ad bc bd cd
abc abd acd bcd

"""
function mlfeatures(fullfeatures, mlfeaturenames, polynomialconnect)
    features = DataFrames.DataFrame()
    for feature in mlfeaturenames
        features[:, feature] = fullfeatures[!, feature]
    end
    featurenames = names(features)  # copy or ref to names? - copy is required
    for poly in 2:polynomialconnect
        for pcf in Combinatorics.combinations(featurenames, poly)
            combiname = ""
            combifeature = fill(1.0f0, (size(features, 1)))
            for feature in pcf
                combiname = combiname == "" ? feature : combiname * "*" * feature
                combifeature .*= features[:, feature]
            end
            features[:, combiname] = combifeature
        end
    end
    return features
end

function mlfeatures_test()
    fnames = ["a", "b", "c", "x", "d"]
    fnamesselect = ["a", "b", "c", "d"]
    df = DataFrames.DataFrame()
    for f in fnames
        df[:, f] = [1.0f0, 2.0f0]
    end
    # println("input: $df")
    df = mlfeatures(df, fnamesselect, 3)
    # println("mlfeatures output: $df")
    return all(v->(v==2.0), df[2, 1:4]) && all(v->(v==4.0), df[2, 5:10]) && all(v->(v==8.0), df[2, 11:14])
end

function prepare(labelthresholds)
    x, y = TestOhlcv.sinesamples(20*24*60, 2, [(150, 0, 0.5)])
    _, grad = Features.rollingregression(y, 50)
    fdf, featuremask = Features.features001set(y)
    fdf = mlfeatures(fdf, featuremask, 2)
    # fdf = mlfeatures(fdf, featuremask, 3)

    labels, relativedist, distances, regressionix, priceix = Targets.continuousdistancelabels(y, grad, labelthresholds)
    # labels, relativedist, distances, priceix = Targets.continuousdistancelabels(y)
    # df = DataFrames.DataFrame()
    # df.x = x
    # df.y = y
    # df.grad = grad
    # df.dist = relativedist
    # df.pp = priceix
    # df.rp = regressionix
    mlfeatures(fdf, featuremask, 1)
    println("size(features): $(size(fdf)) size(relativedist): $(size(relativedist))")
    # println(features[1:3,:])
    # println(relativedist[1:3])
    labels = CategoricalArray(labels, ordered=true)
    println(get_probs(labels))
    levels!(labels, Targets.labellevels)
    # println(levels(labels))
    return labels, relativedist, fdf, x, y
end

function plsdetail(relativedist, features, train, test)
    featuressrc = source(features)
    stdfeaturesnode = MLJ.transform(machine(Standardizer(), featuressrc), featuressrc)
    fit!(stdfeaturesnode, rows=train)
    ftest = features[test, :]
    # println(ftest[1:10, :])
    stdftest = stdfeaturesnode(rows=test)
    # println(stdftest[1:10, :])

    relativedistsrc = source(relativedist)
    stdlabelsmachine = machine(Standardizer(), relativedistsrc)
    stdlabelsnode = MLJ.transform(stdlabelsmachine, relativedistsrc)
    # fit!(stdlabelsnode, rows=train)

    plsnode = predict(machine(PartialLeastSquaresRegressor.PLSRegressor(n_factors=20), stdfeaturesnode, stdlabelsnode), stdfeaturesnode)
    yhat = inverse_transform(stdlabelsmachine, plsnode)
    fit!(yhat, rows=train)
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
    labels, relativedist, features, x, y = prepare(lt)
    train, test = partition(eachindex(relativedist), 0.7, stratify=labels) # 70:30 split
    println("training: $(size(train,1))  test: $(size(test,1))")

    # models(matching(features, relativedist))
    # Regr = @load BayesianRidgeRegressor pkg=ScikitLearn  #load model class
    # regr = Regr()  #instatiate model

    # building a pipeline with scaling on data
    println("hello")
    # println("typeof(regressor) $(typeof(regressor))")
    yhat1, stdfeatures = plsdetail(relativedist, features, train, test)
    predictlabels = Targets.getlabels(yhat1, lt)
    predictlabels = CategoricalArray(predictlabels, ordered=true)
    levels!(predictlabels, Targets.labellevels)

    confusion_matrix(predictlabels, labels[test])
    printresult(relativedist[test], yhat1)
    # yhat2 = plssimple(relativedist, features, train, test)
    # printresult(relativedist[test], yhat2)
    println("label performance $(round(tradeperformance(y[test], labels[test]); digits=1))%")
    println("trade performance $(round(tradeperformance(y[test], predictlabels); digits=1))%")
    # tay = ["$((labels[test])[ix])" for ix in 1:size(y, 1)]
    # tayhat = ["$(predictlabels[ix])" for ix in 1:size(y, 1)]
    traces = [
        scatter(y=y[test], x=x[test], mode="lines", name="input"),
        # scatter(y=stdfeatures, x=x[test], mode="lines", name="std input"),
        scatter(y=relativedist[test], x=x[test], text=labels[test], mode="lines", name="target"),
        scatter(y=yhat1, x=x[test], text=predictlabels, mode="lines", name="predict")
    ]
    plot(traces)
end

end  # module
