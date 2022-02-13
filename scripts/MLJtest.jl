cd("$(@__DIR__)/..")
# println("activated $(pwd())")
using Pkg
Pkg.activate(pwd())
cd(@__DIR__)
include("../src/targets.jl")
include("../src/testohlcv.jl")

using MLJ, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics
using ..Targets, ..TestOhlcv
using RDatasets
using PlotlyJS, WebIO, Dates, DataFrames


function iris1()
    iris = load_iris();
    selectrows(iris, 1:3)  |> pretty
    schema(iris) |> pretty
    iris = DataFrames.DataFrame(iris);
    y, X = unpack(iris, ==(:target); rng=123, wrap_singles=true);

    first(X, 3) |> pretty
    first(y, 3) |> pretty
    # println(y)
    y = y.target
    # models(matching(X,y))
    Tree = @load DecisionTreeClassifier pkg=DecisionTree  #load model class
    tree = Tree()  #instatiate model
    # evaluate(tree, X, y,
    #                 resampling=CV(shuffle=true),
    #                         measures=[log_loss, accuracy],
    #                         verbosity=2)
    mach = machine(tree, X, y)  # wrapping the model in data creates a machine
    train, test = partition(eachindex(y), 0.7); # 70:30 split
    println("training: $(size(train,1))  test: $(size(test,1))")
    fit!(mach, rows=train)
    yhat = predict(mach, X[test,:])
    yhat[3:5]
    log_loss(yhat, y[test]) |> mean
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
    # fdf = mlfeatures(fdf, featuremask, 2)
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
    println("size(features): $(size(fdf)) size(relativedist): $(size(relativedist)) names(features): $(names(fdf))")
    # println(features[1:3,:])
    # println(relativedist[1:3])
    labels = CategoricalArray(labels, ordered=true)
    println(get_probs(labels))
    levels!(labels, ["sell", "close", "hold", "buy"])
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
    println("target: $(typeof(target)), yhat: $(typeof(yhat))")
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
    levels!(predictlabels, ["sell", "close", "hold", "buy"])

    confusion_matrix(predictlabels, labels[test])
    printresult(relativedist[test], yhat1)
    # yhat2 = plssimple(relativedist, features, train, test)
    # printresult(relativedist[test], yhat2)
    traces = [
        scatter(y=y[test], x=x[test], mode="lines", name="input"),
        # scatter(y=stdfeatures, x=x[test], mode="lines", name="std input"),
        scatter(y=relativedist[test], x=x[test], mode="lines", name="target"),
        scatter(y=yhat1, x=x[test], mode="lines", name="predict")
    ]
    plot(traces)


end

function regression2()

    # loading data and selecting some features
    data = dataset("datasets", "longley")[:, 2:5]

    # unpacking the target
    y, X = unpack(data, ==(:GNP), colname -> true)

    # loading the model
    # regressor = PartialLeastSquaresRegressor.PLSRegressor(n_factors=2)

    # building a pipeline with scaling on data
    pls_model = @pipeline Standardizer PartialLeastSquaresRegressor.PLSRegressor(n_factors=2) target=Standardizer

    # a simple hould out
    train, test = partition(eachindex(y), 0.7, shuffle=true)

    pls_machine = machine(pls_model, X, y)

    fit!(pls_machine, rows=train)

    yhat = predict(pls_machine, rows=test)

    mae(yhat, y[test]) |> mean

end

# researchmodels()
regression1()
# mlfeatures_test()
