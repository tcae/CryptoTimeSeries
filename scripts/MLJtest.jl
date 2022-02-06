cd("$(@__DIR__)/..")
# println("activated $(pwd())")
using Pkg
Pkg.activate(pwd())
cd(@__DIR__)
include("../src/targets.jl")
include("../src/testohlcv.jl")

using MLJ, PartialLeastSquaresRegressor, CategoricalArrays
using ..Targets, ..TestOhlcv
import DataFrames
using RDatasets

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

function prepare()
    x, y = TestOhlcv.sinesamples(20*24*60, 2, [(150, 0, 0.5)])
    _, grad = Features.rollingregression(y, 50)
    fdf, featuremask = Features.features001set(y)
    labels, pctdist, distances, regressionix, priceix = Targets.continuousdistancelabels(y, grad)
    # df = DataFrames.DataFrame()
    # df.x = x
    # df.y = y
    # df.grad = grad
    # df.dist = pctdist
    # df.pp = priceix
    # df.rp = regressionix
    features = DataFrames.DataFrame()
    for feature in featuremask
        features[:, feature] = fdf[!, feature]
    end
    println("size(features): $(size(features)) size(pctdist): $(size(pctdist))")
    # println(features[1:3,:])
    # println(pctdist[1:3])
    labels = CategoricalArray(labels, ordered=true)
    println(get_probs(labels))
    levels!(labels, ["sell", "hold", "buy"])
    # println(levels(labels))
    return labels, pctdist, features
end

function regression1()
    labels, pctdist, features = prepare()
    train, test = partition(eachindex(pctdist), 0.7, stratify=labels) # 70:30 split
    println("training: $(size(train,1))  test: $(size(test,1))")

    # models(matching(features, pctdist))
    # Regr = @load BayesianRidgeRegressor pkg=ScikitLearn  #load model class
    # regr = Regr()  #instatiate model

    # building a pipeline with scaling on data
    println("hello")
    # println("typeof(regressor) $(typeof(regressor))")

    pls_model = @pipeline Standardizer PartialLeastSquaresRegressor.PLSRegressor(n_factors=2) target=Standardizer

    pls_machine = machine(pls_model, features, pctdist)

    fit!(pls_machine, rows=train)

    yhat = predict(pls_machine, rows=test)
    df = DataFrame()
    df.target = pctdist[test]
    df.predict = yhat
    df.mae = abs.(df.target - df.predict)
    println(df[1:3,:])
    println("mean(df.mae)=$(sum(df.mae)/size(df,1))")

    mae(yhat, pctdist[test]) #|> mean

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
