# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# using Pkg
# Pkg.activate(pwd())
# cd(@__DIR__)

using MLJ, PartialLeastSquaresRegressor, CategoricalArrays, Combinatorics
using MLJBase, RDatasets, MLJTuning, MLJModels
using Targets, TestOhlcv
# using RDatasets
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

# labels = CategoricalArray(["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"], ordered=true)
# MLJBase.train_test_pairs(StratifiedCV(nfolds=5), 1:10, labels)
# MLJBase.train_test_pairs(TimeSeriesCV(nfolds=8), 1:10)
# MLJBase.train_test_pairs(CV(nfolds=5), 1:10)

# researchmodels()
# Classify.regression1()
# mlfeatures_test()

@load KPLSRegressor pkg=PartialLeastSquaresRegressor

# loading data and selecting some features
data = RDatasets.dataset("datasets", "longley")[:, 2:5]

# unpacking the target
y, X = unpack(data, ==(:GNP), colname -> true)

# loading the model
pls_model = PartialLeastSquaresRegressor.KPLSRegressor()

# defining hyperparams for tunning
r1 = range(pls_model, :width, lower=0.001, upper=100.0, scale=:log)

# attaching tune
self_tuning_pls_model = TunedModel(model =          pls_model,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(resolution = 100),
                                   range = [r1],
                                   measure = mae)

# putting into the machine
self_tuning_pls = machine(self_tuning_pls_model, X, y)

# fitting with tunning
fit!(self_tuning_pls, verbosity=0)

# getting the report
report(self_tuning_pls)
