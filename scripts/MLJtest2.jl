using MLJ, RDatasets
using PartialLeastSquaresRegressor

# loading data and selecting some features
data = dataset("datasets", "longley")[:, 2:5]

# unpacking the target
y, X = unpack(data, ==(:GNP), colname -> true)

# loading the model
regressor = PartialLeastSquaresRegressor.PLSRegressor(n_factors=2)

# building a pipeline with scaling on data
pls_model = @pipeline Standardizer regressor target=Standardizer

# a simple hould out
train, test = partition(eachindex(y), 0.7, shuffle=true)

pls_machine = machine(pls_model, X, y)

fit!(pls_machine, rows=train)

yhat = predict(pls_machine, rows=test)

mae(yhat, y[test]) |> mean
