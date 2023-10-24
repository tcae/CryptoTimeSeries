using MLJ, Dates, MLJFlux, Flux, DataFrames
using Ohlcv, Features, TestOhlcv, Targets

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(20)
ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
f12x, f3 = Features.features12x5m01(ohlcv)
labels, relativedist, _, _, _ = Targets.continuousdistancelabels(Features.ohlcvdataframe(f3).pivot, Features.grad(f3, 5), Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
y = relativedist

scitype(y)
schema(f12x)
(f12x, f12xtest) = partition(f12x, 0.7, multi=false)
(y, ytest) = partition(y, 0.7, multi=false)
println(describe(f12x))
# (f12x, f12xtest), (y, ytest) = partition((f12x, y), 0.7, multi=false)
println(typeof(f12x))

input_size = length(names(f12x))
hidden_size1 = 64
hidden_size2 = 32
output_size = 1


builder = MLJFlux.@builder begin
    init=Flux.glorot_uniform(rng)
    Chain(
        Dense(input_size, hidden_size1, relu, init=init),
        Dense(hidden_size1, hidden_size2, relu, init=init),
        Dense(hidden_size2, output_size, init=init),
    )
end

NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux
model = NeuralNetworkRegressor(
    builder=builder,
    rng=123,
    epochs=20
)

pipe = Standardizer |> TransformedTargetModel(model, transformer=Standardizer())


mach = machine(pipe, f12x, y)
fit!(mach, verbosity=2)

## first element initial loss, 2:end per epoch training losses
report(mach).transformed_target_model_deterministic.model.training_losses



# Define the structure of the network:

# input_size = 12
# hidden_size1 = 48
# hidden_size2 = 20
# output_size = 3

# builder = @builder Chain(
#     Dense(input_size, hidden_size1, relu),
#     Dense(hidden_size1, hidden_size2, relu),
#     Dense(hidden_size2, output_size)
# )

#     # Create the model using the defined builder:

# model = MLJFlux.machine(builder)

#     # Define the evaluation metric:

# metric = cross_entropy

#     # Define the resampling strategy:

# resampling = CV(nfolds=6, shuffle=false, stratify=true)

#     # Create the evaluation plan:

# eval_plan = MLJ.@load Evaluator(
#     model=model,
#     resampling=resampling,
#     measure=metric
# )

#     # Train and evaluate the model:

# X = MLJ.table(ff)  # Assuming ff is a DataFrame
# y = MLJ.coerce(ff.target, Multiclass)

# mach = machine(eval_plan, X, y)
# MLJ.fit!(mach)
# MLJ.evaluate!(mach)



# enddt = DateTime("2022-01-02T22:54:00")
# startdt = enddt - Dates.Day(20)
# ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
# df = Ohlcv.dataframe(ohlcv)
# ol = size(df,1)
# y = Ohlcv.pivot!(ohlcv)
# # println("pivot: $(typeof(y)) $(length(y))")
# _, grad = Features.rollingregression(y, 5)  # use 5 minute regression
# labels, relativedist, distances, regressionix, priceix = Targets.continuousdistancelabels(y, grad, labelthresholds)
# labels = CategoricalArray(labels, ordered=true)
# println(get_probs(labels))
# levels!(labels, levels(Targets.possiblelabels()))

