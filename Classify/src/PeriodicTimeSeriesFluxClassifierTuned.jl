using MLJ, Dates, MLJFlux, Flux, DataFrames, PrettyPrinting, MLJBase, JLSO
using Ohlcv, Features, TestOhlcv, Targets, EnvConfig
using PlotlyJS

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(20)
ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
f12x, f3 = Features.features12x5m01(ohlcv)
# labels, relativedist, _, _, _ = Targets.continuousdistancelabels(Features.ohlcvdataframe(f3).pivot, Features.grad(f3, 5), Targets.LabelThresholds(0.03, 0.0001, -0.0001, -0.03))
labels, relativedist, _, _, _ = Targets.continuousdistancelabels(Features.ohlcvdataframe(f3).pivot, Features.grad(f3, 5), Targets.LabelThresholds(0.03, 0.0001, -0.0001, -0.03))
f12x.labels = labels
f12x = coerce(f12x, :labels=>OrderedFactor)

y, X = unpack(f12x, ==(:labels))
# println("levels: $(levels(y)) max(relativedist)=$(maximum(relativedist)) min(relativedist)=$(minimum(relativedist))")
levels!(y, Targets.possiblelabels())

# (f12xtrain, f12xtest) = partition(f12x, 0.7, multi=false)
# y, X = unpack(f12xtrain, ==(:labels))

hidden_size1 = 64
hidden_size2 = 32

tsbuilder = MLJFlux.@builder begin
                init=Flux.glorot_uniform(rng)
                hidden1 = Dense(n_in, hidden_size1, relu)
                hidden2 = Dense(hidden_size1, hidden_size2, relu)
                outputlayer = Dense(hidden_size2, n_out)
                Chain(hidden1, hidden2, outputlayer)
            end


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

mach = loaded_mach = machpred = loadedmachpred = nothing

if true
    mach = machine(clf, X, y)
    result = evaluate!(mach,
                        resampling=CV(),  # Holdout(fraction_train=0.7),
                        measure=[cross_entropy], verbosity=2
                        )
    # MLJ.save("PeriodicTimeSeriesFluxClassifierTuned.mlj", mach)
    smach = serializable(mach)
    JLSO.save(EnvConfig.logpath("PeriodicTimeSeriesFluxClassifierTuned.jlso"), :machine => smach)

    println("size(result.per_observation)=$(size(result.per_observation))")
    println("size(result.per_observation[1])=$(size(result.per_observation[1]))")
    println("size(result.per_observation[1][1])=$(size(result.per_observation[1][1]))")
    pprint(result.fitted_params_per_fold)
    println()
    println("result.report_per_fold=$((result.report_per_fold))")
    println("result.operation=$((result.operation))")
    println("result.resampling=$((result.resampling))")
    println("result.repeats=$((result.repeats))")
    println("size(result.train_test_rows)=$(size(result.train_test_rows))")
    println("size(result.train_test_rows[1][1])=$(size(result.train_test_rows[1][1]))")
    println("size(result.train_test_rows[1][2])=$(size(result.train_test_rows[1][2]))")
    println()
    machpred = predict(mach, X)
    println("machpred size: $(size(machpred)) \n $(machpred[1:5])")

end

if true
    loadedmach = JLSO.load(EnvConfig.logpath("PeriodicTimeSeriesFluxClassifierTuned.jlso"))[:machine]
    # Deserialize and restore learned parameters to useable form:
    restore!(loadedmach)
    loadedmachpred = predict(loadedmach, X)
    println("loadedmachpred size: $(size(loadedmachpred)) \n $(loadedmachpred[1:5])")
end

println("machpred ≈ loadedmachpred is $(!isnothing(machpred) ? (!isnothing(loadedmachpred) ? machpred ≈ loadedmachpred : "invalid due to no loadedmachpred") : "invalid due to no machpred")")

# fit!(mach, verbosity=2)
# pprint(fitted_params(mach).chain)
# println()

# evaluate!(mach)
# training_loss = cross_entropy(predict(mach, X), y) |> mean
