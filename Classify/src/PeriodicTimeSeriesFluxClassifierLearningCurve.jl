using MLJ, Dates, MLJFlux, Flux, DataFrames, PrettyPrinting
using Ohlcv, Features, TestOhlcv, Targets
using PlotlyJS

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(20)
ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
f12x, f3 = Features.features12x5m01(ohlcv)
labels, relativedist, _, _ = Targets.continuousdistancelabels(f3.f2; labelthresholds=Targets.LabelThresholds(0.03, 0.0001, -0.0001, -0.03), regrwinarr=[5])
f12x.labels = labels[Features.ohlcvix(f3, 1):end]
f12x = coerce(f12x, :labels=>OrderedFactor)

y, X = unpack(f12x, ==(:labels))
# println("levels: $(levels(y)) max(relativedist)=$(maximum(relativedist)) min(relativedist)=$(minimum(relativedist))")
levels!(y, Targets.all_labels)

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
        batch_size = 1,
        lambda = 0.0,
        alpha = 0.0,
        optimiser_changes_trigger_retraining = false)

mach = machine(clf, X, y)

r = MLJ.range(clf, :epochs, lower=1, upper=3, scale=:linear)
curve = learning_curve(mach,
                     range=r,
                     resampling=Holdout(fraction_train=0.7),
                     measure=[cross_entropy], verbosity=2
                     )

# fit!(mach, verbosity=2)
# pprint(fitted_params(mach).chain)
# println()

# evaluate!(mach)
# training_loss = cross_entropy(predict(mach, X), y) |> mean
println("curve=$curve")
p = plot(
    scatter(x=curve.parameter_values, y=curve.measurements, mode="markers+lines", name=curve.parameter_name, showlegend=true),
    Layout(xaxis_title="epochs", yaxis_title="Cross Entropy", title_text="epoch convergence")
)

display(p)  # optional
