using MLJ, Dates, MLJFlux, Flux, DataFrames, PrettyPrinting, JLSO, CategoricalArrays
using Ohlcv, Features, TestOhlcv, Targets, EnvConfig, Classify
using PlotlyJS


#region DataPrep
lth = Targets.LabelThresholds(0.03, 0.0001, -0.0001, -0.03)
features = targets = nothing

folds(data, nfolds) = partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...)

#endregion DataPrep
#region LearningNetwork


#endregion LearningNetwork

#region Evaluation
#endregion Evaluation

mach = loaded_mach = machpred = loadedmachpred = result = nothing
machfile = EnvConfig.logpath("PeriodicTimeSeriesFluxClassifierTuned.jlso")


f3, pe = Classify.loadtestdata()
regrwindow= 5
features, targets = Classify.featurestargets(regrwindow, f3, pe)

# mach, result, machpred, targets = Classify.newmacheval(Classify.clf, f3, pe, regrwindow)

# Classify.savemach(mach, machfile)

# loadedmach = Classify.loadmach(machfile)
# loadedmachpred = predict(loadedmach, features)
# println("loadedmachpred size: $(size(loadedmachpred)) \n ")
# println("machpred ≈ loadedmachpred is $(!isnothing(machpred) ? (!isnothing(loadedmachpred) ? machpred ≈ loadedmachpred : "invalid due to no loadedmachpred") : "invalid due to no machpred")")

mach = Classify.loadmach(machfile)
pred = predict(mach, features)

# Classify.aucscores(pred, targets)
MLJ.confmat(Classify.maxscoreclass(pred), targets)
# test_basecombitestpartitions()