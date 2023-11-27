using Dates, Flux, DataFrames
using Ohlcv, Features, TestOhlcv, Targets, EnvConfig, Classify, ROC
using Plots


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


# f3, pe = Targets.loaddata())
startdt = DateTime("2022-01-02T22:54:00")-Dates.Day(20)
enddt = DateTime("2022-01-02T22:54:00")
f3, pe = Targets.loaddata(;startdt=startdt, enddt=enddt, ohlcv=TestOhlcv.testohlcv("sine", startdt, enddt), labelthresholds=lth)
regrwindow= 5
features, targets = Classify.featurestargets(regrwindow, f3, pe)

fm = Classify.adaptmachine(features, targets)
pred = Classify.predict(fm, features)

# Classify.savemach(mach, machfile)

# loadedmach = Classify.loadmach(machfile)
# loadedmachpred = predict(loadedmach, features)
# println("loadedmachpred size: $(size(loadedmachpred)) \n ")
# println("machpred ≈ loadedmachpred is $(!isnothing(machpred) ? (!isnothing(loadedmachpred) ? machpred ≈ loadedmachpred : "invalid due to no loadedmachpred") : "invalid due to no machpred")")

# mach = Classify.loadmach(machfile)
# pred = predict(mach, features)
# println("size(targets)=$(size(targets)) size(features)=$(size(features)) ")

println("auc=$(Classify.aucscores(pred))")
rc = Classify.roccurves(pred)
# for (k, v) in rc
#    println("key: $k size(cutoffs)=$(size(ROC.cutoffs(v)))")
# end
Classify.plotroccurves(rc)
scores, labels = Classify.maxpredictions(pred)
show(stdout, MIME"text/plain"(), Classify.confusionmatrix(pred, targets))  # prints the table

# test_basecombitestpartitions()
