using Dates
# using Features, TestOhlcv, Targets, ROC
# using Plots
using EnvConfig, Classify

EnvConfig.init(test)
EnvConfig.setlogpath("Test_Relu_200epochs_noignore_oversample_nostandardizer_nocv_losscrossentropy")

# for sc in 0.05:0.05:1.05
#     bix = Classify.score2bin(sc, 10)
#     scr = Classify.bin2score(bix, 10)
#     println("sc:$sc bix=$bix sc range=$scr")
# end
# Classify.evaluate("BTC", DateTime("2022-01-02T22:54:00"), Dates.Day(40); select=[5, "combi"]) # nothing)  #
# Classify.evaluate("BTC", DateTime("2017-09-02T22:54:00"), Dates.Day(120); select=[5]) # nothing)  #
# Classify.evaluate("BTC"; select=[5]) # nothing)  #
Classify.evaluatetest(select=[5]) #

println("done")
