using Dates
# using Features, TestOhlcv, Targets, ROC
# using Plots
using EnvConfig, Classify

EnvConfig.init(production)
# EnvConfig.setlogpath("Relu_ignore_oversample_nostandardizer_nocv_losscrossentropy")
EnvConfig.setlogpath("ix")

# for sc in 0.05:0.05:1.05
#     bix = Classify.score2bin(sc, 10)
#     scr = Classify.bin2score(bix, 10)
#     println("sc:$sc bix=$bix sc range=$scr")
# end
Classify.evaluate("BTC", DateTime("2022-01-02T22:54:00"), Dates.Day(40); select=[5]) # nothing)  #
# Classify.evaluate("BTC", DateTime("2017-09-02T22:54:00"), Dates.Day(120); select=[5]) # nothing)  #
# Classify.evaluate("BTC"; select=[5]) # nothing)  #
# Classify.evaluate("BTC"; select=nothing)  # [5, 15, "combi"]) #

# Classify.evaluatepredictions("BTCUSDT_NN1d_23-12-16_14-06-53_gitSHA-402543a1f9002c04f0129ae473d1e8620286baf6")
# Classify.evaluateclassifier("NNcombi_23-12-16_17-46-41_gitSHA-402543a1f9002c04f0129ae473d1e8620286baf6.bson")
println("done")
