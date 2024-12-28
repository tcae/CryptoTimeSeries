using Dates
# using Features, TestOhlcv, Targets, ROC
using EnvConfig, Classify.Classify

EnvConfig.init(production)
EnvConfig.setlogpath("Test_Relu_1000epochsbreak4_3hiddenlayer_ignore_oversample_nostandardizer_nocv_losscrossentropy")
# EnvConfig.setlogpath("Relu_1000epochsbreak4_3hiddenlayer_ignore_oversample_nostandardizer_nocv_losscrossentropy")
# EnvConfig.setlogpath("Relu_200epochs_3hiddenlayer_ignore_oversample_nostandardizer_nocv_losscrossentropy")
# EnvConfig.setlogpath("Relu_noignore_oversample_nostandardizer_nocv_losscrossentropy")

# for sc in 0.05:0.05:1.05
#     bix = Classify.score2bin(sc, 10)
#     scr = Classify.bin2score(bix, 10)
#     println("sc:$sc bix=$bix sc range=$scr")
# end
# Classify.evaluate("BTC", DateTime("2022-01-02T22:54:00"), Dates.Day(40); select=[5, "combi"]) # nothing)  #
# Classify.evaluate("BTC", DateTime("2017-09-02T22:54:00"), Dates.Day(120); select=[5]) # nothing)  #
Classify.evaluate("BTC"; select=[15]) # nothing)  #
# Classify.evaluate("BTC"; select=[5, 15, "combi"]) # nothing)  #
# Classify.evaluate("BTC"; select=nothing)

# Classify.evaluatepredictions("BTCUSDT_NN5m_23-12-20_01-45-40_gitSHA-b02abf01b3a714054ea6dd92d5b683648878b079.jdf")
# Classify.evaluateclassifier("NN5m_24-01-01_14-44-29_gitSHA-959b51958697b9892bd250f88eed9a9f01beec84.bson")
# Classify.evaluateclassifier("NN15m_24-01-03_00-41-17_gitSHA-959b51958697b9892bd250f88eed9a9f01beec84.bson")
println("done")
