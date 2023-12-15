using Dates
# using Features, TestOhlcv, Targets, ROC
# using Plots
using EnvConfig, Classify

EnvConfig.init(production)
EnvConfig.setlogpath("ReluV3")

# for sc in 0.05:0.05:1.05
#     bix = Classify.score2bin(sc, 10)
#     scr = Classify.bin2score(bix, 10)
#     println("sc:$sc bix=$bix sc range=$scr")
# end
# Classify.evaluate("BTC", DateTime("2022-01-02T22:54:00"), Dates.Day(40); select=[5]) # nothing)  #
# Classify.evaluate("BTC", DateTime("2017-09-02T22:54:00"), Dates.Day(120); select=[5]) # nothing)  #
# Classify.evaluate("BTC"; select=[5]) # nothing)  #
# Classify.evaluate("BTC"; select=nothing)  # [5, 15, "combi"]) #

# Classify.evaluatepredictions("BTCUSDT_NN5m_23-12-10_17-44-51_gitSHA-206f431207caef04d69b647b2f3d4be98b23b2be")
Classify.evaluateclassifier("NNNN5m_23-12-13_18-00-18_gitSHA-430e70a168bb82e8b3a6b818fa777f33d7a1c060")
println("done")
