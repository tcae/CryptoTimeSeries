using Dates
# using Features, TestOhlcv, Targets, ROC
# using Plots
using EnvConfig, Classify

EnvConfig.init(production)

# Classify.evaluate("BTC", DateTime("2022-01-02T22:54:00"), Dates.Day(40); select=[5]) # nothing)  #
# Classify.evaluate("BTC", DateTime("2017-09-02T22:54:00"), Dates.Day(120); select=[5]) # nothing)  #
# Classify.evaluate("BTC"; select=[5]) # nothing)  #
Classify.evaluate("BTC"; select=nothing)  # [5, 15, "combi"]) #

# Classify.evaluatepredictions("BTCUSDT_NN5m_23-12-10_17-44-51_gitSHA-206f431207caef04d69b647b2f3d4be98b23b2be")
# Classify.evaluateclassifier("NNcombi_23-12-10_23-34-42_gitSHA-206f431207caef04d69b647b2f3d4be98b23b2be")
println("done")