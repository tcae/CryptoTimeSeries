using Dates
# using Ohlcv, Features, TestOhlcv, Targets, EnvConfig, ROC
# using Plots
using Classify

# Classify.evaluate("BTC", DateTime("2022-01-02T22:54:00"), Dates.Day(40); select=nothing)  # [5, 15, "combi"]) #
Classify.evaluate("BTC"; select=nothing)  # [5, 15, "combi"]) #

# Classify.evaluatetest()
println("done")