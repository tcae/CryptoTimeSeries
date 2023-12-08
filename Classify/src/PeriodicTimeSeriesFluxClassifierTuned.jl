using Dates
# using Ohlcv, Features, TestOhlcv, Targets, EnvConfig, ROC
# using Plots
using Classify

Classify.evaluate("BTC", DateTime("2017-08-02T22:54:00"), Dates.Year(7); select=nothing)  # [5, 15, "combi"]) #

# Classify.evaluatetest()
println("done")