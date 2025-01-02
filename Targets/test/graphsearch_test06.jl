using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test, CSV, Logging, LoggingExtras

using EnvConfig, Features, Targets, TestOhlcv, Ohlcv

all_logger = ConsoleLogger(stderr, Logging.BelowMinLevel)
logger = EarlyFilteredLogger(all_logger) do args
    r = Logging.Debug <= args.level < Logging.AboveMaxLevel && args._module === Targets
    # r = Logging.Info <= args.level < Logging.AboveMaxLevel && args._module === Targets
    return r
end

with_logger(logger) do

    #! not checked yet
# if a long term longbuy exceeding regr extreme is interrupted by a short term longbuy exceeding regr then the short term has priority
ydata = [1.0f0, 1.1f0, 1.1f0, 0.9f0, 0.75f0, 0.8f0, 0.8f0, 0.85f0, 0.8f0, 0.75f0, 0.7f0, 0.6f0, 0.4f0, 0.6f0, 0.65f0, 0.7f0, 0.71f0, 0.75f0, 0.7f0, 0.6f0]
grad1 = [-0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, 0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f1]
grad2 = [0.2f0, 0.1f0, 0.1f0, -0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, 0.2f0, 0.1f0, 0.2f0, 0.2f0, -0.2f0, -0.2f0, 0.0f0]
f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
#! with graph search no heuristic anymore - removed from regression test set
println("nearer extreme exceeding longbuy thresholds should win against further away longbuy threshold exceeding extreme if near term longbuy has at least 50% of the far term longbuy gain")
df = DataFrame()
df.ydata = ydata
df.grad1 = grad1
df.grad2 = grad2
df.realdist = realdist
df.priceix = priceix
df.relativedist = relativedist
df.labels = labels
println(df)
# println("labels = $labels")
# println("relativedist = $relativedist")
# println("realdist = $realdist")
# println("regressionix = $regressionix")
println("priceix = $priceix")
end