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

    # if a long term buy exceeding regr extreme is interrupted by a short term buy exceeding regr then the short term has priority and the long term focus is resumed if it is still buy exceeded afterwards
ydata = [1.0f0, 1.1f0, 1.31f0, 0.9f0, 0.75f0, 0.8f0, 1.0f0, 0.85f0, 0.8f0, 0.75f0, 0.7f0, 0.6f0, 0.5f0, 0.6f0, 0.65f0, 0.7f0, 0.71f0, 0.75f0, 0.7f0, 0.4f0]
grad1 = [-0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, 0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f1]
grad2 = [0.2f0, 0.1f0, -0.1f0, -0.1f0, -0.01f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, 0.2f0, -0.1f0, 0.1f0, -0.1f0, 0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, -0.02f0]
f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
println("if a long term buy exceeding regr extreme is interrupted by a short term buy exceeding regr then the short term has priority and the long term focus is resumed if it is still buy exceeded afterwards")
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