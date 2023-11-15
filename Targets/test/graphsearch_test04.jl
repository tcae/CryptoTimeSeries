using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test, CSV, Logging, LoggingExtras
using PlotlyJS, WebIO

using EnvConfig, Features, Targets, TestOhlcv, Ohlcv

all_logger = ConsoleLogger(stderr, Logging.BelowMinLevel)
logger = EarlyFilteredLogger(all_logger) do args
    r = Logging.Debug <= args.level < Logging.AboveMaxLevel && args._module === Targets
    # r = Logging.Info <= args.level < Logging.AboveMaxLevel && args._module === Targets
    return r
end

with_logger(logger) do

# nearer extreme exceeding buy thresholds should win against further away buy threshold exceeding extreme
ydata = [1.0f0, 1.31f0, 1.0f0, 1.2f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0]
grad1 = [0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
grad2 = [-0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.01f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
labels, relativedist, realdist, priceix = Targets.continuousdistancelabels2(f2, Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
# labels, relativedist, realdist, regressionix, priceix, df = Targets.continuousdistancelabels(y, [grad1, grad2], Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
println("nearer extreme exceeding buy thresholds should win against further away buy threshold exceeding extreme")
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