using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test, CSV
using PlotlyJS, WebIO

using EnvConfig, Features, Targets, TestOhlcv, Ohlcv

# nearer extreme exceeding buy thresholds should win against further away buy threshold exceeding extreme
y =     [1.0f0, 1.3f0, 1.0f0, 1.3f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0]
grad1 = [0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
grad2 = [0.2f0, 0.1f0, -0.1f0, -0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
labels, relativedist, realdist, regressionix, priceix, df = Targets.continuousdistancelabels(y, [grad1, grad2], Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
println("nearer extreme exceeding buy thresholds should win against further away buy threshold exceeding extreme")
println(df)
# println("labels = $labels")
# println("relativedist = $relativedist")
# println("realdist = $realdist")
# println("regressionix = $regressionix")
println("priceix = $priceix")
