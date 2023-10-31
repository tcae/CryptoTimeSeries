using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test, CSV
using PlotlyJS, WebIO

using EnvConfig, Features, Targets, TestOhlcv, Ohlcv

# if a long term buy exceeding regr extreme is interrupted by a short term buy exceeding regr then the short term has priority except the long term has mor than double gain
y =     [1.0f0, 1.1f0, 1.1f0, 0.9f0, 0.75f0, 0.8f0, 0.8f0, 0.85f0, 0.8f0, 0.75f0, 0.7f0, 0.6f0, 0.4f0, 0.6f0, 0.65f0, 0.7f0, 0.71f0, 0.75f0, 0.7f0, 0.6f0]
grad1 = [-0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, 0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f1]
grad2 = [0.2f0, 0.1f0, 0.1f0, -0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, 0.2f0, 0.1f0, 0.2f0, 0.2f0, -0.2f0, -0.2f0, 0.0f0]
labels, relativedist, realdist, regressionix, priceix, df = Targets.continuousdistancelabels(y, [grad1, grad2], Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
println("nearer extreme exceeding buy thresholds should win against further away buy threshold exceeding extreme if near term buy has at least 50% of the far term buy gain")
println(df)
println("labels = $labels")
println("relativedist = $relativedist")
println("realdist = $realdist")
println("regressionix = $regressionix")
println("priceix = $priceix")
