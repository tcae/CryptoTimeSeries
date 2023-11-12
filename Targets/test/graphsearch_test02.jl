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



    # periodsamples = 150  # minutes per  period
# totalsamples = 20 * 24 * 60  # 20 days in minute frequency
totalsamples = 20
periodsamples = 10  # values per  period
# periodsamples = 10  # minutes per  period
# totalsamples = 60  # 1 hour in minute frequency
yconst = 2.0
x, ydata::Vector{Float32} = TestOhlcv.sinesamples(totalsamples, yconst, [(periodsamples, 0, 0.5)])
_, grad = Features.rollingregression(ydata, Int64(round(periodsamples/2)))

labels1, relativedist1, realdist1, priceix1 = Targets.continuousdistancelabels(ydata, Targets.defaultlabelthresholds)
# labels2, relativedist2, realdist2, regressionix2, priceix2 = Targets.continuousdistancelabels(ydata, [grad, grad], Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
# labels2, relativedist2, realdist2, regressionix2, priceix2 = Targets.continuousdistancelabels(ydata, grad, Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))

with_logger(logger) do
    f2 = Targets.fakef2fromarrays(ydata, [grad])
    labels2, relativedist2, realdist2, priceix2 = Targets.continuousdistancelabels2(f2, Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
    f2 = Targets.fakef2fromarrays(ydata, [grad, grad])
    labels3, relativedist3, realdist3, priceix3 = Targets.continuousdistancelabels2(f2, Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
end
# labels1, realdist1, x, ydata, priceix1 = prepare1(totalsamples, periodsamples, yconst)
# labels2, realdist2, _, _, priceix2, regressionix2 = prepare2(totalsamples, periodsamples, yconst)
df = DataFrame()
df.x = x
df.ydata = ydata
df.grad = grad
df.realdist1 = realdist1
df.priceix1 = priceix1
df.relativedist1 = relativedist1
df.labels1 = labels1
df.realdist2 = realdist2
df.priceix2 = priceix2
df.priceix3 = priceix3
df.delta = abs.(priceix2) .== abs.(priceix3)
df.realdist2 = relativedist2
df.realdist3 = relativedist3
df.relativedist2 = relativedist2
df.relativedist3 = relativedist3
df.labels2 = labels2
df.labels3 = labels3
# df.regressionix2 = regressionix2
@assert ydata == Ohlcv.dataframe(f2.ohlcv).pivot

println("eltype(realdist1)=$(eltype(realdist1)) eltype(relativedist1)=$(eltype(relativedist1))")
println(df)
# traces = [
#     scatter(y=ydata, x=x, mode="lines", name="input"),
#     # scatter(ydata=stdfeatures, x=x[test], mode="lines", name="std input"),
#     # scatter(ydata=realdist1, x=x, mode="lines", name="realdist1", line_dash="dot"),
#     scatter(y=relativedist2, x=x, mode="lines", name="relativedist2", line_dash="dot"),
#     scatter(y=realdist2, x=x, mode="lines", name="realdist2")
# ]
# p = plot(traces)
# display(p)

# println("labels1 = $labels1")
# println("relativedist1 = $relativedist1")
# println("realdist1 = $realdist1")
# println("priceix1 = $priceix1")

# println("labels2 = $labels2")
# println("relativedist2 = $relativedist2")
# println("realdist2 = $realdist2")
# println("regressionix2 = $regressionix2")
println("priceix3 = $priceix3")

priceix3 == Int32[3, 3, -8, -8, -8, -8, -8, 13, 13, 13, 13, 13, -18, -18, -18, -18, -18, 20, 20, -20]