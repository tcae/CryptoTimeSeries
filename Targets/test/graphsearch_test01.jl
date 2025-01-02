using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test, CSV

using EnvConfig, Features, Targets, TestOhlcv, Ohlcv
using Logging, LoggingExtras

all_logger = ConsoleLogger(stderr, Logging.BelowMinLevel)
logger = EarlyFilteredLogger(all_logger) do args
    r = Logging.Debug <= args.level < Logging.AboveMaxLevel && args._module === Targets
    # r = Logging.Info <= args.level < Logging.AboveMaxLevel && args._module === Targets
    return r
end

with_logger(logger) do

# not exceeding longbuy thresholds should result always in next possible extreme
ydata = [1.0f0, 1.29f0, 1.0f0, 1.29f0, 0.97f0, 1.15f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.1f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0]
grad1 = [0.2f0, 0.2f0, -0.2f0, -0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
grad2 = [0.2f0, 0.1f0, 0.1f0, 0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
# labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2, Targets.LabelThresholds(0.1, 0.05, -0.1, -0.1))
df = DataFrame()
df.prices = ydata
df.grad1 = grad1
df.grad2 = grad2
df.priceix = priceix
df.realdist = relativedist
df.relativedist = relativedist
df.labels = labels

refpriceix = Int32[2, -5, -5, -5, 6, -7, 12, 12, 12, 12, 12, -13, 16, 16, 16, -17, 18, -19, 20, -20]
println("not exceeding longbuy thresholds should result always in next possible extreme")
println(df)
println("equal to ref = $(refpriceix == priceix)  priceix = $priceix")

end
