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

    enddt = DateTime("2022-01-02T22:54:00")
    startdt = enddt - Dates.Day(20)
    ohlcv = TestOhlcv.testohlcv("DOUBLESINEUSDT", startdt, enddt)
    df = Ohlcv.dataframe(ohlcv)
    ol = size(df,1)
    f2 = Features.Features002(ohlcv)
    labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2)  # , Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
    df = DataFrame()
    df.realdist = realdist
    df.priceix = priceix
    df.relativedist = relativedist
    df.labels = labels
    println(describe(df))

    # println("labels = $labels")
    # println("relativedist = $relativedist")
    # println("realdist = $realdist")
    # println("regressionix = $regressionix")
    println("priceix = first: $(priceix[firstindex(priceix):min(3, lastindex(priceix))]) last: $(priceix[max(lastindex(priceix)-3, firstindex(priceix)):end])")
end