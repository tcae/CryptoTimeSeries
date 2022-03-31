# using Pkg
# Pkg.add(["LoggingFacilities", "NamedArrays"])

include("../src/targets.jl")

module TargetsTest

using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test, CSV
using PlotlyJS, WebIO

using ..EnvConfig, ..Features, ..Targets, ..TestOhlcv, ..Ohlcv

function regressionlabelsx_test(targetlabelfunction, testdatafilename)
    dffile = projectdir("test", testdatafilename)
    io = CSV.File(dffile, types=Dict(1=>Float32, 2=>Float32, 3=>Int8), comment="#")
    df = DataFrame(io)
    # display(describe(df))
    # println(df.price)
    # println(df.regression)
    targetlabels = targetlabelfunction(df.price, df.regression)
    df.result = targetlabels
    # show(df, allrows=true)
    # println("targetlabels=$targetlabels")
    return targetlabels == df.expectation
end

function regressionlabels1_test()
    dffile = projectdir("test", "regressionlabels1_testdata.csv")
    io = CSV.File(dffile, types=Dict(1=>Float32, 2=>Float32, 3=>Int8), comment="#")
    df = DataFrame(io)
    # display(describe(df))
    # println(df.price)
    # println(df.regression)
    startregr::Float32, endregr::Float32 = Targets.tradegradientthresholds(df.price, df.regression)
    lastgainlossdf = Features.lastgainloss(df.price, df.regression)
    targetlabels = Targets.regressionlabels1(df.price, df.regression)
    df.result = targetlabels
    df.lastgain = lastgainlossdf.lastgain
    df.lastloss = lastgainlossdf.lastloss
    # println("startregr=$startregr  endregr=$endregr")
    # show(df, allrows=true)
    # println("targetlabels=$targetlabels")
    return targetlabels == df.expectation
end

function targets4_test()
    x, y = TestOhlcv.sinesamples(120, 200, [(20, 0, 0.9), (40, 30.0, 0.9)])
    w1 = 6; w2 = 30
    res = Targets.targets4(y, [w1, w2])
    df = DataFrame(price=y, target=res[:target],
                   grad3=res[w1][:gradient], price3=res[w1][:price], gain3=res[w1][:gain], xix3=res[w1][:xix],
                   grad10=res[w2][:gradient], price10=res[w2][:price], gain10=res[w2][:gain], xix10=res[w2][:xix])
    show(df, allrows=true)
end


function prepare1(totalsamples, periodsamples, yconst)
    x, y = TestOhlcv.sinesamples(totalsamples, yconst, [(periodsamples, 0, 0.5)])
    labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(y, Targets.defaultlabelthresholds)
    # df = DataFrames.DataFrame()
    # df.x = x
    # df.y = y
    # df.relativedist = pctdist
    # df.realdist = realdist
    # df.priceix = priceix
    # println(df)
    # println("size(relativedist): $(size(relativedist))")
    return labels, realdist, x, y, priceix
end

function prepare2(totalsamples, periodsamples, yconst)
    x, y = TestOhlcv.sinesamples(totalsamples, yconst, [(periodsamples, 0, 0.5)])
    _, grad = Features.rollingregression(y, Int64(round(periodsamples/2)))

    labels, relativedist, realdist, regressionix, priceix = Targets.continuousdistancelabels(y, grad, Targets.defaultlabelthresholds)
    df = DataFrames.DataFrame()
    # df.x = x
    # df.y = y
    # df.relativedist = relativedist
    # df.realdist = realdist
    # df.regrix = regressionix
    # df.priceix = priceix
    # println(df)
    # println("size(relativedist): $(size(relativedist))")
    return labels, realdist, x, y, priceix, regressionix
end

function continuousdistancelabels_test()
    # periodsamples = 150  # minutes per  period
    # totalsamples = 20 * 24 * 60  # 20 days in minute frequency
    totalsamples = 20
    periodsamples = 10  # values per  period
    # periodsamples = 10  # minutes per  period
    # totalsamples = 60  # 1 hour in minute frequency
    yconst = 2.0
    x, y = TestOhlcv.sinesamples(totalsamples, yconst, [(periodsamples, 0, 0.5)])
    _, grad = Features.rollingregression(y, Int64(round(periodsamples/2)))

    labels1, relativedist1, realdist1, priceix1 = Targets.continuousdistancelabels(y, Targets.defaultlabelthresholds)
    labels2, relativedist2, realdist2, regressionix2, priceix2 = Targets.continuousdistancelabels(y, grad, Targets.defaultlabelthresholds)

    # labels1, realdist1, x, y, priceix1 = prepare1(totalsamples, periodsamples, yconst)
    # labels2, realdist2, _, _, priceix2, regressionix2 = prepare2(totalsamples, periodsamples, yconst)
    df = DataFrame()
    df.x = x
    df.y = y
    df.realdist1 = realdist1
    df.priceix1 = priceix1
    df.relativedist1 = relativedist1
    df.labels1 = labels1
    df.realdist2 = realdist2
    df.priceix2 = priceix2
    df.relativedist2 = relativedist2
    df.labels2 = labels2
    df.regressionix2 = regressionix2
    # println(df)
    # traces = [
    #     scatter(y=y, x=x, mode="lines", name="input"),
    #     # scatter(y=stdfeatures, x=x[test], mode="lines", name="std input"),
    #     scatter(y=realdist1, x=x, mode="lines", name="realdist1", line_dash="dot"),
    #     scatter(y=realdist2, x=x, mode="lines", name="realdist2")
    # ]
    # p = plot(traces)
    # display(p)
    # println(priceix2)
    return priceix2 == [3, 3, 8, 8, 8, 8, 8, 13, 13, 13, 13, 13, 18, 18, 18, 18, 18, 18, 19, 20]
end

function distancetest2()
    # periodsamples = 150  # minutes per  period
    # totalsamples = 20 * 24 * 60  # 20 days in minute frequency
    totalsamples = 20
    periodsamples = 10  # values per  period
    # periodsamples = 10  # minutes per  period
    # totalsamples = 60  # 1 hour in minute frequency
    yconst = 2.0
    labels1, realdist1, x, y, priceix1 = prepare1(totalsamples, periodsamples, yconst)
    labels2, realdist2, _, _, priceix2, regressionix2 = prepare2(totalsamples, periodsamples, yconst)
    df = DataFrame()
    df.x = x
    df.y = y
    df.realdist1 = realdist1
    df.priceix1 = priceix1
    df.realdist2 = realdist2
    df.priceix2 = priceix2
    df.regressionix2 = regressionix2
    # println(df)
    # traces = [
    #     scatter(y=y, x=x, mode="lines", name="input"),
    #     # scatter(y=stdfeatures, x=x[test], mode="lines", name="std input"),
    #     scatter(y=realdist1, x=x, mode="lines", name="realdist1", line_dash="dot"),
    #     scatter(y=realdist2, x=x, mode="lines", name="realdist2")
    # ]
    # p = plot(traces)
    # display(p)
    # println(priceix2)
end

function distancetest1()
    # periodsamples = 150  # minutes per  period
    # totalsamples = 20 * 24 * 60  # 20 days in minute frequency
    periodsamples = 10  # minutes per  period
    totalsamples = 60  # 1 hour in minute frequency
    yconst = 0.0
    labels1, realdist1, x, y = prepare1(totalsamples, periodsamples, yconst)
    df = DataFrame()
    df.x = x
    df.y = y
    df.realdist1 = realdist1
    println(df)
    traces = [
        scatter(y=y, x=x, mode="lines", name="input"),
        # scatter(y=stdfeatures, x=x[test], mode="lines", name="std input"),
        scatter(y=realdist1, x=x, mode="lines", name="target")
    ]
    p = plot(traces)
    display(p)
end



# with_logger(TimestampTransformerLogger(current_logger(), BeginningMessageLocation();
#                                               format = "yyyy-mm-dd HH:MM:SSz")) do
    # regressionlabels1_test()

    EnvConfig.init(test)
    # EnvConfig.init(production)
    println("\nconfig mode = $(EnvConfig.configmode)")
    distancetest2()

    # targets4_test()

    # println("""regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv")=$(regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv"))""")

@testset "Targets tests" begin

    # ohlcv = TestOhlcv.doublesinedata(40, 2)
    # regressions = Features.normrollingregression(ohlcv.df.pivot, 5)

    # @test regressionlabels1_test()
    # @test regressionlabelsx_test(Targets.regressionlabels1, "regressionlabels1_testdata.csv")
    # @test regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv")
    # @test regressionlabelsx_test(Targets.regressionlabels3, "regressionlabels3_testdata.csv")
    @test continuousdistancelabels_test()
end  # of testset

# end  # of with logger
end  # module
