module TargetsTest

using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test, CSV
using PlotlyJS, WebIO

using EnvConfig, Features, Targets, TestOhlcv, Ohlcv

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
    x, y::Vector{Float32} = TestOhlcv.sinesamples(totalsamples, yconst, [(periodsamples, 0, 0.5)])
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

    totalsamples = 20  # minutes
    periodsamples = 10  # minutes per  period
    yconst = 2.0  # y elevation = level
    x, y::Vector{Float32} = TestOhlcv.sinesamples(totalsamples, yconst, [(periodsamples, 0, 0.5)])
    _, grad = Features.rollingregression(y, Int64(round(periodsamples/2)))

    labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(y, Targets.defaultlabelthresholds)
    @test labels == ["longbuy", "longbuy", "close", "shortbuy", "shortbuy", "shortbuy", "shortbuy", "close", "longbuy", "longbuy", "longbuy", "longbuy", "close", "shortbuy", "shortbuy", "shortbuy", "close", "longbuy", "longbuy", "close"]
    @test relativedist ≈ [0.19209163f0, 0.07337247f0, 0.0f0, -0.6238597f0, -0.5047131f0, -0.31192985f0, -0.11914659f0, 0.0f0, 0.38418326f0, 0.31081077f0, 0.19209163f0, 0.07337247f0, 0.0f0, -0.45098034f0, -0.3445183f0, -0.17225915f0, 0.0f0, 0.10646201f0, 0.10646201f0, 0.0f0]
    @test realdist ≈ [0.47552824f0, 0.18163562f0, 0.0f0, -0.9510565f0, -0.76942086f0, -0.47552824f0, -0.18163562f0, 0.0f0, 0.9510565f0, 0.76942086f0, 0.47552824f0, 0.18163562f0, 0.0f0, -0.76942086f0, -0.58778524f0, -0.29389262f0, 0.0f0, 0.18163562f0, 0.18163562f0, 0.0f0]
    @test priceix == [4, 4, 4, 9, 9, 9, 9, 9, 14, 14, 14, 14, 14, 20, 20, 20, 20, 20, 20, 0]

    labels, relativedist, realdist, regressionix, priceix = Targets.continuousdistancelabels(y, grad, Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
    @test labels == ["longhold", "longhold", "shorthold", "shorthold", "shorthold", "shorthold", "shorthold", "longbuy", "longbuy", "longbuy", "longhold", "longhold", "shorthold", "shorthold", "shorthold", "shorthold", "shorthold", "close", "longhold", "close"]
    @test relativedist ≈ [0.23776412, 0.07918227, -0.38418326, -0.38418326, -0.33542147, -0.23776412, -0.10646201, 0.6238597, 0.6238597, 0.45098034, 0.23776412, 0.07918227, -0.38418326, -0.38418326, -0.33542147, -0.23776412, -0.10646201, 0.0, 0.11914659, 0.0]
    @test realdist ≈ [0.47552824, 0.18163562, -0.9510565, -0.9510565, -0.76942086, -0.47552824, -0.18163562, 0.9510565, 0.9510565, 0.76942086, 0.47552824, 0.18163562, -0.9510565, -0.9510565, -0.76942086, -0.47552824, -0.18163562, 0.0, 0.18163562, 0.0]
    @test regressionix == [6, 6, -11, -11, -11, -11, -11, 16, 16, 16, 16, 16, -20, -20, -20, -20, -20, -20, -20, -20]
    @test priceix == [3, 3, -8, -8, -8, -8, -8, 13, 13, 13, 13, 13, -18, -18, -18, -18, -18, -19, -20, -20]

    # regression test that multiple of the same still works
    labels, relativedist, realdist, regressionix, priceix = Targets.continuousdistancelabels(y, [grad, grad], Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
    @test labels == ["longhold", "longhold", "shorthold", "shorthold", "shorthold", "shorthold", "shorthold", "longbuy", "longbuy", "longbuy", "longhold", "longhold", "shorthold", "shorthold", "shorthold", "shorthold", "shorthold", "close", "longhold", "close"]
    @test relativedist ≈ [0.23776412, 0.07918227, -0.38418326, -0.38418326, -0.33542147, -0.23776412, -0.10646201, 0.6238597, 0.6238597, 0.45098034, 0.23776412, 0.07918227, -0.38418326, -0.38418326, -0.33542147, -0.23776412, -0.10646201, 0.0, 0.11914659, 0.0]
    @test realdist ≈ [0.47552824, 0.18163562, -0.9510565, -0.9510565, -0.76942086, -0.47552824, -0.18163562, 0.9510565, 0.9510565, 0.76942086, 0.47552824, 0.18163562, -0.9510565, -0.9510565, -0.76942086, -0.47552824, -0.18163562, 0.0, 0.18163562, 0.0]
    @test regressionix == [6, 6, -11, -11, -11, -11, -11, 16, 16, 16, 16, 16, -20, -20, -20, -20, -20, -20, -20, -20]
    @test priceix == [3, 3, -8, -8, -8, -8, -8, 13, 13, 13, 13, 13, -18, -18, -18, -18, -18, -19, -20, -20]

    # not exceeding buy thresholds should result always in next possible extreme
    y =     [1.0f0, 1.29f0, 1.0f0, 1.29f0, 0.97f0, 1.15f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.1f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0]
    grad1 = [0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
    grad2 = [0.2f0, 0.1f0, -0.1f0, -0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
    labels, relativedist, realdist, regressionix, priceix, df = Targets.continuousdistancelabels(y, [grad1, grad2], Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
    @test labels == ["longhold", "shorthold", "close", "shorthold", "longhold", "close", "longhold", "shorthold", "longhold", "shorthold", "longhold", "shorthold", "longhold", "shorthold", "longhold", "longhold", "longhold", "close", "longhold", "close"]
    @test relativedist ≈ [0.28999996, -0.24806197, -0.029999971, -0.24806197, 0.23711342, 0.043478325, 0.20000005, -0.1666667, 0.20000005, -0.1666667, 0.20000005, -0.1666667, 0.20000005, -0.1666667, 0.20000005, 0.09090911, 0.20000005, 0.0, 0.20000005, 0.0]
    @test realdist ≈ [0.28999996, -0.31999993, -0.029999971, -0.31999993, 0.23000002, 0.05000007, 0.20000005, -0.20000005, 0.20000005, -0.20000005, 0.20000005, -0.20000005, 0.20000005, -0.20000005, 0.20000005, 0.100000024, 0.20000005, 0.0, 0.20000005, 0.0]
    @test regressionix == [3, -6, -6, -6, 9, 9, 9, -11, 13, -16, 17, -18, 19, -20, 20, 20, 20, 20, 20, 20]
    @test priceix == [2, -5, -5, -5, 8, 8, 8, -9, 10, -11, 12, -13, 14, -15, 18, 18, 18, 20, 20, 20]
    # println("not exceeding buy thresholds should result always in next possible extreme")
    # println(df)
    # println("labels = $labels")
    # println("relativedist = $relativedist")
    # println("realdist = $realdist")
    # println("regressionix = $regressionix")
    # println("priceix = $priceix")

    # nearer extreme not exceeding buy thresholds should not win against further away but buy threshold exceeding extreme
    y =     [1.0f0, 1.2f0, 1.0f0, 1.31f0, 0.91f0, 1.1f0, 1.0f0, 0.9f0, 1.0f0, 1.1f0, 1.0f0, 1.3f0, 1.0f0, 1.2f0, 0.91f0, 1.182f0, 1.0f0, 1.1f0, 1.2f0, 1.3f0]
    grad1 = [0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
    grad2 = [0.2f0, 0.1f0, -0.1f0, -0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
    labels, relativedist, realdist, regressionix, priceix, df = Targets.continuousdistancelabels(y, [grad1, grad2], Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
    # println("nearer extreme not exceeding buy thresholds should not win against further away but buy threshold exceeding extreme")
    # println(df)
    # println("labels = $labels")
    # println("relativedist = $relativedist")
    # println("realdist = $realdist")
    # println("regressionix = $regressionix")
    # println("priceix = $priceix")
    @test labels == ["longbuy", "longhold", "longbuy", "shorthold", "longhold", "shorthold", "shorthold", "longbuy", "longhold", "longhold", "longhold", "shorthold", "close", "shorthold", "longbuy", "longhold", "longhold", "longhold", "longhold", "close"]
    @test relativedist ≈ [0.30999994, 0.09166658, 0.30999994, -0.30534345, 0.2087912, -0.18181822, -0.100000024, 0.44444442, 0.29999995, 0.18181811, 0.29999995, -0.29999995, -0.089999974, -0.24166667, 0.42857134, 0.09983072, 0.29999995, 0.18181811, 0.083333254, 0.0]
    @test realdist ≈ [0.30999994, 0.109999895, 0.30999994, -0.39999992, 0.19, -0.20000005, -0.100000024, 0.39999998, 0.29999995, 0.19999993, 0.29999995, -0.38999993, -0.089999974, -0.29000002, 0.38999993, 0.11799991, 0.29999995, 0.19999993, 0.099999905, 0.0]
    @test regressionix == [6, 6, 6, -6, 9, -11, -11, 13, 13, 13, 13, -16, -16, -16, 20, 20, 20, 20, 20, 20]
    @test priceix == [4, 4, 4, -5, 6, -8, -8, 12, 12, 12, 12, -15, -15, -15, 20, 20, 20, 20, 20, 20]

    # nearer extreme exceeding buy thresholds should win against further away buy threshold exceeding extreme
    y =     [1.0f0, 1.3f0, 1.0f0, 1.3f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0]
    grad1 = [0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
    grad2 = [0.2f0, 0.1f0, -0.1f0, -0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
    labels, relativedist, realdist, regressionix, priceix, df = Targets.continuousdistancelabels(y, [grad1, grad2], Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
    # println("nearer extreme exceeding buy thresholds should win against further away buy threshold exceeding extreme")
    # println(df)
    # println("labels = $labels")
    # println("relativedist = $relativedist")
    # println("realdist = $realdist")
    # println("regressionix = $regressionix")
    # println("priceix = $priceix")
    @test labels == ["longhold", "shorthold", "shorthold", "shorthold", "longbuy", "longhold", "longhold", "shorthold", "longhold", "shorthold", "shorthold", "shorthold", "shorthold", "shorthold", "longbuy", "longhold", "longhold", "shorthold", "longhold", "close"]
    @test relativedist ≈ [0.29999995, -0.3076923, -0.100000024, -0.3076923, 0.44444442, 0.083333254, 0.29999995, -0.2307692, 0.29999995, -0.3076923, -0.100000024, -0.3076923, -0.100000024, -0.3076923, 0.44444442, 0.083333254, 0.29999995, -0.2307692, 0.29999995, 0.0]
    @test realdist ≈ [0.29999995, -0.39999998, -0.100000024, -0.39999998, 0.39999998, 0.099999905, 0.29999995, -0.29999995, 0.29999995, -0.39999998, -0.100000024, -0.39999998, -0.100000024, -0.39999998, 0.39999998, 0.099999905, 0.29999995, -0.29999995, 0.29999995, 0.0]
    @test regressionix == [3, -6, -6, -6, 9, 9, 9, -11, 13, -16, -16, -16, -16, -16, 20, 20, 20, -20, 20, 20]
    @test priceix == [2, -5, -5, -5, 8, 8, 8, -9, 10, -15, -15, -15, -15, -15, 18, 18, 18, -19, 20, 20]

end  # of testset

# end  # of with logger
end  # module
