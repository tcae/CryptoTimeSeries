module TargetsTest

using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test
using Logging, LoggingExtras
using EnvConfig, Features, Targets, TestOhlcv, Ohlcv, CryptoXch

# with_logger(TimestampTransformerLogger(current_logger(), BeginningMessageLocation();
#                                               format = "yyyy-mm-dd HH:MM:SSz")) do

EnvConfig.init(test)
# EnvConfig.init(production)
println("\nconfig mode = $(EnvConfig.configmode)")



all_logger = ConsoleLogger(stderr, Logging.BelowMinLevel)
logger = EarlyFilteredLogger(all_logger) do args
    # r = Logging.Debug <= args.level < Logging.AboveMaxLevel && args._module === Targets
    r = Logging.Info <= args.level < Logging.AboveMaxLevel && args._module === Targets
    return r
end

include("trend_test.jl")
# include("fixeddistancegain_test.jl") fails due to changes in Targets

@testset "Targets Labelthresholds" begin
    nt = Targets.thresholds(Targets.defaultlabelthresholds)
    lt = Targets.thresholds(nt)
    # println("nt=$nt, lt=$lt")
    @test Targets.defaultlabelthresholds == lt
end

@testset "Targets tests" begin

    with_logger(logger) do
        # ohlcv = TestOhlcv.doublesinedata(40, 2)
        # regressions = Features.normrollingregression(ohlcv.df.pivot, 5)


        totalsamples = 20  # minutes
        periodsamples = 10  # minutes per  period
        yconst = 2.0  # y elevation = level
        x, ydata::Vector{Float32} = TestOhlcv.sinesamples(totalsamples, yconst, [(periodsamples, 0, 0.5)])
        _, grad = Features.rollingregression(ydata, Int64(round(periodsamples/2)))

        # CryptoTimeSeries/Targets/test/continuousdistancelabels_test02.jl
        labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(ydata, Targets.defaultlabelthresholds)
        @test labels == ["longbuy", "longbuy", "longclose", "shortbuy", "shortbuy", "shortbuy", "shortbuy", "shortclose", "longbuy", "longbuy", "longbuy", "longbuy", "longclose", "shortbuy", "shortbuy", "shortbuy", "shortclose", "longbuy", "longbuy", "longclose"]
        @test relativedist ≈ [0.23776412, 0.07918227, 0.0, -0.38418326, -0.33542147, -0.23776412, -0.10646201, 0.0, 0.6238597, 0.45098034, 0.23776412, 0.07918227, 0.0, -0.31081077, -0.2562392, -0.14694631, 0.0, 0.11914659, 0.11914659, 0.0]
        @test realdist ≈ [0.47552824f0, 0.18163562f0, 0.0f0, -0.9510565f0, -0.76942086f0, -0.47552824f0, -0.18163562f0, 0.0f0, 0.9510565f0, 0.76942086f0, 0.47552824f0, 0.18163562f0, 0.0f0, -0.76942086f0, -0.58778524f0, -0.29389262f0, 0.0f0, 0.18163562f0, 0.18163562f0, 0.0f0]
        @test priceix == [4, 4, 4, 9, 9, 9, 9, 9, 14, 14, 14, 14, 14, 20, 20, 20, 20, 20, 20, 0]

        # CryptoTimeSeries/Targets/test/continuousdistancelabels_test02.jl
        f2 = Targets.fakef2fromarrays(ydata, [grad])
        labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
        @test priceix == Int32[4, 4, 4, -9, -9, -9, -9, -9, 14, 14, 14, 14, 14, -19, -19, -19, -19, -19, 20, 20] 

        # regression test that multiple of the same still works
        f2 = Targets.fakef2fromarrays(ydata, [grad, grad])
        labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
        @test priceix == Int32[4, 4, 4, -9, -9, -9, -9, -9, 14, 14, 14, 14, 14, -19, -19, -19, -19, -19, 20, 20] 

        # not exceeding longbuy thresholds should result always in next possible extreme => with graph search not the case
        # CryptoTimeSeries/Targets/test/graphsearch_test01.jl
        ydata = [1.0f0, 1.29f0, 1.0f0, 1.29f0, 0.97f0, 1.15f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0, 1.0f0, 1.1f0, 1.0f0, 1.2f0, 1.0f0, 1.2f0]
        grad1 = [0.2f0, 0.2f0, -0.2f0, -0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
        grad2 = [0.2f0, 0.1f0, 0.1f0, 0.1f0, 0.0f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
        f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
        labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.1, -0.6))
        @test priceix == Int32[2, -3, 4, -9, -9, -9, -9, -9, 12, 12, 12, -15, -15, -15, 16, -17, 18, -19, 20, 20]

        # CryptoTimeSeries/Targets/test/graphsearch_test03.jl
        ydata =     [1.0f0, 1.2f0, 1.0f0, 1.29f0, 1.1f0, 1.1f0, 0.8f0, 0.9f0, 1.0f0, 1.1f0, 1.0f0, 1.31f0, 1.0f0, 1.2f0, 0.91f0, 1.182f0, 1.0f0, 1.1f0, 1.2f0, 1.3f0]
        grad1 = [0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.05f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
        grad2 = [0.2f0, 0.1f0, -0.1f0, -0.1f0, 0.0f0, 0.1f0, -0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
        f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
        labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
        # println("nearer extreme not exceeding longbuy thresholds should not win against further away but longbuy threshold exceeding extreme")
        @test priceix == Int32[2, -3, 4, -7, -7, -7, 12, 12, 12, 12, 12, -15, -15, -15, 20, 20, 20, 20, 20, 20]

        # CryptoTimeSeries/Targets/test/graphsearch_test04.jl
        ydata = [1.0f0, 1.31f0, 1.0f0, 1.2f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0, 0.9f0, 1.2f0, 1.0f0, 1.3f0, 1.0f0, 1.3f0]
        grad1 = [0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f0, -0.1f0, -0.2f0, -0.2f0, -0.1f0, 0.0f0, 0.1f0, 0.1f0, 0.1f0, 0.2f0, -0.1f0, -0.2f0, -0.2f0, 0.2f0, 0.2f0, 0.0f0]
        grad2 = [-0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.01f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, -0.2f0, 0.1f0, 0.1f0, -0.1f0, -0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, 0.0f0]
        f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
        labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
        # println("nearer extreme exceeding longbuy thresholds should win against further away longbuy threshold exceeding extreme")
        @test priceix == Int32[2, -5, -5, -5, 8, 8, 8, -9, 12, 12, 12, -15, -15, -15, 20, 20, 20, 20, 20, 20]

        # CryptoTimeSeries/Targets/test/graphsearch_test05.jl
        ydata = [1.0f0, 1.1f0, 1.31f0, 0.9f0, 0.75f0, 0.8f0, 1.0f0, 0.85f0, 0.8f0, 0.75f0, 0.7f0, 0.6f0, 0.5f0, 0.6f0, 0.65f0, 0.7f0, 0.71f0, 0.75f0, 0.7f0, 0.4f0]
        grad1 = [-0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.2f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, -0.1f0, 0.2f0, 0.2f0, 0.2f0, 0.1f0, 0.0f1]
        grad2 = [0.2f0, 0.1f0, -0.1f0, -0.1f0, -0.01f0, 0.1f0, 0.2f0, 0.2f0, -0.1f0, 0.2f0, -0.1f0, 0.1f0, -0.1f0, 0.2f0, -0.1f0, 0.2f0, -0.2f0, 0.2f0, -0.2f0, -0.02f0]
        f2 = Targets.fakef2fromarrays(ydata, [grad1, grad2])
        labels, relativedist, realdist, priceix = Targets.continuousdistancelabels(f2; labelthresholds=Targets.LabelThresholds(0.3, 0.05, -0.05, -0.3))
        # println("if a long term longbuy exceeding regr extreme is interrupted by a short term longbuy exceeding regr then the short term has priority and the long term focus is resumed if it is still longbuy exceeded afterwards")
        @test priceix == Int32[-5, -5, -5, -5, 7, 7, -9, -9, 10, -11, 12, -13, 18, 18, 18, 18, 18, -20, -20, -20]


        # enddt = DateTime("2022-01-02T22:54:00")
        # startdt = enddt - Dates.Day(20)
        # ohlcv = TestOhlcv.testohlcv("DOUBLESINE", startdt, enddt)
        # df = Ohlcv.dataframe(ohlcv)
        # ol = size(df,1)
        # f2 = Features.Features002(ohlcv)
        # labels, relativedist, realdist, pricepeakix = Targets.continuousdistancelabels(f2)
        # @test first(pricepeakix) == -1481 broken=true
        # @test last(pricepeakix) == -28801
        # @test length(pricepeakix) == 28801
    end
end  # of testset

# end  # of with logger
end  # module
