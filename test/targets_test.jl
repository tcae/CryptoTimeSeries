using DrWatson
@quickactivate "CryptoTimeSeries"
using Pkg
Pkg.add(["LoggingFacilities", "NamedArrays"])

include("../test/testohlcv.jl")
include("../src/targets.jl")
include("../src/ohlcv.jl")

module TargetsTest

using Dates, DataFrames, Logging, LoggingFacilities, NamedArrays
using Test, CSV

using ..Config, ..Features, ..Targets, ..TestOhlcv, ..Ohlcv

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

# with_logger(TimestampTransformerLogger(current_logger(), BeginningMessageLocation();
#                                               format = "yyyy-mm-dd HH:MM:SSz")) do
    # regressionlabels1_test()

    Config.init(test)
    Config.init(production)
    println("\nconfig mode = $(Config.configmode)")


    # println("""regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv")=$(regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv"))""")
#=
@testset "Targets tests" begin

    ohlcv = TestOhlcv.doublesinedata(40, 2)
    regressions = Features.normrollingregression(ohlcv.df.pivot, 5)

    # @test regressionlabels1_test()
    # @test regressionlabelsx_test(Targets.regressionlabels1, "regressionlabels1_testdata.csv")
    # @test regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv")
    # @test regressionlabelsx_test(Targets.regressionlabels3, "regressionlabels3_testdata.csv")

end  # of testset
=#
# end  # of with logger
end  # module