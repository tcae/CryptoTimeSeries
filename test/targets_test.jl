using DrWatson
@quickactivate "CryptoTimeSeries"

include(srcdir("targets.jl"))

module TargetsTest

using Dates, DataFrames
using Test, CSV, DrWatson

using ..Config, ..Features, ..Targets

function gradientgaphistogram_test()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*1.08, p*1.09, p*1.10, p*0.98, p*1.01, p*1.09, p*1.10, p*0.98, p*1.05, p*1.12]
    regressions = [missing, 0.20,   0.11,  -0.05,  -0.05,   0.02,   0.07,   0.02,   0.02,  -0.12,   0.02,   0.02,   0.02,  -0.12,   0.07,   0.07]
    regbuckets = 3
    histo, regquantiles, gainborders = Targets.gradientgaphistogram(prices, regressions, regbuckets)
    # println("len= $(length(regquantiles))  regquantiles=$regquantiles")
    # println("len= $(length(gainborders))  gainborders=$gainborders")
    # zeroref = zeros(Int32, (regbuckets, regbuckets))
    # for ix in 1:size(histo, 3)
    #     println("histo regbucket=$ix")
    #     if histo[:, :, ix] != zeroref
    #         display(histo[:, :, ix])
    #     end
    # end
    return (histo[3, 2, 8] == 1) && (histo[2, 2, 12] == 2) && (histo[2, 3, 12] == 1)
end

function gradientgaphistogram_test2()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*1.08, p*1.09, p*1.10, p*0.98, p*1.01, p*1.09, p*1.10, p*0.98, p*1.05, p*1.12]
    regressions = [missing, 0.20,   0.11,  -0.05,  -0.05,   0.02,   0.07,   0.02,   0.02,  -0.12,   0.02,   0.02,   0.02,  -0.12,   0.07,   0.07]
    regbuckets = 3
    gainrange = 0.02  # 2% = 0.02 considered OK to decided about gradients for target label
    histo, regquantiles, gainborders = Targets.gradientgaphistogram(prices, regressions, regbuckets, gainrange)
    # println("len= $(length(regquantiles))  regquantiles=$regquantiles")
    # println("len= $(length(gainborders))  gainborders=$gainborders")
    # zeroref = zeros(Int32, (regbuckets, regbuckets))
    # for ix in 1:size(histo, 3)
    #     println("histo regbucket=$ix")
    #     if histo[:, :, ix] != zeroref
    #         display(histo[:, :, ix])
    #     end
    # end
    return (histo[3, 2, 4] == 1) && (histo[2, 2, 4] == 2) && (histo[2, 3, 4] == 1)
end

function gradientthresholdlikelihoods_test()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*1.08, p*1.09, p*1.10, p*0.98, p*1.01, p*1.02, p*1.03, p*0.98, p*1.05, p*1.12]
    regressions = [missing, 0.20,   0.11,  -0.05,  -0.05,   0.02,   0.07,   0.02,   0.02,  -0.12,   0.02,   0.02,   0.02,  -0.12,   0.07,   0.07]
    regbuckets = 3
    histo, regquantiles, gainborders = Targets.gradientgaphistogram(prices, regressions, regbuckets)
    lh = Targets.gradientthresholdlikelihoods(histo, regquantiles, gainborders, 0.04)
    # println("len= $(length(regquantiles))  regquantiles=$regquantiles")
    # println("len= $(length(gainborders))  gainborders=$gainborders")
    # zeroref = zeros(Int32, (regbuckets, regbuckets))
    # for ix in 1:size(histo, 3)
    #     println("histo regbucket=$ix")
    #     if histo[:, :, ix] != zeroref
    #         display(histo[:, :, ix])
    #     end
    # end
    # display(lh)
    return (lh[3, 2] == 0.0) && (lh[2, 2] == 0.25) && (lh[2, 3] == 0.25)
end

function tradegradientthreshold_test()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*1.08, p*1.09, p*1.10, p*0.98, p*1.01, p*1.02, p*1.03, p*0.98, p*1.05, p*1.12]
    regressions = [missing, 0.20,   0.11,  -0.05,  -0.05,   0.02,   0.07,   0.02,   0.02,  -0.12,   0.02,   0.02,   0.02,  -0.12,   0.07,   0.07]
    regbuckets = 3
    histo, regquantiles, gainborders = Targets.gradientgaphistogram(prices, regressions, regbuckets)
    lh = Targets.gradientthresholdlikelihoods(histo, regquantiles, gainborders, 0.04)
    max, ix = findmax(lh)
    startregr = ix[1]>1 ? regquantiles[ix[1]-1] : regquantiles[1]
    endregr = ix[2]>1 ? regquantiles[ix[2]-1] : regquantiles[1]
    # println("len= $(length(regquantiles))  regquantiles=$regquantiles")
    # println("len= $(length(gainborders))  gainborders=$gainborders")
    # zeroref = zeros(Int32, (regbuckets, regbuckets))
    # for ix in 1:size(histo, 3)
    #     println("histo regbucket=$ix")
    #     if histo[:, :, ix] != zeroref
    #         display(histo[:, :, ix])
    #     end
    # end
    # display(lh)
    # println("max=$max  maxix=$ix  regquantiles=$regquantiles  startregr=$startregr  endregr=$endregr")
    return (startregr == -0.015) && (endregr == -0.015)
end

function tradegradientthreshold_test2()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*1.08, p*1.09, p*1.10, p*0.98, p*1.01, p*1.02, p*1.03, p*0.98, p*1.05, p*1.12]
    regressions = [missing, 0.20,   0.11,  -0.05,  -0.05,   0.02,   0.07,   0.02,   0.02,  -0.12,   0.02,   0.02,   0.02,  -0.12,   0.07,   0.07]
    regbuckets = 3
    startregr, endregr = Targets.tradegradientthresholds(prices, regressions, regbuckets, 0.04)
    # println("startregr=$startregr  endregr=$endregr")
    return (startregr == -0.015) && (endregr == -0.015)
end

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

# regressionlabels1_test()

# gradientgaphistogram_test2()
Config.init(test)
println("\nconfig mode = $(Config.configmode)")

println("""regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv")=$(regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv"))""")

@testset "Targets tests" begin

@test gradientgaphistogram_test()
@test gradientgaphistogram_test2()
@test gradientthresholdlikelihoods_test()
@test tradegradientthreshold_test()
@test tradegradientthreshold_test2()
@test regressionlabels1_test()
@test regressionlabelsx_test(Targets.regressionlabels1, "regressionlabels1_testdata.csv")
@test regressionlabelsx_test(Targets.regressionlabels2, "regressionlabels2_testdata.csv")
@test regressionlabelsx_test(Targets.regressionlabels3, "regressionlabels3_testdata.csv")

end  # of testset

end  # module