
using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/features.jl")

module FeaturesTest
using Dates, DataFrames
using Test

using ..Config, ..Ohlcv, ..Features


function config_test()
    ohlcv = Ohlcv.read("test")
    display(ohlcv)
    Features.f4condagg!(ohlcv)
    display(ohlcv)
end

function lastextremes_test()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*0.98]
    regressions = [missing, 0.02,   0.11,  -0.05,  -0.05,   0.02,  -0.01]
    df = Features.lastextremes(prices, regressions)
    refdf = DataFrame(
        pricemax = Float32[0.0, -0.02970297, -0.08411215, 0.04901961, 0.13829787, 0.05940594, 0.030612245],
        timemax = Float32[ 0.0,  1.0,         2.0,        1.0,        2.0,        3.0,        1.0],
        pricemin = Float32[0.0, 0.02970297, 0.08411215, 0.039215688, -0.04255319, 0.06930693, 0.040816326],
        timemin = Float32[ 0.0, 1.0,        2.0,        3.0,          4.0,        1.0,        2.0])
    # println(df)
    # println(refdf)
    # println(df.pricemax)
    # println(df.timemax)
    # println(df.pricemin)
    # println(df.timemin)
    return df == refdf
end

function gradientgaphistogram_test()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*0.98]
    regressions = [missing, 0.02,   0.11,  -0.05,  -0.05,   0.02,  -0.01]
    Features.gradientgaphistogram(prices, regressions, 5)
end

Config.init(test)
# config_test()
# Features.executeconfig()
# gradientgaphistogram_test()
# display(Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4))
@testset "Features tests" begin

@test abs(Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)[7] - 0.310714285714285) < 10^-7
@test isapprox(Features.normrollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4)[4:7], Features.normrollingregression2([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4)[4:7], atol=10^-8)
@test Features.relativevolume([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 3, 5) == [1.0555555555555556; 1.0526315789473684; 1.025]
@test lastextremes_test()
@test Features.regressionaccelerationhistory([missing, 0.1, 0.25, -0.15, -0.3, 0.2, 0.1]) == [0.0 0.0 1.0 -1.0 -2.0 1.0 -1.0]

end

end  # module