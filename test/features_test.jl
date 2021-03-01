
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
    regressions = [0.0, 0.02,   0.11,  -0.05,  -0.05,   0.02,  -0.01]
    df = Features.lastextremes(prices, regressions)
    refdf = DataFrame(
        pricemax = Float32[0.0, -0.02970297, -0.08411215, 0.04901961, 0.13829787, 0.05940594, 0.030612245],
        timemax = Float32[ 0.0,  1.0,         2.0,        1.0,        2.0,        3.0,        1.0],
        pricemin = Float32[0.0, -0.02970297, -0.08411215, -0.039215688, 0.04255319, -0.06930693, -0.040816326],
        timemin = Float32[ 0.0, 1.0,        2.0,        3.0,          4.0,        1.0,        2.0])
    # println(df)
    # println(refdf)
    # println(df.pricemax)
    # println(df.timemax)
    # println(df.pricemin)
    # println(df.timemin)

    # dfarr = [df.pricemax, df.timemax, df.pricemin, df.timemin]
    # display(dfarr)
    # diff = [df.pricemax - refdf.pricemax, df.timemax - refdf.timemax, df.pricemin - refdf.pricemin, df.timemin - refdf.timemin]
    # display(diff)
    return isapprox(df, refdf, atol=10^-5)
end

function lastgainloss_test()
    p = 3  #arbitrary
    prices =     [p*0.98, p*1.01, p*1.07, p*1.02, p*0.94, p*1.01, p*0.98]
    regressions = [0.0, 0.02,   0.11,  -0.05,  -0.05,   0.02,  -0.01]
    df = Features.lastgainloss(prices, regressions)
    refdf = DataFrame(
        lastgain = Float32[0.0, 0.0, 0.0, 0.091836736, 0.091836736, 0.091836736, 0.07446808],
        lastloss = Float32[0.0, 0.0, 0.0, 0.0, 0.0, -0.121495, -0.121495])
    # println(df)
    # println(refdf)
    return isapprox(df, refdf, atol=10^-5)
end

Config.init(test)
# config_test()
# Features.executeconfig()
# display(Features.regressionaccelerationhistory([0, 0.1, 0.25, -0.15, -0.3, 0.2, 0.1]))
# lastextremes_test()
# lastgainloss_test()
# println("rolling regression $(Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4))")
# println("norm rolling regression $(Features.normrollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4))")

@testset "Features tests" begin

a,b = Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)
@test abs(b[7] - 0.31071427) < 10^-7
@test isapprox(a, [2.8535714, 3.1642857, 3.475, 3.7857144, 4.0964284, 4.4071426, 4.7178574], atol=10^-5)
a,b = Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4)
@test isapprox(a, [2.87, 3.19, 3.51, 3.83, 4.06, 4.13, 4.78], atol=10^-5)
@test isapprox(b, [0.32, 0.32, 0.32, 0.32, 0.29, 0.17, 0.37], atol=10^-5)
a,b = Features.normrollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4)
@test isapprox(a, [0.98965514, 1.0290322, 0.975, 1.0078948, 1.015, 1.0073171, 0.95600003], atol=10^-5)
@test isapprox(b, [0.11034483, 0.103225805, 0.08888888, 0.08421052, 0.0725, 0.041463416, 0.074], atol=10^-5)
@test Features.relativevolume([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 3, 5) == [1.0555555555555556; 1.0526315789473684; 1.025]
@test lastextremes_test()
@test lastgainloss_test()
@test isapprox(Features.regressionaccelerationhistory([0, 0.1, 0.25, -0.15, -0.3, 0.2, 0.1]), [0.0  0.1  0.25  -0.4  -0.55  0.5  -0.1], atol=10^-5)

end

end  # module