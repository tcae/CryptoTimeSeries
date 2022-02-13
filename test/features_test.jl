
include("../src/features.jl")
include("../src/testohlcv.jl")

module FeaturesTest
using Dates, DataFrames
using Test

using ..EnvConfig, ..Ohlcv, ..Features, ..TestOhlcv



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

function distancepeaktest()
    x, y = TestOhlcv.sinesamples(400, 2, [(150, 0, 0.5)])
    _, grad = Features.rollingregression(y, 50)
    distances, regressionix, priceix = Features.distancesregressionpeak(y, grad)
    df = DataFrame()
    df.x = x
    df.y = y
    df.grad = grad
    df.dist = distances
    df.pp = priceix
    df.rp = regressionix
    println(df)
end

function nextpeakindices_test(prices, target)
    df = DataFrame()
    df.prices = prices
    df[:, :targetdist] = [(target[ix] == 0 ? 0.0 : prices[target[ix]] - prices[ix]) for ix in 1:length(prices)]
    df.targetgain = [(target[ix] == 0) ? 0.0 : Features.gain(prices, ix, target[ix]) for ix in 1:length(prices)]
    df.target = target
    df[:, :dist], df[:, :distix] = Features.nextpeakindices(prices, 0.05, -0.05)
    df[:, :distgain] = [(df[ix, :distix] == 0 ? 0.0 : Features.gain(prices, ix, df[ix, :distix])) for ix in 1:length(prices)]
    res = sum(df.distgain .- df.targetgain)
    # println(df)
    # println(res)
    return res == 0.0
end

function nextpeakdistance_test()
    res = true

    # start and end without significant changes but have some in between
    prices = [100, 97, 99, 98, 103, 100, 104, 98, 99, 100]
    target = [  0,  7,  7,  7,   7,   7,   8,  0,  0,   0]
    res = nextpeakindices_test(prices, target) && res

    # start and end with significant changes
    prices = [103, 97, 99, 98, 103, 100, 104, 98, 99, 104]
    target = [  2,  7,  7,  7,   7,   7,   8, 10, 10,   0]
    res = nextpeakindices_test(prices, target) && res

    # only significant changes at start
    prices = [103, 97, 99, 96, 103, 100, 104, 98, 99, 104]
    target = [  4,  4,  4,  7,   7,   7,   8, 10, 10,   0]
    res = nextpeakindices_test(prices, target) && res

    # no significant changes at all
    prices = [100, 97, 99, 98, 100, 99, 101, 98, 99, 100]
    target = [  0,  0,  0,  0,   0,   0,   0,  0,  0,   0]
    res = nextpeakindices_test(prices, target) && res
    return res
end


EnvConfig.init(test)
# config_test()
# display(Features.regressionaccelerationhistory([0, 0.1, 0.25, -0.15, -0.3, 0.2, 0.1]))
# lastextremes_test()
# lastgainloss_test()
# println("rolling regression $(Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4))")
# println("norm rolling regression $(Features.normrollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4))")


# ! features001set test to be added
# ! rollingregressionstd test to be added

@testset "Features tests" begin

a,b = Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)
@test abs(b[7] - 0.31071427) < 10^-7
@test isapprox(a, [2.8535714, 3.1642857, 3.475, 3.7857144, 4.0964284, 4.4071426, 4.7178574], atol=10^-5)
a,b = Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4)
@test isapprox(a, [2.87, 3.19, 3.51, 3.83, 4.06, 4.13, 4.78], atol=10^-5)
@test isapprox(b, [0.32, 0.32, 0.32, 0.32, 0.29, 0.17, 0.37], atol=10^-5)
# a,b = Features.normrollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4)
# @test isapprox(a, [0.98965514, 1.0290322, 0.975, 1.0078948, 1.015, 1.0073171, 0.95600003], atol=10^-5)
# @test isapprox(b, [0.11034483, 0.103225805, 0.08888888, 0.08421052, 0.0725, 0.041463416, 0.074], atol=10^-5)
# @test Features.relativevolume([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 3, 5) == [1.0555555555555556; 1.0526315789473684; 1.025]
@test Features.relativevolume([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 3, 5) == [1.0, 1.0, 1.0, 1.0746268656716418, 1.0555555555555556, 1.0526315789473684, 1.025]
@test lastextremes_test()
@test lastgainloss_test()
@test isapprox(Features.regressionaccelerationhistory([0, 0.1, 0.25, -0.15, -0.3, 0.2, 0.1]), [0.0  0.1  0.25  -0.4  -0.55  0.5  -0.1], atol=10^-5)
@test nextpeakdistance_test()
end

# distancepeaktest()

end  # module