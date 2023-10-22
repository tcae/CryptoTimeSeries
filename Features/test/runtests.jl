module FeaturesTest
using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, Features, TestOhlcv



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
    distances, regressionix, priceix = Features.pricediffregressionpeak(y, grad)
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
    df[:, :targetdist] = [(target[ix] == 0 ? 0.0 : prices[target[ix]] - prices[ix]) for ix in eachindex(prices)]  # 1:length(prices)]
    df.targetgain = [(target[ix] == 0) ? 0.0 : Features.gain(prices, ix, target[ix]) for ix in eachindex(prices)]  # 1:length(prices)]
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

function testfeatures()
    enddt = DateTime("2022-01-02T22:54:00")
    startdt = enddt - Dates.Day(3)
    ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
    ol = size(Ohlcv.dataframe(ohlcv),1)
    f2 = Features.Features002(ohlcv)
    return ol, f2

end

EnvConfig.init(test)
# config_test()
# display(Features.regressionaccelerationhistory([0, 0.1, 0.25, -0.15, -0.3, 0.2, 0.1]))
# lastextremes_test()
# lastgainloss_test()
# println("rolling regression $(Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4))")
# println("norm rolling regression $(Features.normrollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 4))")

@testset "Features tests" begin

ol, f2 = testfeatures()
@test ol == 4321
@test length(keys(f2.regr)) == length(Features.regressionwindows002)
for (win, f2r) in pairs(f2.regr)
    @test 0 < length(f2r.grad) <= ol
    @test 0 < length(f2r.regry) <= ol
    @test 0 < length(f2r.std) <= ol
    @test 0 < length(f2r.xtrmix) <= ol
end

yvec = [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
regr,grad = Features.rollingregression(yvec, 7)
@test abs(grad[7] - 0.31071427) < 10^-7
@test isapprox(regr, [2.8535714, 3.1642857, 3.475, 3.7857144, 4.0964284, 4.4071426, 4.7178574], atol=10^-5)
regr2,grad2 = Features.rollingregression(yvec, 7, 1)
@test regr == regr2
@test grad == grad2
regr,grad = Features.rollingregression(yvec, 4)
@test isapprox(regr, [2.87, 3.19, 3.51, 3.83, 4.06, 4.13, 4.78], atol=10^-5)
@test isapprox(grad, [0.32, 0.32, 0.32, 0.32, 0.29, 0.17, 0.37], atol=10^-5)
regr2,grad2 = Features.rollingregression(yvec, 4, 1)
@test regr == regr2
@test grad == grad2
# regr,grad = Features.normrollingregression(yvec, 4)
# @test isapprox(regr, [0.98965514, 1.0290322, 0.975, 1.0078948, 1.015, 1.0073171, 0.95600003], atol=10^-5)
# @test isapprox(grad, [0.11034483, 0.103225805, 0.08888888, 0.08421052, 0.0725, 0.041463416, 0.074], atol=10^-5)
# @test Features.relativevolume(yvec, 3, 5) == [1.0555555555555556; 1.0526315789473684; 1.025]
@test Features.relativevolume(yvec, 3, 5) == [1.0, 1.0, 1.0, 1.0746268656716418, 1.0555555555555556, 1.0526315789473684, 1.025]
@test lastextremes_test()
@test lastgainloss_test()
@test isapprox(Features.regressionaccelerationhistory([0, 0.1, 0.25, -0.15, -0.3, 0.2, 0.1]), [0.0  0.1  0.25  -0.4  -0.55  0.5  -0.1], atol=10^-5)
@test nextpeakdistance_test()
@test Features.mlfeatures_test()

regr,grad = Features.rollingregression(yvec, 2)
regr2,grad2 = Features.rollingregression(yvec, 2, 4)
@test length(regr) - 4 + 2 == length(regr2)
@test length(grad) - 4 + 2 == length(grad2)
@test regr[4:end] == regr2[2:end]
@test grad[4:end] == grad2[2:end]
# @test grad == grad2

regr2,grad2 = Features.rollingregression!(nothing, nothing, yvec[1:4], 2)
@test regr[1:4] == regr2
@test grad[1:4] == grad2
regr2,grad2 = Features.rollingregression!(Float64[], Float64[], yvec[1:4], 2)
@test regr[1:4] == regr2
@test grad[1:4] == grad2
regr2,grad2 = Features.rollingregression(yvec[1:4], 2)
@test regr[1:4] == regr2
@test grad[1:4] == grad2
regr2,grad2 = Features.rollingregression!(regr2, grad2, yvec[1:5], 2)
@test regr[1:5] == regr2
@test grad[1:5] == grad2
regr2,grad2 = Features.rollingregression!(regr2, grad2, yvec, 2)
@test regr == regr2
@test grad == grad2

regr,grad = Features.rollingregression(yvec, 5)
regr2,grad2 = Features.rollingregression(yvec[1:5], 5)
@test regr[1:5] == regr2
@test grad[1:5] == grad2
regr2,grad2 = Features.rollingregression!(regr2, grad2, yvec, 5)
@test regr == regr2
@test grad == grad2

s1, m1, n1 = Features.rollingregressionstd(yvec, regr, grad, 5)
s2, m2, n2 = Features.rollingregressionstdxt(yvec, regr, grad, 5)
@test s1 == s2
@test m1 == m2

ymv = [yvec]
smv = Features.rollingregressionstdmv(ymv, regr, grad, 5, 1)
@test s1 == smv

regr,grad = Features.rollingregression(yvec, 2)
s1, m1, n1 = Features.rollingregressionstd(yvec, regr, grad, 3)
s2, m2, n2 = Features.rollingregressionstd!(nothing, yvec[1:4], regr[1:4], grad[1:4], 3)
# s2, m2, n2 = Features.rollingregressionstd(yvec[1:4], regr[1:4], grad[1:4], 3)
@test s1[1:4] == s2
@test m1[1:4] == m2
@test n1[1:4] == n2

s2, m2, n2 = Features.rollingregressionstd!(s2, yvec, regr, grad, 3)
@test s1 == s2
@test m1[5:7] == m2
@test n1[5:7] == n2

ymv = [yvec]
smv1 = Features.rollingregressionstdmv(ymv, regr, grad, 3, 1)
@test s1 == smv1
ymv = [yvec[1:4]]
smv2 = Features.rollingregressionstdmv!(nothing, ymv, regr[1:4], grad[1:4], 3)
@test smv1[1:4] == smv2
ymv = [yvec]
smv2 = Features.rollingregressionstdmv!(smv2, ymv, regr, grad, 3)
@test smv1 == smv2

ymv = [yvec, yvec]
smv1 = Features.rollingregressionstdmv(ymv, regr, grad, 3, 1)
@test all(s1[2:end] .> smv1[2:end])
@test length(s1) == length(smv1)
ymv = [yvec[1:4], yvec[1:4]]
smv2 = Features.rollingregressionstdmv!(nothing, ymv, regr[1:4], grad[1:4], 3)
@test smv1[1:4] == smv2
ymv = [yvec, yvec]
smv2 = Features.rollingregressionstdmv!(smv2, ymv, regr, grad, 3)
@test smv1 == smv2


reggrad = [1, 2, 0, 0, -1, 0, 1, -3, 1, 1, 0]
xix = Int32[]
xix = Features.regressionextremesix!(xix, reggrad, 1; forward=true)
@test xix == [3, -7, 8, -9, 11]
xix = Features.regressionextremesix!(nothing, reggrad, 1; forward=true)
@test xix == [3, -7, 8, -9, 11]
xix2 = xix[1:3]
xix2 = Features.regressionextremesix!(xix2, reggrad, 8; forward=true)
@test xix2 == [3, -7, 8, -9, 11]
xix = Features.regressionextremesix!(nothing, reggrad, length(reggrad); forward=false)
@test xix == [2, -6, 7, -8, 10]
xix = Features.regressionextremesix!(xix[3:5], reggrad, 7; forward=false)
@test xix == [2, -6, 7, -8, 10]

@test "121m" == Features.periodlabels(2*60+1)
@test "2h" == Features.periodlabels(2*60)
@test "2d" == Features.periodlabels(2*24*60)

md = DateTime("2022-06-02T12:54:00")
@test Features.relativedayofyear(md) == 0.4192f0
@test Features.relativedayofweek(md) == 0.5714f0
@test Features.relativeminuteofday(md) == 0.5375f0

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(2)
ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
df = Ohlcv.dataframe(ohlcv)
f2 = Features.Features002(ohlcv)

# f12x = Features.features12x1m01(f2)
# f12xd = describe(f12x)
# @test all(f12xd.eltype .== Float32)
# @test all(f12xd.nmissing .== 0)
# @test all(-1.9 .< f12xd.min .< 1.9)
# @test all(-1.9 .< f12xd.median .< 1.9)
# @test all(-1.9 .< f12xd.max .< 2.9)
# @test all(-1.2 .< f12xd.mean .< 1.2)
# @test size(f12xd, 1) == 60

rdf = DataFrame()
df = DataFrame((colA = [1.1f0, 3.5f0, 8.0f0]))
rdf, colname = Features.lookbackrow!(nothing, df, "colA",1, 1, size(df,1); fill=nothing)
@test rdf[!, "colA01"] == [1.1f0, 1.1f0, 3.5f0]
rdf, colname = Features.lookbackrow!(nothing, df, "colA",2, 1, size(df,1); fill=nothing)
@test rdf[!, "colA02"] == [1.1f0, 1.1f0, 1.1f0]
rdf, colname = Features.lookbackrow!(nothing, df, "colA",3, 1, size(df,1); fill=nothing)
@test rdf[!, "colA03"] == [1.1f0, 1.1f0, 1.1f0]
rdf, colname = Features.lookbackrow!(nothing, df, "colA",3, 1, size(df,1); fill=0)
@test rdf[!, "colA03"] == [0.0f0, 0.0f0, 0.0f0]
rdf, colname = Features.lookbackrow!(nothing, df, "colA",2, 2, size(df,1); fill=0)
@test rdf[!, "colA02"] == [0.0f0, 1.1f0]
rdf, colname = Features.lookbackrow!(nothing, df, "colA",1, 1, size(df,1); fill=nothing)
rdf, colname = Features.lookbackrow!(rdf, df, "colA",2, 1, size(df,1); fill=nothing)
@test size(rdf) == (3, 2)
rdf, colname = Features.lookbackrow!(nothing, df, "colA",1, 1, size(df,1); fill=nothing)
@test_throws AssertionError Features.lookbackrow!(rdf, df, "colA",2, 2, size(df,1); fill=0)
rdf, colname = Features.lookbackrow!(nothing, df, "colA",0, 1, size(df,1); fill=nothing)
@test rdf[!, "colA00"] == [1.1f0, 3.5f0, 8.0f0]

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(20)
ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
f12x, f3 = Features.regressionfeatures01(ohlcv,  11, 5, [15, 60], 5, 4*60, "relminuteofday")
# println("regressionfeatures01(ohlcv)")
# println("size(f12x)=$(size(f12x))")
# println("size(regr[5].grad(f3))=$(size(Features.grad(f3, 5)))")
# println("size(regr[15].grad(f3))=$(size(Features.grad(f3, 15)))")
# println("size(ohlcvdataframe(f3))=$(size(Features.ohlcvdataframe(f3)))")
f12xd = describe(f12x)
# println("describe(f12x)=$f12xd")
@test size(f12xd, 1) == 30
@test minimum(f12xd[!, :min]) > -3.2
@test maximum(f12xd[!, :max]) < 3.2
@test all(y-> eltype(y) == Float32, eachcol(f12x))
f12x, f3 = Features.features12x1m01(ohlcv)
# println("features12x1m01(ohlcv)")
# println("size(f12x)=$(size(f12x))")
# println("size(regr[5].grad(f3))=$(size(Features.grad(f3, 5)))")
# println("size(regr[15].grad(f3))=$(size(Features.grad(f3, 15)))")
# println("size(ohlcvdataframe(f3))=$(size(Features.ohlcvdataframe(f3)))")
f12xd = describe(f12x)
# println("describe(f12x)=$f12xd")
@test size(f12xd, 1) == 60
@test minimum(f12xd[!, :min]) > -3.2
@test maximum(f12xd[!, :max]) < 3.2
@test all(y-> eltype(y) == Float32, eachcol(f12x))

end # testset

# distancepeaktest()

end  # module