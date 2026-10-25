using Dates, DataFrames, Statistics, Test
using EnvConfig, Ohlcv, Features, TestOhlcv

"""Configure normalized gradients with raw gradients and volume for integration checks."""
function stabilizedtestfeatures()
    f6 = Features.Features006()
    Features.addnormgrad!(f6, window=5, offset=2)
    Features.addgrad!(f6, window=5)
    Features.addrelvol!(f6, short=3, long=15)
    return f6
end

@testset "positive volume baseline" begin
    @test Features.relativevolume(zeros(Float32, 20), 3, 15) == zeros(Float32, 20)
    @test Features.relativevolume(Float32[7], 3, 15) == Float32[1]
    @test Features.relativevolume(Float32[0, 2], 3, 15) == Float32[0, 1]
    onset = vcat(zeros(Float32, 15), fill(0.001f0, 3))
    @test Features.relativevolume(onset, 3, 15)[end] == 1f0
    vol = vcat(zeros(Float32, 12), Float32[2, 4, 6])
    @test Features.relativevolume(vol, 3, 15)[end] == 1f0
    vol = Float32[1, 0, 0, 0, 0, 0, 0, 8, 16, 32]
    result = Features.relativevolume(vol, 3, 7)
    for ix in eachindex(vol)
        trailing = vol[max(1, ix - 6):ix]
        baseline = median(trailing)
        positive = filter(>(0), trailing)
        baseline = iszero(baseline) && !isempty(positive) ? median(positive) : baseline
        shortmedian = median(vol[max(1, ix - 2):ix])
        @test result[ix] == (iszero(baseline) ? 0f0 : shortmedian / baseline)
        @test Features.relativevolume(vol[1:ix], 3, 7)[end] == result[ix]
    end
    tiny = fill(eps(Float32) / 8, 20)
    @test Features.relativevolume(tiny, 3, 15) == ones(Float32, 20)
    @test_throws AssertionError Features.relativevolume(Float32[-1, 2, 3], 2, 3)
    @test_throws AssertionError Features.relativevolume(Float32[NaN, 2, 3], 2, 3)
    @test_throws AssertionError Features.relativevolume(Float32[], 2, 3)
    @test_throws AssertionError Features.relativevolume(ones(Float32, 5), 3, 3)
end

@testset "normalized Features006 and bounded volume" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime("2025-02-17T13:30:00")
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Hour(2) - Minute(1))
    odf = Ohlcv.dataframe(ohlcv)
    odf[!, :basevolume] .= 1f0
    odf[1:20, :basevolume] .= 0f0
    odf[18:20, :basevolume] .= 0.001f0
    odf[30:32, :basevolume] .= 59f0
    odf[60:62, :basevolume] .= 60f0
    odf[90:92, :basevolume] .= 10000f0
    odf[95:115, :basevolume] .= 0f0
    odf[113:115, :basevolume] .= 0.001f0
    f6 = stabilizedtestfeatures()
    Features.setbase!(f6, ohlcv)
    @test Features.featurecount(f6) == 3
    @test Features.requestedcolumns(f6) == ["nrg+5+2", "rg+5+0", "rv+3+15+0"]
    @test Features.regressionwindows(f6) == ([5], Set{Int}())
    ry, rg = Features.rollingregression(odf[!, :pivot], 5)
    @test f6.fdfno[!, "ry+5"] == ry
    @test f6.fdfno[!, "rg+5"] == rg
    @test f6.fdfno[!, "nrg+5"] ≈ rg ./ vcat([Statistics.mean(odf[1:ix, :pivot]) for ix in 1:4],
                                          [Statistics.mean(odf[(ix-4):ix, :pivot]) for ix in 5:nrow(odf)])
    @test f6.fdf[!, "nrg+5+2"] == f6.fdfno[1:(end-2), "nrg+5"]
    @test f6.fdfno[32, "rv+3+15"] == 59f0
    @test f6.fdfno[62, "rv+3+15"] == 60f0
    @test f6.fdfno[92, "rv+3+15"] == 60f0
    @test f6.fdfno[20, "rv+3+15"] == 1f0
    @test f6.fdfno[115, "rv+3+15"] == 1f0
    @test all(v -> 0 <= v <= 60, f6.fdfno[!, "rv+3+15"])

    scaled = Ohlcv.defaultohlcv("SINE")
    scaleddf = DataFrame(odf)
    scaleddf[!, :pivot] .*= 10f0
    Ohlcv.setdataframe!(scaled, scaleddf)
    scaledf6 = stabilizedtestfeatures()
    Features.setbase!(scaledf6, scaled)
    @test scaledf6.fdfno[!, "nrg+5"] ≈ f6.fdfno[!, "nrg+5"] atol=1f-5

    live = Ohlcv.defaultohlcv("SINE")
    Ohlcv.setdataframe!(live, DataFrame(odf[1:45, :]))
    livef6 = stabilizedtestfeatures()
    Features.setbase!(livef6, live)
    for lastrow in 46:nrow(odf)
        Ohlcv.setdataframe!(live, DataFrame(odf[1:lastrow, :]))
        Features.supplement!(livef6)
    end
    for col in Features.requestedcolumns(f6)
        @test livef6.fdf[!, col] ≈ f6.fdf[!, col]
    end

    onlynrg = Features.Features006()
    feature = Features.addnormgrad!(onlynrg, window=5)
    @test feature == Features._normgrad(onlynrg, window=5, offset=0)
    @test Features.requestedcolumns(onlynrg) == ["nrg+5+0"]
    @test Features.configdescription(onlynrg) == "Features006(nrg+5+0;nrg=rg/meanpivot)"
    Features.setbase!(onlynrg, ohlcv)
    regr = Features.regressionat(onlynrg, 5, odf[end, :opentime])
    @test regr == (regry=ry[end], grad=rg[end])
    @test Features.relativegain(regr.regry, regr.grad, 60) ==
          Features.relativegain(ry[end], rg[end], 60)
    @test_throws AssertionError Features.addnormgrad!(Features.Features006(), window=1)
    @test_throws AssertionError Features.addnormgrad!(Features.Features006(), offset=-1)
    for meanpivot in (0f0, -1f0, NaN32, Inf32)
        raw = DataFrame("ry+5" => fill(meanpivot, nrow(odf)), "rg+5" => zeros(Float32, nrow(odf)))
        @test_throws AssertionError Features._normgrad!(raw, onlynrg, feature, odf, nothing, nothing)
    end
    raw = DataFrame("ry+5" => ones(Float32, nrow(odf)), "rg+5" => fill(Inf32, nrow(odf)))
    @test_throws AssertionError Features._normgrad!(raw, onlynrg, feature, odf, nothing, nothing)
    raw = DataFrame("ry+5" => fill(floatmin(Float32), nrow(odf)), "rg+5" => fill(floatmax(Float32), nrow(odf)))
    @test_throws AssertionError Features._normgrad!(raw, onlynrg, feature, odf, nothing, nothing)
end

@testset "nrg mean pivot denominator" begin
    @test Features.regressionmeanpivot(10f0, 1f0, 5) == 8f0
    # abrupt drop like CTC 2022-09-21: positive prices, negative regression end point
    piv = vcat(fill(0.303649f0, 30), fill(8f-5, 30))
    crash = DataFrame(opentime=[DateTime(2022, 9, 21, 11, 0) + Minute(ix) for ix in 0:59],
                      open=piv, high=piv, low=piv, close=piv, basevolume=ones(Float32, 60))
    crashohlcv = Ohlcv.defaultohlcv("CRASH")
    Ohlcv.setdataframe!(crashohlcv, crash)
    Ohlcv.addpivot!(crashohlcv)
    for window in (5, 15)
        ry, rg = Features.rollingregression(Ohlcv.dataframe(crashohlcv)[!, :pivot], window)
        @test minimum(ry) < 0
        f6 = Features.Features006()
        Features.addnormgrad!(f6, window=window)
        Features.setbase!(f6, crashohlcv)
        nrg = f6.fdfno[!, "nrg+$window"]
        @test all(isfinite, nrg)
        @test maximum(abs, nrg) <= 6 / (window + 1) * (1 + 1f-4)
        @test f6.fdfno[!, "ry+$window"] == ry
        @test f6.fdfno[!, "rg+$window"] == rg
        fullix = window:60
        @test Features.regressionmeanpivot.(ry[fullix], rg[fullix], window) ≈
              [Statistics.mean(piv[(ix-window+1):ix]) for ix in fullix] rtol=1f-4
    end
end
