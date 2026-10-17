function trend05_test_ohlcv(pivots::Vector{Float32}; base::String="TEST")
    startdt = DateTime("2025-01-01T00:00:00")
    df = DataFrame(
        opentime=[startdt + Minute(ix - 1) for ix in eachindex(pivots)],
        open=copy(pivots),
        high=copy(pivots),
        low=copy(pivots),
        close=copy(pivots),
        basevolume=fill(1f0, length(pivots)),
        pivot=copy(pivots),
    )
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.setdataframe!(ohlcv, df)
    return ohlcv
end

@testset "featurestargetsdf supports Trend05 targets" begin
    fullohlcv = trend05_test_ohlcv(Float32.(100:1:599); base="TREND05RANGE")
    ohlcv = Ohlcv.ohlcvview(fullohlcv, 400:500)
    f6 = Features.Features006()
    for window in (2, 3, 4)
        Features.addgrad!(f6; window, offset=0)
    end
    Features.addrelvol!(f6; short=5, long=360, offset=0)

    nn = Classify.NN(nothing, nothing, nothing, Targets.TradeLabel[], "Trend05 test")
    cl = Classify.TrendClassifier001(nn; featconfig=() -> f6)
    trd = Targets.Trend05(leadregr=2, supportregr=3, trendregr=4)

    results, featurestable = Classify.featurestargetsdf(cl, ohlcv, trd)

    @test nrow(results) == length(Ohlcv.dataframe(ohlcv)[!, :pivot])
    @test nrow(featurestable) == nrow(results)
    @test trd.f6 === f6
    @test trd.ohlcv === ohlcv
    @test nrow(f6.fdfno) == nrow(Ohlcv.dataframe(ohlcv)) + 359
    @test results[!, :opentime] == Ohlcv.dataframe(ohlcv)[!, :opentime]
    @test issubset(Set(results[!, :target]), Set(Targets.uniquelabels(trd)))
end
