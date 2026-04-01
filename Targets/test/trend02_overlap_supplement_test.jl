using Dates, DataFrames
using Test
using Targets, Ohlcv

function testohlcvfrompivots(pivots::Vector{Float32}; startdt::DateTime=DateTime("2025-01-01T00:00:00"))
    rows = length(pivots)
    df = DataFrame(
        opentime=[startdt + Minute(ix - 1) for ix in 1:rows],
        open=copy(pivots),
        high=copy(pivots),
        low=copy(pivots),
        close=copy(pivots),
        basevolume=fill(1f0, rows),
        pivot=copy(pivots)
    )
    ohlcv = Ohlcv.defaultohlcv("TEST")
    Ohlcv.setdataframe!(ohlcv, df)
    return ohlcv
end

@testset "Targets::Trend02 overlap supplement tests" begin
    thres = Targets.LabelThresholds(longbuy=0.10f0, longhold=0f0, shorthold=0f0, shortbuy=-0.10f0)

    @testset "extends existing trend with original anchor" begin
        trd = Targets.Trend02(3, 4, thres)
        pivots = Float32[100, 100, 90, 92, 95, 99]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        @test trd.df[3, :label] == allclose
        @test trd.df[6, :label] == longbuy
        @test trd.df[6, :relix] == 3

        pivots2 = Float32[100, 100, 90, 92, 95, 99, 103]
        Ohlcv.setdataframe!(ohlcv, testohlcvfrompivots(pivots2).df)
        Targets.supplement!(trd)

        @test trd.df[7, :label] == longbuy
        @test trd.df[7, :relix] == 3
    end

    @testset "overlap start becomes confirmed after append" begin
        trd = Targets.Trend02(3, 6, thres)
        pivots = Float32[100, 101, 102, 103, 104, 105]
        ohlcv = testohlcvfrompivots(pivots; startdt=DateTime("2025-01-02T00:00:00"))
        Targets.setbase!(trd, ohlcv)

        @test all(trd.df[!, :label] .== allclose)

        pivots2 = Float32[100, 101, 102, 103, 104, 105, 112]
        Ohlcv.setdataframe!(ohlcv, testohlcvfrompivots(pivots2; startdt=DateTime("2025-01-02T00:00:00")).df)
        Targets.supplement!(trd)

        @test trd.df[1, :label] == allclose
        @test all(trd.df[2:6, :label] .== allclose)
        @test trd.df[1, :relix] == 1
        @test all(trd.df[2:6, :relix] .== 2)
        @test trd.df[7, :label] == longbuy
        @test trd.df[7, :relix] == 2
    end
end
