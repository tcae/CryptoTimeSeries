using Test
using Targets

@testset "Targets::Trend02 directional detection tests" begin
    thres = Targets.LabelThresholds(longbuy=0.10f0, longhold=0f0, shorthold=0f0, shortbuy=-0.10f0)

    @testset "straight long trend detection" begin
        trd = Targets.Trend02(3, 8, thres)
        pivots = Float32[100, 102, 104, 106, 112]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        @test all(trd.df[1:end-1, :label] .== allclose)
        @test trd.df[end, :label] == longbuy
        @test trd.df[end, :relix] == 1
        @test trd.df[end, :reldiff] > 0f0
    end

    @testset "straight short trend detection" begin
        trd = Targets.Trend02(3, 8, thres)
        pivots = Float32[100, 98, 96, 94, 88]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        @test all(trd.df[1:end-1, :label] .== allclose)
        @test trd.df[end, :label] == shortbuy
        @test trd.df[end, :relix] == 1
        @test trd.df[end, :reldiff] < 0f0
    end

    @testset "overarching long broken by opposite short" begin
        trd = Targets.Trend02(3, 8, thres)
        pivots = Float32[100, 108, 120, 108, 95, 92, 96, 106, 118]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        @test trd.df[3, :label] == longbuy
        @test trd.df[6, :label] == shortbuy
        @test trd.df[7, :label] == shorthold
        @test trd.df[8, :label] == shorthold
        @test trd.df[9, :label] == allclose
    end

    @testset "trend closes when no new long extreme within maxwindow" begin
        trd = Targets.Trend02(3, 4, thres)
        pivots = Float32[100, 100, 90, 95, 99, 98, 97, 96, 95]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        @test trd.df[5, :label] == longbuy
        @test trd.df[6, :label] == longhold
        @test trd.df[8, :label] == longhold
        @test trd.df[9, :label] == allclose
        @test trd.df[9, :relix] == 9
    end
end
