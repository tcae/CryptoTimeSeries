using Test
using Targets

@testset "Targets::Trend02 crosscheck tests" begin
    thres = Targets.LabelThresholds(longbuy=0.10f0, longhold=0f0, shorthold=0f0, shortbuy=-0.10f0)

    @testset "valid handcrafted labels pass" begin
        trd = Targets.Trend02(3, 8, thres)
        labels = TradeLabel[allclose, allclose, longbuy, longbuy, longbuy, longbuy, allclose]
        pivots = Float32[100, 95, 100, 106, 112, 113, 110]
        check = Targets.crosscheck(trd, labels, pivots)
        @test isempty(check)
    end

    @testset "hold-only range fails" begin
        trd = Targets.Trend02(3, 8, thres)
        labels = TradeLabel[allclose, longhold, longhold, allclose]
        pivots = Float32[100, 101, 102, 103]

        check = Targets.crosscheck(trd, labels, pivots)
        @test !isempty(check)
        @test any(occursin("must be preceded by", msg) for msg in check)
    end

    @testset "buy without minwindow anchor fails" begin
        trd = Targets.Trend02(3, 8, thres)
        labels = TradeLabel[allclose, longbuy]
        pivots = Float32[100, 120]

        check = Targets.crosscheck(trd, labels, pivots)
        @test !isempty(check)
        @test any(occursin("violates minwindow", msg) for msg in check)
    end

    @testset "rolling maxwindow buy confirmation is required" begin
        trd = Targets.Trend02(2, 3, thres)
        labels = TradeLabel[longbuy, longbuy, longbuy, longbuy, longbuy]
        pivots = Float32[100, 103, 106, 109, 112]

        check = Targets.crosscheck(trd, labels, pivots)
        @test !isempty(check)
        @test any(occursin("lacks longbuy confirmation within maxwindow", msg) for msg in check)
    end

    @testset "segment start must be directional extreme" begin
        trd = Targets.Trend02(2, 8, thres)
        labels = TradeLabel[longbuy, longbuy, longbuy]
        pivots = Float32[101, 100, 120]

        check = Targets.crosscheck(trd, labels, pivots)
        @test !isempty(check)
        @test any(occursin("must start at a segment low extreme", msg) for msg in check)
    end

    @testset "instance overload uses trd state" begin
        trd = Targets.Trend02(3, 8, thres)
        pivots = Float32[100, 95, 100, 106, 112, 113, 110]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)
        trd.df[!, :label] = TradeLabel[allclose, allclose, longbuy, longbuy, longbuy, longbuy, allclose]

        check = Targets.crosscheck(trd)
        @test isempty(check)
    end
end
