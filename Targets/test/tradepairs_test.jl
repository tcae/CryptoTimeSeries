using Test
using Targets

@testset "TradePairs target derivation" begin
    @testset "constructor assertions and label set" begin
        trd = Targets.Trend04(10, 120, Targets.thresholds((longbuy=0.01f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.01f0)))
        tp = Targets.TradePairs(trd; entryfraction=0.1f0, exitfraction=0.1f0)
        @test tp.entryfraction == 0.1f0
        @test tp.exitfraction == 0.1f0
        @test Targets.longclose in Targets.uniquelabels(tp)
        @test Targets.shortclose in Targets.uniquelabels(tp)
    end

    @testset "vector label derivation creates sparse buy and close zones" begin
        trd = Targets.Trend04(10, 120, Targets.thresholds((longbuy=0.01f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.01f0)))
        tp = Targets.TradePairs(trd; entryfraction=0.1f0, exitfraction=0.1f0)

        baselabels = TradeLabel[
            longbuy, longbuy, longhold, longhold, longhold,
            allclose,
            shortbuy, shorthold, shorthold, shorthold,
        ]
        pivots = Float32[100.0, 100.05, 100.30, 100.92, 101.0, 101.0, 101.0, 100.95, 100.70, 100.10]
        groups = Int[1, 1, 1, 1, 1, 1, 2, 2, 2, 2]

        pairlabels = Targets.tradepairlabels(tp, baselabels, pivots; groups=groups)

        @test pairlabels[1:5] == TradeLabel[longbuy, longbuy, longhold, longclose, longclose]
        @test pairlabels[6] == allclose
        @test pairlabels[7:10] == TradeLabel[shortbuy, shortbuy, shorthold, shortclose]
    end

    @testset "setbase! populates derived labels" begin
        pivots = Float32[1.0, 1.001, 1.002, 1.006, 1.010, 1.009, 1.008, 1.002, 1.0]
        ohlcv = testohlcvfrompivots(pivots)
        tp = Targets.TradePairs(
            Targets.Trend04(2, 10, Targets.thresholds((longbuy=0.01f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.01f0)));
            entryfraction=0.1f0,
            exitfraction=0.1f0,
        )

        Targets.setbase!(tp, ohlcv)
        lbls = collect(Targets.labels(tp))
        @test length(lbls) == length(pivots)
        @test all(lbl -> lbl in Targets.uniquelabels(tp), lbls)
    end
end
