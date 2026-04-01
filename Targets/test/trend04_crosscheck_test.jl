using Test
using Targets

@testset "Targets::Trend04 crosscheck mitigation tests" begin
    @testset "monotonic continuations keep buy extension" begin
        thres = Targets.LabelThresholds(longbuy=0.025f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.025f0)

        longtrd = Targets.Trend04(2, 10, thres)
        longohlcv = testohlcvfrompivots(Float32[100.0, 103.0, 105.0, 106.0, 106.5])
        Targets.setbase!(longtrd, longohlcv)
        @test count(==(longbuy), longtrd.df.label) > 0
        @test count(==(longhold), longtrd.df.label) == 0

        shorttrd = Targets.Trend04(2, 10, thres)
        shortohlcv = testohlcvfrompivots(Float32[106.0, 103.0, 101.0, 100.0, 99.5])
        Targets.setbase!(shorttrd, shortohlcv)
        @test count(==(shortbuy), shorttrd.df.label) > 0
        @test count(==(shorthold), shorttrd.df.label) == 0
    end

    @testset "state machine starts long at local low" begin
        thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.005f0, shorthold=-0.005f0, shortbuy=-0.03f0)
        trd = Targets.Trend04(2, 10, thres)
        pivots = Float32[100.0, 95.0, 100.0, 106.0, 112.0]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        firstlong = findfirst(in((longbuy, longhold)), trd.df.label)
        @test !isnothing(firstlong)
        @test firstlong >= 2
        @test trd.df[firstlong, :relix] == 2
    end

    @testset "state machine avoids hold-only short runs" begin
        thres = Targets.LabelThresholds(longbuy=0.01f0, longhold=0.005f0, shorthold=-0.005f0, shortbuy=-0.01f0)
        trd = Targets.Trend04(2, 10, thres)
        pivots = Float32[4631.0, 4632.0, 4610.0, 4590.0, 4558.94, 4558.94]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        labels = trd.df.label
        @test count(==(shorthold), labels) >= 0
        @test (count(==(shorthold), labels) == 0) || (count(==(shortbuy), labels) > 0)
        for ix in eachindex(labels)
            if labels[ix] == shorthold
                @test any(labels[1:ix] .== shortbuy)
            end
        end
    end

    @testset "short range closes at last low extreme" begin
        thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.005f0, shorthold=-0.005f0, shortbuy=-0.03f0)
        trd = Targets.Trend04(10, 240, thres)

        # Reproduces the BTC crosscheck pattern where the short segment continued
        # beyond the last low extreme.
        pivots = Float32[3819.6074, 3816.335, 3890.0, 3890.0, 3885.0, 3850.0, 3804.82, 3779.245, 3754.7, 3754.7, 3751.405, 3758.0, 3758.0, 3830.6875]
        ohlcv = testohlcvfrompivots(pivots)
        Targets.setbase!(trd, ohlcv)

        check = Targets.crosscheck(trd)
        @test !any(occursin("must end at a segment low extreme", msg) for msg in check)
    end

    @testset "slow short continuation does not require per-window shortbuy" begin
        thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.005f0, shorthold=-0.005f0, shortbuy=-0.03f0)
        trd = Targets.Trend04(2, 3, thres)

        labels = TradeLabel[shortbuy, shortbuy, shortbuy, shortbuy, shortbuy]
        pivots = Float32[100.0, 98.0, 97.0, 96.0, 95.0]

        check = Targets.crosscheck(trd, labels, pivots)
        @test isempty(check)
    end
end
