using Test
using DataFrames
using Dates
using TradingStrategy
using Targets

function init_strategy_columns!(tdf::DataFrame)
    if :longopenlimit ∉ propertynames(tdf)
        tdf[!, :longopenlimit] = fill(0f0, nrow(tdf))
    end
    if :longcloselimit ∉ propertynames(tdf)
        tdf[!, :longcloselimit] = fill(0f0, nrow(tdf))
    end
    if :shortopenlimit ∉ propertynames(tdf)
        tdf[!, :shortopenlimit] = fill(0f0, nrow(tdf))
    end
    if :shortcloselimit ∉ propertynames(tdf)
        tdf[!, :shortcloselimit] = fill(0f0, nrow(tdf))
    end
    if :tradelabel ∉ propertynames(tdf)
        tdf[!, :tradelabel] = fill(Targets.ignore, nrow(tdf))
    end
    if :labelscore ∉ propertynames(tdf)
        tdf[!, :labelscore] = zeros(Float32, nrow(tdf))
    end
    return tdf
end

function test_strategy(; minpricedelta=0f0)
    return TradingStrategy.StrategyConfig(
        openthreshold=0.6f0,
        buygain=0.001f0,
        sellgain=0.01f0,
        limitreduction=0f0,
        maxwindow=4 * 60,
        minpricedelta=minpricedelta,
    )
end

@testset "gain_limit_reversal direction" begin
    @testset "long signal encodes long guidance" begin
        tdf = DataFrame(
            opentime=[DateTime(2026, 1, 1, 0, 0)],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            tradelabel=Targets.TradeLabel[Targets.longopen],
            labelscore=Float32[0.9f0],
        )
        init_strategy_columns!(tdf)
        TradingStrategy.gain_limit_reversal!(
            test_strategy(),
            tdf,
            1,
            tdf[1, :tradelabel],
            tdf[1, :labelscore],
            tdf[1, :close],
        )
        @test tdf[1, :tradelabel] == Targets.longopen
        @test isapprox(tdf[1, :longopenlimit], 99.9f0; atol=1f-4)
        @test isapprox(tdf[1, :longcloselimit], 101f0; atol=1f-4)
        @test tdf[1, :shortopenlimit] == 0f0
    end

    @testset "short signal encodes short guidance" begin
        tdf = DataFrame(
            opentime=[DateTime(2026, 1, 1, 0, 0)],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            tradelabel=Targets.TradeLabel[Targets.shortopen],
            labelscore=Float32[0.9f0],
        )
        init_strategy_columns!(tdf)
        TradingStrategy.gain_limit_reversal!(
            test_strategy(),
            tdf,
            1,
            tdf[1, :tradelabel],
            tdf[1, :labelscore],
            tdf[1, :close],
        )
        @test tdf[1, :tradelabel] == Targets.shortopen
        @test isapprox(tdf[1, :shortopenlimit], 100.1f0; atol=1f-4)
        @test isapprox(tdf[1, :shortcloselimit], 99f0; atol=1f-4)
        @test tdf[1, :longopenlimit] == 0f0
    end

    @testset "pricedelta gating keeps limits unchanged below threshold" begin
        tdf = DataFrame(
            opentime=[DateTime(2026, 1, 1, 0, 0), DateTime(2026, 1, 1, 0, 1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100.02f0],
            tradelabel=Targets.TradeLabel[Targets.longopen, Targets.longopen],
            labelscore=Float32[0.9f0, 0.9f0],
        )
        init_strategy_columns!(tdf)
        TradingStrategy.gain_limit_reversal!(
            test_strategy(minpricedelta=0.01f0),
            tdf,
            1,
            tdf[1, :tradelabel],
            tdf[1, :labelscore],
            tdf[1, :close],
        )
        TradingStrategy.gain_limit_reversal!(
            test_strategy(minpricedelta=0.01f0),
            tdf,
            2,
            tdf[2, :tradelabel],
            tdf[2, :labelscore],
            tdf[2, :close],
        )
        @test isapprox(tdf[2, :longopenlimit], tdf[1, :longopenlimit]; atol=1f-6)
        @test isapprox(tdf[2, :longcloselimit], tdf[1, :longcloselimit]; atol=1f-6)
    end
end
