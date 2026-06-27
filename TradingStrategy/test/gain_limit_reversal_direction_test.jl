using Test
using DataFrames
using Dates
using TradingStrategy
using Targets

function init_strategy_columns!(tdf::DataFrame)
    if :longopenlimit ∉ propertynames(tdf)
        tdf[!, :longopenlimit] = Vector{Union{Missing, Float32}}(missing, nrow(tdf))
    end
    if :longcloselimit ∉ propertynames(tdf)
        tdf[!, :longcloselimit] = Vector{Union{Missing, Float32}}(missing, nrow(tdf))
    end
    if :shortopenlimit ∉ propertynames(tdf)
        tdf[!, :shortopenlimit] = Vector{Union{Missing, Float32}}(missing, nrow(tdf))
    end
    if :shortcloselimit ∉ propertynames(tdf)
        tdf[!, :shortcloselimit] = Vector{Union{Missing, Float32}}(missing, nrow(tdf))
    end
    if :tradelabel ∉ propertynames(tdf)
        tdf[!, :tradelabel] = fill(Targets.ignore, nrow(tdf))
    end
    if :labelscore ∉ propertynames(tdf)
        tdf[!, :labelscore] = zeros(Float32, nrow(tdf))
    end
    return tdf
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
            tdf,
            1,
            tdf[1, :tradelabel],
            tdf[1, :labelscore],
            tdf[1, :close];
            openthreshold=0.6f0,
            buygain=0.001f0,
            sellgain=0.01f0,
            limitreduction=0f0,
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
            tdf,
            1,
            tdf[1, :tradelabel],
            tdf[1, :labelscore],
            tdf[1, :close];
            openthreshold=0.6f0,
            buygain=0.001f0,
            sellgain=0.01f0,
            limitreduction=0f0,
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
            tdf,
            1,
            tdf[1, :tradelabel],
            tdf[1, :labelscore],
            tdf[1, :close];
            openthreshold=0.6f0,
            buygain=0.001f0,
            sellgain=0.01f0,
            limitreduction=0f0,
            minpricedelta=0.01f0,
        )
        TradingStrategy.gain_limit_reversal!(
            tdf,
            2,
            tdf[2, :tradelabel],
            tdf[2, :labelscore],
            tdf[2, :close];
            openthreshold=0.6f0,
            buygain=0.001f0,
            sellgain=0.01f0,
            limitreduction=0f0,
            minpricedelta=0.01f0,
        )
        @test isapprox(tdf[2, :longopenlimit], tdf[1, :longopenlimit]; atol=1f-6)
        @test isapprox(tdf[2, :longcloselimit], tdf[1, :longcloselimit]; atol=1f-6)
    end
end
