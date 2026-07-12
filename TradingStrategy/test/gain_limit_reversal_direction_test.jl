using Test
using DataFrames
using Dates
using TradingStrategy
using Targets
using Xch

function init_trade_columns!(tdf::DataFrame)
    if :lo_amount ∉ propertynames(tdf)
        tdf[!, :lo_amount] = fill(0f0, nrow(tdf))
    end
    if :lc_amount ∉ propertynames(tdf)
        tdf[!, :lc_amount] = fill(0f0, nrow(tdf))
    end
    if :so_amount ∉ propertynames(tdf)
        tdf[!, :so_amount] = fill(0f0, nrow(tdf))
    end
    if :sc_amount ∉ propertynames(tdf)
        tdf[!, :sc_amount] = fill(0f0, nrow(tdf))
    end
    return tdf
end

function init_strategy_columns!(tdf::DataFrame)
    for contributor in Xch.tradesdf_all_contributors()
        contributor(tdf)
    end
    init_trade_columns!(tdf)
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
            label=Targets.TradeLabel[Targets.longopen],
            score=Float32[0.9f0],
        )
        init_strategy_columns!(tdf)
        TradingStrategy.gain_limit_reversal!(
            test_strategy(),
            tdf,
            1,
        )
        @test tdf[1, :label] == Targets.longopen
        @test isapprox(tdf[1, :lo_limit], 99.9f0; atol=1f-4)
        @test isapprox(tdf[1, :lc_limit], 101f0; atol=1f-4)
        @test tdf[1, :so_limit] == 0f0
    end

    @testset "short signal encodes short guidance" begin
        tdf = DataFrame(
            opentime=[DateTime(2026, 1, 1, 0, 0)],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            label=Targets.TradeLabel[Targets.shortopen],
            score=Float32[0.9f0],
        )
        init_strategy_columns!(tdf)
        TradingStrategy.gain_limit_reversal!(
            test_strategy(),
            tdf,
            1,
        )
        @test tdf[1, :label] == Targets.shortopen
        @test isapprox(tdf[1, :so_limit], 100.1f0; atol=1f-4)
        @test isapprox(tdf[1, :sc_limit], 99f0; atol=1f-4)
        @test tdf[1, :lo_limit] == 0f0
    end

    @testset "pricedelta gating keeps limits unchanged below threshold" begin
        tdf = DataFrame(
            opentime=[DateTime(2026, 1, 1, 0, 0), DateTime(2026, 1, 1, 0, 1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100.02f0],
            label=Targets.TradeLabel[Targets.longopen, Targets.longopen],
            score=Float32[0.9f0, 0.9f0],
        )
        init_strategy_columns!(tdf)
        TradingStrategy.gain_limit_reversal!(
            test_strategy(minpricedelta=0.01f0),
            tdf,
            1,
        )
        TradingStrategy.gain_limit_reversal!(
            test_strategy(minpricedelta=0.01f0),
            tdf,
            2,
        )
        @test isapprox(tdf[2, :lo_limit], tdf[1, :lo_limit]; atol=1f-6)
        @test isapprox(tdf[2, :lc_limit], tdf[1, :lc_limit]; atol=1f-6)
    end
end
