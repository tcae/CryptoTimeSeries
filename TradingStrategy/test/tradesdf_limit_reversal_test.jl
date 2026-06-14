using Test
using Dates
using DataFrames
using TradingStrategy
using Targets

@testset "TradesDF limit-reversal variants" begin
    dt = DateTime(2026, 1, 8)
    tradesdf = DataFrame(
        opentime=[dt, dt + Minute(1), dt + Minute(2)],
        high=Float32[101f0, 102f0, 103f0],
        low=Float32[99f0, 100f0, 101f0],
        close=Float32[100f0, 101f0, 101f0],
        score=Float32[0.9f0, 0.1f0, 0.9f0],
        label=Targets.TradeLabel[Targets.longopen, Targets.allclose, Targets.shortopen],
    )

    TradingStrategy.gain_limit_reversal!(
        tradesdf,
        3;
        openthreshold=0.6f0,
        buygain=0.001f0,
        sellgain=0.01f0,
        limitreduction=1f0,
    )

    @test :longopenlimit in propertynames(tradesdf)
    @test :longcloselimit in propertynames(tradesdf)
    @test :shortopenlimit in propertynames(tradesdf)
    @test :shortcloselimit in propertynames(tradesdf)
    @test :tradelabel in propertynames(tradesdf)
    @test :labelscore in propertynames(tradesdf)
    @test :lastopentrade in propertynames(tradesdf)

    @test tradesdf[1, :tradelabel] == Targets.longopen
    @test isapprox(tradesdf[1, :longopenlimit], 99.9f0; atol=1f-4)
    @test isapprox(tradesdf[1, :longcloselimit], 101f0; atol=1f-4)
    @test ismissing(tradesdf[1, :lastopentrade])

    @test tradesdf[2, :tradelabel] == Targets.ignore
    @test isapprox(tradesdf[2, :longopenlimit], tradesdf[1, :longopenlimit]; atol=1f-4)
    @test tradesdf[2, :longcloselimit] <= tradesdf[1, :longcloselimit]
    @test ismissing(tradesdf[2, :lastopentrade])

    @test tradesdf[3, :tradelabel] == Targets.shortopen
    @test isapprox(tradesdf[3, :shortopenlimit], 101.101f0; atol=1f-3)
    @test isapprox(tradesdf[3, :shortcloselimit], 99.99f0; atol=1f-3)
    @test ismissing(tradesdf[3, :lastopentrade])

    gaindf = TradingStrategy.materialize_gains_from_trades(tradesdf, tradesdf)
    @test nrow(gaindf) >= 1
    @test gaindf[1, :trend] == Targets.up
    @test gaindf[1, :startix] == 1
    @test gaindf[1, :endix] == 1

    @test ismissing(tradesdf[1, :lastopentrade])
    @test ismissing(tradesdf[2, :lastopentrade])
    @test ismissing(tradesdf[3, :lastopentrade])
end

@testset "TsTp wrapper updates last_update_dt" begin
    dt = DateTime(2026, 1, 9)
    tp = TradingStrategy.TsTp(
        pair="BTCUSDT",
        tradesdf=DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.8f0, 0.2f0],
            label=Targets.TradeLabel[longopen, allclose],
        ),
    )

    TradingStrategy.gain_limit_reversal!(tp, 2; openthreshold=0.6f0)
    @test tp.last_update_dt == dt + Minute(1)
    @test tp.tradesdf[1, :tradelabel] == longopen
end
