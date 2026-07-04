using Test
using Dates
using DataFrames
using TradingStrategy
using Targets
using Xch

function init_limit_reversal_columns!(tdf::DataFrame)
    for contributor in Xch.tradesdf_contributors()
        contributor(tdf)
    end
    for contributor in TradingStrategy.tradesdf_contributors()
        contributor(tdf)
    end
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

function limit_reversal_strategy(; maxwindow=4 * 60, minpricedelta=0f0)
    return TradingStrategy.StrategyConfig(
        openthreshold=0.6f0,
        buygain=0.001f0,
        sellgain=0.01f0,
        limitreduction=1f0,
        maxwindow=maxwindow,
        minpricedelta=minpricedelta,
    )
end

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
    init_limit_reversal_columns!(tradesdf)

    tpdf = TradingStrategy.TsTp(
        pair="BTCUSDT",
        tradesdf=tradesdf,
    )
    TradingStrategy.simulate_gains!(limit_reversal_strategy(), tpdf, 3)

    @test :lo_limit in propertynames(tradesdf)
    @test :lc_limit in propertynames(tradesdf)
    @test :so_limit in propertynames(tradesdf)
    @test :sc_limit in propertynames(tradesdf)
    @test :label in propertynames(tradesdf)
    @test :score in propertynames(tradesdf)
    @test :lastopentrade in propertynames(tradesdf)

    @test tradesdf[1, :label] == Targets.longopen
    @test isapprox(tradesdf[1, :lo_limit], 99.9f0; atol=1f-4)
    @test isapprox(tradesdf[1, :lc_limit], 101f0; atol=1f-4)
    @test ismissing(tradesdf[1, :lastopentrade])

    @test tradesdf[2, :label] == Targets.allclose
    @test isapprox(tradesdf[2, :lo_limit], tradesdf[1, :lo_limit]; atol=1f-4)
    @test tradesdf[2, :lc_limit] <= tradesdf[1, :lc_limit]
    @test ismissing(tradesdf[2, :lastopentrade])

    @test tradesdf[3, :label] == Targets.shortopen
    @test isapprox(tradesdf[3, :so_limit], 101.101f0; atol=1f-3)
    @test isapprox(tradesdf[3, :sc_limit], 99.99f0; atol=1f-3)
    @test ismissing(tradesdf[3, :lastopentrade])

    tp = TradingStrategy.TsTp(
        pair="BTCUSDT",
        tradesdf=DataFrame(
            opentime=tradesdf[!, :opentime],
            high=tradesdf[!, :high],
            low=tradesdf[!, :low],
            close=tradesdf[!, :close],
            score=tradesdf[!, :score],
            label=Targets.TradeLabel[tradesdf[ix, :label] for ix in 1:nrow(tradesdf)],
        ),
    )
    init_limit_reversal_columns!(tp.tradesdf)
    gaindf = TradingStrategy.emptygaindf()
    TradingStrategy.simulate_gains!(limit_reversal_strategy(), tp, nrow(tp.tradesdf), gaindf)
    @test names(gaindf) == names(TradingStrategy.emptygaindf())
    if nrow(gaindf) > 0
        @test gaindf[1, :startix] >= 1
        @test gaindf[1, :endix] >= gaindf[1, :startix]
    end

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
    init_limit_reversal_columns!(tp.tradesdf)

    TradingStrategy.simulate_gains!(limit_reversal_strategy(), tp, 2)
    @test tp.last_update_dt == dt + Minute(1)
    @test tp.tradesdf[1, :label] == longopen
end
