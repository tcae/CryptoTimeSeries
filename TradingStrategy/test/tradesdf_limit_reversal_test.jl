using Test
using Dates
using DataFrames
using TradingStrategy
using Targets
using Xch

function init_limit_reversal_columns!(tdf::DataFrame)
    for contributor in Xch.tradesdf_all_contributors()
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
        score=Float32[0.9f0, 0f0, 0.9f0],
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

    @testset "open spec is scheduled after advice" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=Targets.TradeLabel[Targets.longopen],
        )
        init_limit_reversal_columns!(probe)
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), probe, 1)
        @test isnothing(TradingStrategy._open_hit_spec(probe, 1))
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), probe, 1)
        openhit = TradingStrategy._open_hit_spec(probe, 1)
        @test !isnothing(openhit)
        @test openhit.side == :long
        @test openhit.amount == 100f0
    end

    @testset "scheduled open materializes on next row" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 100f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0f0],
            label=Targets.TradeLabel[Targets.longopen, Targets.allclose],
        )
        init_limit_reversal_columns!(probe)
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), probe, 1)
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), probe, 1)
        openhit = TradingStrategy._open_hit_spec(probe, 1)
        @test !isnothing(openhit)
        TradingStrategy._rowtakeover!(probe, 2)
        TradingStrategy._apply_open_hit!(probe, 2, openhit.side, openhit.limitprice, openhit.amount)
        @test ismissing(probe[1, :lastopentrade])
        @test probe[2, :lastopentrade] == probe[2, :opentime]
        @test probe[2, :lp_amount] == openhit.amount
    end

    @testset "same-side open hit extends amount" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=Targets.TradeLabel[Targets.longopen, Targets.longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[2, :lp_amount] = 100f0
        probe[2, :lo_pavg] = 98f0
        probe[2, :lastopentrade] = probe[1, :opentime]
        openhit = (side=:long, limitprice=99f0, amount=25f0)
        TradingStrategy._apply_open_hit!(probe, 2, openhit.side, openhit.limitprice, openhit.amount)
        @test probe[2, :lp_amount] == 125f0
        @test isapprox(probe[2, :lo_pavg], 98.2f0; atol=1f-4)
        @test probe[2, :lastopentrade] == probe[1, :opentime]
    end

    @testset "advice row enables same-side short extension" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=Targets.TradeLabel[Targets.shortopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :sp_amount] = 100f0
        probe[1, :so_pavg] = 2.28228f0
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), probe, 1)
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), probe, 1)
        @test probe[1, :so_amount] == 100f0
        @test probe[1, :so_pavg] == 2.28228f0
        openhit = TradingStrategy._open_hit_spec(probe, 1)
        @test !isnothing(openhit)
        @test openhit.side == :short
        @test openhit.amount == 100f0
    end

    @testset "advice row clears stale opposite open amount" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=Targets.TradeLabel[Targets.longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :lp_amount] = 100f0
        probe[1, :lo_pavg] = 98f0
        probe[1, :so_amount] = 100f0
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), probe, 1)
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), probe, 1)
        @test probe[1, :so_amount] == 0f0
        @test probe[1, :lo_amount] == 100f0
    end

    @testset "flip row closes before queued open" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=Targets.TradeLabel[Targets.longopen, Targets.longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :sp_amount] = 100f0
        probe[1, :so_pavg] = 101f0
        probe[1, :so_limit] = 101f0
        probe[1, :sc_limit] = 99f0
        probe[1, :lastopentrade] = probe[1, :opentime]
        probe[1, :lo_limit] = 99f0
        probe[1, :lo_amount] = 100f0

        openhit = TradingStrategy._open_hit_spec(probe, 1)
        @test !isnothing(openhit)

        TradingStrategy._rowtakeover!(probe, 2)
        gaindf_flip = TradingStrategy.emptygaindf()
        last_openix = TradingStrategy._materialize_gains_sample_from_trades!(gaindf_flip, probe, 2, 1; lastix=2)
        @test last_openix == 0
        @test probe[2, :sp_amount] == 0f0

        TradingStrategy._apply_open_hit!(probe, 2, openhit.side, openhit.limitprice, openhit.amount)
        @test probe[2, :lp_amount] == openhit.amount
        @test probe[2, :sp_amount] == 0f0
        @test probe[2, :lastopentrade] == probe[2, :opentime]
    end

    @testset "materialized gains use pavg not reset open limit" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1), dt + Minute(2)],
            high=Float32[101f0, 101f0, 101f0],
            low=Float32[99f0, 99f0, 99f0],
            close=Float32[100f0, 100f0, 100f0],
            score=Float32[0.9f0, 0.9f0, 0.2f0],
            label=Targets.TradeLabel[Targets.shortopen, Targets.ignore, Targets.allclose],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :sp_amount] = 100f0
        probe[1, :so_pavg] = 2f0
        probe[1, :lastopentrade] = probe[1, :opentime]
        probe[1, :sc_limit] = 1.9f0
        probe[1, :so_limit] = 0f0
        probe[2, :] = probe[1, :]
        probe[2, :opentime] = dt + Minute(1)
        probe[3, :] = probe[2, :]
        probe[3, :opentime] = dt + Minute(2)
        probe[3, :low] = 1.85f0
        probe[3, :high] = 2.05f0

        gaindf_probe = TradingStrategy.emptygaindf()
        last_openix = TradingStrategy._materialize_gains_sample_from_trades!(gaindf_probe, probe, 3, 1; lastix=3)
        @test last_openix == 0
        @test nrow(gaindf_probe) == 1
        @test isfinite(gaindf_probe[1, :gain])
        @test isapprox(gaindf_probe[1, :gain], 0.05f0; atol=1f-6)
    end

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

    @test ismissing(tp.tradesdf[1, :lastopentrade])
    @test tp.tradesdf[2, :lastopentrade] == tp.tradesdf[2, :opentime]
    if (tp.tradesdf[3, :lp_amount] > 0f0) || (tp.tradesdf[3, :sp_amount] > 0f0)
        @test !ismissing(tp.tradesdf[3, :lastopentrade])
    else
        @test ismissing(tp.tradesdf[3, :lastopentrade])
    end
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
