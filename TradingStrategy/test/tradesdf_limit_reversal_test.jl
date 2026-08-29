using Test
using Dates
using DataFrames
using TradingStrategy
using Targets
using Xch
using TSM

function init_limit_reversal_columns!(tdf::DataFrame)
    for contributor in TSM.tradesdf_all_contributors()
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
    # Opens are funded from freequote, so probes need an account that can pay for them.
    tdf[!, :freequote] = fill(1f6, nrow(tdf))
    tdf[!, :freemargin] = fill(1f6, nrow(tdf))
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
        score=Float32[0.9f0, 0.9f0, 0.9f0],
        label=TradeLabel[longopen, allclose, shortopen],
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

    @test tradesdf[1, :label] == longopen
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
            label=TradeLabel[longopen],
        )
        init_limit_reversal_columns!(probe)
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 1)
        @test isnothing(TradingStrategy._open_hit_spec(tcols(probe), 1))
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), tcols(probe), 1)
        openhit = TradingStrategy._open_hit_spec(tcols(probe), 1)
        @test !isnothing(openhit)
        @test openhit.side == :long
        # flat lane: the full quote budget is spent at the open limit
        @test isapprox(openhit.amount, limit_reversal_strategy().maxbudgetquote / openhit.limitprice; atol=1f-4)
    end

    @testset "scheduled open materializes on next row" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 100f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0f0],
            label=TradeLabel[longopen, allclose],
        )
        init_limit_reversal_columns!(probe)
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 1)
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), tcols(probe), 1)
        openhit = TradingStrategy._open_hit_spec(tcols(probe), 1)
        @test !isnothing(openhit)
        TradingStrategy._rowtakeover!(TSM.TradesColumns(probe), 2)
        TradingStrategy._apply_open_hit!(limit_reversal_strategy(), tcols(probe), 2, openhit.side, openhit.limitprice, openhit.amount)
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
            label=TradeLabel[longopen, longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[2, :lp_amount] = 100f0
        probe[2, :lol_pavg] = 98f0
        probe[2, :lastopentrade] = probe[1, :opentime]
        openhit = (side=:long, limitprice=99f0, amount=25f0)
        TradingStrategy._apply_open_hit!(limit_reversal_strategy(), tcols(probe), 2, openhit.side, openhit.limitprice, openhit.amount)
        @test probe[2, :lp_amount] == 125f0
        @test isapprox(probe[2, :lol_pavg], 98.2f0; atol=1f-4)
        @test probe[2, :lastopentrade] == probe[1, :opentime]
    end

    @testset "advice row suppresses same-side extension once budget is spent" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=TradeLabel[shortopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :sp_amount] = 100f0
        probe[1, :sol_pavg] = 2.28228f0
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 1)
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), tcols(probe), 1)
        @test probe[1, :so_amount] == 0f0
        @test probe[1, :sol_pavg] == 2.28228f0
        @test isnothing(TradingStrategy._open_hit_spec(tcols(probe), 1))
    end

    @testset "advice row caps the open by available freequote" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=TradeLabel[longopen],
        )
        init_limit_reversal_columns!(probe)
        strategy = limit_reversal_strategy()
        # less free quote than the lane budget, so equity is the binding constraint
        probe[1, :freequote] = strategy.maxbudgetquote / 5f0
        TradingStrategy.gain_limit_reversal!(strategy, tcols(probe), 1)
        TradingStrategy._process_advice_row!(strategy, tcols(probe), 1)
        openhit = TradingStrategy._open_hit_spec(tcols(probe), 1)
        @test !isnothing(openhit)
        @test isapprox(openhit.amount, probe[1, :freequote] / openhit.limitprice; atol=1f-4)
    end

    @testset "advice row posts no open without free quote" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=TradeLabel[longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :freequote] = 0f0
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 1)
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), tcols(probe), 1)
        @test probe[1, :lo_amount] == 0f0
        @test isnothing(TradingStrategy._open_hit_spec(tcols(probe), 1))
    end

    @testset "advice row posts no dust order when the lane sits at its budget" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=TradeLabel[longopen],
        )
        init_limit_reversal_columns!(probe)
        strategy = limit_reversal_strategy()
        # lane invested to just under the budget, leaving far less than one minimum order
        probe[1, :lol_pavg] = 0.0102697f0
        probe[1, :lp_amount] = (strategy.maxbudgetquote - 1f-3) / probe[1, :lol_pavg]
        TradingStrategy.gain_limit_reversal!(strategy, tcols(probe), 1)
        TradingStrategy._process_advice_row!(strategy, tcols(probe), 1)
        @test probe[1, :lo_amount] == 0f0
        @test isnothing(TradingStrategy._open_hit_spec(tcols(probe), 1))
    end

    @testset "advice row tops up a lane that is below its budget" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=TradeLabel[shortopen],
        )
        init_limit_reversal_columns!(probe)
        strategy = limit_reversal_strategy()
        # invest a quarter of the lane budget, so three quarters remain available
        invested_quote = strategy.maxbudgetquote / 4f0
        probe[1, :sol_pavg] = 2.28228f0
        probe[1, :sp_amount] = invested_quote / probe[1, :sol_pavg]
        TradingStrategy.gain_limit_reversal!(strategy, tcols(probe), 1)
        TradingStrategy._process_advice_row!(strategy, tcols(probe), 1)
        openhit = TradingStrategy._open_hit_spec(tcols(probe), 1)
        @test !isnothing(openhit)
        @test openhit.side == :short
        @test isapprox(openhit.amount, (strategy.maxbudgetquote - invested_quote) / openhit.limitprice; atol=1f-3)
    end

    @testset "advice row clears stale opposite open amount" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=TradeLabel[longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :lp_amount] = 100f0
        probe[1, :lol_pavg] = 98f0
        probe[1, :so_amount] = 100f0
        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 1)
        TradingStrategy._process_advice_row!(limit_reversal_strategy(), tcols(probe), 1)
        @test probe[1, :so_amount] == 0f0
        # the long lane already holds more than its budget, so no additional open is posted
        @test probe[1, :lo_amount] == 0f0
    end

    @testset "flip row closes before queued open" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[longopen, longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :sp_amount] = 100f0
        probe[1, :sol_pavg] = 101f0
        probe[1, :so_limit] = 101f0
        probe[1, :sc_limit] = 99f0
        probe[1, :lastopentrade] = probe[1, :opentime]
        probe[1, :lo_limit] = 99f0
        probe[1, :lo_amount] = 100f0

        openhit = TradingStrategy._open_hit_spec(tcols(probe), 1)
        @test !isnothing(openhit)

        TradingStrategy._rowtakeover!(TSM.TradesColumns(probe), 2)
        gaindf_flip = TradingStrategy.emptygaindf()
        last_openix = TradingStrategy._materialize_gains_sample_from_trades!(gaindf_flip, tcols(probe), 2, 1)
        @test last_openix == 0
        @test probe[2, :sp_amount] == 0f0

        TradingStrategy._apply_open_hit!(limit_reversal_strategy(), tcols(probe), 2, openhit.side, openhit.limitprice, openhit.amount)
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
            label=TradeLabel[shortopen, ignore, allclose],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :sp_amount] = 100f0
        probe[1, :sol_pavg] = 2f0
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
        last_openix = TradingStrategy._materialize_gains_sample_from_trades!(gaindf_probe, tcols(probe), 3, 1)
        @test last_openix == 0
        @test nrow(gaindf_probe) == 1
        @test isfinite(gaindf_probe[1, :gain])
        @test isapprox(gaindf_probe[1, :gain], 0.05f0; atol=1f-6)
    end

    @testset "gain segment open at range end is dropped, not closed at close price" begin
        # No limit is ever hit, so the position is still open when the range ends.
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[longopen, longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :lp_amount] = 100f0
        probe[1, :lol_pavg] = 98f0
        probe[1, :lastopentrade] = probe[1, :opentime]
        probe[1, :lc_limit] = 150f0     # unreachable within the bar
        probe[1, :lcsl_limit] = 50f0    # unreachable within the bar
        probe[2, :] = probe[1, :]
        probe[2, :opentime] = dt + Minute(1)

        gaindf_probe = TradingStrategy.emptygaindf()
        lastix = nrow(probe)
        last_openix = TradingStrategy._materialize_gains_sample_from_trades!(gaindf_probe, tcols(probe), lastix, 1)
        @test nrow(gaindf_probe) == 0
        @test last_openix == 1
        @test probe[lastix, :lp_amount] == 100f0
        @test probe[lastix, :lcl_pavg] == 0f0
    end

    @testset "close bracket re-anchors after position close" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[121f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[120f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[longopen, longopen],
        )
        init_limit_reversal_columns!(probe)
        # row 1 state of a position that closed at its take profit: limits are still resting
        probe[1, :lc_limit] = 121f0
        probe[1, :lcsl_limit] = 114f0
        probe[1, :lo_limit] = 119.88f0

        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 2)
        @test isapprox(probe[2, :lc_limit], 101f0; atol=1f-4)
        @test probe[2, :lcsl_limit] == 0f0
    end

    @testset "held long refreshes target from current close and stop follows close" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[longopen, longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :lc_limit] = 105f0
        probe[2, :lp_amount] = 100f0
        probe[2, :lol_pavg] = 98f0
        probe[2, :lastopentrade] = probe[1, :opentime]

        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 2)
        @test isapprox(probe[2, :lc_limit], 101f0; atol=1f-4)
        @test isapprox(probe[2, :lcsl_limit], 100f0 * 0.95f0; atol=1f-4)
    end

    @testset "short close bracket is symmetric" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[80f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[shortopen, shortopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :sc_limit] = 79.2f0
        probe[1, :scsl_limit] = 84f0
        probe[2, :sp_amount] = 100f0
        probe[2, :sol_pavg] = 80f0
        probe[2, :lastopentrade] = probe[1, :opentime]

        TradingStrategy.gain_limit_reversal!(limit_reversal_strategy(), tcols(probe), 2)
        @test isapprox(probe[2, :sc_limit], 99f0; atol=1f-4)
        @test isapprox(probe[2, :scsl_limit], 105f0; atol=1f-4)
    end

    @testset "zero close limit keeps the stop leg" begin
        probe = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
            score=Float32[0.9f0],
            label=TradeLabel[longopen],
        )
        init_limit_reversal_columns!(probe)
        probe[1, :lp_amount] = 100f0
        probe[1, :lol_pavg] = 98f0
        probe[1, :lastopentrade] = probe[1, :opentime]

        TradingStrategy._setclosebracket!(limit_reversal_strategy(), tcols(probe), 1, longclose, probe[1, :close], 0f0)
        @test probe[1, :lc_limit] == 0f0
        @test isapprox(probe[1, :lcsl_limit], 95f0; atol=1f-4)
    end

    @testset "open fill anchors both bracket legs on last close" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 101f0],
            low=Float32[99f0, 99f0],
            close=Float32[100f0, 100f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[longopen, longopen],
        )
        init_limit_reversal_columns!(probe)
        TradingStrategy._apply_open_hit!(limit_reversal_strategy(), tcols(probe), 2, :long, 99.9f0, 100f0)
        @test isapprox(probe[2, :lc_limit], 101f0; atol=1f-4)
        @test isapprox(probe[2, :lcsl_limit], 95f0; atol=1f-4)
    end

    @testset "long stop loss materializes a loss and clears the bracket" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 100f0],
            low=Float32[99f0, 94f0],
            close=Float32[100f0, 95f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[longopen, longhold],
        )
        init_limit_reversal_columns!(probe)
        probe[2, :lp_amount] = 100f0
        probe[2, :lol_pavg] = 100f0
        probe[2, :lastopentrade] = probe[1, :opentime]
        probe[2, :lc_limit] = 101f0
        probe[2, :lcsl_limit] = 95f0

        gaindf_stop = TradingStrategy.emptygaindf()
        last_openix = TradingStrategy._materialize_gains_sample_from_trades!(gaindf_stop, tcols(probe), 2, 1)
        @test last_openix == 0
        @test nrow(gaindf_stop) == 1
        @test isapprox(gaindf_stop[1, :gain], -0.05f0; atol=1f-6)
        @test probe[2, :lc_limit] == 0f0
        @test probe[2, :lcsl_limit] == 0f0
    end

    @testset "short stop loss materializes a loss" begin
        probe = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[101f0, 106f0],
            low=Float32[99f0, 100f0],
            close=Float32[100f0, 105f0],
            score=Float32[0.9f0, 0.9f0],
            label=TradeLabel[shortopen, shorthold],
        )
        init_limit_reversal_columns!(probe)
        probe[2, :sp_amount] = 100f0
        probe[2, :sol_pavg] = 100f0
        probe[2, :lastopentrade] = probe[1, :opentime]
        probe[2, :sc_limit] = 99f0
        probe[2, :scsl_limit] = 105f0

        gaindf_stop = TradingStrategy.emptygaindf()
        last_openix = TradingStrategy._materialize_gains_sample_from_trades!(gaindf_stop, tcols(probe), 2, 1)
        @test last_openix == 0
        @test nrow(gaindf_stop) == 1
        @test isapprox(gaindf_stop[1, :gain], -0.05f0; atol=1f-6)
        @test probe[2, :sc_limit] == 0f0
        @test probe[2, :scsl_limit] == 0f0
    end

    tp = TradingStrategy.TsTp(
        pair="BTCUSDT",
        tradesdf=DataFrame(
            opentime=tradesdf[!, :opentime],
            high=tradesdf[!, :high],
            low=tradesdf[!, :low],
            close=tradesdf[!, :close],
            score=tradesdf[!, :score],
            label=TradeLabel[tradesdf[ix, :label] for ix in 1:nrow(tradesdf)],
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
            label=TradeLabel[longopen, allclose],
        ),
    )
    init_limit_reversal_columns!(tp.tradesdf)

    TradingStrategy.simulate_gains!(limit_reversal_strategy(), tp, 2)
    @test tp.last_update_dt == dt + Minute(1)
    @test tp.tradesdf[1, :label] == longopen
end
