using Test
using Dates
using DataFrames
using EnvConfig, Trade, CryptoXch, Ohlcv

@testset "Objective 4 scaffolding" begin
    EnvConfig.init(test)

    cache = Trade.TradeCache(xc=CryptoXch.XchCache())
    @test haskey(cache.mc, :async_engine_enabled)
    @test haskey(cache.mc, :async_shadow_mode)
    @test haskey(cache.mc, :ws_marketdata_enabled)
    @test haskey(cache.mc, :exchange_balance_cache_owner)
    @test haskey(cache.mc, :ohlcv_gap_backfill_on_tradable)
    @test haskey(cache.mc, :async_worker_channels)
    @test haskey(cache.mc, :async_worker_heartbeats)
    @test haskey(cache.mc, :async_worker_watchdog_breaches)
    @test haskey(cache.mc, :marketdata_source)
    @test haskey(cache.mc, :marketdata_ws_fallback_active)
    @test haskey(cache.mc, :marketdata_ws_fallback_switches)
    @test haskey(cache.mc, :tradable_ohlcv_state_by_base)
    @test haskey(cache.mc, :tradable_ohlcv_state_dt_by_base)
    @test haskey(cache.mc, :strategy_last_closed_candle_dt)
    @test haskey(cache.mc, :async_canary_bases_raw)
    @test haskey(cache.mc, :objective4_cycle_count)
    @test haskey(cache.mc, :objective4_canary_cycles)
    @test haskey(cache.mc, :objective4_canary_last_bases)
    @test haskey(cache.mc, :objective4_order_rejects)
    @test haskey(cache.mc, :objective4_permission_rejects)
    @test haskey(cache.mc, :objective4_privatecooldown_skips)
    @test haskey(cache.mc, :objective4_marketdata_fallback_activations)
    @test haskey(cache.mc, :objective4_watchdog_breaches_total)
    @test haskey(cache.mc, :objective4_last_worker_latency_ms)

    sync_advices = [
        Trade.StrategyAdvice(
            classifier=cache.cl,
            base="BTC",
            datetime=DateTime("2026-05-30T00:00:00"),
            tradelabel=Trade.longbuy,
            relativeamount=1f0,
            price=100f0,
        )
    ]

    async_advices = deepcopy(sync_advices)
    @test Trade._run_async_shadow_compare!(cache, sync_advices, async_advices)
    @test cache.mc[:async_shadow_last_compare].ok

    eth_advice = Trade.StrategyAdvice(
        classifier=cache.cl,
        base="ETH",
        datetime=DateTime("2026-05-30T00:00:00"),
        tradelabel=Trade.longbuy,
        relativeamount=1f0,
        price=100f0,
    )
    canary_pool = [sync_advices[1], eth_advice]
    cache.mc[:async_canary_bases_raw] = "BTCUSDT, ETH/USDT"
    canary_selected = Trade._select_async_shadow_advices(cache, canary_pool)
    @test canary_selected.canary_enabled
    @test sort([ta.base for ta in canary_selected.advices]) == ["BTC", "ETH"]
    @test canary_selected.canary_bases == ["BTC", "ETH"]
    cache.mc[:async_canary_bases_raw] = ""
    canary_off = Trade._select_async_shadow_advices(cache, canary_pool)
    @test !canary_off.canary_enabled
    @test length(canary_off.advices) == length(canary_pool)

    cache.mc[:async_engine_enabled] = true
    divergent_advices = deepcopy(sync_advices)
    divergent_advices[1].tradelabel = Trade.shortbuy
    @test !Trade._run_async_shadow_compare!(cache, sync_advices, divergent_advices)
    @test cache.mc[:async_shadow_autodisabled]
    @test !cache.mc[:async_engine_enabled]
    @test !isnothing(cache.mc[:async_shadow_autodisable_reason])

    assets = DataFrame(
        coin=["USDT", "BTC"],
        free=[1000f0, 0.1f0],
        locked=[0f0, 0f0],
        borrowed=[0f0, 0f0],
        usdtprice=[1f0, 50000f0],
        usdtvalue=[1000f0, 5000f0],
    )

    cache.mc[:exchange_balance_cache_owner] = false
    Trade._sync_exchange_balances_snapshot!(cache, assets)
    @test size(cache.mc[:exchange_balances_snapshot], 1) == 0

    cache.mc[:exchange_balance_cache_owner] = true
    cache.xc.currentdt = DateTime("2026-05-30T12:00:00")
    Trade._sync_exchange_balances_snapshot!(cache, assets)
    @test cache.mc[:exchange_balances_snapshot] isa DataFrame
    @test cache.mc[:exchange_balances_snapshot_dt] == DateTime("2026-05-30T12:00:00")

    refreshed = CryptoXch.balancessnapshot(cache.xc; force_refresh=true, ignoresmallvolume=false)
    @test refreshed.fresh
    @test refreshed.datetime == cache.xc.currentdt
    @test refreshed.snapshot isa DataFrame

    cache.xc.currentdt = cache.xc.currentdt + Minute(1)
    stale = CryptoXch.balancessnapshot(cache.xc; force_refresh=false, max_age=Second(0), ignoresmallvolume=false)
    @test !stale.fresh
    @test stale.datetime == DateTime("2026-05-30T12:00:00")

    cache.mc[:async_engine_enabled] = true
    topology_advices = Trade._run_async_shadow_topology!(cache, assets, DataFrame(), sync_advices)
    @test length(topology_advices) == 1
    @test cache.mc[:async_worker_topology_started]
    @test cache.mc[:objective4_last_worker_latency_ms] isa Dict{Symbol, Float64}
    for worker in (:marketdata, :balance_sync, :order_management)
        @test haskey(cache.mc[:async_worker_channels], worker)
        @test haskey(cache.mc[:async_worker_heartbeats], worker)
        @test haskey(cache.mc[:async_worker_last_latency_ms], worker)
        @test haskey(cache.mc[:async_worker_watchdog_breaches], worker)
        @test haskey(cache.mc[:objective4_last_worker_latency_ms], worker)
    end

    cache.mc[:async_worker_watchdog_timeout] = Second(0)
    cache.xc.currentdt = cache.xc.currentdt + Minute(2)
    breaches_total_before = cache.mc[:objective4_watchdog_breaches_total]
    Trade._update_async_worker_watchdog!(cache)
    @test any(values(cache.mc[:async_worker_watchdog_breaches]) .> 0)
    @test cache.mc[:objective4_watchdog_breaches_total] > breaches_total_before

    cache.xc.currentdt = DateTime("2026-05-30T12:05:00")
    cache.mc[:ws_marketdata_enabled] = false
    md_disabled = Trade._update_marketdata_freshness_policy!(cache)
    @test md_disabled.source == :http
    @test !md_disabled.fallback_active
    @test md_disabled.fallback_reason == :ws_disabled

    cache.mc[:ws_marketdata_enabled] = true
    cache.mc[:marketdata_ws_freshness_sla] = Second(30)
    cache.mc[:marketdata_ws_last_update_dt] = nothing
    fallback_activations_before = cache.mc[:objective4_marketdata_fallback_activations]
    md_missing = Trade._update_marketdata_freshness_policy!(cache)
    @test md_missing.source == :http
    @test md_missing.fallback_active
    @test md_missing.fallback_reason == :ws_no_updates
    @test cache.mc[:objective4_marketdata_fallback_activations] == fallback_activations_before + 1

    Trade._mark_marketdata_ws_update!(cache; datetime=cache.xc.currentdt)
    md_fresh = Trade._update_marketdata_freshness_policy!(cache)
    @test md_fresh.source == :ws
    @test !md_fresh.fallback_active
    @test isnothing(md_fresh.fallback_reason)

    cache.mc[:marketdata_ws_last_update_dt] = nothing
    CryptoXch.setmarketdataheartbeat!(cache.xc, cache.xc.currentdt)
    md_from_exchange = Trade._update_marketdata_freshness_policy!(cache)
    @test md_from_exchange.source == :ws
    @test cache.mc[:marketdata_ws_last_update_dt] == cache.xc.currentdt

    switches_before = cache.mc[:marketdata_ws_fallback_switches]
    cache.xc.currentdt = cache.xc.currentdt + Minute(2)
    md_stale = Trade._update_marketdata_freshness_policy!(cache)
    @test md_stale.source == :http
    @test md_stale.fallback_active
    @test md_stale.fallback_reason == :ws_stale
    @test cache.mc[:marketdata_ws_fallback_switches] == switches_before + 1
    @test cache.mc[:objective4_marketdata_fallback_activations] == fallback_activations_before + 2

    Trade._mark_marketdata_ws_update!(cache; datetime=cache.xc.currentdt)
    md_recovered = Trade._update_marketdata_freshness_policy!(cache)
    @test md_recovered.source == :ws
    @test !md_recovered.fallback_active

    cache.xc.currentdt = DateTime("2026-05-30T12:10:00")
    cache.mc[:ws_marketdata_enabled] = true
    cache.mc[:marketdata_ws_freshness_sla] = Second(30)
    Trade._mark_marketdata_ws_update!(cache; datetime=cache.xc.currentdt)
    CryptoXch.setmarketdataheartbeat!(cache.xc, "BTCUSDT", cache.xc.currentdt - Minute(2))
    md_symbol_stale = Trade._update_marketdata_freshness_policy!(cache; symbols=["BTCUSDT"])
    @test md_symbol_stale.source == :http
    @test md_symbol_stale.fallback_active
    @test md_symbol_stale.fallback_reason == :ws_symbol_stale
    @test "BTCUSDT" in md_symbol_stale.stale_symbols

    CryptoXch.setmarketdataheartbeat!(cache.xc, "BTCUSDT", cache.xc.currentdt)
    md_symbol_fresh = Trade._update_marketdata_freshness_policy!(cache; symbols=["BTCUSDT"])
    @test md_symbol_fresh.source == :ws
    @test isempty(md_symbol_fresh.stale_symbols)

    cache.mc[:ohlcv_gap_backfill_on_tradable] = true
    gapped = Ohlcv.defaultohlcv("BTC")
    Ohlcv.setdataframe!(gapped, DataFrame(
        opentime=[DateTime("2026-05-30T00:00:00"), DateTime("2026-05-30T00:02:00")],
        open=[100f0, 102f0],
        high=[101f0, 103f0],
        low=[99f0, 101f0],
        close=[100.5f0, 102.5f0],
        basevolume=[1f0, 1f0],
        pivot=[0f0, 0f0],
    ))
    readiness = Trade._prepare_tradable_ohlcv!(cache, gapped; datetime=DateTime("2026-05-30T00:02:00"))
    @test readiness.state == :data_ready
    @test readiness.ready

    emptyohlcv = Ohlcv.defaultohlcv("ETH")
    emptystate = Trade._prepare_tradable_ohlcv!(cache, emptyohlcv; datetime=DateTime("2026-05-30T00:02:00"))
    @test emptystate.state == :backfill_required
    @test !emptystate.ready

    cache.xc.currentdt = DateTime("2026-05-30T12:20:00")
    @test Trade._tradable_ohlcv_state(cache, "BTC") == :discovered
    Trade._set_tradable_ohlcv_state!(cache, "BTC", :backfill_required; datetime=cache.xc.currentdt)
    @test Trade._tradable_ohlcv_state(cache, "BTC") == :backfill_required
    @test cache.mc[:tradable_ohlcv_state_dt_by_base]["BTC"] == DateTime("2026-05-30T12:20:00")

    resumed = Trade.TradeCache(xc=CryptoXch.XchCache())
    resumed.mc[:tradable_ohlcv_state_by_base] = deepcopy(cache.mc[:tradable_ohlcv_state_by_base])
    resumed.mc[:tradable_ohlcv_state_dt_by_base] = deepcopy(cache.mc[:tradable_ohlcv_state_dt_by_base])
    @test Trade._tradable_ohlcv_state(resumed, "BTC") == :backfill_required
    Trade._set_tradable_ohlcv_state!(resumed, "BTC", :data_ready; datetime=DateTime("2026-05-30T12:21:00"))
    @test Trade._tradable_ohlcv_state(resumed, "BTC") == :data_ready
    @test resumed.mc[:tradable_ohlcv_state_dt_by_base]["BTC"] == DateTime("2026-05-30T12:21:00")
    Trade._set_tradable_ohlcv_state!(resumed, "BTC", :tradable; datetime=DateTime("2026-05-30T12:22:00"))
    @test Trade._tradable_ohlcv_state(resumed, "BTC") == :tradable

    cache.xc.currentdt = DateTime("2026-05-30T12:11:00")
    next_closed = Trade._next_closed_candle_dt!(cache)
    @test next_closed == DateTime("2026-05-30T12:10:00")
    Trade._mark_closed_candle_consumed!(cache, next_closed)
    @test isnothing(Trade._next_closed_candle_dt!(cache))
    @test cache.mc[:strategy_closed_candle_pending_reason] == :no_new_closed_candle

    cache.xc.currentdt = DateTime("2026-05-30T12:12:00")
    progressed_closed = Trade._next_closed_candle_dt!(cache)
    @test progressed_closed == DateTime("2026-05-30T12:11:00")
    @test isnothing(cache.mc[:strategy_closed_candle_pending_reason])
end
