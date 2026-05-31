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
    for worker in (:marketdata, :strategy, :order_intent, :order_execution, :order_reconcile, :balance_sync)
        @test haskey(cache.mc[:async_worker_channels], worker)
        @test haskey(cache.mc[:async_worker_heartbeats], worker)
        @test haskey(cache.mc[:async_worker_last_latency_ms], worker)
        @test haskey(cache.mc[:async_worker_watchdog_breaches], worker)
    end

    cache.mc[:async_worker_watchdog_timeout] = Second(0)
    cache.xc.currentdt = cache.xc.currentdt + Minute(2)
    Trade._update_async_worker_watchdog!(cache)
    @test any(values(cache.mc[:async_worker_watchdog_breaches]) .> 0)

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
end
