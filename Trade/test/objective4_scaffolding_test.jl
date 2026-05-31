using Test
using Dates
using DataFrames
using EnvConfig, Trade, CryptoXch

@testset "Objective 4 scaffolding" begin
    EnvConfig.init(test)

    cache = Trade.TradeCache(xc=CryptoXch.XchCache())
    @test haskey(cache.mc, :async_engine_enabled)
    @test haskey(cache.mc, :async_shadow_mode)
    @test haskey(cache.mc, :ws_marketdata_enabled)
    @test haskey(cache.mc, :exchange_balance_cache_owner)
    @test haskey(cache.mc, :ohlcv_gap_backfill_on_tradable)

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
    @test size(cache.mc[:exchange_balances_snapshot], 1) == 2
    @test cache.mc[:exchange_balances_snapshot_dt] == DateTime("2026-05-30T12:00:00")
end
