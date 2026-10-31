module SimulatedMarketviewTest

include("testfixtures.jl")
using .TradeTestFixtures
using Test
using Dates
using DataFrames
using EnvConfig, Trade, Xch, Ohlcv, TradingStrategy

@testset "TradeCache requires explicit strategy configuration" begin
    EnvConfig.init(EnvConfig.test)
    xc = Xch.XchCache()
    @test_throws UndefKeywordError Trade.TradeCache(; xc=xc)
    @test_throws ArgumentError Trade.TradeCache(TradingStrategy.StrategyConfig(); xc=xc)
    strategy = TradeTestFixtures.strategy()
    runtime = TradingStrategy.TsCache(strategy=strategy, source="test")
    @test Trade.TradeCache(runtime; xc=xc).ts === runtime
    @test Trade.TradeCache(; ts=runtime, xc=xc).ts === runtime
    cache = Trade.TradeCache(strategy; xc=xc)
    @test cache.ts.cfg.classifier === strategy.classifier
    @test cache.ts.cfg.algorithm === strategy.algorithm
    @test cache.ts.cfg.algorithmconfig === strategy.algorithmconfig
    @test Trade._tradeselection_history_minutes(cache) >= 24 * 60
end

@testset "simulated marketview" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-05-18T00:00:00")
    dt = startdt + Minute(1)

    ohlcv = Ohlcv.OhlcvData(
        DataFrame(
            opentime=[startdt, dt],
            open=Float32[100, 110],
            high=Float32[101, 111],
            low=Float32[99, 109],
            close=Float32[100, 110],
            basevolume=Float32[2, 3],
            pivot=Float32[100, 110],
        ),
        "BTC",
        uppercase(EnvConfig.pairquote),
        "1m",
        2,
        nothing,
    )

    xc = Xch.XchCache(startdt=startdt, enddt=dt)
    Xch.addbase!(xc, ohlcv)
    tc = Trade.TradeCache(TradeTestFixtures.strategy(); xc=xc, trademode=Trade.notrade)

    mdf = Trade._simulated_usdtmarketview(tc, dt, Set(["BTC"]), startdt)
    @test nrow(mdf) == 1
    @test mdf[1, :basecoin] == "BTC"
    @test isapprox(mdf[1, :quotevolume24h], 530.0; atol=1e-6)
    @test isapprox(mdf[1, :pricechangepercent], 10.0f0; atol=1e-5)
    @test mdf[1, :lastprice] == 110.0f0

    @test Trade._uses_simulated_marketview(tc)

    xc_live = Xch.XchCache(startdt=startdt, enddt=nothing)
    tc_live = Trade.TradeCache(TradeTestFixtures.strategy(); xc=xc_live, trademode=Trade.notrade)
    @test !Trade._uses_simulated_marketview(tc_live)
end

end
