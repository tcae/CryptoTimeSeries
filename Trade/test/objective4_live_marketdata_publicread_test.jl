using Test
using Dates
using EnvConfig, Trade, CryptoXch

@testset "Objective 4 live marketdata public-read (env gated)" begin
    if get(ENV, "CTS_RUN_LIVE_MARKETDATA_TESTS", "0") != "1"
        @info "Skipping live public-read marketdata test (set CTS_RUN_LIVE_MARKETDATA_TESTS=1 to enable)"
        @test true
        return
    end

    prevmode = EnvConfig.configmode
    try
        EnvConfig.init(EnvConfig.production)

        exchange = get(ENV, "CTS_LIVE_MARKETDATA_EXCHANGE", CryptoXch.EXCHANGE_KRAKENSPOT)
        xc = CryptoXch.XchCache(exchange=String(exchange))
        tc = Trade.TradeCache(xc=xc, trademode=Trade.notrade)
        tc.xc.currentdt = floor(Dates.now(Dates.UTC), Minute(1))

        # Public-read market snapshot should be available from live exchange endpoints.
        marketdf = CryptoXch.screeningUSDTmarket(tc.xc; dt=tc.xc.currentdt)
        @test size(marketdf, 1) > 0
        @test "basecoin" in names(marketdf)
        @test "lastprice" in names(marketdf)

        # Objective 4.5 canary acceptance scaffold: fallback source switching should remain observable.
        tc.mc[:ws_marketdata_enabled] = true
        tc.mc[:marketdata_ws_freshness_sla] = Second(30)
        tc.mc[:marketdata_ws_last_update_dt] = nothing
        http_state = Trade._update_marketdata_freshness_policy!(tc; symbols=["BTCUSDT"])
        @test http_state.source == :http
        @test http_state.fallback_active

        CryptoXch.setmarketdataheartbeat!(tc.xc, "BTCUSDT", tc.xc.currentdt)
        ws_state = Trade._update_marketdata_freshness_policy!(tc; symbols=["BTCUSDT"])
        @test ws_state.source == :ws
        @test !ws_state.fallback_active

        @test haskey(tc.mc, :marketdata_source)
        @test haskey(tc.mc, :marketdata_ws_fallback_switches)
        @test Int(tc.mc[:marketdata_ws_fallback_switches]) >= 0
    finally
        EnvConfig.init(prevmode)
    end
end
