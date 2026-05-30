module CryptoXchRoutingTest
using Test
using DataFrames
using EnvConfig, CryptoXch

EnvConfig.init(test)

@testset "CryptoXch routing" begin
    xc = CryptoXch.XchCache()
    CryptoXch.setrole!(xc, CryptoXch.data_exchange, CryptoXch.EXCHANGE_BYBIT)
    CryptoXch.setrole!(xc, CryptoXch.trade_exchange_spot, CryptoXch.EXCHANGE_KRAKENSPOT)
    CryptoXch.setrole!(xc, CryptoXch.trade_exchange_futures, CryptoXch.EXCHANGE_KRAKENFUTURES)

    @test CryptoXch._routeexchange(xc.routing, CryptoXch.data_exchange, xc.exchange) == CryptoXch.EXCHANGE_BYBIT
    @test CryptoXch._routeexchange(xc.routing, CryptoXch.trade_exchange_spot, xc.exchange) == CryptoXch.EXCHANGE_KRAKENSPOT
    @test CryptoXch._routeexchange(xc.routing, CryptoXch.trade_exchange_futures, xc.exchange) == CryptoXch.EXCHANGE_KRAKENFUTURES
    @test CryptoXch._normalizeexchange("bybitsim") == CryptoXch.EXCHANGE_BYBITSIM
    @test_throws ArgumentError CryptoXch._normalizeexchange("testxch")

    @test string(CryptoXch._routedModule(xc, CryptoXch.data_exchange)) == "Bybit"
    @test string(CryptoXch._routedModule(xc, CryptoXch.trade_exchange_spot)) == "KrakenSpot"
    @test string(CryptoXch._routedModule(xc, CryptoXch.trade_exchange_futures)) == "KrakenFutures"
    @test CryptoXch._routedbc(xc, CryptoXch.data_exchange) isa CryptoXch.Bybit.BybitCache

    xc_live = CryptoXch.XchCache()
    xc_live.mc[:simmode] = CryptoXch.nosimulation
    xc_live.bc = "primary-adapter"
    @test CryptoXch._routedbc(xc_live, CryptoXch.data_exchange) == "primary-adapter"

    spotbc = CryptoXch.KrakenSpot.KrakenSpotCache(DataFrame(
        symbol=["BTCUSDT"],
        basecoin=["BTC"],
        quotecoin=["USDT"],
        krakenpairname=["XBT/USDT"],
        wsname=["BTC/USDT"],
        status=["online"],
    ), "", "", "")
    futuresbc = CryptoXch.KrakenFutures.KrakenFuturesCache(DataFrame(
        symbol=["BTCUSD"],
        basecoin=["BTC"],
        quotecoin=["USD"],
        krakenpairname=["PI_XBTUSD"],
        wsname=["PI_XBTUSD"],
        status=["trading"],
    ), "", "", "")
    bybitbc = CryptoXch.Bybit.BybitCache(DataFrame(
        symbol=["BTCUSDT"],
        basecoin=["BTC"],
        quotecoin=["USDT"],
        status=["Trading"],
        innovation=[0],
    ), "", "", "", nothing, nothing, nothing, nothing)

    @test CryptoXch.KrakenSpot.symboltoken(spotbc, "BTC", "USDT") == "BTCUSDT"
    @test CryptoXch.KrakenFutures.symboltoken(futuresbc, "BTC", "USD") == "BTCUSD"
    @test CryptoXch.Bybit.symboltoken(bybitbc, "BTC", "USDT") == "BTCUSDT"

    @test CryptoXch.KrakenSpot.validsymbol(spotbc, "BTC", "USDT")
    @test CryptoXch.KrakenFutures.validsymbol(futuresbc, "BTC", "USD")
    @test CryptoXch.Bybit.validsymbol(bybitbc, "BTC", "USDT")

    bybitsim_cache = CryptoXch._exchangecache(CryptoXch.EXCHANGE_BYBITSIM, CryptoXch.bybitsim)
    @test bybitsim_cache isa CryptoXch.Bybit.BybitCache
    @test !isnothing(bybitsim_cache.assets)
    @test !isnothing(bybitsim_cache.orders)
    @test !isnothing(bybitsim_cache.closedorders)
    @test !isnothing(CryptoXch.Bybit.symbolinfo(bybitsim_cache, "SINEUSDT"))
    @test CryptoXch.Bybit.validsymbol(bybitsim_cache, "SINEUSDT")
    @test !isnothing(CryptoXch.Bybit.symbolinfo(bybitsim_cache, "DOUBLESINEUSDT"))
    @test CryptoXch.Bybit.validsymbol(bybitsim_cache, "DOUBLESINEUSDT")

    @test_throws ArgumentError CryptoXch._exchangecache(CryptoXch.EXCHANGE_KRAKENSPOT, CryptoXch.bybitsim)
    @test_throws ArgumentError CryptoXch._exchangecache(CryptoXch.EXCHANGE_KRAKENFUTURES, CryptoXch.bybitsim)
end

end