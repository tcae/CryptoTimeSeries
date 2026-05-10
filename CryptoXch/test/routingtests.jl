module CryptoXchRoutingTest
using Test
using DataFrames
using EnvConfig, CryptoXch

EnvConfig.init(test)

@testset "CryptoXch routing" begin
    xc = CryptoXch.XchCache()
    CryptoXch.setrole!(xc, CryptoXch.data_exchange, CryptoXch.EXCHANGE_BYBIT)
    CryptoXch.setrole!(xc, CryptoXch.trade_exchange_spot, CryptoXch.EXCHANGE_KRAKENSPOT, "krakenspot-tcae1")
    CryptoXch.setrole!(xc, CryptoXch.trade_exchange_futures, CryptoXch.EXCHANGE_KRAKENFUTURES, "krakenfutures-tcae2")

    @test CryptoXch._routeexchange(xc.routing, CryptoXch.data_exchange, xc.exchange) == CryptoXch.EXCHANGE_BYBIT
    @test CryptoXch._routeexchange(xc.routing, CryptoXch.trade_exchange_spot, xc.exchange) == CryptoXch.EXCHANGE_KRAKENSPOT
    @test CryptoXch._routeexchange(xc.routing, CryptoXch.trade_exchange_futures, xc.exchange) == CryptoXch.EXCHANGE_KRAKENFUTURES

    @test string(CryptoXch._routedModule(xc, CryptoXch.data_exchange)) == "Bybit"
    @test string(CryptoXch._routedModule(xc, CryptoXch.trade_exchange_spot)) == "KrakenSpot"
    @test string(CryptoXch._routedModule(xc, CryptoXch.trade_exchange_futures)) == "KrakenFutures"
    @test CryptoXch._routedbc(xc, CryptoXch.data_exchange) === nothing

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
    ), "", "", "")

    @test CryptoXch.KrakenSpot.symboltoken(spotbc, "BTC", "USDT") == "BTCUSDT"
    @test CryptoXch.KrakenFutures.symboltoken(futuresbc, "BTC", "USD") == "BTCUSD"
    @test CryptoXch.Bybit.symboltoken(bybitbc, "BTC", "USDT") == "BTCUSDT"

    @test CryptoXch.KrakenSpot.validsymbol(spotbc, "BTC", "USDT")
    @test CryptoXch.KrakenFutures.validsymbol(futuresbc, "BTC", "USD")
    @test CryptoXch.Bybit.validsymbol(bybitbc, "BTC", "USDT")
end

end