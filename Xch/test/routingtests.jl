module XchRoutingTest
using Test
using Dates
using DataFrames
using EnvConfig, Xch

EnvConfig.init(test)

@testset "Xch routing" begin
    xc = Xch.XchCache()
    Xch.setrole!(xc, Xch.data_exchange, Xch.EXCHANGE_BYBIT)
    Xch.setrole!(xc, Xch.trade_exchange_spot, Xch.EXCHANGE_KRAKENSPOT)
    Xch.setrole!(xc, Xch.trade_exchange_futures, Xch.EXCHANGE_KRAKENFUTURES)

    @test Xch._routeexchange(xc.routing, Xch.data_exchange, xc.exchange) == Xch.EXCHANGE_BYBIT
    @test Xch._routeexchange(xc.routing, Xch.trade_exchange_spot, xc.exchange) == Xch.EXCHANGE_KRAKENSPOT
    @test Xch._routeexchange(xc.routing, Xch.trade_exchange_futures, xc.exchange) == Xch.EXCHANGE_KRAKENFUTURES
    @test Xch._normalizeexchange("bybitsim") == Xch.EXCHANGE_BYBITSIM
    @test_throws ArgumentError Xch._normalizeexchange("testxch")

    @test string(Xch._routedModule(xc, Xch.data_exchange)) == "Bybit"
    @test string(Xch._routedModule(xc, Xch.trade_exchange_spot)) == "KrakenSpot"
    @test string(Xch._routedModule(xc, Xch.trade_exchange_futures)) == "KrakenFutures"
    @test Xch._routedbc(xc, Xch.data_exchange) isa Xch.Bybit.BybitCache

    xc_live = Xch.XchCache()
    xc_live.mc[:simmode] = Xch.nosimulation
    xc_live.bc = "primary-adapter"
    @test Xch._routedbc(xc_live, Xch.data_exchange) == "primary-adapter"

    spotbc = Xch.KrakenSpot.KrakenSpotCache(DataFrame(
        symbol=["BTCUSDT"],
        basecoin=["BTC"],
        quotecoin=["USDT"],
        krakenpairname=["XBT/USDT"],
        wsname=["BTC/USDT"],
        status=["online"],
    ), "", "", "")
    futuresbc = Xch.KrakenFutures.KrakenFuturesCache(DataFrame(
        symbol=["BTCUSD"],
        basecoin=["BTC"],
        quotecoin=["USD"],
        krakenpairname=["PI_XBTUSD"],
        wsname=["PI_XBTUSD"],
        status=["trading"],
    ), "", "", "")
    bybitbc = Xch.Bybit.BybitCache(DataFrame(
        symbol=["BTCUSDT"],
        basecoin=["BTC"],
        quotecoin=["USDT"],
        status=["Trading"],
        innovation=[0],
    ), "", "", "", nothing, nothing, nothing, nothing)

    @test Xch.KrakenSpot.symboltoken(spotbc, "BTC", "USDT") == "BTCUSDT"
    @test Xch.KrakenFutures.symboltoken(futuresbc, "BTC", "USD") == "BTCUSD"
    @test Xch.Bybit.symboltoken(bybitbc, "BTC", "USDT") == "BTCUSDT"

    @test Xch.KrakenSpot.validsymbol(spotbc, "BTC", "USDT")
    @test Xch.KrakenFutures.validsymbol(futuresbc, "BTC", "USD")
    @test Xch.Bybit.validsymbol(bybitbc, "BTC", "USDT")

    bybitsim_cache = Xch._exchangecache(Xch.EXCHANGE_BYBITSIM, Xch.bybitsim)
    @test bybitsim_cache isa Xch.Bybit.BybitCache
    @test !isnothing(bybitsim_cache.assets)
    @test !isnothing(bybitsim_cache.orders)
    @test !isnothing(bybitsim_cache.closedorders)
    @test !isnothing(Xch.Bybit.symbolinfo(bybitsim_cache, "SINEUSDT"))
    @test Xch.Bybit.validsymbol(bybitsim_cache, "SINEUSDT")
    @test !isnothing(Xch.Bybit.symbolinfo(bybitsim_cache, "DOUBLESINEUSDT"))
    @test Xch.Bybit.validsymbol(bybitsim_cache, "DOUBLESINEUSDT")

    @test_throws ArgumentError Xch._exchangecache(Xch.EXCHANGE_KRAKENSPOT, Xch.bybitsim)
    @test_throws ArgumentError Xch._exchangecache(Xch.EXCHANGE_KRAKENFUTURES, Xch.bybitsim)

    @test Xch.tradingpairkey("btc", "usdt") == "BTCUSDT"
    @test !Xch.hastrades(xc, "BTCUSDT")

    btcusdt = Xch.trades(xc, "btc", "usdt")
    @test btcusdt isa DataFrame
    @test nrow(btcusdt) == 0
    @test "opentime" in names(btcusdt)
    @test "pair" in names(btcusdt)
    @test "lastopentrade" in names(btcusdt)
    @test eltype(btcusdt[!, :opentime]) == DateTime
    @test Xch.hastrades(xc, "BTCUSDT")

    seeded = DataFrame(opentime=[DateTime("2025-01-01T00:00:00")], action=["open"], qty=[1.0f0])
    Xch.settrades!(xc, "eth", "usdt", seeded)
    @test Xch.tradingpairs(xc) == ["BTCUSDT", "ETHUSDT"]
    @test nrow(Xch.trades(xc, "ETHUSDT")) == 1
    @test "opentime" in names(Xch.trades(xc, "ETHUSDT"))
    @test "action" in names(Xch.trades(xc, "ETHUSDT"))
    @test "qty" in names(Xch.trades(xc, "ETHUSDT"))
    @test "pair" in names(Xch.trades(xc, "ETHUSDT"))
    @test "lastopentrade" in names(Xch.trades(xc, "ETHUSDT"))
    @test Xch.trades(xc, "ETHUSDT")[1, :pair] == "ETHUSDT"
end

end