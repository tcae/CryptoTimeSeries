using Test
using DataFrames
using EnvConfig, Trade, CryptoXch

@testset "Active buy cache" begin
    EnvConfig.init(test)

    cache = Trade.TradeCache(xc=CryptoXch.XchCache())
    oo = DataFrame(
        orderid=["oid-btc-buy", "oid-eth-sell", "oid-btc-leverage-buy"],
        symbol=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        side=["Buy", "Sell", "Buy"],
        status=["New", "New", "New"],
        isLeverage=[false, false, true],
    )

    Trade._refreshactiveopenbuysymbols!(cache, oo)

    @test Trade._hasactiveopenbuy(cache, "BTCUSDT")
    @test !Trade._hasactiveopenbuy(cache, "ETHUSDT")
    @test Trade._hasactiveopensell(cache, "ETHUSDT") == false

    Trade._rememberactiveopenbuy!(cache, "ethusdt")
    @test Trade._hasactiveopenbuy(cache, "ETHUSDT")

    Trade._rememberactiveopensell!(cache, "ethusdt")
    @test Trade._hasactiveopensell(cache, "ETHUSDT")

    Trade._refreshactiveopenbuysymbols!(cache, DataFrame(orderid=String[], symbol=String[], side=String[], status=String[]))
    Trade._refreshactiveopensellsymbols!(cache, DataFrame(orderid=String[], symbol=String[], side=String[], status=String[]))
    @test !Trade._hasactiveopenbuy(cache, "BTCUSDT")
    @test !Trade._hasactiveopenbuy(cache, "ETHUSDT")
    @test !Trade._hasactiveopensell(cache, "ETHUSDT")
end