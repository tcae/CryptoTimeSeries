using Test
using DataFrames
using EnvConfig, Trade, Xch

@testset "Active buy cache" begin
    EnvConfig.init(test)

    cache = Trade.TradeCache(xc=Xch.XchCache())
    oo = DataFrame(
        orderid=["oid-btc-buy", "oid-eth-sell", "oid-btc-leverage-buy"],
        symbol=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        side=["Buy", "Sell", "Buy"],
        status=["New", "New", "New"],
        isLeverage=[false, false, true],
    )

    Trade._refreshactiveopenlongsymbols!(cache, oo)

    @test Trade._hasactiveopenlong(cache, "BTCUSDT")
    @test !Trade._hasactiveopenlong(cache, "ETHUSDT")
    @test Trade._hasactiveopenshort(cache, "ETHUSDT") == false

    Trade._rememberactiveopenlong!(cache, "ethusdt")
    @test Trade._hasactiveopenlong(cache, "ETHUSDT")

    Trade._rememberactiveopenshort!(cache, "ethusdt")
    @test Trade._hasactiveopenshort(cache, "ETHUSDT")

    Trade._refreshactiveopenlongsymbols!(cache, DataFrame(orderid=String[], symbol=String[], side=String[], status=String[]))
    Trade._refreshactiveopenshortsymbols!(cache, DataFrame(orderid=String[], symbol=String[], side=String[], status=String[]))
    @test !Trade._hasactiveopenlong(cache, "BTCUSDT")
    @test !Trade._hasactiveopenlong(cache, "ETHUSDT")
    @test !Trade._hasactiveopenshort(cache, "ETHUSDT")
end