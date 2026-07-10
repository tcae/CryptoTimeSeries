using Test
using DataFrames
using EnvConfig, Trade, Xch

@testset "Active buy cache" begin
    EnvConfig.init(test)

    cache = Trade.TradeCache(xc=Xch.XchCache())
    oo = DataFrame(
        orderid=["oid-btc-buy", "oid-eth-sell", "oid-btc-shortclose", "oid-ada-shortopen"],
        symbol=["BTCUSDT", "ETHUSDT", "BTCUSDT", "ADAUSDT"],
        side=["Buy", "Sell", "Buy", "Sell"],
        status=["New", "New", "New", "New"],
        isLeverage=[false, false, true, true],
        reduceonly=[false, false, true, false],
    )

    Trade._refreshactiveopenlongsymbols!(cache, oo)
    Trade._refreshactiveopenshortsymbols!(cache, oo)

    @test Trade._hasactiveopenlong(cache, "BTCUSDT")
    @test !Trade._hasactiveopenlong(cache, "ETHUSDT")
    @test !Trade._hasactiveopenlong(cache, "ADAUSDT")
    @test Trade._hasactiveopenshort(cache, "ADAUSDT")
    @test Trade._hasactiveopenshort(cache, "ETHUSDT") == false
    @test Trade._hasactiveopenshort(cache, "BTCUSDT") == false

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