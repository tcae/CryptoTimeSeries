
module CryptoXchTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, CryptoXch

function balances_test()
    result = CryptoXch.balances()
    display(result)
    display(EnvConfig.bases)
    display(EnvConfig.trainingbases)
    display(EnvConfig.datapath)
end

# EnvConfig.init(test)
EnvConfig.init(production)
# balances_test()

userdataChannel = Channel(10)
startdt = DateTime("2020-08-11T22:45:00")
enddt = DateTime("2020-09-11T22:49:00")
# res = Binance.getKlines("BTCUSDT"; startDateTime=startdt, endDateTime=enddt, interval="1m")
# display(res)
# display(last(res[:body], 3))
# display(first(res[:body], 3))
# display(res[:body][1:3, :])
# display(res[:body][end-3:end, :])

# Binance.wsKlineStreams(cb, ["BTCUSDT", "XRPUSDT"])

function gethistoryohlcv_test()
    startdt = DateTime("2020-08-11T22:45:00")
    enddt = DateTime("2020-08-12T22:49:00")
    df = CryptoXch.gethistoryohlcv("btc", startdt, enddt)
    # display(first(df, 2))
    # display(last(df, 2))
    # println("saved btc from $(df[1, :opentime]) until $(df[end, :opentime])")
    return df
end

function addstartgapbtcdownload()
    startdt = DateTime("2022-01-02T22:40:03")
    enddt = DateTime("2022-01-02T22:41:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function appendgapbtcdownload()
    startdt = DateTime("2022-01-02T22:53:03")
    enddt = DateTime("2022-01-02T22:55:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function appendoverlapbtcdownload()
    startdt = DateTime("2022-01-02T22:47:03")
    enddt = DateTime("2022-01-02T22:51:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function addfulloverlapbtcdownload()
    startdt = DateTime("2022-01-02T22:44:01")
    enddt = DateTime("2022-01-02T22:50:45")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function addstartoverlapbtcdownload()
    startdt = DateTime("2022-01-02T22:43:03")
    enddt = DateTime("2022-01-02T22:47:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function initialbtcdownload()
    startdt = DateTime("2022-01-02T22:45:03")
    enddt = DateTime("2022-01-02T22:49:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function orderstring2values!_test()
    ood = [
        Dict("symbol" => "LTCUSDT", "orderId" => 1, "isWorking" => true, "price" => "0.1", "time" => 1499827319559,
        "fills" => [
            Dict("price" => "4000.00000000", "qty" => "1.00000000","commission" => "4.00000000", "commissionAsset" => "USDT", "tradeId" => 56),
            Dict("price" => "4000.10000000", "qty" => "1.10000000","commission" => "4.10000000", "commissionAsset" => "USDT", "tradeId" => 57)
            ]
        )
    ]
    # println("before value conversion: $ood")
    ood = CryptoXch.orderstring2values!(ood)
    # println("after value conversion: $ood")
    return ood
end

@testset "CryptoXch tests" begin

    df = CryptoXch.klines2jdf(missing)
    @test nrow(df) == 0
    mdf = CryptoXch.getUSDTmarket()
    # println(mdf)
    @test names(mdf) == ["base", "qte", "quotevolume24h", "pricechangepercent", "lastprice"]
    @test nrow(mdf) > 10

    EnvConfig.init(EnvConfig.test)
    ohlcv = Ohlcv.defaultohlcv("btc")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:45:03"), DateTime("2022-01-02T22:49:35"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5
    @test names(Ohlcv.dataframe(ohlcv)) == ["opentime", "open", "high", "low", "close", "basevolume", "pivot"]

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:45:01"), DateTime("2022-01-02T22:49:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:48:01"), DateTime("2022-01-02T22:51:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 7

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:53:03"), DateTime("2022-01-02T22:55:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 13

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:40:03"), DateTime("2022-01-02T22:41:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 16

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:38:03"), DateTime("2022-01-02T22:57:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 20

    ohlcv1 = Ohlcv.copy(ohlcv)
    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 4

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:47:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 5

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:49:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 7

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:42:00"), DateTime("2022-01-02T22:49:45"))
    # does not add anything for DateTime("2022-01-02T22:42:03")
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 8

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 6

    Ohlcv.delete(ohlcv)
    @test size(Ohlcv.dataframe(Ohlcv.read!(ohlcv)), 1) == 0

    @test CryptoXch.onlyconfiguredsymbols("BTCUSDT")
    @test !CryptoXch.onlyconfiguredsymbols("BTCBNB")
    @test !CryptoXch.onlyconfiguredsymbols("EURUSDT")

    ood = orderstring2values!_test()
    @test ood[1]["price"] isa AbstractFloat
    @test ood[1]["time"] isa DateTime
    @test ood[1]["fills"][1]["qty"] isa AbstractFloat

    oo1 = CryptoXch.createorder("btc", "BUY", 19001.0, 20.0)
    println("createorder: $oo1")
    oo2 = CryptoXch.getorder("btc", oo1["orderId"])
    println("getorder: $oo2")
    # @test oo1["orderId"] == oo2["orderId"]
    ooarray = CryptoXch.getopenorders(nothing)
    println("getopenorders(nothing): $ooarray")
    ooarray = CryptoXch.getopenorders("btc")
    println("getopenorders(\"btc\"): $ooarray")
    oo2 = CryptoXch.cancelorder("btc", oo1["orderId"])
    println("cancelorder: $oo2")
    @test oo1["orderId"] == oo2["orderId"]

end


end  # module