
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

EnvConfig.init(test)
# EnvConfig.init(production)
# balances_test()

userdataChannel = Channel(10)
startdt = DateTime("2020-08-11T22:45:00")
enddt = DateTime("2020-09-11T22:49:00")
# res = Bybit.getklines("BTCUSDT"; startDateTime=startdt, endDateTime=enddt, interval="1m")
# display(res)
# display(last(res[:body], 3))
# display(first(res[:body], 3))
# display(res[:body][1:3, :])
# display(res[:body][end-3:end, :])

# Binance.wsKlineStreams(cb, ["BTCUSDT", "XRPUSDT"])


# function orderstring2values!_test()
#     ood = [
#         Dict("symbol" => "LTCUSDT", "orderId" => 1, "isWorking" => true, "price" => "0.1", "time" => 1499827319559,
#         "fills" => [
#             Dict("price" => "4000.00000000", "qty" => "1.00000000","commission" => "4.00000000", "commissionAsset" => "USDT", "tradeId" => 56),
#             Dict("price" => "4000.10000000", "qty" => "1.10000000","commission" => "4.10000000", "commissionAsset" => "USDT", "tradeId" => 57)
#             ]
#         )
#     ]
#     # println("before value conversion: $ood")
#     ood = CryptoXch.orderstring2values!(ood)
#     # println("after value conversion: $ood")
#     return ood
# end

@testset "CryptoXch tests" begin

    EnvConfig.init(production)
    # df = CryptoXch.klines2jdf(missing)
    # @test nrow(df) == 0
    mdf = CryptoXch.getUSDTmarket()
    @test size(mdf, 1) > 100
    # println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])
    @test nrow(mdf) > 10

    EnvConfig.init(production; newdatafolder=true) #! stay with newdatafolder because deleting the data is part of it
    # EnvConfig.init(EnvConfig.production)
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
    CryptoXch.timerangecut!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
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
    CryptoXch.timerangecut!(ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 6

    Ohlcv.delete(ohlcv)
    @test size(Ohlcv.dataframe(Ohlcv.read!(ohlcv)), 1) == 0
    rm(EnvConfig.datafolderpath())

    @test CryptoXch.onlyconfiguredsymbols("BTCUSDT")
    @test !CryptoXch.onlyconfiguredsymbols("BTCBNB")
    @test !CryptoXch.onlyconfiguredsymbols("EURUSDT")


    # EnvConfig.init(EnvConfig.test)
    EnvConfig.init(EnvConfig.production)
    btcprice = mdf[mdf.basecoin .== "btc", :lastprice][1,1]
    println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = CryptoXch.balances()
    @test size(bdf, 2) == 3
    # println(bdf)

    oodf = CryptoXch.getopenorders(nothing)
    println("getopenorders(nothing): $oodf")

    oo2 = CryptoXch.getorder("invalid_or_unknown_id")
    @test isnothing(oo2)

    oid = CryptoXch.createbuyorder("btc", limitprice=91001.03, usdtquantity=26.01) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    oid = CryptoXch.createbuyorder("btc", limitprice=btcprice * 1.01, usdtquantity=26.01) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    # println("createbuyorder: $(string(oid)) - reject expected")
    oo2 = CryptoXch.getorder(oid)
    # println("getorder: $oo2")
    @test oo2.orderid == oid
    @test oo2.status == "Rejected"

    oid = CryptoXch.createbuyorder("btc", limitprice=19001.0003, usdtquantity=26.01)
    # println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(oid)
    # println("getorder: $oo2")
    @test oid == oo2.orderid

    oodf = CryptoXch.getopenorders(nothing)
    # println("getopenorders(nothing): $oodf")
    oodf = CryptoXch.getopenorders("xrp")
    # println("getopenorders(\"xrp\"): $oodf")
    oid2 = CryptoXch.cancelorder("btc", oid)
    # println("cancelorder: $(string(oid2))")
    @test oid == oid2
    oo2 = CryptoXch.getorder(oid)
    # println("getorder: $oo2")
    oodf = CryptoXch.getopenorders()
    # println("getopenorders(nothing): $oodf")

    # println("test IP with CLI: wget -qO- http://ipecho.net/plain | xargs echo")
end


end  # module