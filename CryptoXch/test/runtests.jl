
module CryptoXchTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, CryptoXch, Bybit

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

    # EnvConfig.init(test)  # test production
    EnvConfig.init(production)  # test production
    CryptoXch.init()
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

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:48:01"), DateTime("2022-01-02T22:51:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 7

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:53:03"), DateTime("2022-01-02T22:55:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 13

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:40:03"), DateTime("2022-01-02T22:41:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 16

    Ohlcv.write(ohlcv)
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
    @test size(Ohlcv.dataframe(Ohlcv.read!(ohlcv)), 1) == 0  # will result in no data found message
    rm(EnvConfig.datafolderpath())

    @test CryptoXch.onlyconfiguredsymbols("BTCUSDT")
    @test !CryptoXch.onlyconfiguredsymbols("BTCBNB")
    @test !CryptoXch.onlyconfiguredsymbols("EURUSDT")


    # EnvConfig.init(test)
    EnvConfig.init(production)
    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1,1]
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = CryptoXch.balances()
    @test size(bdf, 2) == 3
    # println(bdf)

    pdf = CryptoXch.portfolio!(bdf, mdf)
    @test size(pdf, 2) == 4
    # println(pdf)

    oodf = CryptoXch.getopenorders(nothing)
    @test isa(oodf, AbstractDataFrame)
    # println("getopenorders(nothing): $oodf")

    oo2 = CryptoXch.getorder("invalid_or_unknown_id")
    @test isnothing(oo2)

    oid = CryptoXch.createbuyorder("btc", limitprice=btcprice*1.2, basequantity=26.01/btcprice) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    oid = CryptoXch.createbuyorder("btc", limitprice=btcprice * 1.01, basequantity=26.01/btcprice) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    # println("createbuyorder: $(string(oid)) - reject expected")
    oo2 = CryptoXch.getorder(oid)
    # println("getorder: $oo2")
    @test oo2.orderid == oid
    @test oo2.status == "Rejected"

    oid = CryptoXch.createbuyorder("btc", limitprice=btcprice * 0.9, basequantity=6.01/btcprice)
    # println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(oid)
    # println("getorder: $oo2")
    @test oid == oo2.orderid
    # println("getorder: $(CryptoXch.getorder(oid))")

    oidc = CryptoXch.changeorder(oid; basequantity=4.02/btcprice)
    @test oidc == oid
    # println("getorder: $(CryptoXch.getorder(oid))")

    oidc = CryptoXch.changeorder(oid; limitprice=btcprice * 0.8)
    @test oidc == oid
    # println("getorder: $(CryptoXch.getorder(oid))")


    oodf = CryptoXch.getopenorders(nothing)
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) > 0)
    @test (size(oodf, 2) == 12)
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


    println("Now testing CryptoXch simulation mode")

    EnvConfig.init(training)
    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(20)
    CryptoXch.init(bases=[])
    ohlcv = CryptoXch.cryptodownload("BTC", "1m", startdt, enddt)
    CryptoXch.timerangecut!(ohlcv, startdt, enddt)
    Ohlcv.setix!(ohlcv, 1)
    # println(ohlcv)

    mdf = CryptoXch.getUSDTmarket()
    @test size(mdf, 1) > 0
    # println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])

    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1]
    # at 2022-01-01T01:00:00 price falls 0.1% in the first 5 minutes and then rises 0.5% within 5.5 hours
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = CryptoXch.balances()
    @test size(bdf, 2) == 3
    # println(bdf)

    pdf = CryptoXch.portfolio!(bdf, mdf)
    @test size(pdf, 2) == 4
    # println(pdf)

    oid = CryptoXch.createbuyorder("BTC", limitprice=btcprice*1.2, basequantity=26.01/btcprice) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    oid = CryptoXch.createbuyorder("BTC", limitprice=btcprice * 1.01, basequantity=26.01/btcprice) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    # println("createbuyorder: $(string(oid)) - reject expected")
    oo2 = CryptoXch.getorder(oid)
    # println("get rejected order: $(DataFrame([oo2]))")
    @test oo2.orderid == oid
    @test oo2.status == "Rejected"

    oo2 = CryptoXch.getorder("invalid_or_unknown_id")
    @test isnothing(oo2)

    oidx = CryptoXch.createbuyorder("BTC", limitprice=btcprice * 0.9, basequantity=8.01/btcprice)
    oid = CryptoXch.createbuyorder("BTC", limitprice=btcprice * 0.999, basequantity=6.01/btcprice)
    oodf = CryptoXch.getopenorders()
    # println("getopenorders(nothing): $oodf")
    # println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(oid)
    # println("getorder: $oo2")
    @test oid == oo2.orderid

    currenttime = CryptoXch._ordercurrenttime(oid)
    oidc = CryptoXch.changeorder(oid; basequantity=4.02/btcprice)
    @test oidc == oid
    # println("changeorder after basequatity change: $(DataFrame([CryptoXch.getorder(oid)]))")

    CryptoXch.setcurrenttime!(currenttime + Minute(4))
    oidc = CryptoXch.changeorder(oidx; limitprice=btcprice * 0.998)
    @test oidc == oidx
    # println("changeorder after limitprice change: $(DataFrame([CryptoXch.getorder(oidx)]))")

    CryptoXch.setcurrenttime!(currenttime + Minute(10))
    oo2 = CryptoXch.getorder(oid)
    @test oid == oo2.orderid
    currenttime = CryptoXch._ordercurrenttime(oid)
    # println("getorder after fill: $(DataFrame([oo2]))")

    oodf = CryptoXch.getopenorders("BTC")
    # println("getopenorders(BTC) after fill: $oodf")

    pdf = CryptoXch.balances()
    @test size(pdf, 1) == 2
    btcqty = sum(pdf[pdf.coin .== "BTC", :free])
    # println("portfolio with btcqty=$btcqty $pdf")

    btcprice = CryptoXch._ordercurrentprice(oid)
    oid = CryptoXch.createsellorder("BTC", limitprice=btcprice * 1.005, basequantity=btcqty)

    oodf = CryptoXch.getopenorders("BTC")
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) > 0)
    @test (size(oodf, 2) >= 12)
    # println("getopenorders(\"BTC\"): $oodf")
    oodf = CryptoXch.getopenorders("XRP")
    # println("getopenorders(\"XRP\"): $oodf")

    CryptoXch.setcurrenttime!(currenttime + Minute(990))
    # println("createsellorder: $(DataFrame([CryptoXch.getorder(oid)]))")

    oodf = CryptoXch.getopenorders()
    # println("getopenorders(nothing) - expect 2 open orders: $oodf")

    oid2 = CryptoXch.cancelorder("BTC", oidx)
    # println("cancelorder: $(string(oid2))")
    @test oidx == oid2
    oo2 = CryptoXch.getorder(oidx)
    # println("get cancelled order: $(DataFrame([oo2]))")
    oodf = CryptoXch.getopenorders()
    # println("getopenorders(nothing) - expect no open order: $oodf")

end


end  # module