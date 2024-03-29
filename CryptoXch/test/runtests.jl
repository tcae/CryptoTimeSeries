
module CryptoXchTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, CryptoXch


EnvConfig.init(test)
# EnvConfig.init(production)

userdataChannel = Channel(10)
startdt = DateTime("2020-08-11T22:45:00")
enddt = DateTime("2020-09-11T22:49:00")

@testset "CryptoXch tests" begin

    # EnvConfig.init(test)  # test production
    EnvConfig.init(production)  # test production
    xc = CryptoXch.XchCache(true)
    # df = CryptoXch.klines2jdf(xc, missing)
    # @test nrow(df) == 0
    mdf = CryptoXch.getUSDTmarket(xc)
    @test size(mdf, 1) > 100
    # println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])
    @test nrow(mdf) > 10

    EnvConfig.init(production; newdatafolder=true) #! stay with newdatafolder because deleting the data is part of it
    # EnvConfig.init(EnvConfig.production)
    ohlcv = Ohlcv.defaultohlcv("btc")

    testcoins = CryptoXch.testbasecoin()
    for tc in testcoins
        ohlcv = CryptoXch.cryptodownload(xc, tc, "1m", DateTime("2022-01-02T22:38:03"), DateTime("2022-01-02T22:57:45"))
        # println(ohlcv)
        @test size(Ohlcv.dataframe(ohlcv), 1) == 20
        @test all([name in names(Ohlcv.dataframe(ohlcv)) for name in names(Ohlcv.defaultohlcvdataframe())])
    end

    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:45:03"), DateTime("2022-01-02T22:49:35"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5
    @test names(Ohlcv.dataframe(ohlcv)) == ["opentime", "open", "high", "low", "close", "basevolume", "pivot"]

    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:45:01"), DateTime("2022-01-02T22:49:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:48:01"), DateTime("2022-01-02T22:51:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 7

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:53:03"), DateTime("2022-01-02T22:55:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 3  # not using canned data if no overlap

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:40:03"), DateTime("2022-01-02T22:41:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 2  # not using canned data if no overlap

    Ohlcv.write(ohlcv)
    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:38:03"), DateTime("2022-01-02T22:57:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 20

    ohlcv1 = Ohlcv.copy(ohlcv)
    CryptoXch.timerangecut!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 4

    CryptoXch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:47:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 5

    CryptoXch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:49:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 7

    CryptoXch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:42:00"), DateTime("2022-01-02T22:49:45"))
    # does not add anything for DateTime("2022-01-02T22:42:03")
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 8

    CryptoXch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    CryptoXch.timerangecut!(ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 6

    Ohlcv.delete(ohlcv)
    sleep(1)
    rm(EnvConfig.datafolderpath(); force=true, recursive=true)

    @test CryptoXch.onlyconfiguredsymbols("BTCUSDT")
    @test !CryptoXch.onlyconfiguredsymbols("BTCBNB")
    @test !CryptoXch.onlyconfiguredsymbols("EURUSDT")


    # EnvConfig.init(test)
    EnvConfig.init(production)
    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1,1]
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = CryptoXch.balances(xc)
    @test size(bdf, 2) == 3
    # println(bdf)

    pdf = CryptoXch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) == 5
    # println(pdf)

    oodf = CryptoXch.getopenorders(xc, nothing)
    @test isa(oodf, AbstractDataFrame)
    # println("getopenorders(xc, nothing): $oodf")

    oo2 = CryptoXch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oo2)

    oid = CryptoXch.createbuyorder(xc, "btc", limitprice=btcprice*1.2, basequantity=26.01/btcprice, maker=false) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    oid = CryptoXch.createbuyorder(xc, "btc", limitprice=btcprice * 1.01, basequantity=26.01/btcprice, maker=false) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    # println("createbuyorder: $(string(oid)) - reject expected")
    oo2 = CryptoXch.getorder(xc, oid)
    # println("getorder: $oo2")
    @test oo2.orderid == oid
    # @test oo2.status == "Rejected"  # applicable for PostOnly as soon as taker fee > maker fee
    @test oo2.status == "Filled"  # due to GTC as long as taker fee == maker fee

    oid = CryptoXch.createbuyorder(xc, "btc", limitprice=btcprice * 0.9, basequantity=6.01/btcprice, maker=false)
    # println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(xc, oid)
    # println("getorder: $oo2")
    @test oid == oo2.orderid
    # println("getorder: $(CryptoXch.getorder(xc, oid))")

    oidc = CryptoXch.changeorder(xc, oid; basequantity=4.02/btcprice)
    @test oidc == oid
    # println("getorder: $(CryptoXch.getorder(xc, oid))")

    oidc = CryptoXch.changeorder(xc, oid; limitprice=btcprice * 0.8)
    @test oidc == oid
    # println("getorder: $(CryptoXch.getorder(xc, oid))")


    oodf = CryptoXch.getopenorders(xc, nothing)
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) > 0)
    @test (size(oodf, 2) == 13)
    # println("getopenorders(nothing): $oodf")
    oodf = CryptoXch.getopenorders(xc, "xrp")
    # println("getopenorders(\"xrp\"): $oodf")
    oid2 = CryptoXch.cancelorder(xc, "btc", oid)
    # println("cancelorder: $(string(oid2))")
    @test oid == oid2
    oo2 = CryptoXch.getorder(xc, oid)
    # println("getorder: $oo2")
    oodf = CryptoXch.getopenorders(xc)
    # println("getopenorders(nothing): $oodf")

    # println("test IP with CLI: wget -qO- http://ipecho.net/plain | xargs echo")

    println("Now testing CryptoXch simulation mode")

    EnvConfig.init(training)
    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(20)
    xc = CryptoXch.XchCache([], startdt, enddt, 1000)
    ohlcv = CryptoXch.cryptodownload(xc, "BTC", "1m", startdt, enddt)
    CryptoXch.timerangecut!(ohlcv, startdt, enddt)
    Ohlcv.setix!(ohlcv, 1)
    # println(ohlcv)

    mdf = CryptoXch.getUSDTmarket(xc)
    @test size(mdf, 1) > 0
    # println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])

    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1]
    # at 2022-01-01T01:00:00 price falls 0.1% in the first 5 minutes and then rises 0.5% within 5.5 hours
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01)) BTC.close=$(xc.bases["BTC"].df[xc.bases["BTC"].ix, :close])")

    bdf = CryptoXch.balances(xc)
    @test size(bdf, 2) == 3
    # println(bdf)

    pdf = CryptoXch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) == 5
    # println(pdf)

    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice*1.2, basequantity=26.01/btcprice, maker=false) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    # println("limitprice=$(btcprice * 1.01)")

    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice * 1.01, basequantity=26.01/btcprice, maker=false) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    # println("createbuyorder: $(string(oid)) - reject expected")  # not applicable anymore because timeinforce is by default changed from PostOnly to GTC
    oo2 = CryptoXch.getorder(xc, oid)
    # println("get (not rejected) order: $(DataFrame([oo2]))")
    @test oo2.orderid == oid
    # @test oo2.status == "Rejected"  - not Rejected because default was changed from PostOnly to GTC
    @test oo2.status == "Filled"

    oo2 = CryptoXch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oo2)

    oidx = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice * 0.9, basequantity=8.01/btcprice, maker=false)
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice * 0.999, basequantity=6.01/btcprice, maker=false)
    oodf = CryptoXch.getopenorders(xc)
    # println("getopenorders(nothing): $oodf")
    # println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(xc, oid)
    # println("getorder: $oo2")
    @test oid == oo2.orderid

    currenttime = CryptoXch._ordercurrenttime(xc, oid)
    oidc = CryptoXch.changeorder(xc, oid; basequantity=4.02/btcprice)
    @test oidc == oid
    # println("changeorder after basequatity change: $(DataFrame([CryptoXch.getorder(oid)]))")

    CryptoXch.setcurrenttime!(xc, currenttime + Minute(4))
    oidc = CryptoXch.changeorder(xc, oidx; limitprice=btcprice * 0.998)
    @test oidc == oidx
    # println("changeorder after limitprice change: $(DataFrame([CryptoXch.getorder(oidx)]))")

    CryptoXch.setcurrenttime!(xc, currenttime + Minute(10))
    oo2 = CryptoXch.getorder(xc, oid)
    @test oid == oo2.orderid
    currenttime = CryptoXch._ordercurrenttime(xc, oid)
    # println("getorder after fill: $(DataFrame([oo2]))")

    oodf = CryptoXch.getopenorders(xc, "BTC")
    # println("getopenorders(BTC) after fill: $oodf")

    pdf = CryptoXch.balances(xc)
    @test size(pdf, 1) == 2
    btcqty = sum(pdf[pdf.coin .== "BTC", :free])
    # println("portfolio with btcqty=$btcqty $pdf")

    btcprice = CryptoXch._ordercurrentprice(xc, oid)
    oid = CryptoXch.createsellorder(xc, "BTC", limitprice=btcprice * 1.005, basequantity=btcqty, maker=false)

    oodf = CryptoXch.getopenorders(xc, "BTC")
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) > 0)
    @test (size(oodf, 2) >= 12)
    # println("getopenorders(\"BTC\"): $oodf")
    oodf = CryptoXch.getopenorders(xc, "XRP")
    # println("getopenorders(\"XRP\"): $oodf")

    CryptoXch.setcurrenttime!(xc, currenttime + Minute(990))

    oodf = CryptoXch.getopenorders(xc)
    # println("getopenorders(nothing) - expect 2 open orders: $oodf")

    oid2 = CryptoXch.cancelorder(xc, "BTC", oidx)
    # println("cancelorder: $(string(oid2))")
    @test oidx == oid2
    oo2 = CryptoXch.getorder(xc, oidx)
    # println("get cancelled order: $(DataFrame([oo2]))")
    oodf = CryptoXch.getopenorders(xc)
    # println("getopenorders(nothing) - expect no open order: $oodf")

end


end  # module