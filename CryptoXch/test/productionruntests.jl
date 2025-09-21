
module CryptoXchProductionTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, CryptoXch, TestOhlcv



# EnvConfig.init(test)  # test CryptoXch Testnet production
EnvConfig.init(production)  # test CryptoXch real production
println("CryptoXchTest runtests")

@testset "CryptoXch production tests" begin
    CryptoXch.verbosity =0

    xc = CryptoXch.XchCache()
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

    testcoins = TestOhlcv.testbasecoin()
    for tc in testcoins
        @test CryptoXch.validbase(xc, tc)
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
    Ohlcv.timerangecut!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
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
    Ohlcv.timerangecut!(ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 6

    Ohlcv.delete(ohlcv)
    sleep(1)
    rm(EnvConfig.datafolderpath(); force=true, recursive=true)

    @test CryptoXch.onlyconfiguredsymbols("BTCUSDT")
    @test !CryptoXch.onlyconfiguredsymbols("BTCBNB")
    @test !CryptoXch.onlyconfiguredsymbols("EURUSDT")


    # EnvConfig.init(test)
    EnvConfig.init(production)
    mdf = CryptoXch.getUSDTmarket(xc)
    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1,1]
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = CryptoXch.balances(xc)
    @test size(bdf, 2) >= 3
    # println(bdf)

    pdf = CryptoXch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) >= 5
    # println(pdf)

    oodf = CryptoXch.getopenorders(xc, nothing)
    @test isa(oodf, AbstractDataFrame)
    # println("getopenorders(xc, nothing): $oodf")

    oo2 = CryptoXch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oo2)

    oid = CryptoXch.createbuyorder(xc, "btc", limitprice=btcprice*1.2, basequantity=26.01/btcprice, maker=false) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    oid = CryptoXch.createbuyorder(xc, "btc", limitprice=btcprice * 1.001, basequantity=26.01/btcprice, maker=false) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    oo2 = CryptoXch.getorder(xc, oid)
    # println("getorder: $oo2")
    @test oo2.orderid == oid
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
    @test (size(oodf, 2) >= 13)
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

end


end  # module