module SimuTest
using Dates, DataFrames
using Test
using Ohlcv, EnvConfig, CryptoXch, Bybit

@testset "CryptoXch tests" begin

    EnvConfig.init(training)
    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(20)
    xc = CryptoXch.XchCache(bases=[])
    ohlcv = CryptoXch.cryptodownload(xc, "BTC", "1m", startdt, enddt)
    CryptoXch.timerangecut!(ohlcv, startdt, enddt)
    Ohlcv.setix!(ohlcv, 1)
    println(ohlcv)

    mdf = CryptoXch.getUSDTmarket(xc)
    @test size(mdf, 1) > 0
    println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])

    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1]
    # at 2022-01-01T01:00:00 price falls 0.1% in the first 5 minutes and then rises 0.5% within 15 minutes
    println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = CryptoXch.balances(xc)
    @test size(bdf, 2) == 3
    println(bdf)

    pdf = CryptoXch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) == 4
    println(pdf)

    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice*1.2, basequantity=26.01/btcprice, maker=false) # limitprice out of allowed range
    @test isnothing(oid)
    println("createbuyorder: $(string(oid)) - error expected")
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice * 1.01, basequantity=26.01/btcprice, maker=false) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    println("createbuyorder: $(string(oid)) - reject expected")
    oo2 = CryptoXch.getorder(xc, oid)
    println("get rejected order: $(DataFrame([oo2]))")
    @test oo2.orderid == oid
    @test oo2.status == "Rejected"

    oo2 = CryptoXch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oo2)

    oidx = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice * 0.9, basequantity=8.01/btcprice, maker=false)
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice * 0.999, basequantity=6.01/btcprice, maker=false)
    oodf = CryptoXch.getopenorders(xc)
    println("getopenorders(nothing): $oodf")
    println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(xc, oid)
    # println("getorder: $oo2")
    @test oid == oo2.orderid

    currenttime = CryptoXch._ordercurrenttime(xc, oid)
    oidc = CryptoXch.changeorder(xc, oid; basequantity=4.02/btcprice)
    @test oidc == oid
    println("changeorder after basequatity change: $(DataFrame([CryptoXch.getorder(xc, oid)]))")

    CryptoXch.setcurrenttime!(xc, currenttime + Minute(4))
    oidc = CryptoXch.changeorder(xc, oidx; limitprice=btcprice * 0.998)
    @test oidc == oidx
    println("changeorder after limitprice change: $(DataFrame([CryptoXch.getorder(xc, oidx)]))")

    CryptoXch.setcurrenttime!(xc, currenttime + Minute(10))
    oo2 = CryptoXch.getorder(xc, oid)
    @test oid == oo2.orderid
    currenttime = CryptoXch._ordercurrenttime(xc, oid)
    println("getorder after fill: $(DataFrame([oo2]))")

    oodf = CryptoXch.getopenorders(xc, "BTC")
    println("getopenorders(BTC) after fill: $oodf")

    pdf = CryptoXch.balances(xc)
    @test size(pdf, 1) == 2
    btcqty = sum(pdf[pdf.coin .== "BTC", :free])
    println("portfolio with btcqty=$btcqty $pdf")

    btcprice = CryptoXch._ordercurrentprice(xc, oid)
    oid = CryptoXch.createsellorder(xc, "BTC", limitprice=btcprice * 1.005, basequantity=btcqty, maker=false)

    oodf = CryptoXch.getopenorders(xc, "BTC")
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) > 0)
    @test (size(oodf, 2) >= 12)
    println("getopenorders(\"BTC\"): $oodf")
    # println("getopenorders(nothing): $oodf")
    oodf = CryptoXch.getopenorders(xc, "XRP")
    println("getopenorders(\"XRP\"): $oodf")

    CryptoXch.setcurrenttime!(xc, currenttime + Minute(990))
    println("createsellorder: $(DataFrame([CryptoXch.getorder(xc, oid)]))")

    oodf = CryptoXch.getopenorders(xc)
    println("getopenorders(nothing) - expect 2 open orders: $oodf")

    oid2 = CryptoXch.cancelorder(xc, "BTC", oidx)
    println("cancelorder: $(string(oid2))")
    @test oidx == oid2
    oo2 = CryptoXch.getorder(xc, oidx)
    println("get cancelled order: $(DataFrame([oo2]))")
    oodf = CryptoXch.getopenorders(xc)
    println("getopenorders(nothing) - expect no open order: $oodf")


end


end  # module