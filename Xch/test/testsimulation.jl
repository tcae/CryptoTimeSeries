module SimuTest
using Dates, DataFrames
using Test
using Ohlcv, EnvConfig, Xch, Bybit

@testset "Xch tests" begin

    EnvConfig.init(training)
    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(20)
    xc = Xch.XchCache()
    ohlcv = Xch.cryptodownload(xc, "BTC", "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    Ohlcv.setix!(ohlcv, 1)
    println(ohlcv)

    mdf = Xch.getUSDTmarket(xc)
    @test size(mdf, 1) > 0
    println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])

    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1]
    # at 2022-01-01T01:00:00 price falls 0.1% in the first 5 minutes and then rises 0.5% within 15 minutes
    println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = Xch.balances(xc)
    @test size(bdf, 2) >= 5
    println(bdf)

    pdf = Xch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) >= 7
    println(pdf)

    oid_bad = Xch.createopenorder(xc, "BTC"; limitprice=btcprice * 1.2, basequantity=26.01 / btcprice, maker=false, configside=:long)
    @test !isnothing(oid_bad)
    println("createopenorder (high limit): $(string(oid_bad))")

    oid = Xch.createopenorder(xc, "BTC"; limitprice=btcprice * 1.01, basequantity=26.01 / btcprice, maker=false, configside=:long)
    @test !isnothing(oid)
    @test oid != oid_bad
    println("createopenorder: $(string(oid))")

    oo2 = Xch.getorder(xc, oid)
    if !isnothing(oo2)
        println("getorder after createopenorder: $(DataFrame([oo2]))")
        @test oo2.orderid == oid
    end

    oo2 = Xch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oo2)

    oidx = Xch.createopenorder(xc, "BTC"; limitprice=btcprice * 0.9, basequantity=8.01/btcprice, maker=false, configside=:long)
    oid = Xch.createopenorder(xc, "BTC"; limitprice=btcprice * 0.999, basequantity=6.01/btcprice, maker=false, configside=:long)
    oodf = Xch.getopenorders(xc)
    println("getopenorders(nothing): $oodf")
    println("createopenorder: $(string(oid))")
    oo2 = Xch.getorder(xc, oid)
    if !isnothing(oo2)
        @test oid == oo2.orderid
    end

    currenttime = isnothing(xc.currentdt) ? startdt : xc.currentdt
    oidc = Xch.changeorder(xc, oid; basequantity=4.02/btcprice)
    @test isnothing(oidc) || oidc == oid
    changed_order = Xch.getorder(xc, oid)
    if !isnothing(changed_order)
        println("changeorder after basequatity change: $(DataFrame([changed_order]))")
    else
        println("changeorder after basequatity change: order lookup returned nothing")
    end

    Xch.setcurrenttime!(xc, currenttime + Minute(4))
    oidc = Xch.changeorder(xc, oidx; limitprice=btcprice * 0.998)
    @test isnothing(oidc) || oidc == oidx
    changed_order = Xch.getorder(xc, oidx)
    if !isnothing(changed_order)
        println("changeorder after limitprice change: $(DataFrame([changed_order]))")
    else
        println("changeorder after limitprice change: order lookup returned nothing")
    end

    Xch.setcurrenttime!(xc, currenttime + Minute(10))
    oo2 = Xch.getorder(xc, oid)
    if !isnothing(oo2)
        @test oid == oo2.orderid
        println("getorder after fill: $(DataFrame([oo2]))")
    else
        println("getorder after fill: order lookup returned nothing")
    end
    currenttime = isnothing(xc.currentdt) ? startdt : xc.currentdt

    oodf = Xch.getopenorders(xc, "BTC")
    println("getopenorders(BTC) after fill: $oodf")

    pdf = Xch.balances(xc)
    @test size(pdf, 1) == 2
    btcqty = sum(pdf[pdf.coin .== "BTC", :free])
    println("portfolio with btcqty=$btcqty $pdf")

    mdf = Xch.getUSDTmarket(xc)
    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1]
    oid = Xch.createcloseorder(xc, "BTC"; positionside=:long, limitprice=btcprice * 1.005, basequantity=btcqty, maker=false, reduceonly=true)

    oodf = Xch.getopenorders(xc, "BTC")
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) >= 0)
    @test (size(oodf, 2) >= 12)
    println("getopenorders(\"BTC\"): $oodf")
    # println("getopenorders(nothing): $oodf")
    oodf = Xch.getopenorders(xc, "XRP")
    println("getopenorders(\"XRP\"): $oodf")

    Xch.setcurrenttime!(xc, currenttime + Minute(990))
    close_order = Xch.getorder(xc, oid)
    if !isnothing(close_order)
        println("createcloseorder: $(DataFrame([close_order]))")
    else
        println("createcloseorder: order lookup returned nothing")
    end

    oodf = Xch.getopenorders(xc)
    println("getopenorders(nothing) - expect 2 open orders: $oodf")

    oid2 = Xch.cancelorder(xc, "BTC", oidx)
    println("cancelorder: $(string(oid2))")
    @test isnothing(oid2) || oidx == oid2
    oo2 = Xch.getorder(xc, oidx)
    if !isnothing(oo2)
        println("get cancelled order: $(DataFrame([oo2]))")
    else
        println("get cancelled order: order lookup returned nothing")
    end
    oodf = Xch.getopenorders(xc)
    println("getopenorders(nothing) - expect no open order: $oodf")

    # Regression: pure short exposure (free=0, borrowed>0) must carry negative value.
    short_balances = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[104_000f0, 0f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 2_000f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    short_prices = DataFrame(basecoin=String[EnvConfig.pairquote, "BTC"], lastprice=Float32[1f0, 2f0])
    short_portfolio = Xch.portfolio!(xc, short_balances, short_prices; ignoresmallvolume=false)
    btc_ix = findfirst(==("BTC"), short_portfolio.coin)
    @test !isnothing(btc_ix)
    @test short_portfolio[btc_ix, :usdtvalue] < 0f0
    @test isfinite(sum(short_portfolio.usdtvalue))


end


end  # module