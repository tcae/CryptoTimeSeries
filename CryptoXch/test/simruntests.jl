
module CryptoXchSimTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, CryptoXch

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 0
Ohlcv.verbosity = 0
CryptoXch.verbosity = 0

# EnvConfig.init(test)
# EnvConfig.init(production)

println("CryptoXchSimTest simruntests")

@testset "CryptoXch simulation tests" begin

    EnvConfig.init(training)
    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(20)
    xc = CryptoXch.XchCache(true; startdt=startdt, enddt=enddt)
    # usdtbudget = 10000f0
    # btcbudget = 0f0
    assetbtc = (free=0f0, locked=0f0,)
    assetusdt = (free=10000f0, locked=0f0)
    CryptoXch.updateasset!(xc, "USDT", assetusdt.locked, assetusdt.free)
    CryptoXch.setcurrenttime!(xc, startdt)
    ohlcv = CryptoXch.cryptodownload(xc, "BTC", "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
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

    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice*1.2, basequantity=26.01/btcprice, maker=false) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    # println("limitprice=$(btcprice * 1.01)")

    adf = CryptoXch.balances(xc)
    (verbosity  >= 3) && println("0) btc=$assetbtc usdt=$assetusdt simassets=$adf")

    qteqty = 26.01f0
    o1 = (qteqty=qteqty, limit=assetbtc.price * 1.01, baseqty=qteqty/assetbtc.price)
    # usdtbudget = usdtbudget - qteqty
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=o1.limit, basequantity=o1.baseqty, maker=false) # PostOnly will cause reject if price < limitprice due to taker order
    o1 = (o1..., id = oid)
    @test !isnothing(oid)
    assetusdt = (assetusdt..., free = assetusdt.free - o1.qteqty)
    assetbtc = (assetbtc..., free = assetbtc.free + o1.qteqty / assetbtc.price * (1 - xc.feerate))
    # println("createbuyorder: $(string(oid)) - reject expected")  # not applicable anymore because timeinforce is by default changed from PostOnly to GTC
    oo2 = CryptoXch.getorder(xc, o1.id)
    # println("get (not rejected) order: $(DataFrame([oo2]))")
    @test oo2.orderid == o1.id
    # @test oo2.status == "Rejected"  - not Rejected because default was changed from PostOnly to GTC
    @test oo2.status == "Filled"

    USDTEPS = 0.001
    BTCEPS = 0.00000001

    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("1) btc=$assetbtc usdt=$assetusdt simassets=$adf")
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :free]) - assetusdt.free) < USDTEPS
    @test abs(sum(adf[adf[!, :coin] .== "BTC", :free]) - assetbtc.free) < BTCEPS

    oid = CryptoXch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oid)

    qteqty = 8.01
    o2 = (qteqty=qteqty, limit = assetbtc.price * 0.9, baseqty = qteqty / assetbtc.price)
    oidx = CryptoXch.createbuyorder(xc, "BTC", limitprice=o2.limit, basequantity=o2.baseqty, maker=false)
    o2 = (o2..., id = oidx)

    assetusdt = (assetusdt..., locked = assetusdt.locked + o2.baseqty * o2.limit, free = assetusdt.free - Float32(o2.baseqty * o2.limit))
    adf = CryptoXch.balances(xc)
    (verbosity  >= 3) && println("2) btc=$assetbtc usdt=$assetusdt simassets=$adf")
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :locked]) - assetusdt.locked) < USDTEPS
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :free]) - assetusdt.free) < USDTEPS

    qteqty = 6.01
    o3 = (qteqty=qteqty, limit = assetbtc.price * 0.999, baseqty = qteqty / assetbtc.price)
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=o3.limit, basequantity=o3.baseqty, maker=false)
    o3 = (o3..., id = oid)

    assetusdt = (assetusdt..., locked = assetusdt.locked + o3.baseqty * o3.limit, free = assetusdt.free - Float32(o3.baseqty * o3.limit))
    adf = CryptoXch.balances(xc)
    (verbosity  >= 3) && println("3) btc=$assetbtc usdt=$assetusdt simassets=$adf")
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :locked]) - assetusdt.locked) < USDTEPS
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :free]) - assetusdt.free) < USDTEPS

    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("getopenorders(nothing): $oodf")
    # println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(xc, o3.id)
    # println("getorder: $oo2")
    @test o3.id == oo2.orderid

    qteqty = 4.02
    o4 = (o3..., qteqty = qteqty, baseqty = qteqty / assetbtc.price)
    (verbosity  >= 3) && println("4) changeorder assetbtc.price=$(assetbtc.price) \no4=$o4 \no3=$o3")
    oidc = CryptoXch.changeorder(xc, o4.id; basequantity=o4.baseqty)
    @test oidc == o4.id
    # println("changeorder after basequatity change: $(DataFrame([CryptoXch.getorder(oid)]))")
    assetusdt = (assetusdt..., locked = assetusdt.locked + (o4.baseqty - o3.baseqty) * o4.limit, free = assetusdt.free + (o3.baseqty - o4.baseqty) * o4.limit)
    adf = CryptoXch.balances(xc)
    (verbosity  >= 3) && println("4) changeorder btc=$assetbtc usdt=$assetusdt simassets=$adf")
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("4) getopenorders(nothing): $oodf")
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :locked]) - assetusdt.locked) < USDTEPS
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :free]) - assetusdt.free) < USDTEPS

    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(4))
    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    assetusdt = (assetusdt..., locked = assetusdt.locked - o4.baseqty * o4.limit)
    assetbtc = (assetbtc..., free = assetbtc.free + o4.baseqty * (1 - xc.feerate))
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("5) getopenorders(nothing): $oodf")
    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("5) order filled over time by price increase btc=$assetbtc usdt=$assetusdt simassets=$adf")
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :locked]) - assetusdt.locked) < USDTEPS
    @test abs(sum(adf[adf[!, :coin] .== "BTC", :free]) - assetbtc.free) < BTCEPS

    o5 = (o2..., limit=assetbtc.price * 0.998)
    assetusdt = (assetusdt..., locked = assetusdt.locked + (o5.limit - o2.limit) * o5.baseqty, free = assetusdt.free + (o2.limit - o5.limit) * o5.baseqty)
    oidc = CryptoXch.changeorder(xc, o5.id; limitprice=o5.limit)
    @test oidc == o5.id
    # println("changeorder after limitprice change: $(DataFrame([CryptoXch.getorder(oidx)]))")
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("6) getopenorders(nothing): $oodf")
    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("6) changeorder btc=$assetbtc usdt=$assetusdt simassets=$adf")
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :locked]) - assetusdt.locked) < USDTEPS
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :free]) - assetusdt.free) < USDTEPS

    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(6))
    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    oo2 = CryptoXch.getorder(xc, oid)
    @test oid == oo2.orderid
    # println("getorder after fill: $(DataFrame([oo2]))")

    # oodf = CryptoXch.getopenorders(xc, "BTC")
    # println("getopenorders(BTC) after fill: $oodf")  # there is no fill

    pdf = CryptoXch.balances(xc)
    @test size(pdf, 1) == 2
    btcqty = sum(pdf[pdf.coin .== "BTC", :free])
    # println("portfolio with btcqty=$btcqty $pdf")

    qteqty = 8.01
    o6 = (qteqty=btcqty * assetbtc.price, limit = assetbtc.price * 1.005, baseqty = btcqty)
    oid = CryptoXch.createsellorder(xc, "BTC", limitprice=o6.limit, basequantity=o6.baseqty, maker=false)
    oo6 = (o6..., id=oid)
    assetbtc = (assetbtc..., free = assetbtc.free - o6.baseqty, locked = assetbtc.locked + o6.baseqty)
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("getopenorders(nothing) - expect 1 longbuy and 1 longclose order: $oodf")
    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("7) changeorder btc=$assetbtc usdt=$assetusdt simassets=$adf")
    @test abs(sum(adf[adf[!, :coin] .== "BTC", :free]) - assetbtc.free) < BTCEPS
    @test abs(sum(adf[adf[!, :coin] .== "BTC", :locked]) - assetbtc.locked) < BTCEPS

    oodf = CryptoXch.getopenorders(xc, "BTC")
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) == 2)
    @test (size(oodf, 2) >= 12)
    # println("getopenorders(\"BTC\"): $oodf")
    oodf = CryptoXch.getopenorders(xc, "XRP")
    @test (size(oodf, 1) == 0)
    @test (size(oodf, 2) >= 12)
    # println("getopenorders(\"XRP\"): $oodf")

    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(990))
    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    assetbtc = (assetbtc..., locked = assetbtc.locked - o6.baseqty)
    assetusdt = (assetusdt..., free = assetusdt.free + o6.baseqty * o6.limit * (1 - xc.feerate))

    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("8) getopenorders(nothing) - expect 1 longbuy order open: $oodf")
    @test (size(oodf, 1) == 1)

    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("8) 990 minutes later btc=$assetbtc usdt=$assetusdt simassets=$adf")
    @test abs(sum(adf[adf[!, :coin] .== "BTC", :locked]) - assetbtc.locked) < BTCEPS
    @test abs(sum(adf[adf[!, :coin] .== "USDT", :free]) - assetusdt.free) < USDTEPS

    oid2 = CryptoXch.cancelorder(xc, "BTC", o2.id)
    # println("cancelorder: $(string(oid2))")
    @test o2.id == oid2


    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("9) $adf")

    oo2 = CryptoXch.getorder(xc, o2.id)
    # println("get cancelled order: $(DataFrame([oo2]))")
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("getopenorders(nothing) - expect no open order: $oodf")

end


end  # module