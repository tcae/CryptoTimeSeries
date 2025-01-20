
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
verbosity = 3 # 3
Ohlcv.verbosity = 0
CryptoXch.verbosity = 0

# EnvConfig.init(test)
# EnvConfig.init(production)

println("CryptoXchSimTest simruntests")
USDTEPS = 2 * eps(Float32)  # 0.00000001
BTCEPS = 2 * eps(Float32)  # 0.00000001
function testapprox(v1, v2)
    if isapprox(v1, v2, atol = 3 * max(max(v1,v2) * eps(Float32), eps(Float32)))
        return true
    else
        @error "actual=$v1 not approximate to expected=$v2 => absdelta= $(abs(v1-v2)) tolerated= $(max(v1,v2)*eps(Float32))"
        return false
    end
end

function showorders(order)
    # println("showorders: $(typeof(order)) - $order")
    if (typeof(order) <: DataFrameRow) || (typeof(order) <: NamedTuple)
        df = DataFrame()
        push!(df, order)
        order = df
    elseif typeof(order) <: DataFrame
        order = copy(order)
    end
    order.qteqty = order.baseqty .* order.limitprice
    order.exeqteqty = order.executedqty .* order.limitprice
    order.remainqty = (order.baseqty - order.executedqty) .* order.limitprice
    # println(order)
    return "$order"
end

function showsimassets(simassets)
    df = copy(simassets)
    df.qtelocked = df.locked .* df.usdtprice
    df.qtefree = df.free .* df.usdtprice
    df.qteborrowed = df.borrowed .* df.usdtprice
    df.qteaccruedinterest = df.accruedinterest .* df.usdtprice
    # println(df)
    return "$df"
end

function showassets(assets)
    # df = copy(assets)
    # df.qtelocked = df.locked .* df.usdtprice
    # df.qtefree = df.free .* df.usdtprice
    # df.qtemargin = df.margin .* df.usdtprice
    # df.qteassetborrowed = df.assetborrowed .* df.usdtprice
    # df.qteorderborrowed = df.orderborrowed .* df.usdtprice
    # df.qteaccruedinterest = df.accruedinterest .* df.usdtprice
    # println(df)
    return "$assets"
end

function check(xc, assetusdt, assetbtc, checkpoint, action)
    oodf = CryptoXch.getopenorders(xc)
    adf = CryptoXch.portfolio!(xc)
    if verbosity >= 3
        println()
        println("$(CryptoXch.ttstr(xc)) $checkpoint: $action")
        println("$checkpoint: openorders: $(showorders(oodf))")
        println("$checkpoint: portfolio: $(showsimassets(adf))")
        println("$checkpoint: xc.assets: $(xc.assets)")
        println("$checkpoint: assetbtc: $assetbtc")
        println("$checkpoint: assetusdt: $assetusdt")
    end
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :free]), assetusdt.free)
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :locked]), assetusdt.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :borrowed]), assetusdt.borrowed)
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :free]), assetbtc.free)
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :locked]), assetbtc.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :borrowed]), assetbtc.borrowed)
end

currenttime(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :opentime]
currentclose(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :close]
currenthigh(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :high]
currentlow(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :low]

EnvConfig.init(training)
#region shorttrades
@testset "CryptoXch margin trade simulation tests" begin

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(20)
    xc = CryptoXch.XchCache(startdt=startdt, enddt=enddt)
    # usdtbudget = 10000f0
    # btcbudget = 0f0
    assetbtc = (free=0f0, locked=0f0, borrowed=0f0)
    assetusdt = (free=10000f0, locked=0f0, borrowed=0f0)
    empty!(xc.orders)
    empty!(xc.closedorders)
    empty!(xc.assets)
    CryptoXch._updateasset!(xc, "USDT", assetusdt.free)
    CryptoXch.setcurrenttime!(xc, startdt)
    ohlcv = CryptoXch.cryptodownload(xc, "BTC", "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    Ohlcv.setix!(ohlcv, 1)
    # println(ohlcv)

    mdf = CryptoXch.getUSDTmarket(xc)
    @test size(mdf, 1) > 0
    # println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])

    btcprice = currentclose(ohlcv)
    assetbtc = (assetbtc..., price = btcprice)
    # at 2022-01-01T01:00:00 price falls 0.1% in the first 5 minutes and then rises 0.5% within 5.5 hours
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01)) BTC.close=$(xc.bases["BTC"].df[xc.bases["BTC"].ix, :close])")

    bdf = CryptoXch.balances(xc)
    @test size(bdf, 2) == 5
    # println(bdf)

    pdf = CryptoXch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) == 7
    # println(pdf)

    qteqty = 10f0
    mo1 = (qteqty=qteqty, limitprice = assetbtc.price * 1.005, baseqty = qteqty / (assetbtc.price * 1.005), time=currenttime(ohlcv), marginleverage=3)
    check(xc, assetusdt, assetbtc, "m1", "initial setup")

    # create margin short sell order
    oid = CryptoXch.createsellorder(xc, "BTC", limitprice=mo1.limitprice, basequantity=mo1.baseqty, maker=false, marginleverage=mo1.marginleverage)
    mo1 = (mo1..., oid=oid)
    so1locking = mo1.baseqty * mo1.limitprice / mo1.marginleverage 
    @test !isnothing(mo1.oid)
    assetbtc = (assetbtc..., borrowed = mo1.baseqty * 2f0 / 3f0)
    assetusdt = (assetusdt..., free = assetusdt.free - so1locking, locked = assetusdt.locked + so1locking)
    check(xc, assetusdt, assetbtc, "m2", "margin short open sell order $qteqty USDT")

    # execute margin short sell order
    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(29))
    order = CryptoXch.getorder(xc, mo1.oid)
    mo1 = (mo1..., time=currenttime(ohlcv))
    @test abs(mo1.limitprice - order.limitprice) < USDTEPS * order.limitprice
    assetbtc = (assetbtc..., free = -mo1.baseqty)
    assetusdt = (assetusdt..., locked = 0f0, free = assetusdt.free - mo1.baseqty * mo1.limitprice * xc.feerate) # free to be adjusted for fee
    check(xc, assetusdt, assetbtc, "m3", "margin short open sell order executed")

    # create margin buy with borrow at creation and extending borrow
    adf = CryptoXch.portfolio!(xc)
    assetbtc = (free=sum(adf[adf[!, :coin] .== "BTC", :free]), locked=sum(adf[adf[!, :coin] .== "BTC", :locked]), borrowed=sum(adf[adf[!, :coin] .== "BTC", :borrowed]))
    assetusdt = (free=sum(adf[adf[!, :coin] .== "USDT", :free]), locked=sum(adf[adf[!, :coin] .== "USDT", :locked]), borrowed=sum(adf[adf[!, :coin] .== "USDT", :borrowed]))
    limitprice = currentlow(ohlcv) * (1 - 0.0001)
    qteqty = 25f0
    mo2 = (qteqty = qteqty, limitprice = limitprice, baseqty = qteqty / limitprice, marginleverage = 3) 
    reduceqty = abs(assetbtc.free)
    extendqty = mo2.baseqty - reduceqty
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=mo2.limitprice, basequantity=mo2.baseqty, maker=false, marginleverage=mo2.marginleverage)
    newlockeddelta = extendqty * mo2.limitprice / mo2.marginleverage
    mo2 = (mo2..., oid = oid)
    @test !isnothing(mo1.oid)
    newborrowed = extendqty * (mo2.marginleverage - 1) / mo2.marginleverage
    assetbtc = (assetbtc..., free=0f0, borrowed = assetbtc.borrowed + newborrowed, locked = -reduceqty) # was previously a sell and now compensated by buy
    assetusdt = (assetusdt..., free = assetusdt.free - newlockeddelta, locked = assetusdt.locked + newlockeddelta)
    check(xc, assetusdt, assetbtc, "m4", "margin long open with borrowed short of $qteqty USDT")

    # execute margin buy with borrow at creation and extending borrow
    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(29))
    order = CryptoXch.getorder(xc, mo2.oid)
    @test abs(mo2.limitprice - order.limitprice) < USDTEPS * order.limitprice
    assetbtc = (assetbtc..., free = extendqty , locked = 0f0, borrowed = newborrowed) 
    assetusdt = (assetusdt..., locked = assetusdt.locked - newlockeddelta, free = assetusdt.free + reduceqty  * mo2.limitprice / mo2.marginleverage - mo2.baseqty * mo2.limitprice * xc.feerate) # free to be adjusted for fee
    check(xc, assetusdt, assetbtc, "m5", "margin long open with borrowed short executed")
    
    # create margin short open sell without borrow at creation using base coins that are in the protfolio
    qteqty = 10f0
    limitprice = currentclose(ohlcv) * (1+0.001)
    mo3 = (qteqty = qteqty, limitprice = limitprice, baseqty = qteqty / limitprice, marginleverage = 3) 
    oid = CryptoXch.createsellorder(xc, "BTC", limitprice=mo3.limitprice, basequantity=mo3.baseqty, maker=false, marginleverage=mo3.marginleverage)
    mo3 = (mo3..., oid=oid)
    locking = mo3.baseqty  # complete reduce  * mo3.limitprice / mo3.marginleverage 
    @test !isnothing(mo3.oid)
    assetbtc = (assetbtc..., locked = locking, free = assetbtc.free - locking)
    # assetusdt = (assetusdt..., free = assetusdt.free - locking, locked = assetusdt.locked + locking) no change due to complete reduce
    check(xc, assetusdt, assetbtc, "m6", "reduce only: create margin short open sell without borrow at creation using base coins that are in the protfolio")

    # execute margin sell without borrow at creation
    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(29))
    oodf = CryptoXch.getopenorders(xc)
    @test size(oodf, 1) == 0  # expecting order to be executed
    order = CryptoXch.getorder(xc, mo3.oid)
    mo3 = (mo3..., time=currenttime(ohlcv))
    @test abs(mo3.limitprice - order.limitprice) < USDTEPS * order.limitprice
    assetbtc = (assetbtc..., locked = 0f0, borrowed = assetbtc.borrowed - mo3.baseqty * (mo3.marginleverage - 1) / mo3.marginleverage)
    assetusdt = (assetusdt..., free = assetusdt.free +locking * mo3.limitprice / mo3.marginleverage - mo3.baseqty * mo3.limitprice * xc.feerate) # free to be adjusted for fee
    check(xc, assetusdt, assetbtc, "m7", "reduce only: executed margin short open sell without borrow at creation using base coins that are in the protfolio")

    # intermediate step to create negative wallet base on free
    qteqty = 30f0
    limitprice = currentclose(ohlcv) * (1+0.001)
    mo4 = (qteqty = qteqty, limitprice = limitprice, baseqty = qteqty / limitprice, marginleverage = 3) 
    oid = CryptoXch.createsellorder(xc, "BTC", limitprice=mo4.limitprice, basequantity=mo4.baseqty, maker=false, marginleverage=mo4.marginleverage)
    mo4 = (mo4..., oid=oid)
    @test !isnothing(mo4.oid)
    reduceqty = abs(assetbtc.free)
    extendqty = mo4.baseqty - reduceqty
    locking = extendqty * mo4.limitprice / mo4.marginleverage 
    assetbtc = (assetbtc..., free = 0f0, locked = reduceqty, borrowed = assetbtc.borrowed + extendqty * (mo4.marginleverage -1) / mo4.marginleverage )
    assetusdt = (assetusdt..., free = assetusdt.free - locking, locked = assetusdt.locked + locking)
    check(xc, assetusdt, assetbtc, "m8", "create margin short open sell with borrow at creation using base coins that are in the protfolio + borrow")

    # execute margin sell without borrow at creation
    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(240))
    oodf = CryptoXch.getopenorders(xc)
    @test size(oodf, 1) == 0  # expecting order to be executed
    order = CryptoXch.getorder(xc, mo4.oid)
    mo4 = (mo4..., time=currenttime(ohlcv))
    @test abs(mo4.limitprice - order.limitprice) < USDTEPS * order.limitprice
    assetbtc = (assetbtc..., free = -extendqty, locked = 0f0, borrowed = assetbtc.borrowed - reduceqty * (mo4.marginleverage -1) / mo4.marginleverage)
    assetusdt = (assetusdt..., locked = assetusdt.locked - locking, free = assetusdt.free + reduceqty * mo4.limitprice / mo4.marginleverage - mo4.baseqty * mo4.limitprice * xc.feerate) # free to be adjusted for fee
    check(xc, assetusdt, assetbtc, "m9", "executed margin short open sell with borrow at creation using base coins that are in the protfolio + borrow")

    
    # create margin long open buy without borrow at creation by reducing borrow
    limitprice = currentlow(ohlcv) * (1 - 0.0001)
    qteqty = 5f0
    mo5 = (qteqty = qteqty, limitprice = limitprice, baseqty = qteqty / limitprice, marginleverage = 3) 
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=mo5.limitprice, basequantity=mo5.baseqty, maker=false, marginleverage=mo5.marginleverage)
    newlockeddelta = mo5.baseqty * mo5.limitprice / mo5.marginleverage
    mo5 = (mo5..., oid = oid)
    @test !isnothing(mo1.oid)
    newborrowed = mo5.baseqty * (mo5.marginleverage - 1) / mo5.marginleverage
    assetbtc = (assetbtc..., locked = -mo5.baseqty, free = assetbtc.free + mo5.baseqty)
    # assetusdt = (assetusdt..., free = assetusdt.free - newlockeddelta, locked = assetusdt.locked + newlockeddelta) # no quote requires with pure reduce
    check(xc, assetusdt, assetbtc, "m10", "create margin long open buy without borrow at creation by reducing borrow of $qteqty USDT")

    # execute margin buy with borrow at creation and extending borrow
    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(29))
    order = CryptoXch.getorder(xc, mo5.oid)
    @test abs(mo5.limitprice - order.limitprice) < USDTEPS * order.limitprice
    oodf = CryptoXch.getopenorders(xc)
    @test size(oodf, 1) == 0  # expecting order to be executed
    assetbtc = (assetbtc..., locked = 0f0, borrowed = assetbtc.borrowed - mo5.baseqty * (mo5.marginleverage - 1) / mo5.marginleverage)
    assetusdt = (assetusdt..., free = assetusdt.free + mo5.baseqty * mo5.limitprice / mo5.marginleverage - mo5.baseqty * mo5.limitprice * xc.feerate) # free to be adjusted for fee
    check(xc, assetusdt, assetbtc, "m11", "executed margin long open buy without borrow at creation by reducing borrow")
    

end
#endregion shorttrades
    
#region longtrades
@testset "CryptoXch spot trade simulation tests" begin

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(20)
    xc = CryptoXch.XchCache(startdt=startdt, enddt=enddt)
    # usdtbudget = 10000f0
    # btcbudget = 0f0
    assetbtc = (free=0f0, locked=0f0, borrowed=0f0)
    assetusdt = (free=10000f0, locked=0f0, borrowed=0f0)
    empty!(xc.orders)
    empty!(xc.closedorders)
    empty!(xc.assets)
    CryptoXch._updateasset!(xc, "USDT", assetusdt.free)
    CryptoXch.setcurrenttime!(xc, startdt)
    ohlcv = CryptoXch.cryptodownload(xc, "BTC", "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    Ohlcv.setix!(ohlcv, 1)
    # println(ohlcv)

    mdf = CryptoXch.getUSDTmarket(xc)
    @test size(mdf, 1) > 0
    # println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])

    btcprice = currentclose(ohlcv)
    assetbtc = (assetbtc..., price = btcprice)
    # at 2022-01-01T01:00:00 price falls 0.1% in the first 5 minutes and then rises 0.5% within 5.5 hours
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01)) BTC.close=$(xc.bases["BTC"].df[xc.bases["BTC"].ix, :close])")

    bdf = CryptoXch.balances(xc)
    @test size(bdf, 2) == 5
    # println(bdf)

    pdf = CryptoXch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) == 7
    # println(pdf)

    # reset assets for long tests
    assetbtc = (free=0f0, locked=0f0,)
    assetusdt = (free=10000f0, locked=0f0)
    empty!(xc.orders)
    empty!(xc.closedorders)
    empty!(xc.assets)
    CryptoXch._updateasset!(xc, "USDT", assetusdt.free)
    CryptoXch.setcurrenttime!(xc, startdt)

    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=btcprice*1.2, basequantity=26.01/btcprice, maker=false) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    # println("limitprice=$(btcprice * 1.01)")

    adf = CryptoXch.balances(xc)
    (verbosity  >= 4) && println("$(CryptoXch.ttstr(xc)) s0) btc=$assetbtc \nusdt=$assetusdt \nsimassets=$(showsimassets(adf))")

    qteqty = 26.01f0
    o1 = (qteqty=qteqty, limitprice=assetbtc.price * 1.01, baseqty=qteqty/assetbtc.price)
    # usdtbudget = usdtbudget - qteqty
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=o1.limitprice, basequantity=o1.baseqty, maker=false) # maker=false to avoid PostOnly that will cause reject if price < limitprice due to taker order
    o1 = (o1..., id = oid)
    @test !isnothing(oid)
    assetusdt = (assetusdt..., free = assetusdt.free - o1.qteqty * (1 + xc.feerate))
    assetbtc = (assetbtc..., free = assetbtc.free + o1.qteqty / assetbtc.price)
    # println("createbuyorder: $(string(oid)) - reject expected")  # not applicable anymore because timeinforce is by default changed from PostOnly to GTC
    oo2 = CryptoXch.getorder(xc, o1.id)
    # println("get (not rejected) order: $(DataFrame([oo2]))")
    @test oo2.orderid == o1.id
    # @test oo2.status == "Rejected"  - not Rejected because default was changed from PostOnly to GTC
    @test oo2.status == "Filled"

    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 4) && println("$(CryptoXch.ttstr(xc)) s1) btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :free]), assetusdt.free)
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :free]), assetbtc.free)

    oid = CryptoXch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oid)

    qteqty = 8.01
    o2 = (qteqty=qteqty, limitprice = assetbtc.price * 0.9, baseqty = qteqty / assetbtc.price)
    oidx = CryptoXch.createbuyorder(xc, "BTC", limitprice=o2.limitprice, basequantity=o2.baseqty, maker=false)
    o2 = (o2..., id = oidx)

    assetusdt = (assetusdt..., locked = assetusdt.locked + o2.baseqty * o2.limitprice, free = assetusdt.free - Float32(o2.baseqty * o2.limitprice))
    adf = CryptoXch.balances(xc)
    (verbosity  >= 4) && println("$(CryptoXch.ttstr(xc)) s2) btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :locked]), assetusdt.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :free]), assetusdt.free)

    qteqty = 6.01
    o3 = (qteqty=qteqty, limitprice = assetbtc.price * 0.999, baseqty = qteqty / assetbtc.price)
    oid = CryptoXch.createbuyorder(xc, "BTC", limitprice=o3.limitprice, basequantity=o3.baseqty, maker=false)
    o3 = (o3..., id = oid)

    assetusdt = (assetusdt..., locked = assetusdt.locked + o3.baseqty * o3.limitprice, free = assetusdt.free - Float32(o3.baseqty * o3.limitprice))
    adf = CryptoXch.balances(xc)
    (verbosity  >= 4) && println("$(CryptoXch.ttstr(xc)) s3) btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :locked]), assetusdt.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :free]), assetusdt.free)

    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 4) && println("getopenorders(nothing): $(showorders(oodf))")
    # println("createbuyorder: $(string(oid))")
    oo2 = CryptoXch.getorder(xc, o3.id)
    # println("getorder: $oo2")
    @test o3.id == oo2.orderid

    qteqty = 4.02
    o4 = (o3..., qteqty = qteqty, baseqty = qteqty / assetbtc.price)
    (verbosity  >= 4) && println("\n$(CryptoXch.ttstr(xc)) s4a) changeorder assetbtc.price=$(assetbtc.price) \no4=$o4 \no3=$o3")
    oidc = CryptoXch.changeorder(xc, o4.id; basequantity=o4.baseqty)
    @test oidc == o4.id
    # println("changeorder after basequatity change: $(DataFrame([CryptoXch.getorder(oid)]))")
    assetusdt = (assetusdt..., locked = assetusdt.locked + (o4.baseqty - o3.baseqty) * o4.limitprice, free = assetusdt.free + (o3.baseqty - o4.baseqty) * o4.limitprice)
    adf = CryptoXch.balances(xc)
    (verbosity  >= 4) && println("$(CryptoXch.ttstr(xc)) s4b) changeorder btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 4) && println("$(CryptoXch.ttstr(xc)) s4c) getopenorders(nothing): $(showorders(oodf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :locked]), assetusdt.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :free]), assetusdt.free)

    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(4))
    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    assetusdt = (assetusdt..., locked = assetusdt.locked - o4.baseqty * o4.limitprice)
    assetbtc = (assetbtc..., free = assetbtc.free + o4.baseqty * (1 - xc.feerate))
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("\n$(CryptoXch.ttstr(xc)) s5a) getopenorders(nothing): $(showorders(oodf))")
    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s5b) order filled over time by price increase btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :locked]), assetusdt.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :free]), assetbtc.free)

    order = CryptoXch.getorder(xc, o2.id)
    o5 = (o2..., limitprice=assetbtc.price * 0.998)
    o5locked = (o5.limitprice - o2.limitprice) * (o5.baseqty - order.executedqty)
    assetusdt = (assetusdt..., locked = assetusdt.locked + o5locked, free = assetusdt.free - o5locked)
    oidc = CryptoXch.changeorder(xc, o5.id; limitprice=o5.limitprice)
    @test oidc == o5.id
    # println("changeorder after limitprice change: $(DataFrame([CryptoXch.getorder(oidx)]))")
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("\n$(CryptoXch.ttstr(xc)) s6a) previously placed order=$(showorders(order)) \no2=$o2 \no5=$o5 \ngetopenorders(nothing): $(showorders(oodf))")
    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s6b) changeorder btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :locked]), assetusdt.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :free]), assetusdt.free)

    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(6))
    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    oo2 = CryptoXch.getorder(xc, oid)
    @test oid == oo2.orderid
    # println("getorder after fill: $(DataFrame([oo2]))")

    # oodf = CryptoXch.getopenorders(xc, "BTC")
    # println("getopenorders(BTC) after fill: $(showorders(oodf))")  # there is no fill

    pdf = CryptoXch.balances(xc)
    @test size(pdf, 1) == 2
    btcqty = sum(pdf[pdf.coin .== "BTC", :free])
    # println("portfolio with btcqty=$btcqty $pdf")

    qteqty = 8.01
    o6 = (qteqty=btcqty * assetbtc.price, limitprice = assetbtc.price * 1.005, baseqty = btcqty)
    oid = CryptoXch.createsellorder(xc, "BTC", limitprice=o6.limitprice, basequantity=o6.baseqty, maker=false)
    oo6 = (o6..., id=oid)
    assetbtc = (assetbtc..., free = assetbtc.free - o6.baseqty, locked = assetbtc.locked + o6.baseqty)
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s7a) getopenorders(nothing) - expect 1 longbuy and 1 longclose order: $(showorders(oodf))")
    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s7b) changeorder btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :free]), assetbtc.free)
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :locked]), assetbtc.locked)

    oodf = CryptoXch.getopenorders(xc, "BTC")
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) == 2)
    @test (size(oodf, 2) >= 12)
    # println("getopenorders(\"BTC\"): $(showorders(oodf))")
    oodf = CryptoXch.getopenorders(xc, "XRP")
    @test (size(oodf, 1) == 0)
    @test (size(oodf, 2) >= 12)
    # println("getopenorders(\"XRP\"): $(showorders(oodf))")

    CryptoXch.setcurrenttime!(xc, xc.currentdt + Minute(990))
    btcprice = Ohlcv.current(CryptoXch.ohlcv(xc, "BTC")).close
    assetbtc = (assetbtc..., price = btcprice)
    assetbtc = (assetbtc..., locked = assetbtc.locked - o6.baseqty)
    assetusdt = (assetusdt..., free = assetusdt.free + o6.baseqty * o6.limitprice * (1 - xc.feerate))

    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s8a) getopenorders(nothing) - expect 1 longbuy order open: $(showorders(oodf))")
    @test (size(oodf, 1) == 1)

    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s8b) 990 minutes later btc=$assetbtc usdt=$assetusdt simassets=$(showsimassets(adf))")
    @test testapprox(sum(adf[adf[!, :coin] .== "BTC", :locked]), assetbtc.locked)
    @test testapprox(sum(adf[adf[!, :coin] .== "USDT", :free]), assetusdt.free)

    oid2 = CryptoXch.cancelorder(xc, "BTC", o2.id)
    # println("cancelorder: $(string(oid2))")
    @test o2.id == oid2


    adf = CryptoXch.portfolio!(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s9a) $(showsimassets(adf))")

    oo2 = CryptoXch.getorder(xc, o2.id)
    # println("get cancelled order: $(DataFrame([oo2]))")
    oodf = CryptoXch.getopenorders(xc)
    (verbosity  >= 3) && println("$(CryptoXch.ttstr(xc)) s9b) getopenorders(nothing) - expect no open order: $(showorders(oodf))")

    #endregion longtrades

end


end  # module