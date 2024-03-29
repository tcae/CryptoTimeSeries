using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(production)  # test production

@testset "Bybit tests" begin
    bc = Bybit.BybitCache()
    syminfo = Bybit.exchangeinfo(bc)
    @test isa(syminfo, AbstractDataFrame)
    @test size(syminfo, 1) > 100

    @test (Dates.now(UTC) + Dates.Second(15)) > Bybit.servertime(bc) > (Dates.now(UTC) - Dates.Second(15))

    acc = Bybit.account(bc)
    @test acc["marginMode"] == "ISOLATED_MARGIN"  # "REGULAR_MARGIN"
    @test isa(acc, AbstractDict)
    @test length(acc) > 1

    syminfo = Bybit.symbolinfo(bc, "BTCUSDT")
    @test isa(syminfo, NamedTuple)

    dayresult = Bybit.get24h(bc)
    @test isa(dayresult, AbstractDataFrame)
    @test size(dayresult, 1) > 100

    dayresult = Bybit.get24h(bc, "BTCUSDT")
    @test isa(dayresult, DataFrameRow)
    @test length(dayresult) >= 6
    @test all([s in ["askprice", "bidprice", "lastprice", "quotevolume24h", "pricechangepercent", "symbol"] for s in names(dayresult)])
    btcprice = dayresult.lastprice

    klines = Bybit.getklines(bc, "BTCUSDT")
    @test isa(klines, AbstractDataFrame)


    oid = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00001, btcprice * 0.9, false)

    oo = Bybit.order(bc, oid)
    @test isa(oo, DataFrameRow)
    @test length(oo) >= 13
    @test oo.orderid == oid

    oidc = Bybit.amendorder(bc, "BTCUSDT", oid; basequantity=0.00011)
    @test oidc == oid

    println("runtests $(Bybit.order(bc, oid))")
    oidc = Bybit.amendorder(bc, "BTCUSDT", oid; limitprice=btcprice * 0.8)
    @test oidc == oid

    oo = Bybit.openorders(bc)
    @test isa(oo, AbstractDataFrame)
    @test (size(oo, 1) > 0)
    @test (size(oo, 2) >= 13)

    coid = Bybit.cancelorder(bc, "BTCUSDT", oid)
    @test coid == oid

    oo = Bybit.order(bc, oid)
    @test oo.status == "Cancelled"

    wb = Bybit.balances(bc)
    @test isa(wb, AbstractDataFrame)
    @test size(wb, 2) == 3

end
