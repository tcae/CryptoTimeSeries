using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(production)

@testset "Bybit tests" begin

    syminfo = Bybit.exchangeinfo()
    @test isa(syminfo, AbstractDataFrame)
    @test size(syminfo, 1) > 100

    @test (Dates.now(UTC) + Dates.Second(15)) > Bybit.servertime() > (Dates.now(UTC) - Dates.Second(15))

    acc = Bybit.account()
    @test acc["marginMode"] == "REGULAR_MARGIN"
    @test isa(acc, AbstractDict)
    @test length(acc) > 1

    syminfo = Bybit.symbolinfo("BTCUSDT")
    @test isa(syminfo, NamedTuple)

    dayresult = Bybit.get24h()
    @test isa(dayresult, AbstractDataFrame)
    @test size(dayresult, 1) > 100

    dayresult = Bybit.get24h("BTCUSDT")
    @test isa(dayresult, AbstractDataFrame)
    @test size(dayresult, 2) >= 6
    @test all([s in ["askprice", "bidprice", "lastprice", "quotevolume24h", "pricechangepercent", "symbol"] for s in names(dayresult)])
    @test size(dayresult, 1) == 1

    klines = Bybit.getklines("BTCUSDT")
    @test isa(klines, AbstractDataFrame)


    oid = Bybit.createorder("BTCUSDT", "Buy", 0.00001, 39899)

    oo = Bybit.order(oid)
    @test isa(oo, NamedTuple)
    @test length(oo) == 12
    @test oo.orderid == oid

    oo = Bybit.openorders()
    @test isa(oo, AbstractDataFrame)
    @test (size(oo, 1) > 0)
    @test (size(oo, 2) == 12)

    coid = Bybit.cancelorder("BTCUSDT", oid)
    @test coid == oid

    oo = Bybit.order(oid)
    @test oo.status == "Cancelled"

    wb = Bybit.balances()
    @test isa(wb, AbstractDataFrame)
    @test size(wb, 2) >= 18

end
