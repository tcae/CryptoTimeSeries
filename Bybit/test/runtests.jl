using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(production)  # test production

@testset "Bybit tests" begin
    bc = Bybit.BybitCache()
    syminfo = Bybit.exchangeinfo(bc)
    @test isa(syminfo, AbstractDataFrame)
    @test size(syminfo, 1) > 100

    @test (Dates.now(UTC) + Dates.Second(15)) > Bybit.servertime(bc) > (Dates.now(UTC) - Dates.Second(15))

    # acc = Bybit.account(bc)
    # @test acc["marginMode"] == "ISOLATED_MARGIN"  broken=true # "REGULAR_MARGIN"
    # @test isa(acc, AbstractDict)
    # @test length(acc) > 1

    syminfo = Bybit.symbolinfo(bc, "BTCUSDT")
    @test isa(syminfo, DataFrameRow)

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

    # BybitSim: TestOhlcv symbols must provide klines and support simulated trading.
    bc_sim = Bybit.BybitCache()
    Bybit._init_simulation!(bc_sim)
    Bybit.seedportfolio!(bc_sim, EnvConfig.pairquote, 1_000f0)

    sdt = DateTime("2025-01-05T00:00:00")
    edt = DateTime("2025-01-05T01:00:00")
    sine_klines = Bybit.getklines(bc_sim, "SINEUSDT"; startDateTime=sdt, endDateTime=edt, interval="1m")
    dsine_klines = Bybit.getklines(bc_sim, "DOUBLESINEUSDT"; startDateTime=sdt, endDateTime=edt, interval="1m")
    @test size(sine_klines, 1) > 0
    @test size(dsine_klines, 1) > 0

    o_sine = Bybit.createorder(bc_sim, "SINEUSDT", "Buy", 2.0f0, nothing, false)
    @test !isnothing(o_sine)
    @test o_sine.symbol == "SINEUSDT"
    @test o_sine.status == "Filled"

    sim_balances = Bybit.balances(bc_sim)
    @test any(sim_balances.coin .== "SINE")
    @test sim_balances[sim_balances.coin .== "SINE", :free][1] > 0f0

    sim_capacity = Bybit.accountcapacity(bc_sim)
    @test sim_capacity.available_opening_quote > 0.0
    @test sim_capacity.available_long_quote == sim_capacity.available_opening_quote
    @test sim_capacity.available_short_quote == sim_capacity.available_opening_quote
    @test sim_capacity.equity_quote > sim_capacity.available_opening_quote
    @test sim_capacity.source == "Bybit:sim_wallet"


    # oocreate = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00001, btcprice * 0.9, false)
    # oid = isnothing(oocreate) ? nothing : oocreate.orderid

    # oo = Bybit.order(bc, oid)
    # @test isa(oo, DataFrameRow)
    # @test length(oo) >= 13
    # @test oo.orderid == oid

    # ooamend = Bybit.amendorder(bc, "BTCUSDT", oid; basequantity=0.00011)
    # @test ooamend.orderid == oid

    # ooamend = Bybit.amendorder(bc, "BTCUSDT", oid; limitprice=btcprice * 0.8)
    # @test ooamend.orderid == oid

    # oo = Bybit.openorders(bc)
    # @test isa(oo, AbstractDataFrame)
    # @test (size(oo, 1) > 0)
    # @test (size(oo, 2) >= 13)

    # coid = Bybit.cancelorder(bc, "BTCUSDT", oid)
    # @test coid == oid

    # oo = Bybit.order(bc, oid)
    # @test oo.status == "Cancelled"

    # wb = Bybit.balances(bc)
    # @test isa(wb, AbstractDataFrame)
    # @test size(wb, 2) == 3

end
