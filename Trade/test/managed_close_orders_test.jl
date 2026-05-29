using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, CryptoXch, Targets

@testset "Managed close order state" begin
    EnvConfig.init(EnvConfig.test)

    xc = CryptoXch.XchCache()
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.notrade)
    tc.cfg = DataFrame(
        basecoin=["BTC"],
        buyenabled=[true],
        sellenabled=[true],
        classifieraccepted=[true],
        minquotevol=[true],
        continuousminvol=[true],
        whitelisted=[true],
        robotownedlongqty=[1.0f0],
        robotownedshortqty=[0.0f0],
        datetime=[DateTime("2025-01-01T00:00:00")],
    )

    assets = DataFrame(
        coin=["BTC", EnvConfig.cryptoquote],
        free=Float32[1.0f0, 1000.0f0],
        locked=Float32[0.0f0, 0.0f0],
        borrowed=Float32[0.0f0, 0.0f0],
        usdtprice=Float32[100.0f0, 1.0f0],
        usdtvalue=Float32[100.0f0, 1000.0f0],
    )

    symbol = CryptoXch.symboltoken(tc.xc, "BTC", EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    oo = DataFrame(
        orderid=["oid-close", "oid-entry"],
        symbol=[symbol, symbol],
        side=["Sell", "Buy"],
        baseqty=Float32[1.0f0, 0.5f0],
        ordertype=["Limit", "Limit"],
        timeinforce=["PostOnly", "PostOnly"],
        limitprice=Float32[110.0f0, 90.0f0],
        executedqty=Float32[0.0f0, 0.0f0],
        status=["New", "New"],
        created=[DateTime("2025-01-01T00:00:00"), DateTime("2025-01-01T00:00:00")],
        updated=[DateTime("2025-01-01T00:00:00"), DateTime("2025-01-01T00:00:00")],
        rejectreason=["", ""],
    )

    Trade._reconstruct_managed_close_orders!(tc, assets, oo)
    managed = tc.mc[:managed_close_orders]
    longkey = Trade._managedclosekey("BTC", Targets.longclose)
    @test haskey(managed, longkey)
    @test managed[longkey][:orderid] == "oid-close"
    @test managed[longkey][:tradelabel] == Targets.longclose

    gs = TradingStrategy.GainSegment(; algorithm=TradingStrategy.gain_limit_reversal!)
    gs.longta = TradingStrategy.TradeAction(longclose, 111.0f0, 100.0f0, 1)
    tc.mc[:strategy_engine] = :getgainsalgo
    tc.mc[:strategy_state]["BTC"] = gs

    @test Trade._strategy_sell_limitprice(tc, "BTC", Targets.longclose) == 111.0f0
    @test isnothing(Trade._strategy_sell_limitprice(tc, "BTC", Targets.shortclose))

    Trade._managedcloseset!(tc, "BTC", "oid-x", Targets.longclose; limitprice=111.0f0, baseqty=0.5f0)
    @test haskey(tc.mc[:managed_close_orders], longkey)
    Trade.apply_tradingstrategy!(tc, gs; strategy_engine=:getgainsalgo, source="test")
    @test isempty(tc.mc[:managed_close_orders])
end
