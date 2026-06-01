using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, CryptoXch, Targets

Base.@kwdef mutable struct ClosePriceRuntime <: TradingStrategy.AbstractStrategyRuntime
    snap::Union{Nothing, TradingStrategy.StrategySnapshot} = nothing
end

function TradingStrategy.getsnapshot!(
    rt::ClosePriceRuntime,
    xc::CryptoXch.XchCache,
    base::AbstractString,
    datetime::DateTime;
    reconciliation::TradingStrategy.StrategyReconciliationInput=TradingStrategy.StrategyReconciliationInput(),
)::Union{Nothing, TradingStrategy.StrategySnapshot}
    _ = xc
    _ = base
    _ = datetime
    _ = reconciliation
    return rt.snap
end

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

    tc.mc[:strategy_runtime] = ClosePriceRuntime(
        snap=TradingStrategy.StrategySnapshot(
            base="BTC",
            datetime=DateTime("2025-01-01T00:00:00"),
            label=Targets.ignore,
            long_closeprice=111.0f0,
            short_closeprice=0.0f0,
        ),
    )

    @test Trade._strategy_sell_limitprice(tc, "BTC", Targets.longclose; assets=assets) == 111.0f0
    @test isnothing(Trade._strategy_sell_limitprice(tc, "BTC", Targets.shortclose; assets=assets))

    gs = TradingStrategy.GainSegment(; algorithm=TradingStrategy.gain_limit_reversal!)
    gs.longta = TradingStrategy.TradeAction(longclose, 111.0f0, 100.0f0, 1)

    Trade._managedcloseset!(tc, "BTC", "oid-x", Targets.longclose; limitprice=111.0f0, baseqty=0.5f0)
    @test haskey(tc.mc[:managed_close_orders], longkey)
    Trade.apply_tradingstrategy!(tc, gs; source="test")
    @test isempty(tc.mc[:managed_close_orders])
end

@testset "Missing close-order check includes managed state" begin
    EnvConfig.init(EnvConfig.test)

    xc = CryptoXch.XchCache()
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.buysell)

    assets = DataFrame(
        coin=["BTC", EnvConfig.cryptoquote],
        free=Float32[0.0f0, 2000.0f0],
        locked=Float32[0.0f0, 0.0f0],
        borrowed=Float32[1.0f0, 0.0f0],
        usdtprice=Float32[100.0f0, 1.0f0],
        usdtvalue=Float32[-100.0f0, 2000.0f0],
    )

    Trade._managedcloseset!(tc, "BTC", "oid-pending-shortclose", Targets.shortclose; limitprice=100.0f0, baseqty=1.0f0)
    missing = Trade._positions_without_close_orders(tc, assets, DataFrame())
    @test isempty(missing)
end
