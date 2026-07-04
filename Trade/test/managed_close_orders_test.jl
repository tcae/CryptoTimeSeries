using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, Xch, Targets

@testset "Trade leverage defaults" begin
    df = DataFrame(opentime=[DateTime("2025-01-01T00:00:00")], pair=["BTCUSDT"])
    for contributor in Trade.tradesdf_contributors()
        contributor(df)
    end

    @test eltype(df[!, :longleverage]) == UInt8
    @test eltype(df[!, :shortleverage]) == UInt8
    @test df[1, :longleverage] == UInt8(1)
    @test df[1, :shortleverage] == UInt8(1)
end

@testset "Managed close order state" begin
    EnvConfig.init(EnvConfig.test)

    xc = Xch.XchCache()
    Xch.ensuretradesschema(
        xc,
        vcat(
            Xch.tradesdf_contributors(),
            TradingStrategy.tradesdf_contributors(),
            Trade.tradesdf_contributors(),
        ),
    )
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.notrade)
    tc.cfg = DataFrame(
        basecoin=["BTC"],
        openenabled=[true],
        closeenabled=[true],
        classifieraccepted=[true],
        minquotevol=[true],
        continuousminvol=[true],
        whitelisted=[true],
        datetime=[DateTime("2025-01-01T00:00:00")],
    )

    assets = DataFrame(
        coin=["BTC", EnvConfig.pairquote],
        free=Float32[1.0f0, 1000.0f0],
        locked=Float32[0.0f0, 0.0f0],
        borrowed=Float32[0.0f0, 0.0f0],
        usdtprice=Float32[100.0f0, 1.0f0],
        usdtvalue=Float32[100.0f0, 1000.0f0],
    )

    symbol = Xch.symboltoken(tc.xc, "BTC", EnvConfig.pairquote; role=Xch.trade_exchange_spot)
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
    @test managed[longkey][:label] == Targets.longclose

    gs = TradingStrategy.StrategyConfig(; algorithm=TradingStrategy.gain_limit_reversal!)

    Trade._managedcloseset!(tc, "BTC", "oid-x", Targets.longclose; limitprice=111.0f0, baseqty=0.5f0)
    @test haskey(tc.mc[:managed_close_orders], longkey)
    Trade.apply_tradingstrategy!(tc, gs; source="test")
    @test isempty(tc.mc[:managed_close_orders])
end

@testset "Missing close-order check includes managed state" begin
    EnvConfig.init(EnvConfig.test)

    xc = Xch.XchCache()
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.buysell)

    assets = DataFrame(
        coin=["BTC", EnvConfig.pairquote],
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

@testset "Missing close-order check ignores dust quantities" begin
    EnvConfig.init(EnvConfig.test)

    xc = Xch.XchCache()
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.buysell)

    assets = DataFrame(
        coin=["BTC", EnvConfig.pairquote],
        free=Float32[5f-7, 2000.0f0],
        locked=Float32[0.0f0, 0.0f0],
        borrowed=Float32[0.0f0, 0.0f0],
        usdtprice=Float32[100.0f0, 1.0f0],
        usdtvalue=Float32[0.0f0, 2000.0f0],
    )

    missing = Trade._positions_without_close_orders(tc, assets, DataFrame())
    @test isempty(missing)
end

@testset "Order amend threshold default and material change boundary" begin
    tc = Trade.TradeCache(xc=Xch.XchCache(), cl=Classify.Classifier011(), trademode=Trade.notrade)
    amend_price_reltol = 1f-3

    oldp = 100f0
    newp_small = 100.05f0   # 0.05%
    newp_large = 100.2f0    # 0.2%

    @test !Trade._material_order_change(oldp, newp_small, 1f0, 1f0; price_reltol=amend_price_reltol)
    @test Trade._material_order_change(oldp, newp_large, 1f0, 1f0; price_reltol=amend_price_reltol)
end
