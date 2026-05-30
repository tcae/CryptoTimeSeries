module CryptoXchLogIntegrationTest
using Test
using Dates

using EnvConfig, CryptoXch, TradeLog

@testset "CryptoXch log integration" begin
    oldroot = get(ENV, "CTS_TRADELOG_ROOT", nothing)
    tmpdir = mktempdir()
    try
        ENV["CTS_TRADELOG_ROOT"] = tmpdir
        EnvConfig.init(test)

        xc = CryptoXch.XchCache()
        orderinfo = (
            orderid="order-123",
            timeinforce="GTC",
            status="New",
            rejectreason="NO ERROR",
            executedqty=0.0f0,
            avgprice=0.0f0,
        )
        CryptoXch._tradelogcreatedorder!(xc, CryptoXch.trade_exchange_spot, "BTCUSDT", "Buy", 0.01f0, 65000.0f0, 0, orderinfo)
        CryptoXch._tradelogordererror!(xc, CryptoXch.trade_exchange_spot, "BTCUSDT", "Buy", 0.01f0, 65000.0f0, 0, ErrorException("guardrail"))

        submitted = TradeLog.AuditEventRow(
            event_type=TradeLog.ORDER_SUBMITTED,
            environment=string(Symbol(EnvConfig.configmode)),
            run_mode=CryptoXch.tradelogrunmode(xc),
            exchange=CryptoXch.exchange(xc),
            account_alias=CryptoXch.exchange(xc),
            asset_class=TradeLog.crypto,
            instrument_type=TradeLog.spot_pair,
        )
        auditpath = TradeLog.auditfile(submitted)
        @test isfile(auditpath)
        events = readlines(auditpath)
        @test length(events) == 2
        @test occursin("\"event_type\":\"ORDER_SUBMITTED\"", events[1])
        @test occursin("\"symbol\":\"BTCUSDT\"", events[1])
        @test occursin("\"side\":\"Buy\"", events[1])
        @test occursin("\"requested_base_qty\":", events[1])
        @test occursin("\"run_mode\":\"simulation\"", events[1])
        @test occursin("\"run_id\":\"", events[1])
        @test occursin("\"event_type\":\"ORDER_REJECTED\"", events[2])
        @test occursin("\"status_reason\":\"guardrail\"", events[2])

        # Reconcile status transitions through getopenorders/getorder in simulation mode.
        dt = DateTime("2025-01-01T00:00:00")
        template_order = (
            orderid="template",
            symbol="BTCUSDT",
            side="Buy",
            baseqty=0.5f0,
            ordertype="Limit",
            marginleverage=0,
            timeinforce="GTC",
            limitprice=60000.0f0,
            avgprice=0.0f0,
            executedqty=0.0f0,
            status="New",
            created=dt,
            updated=dt,
            rejectreason="NO ERROR",
            lastcheck=dt,
        )
        ack_order = (; template_order..., orderid="order-ack-1", symbol="BTCUSDT", status="New")
        filled_order = (; template_order..., orderid="order-filled-1", symbol="ETHUSDT", side="Sell", baseqty=1.0f0, limitprice=3000.0f0, avgprice=3001.0f0, executedqty=1.0f0, status="Filled")
        cancelled_order = (; template_order..., orderid="order-cancel-1", symbol="SOLUSDT", side="Buy", baseqty=2.0f0, limitprice=150.0f0, avgprice=0.0f0, executedqty=0.0f0, status="Cancelled")
        signal_filled_order = (; template_order..., orderid="order-filled-signal-1", symbol="BTCUSDT", side="Buy", baseqty=0.25f0, limitprice=65000.0f0, avgprice=65130.0f0, executedqty=0.25f0, status="Filled")
        simbc = CryptoXch._routedbc(xc, CryptoXch.trade_exchange_spot)
        @test !isnothing(simbc)
        push!(simbc.orders, (isLeverage=false, ack_order...); cols=:subset)
        push!(simbc.orders, (isLeverage=false, filled_order...); cols=:subset)
        push!(simbc.orders, (isLeverage=false, cancelled_order...); cols=:subset)

        CryptoXch.getopenorders(xc)
        CryptoXch.getorder(xc, "order-ack-1")
        CryptoXch.getorder(xc, "order-filled-1")
        CryptoXch.getorder(xc, "order-cancel-1")
        CryptoXch.settradelogcontext!(xc; strategy_engine="classifier", strategy_config_ref="trenddetector:test", signal_label="longbuy", signal_score=0.91, leg_group_id="grp-1", leg_label="take_profit")
        push!(simbc.orders, (isLeverage=false, signal_filled_order...); cols=:subset)
        CryptoXch.getorder(xc, "order-filled-signal-1")
        CryptoXch.cleartradelogcontext!(xc)

        child_order = (; template_order..., orderid="order-child-1", symbol="BTCUSDT", status="New")
        CryptoXch._tradelogsetorderparent!(xc, "order-child-1", "order-ack-1")
        CryptoXch._tradelogreconcileorderstate!(xc, child_order; source="test-parent")

        events = readlines(auditpath)
        @test any(line -> occursin("\"event_type\":\"ORDER_ACK\"", line) && occursin("\"exchange_order_id\":\"order-ack-1\"", line), events)
        @test any(line -> occursin("\"event_type\":\"ORDER_FILLED\"", line) && occursin("\"exchange_order_id\":\"order-filled-1\"", line), events)
        @test any(line -> occursin("\"event_type\":\"ORDER_CANCELED\"", line) && occursin("\"exchange_order_id\":\"order-cancel-1\"", line), events)
        @test any(line -> occursin("\"event_type\":\"ORDER_FILLED\"", line) && occursin("\"exchange_order_id\":\"order-filled-1\"", line) && occursin("\"fee_currency\":\"USDT\"", line) && occursin("\"fee_amount\":", line) && !occursin("\"fee_amount\":null", line), events)
        @test any(line -> occursin("\"event_type\":\"ORDER_ACK\"", line) && occursin("\"exchange_order_id\":\"order-child-1\"", line) && occursin("\"correlation_id\":\"order-ack-1\"", line) && !occursin("\"parent_event_id\":null", line), events)
        @test any(line -> occursin("\"event_type\":\"ORDER_FILLED\"", line) && occursin("\"exchange_order_id\":\"order-filled-signal-1\"", line) && occursin("\"strategy_engine\":\"classifier\"", line) && occursin("\"strategy_config_ref\":\"trenddetector:test\"", line) && occursin("\"signal_label\":\"longbuy\"", line), events)
        @test any(line -> occursin("\"event_type\":\"ORDER_FILLED\"", line) && occursin("\"exchange_order_id\":\"order-filled-signal-1\"", line) && occursin("\"slippage_bps\":", line) && !occursin("\"slippage_bps\":null", line), events)
        @test any(line -> occursin("\"event_type\":\"ORDER_FILLED\"", line) && occursin("\"exchange_order_id\":\"order-filled-signal-1\"", line) && occursin("leg_group_id=grp-1", line) && occursin("leg_label=take_profit", line), events)
    finally
        if isnothing(oldroot)
            delete!(ENV, "CTS_TRADELOG_ROOT")
        else
            ENV["CTS_TRADELOG_ROOT"] = oldroot
        end
        rm(tmpdir; force=true, recursive=true)
    end
end

end
