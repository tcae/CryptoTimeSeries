module TradeLogTest

using Dates, JSON3, Test
using TradeLog

function sampleevent()::TradeLog.AuditEventRow
    return TradeLog.AuditEventRow(
        event_id="evt-001",
        event_type=TradeLog.ORDER_SUBMITTED,
        event_time_utc=DateTime("2026-05-10T12:34:56"),
        created_at_utc=DateTime("2026-05-10T12:34:57"),
        source_module="Trade",
        environment="production",
        run_mode="live",
        run_id="run-001",
        correlation_id="corr-001",
        exchange="KrakenSpot",
        account_alias="krakenspot-tcae1",
        routing_role=TradeLog.routing_trade_exchange_spot,
        market_type=TradeLog.market_spot,
        asset_class=TradeLog.crypto,
        instrument_type=TradeLog.spot_pair,
        venue_instrument_type="spot",
        symbol="BTCUSDT",
        baseasset="BTC",
        quoteasset="USDT",
        client_order_id="client-001",
        exchange_order_id="exchange-001",
        side="Buy",
        order_type="Limit",
        time_in_force="GTC",
        status="New",
        requested_base_qty=0.5,
        requested_quote_qty=50000.0,
        requested_limit_price=100000.0,
        requested_notional=50000.0,
        strategy_engine="getgainsalgo",
        strategy_config_ref="cfg-001",
        signal_label="longbuy",
        signal_score=0.75,
        algorithm_version="v1",
    )
end

@testset "TradeLog tests" begin
    event = sampleevent()

    @test event.event_id == "evt-001"
    @test event.instrument_type == TradeLog.spot_pair

    payload = TradeLog.eventpayload(event)
    @test payload["event_type"] == "ORDER_SUBMITTED"
    @test payload["routing_role"] == "routing_trade_exchange_spot"
    @test payload["asset_class"] == "crypto"
    @test payload["event_time_utc"] == "2026-05-10T12:34:56.000Z"
    @test payload["run_mode"] == "live"
    @test payload["baseasset"] == "BTC"
    @test ismissing(payload["notes"])

    mktempdir() do root
        folder = TradeLog.auditfolder(event; root=root)
        @test endswith(folder, joinpath(
            "environment=production",
            "run_mode=live",
            "exchange=KrakenSpot",
            "account=krakenspot-tcae1",
            "asset_class=crypto",
            "instrument_type=spot_pair",
            "date=2026-05-10",
        ))

        path = TradeLog.writeevent(event; root=root)
        @test path == joinpath(folder, "events.jsonl")
        @test isfile(path)

        TradeLog.writeevent(TradeLog.AuditEventRow(event_id="evt-002", event_type=TradeLog.ORDER_ACK, event_time_utc=event.event_time_utc, created_at_utc=event.created_at_utc, source_module="CryptoXch", environment="production", run_mode="live", exchange="KrakenSpot", account_alias="krakenspot-tcae1", routing_role=TradeLog.routing_trade_exchange_spot, market_type=TradeLog.market_spot, asset_class=TradeLog.crypto, instrument_type=TradeLog.spot_pair, symbol="BTCUSDT"); root=root)

        lines = readlines(path)
        @test length(lines) == 2

        firstrow = JSON3.read(lines[1])
        secondrow = JSON3.read(lines[2])
        @test firstrow["event_id"] == "evt-001"
        @test secondrow["event_type"] == "ORDER_ACK"
    end

    old_enabled = get(ENV, "CTS_TRADELOG_ENABLED", nothing)
    old_sim_enabled = get(ENV, "CTS_TRADELOG_SIMULATION_ENABLED", nothing)
    try
        ENV["CTS_TRADELOG_ENABLED"] = "false"
        mktempdir() do root
            path = TradeLog.writeevent(event; root=root)
            @test path == ""
            @test isempty(readdir(root))
        end

        ENV["CTS_TRADELOG_ENABLED"] = "true"
        ENV["CTS_TRADELOG_SIMULATION_ENABLED"] = "false"

        simulation_event = TradeLog.AuditEventRow(
            event_id="evt-sim-001",
            event_type=TradeLog.ORDER_SUBMITTED,
            event_time_utc=DateTime("2026-05-10T12:34:56"),
            created_at_utc=DateTime("2026-05-10T12:34:57"),
            source_module="Trade",
            environment="test",
            run_mode="simulation",
            exchange="KrakenSpot",
            account_alias="krakenspot-tcae1",
            routing_role=TradeLog.routing_trade_exchange_spot,
            market_type=TradeLog.market_spot,
            asset_class=TradeLog.crypto,
            instrument_type=TradeLog.spot_pair,
            symbol="BTCUSDT",
        )
        mktempdir() do root
            path = TradeLog.writeevent(simulation_event; root=root)
            @test path == ""
            @test isempty(readdir(root))
        end
    finally
        if isnothing(old_enabled)
            delete!(ENV, "CTS_TRADELOG_ENABLED")
        else
            ENV["CTS_TRADELOG_ENABLED"] = old_enabled
        end
        if isnothing(old_sim_enabled)
            delete!(ENV, "CTS_TRADELOG_SIMULATION_ENABLED")
        else
            ENV["CTS_TRADELOG_SIMULATION_ENABLED"] = old_sim_enabled
        end
    end
end

include("hash_chain_test.jl")
include("arrow_export_test.jl")

end