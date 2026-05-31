"""
Unit tests for TradeLog Arrow export functionality.
"""

using Arrow, Dates, JSON3, Test, TradeLog, UUIDs, EnvConfig

const UTC = Dates.UTC

@testset "Arrow export functionality" begin
    # Setup test audit folder
    test_root = mktempdir()
    old_tradelog = get(ENV, "CTS_TRADELOG_ENABLED", "false")
    ENV["CTS_TRADELOG_ENABLED"] = "true"
    
    try
        @testset "read JSONL audit events" begin
            test_root_jsonl = mktempdir()
            # Create test events
            events_to_write = [
                TradeLog.AuditEventRow(
                    event_type=TradeLog.ORDER_SUBMITTED,
                    event_time_utc=Dates.now(UTC),
                    source_module="test_module",
                    environment="test",
                    run_mode="simulation",
                    exchange="test_exchange",
                    symbol="BTC/USDT"
                ),
                TradeLog.AuditEventRow(
                    event_type=TradeLog.ORDER_ACK,
                    event_time_utc=Dates.now(UTC) + Dates.Second(1),
                    source_module="test_module",
                    environment="test",
                    run_mode="simulation",
                    exchange="test_exchange",
                    symbol="BTC/USDT",
                    status="ACK"
                )
            ]
            
            for event in events_to_write
                TradeLog.writeeventwithhash(event; root=test_root_jsonl)
            end
            
            # Read events back
            jsonl_path = TradeLog.auditfile(events_to_write[1]; root=test_root_jsonl)
            events = TradeLog.readjsonlauditevents(jsonl_path)
            @test length(events) == 2
            @test events[1]["symbol"] == "BTC/USDT"
            @test events[2]["status"] == "ACK"
        end
        
        @testset "arrow export file path" begin
            event = TradeLog.AuditEventRow(
                event_type=TradeLog.ORDER_SUBMITTED,
                exchange="test_exchange",
                symbol="BTC/USDT"
            )
            arrow_path = TradeLog.arrowexportfile(event; root=test_root)
            @test contains(arrow_path, "events.arrow")
            @test contains(arrow_path, "test_exchange")
        end
        
        @testset "write arrow export" begin
            test_root_arrow = mktempdir()
            # Create test events
            num_events = 5
            for i in 1:num_events
                event = TradeLog.AuditEventRow(
                    event_type=i % 2 == 0 ? TradeLog.ORDER_ACK : TradeLog.ORDER_SUBMITTED,
                    event_time_utc=Dates.now(UTC) + Dates.Second(i-1),
                    source_module="test_module",
                    environment="test",
                    run_mode="simulation",
                    exchange="test_exchange_arrow",
                    symbol="BTC/USDT",
                    requested_base_qty=1.0 * i
                )
                TradeLog.writeeventwithhash(event; root=test_root_arrow)
            end
            
            # Export to Arrow (use same environment/run_mode as the events)
            jsonl_event = TradeLog.AuditEventRow(
                environment="test",
                run_mode="simulation",
                exchange="test_exchange_arrow",
                symbol="BTC/USDT"
            )
            jsonl_path = TradeLog.auditfile(jsonl_event; root=test_root_arrow)
            
            arrow_path = TradeLog.writearrowauditexport(jsonl_path)
            @test !isempty(arrow_path)
            @test isfile(arrow_path)
        end
        
        @testset "read arrow export" begin
            test_root_read = mktempdir()
            # Create test events
            num_events = 3
            for i in 1:num_events
                event = TradeLog.AuditEventRow(
                    event_type=TradeLog.ORDER_SUBMITTED,
                    event_time_utc=Dates.now(UTC) + Dates.Second(i-1),
                    source_module="test_module",
                    environment="test",
                    run_mode="simulation",
                    exchange="test_exchange_read",
                    symbol="BTC/USDT",
                    requested_quote_qty=1000.0 * i
                )
                TradeLog.writeeventwithhash(event; root=test_root_read)
            end
            
            # Export to Arrow (use same environment/run_mode)
            jsonl_event = TradeLog.AuditEventRow(
                environment="test",
                run_mode="simulation",
                exchange="test_exchange_read",
                symbol="BTC/USDT"
            )
            jsonl_path = TradeLog.auditfile(jsonl_event; root=test_root_read)
            
            arrow_path = TradeLog.writearrowauditexport(jsonl_path)
            
            # Read back Arrow table
            table = TradeLog.readarrowauditexport(arrow_path)
            @test !isempty(table)
            # Arrow table successfully read and contains data
        end
        
    finally
        ENV["CTS_TRADELOG_ENABLED"] = old_tradelog
    end
end

println("✓ Arrow export tests passed")
