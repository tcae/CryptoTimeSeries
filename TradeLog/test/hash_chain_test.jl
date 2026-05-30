"""
Unit tests for TradeAudit hash-chain integrity and manifest validation.
"""

using Dates, JSON3, SHA, Test, TradeAudit, UUIDs, EnvConfig

const UTC = Dates.UTC
const Second = Dates.Second

@testset "Hash-chain integrity" begin
    # Setup test audit folder
    test_root = mktempdir()
    
    @testset "compute event hash" begin
        event = TradeAudit.AuditEventRow(
            event_type=TradeAudit.ORDER_SUBMITTED,
            exchange="test_exchange",
            symbol="BTC/USDT"
        )
        hash = TradeAudit.computeevenhash(event)
        @test !isempty(hash)
        @test length(hash) == 64  # SHA256 hex is 64 chars
        @test all(c in "0123456789abcdef" for c in hash)
    end
    
    @testset "manifest file path" begin
        event = TradeAudit.AuditEventRow(
            event_type=TradeAudit.ORDER_SUBMITTED,
            exchange="test_exchange",
            symbol="BTC/USDT"
        )
        manifest_path = TradeAudit.manifestfile(event; root=test_root)
        @test contains(manifest_path, "manifest.json")
        @test contains(manifest_path, "test_exchange")
    end
    
    @testset "read/write manifest" begin
        test_root_rw = mktempdir()
        event = TradeAudit.AuditEventRow(
            event_type=TradeAudit.ORDER_SUBMITTED,
            exchange="test_exchange_rw",
            symbol="BTC/USDT"
        )
        manifest_path = TradeAudit.manifestfile(event; root=test_root_rw)
        
        # Read empty manifest
        manifest = TradeAudit.readmanifest(manifest_path)
        @test haskey(manifest, "date")
        @test haskey(manifest, "event_hashes")
        @test isempty(manifest["event_hashes"])
        
        # Write updated manifest
        event_hash = TradeAudit.computeevenhash(event)
        manifest["event_hashes"] = [Dict("event_id" => event.event_id, "hash" => event_hash)]
        TradeAudit.writemanifest(manifest_path, manifest)
        
        # Read back
        read_manifest = TradeAudit.readmanifest(manifest_path)
        @test !isempty(read_manifest["event_hashes"])
        @test read_manifest["event_hashes"][1]["event_id"] == event.event_id
    end
    
    @testset "prior event hash" begin
        test_root_prior = mktempdir()
        event = TradeAudit.AuditEventRow(
            event_type=TradeAudit.ORDER_SUBMITTED,
            exchange="test_exchange_prior",
            symbol="BTC/USDT"
        )
        manifest_path = TradeAudit.manifestfile(event; root=test_root_prior)
        
        # Empty manifest
        prior = TradeAudit.priorevenhash(manifest_path)
        @test isnothing(prior)
        
        # Add event hash
        manifest = TradeAudit.readmanifest(manifest_path)
        event_hash = TradeAudit.computeevenhash(event)
        manifest["event_hashes"] = [Dict("event_id" => event.event_id, "hash" => event_hash)]
        TradeAudit.writemanifest(manifest_path, manifest)
        
        prior = TradeAudit.priorevenhash(manifest_path)
        @test !isnothing(prior)
        @test prior == event_hash
    end
    
    @testset "write event with hash" begin
        # Suppress actual file writes unless audit enabled
        old_audit = get(ENV, "CTS_AUDIT_ENABLED", "false")
        ENV["CTS_AUDIT_ENABLED"] = "true"
        
        try
            event = TradeAudit.AuditEventRow(
                event_type=TradeAudit.ORDER_SUBMITTED,
                event_time_utc=Dates.now(UTC),
                source_module="test_module",
                environment="test",
                run_mode="simulation",
                exchange="test_exchange",
                symbol="BTC/USDT"
            )
            
            path = TradeAudit.writeeventwithhash(event; root=test_root)
            @test !isempty(path)
            @test isfile(path)
            
            # Verify manifest was created
            manifest_path = TradeAudit.manifestfile(event; root=test_root)
            @test isfile(manifest_path)
            manifest = TradeAudit.readmanifest(manifest_path)
            @test !isempty(manifest["event_hashes"])
        finally
            ENV["CTS_AUDIT_ENABLED"] = old_audit
        end
    end
    
    @testset "hash chain validation" begin
        old_audit = get(ENV, "CTS_AUDIT_ENABLED", "false")
        ENV["CTS_AUDIT_ENABLED"] = "true"
        
        try
            # Create multiple events
            events = [
                TradeAudit.AuditEventRow(
                    event_type=TradeAudit.ORDER_SUBMITTED,
                    event_time_utc=Dates.now(UTC),
                    source_module="test_module",
                    environment="test",
                    run_mode="simulation",
                    exchange="test_exchange",
                    symbol="BTC/USDT"
                ),
                TradeAudit.AuditEventRow(
                    event_type=TradeAudit.ORDER_ACK,
                    event_time_utc=Dates.now(UTC) + Second(1),
                    source_module="test_module",
                    environment="test",
                    run_mode="simulation",
                    exchange="test_exchange",
                    symbol="BTC/USDT"
                )
            ]
            
            for event in events
                TradeAudit.writeeventwithhash(event; root=test_root)
            end
            
            # Validate chain
            folder = TradeAudit.auditfolder(events[1]; root=test_root)
            is_valid, issues = TradeAudit.validatehashchain(folder)
            @test is_valid
            @test isempty(issues)
        finally
            ENV["CTS_AUDIT_ENABLED"] = old_audit
        end
    end
    
    @testset "detect corrupted event" begin
        old_audit = get(ENV, "CTS_AUDIT_ENABLED", "false")
        ENV["CTS_AUDIT_ENABLED"] = "true"
        
        try
            event = TradeAudit.AuditEventRow(
                event_type=TradeAudit.ORDER_SUBMITTED,
                event_time_utc=Dates.now(UTC),
                source_module="test_module",
                environment="test",
                run_mode="simulation",
                exchange="test_exchange",
                symbol="BTC/USDT"
            )
            
            path = TradeAudit.writeeventwithhash(event; root=test_root)
            
            # Corrupt the event by modifying a field
            lines = readlines(path)
            @test !isempty(lines)
            json_obj = JSON3.read(lines[end])
            # Convert to Dict for modification
            payload = Dict{String, Any}()
            for (k, v) in pairs(json_obj)
                payload[String(k)] = v
            end
            payload["symbol"] = "ETH/USDT"  # Modify field
            lines[end] = JSON3.write(payload)
            open(path, "w") do io
                for line in lines
                    write(io, line)
                    write(io, "\n")
                end
            end
            
            # Validate should detect corruption
            folder = TradeAudit.auditfolder(event; root=test_root)
            is_valid, issues = TradeAudit.validatehashchain(folder)
            @test !is_valid
            @test !isempty(issues)
            @test any(contains(issue, "Hash mismatch") for issue in issues)
        finally
            ENV["CTS_AUDIT_ENABLED"] = old_audit
        end
    end
end

println("✓ Hash-chain integrity tests passed")
