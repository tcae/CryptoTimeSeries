"""
Unit tests for TradeLog compatibility hash/manifest helpers.

TradeLog intentionally relaxes hash-chain guarantees, so these tests verify
compatibility surfaces and current expected behavior.
"""

using Dates, JSON3, Test, TradeLog

const UTC = Dates.UTC

@testset "TradeLog hash compatibility" begin
    test_root = mktempdir()

    @testset "compute compatibility hash" begin
        event = TradeLog.AuditEventRow(
            event_type=TradeLog.ORDER_SUBMITTED,
            exchange="test_exchange",
            symbol="BTC/USDT",
        )
        h = TradeLog.computeevenhash(event)
        @test !isempty(h)
        @test h isa String
    end

    @testset "manifest helpers roundtrip" begin
        event = TradeLog.AuditEventRow(
            event_type=TradeLog.ORDER_SUBMITTED,
            exchange="test_exchange_rw",
            symbol="BTC/USDT",
        )
        manifest_path = TradeLog.manifestfile(event; root=test_root)
        manifest = TradeLog.readmanifest(manifest_path)
        @test haskey(manifest, "event_hashes")

        event_hash = TradeLog.computeevenhash(event)
        manifest["event_hashes"] = [Dict("event_id" => event.event_id, "hash" => event_hash)]
        TradeLog.writemanifest(manifest_path, manifest)

        prior = TradeLog.priorevenhash(manifest_path)
        @test prior == event_hash
    end

    @testset "writeeventwithhash compatibility path" begin
        old_tradelog = get(ENV, "CTS_TRADELOG_ENABLED", "false")
        ENV["CTS_TRADELOG_ENABLED"] = "true"
        try
            event = TradeLog.AuditEventRow(
                event_type=TradeLog.ORDER_SUBMITTED,
                event_time_utc=Dates.now(UTC),
                source_module="test_module",
                environment="test",
                run_mode="simulation",
                exchange="test_exchange",
                symbol="BTC/USDT",
            )
            path = TradeLog.writeeventwithhash(event; root=test_root)
            @test !isempty(path)
            @test isfile(path)
        finally
            ENV["CTS_TRADELOG_ENABLED"] = old_tradelog
        end
    end

    @testset "validatehashchain is explicitly disabled" begin
        folder = mktempdir()
        is_valid, issues = TradeLog.validatehashchain(folder)
        @test is_valid
        @test !isempty(issues)
        @test any(contains(issue, "disabled") for issue in issues)
    end

    @testset "corrupted row does not trigger strict-chain failure" begin
        old_tradelog = get(ENV, "CTS_TRADELOG_ENABLED", "false")
        ENV["CTS_TRADELOG_ENABLED"] = "true"
        try
            event = TradeLog.AuditEventRow(
                event_type=TradeLog.ORDER_SUBMITTED,
                event_time_utc=Dates.now(UTC),
                source_module="test_module",
                environment="test",
                run_mode="simulation",
                exchange="test_exchange",
                symbol="BTC/USDT",
            )
            path = TradeLog.writeeventwithhash(event; root=test_root)

            lines = readlines(path)
            @test !isempty(lines)
            payload = Dict{String, Any}(String(k) => v for (k, v) in pairs(JSON3.read(lines[end])))
            payload["symbol"] = "ETH/USDT"
            lines[end] = JSON3.write(payload)
            open(path, "w") do io
                for line in lines
                    write(io, line)
                    write(io, "\n")
                end
            end

            folder = TradeLog.auditfolder(event; root=test_root)
            is_valid, issues = TradeLog.validatehashchain(folder)
            @test is_valid
            @test any(contains(issue, "disabled") for issue in issues)
        finally
            ENV["CTS_TRADELOG_ENABLED"] = old_tradelog
        end
    end
end
