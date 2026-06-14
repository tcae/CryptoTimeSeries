using Test, DataFrames
using EnvConfig, Xch, Trade, TradeLog

@testset "Trade portfolio audit snapshots" begin
    oldauditroot = get(ENV, "CTS_TRADELOG_ROOT", nothing)
    tmpdir = mktempdir()
    try
        ENV["CTS_TRADELOG_ROOT"] = tmpdir
        EnvConfig.init(test)

        xc = Xch.XchCache()
        tc = Trade.TradeCache(xc=xc)
        assets = DataFrame(
            coin=["USDT", "BTC"],
            locked=Float32[0, 0],
            free=Float32[1500, 0.25],
            borrowed=Float32[0, 0],
            accruedinterest=Float32[0, 0],
            usdtprice=Float32[1, 60000],
            usdtvalue=Float32[1500, 15000],
        )

        Trade._writeportfoliosnapshot!(tc, assets; source_module="TradeTest")

        snapshot = TradeLog.AuditEventRow(
            event_type=TradeLog.PORTFOLIO_SNAPSHOT,
            environment=string(Symbol(EnvConfig.configmode)),
            run_mode=Xch.auditrunmode(xc),
            exchange=Xch.exchange(xc),
            account_alias=Xch.exchange(xc),
            asset_class=TradeLog.crypto,
            instrument_type=TradeLog.spot_pair,
        )
        auditpath = TradeLog.auditfile(snapshot)
        @test isfile(auditpath)
        events = readlines(auditpath)
        @test length(events) == 2
        @test all(line -> occursin("\"event_type\":\"PORTFOLIO_SNAPSHOT\"", line), events)
        @test any(line -> occursin("\"symbol\":\"USDT\"", line), events)
        @test any(line -> occursin("\"symbol\":\"BTC\"", line), events)
        @test any(line -> occursin("\"cash_after\":1500.0", line), events)
        @test all(line -> occursin("\"portfolio_value_after\":16500.0", line), events)
        @test all(line -> occursin("\"run_mode\":\"simulation\"", line), events)
        @test all(line -> occursin("\"run_id\":\"", line), events)
        @test any(line -> occursin("\"notes\":\"asset=USDT", line) && occursin("rows=2", line) && occursin("simmode=", line), events)
        @test any(line -> occursin("\"notes\":\"asset=BTC", line) && occursin("rows=2", line) && occursin("simmode=", line), events)
    finally
        if isnothing(oldauditroot)
            delete!(ENV, "CTS_TRADELOG_ROOT")
        else
            ENV["CTS_TRADELOG_ROOT"] = oldauditroot
        end
        rm(tmpdir; force=true, recursive=true)
    end
end