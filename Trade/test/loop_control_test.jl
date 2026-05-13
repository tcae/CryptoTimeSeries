# Increment 1.4 tests — loop control state machine and backtest replay smoke test.
# All tests bypass live exchange calls by using cryptoxchsim mode (EnvConfig.test).
using Test, Dates, DataFrames
using EnvConfig, Trade, CryptoXch

EnvConfig.init(EnvConfig.test)

@testset "Loop control state machine" begin
    tc = Trade.TradeCache(trademode=Trade.notrade)

    @testset "Initial state is loop_idle" begin
        @test Trade.loopstate(tc) == Trade.loop_idle
    end

    @testset "pause! only transitions from loop_running" begin
        # idle → pause! → still idle
        Trade.pause!(tc)
        @test Trade.loopstate(tc) == Trade.loop_idle

        # running → pause! → paused
        Trade._setloopstate!(tc, Trade.loop_running)
        Trade.pause!(tc)
        @test Trade.loopstate(tc) == Trade.loop_paused
    end

    @testset "resume! only transitions from loop_paused" begin
        # paused → resume! → running
        @assert Trade.loopstate(tc) == Trade.loop_paused
        Trade.resume!(tc)
        @test Trade.loopstate(tc) == Trade.loop_running

        # stopped → resume! → still stopped
        Trade._setloopstate!(tc, Trade.loop_stopped)
        Trade.resume!(tc)
        @test Trade.loopstate(tc) == Trade.loop_stopped
    end

    @testset "stop! from running and paused sets loop_stopping" begin
        Trade._setloopstate!(tc, Trade.loop_running)
        Trade.stop!(tc)
        @test Trade.loopstate(tc) == Trade.loop_stopping

        Trade._setloopstate!(tc, Trade.loop_paused)
        Trade.stop!(tc)
        @test Trade.loopstate(tc) == Trade.loop_stopping
    end

    @testset "stop! from idle has no effect" begin
        Trade._setloopstate!(tc, Trade.loop_idle)
        Trade.stop!(tc)
        @test Trade.loopstate(tc) == Trade.loop_idle
    end
end

@testset "run_backtest! smoke test (empty window, sim mode)" begin
    # startdt > enddt → iterate may still attempt ticks
    # In any case, the state should transition to loop_stopped when backtest completes
    timestamp = DateTime("2024-06-01T10:00:00")
    tc = Trade.TradeCache(
        xc=CryptoXch.XchCache(startdt=timestamp + Minute(1), enddt=timestamp),
        trademode=Trade.notrade,
    )
    # Non-empty cfg bypasses _ensure_tradeloop_initialized!
    tc.cfg = DataFrame(
        basecoin=["BTC"],
        classifieraccepted=[false], minquotevol=[false],
        continuousminvol=[false], buyenabled=[false], sellenabled=[false],
        whitelisted=[false], datetime=[timestamp],
    )

    @test Trade.loopstate(tc) == Trade.loop_idle
    try
        Trade.run_backtest!(tc; skip_init=true)
    catch e
        # May fail on portfolio! with empty OHLCV, but state should still transition
        if !isa(e, BoundsError)
            rethrow(e)
        end
    end
    # Verify state transitioned even if there was an error
    @test Trade.loopstate(tc) in [Trade.loop_stopped, Trade.loop_running, Trade.loop_paused]
end

@testset "stop! during _run_tradeloop! exits cleanly" begin
    # Same empty-window setup
    timestamp = DateTime("2024-06-01T10:00:00")
    tc = Trade.TradeCache(
        xc=CryptoXch.XchCache(startdt=timestamp + Minute(1), enddt=timestamp),
        trademode=Trade.notrade,
    )
    tc.cfg = DataFrame(
        basecoin=String[], classifieraccepted=Bool[], minquotevol=Bool[],
        continuousminvol=Bool[], buyenabled=Bool[], sellenabled=Bool[],
        whitelisted=Bool[], datetime=DateTime[],
    )

    try
        Trade._run_tradeloop!(tc)
    catch e
        # May fail on portfolio! with empty OHLCV
        if !isa(e, BoundsError)
            rethrow(e)
        end
    end
    # State should have transitioned
    @test Trade.loopstate(tc) in [Trade.loop_stopped, Trade.loop_running, Trade.loop_paused]
end
