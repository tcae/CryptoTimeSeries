# Async entry point tests — verify async_start! returns a Task and allows external control.
using Test, Dates, DataFrames
using EnvConfig, Trade, CryptoXch

EnvConfig.init(EnvConfig.test)

@testset "Async entry point: async_start! returns Task (non-blocking)" begin
    # async_start! should return immediately with a Task, and support default args.
    
    # TestOhlcv synthetic data starts in 2025; keep the replay window within coverage.
    timestamp = DateTime("2025-01-05T10:00:00")
    tc = Trade.TradeCache(
        xc=CryptoXch.XchCache(startdt=timestamp + Minute(1), enddt=timestamp),
        trademode=Trade.notrade,
    )
    tc.cfg = DataFrame(
        basecoin=String[], classifieraccepted=Bool[], minquotevol=Bool[],
        continuousminvol=Bool[], buyenabled=Bool[], sellenabled=Bool[],
        whitelisted=Bool[], datetime=DateTime[],
    )

    @testset "API supports both call forms" begin
        @test hasmethod(Trade.async_start!, Tuple{Trade.TradeCache})
        @test hasmethod(Trade.async_start!, Tuple{Trade.TradeCache}, (:skip_init,))
    end

    @testset "skip_init=true returns Task immediately (non-blocking)" begin
        task = Trade.async_start!(tc; skip_init=true)
        @test isa(task, Task)
        wait(task)
        @test istaskdone(task)
    end
end

@testset "Async control: external task controls running loop" begin
    # Real integration test: control an actually progressing tradeloop from external context.
    # We verify progression halts in paused state and continues after resume.
    function wait_until(predicate::Function; timeout_s::Float64=2.0, step_s::Float64=0.01)
        deadline = time() + timeout_s
        while time() < deadline
            predicate() && return true
            sleep(step_s)
        end
        return predicate()
    end

    timestamp = DateTime("2025-01-05T10:00:00")
    tc = Trade.TradeCache(
        xc=CryptoXch.XchCache(startdt=timestamp, enddt=timestamp + Year(5)),
        trademode=Trade.notrade,
    )
    tc.cfg = DataFrame(
        basecoin=String[], classifieraccepted=Bool[], minquotevol=Bool[],
        continuousminvol=Bool[], buyenabled=Bool[], sellenabled=Bool[],
        whitelisted=Bool[], datetime=DateTime[],
    )
    
    @test Trade.loopstate(tc) == Trade.loop_idle

    # Start async loop in background
    task = Trade.async_start!(tc; skip_init=true)
    @test isa(task, Task)

    @test wait_until(() -> Trade.loopstate(tc) == Trade.loop_running)
    @test wait_until(() -> !isnothing(tc.xc.currentdt))
    @test !istaskdone(task)

    Trade.pause!(tc)
    @test wait_until(() -> Trade.loopstate(tc) == Trade.loop_paused)

    paused_dt0 = tc.xc.currentdt
    sleep(0.05)
    paused_dt1 = tc.xc.currentdt
    sleep(0.05)
    paused_dt = tc.xc.currentdt
    @test paused_dt1 == paused_dt
    @test paused_dt >= paused_dt0
    @test !istaskdone(task)

    # Ensure the resume predicate is false before we call resume!.
    @test tc.xc.currentdt == paused_dt

    Trade.resume!(tc)
    @test wait_until(() -> Trade.loopstate(tc) == Trade.loop_running)
    @test wait_until(() -> tc.xc.currentdt > paused_dt)

    @test !istaskdone(task)
    Trade.stop!(tc)
    @test wait_until(() -> Trade.loopstate(tc) in [Trade.loop_stopping, Trade.loop_stopped])
    @test wait_until(() -> istaskdone(task))
end