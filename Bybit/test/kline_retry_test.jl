using Test, Bybit

"""Wrap a Bybit body into the `(body, headers)` response shape of public requests."""
klineresponse(code, message; headers=Dict{String,String}()) =
    (body=Dict("retCode" => code, "retMsg" => message, "result" => Dict("list" => [])), headers=headers)

@testset "Kline API errors are not missing history" begin
    success = klineresponse(0, "OK")
    transient = klineresponse(10016, "internal error")
    permanent = klineresponse(10001, "invalid parameter")
    @test Bybit._checkedklinerequest(() -> success; wait=delay -> error("unexpected retry")) === success.body

    calls = Ref(0)
    delays = Float64[]
    request = () -> (calls[] += 1; calls[] < 4 ? transient : success)
    @test Bybit._checkedklinerequest(request; wait=delay -> push!(delays, delay)) === success.body
    @test calls[] == 4
    @test delays == [1, 2, 4]

    calls[] = 0
    empty!(delays)
    failing = () -> (calls[] += 1; transient)
    @test_throws r"retCode=10016.*attempt=12" Bybit._checkedklinerequest(failing; wait=delay -> push!(delays, delay))
    @test calls[] == Bybit.KLINE_MAXATTEMPTS
    @test delays == [1, 2, 4, 8, 16, 32, 64, 128, 256, 300, 300]

    @test_throws r"retCode=10001.*attempt=1" Bybit._checkedklinerequest(() -> permanent; wait=delay -> error("unexpected retry"))
    @test_throws InterruptException Bybit._checkedklinerequest(() -> throw(InterruptException()))
    @test_throws InterruptException Bybit._checkedklinerequest(() -> transient; wait=delay -> throw(InterruptException()))
end

@testset "Kline transport exceptions back off exponentially" begin
    calls = Ref(0)
    delays = Float64[]
    flaky = () -> (calls[] += 1; calls[] < 3 ? error("transport failure") : klineresponse(0, "OK"))
    @test Bybit._checkedklinerequest(flaky; wait=delay -> push!(delays, delay))["retCode"] == 0
    @test delays == [1, 2]
    empty!(delays)
    @test_throws ErrorException Bybit._checkedklinerequest(() -> error("server down"); wait=delay -> push!(delays, delay))
    @test length(delays) == Bybit.KLINE_MAXATTEMPTS - 1
end

@testset "Kline rate limit waits for reset timestamp" begin
    nowms = 1_700_000_000_000
    limited = klineresponse(10006, "Too many visits. Exceeded the API Rate Limit.";
        headers=Dict("x-bapi-limit" => "600", "x-bapi-limit-status" => "0",
            "x-bapi-limit-reset-timestamp" => string(nowms + 2_500)))
    calls = Ref(0)
    delays = Float64[]
    request = () -> (calls[] += 1; calls[] == 1 ? limited : klineresponse(0, "OK"))
    @test Bybit._checkedklinerequest(request; wait=delay -> push!(delays, delay), clockms=() -> nowms)["retCode"] == 0
    @test delays == [2.5]
    @test Bybit._ratelimitdelay(limited.headers, nowms + 10_000, 4.0) == 1.0
    @test Bybit._ratelimitdelay(Dict{String,String}(), nowms, 4.0) == 4.0
end
