using Test, Bybit

@testset "Kline API errors are not missing history" begin
    success = Dict("retCode" => 0, "retMsg" => "OK", "result" => Dict("list" => []))
    transient = Dict("retCode" => 10016, "retMsg" => "internal error", "result" => Dict())
    permanent = Dict("retCode" => 10001, "retMsg" => "invalid parameter", "result" => Dict())
    @test Bybit._checkedklinerequest(() -> success; wait=delay -> error("unexpected retry")) === success
    calls = Ref(0)
    delays = Int[]
    request = () -> begin
        calls[] += 1
        calls[] < 4 ? transient : success
    end
    @test Bybit._checkedklinerequest(request; wait=delay -> push!(delays, delay)) === success
    @test calls[] == 4
    @test delays == [1, 2, 4]
    calls[] = 0
    empty!(delays)
    failing = () -> begin
        calls[] += 1
        transient
    end
    @test_throws r"retCode=10016.*attempt=4" Bybit._checkedklinerequest(failing; wait=delay -> push!(delays, delay))
    @test calls[] == 4
    @test delays == [1, 2, 4]
    @test_throws r"retCode=10001.*attempt=1" Bybit._checkedklinerequest(() -> permanent; wait=delay -> error("unexpected retry"))
    @test_throws InterruptException Bybit._checkedklinerequest(() -> throw(InterruptException()))
    @test_throws InterruptException Bybit._checkedklinerequest(() -> transient; wait=delay -> throw(InterruptException()))
    @test_throws ErrorException Bybit._checkedklinerequest(() -> error("transport failure"))
end
