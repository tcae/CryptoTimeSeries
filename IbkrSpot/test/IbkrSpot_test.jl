using IbkrSpot, Test, Dates

# Offline tests: no TWS connection required.

@testset "IbkrSpot execution config" begin
    cfg = IbkrSpot.executionconfig()
    @test String(cfg["exchange"]) == "IbkrSpot"

    conn = IbkrSpot._connectionconfig()
    @test conn.host == "127.0.0.1"
    @test conn.port in (7496, 7497, 4001, 4002)
    @test conn.clientid > 0
    @test conn.currency == "USD"
    @test conn.exchange_route == "SMART"

    long = IbkrSpot._executionorderspec(:long)
    @test long.side == :long
    @test long.instrument == "stock"
    @test long.max_quote > 0

    short = IbkrSpot._executionorderspec(:short)
    @test short.side == :short
    # IBKR nets positions per contract; a short is a margin sale of the same stock contract.
    @test short.instrument == "stock_margin"
    @test short.leverage >= 1
    @test short.max_quote > 0

    @test_throws ErrorException IbkrSpot._executionorderspec(:sideways)
end

@testset "IbkrSpot request correlation" begin
    @testset "ids are unique and increasing" begin
        ids = [IbkrSpot._nextreqid() for _ in 1:100]
        @test length(unique(ids)) == 100
        @test issorted(ids)
    end

    @testset "collect then complete returns buffered values" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._push!(reqid, "a")
        IbkrSpot._push!(reqid, "b")
        IbkrSpot._complete!(reqid)
        @test IbkrSpot._await(reqid, "unit test") == ["a", "b"]
    end

    @testset "completion with no values returns empty" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._complete!(reqid)
        @test isempty(IbkrSpot._await(reqid, "unit test"))
    end

    @testset "unsolicited ids are ignored" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._push!(reqid + 1, "stray")
        IbkrSpot._push!(reqid, "kept")
        IbkrSpot._complete!(reqid)
        @test IbkrSpot._await(reqid, "unit test") == ["kept"]
    end

    @testset "TWS request error throws instead of returning partial data" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._push!(reqid, "partial")
        IbkrSpot._fail!(reqid, "code=200 No security definition found")
        @test_throws ErrorException IbkrSpot._await(reqid, "contract details")
    end

    @testset "missing response times out instead of hanging" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        @test_throws ErrorException IbkrSpot._await(reqid, "unit test"; timeout=Dates.Millisecond(150))
    end

    @testset "await releases the buffer" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._complete!(reqid)
        IbkrSpot._await(reqid, "unit test")
        # a second await on a released id must not silently succeed with stale data
        @test_throws ErrorException IbkrSpot._await(reqid, "unit test"; timeout=Dates.Millisecond(100))
    end
end

@testset "IbkrSpot order id allocation" begin
    # Without a connection TWS has not supplied nextValidId, so allocation must fail fast
    # rather than invent an id that the venue would reject.
    if IbkrSpot._nextvalidid[] == 0
        @test_throws AssertionError IbkrSpot._takeorderid()
    else
        first = IbkrSpot._takeorderid()
        @test IbkrSpot._takeorderid() == first + 1
    end
end
