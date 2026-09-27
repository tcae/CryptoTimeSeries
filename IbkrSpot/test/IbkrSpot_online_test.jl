using IbkrSpot, Test, Sockets, Jib, Dates

# Read-only tests against a running TWS / IB Gateway.
# Port override: IBKR_PORT=7497 (paper) or 7496 (live). Defaults to the execution config port.
#
# These tests never place, amend or cancel an order, so they are safe against a live session.

const TWS_HANDSHAKE_TIMEOUT_SECONDS = 5.0

"""
Check that `host:port` accepts a TCP connection and completes the Jib API version handshake.

Return `nothing` when the TWS handshake succeeds, otherwise return a diagnostic string.
The handshake socket is closed on success, failure, and timeout.
"""
function _tws_connection_issue(host::AbstractString, port::Int)::Union{Nothing,String}
    socket = try
        Sockets.connect(host, port)
    catch exception
        return "TCP connection failed: $(sprint(showerror, exception))"
    end

    timed_out = Ref(false)
    timer = Timer(TWS_HANDSHAKE_TIMEOUT_SECONDS) do _
        timed_out[] = true
        isopen(socket) && close(socket)
    end

    try
        minimum_version, maximum_version = Jib.Client.Version .|> (typemin, typemax) .|> Int
        handshake = Jib.Client.buffer(true)
        print(handshake, minimum_version == maximum_version ? "v$(minimum_version)" : "v$(minimum_version)..$(maximum_version)")
        Jib.Client.write_one(socket, handshake)

        server_version, _ = Jib.Client.read_init(socket)
        minimum_version <= server_version <= maximum_version ||
            return "TWS API version $(server_version) is outside the supported range $(minimum_version)..$(maximum_version)"
        return nothing
    catch exception
        if timed_out[]
            return "TWS API handshake timed out after $(TWS_HANDSHAKE_TIMEOUT_SECONDS) seconds"
        end
        return "Jib API handshake failed: $(sprint(showerror, exception))"
    finally
        close(timer)
        isopen(socket) && close(socket)
    end
end

@testset "IbkrSpot online read-only tests" begin
    connection = IbkrSpot._connectionconfig()
    port = haskey(ENV, "IBKR_PORT") ? parse(Int, ENV["IBKR_PORT"]) : connection.port
    @assert 1 <= port <= 65535 "IBKR_PORT=$(port) must be between 1 and 65535"

    issue = _tws_connection_issue(connection.host, port)
    if isnothing(issue)
        clientid = 10_000 + mod(getpid(), 1_000_000)
        cache = IbkrSpot.IbkrSpotCache(port=port, clientid=clientid)
        try
            @test IbkrSpot.servertime(cache) isa DateTime
            info = IbkrSpot.symbolinfo(cache, "AAPLUSD")
            @test !isnothing(info)
            @test IbkrSpot.validsymbol(cache, info)
            @test !IbkrSpot.validsymbol(cache, "NOTAREALSTOCKUSD")
            scan = IbkrSpot.get24h(cache; symbols=["AAPLUSD", "MSFTUSD"])
            @test scan.symbol == ["AAPLUSD", "MSFTUSD"]
            @test all(volume -> volume >= cache.minimum_daily_usd_volume, scan.quotevolume24h)
        finally
            IbkrSpot.disconnect!(cache)
        end
    else
        @warn "Skipping IbkrSpot tests that require TWS" host=connection.host port=port reason=issue
        @test_skip "TWS-dependent tests require a successful TCP connection and Jib API handshake"
    end
end
