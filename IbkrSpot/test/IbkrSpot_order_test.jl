using IbkrSpot, Test

# These tests transmit a real minimum-quantity order and therefore require explicit opt-in.
# The limit price is deliberately distant from the current bid and the order is canceled.
#
# Enable with: IBKR_ORDER_TESTS=true IBKR_PORT=7496

const IBKR_API_PORTS = (7496, 7497, 4001, 4002)

@testset "IbkrSpot order lifecycle tests" begin
    enabled = lowercase(get(ENV, "IBKR_ORDER_TESTS", "false")) in ["1", "true", "yes", "on"]
    port = parse(Int, get(ENV, "IBKR_PORT", string(IbkrSpot._connectionconfig().port)))
    if !enabled
        @info "Skipping IbkrSpot order tests. Set IBKR_ORDER_TESTS=true to enable the minimum-quantity limit-order test."
        @test_skip "order lifecycle requires explicit IBKR_ORDER_TESTS=true"
    elseif !(port in IBKR_API_PORTS)
        error("IbkrSpot order tests refuse unsupported API port=$(port); recognized ports are $(IBKR_API_PORTS)")
    else
        connection = IbkrSpot._connectionconfig()
        issue = _tws_connection_issue(connection.host, port)
        if !isnothing(issue)
            @warn "Skipping IbkrSpot order test because TWS is unavailable" host=connection.host port=port reason=issue
            @test_skip "order lifecycle requires a successful TCP connection and Jib API handshake"
        else
            clientid = 10_000 + mod(getpid(), 1_000_000)
            cache = IbkrSpot.IbkrSpotCache(port=port, clientid=clientid)
            orderid = nothing
            try
                ticker = IbkrSpot.get24h(cache, "AAPLUSD")
                @test !isnothing(ticker)
                if !isnothing(ticker)
                    spec = IbkrSpot.executionorderspec(cache, :long)
                    info = IbkrSpot.symbolinfo(cache, "AAPLUSD")
                    @test !isnothing(info)
                    if !isnothing(info)
                        testquantity = info.minbaseqty
                        limitprice = min(ticker.bidprice * 0.75, spec.max_quote * 0.5)
                        limitprice = floor(limitprice / info.ticksize) * info.ticksize
                        @test limitprice > 0
                        @test limitprice < ticker.bidprice * 0.9
                        created = IbkrSpot.createorder(cache, "AAPLUSD", "Buy", testquantity, limitprice, true; configside=:long)
                        @test !isnothing(created)
                        if !isnothing(created)
                            orderid = String(created.orderid)
                            @test created.baseqty == testquantity
                            @test created.limitprice == limitprice
                        end
                    end
                end
            finally
                try
                    if !isnothing(orderid)
                        cancelled = IbkrSpot.cancelorder(cache, "AAPLUSD", orderid)
                        if isnothing(cancelled)
                            current = IbkrSpot.order(cache, orderid)
                            @test !isnothing(current)
                            if !isnothing(current)
                                @test current.status in ("Rejected", "Cancelled")
                                current.status == "Filled" && @warn "IBKR order test order filled before cancellation; review AAPL position" orderid=orderid
                            end
                        else
                            @test cancelled == orderid
                        end
                    end
                finally
                    IbkrSpot.disconnect!(cache)
                end
            end
        end
    end
end
