using IbkrSpot, Test, Dates, Jib

# These tests transmit small paper orders and therefore require explicit opt-in.
#
# Enable with: IBKR_ORDER_TESTS=true IBKR_PORT=7497

const IBKR_API_PORTS = (7496, 7497, 4001, 4002)
const IBKR_ORDER_FILL_TIMEOUT = Dates.Second(60)

"""Wait for an IBKR order fill and return the executed quantity and average price."""
function _wait_for_order_fill(cache::IbkrSpot.IbkrSpotCache, orderid::String; timeout::Dates.Period=IBKR_ORDER_FILL_TIMEOUT)
    deadline = Dates.now(Dates.UTC) + timeout
    numericid = parse(Int, orderid)
    executionfilter = Jib.ExecutionFilter(clientId=cache.config.clientid)
    while Dates.now(Dates.UTC) < deadline
        rows = IbkrSpot._requestvalues(cache, "executions for order $(orderid)",
            (connection, reqid) -> Jib.reqExecutions(connection, reqid, executionfilter))
        matching = filter(row -> row.execution.orderId == numericid, rows)
        if !isempty(matching)
            execution = last(matching).execution
            filled = execution.cumQty > 0 ? execution.cumQty : execution.shares
            averageprice = execution.avgPrice > 0 ? execution.avgPrice : execution.price
            filled > 0 && averageprice > 0 && return (filled=filled, averageprice=averageprice)
        end
        sleep(0.5)
    end
    error("IBKR order $(orderid) has no execution within $(timeout)")
end

"""Cancel a still-open paper order, accepting a confirmed disappearance after timeout."""
function _cancel_if_open(cache::IbkrSpot.IbkrSpotCache, symbol::String, orderid::Union{Nothing, String})
    isnothing(orderid) && return nothing
    isempty(IbkrSpot.openorders(cache; symbol=symbol, orderid=orderid)) && return nothing
    try
        IbkrSpot.cancelorder(cache, symbol, orderid)
    catch
        isempty(IbkrSpot.openorders(cache; symbol=symbol, orderid=orderid)) || rethrow()
    end
    return nothing
end

@testset "IbkrSpot order lifecycle tests" begin
    enabled = lowercase(get(ENV, "IBKR_ORDER_TESTS", "false")) in ["1", "true", "yes", "on"]
    port = parse(Int, get(ENV, "IBKR_PORT", string(IbkrSpot._connectionconfig().port)))
    if !enabled
        @info "Skipping IbkrSpot order tests. Set IBKR_ORDER_TESTS=true to enable the paper market and limit-order test."
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
            marketorderid = nothing
            followuporderid = nothing
            testquantity = 0.0
            testreferenceprice = nothing
            marketfilledquantity = 0.0
            startingpositionquantity = 0.0
            try
                info = IbkrSpot.symbolinfo(cache, "TUSD")
                @test !isnothing(info)
                if !isnothing(info)
                    testquantity = max(info.minbaseqty, 1.0)
                    spec = (side=:long, instrument="stock", leverage=1, max_quote=300.0)
                    testreferenceprice = 50.0
                    @assert testquantity * testreferenceprice <= spec.max_quote "market test quantity=$(testquantity) reference notional=$(testquantity * testreferenceprice) exceeds test cap=$(spec.max_quote)"
                    startingpositions = IbkrSpot.positionsnapshot(cache)
                    startingpositionix = findfirst(==("T"), startingpositions.coin)
                    !isnothing(startingpositionix) && (startingpositionquantity = startingpositions.long_qty[startingpositionix])
                    marketorder = IbkrSpot.createorder(
                        cache,
                        "TUSD",
                        "Buy",
                        testquantity,
                        nothing,
                        false;
                        configside=:long,
                        execution_spec=spec,
                        market_referenceprice=testreferenceprice,
                    )
                    @test !isnothing(marketorder)
                    if !isnothing(marketorder)
                        marketorderid = String(marketorder.orderid)
                        fill = _wait_for_order_fill(cache, marketorderid)
                        marketfilledquantity = fill.filled
                        @test fill.filled == testquantity
                        @test fill.averageprice > 0

                        limitprice = fill.averageprice + info.ticksize
                        followuporder = IbkrSpot.createorder(
                            cache,
                            "TUSD",
                            "Sell",
                            fill.filled,
                            limitprice,
                            true;
                            configside=:long,
                            execution_spec=spec,
                            reduceonly=true,
                        )
                        @test !isnothing(followuporder)
                        if !isnothing(followuporder)
                            followuporderid = String(followuporder.orderid)
                            @test followuporder.limitprice >= fill.averageprice
                            _cancel_if_open(cache, "TUSD", followuporderid)
                            @test isempty(IbkrSpot.openorders(cache; symbol="TUSD", orderid=followuporderid))
                        end
                    end
                end
            finally
                try
                    _cancel_if_open(cache, "TUSD", followuporderid)
                    _cancel_if_open(cache, "TUSD", marketorderid)
                    positions = IbkrSpot.positionsnapshot(cache)
                    positionix = findfirst(==("T"), positions.coin)
                    if !isnothing(positionix) && positions.long_qty[positionix] > 0
                        additionalquantity = max(0.0, positions.long_qty[positionix] - startingpositionquantity)
                        closequantity = min(additionalquantity, marketfilledquantity)
                        closeorder = IbkrSpot.createorder(
                            cache,
                            "TUSD",
                            "Sell",
                            closequantity,
                            nothing,
                            false;
                            configside=:long,
                            execution_spec=(side=:long, instrument="stock", leverage=1, max_quote=300.0),
                            reduceonly=true,
                            market_referenceprice=testreferenceprice,
                        )
                        @test !isnothing(closeorder)
                        if !isnothing(closeorder)
                            _wait_for_order_fill(cache, String(closeorder.orderid))
                        end
                    end
                    @test isempty(IbkrSpot.openorders(cache; symbol="TUSD"))
                finally
                    IbkrSpot.disconnect!(cache)
                end
            end
        end
    end
end
