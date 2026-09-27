using IbkrSpot, Test, Dates

# Read-only tests against a running TWS / IB Gateway.
# Enable with: IBKR_ONLINE_TESTS=true
# Port override: IBKR_PORT=7497 (paper) or 7496 (live). Defaults to the execution config port.
#
# These tests never place, amend or cancel an order, so they are safe against a live session.

@testset "IbkrSpot online read-only tests" begin
    enabled = lowercase(get(ENV, "IBKR_ONLINE_TESTS", "false")) in ["1", "true", "yes", "on"]
    if !enabled
        @info "Skipping IbkrSpot online tests. Set IBKR_ONLINE_TESTS=true with TWS running to enable."
    else
        @test_skip "online read-only tests pending market data and symbol metadata implementation"
    end
end
