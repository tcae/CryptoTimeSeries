using IbkrSpot, Test, Dates

# Order-placing tests. These transmit real orders and are therefore gated twice:
# the opt-in flag must be set AND the connected session must be a paper session.
#
# Enable with: IBKR_ORDER_TESTS=true IBKR_PORT=7497
#
# Refusing to run against 7496/4001 is deliberate: an accidental run must not be able to
# reach the live account, so the guard is a port check rather than a config flag.

const PAPER_PORTS = (7497, 4002)

@testset "IbkrSpot order lifecycle tests" begin
    enabled = lowercase(get(ENV, "IBKR_ORDER_TESTS", "false")) in ["1", "true", "yes", "on"]
    port = parse(Int, get(ENV, "IBKR_PORT", string(IbkrSpot._connectionconfig().port)))
    if !enabled
        @info "Skipping IbkrSpot order tests. Set IBKR_ORDER_TESTS=true with a paper session to enable."
    elseif !(port in PAPER_PORTS)
        error("IbkrSpot order tests refuse to run against port=$(port); paper ports are $(PAPER_PORTS)")
    else
        @test_skip "order lifecycle tests pending order implementation"
    end
end
