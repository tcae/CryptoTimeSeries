using Test

function _hastadeauditpkg()::Bool
    try
        @eval import TradeAudit
        return true
    catch
        return false
    end
end

@testset "Trade tests" begin
    include("simulated_marketview_test.jl")
    include("managed_close_orders_test.jl")
    include("active_buy_cache_test.jl")
    include("reload_cadence_test.jl")
    # bybit_guardrail_test removed - tests guardrail that prevents trading on data-only exchanges
    # This concept no longer applies with single-exchange model
    if _hastadeauditpkg()
        include("audit_snapshot_test.jl")
    else
        @info "Skipping TradeAudit-dependent tests because TradeAudit package is not available in current test environment"
    end
end
