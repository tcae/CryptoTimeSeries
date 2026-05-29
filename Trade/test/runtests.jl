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
    include("reload_cadence_test.jl")
    include("bybit_guardrail_test.jl")
    if _hastadeauditpkg()
        include("audit_snapshot_test.jl")
        include("ownership_selection_test.jl")
    else
        @info "Skipping TradeAudit-dependent tests because TradeAudit package is not available in current test environment"
    end
end
