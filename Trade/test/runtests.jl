using Test

@testset "Trade tests" begin
    include("simulated_marketview_test.jl")
    include("managed_close_orders_test.jl")
    include("active_buy_cache_test.jl")
    include("reload_cadence_test.jl")
    include("bybit_guardrail_test.jl")
    include("objective4_scaffolding_test.jl")
    include("objective4_live_marketdata_publicread_test.jl")
    include("runtime_api_integration_test.jl")
    include("audit_snapshot_test.jl")
    include("ownership_selection_test.jl")
end
