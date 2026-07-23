using Test

@testset "Trade tests" begin
    include("simulated_marketview_test.jl")
    include("reload_cadence_test.jl")
    # bybit_guardrail_test removed - tests guardrail that prevents trading on data-only exchanges
    # This concept no longer applies with single-exchange model
end
