using Test

@testset "Trade tests" begin
    include("storage_format_test.jl")
    include("algorithm03_adapter_test.jl")
    include("strategy_runtime_config_test.jl")
    include("loop_control_test.jl")
    include("backtest_integration_test.jl")
    include("bybit_guardrail_test.jl")
    include("audit_snapshot_test.jl")
end
