using Test

@testset "Trade tests" begin
    include("storage_format_test.jl")
    include("getgainsalgo_adapter_test.jl")
    include("strategy_runtime_config_test.jl")
    include("simulated_marketview_test.jl")
    include("strategy_advice_test.jl")
    include("reload_cadence_test.jl")
    include("trade_vs_tradingstrategy_regression_test.jl")
    include("loop_control_test.jl")
    include("async_control_test.jl")
    include("backtest_integration_test.jl")
    include("bybit_guardrail_test.jl")
    include("audit_snapshot_test.jl")
end
