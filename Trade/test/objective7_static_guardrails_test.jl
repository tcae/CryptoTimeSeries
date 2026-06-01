using Test

@testset "Objective 7 static guardrails" begin
    trade_src_path = normpath(joinpath(@__DIR__, "..", "src", "Trade.jl"))
    trade_src = read(trade_src_path, String)

    @test !occursin("Classify.advice(", trade_src)
    @test !occursin("Classify.TradeAdvice", trade_src)
    @test !occursin("CTS_USE_STRATEGY_RUNTIME_API", trade_src)
end
