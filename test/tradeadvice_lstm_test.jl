module TradeAdviceLstmTests

using Test
using DataFrames
using EnvConfig

include("../scripts/TradeAdviceLstm.jl")

@testset "TradeAdviceLstm resolves explicit test mode" begin
    mode = TradeAdviceLstm._resolve_runmode(["test", "retrain"])
    @test mode.retrain == true
    @test mode.testmode == true
    @test mode.trainmode == false

    default_retrain = TradeAdviceLstm._resolve_runmode(["retrain"])
    @test default_retrain.retrain == true
    @test default_retrain.trainmode == true
    @test default_retrain.testmode == false
end

@testset "TradeAdviceLstm missing checkpoint returns nothing" begin
    EnvConfig.init(test)
    cfg = TradeAdviceLstm.TradeAdviceLstmConfig(
        configname="ut-lstm-missing-checkpoint",
        folder="TradeAdviceLstm-ut-lstm-missing-checkpoint-test",
        trendconfig=TradeAdviceLstm._resolve_trendconfig("029"),
        mode=EnvConfig.configmode,
    )

    checkpoint = TradeAdviceLstm._with_log_subfolder(cfg.folder) do
        TradeAdviceLstm.load_lstm(cfg)
    end

    @test isnothing(checkpoint)
end

@testset "TradeAdviceLstm comparison transpose is terminal-only" begin
    comparison = DataFrame(
        configname=["001", "002"],
        trendconfig=["029", "025E"],
        final_eval_loss=Float32[0.1, 0.2],
    )

    terminaldf = TradeAdviceLstm._transpose_table_for_terminal(comparison)

    @test names(comparison) == ["configname", "trendconfig", "final_eval_loss"]
    @test names(terminaldf) == ["metric", "001", "002"]
    @test terminaldf[1, :metric] == "configname"
    @test terminaldf[2, Symbol("001")] == "029"
    @test terminaldf[3, Symbol("002")] == Float32(0.2)
end

end
