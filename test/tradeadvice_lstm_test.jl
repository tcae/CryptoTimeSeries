module TrendLstmTests

using Test
using DataFrames
using EnvConfig

include("../scripts/TrendLstm.jl")

@testset "TrendLstm resolves explicit test mode" begin
    mode = TrendLstm._resolve_runmode(["test", "retrain"])
    @test mode.retrain == true
    @test mode.testmode == true
    @test mode.trainmode == false

    default_retrain = TrendLstm._resolve_runmode(["retrain"])
    @test default_retrain.retrain == true
    @test default_retrain.trainmode == true
    @test default_retrain.testmode == false
end

@testset "TrendLstm missing checkpoint returns nothing" begin
    EnvConfig.init(test)
    cfg = TrendLstm.TrendLstmConfig(
        configname="ut-lstm-missing-checkpoint",
        folder="TrendLstm-ut-lstm-missing-checkpoint-test",
        trendconfig=TrendLstm._resolve_trendconfig("029"),
        mode=EnvConfig.configmode,
    )

    checkpoint = TrendLstm._with_log_subfolder(cfg.folder) do
        TrendLstm.load_lstm(cfg)
    end

    @test isnothing(checkpoint)
end

@testset "TrendLstm comparison transpose is terminal-only" begin
    comparison = DataFrame(
        configname=["001", "002"],
        trendconfig=["029", "025E"],
        final_eval_loss=Float32[0.1, 0.2],
    )

    terminaldf = TrendLstm._transpose_table_for_terminal(comparison)

    @test names(comparison) == ["configname", "trendconfig", "final_eval_loss"]
    @test names(terminaldf) == ["metric", "001", "002"]
    @test terminaldf[1, :metric] == "configname"
    @test terminaldf[2, Symbol("001")] == "029"
    @test terminaldf[3, Symbol("002")] == Float32(0.2)
end

@testset "TrendLstm config preset uses config key" begin
    EnvConfig.init(test)
    cfg = TrendLstm.buildcfg(["config=001", "trend=029", "test"])
    @test cfg.configname == "001"
    @test cfg.trendconfig.configname == "029"
    @test cfg.folder == "TrendLstm-001-test"
end

end
