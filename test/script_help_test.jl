module ScriptHelpTests

using Test, Dates, Statistics
using Classify, Targets, Distributions, EnvConfig

include("../scripts/BoundsEstimator.jl")
include("../scripts/TrendDetector.jl")
include("../scripts/TrendLstm.jl")

@testset "script help text includes defaults" begin
    bhelp = BoundsEstimator.boundsestimatorhelp()
    @test occursin("help, --help, -h", bhelp)
    @test occursin("config=<configname>", bhelp)
    @test occursin("test startdt: `2025-01-17T20:56:00`", bhelp)

    thelp = TrendDetector.trenddetectorhelp()
    @test occursin("TREND_DETECTOR_CONFIGS", thelp)
    @test occursin("classbalancing=<Bool>", thelp)
    @test !occursin("oversampling=<Bool>", thelp)
    @test occursin("help, --help, -h", thelp)

    lhelp = TrendLstm.trendlstmhelp()
    @test occursin("config=<configname>", lhelp)
    @test occursin("trend=<configname>", lhelp)
    @test occursin("openthresholds=v1,v2,...", lhelp)
    @test occursin("Default: `trend025`", lhelp)
    @test occursin("scripts/TrendLstm.jl", lhelp)
end

@testset "script help flag detection" begin
    @test BoundsEstimator._wants_help(["help"])
    @test BoundsEstimator._wants_help(["--help"])
    @test BoundsEstimator._wants_help(["help=true"])
    @test !BoundsEstimator._wants_help(["help=false"])

    @test TrendDetector._wants_help(["-h"])
    @test !TrendDetector._wants_help(["train"])

    @test TrendLstm._wants_help(["help=yes"])
    @test !TrendLstm._wants_help(["train", "trend=025"])
end

@testset "script config selection by configname" begin
    startdt = DateTime("2025-01-17T20:56:00")
    enddt = DateTime("2025-08-10T15:00:00")

    trendcfg = TrendDetector.buildcfg(["config=029"], ["SINE"], startdt, enddt)
    @test trendcfg.configname == "029"
    @test trendcfg.classbalancing == false

    boundscfg = BoundsEstimator.buildcfg(["config=001"], ["SINE"], startdt, enddt)
    @test boundscfg.configname == "001"

    lstmcfg = TrendLstm.buildcfg(["config=001", "trend=029"])
    @test lstmcfg.configname == "001"
    @test lstmcfg.trendconfig.configname == "029"
    @test lstmcfg.openthresholds == Float32[0.8f0, 0.7f0, 0.6f0, 0.5f0, 0.4f0, 0.3f0]
    @test lstmcfg.closethresholds == Float32[0.1f0]
end

@testset "class weighting uses inverse class frequency" begin
    targets = [Targets.longbuy, Targets.longbuy, Targets.allclose]
    info = Classify.classweighting(targets, [Targets.longbuy, Targets.allclose, Targets.shortbuy])

    @test isapprox(pdf(info.dist, "longbuy"), 2 / 3; atol=1e-6)
    @test isapprox(pdf(info.dist, "allclose"), 1 / 3; atol=1e-6)
    @test info.classweights[1] < info.classweights[2]
    @test info.classweights[3] == 0f0
    @test isapprox(mean(info.sampleweights), 1.0; atol=1e-6)
end

@testset "weighted classifier adaptation accepts sample weights" begin
    EnvConfig.init(test)
    features = Float32[0 0 0 1; 0 1 2 3]
    targets = [Targets.longbuy, Targets.longbuy, Targets.longbuy, Targets.allclose]
    nn = Classify.model006(size(features, 1), [Targets.longbuy, Targets.allclose], "ut_weighted_trend")
    info = Classify.classweighting(targets, nn.labels)
    adapted = Classify.adaptnn!(nn, features, targets; sampleweights=info.sampleweights)

    @test length(adapted.losses) > 0
end

end
