module ScriptHelpTests

using Test, Dates, Statistics
using Classify, Targets, Distributions, EnvConfig

include("../scripts/BoundsEstimator.jl")
include("../scripts/TrendDetector.jl")

@testset "script help text includes defaults" begin
    bhelp = BoundsEstimator.boundsestimatorhelp()
    @test occursin("help, --help, -h", bhelp)
    @test occursin("config=<configname>", bhelp)
    @test occursin("test startdt: `2025-01-17T20:56:00`", bhelp)

    thelp = TrendDetector.trenddetectorhelp()
    @test occursin("TS_CONFIGS", thelp)
    @test occursin("stage=<stores|train|predict|evaluate|gain>", thelp)
    @test occursin("help, --help, -h", thelp)

end

@testset "script help flag detection" begin
    @test BoundsEstimator._wants_help(["help"])
    @test BoundsEstimator._wants_help(["--help"])
    @test BoundsEstimator._wants_help(["help=true"])
    @test !BoundsEstimator._wants_help(["help=false"])

    @test TrendDetector._wants_help(["-h"])
    @test !TrendDetector._wants_help(["train"])

    @test isnothing(TrendDetector._assertknownargs(["train", "ts=050", "stage=train", "coins=SINE", "runid=x"]))
    @test_throws AssertionError TrendDetector._assertknownargs(["config=050"])
    @test_throws AssertionError TrendDetector._assertknownargs(["special"])
    @test_throws AssertionError TrendDetector._assertknownargs(["TRAIN"])

end

@testset "script config selection by configname" begin
    startdt = DateTime("2025-01-17T20:56:00")
    enddt = DateTime("2025-08-10T15:00:00")

    EnvConfig.init(test)
    rc = TrendDetector.buildrunconfig(["ts=029", "stage=train", "coins=SINE"])
    @test rc.tscfg.configname == "029"
    @test rc.clcfg.classbalancing == false
    @test rc.coinset == "CS00"
    @test rc.coins == ["SINE"]
    @test TrendDetector.classifiername(rc) == "F01-T01-P02-M002-B0-CS00-D00"
    @test TrendDetector.runsstage(rc, "stores") && !TrendDetector.runsstage(rc, "predict")
    @test_throws AssertionError TrendDetector.buildrunconfig(["stage=unknown"])
    @test_throws AssertionError TrendDetector.buildrunconfig(["startdt=2025-07-01T01:00:00"])

    boundscfg = BoundsEstimator.buildcfg(["config=001"], ["SINE"], startdt, enddt)
    @test boundscfg.configname == "001"

end

@testset "class weights use inverse class frequency" begin
    weights = Classify.classweights([2, 1, 0])
    @test weights[1] < weights[2]
    @test weights[3] == 0f0
    @test isapprox((2 * weights[1] + weights[2]) / 3, 1.0; atol=1e-6)
end

@testset "weighted classifier adaptation accepts sample weights" begin
    features = Float32[0 0 0 1; 0 1 2 3]
    targets = [longopen, longopen, longopen, allclose]
    nn = Classify.model006(size(features, 1), [longopen, allclose], "ut_weighted_trend")
    weights = Classify.classweights([3, 1])[[1, 1, 1, 2]]
    mktempdir() do tmpdir
        adapted = Classify.adaptnn!(nn, Classify.MatrixSampleSource(nn, features, targets, weights); maxepochs=3, folderpath=tmpdir)
        @test length(adapted.losses) == 3
    end
end

end
