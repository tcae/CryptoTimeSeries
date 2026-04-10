module ScriptHelpTests

using Test, Dates

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
    @test occursin("oversampling=<Bool>", thelp)
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
    @test trendcfg.oversampling == false

    boundscfg = BoundsEstimator.buildcfg(["config=001"], ["SINE"], startdt, enddt)
    @test boundscfg.configname == "001"

    lstmcfg = TrendLstm.buildcfg(["config=001", "trend=029"])
    @test lstmcfg.configname == "001"
    @test lstmcfg.trendconfig.configname == "029"
end

end
