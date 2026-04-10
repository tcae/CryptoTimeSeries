module ScriptHelpTests

using Test

include("../scripts/BoundsEstimator.jl")
include("../scripts/TrendDetector.jl")
include("../scripts/TradeAdviceLstm.jl")

@testset "script help text includes defaults" begin
    bhelp = BoundsEstimator.boundsestimatorhelp()
    @test occursin("help, --help, -h", bhelp)
    @test occursin("boundsmk001config()", bhelp)
    @test occursin("test startdt: `2025-01-17T20:56:00`", bhelp)

    thelp = TrendDetector.trenddetectorhelp()
    @test occursin("mk029config()", thelp)
    @test occursin("oversampling: `false`", thelp)
    @test occursin("help, --help, -h", thelp)

    lhelp = TradeAdviceLstm.tradeadvicelstmhelp()
    @test occursin("tradeadvice=<ref>", lhelp)
    @test occursin("trend=<ref>", lhelp)
    @test occursin("openthresholds=v1,v2,...", lhelp)
    @test occursin("Default: `trend025`", lhelp)
end

@testset "script help flag detection" begin
    @test BoundsEstimator._wants_help(["help"])
    @test BoundsEstimator._wants_help(["--help"])
    @test BoundsEstimator._wants_help(["help=true"])
    @test !BoundsEstimator._wants_help(["help=false"])

    @test TrendDetector._wants_help(["-h"])
    @test !TrendDetector._wants_help(["train"])

    @test TradeAdviceLstm._wants_help(["help=yes"])
    @test !TradeAdviceLstm._wants_help(["train", "trend=025"])
end

end
