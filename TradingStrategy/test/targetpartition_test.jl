"Test target implementation deliberately missing a declared look-ahead."
struct UndeclaredHorizonTarget <: Targets.AbstractTargets end

@testset "Target horizon fits partition gap" begin
    @test_throws r"lookaheadminutes not implemented" TradingStrategy.checktargetpartition(UndeclaredHorizonTarget(), 480)
    @test isnothing(TradingStrategy.checktargetpartition(Targets.Trend05(targetminutes=480), 480))
    @test_throws r"lookaheadminutes=481.*gapsize=480" TradingStrategy.checktargetpartition(Targets.Trend05(targetminutes=481), 480)
    @test_throws AssertionError TradingStrategy.checktargetpartition(Targets.Trend05(), 0)
    @test isnothing(TradingStrategy.checktargetpartition(Targets.TrendRegression(480, 0.01, -0.01), 480))
    @test_throws AssertionError TradingStrategy.checktargetpartition(Targets.TrendRegression(481, 0.01, -0.01), 480)
    @test isnothing(TradingStrategy.checktargetpartition(TradingStrategy.targetconfig("T09"), 480))
    @test_throws AssertionError TradingStrategy.classifierconfig("F09-T19-P02-M002-B1")
    for id in ("T24", "T25", "T26")
        @test TradingStrategy.targetconfig(id).targetminutes == 480
        @test occursin("targetminutes=480", Targets.configdescription(TradingStrategy.targetconfig(id)))
    end
    include("../../test/investigate_trend05_horizon_test.jl")
end
