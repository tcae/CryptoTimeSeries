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
    @test TradingStrategy.regimepurge(TradingStrategy.targetconfig("T09")) == 120
    feature = TradingStrategy.featureconfig("F22")
    basecolumns = ["nrg+5+0","nrg+15+0","nrg+60+0","nrg+240+0",
        "maxd+60+0","mind+60+0","rv+5+360+0"]
    @test Features.requestedcolumns(feature)[1:length(basecolumns)] == basecolumns
    @test Features.requestedcolumns(feature)[(length(basecolumns)+1):end] ==
        ["minnrg+0.01+240","minage+0.01+240","maxnrg+0.01+240","maxage+0.01+240"]
    target = TradingStrategy.targetconfig("T28")
    @test (target.bandratio,target.openwindow,target.maxwindow) == (0.01,120,240)
    @test TradingStrategy.targetconfig("T28") isa Targets.Trend07
    @test TradingStrategy.regimepurge(TradingStrategy.targetconfig("T28")) == 240
    @test isnothing(TradingStrategy.checktargetpartition(TradingStrategy.targetconfig("T28"),480))
    @test_throws AssertionError TradingStrategy.checktargetpartition(TradingStrategy.targetconfig("T28"),239)
    @test TradingStrategy.tsconfig("071").classifier == "F22-T28-P02-M011-B1"
    @test TradingStrategy.tsconfig("071").strategy == "S09"
    @test TradingStrategy.classifierconfig("F22-T28-P02-M011-B1").model == "M011"
    @test_throws AssertionError TradingStrategy.classifierconfig("F09-T19-P02-M002-B1")
    for id in ("T24", "T25", "T26")
        @test TradingStrategy.targetconfig(id).targetminutes == 480
        @test occursin("targetminutes=480", Targets.configdescription(TradingStrategy.targetconfig(id)))
    end
end
