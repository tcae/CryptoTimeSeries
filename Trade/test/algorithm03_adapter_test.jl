using Test
using TradingStrategy
using Trade
using Targets

@testset "algorithm03 adapter mapping" begin
    gs = TradingStrategy.GainSegment(; algorithm=TradingStrategy.algorithm03!)

    gs.buyta.orderlabel = longbuy
    @test Trade._algorithm03_action2label(gs, allclose) == longbuy

    gs.buyta.orderlabel = shortbuy
    @test Trade._algorithm03_action2label(gs, allclose) == shortbuy

    gs.buyta.orderlabel = ignore
    gs.sellta.orderlabel = longclose
    @test Trade._algorithm03_action2label(gs, allclose) == longclose

    gs.sellta.orderlabel = shortclose
    @test Trade._algorithm03_action2label(gs, allclose) == shortclose

    gs.sellta.orderlabel = ignore
    @test Trade._algorithm03_action2label(gs, allclose) == allclose
end
