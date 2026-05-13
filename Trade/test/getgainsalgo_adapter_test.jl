using Test
using Trade
using TradingStrategy
using Targets

@testset "getgainsalgo adapter mapping" begin
    gs = TradingStrategy.GainSegment(; algorithm=TradingStrategy.gain_limit_reversal!)
    gs.buyta = TradingStrategy.TradeAction()
    gs.sellta = TradingStrategy.TradeAction()
    gs.buyta.orderlabel = longbuy

    @test Trade._getgainsalgo_action2label(gs, allclose) == longbuy

    gs.buyta.orderlabel = shortbuy
    @test Trade._getgainsalgo_action2label(gs, allclose) == shortbuy

    gs.buyta.orderlabel = ignore
    gs.sellta.orderlabel = longclose
    @test Trade._getgainsalgo_action2label(gs, allclose) == longclose

    gs.sellta.orderlabel = shortclose
    @test Trade._getgainsalgo_action2label(gs, allclose) == shortclose

    gs.sellta.orderlabel = ignore
    @test Trade._getgainsalgo_action2label(gs, allclose) == allclose
end