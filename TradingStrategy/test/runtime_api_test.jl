using Test
using Dates
using Targets
using TradingStrategy

@testset "Runtime API compatibility adapter" begin
    rt = TradingStrategy.GainSegmentRuntime()

    @test TradingStrategy.requiredhistoryminutes(rt) >= 0
    @test isempty(TradingStrategy.acceptedbases(rt))

    snap = TradingStrategy.StrategySnapshot(
        base="BTC",
        datetime=DateTime(2026, 1, 1),
        label=Targets.longbuy,
        long_openprice=100f0,
        long_closeprice=101f0,
        long_openix=1,
    )
    @test snap.label == Targets.longbuy
    @test snap.long_openix == 1

    gs = TradingStrategy.GainSegment(maxwindow=12)
    TradingStrategy.apply_strategy!(rt, gs; source="test")
    @test isempty(rt.strategy_state)
    @test isempty(rt.strategy_history)

    TradingStrategy.dropbase!(rt, "BTC")
    TradingStrategy.reset!(rt)
    @test isempty(TradingStrategy.acceptedbases(rt))

    recon = TradingStrategy.StrategyReconciliationInput(has_long_open=true, long_avg_entry=100f0, long_open_ix=7)
    @test recon.has_long_open
    @test recon.long_open_ix == 7
end
