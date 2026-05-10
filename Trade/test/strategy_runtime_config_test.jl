using Test
using Trade
using TradingStrategy

@testset "strategy runtime config bridge" begin
    mc = Dict{Symbol, Any}(
        :strategy_engine => :classifier,
        :strategy_state => Dict{String, Any}(),
        :strategy_history => Dict{String, Any}(),
        :strategy_openthreshold => 0.6f0,
        :strategy_closethreshold => 0.5f0,
        :strategy_buygain => 0.001f0,
        :strategy_sellgain => 0.01f0,
        :strategy_limitreduction => 0f0,
        :strategy_maxwindow => 4 * 60,
        :strategy_source => "default",
    )

    gs = TradingStrategy.GainSegment(
        ;
        maxwindow=123,
        openthreshold=0.65f0,
        closethreshold=0.45f0,
        algorithm=TradingStrategy.algorithm03!,
        limitreduction=0.2f0,
    )
    gs.buygain = 0.003f0
    gs.sellgain = 0.012f0

    mc[:strategy_state]["BTC"] = gs
    mc[:strategy_history]["BTC"] = (predictionsdf=nothing, scores=Float32[], labels=Any[])

    tdref = (configname="046", tradingstrategy=gs)
    Trade.apply_trenddetector_strategy!(mc, tdref)

    @test mc[:strategy_engine] == :algorithm03
    @test mc[:strategy_source] == "trenddetector:046"
    @test mc[:strategy_openthreshold] == Float32(0.65)
    @test mc[:strategy_closethreshold] == Float32(0.45)
    @test mc[:strategy_buygain] == Float32(0.003)
    @test mc[:strategy_sellgain] == Float32(0.012)
    @test mc[:strategy_limitreduction] == Float32(0.2)
    @test mc[:strategy_maxwindow] == 123
    @test isempty(mc[:strategy_state])
    @test isempty(mc[:strategy_history])

    gs_invalid = TradingStrategy.GainSegment(
        ;
        maxwindow=1,
        openthreshold=1.2f0,
        closethreshold=0.2f0,
        algorithm=TradingStrategy.algorithm03!,
        limitreduction=0f0,
    )
    @test_throws AssertionError Trade.apply_tradingstrategy!(mc, gs_invalid)
end
