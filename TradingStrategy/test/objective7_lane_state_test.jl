using Test
using Dates
using DataFrames
using Targets
using TradingStrategy

@testset "Objective 7 lane state sync" begin
    @testset "long open intent is mirrored into long lane" begin
        gs = TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!)
        dt = DateTime(2026, 1, 1)
        predictionsdf = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
        )

        TradingStrategy.getgains(gs, predictionsdf, Float32[0.9f0], TradeLabel[longbuy], false; lastix=1)

        @test gs.longta.label == longbuy
        @test isapprox(gs.longta.openprice, 99.9f0; atol=1f-4)
        @test isapprox(gs.longta.closeprice, 101f0; atol=1f-4)
        @test gs.shortta.label == ignore
    end

    @testset "close guidance stays in long lane" begin
        gs = TradingStrategy.GainSegment()
        gs.longta.closeprice = 110f0
        gs.longta.openprice = 100f0
        gs.longta.openix = 2
        gs.longta.label = ignore

        TradingStrategy.synclanes!(gs)

        @test gs.longta.closeprice == 110f0
        @test gs.longta.openprice == 100f0
        @test gs.longta.label == ignore
        @test gs.shortta.closeprice == 0f0
    end

    @testset "short open intent stays in short lane" begin
        gs = TradingStrategy.GainSegment()
        gs.shortta = TradingStrategy.TradeAction(shortbuy, 99f0, 101f0, 0)

        TradingStrategy.synclanes!(gs)

        @test gs.shortta.label == shortbuy
        @test gs.shortta.openprice == 101f0
        @test gs.longta.label == ignore
    end
end

@testset "Objective 7 no implicit full-fill swap" begin
    gs = TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!)
    dt = DateTime(2026, 1, 2)
    predictionsdf = DataFrame(
        opentime=[dt, dt + Minute(1)],
        high=Float32[101f0, 102f0],
        low=Float32[99f0, 98f0],
        close=Float32[100f0, 100f0],
    )

    TradingStrategy.getgains(gs, predictionsdf, Float32[0.9f0, 0.9f0], TradeLabel[longbuy, longbuy], false; lastix=2)

    @test gs.longta.label == longbuy
    @test gs.longta.openix == 0
    @test gs.longta.closeprice > 0f0
end

@testset "Objective 7 reconciliation hooks" begin
    @testset "long reconciliation synthesizes close guidance" begin
        gs = TradingStrategy.GainSegment()
        gs.sellgain = 0.02f0
        TradingStrategy.setreconciliation!(gs; long_open_qty=3.5f0, long_avg_entry=10f0, long_open_ix=7)

        TradingStrategy.synclanes!(gs)

        @test gs.longta.openprice == 10f0
        @test gs.longta.openix == 7
        @test isapprox(gs.longta.closeprice, 10.2f0; atol=1f-4)
    end

    @testset "short reconciliation synthesizes close guidance" begin
        gs = TradingStrategy.GainSegment()
        gs.sellgain = 0.03f0
        TradingStrategy.setreconciliation!(gs; short_open_qty=2f0, short_avg_entry=20f0, short_open_ix=11)

        TradingStrategy.synclanes!(gs)

        @test gs.shortta.openprice == 20f0
        @test gs.shortta.openix == 11
        @test isapprox(gs.shortta.closeprice, 19.4f0; atol=1f-4)
    end
end

@testset "Objective 7 lane openix and gain realization" begin
    @testset "long lane sets openix and records gain on closeprice hit" begin
        gs = TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!)
        dt = DateTime(2026, 1, 3)
        predictionsdf = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[100f0, 102f0],
            low=Float32[100f0, 99f0],
            close=Float32[100f0, 100f0],
        )

        TradingStrategy.getgains(gs, predictionsdf, Float32[0.9f0, 0.9f0], TradeLabel[longbuy, longbuy], false; lastix=2)

        @test nrow(gs.gaindf) >= 1
        @test gs.gaindf[end, :trend] == up
        @test gs.longta.openix == 0
    end

    @testset "short lane sets openix and records gain on closeprice hit" begin
        gs = TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!)
        dt = DateTime(2026, 1, 4)
        predictionsdf = DataFrame(
            opentime=[dt, dt + Minute(1)],
            high=Float32[100f0, 101f0],
            low=Float32[100f0, 98f0],
            close=Float32[100f0, 100f0],
        )

        TradingStrategy.getgains(gs, predictionsdf, Float32[0.9f0, 0.9f0], TradeLabel[shortbuy, shortbuy], false; lastix=2)

        @test nrow(gs.gaindf) >= 1
        @test gs.gaindf[end, :trend] == down
        @test gs.shortta.openix == 0
    end
end

@testset "Objective 7 lane-native gain_limit_reversal control path" begin
    gs = TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!, limitreduction=1f0)
    gs.longta.openprice = 100f0
    gs.longta.openix = 3
    gs.longta.closeprice = 110f0
    gs.longta.label = ignore

    dt = DateTime(2026, 1, 5)
    predictionsdf = DataFrame(
        opentime=[dt],
        high=Float32[105f0],
        low=Float32[95f0],
        close=Float32[100f0],
    )

    TradingStrategy.getgains(gs, predictionsdf, Float32[0.1f0], TradeLabel[allclose], false; lastix=1)

    @test isapprox(gs.longta.closeprice, 108.9f0; atol=1f-4)
    @test gs.longta.label == ignore
end

@testset "Objective 7 reconciliation persistence boundaries" begin
    gs = TradingStrategy.GainSegment()
    TradingStrategy.setreconciliation!(gs; long_open_qty=1f0, long_avg_entry=10f0, long_open_ix=7)
    TradingStrategy.synclanes!(gs)
    @test gs.longta.openprice == 10f0
    @test gs.longta.openix == 7

    TradingStrategy.clearreconciliation!(gs)
    TradingStrategy._clearactionlane!(gs.longta)
    TradingStrategy.synclanes!(gs)
    @test gs.longta.openprice == 0f0
    @test gs.longta.openix == 0

    TradingStrategy.setreconciliation!(gs; short_open_qty=1f0, short_avg_entry=20f0, short_open_ix=8)
    TradingStrategy.reset!(gs)
    @test isnothing(gs.lane_reconciliation)
end

@testset "Objective 7 overlapping lanes partial-fill and cancel keep rules" begin
    @testset "one tick can keep overlapping lane entries without forced cancellation" begin
        gs = TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!)
        gs.longta = TradingStrategy.TradeAction(longbuy, 120f0, 100f0, 0)
        gs.shortta = TradingStrategy.TradeAction(shortbuy, 80f0, 100f0, 0)

        dt = DateTime(2026, 1, 6)
        predictionsdf = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
        )

        TradingStrategy.getgains(gs, predictionsdf, Float32[0.1f0], TradeLabel[allclose], false; lastix=1)

        @test gs.longta.openix == 1
        @test gs.shortta.openix == 1
        @test nrow(gs.gaindf) == 0
    end

    @testset "lane-specific close hit realizes one lane and keeps the other" begin
        gs = TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!)
        gs.longta = TradingStrategy.TradeAction(longbuy, 101f0, 100f0, 1)
        gs.shortta = TradingStrategy.TradeAction(shortbuy, 97f0, 100f0, 1)

        dt = DateTime(2026, 1, 7)
        predictionsdf = DataFrame(
            opentime=[dt],
            high=Float32[101f0],
            low=Float32[99f0],
            close=Float32[100f0],
        )

        TradingStrategy.getgains(gs, predictionsdf, Float32[0.1f0], TradeLabel[allclose], false; lastix=1)

        @test nrow(gs.gaindf) == 1
        @test gs.gaindf[end, :trend] == up
        @test gs.longta.openix == 0
        @test gs.longta.openprice > 0f0
        @test gs.longta.closeprice > 0f0
        @test gs.shortta.openix == 1
        @test gs.shortta.openprice > 0f0
    end
end
