using Test
using Dates
using DataFrames
using Targets
using TradingStrategy

@testset "Limit trade simulation" begin
    @testset "missed exit falls back to timed market exit" begin
        base = DateTime(2024, 1, 1)
        df = DataFrame(
            opentime=[base + Minute(i - 1) for i in 1:3],
            high=Float32[101, 96, 95],
            low=Float32[98, 95, 93],
            close=Float32[100, 96, 94],
            centerpred=Float32[100, 96, 96],
            widthpred=Float32[4, 2, 2],
            score=Float32[0.95, 0.90, 0.90],
            label=TradeLabel[longbuy, longclose, longclose],
        )

        tradedf = TradingStrategy.simulate_limit_trade_pairs(
            df,
            df[!, :score],
            df[!, :label];
            openthreshold=0.6f0,
            closethreshold=0.5f0,
            entrytimeout=1,
            exittimeout=1,
            makerfee=0.0015f0,
            takerfee=0.002f0,
        )

        @test nrow(tradedf) == 1
        @test tradedf[1, :trend] == up
        @test tradedf[1, :entryfilled] == true
        @test tradedf[1, :exitfilled] == false
        @test tradedf[1, :missedexit] == true
        @test tradedf[1, :exitreason] == "timeout_market"
        @test tradedf[1, :gainfee] < 0f0
    end

    @testset "missed entry expires without trade" begin
        base = DateTime(2024, 1, 2)
        df = DataFrame(
            opentime=[base + Minute(i - 1) for i in 1:3],
            high=Float32[101, 102, 103],
            low=Float32[99, 100, 101],
            close=Float32[100, 101, 102],
            centerpred=Float32[100, 101, 102],
            widthpred=Float32[4, 4, 4],
            score=Float32[0.95, 0.20, 0.20],
            label=TradeLabel[longbuy, allclose, allclose],
        )

        tradedf = TradingStrategy.simulate_limit_trade_pairs(
            df,
            df[!, :score],
            df[!, :label];
            openthreshold=0.6f0,
            closethreshold=0.5f0,
            entrytimeout=1,
            exittimeout=1,
        )

        @test nrow(tradedf) == 0
    end
end
