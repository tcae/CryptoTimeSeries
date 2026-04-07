using Test
using Dates
using DataFrames
using Targets
using TradingStrategy

@testset "Limit trade simulation" begin
    @testset "market trade simulation uses close prices on phase transitions" begin
        base = DateTime(2024, 1, 1)
        df = DataFrame(
            opentime=[base + Minute(i - 1) for i in 1:4],
            high=Float32[101, 103, 105, 104],
            low=Float32[99, 101, 103, 102],
            close=Float32[100, 102, 104, 103],
            score=Float32[0.9, 0.95, 0.9, 0.95],
            label=["flat", "up", "up", "flat"],
        )

        tradedf = TradingStrategy.simulate_market_trade_pairs(
            df,
            df[!, :score],
            df[!, :label];
            openthreshold=0.6f0,
            closethreshold=0.5f0,
            makerfee=0f0,
            takerfee=0f0,
        )

        @test nrow(tradedf) == 1
        @test tradedf[1, :trend] == up
        @test tradedf[1, :startix] == 2
        @test tradedf[1, :endix] == 4
        @test tradedf[1, :gain] == (103f0 - 102f0) / 102f0
    end

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

    @testset "phase labels are converted internally" begin
        base = DateTime(2024, 1, 3)
        df = DataFrame(
            opentime=[base + Minute(i - 1) for i in 1:3],
            high=Float32[101, 97, 96],
            low=Float32[98, 95, 94],
            close=Float32[100, 96, 95],
            centerpred=Float32[100, 96, 96],
            widthpred=Float32[4, 2, 2],
            score=Float32[0.95, 0.90, 0.90],
            label=["up", "flat", "flat"],
        )

        tradedf = TradingStrategy.simulate_limit_trade_pairs(
            df,
            df[!, :score],
            df[!, :label];
            openthreshold=0.6f0,
            closethreshold=0.5f0,
            entrytimeout=1,
            exittimeout=1,
            makerfee=0f0,
            takerfee=0f0,
        )

        @test nrow(tradedf) == 1
        @test tradedf[1, :trend] == up
        @test tradedf[1, :entryfilled] == true
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
