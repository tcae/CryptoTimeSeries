using Test
using Targets
using TradingStrategy
using Classify
using DataFrames

@testset "LstmTradeDecider" begin
    @testset "constructor maps labels and thresholds" begin
        decider = TradingStrategy.LstmTradeDecider(
            labels=["longbuy", "longclose", "shortbuy", "shortclose"],
            scorethresholds=(longbuy=0.7f0, longclose=0.8f0, shortbuy=0.6f0, shortclose=0.55f0),
            fallbacklabel=allclose,
        )
        @test decider.labels == [longbuy, longclose, shortbuy, shortclose]
        @test decider.scorethresholds[longbuy] == 0.7f0
        @test decider.scorethresholds[longclose] == 0.8f0
        @test decider.fallbacklabel == allclose
    end

    @testset "single sample action mapping" begin
        decider = TradingStrategy.LstmTradeDecider(scorethresholds=(longbuy=0.6f0, longclose=0.6f0, shortbuy=0.6f0, shortclose=0.6f0))

        ta1 = TradingStrategy.lstm_trade_action(decider, Float32[0.9, 0.05, 0.03, 0.02])
        @test ta1.orderlabel == longbuy

        ta2 = TradingStrategy.lstm_trade_action(decider, Float32[0.05, 0.1, 0.8, 0.05])
        @test ta2.orderlabel == shortbuy
    end

    @testset "fallback allclose uses asset type" begin
        decider = TradingStrategy.LstmTradeDecider(scorethresholds=(longbuy=0.95f0, longclose=0.95f0, shortbuy=0.95f0, shortclose=0.95f0), fallbacklabel=allclose)

        probs = Float32[0.7, 0.1, 0.1, 0.1]

        ta_up = TradingStrategy.lstm_trade_action(decider, probs; assettype=up)
        @test ta_up.orderlabel == longclose

        ta_down = TradingStrategy.lstm_trade_action(decider, probs; assettype=down)
        @test ta_down.orderlabel == shortclose

        ta_flat = TradingStrategy.lstm_trade_action(decider, probs; assettype=flat)
        @test isnothing(ta_flat.orderlabel)
    end

    @testset "batch mapping" begin
        decider = TradingStrategy.LstmTradeDecider(scorethresholds=(longbuy=0.5f0, longclose=0.5f0, shortbuy=0.5f0, shortclose=0.5f0))
        probs = Float32[
            0.8  0.1  0.1;
            0.1  0.7  0.1;
            0.05 0.1  0.75;
            0.05 0.1  0.05;
        ]
        actions = TradingStrategy.lstm_trade_actions(decider, probs)
        @test length(actions) == 3
        @test actions[1].orderlabel == longbuy
        @test actions[2].orderlabel == longclose
        @test actions[3].orderlabel == shortbuy
    end

    @testset "shape checks" begin
        decider = TradingStrategy.LstmTradeDecider()
        @test_throws AssertionError TradingStrategy.lstm_trade_action(decider, Float32[0.1, 0.2, 0.3])
        @test_throws AssertionError TradingStrategy.lstm_trade_actions(decider, rand(Float32, 3, 2))
    end

    @testset "integration with Phase-2 prediction (tensor input)" begin
        decider = TradingStrategy.LstmTradeDecider(scorethresholds=(longbuy=0.5f0, longclose=0.5f0, shortbuy=0.5f0, shortclose=0.5f0))
        model = x -> begin
            nclasses = 4
            seqlen = size(x, 2)
            batch = size(x, 3)
            y = zeros(Float32, nclasses, seqlen, batch)
            y[1, :, :] .= 5f0
            return y
        end
        X = rand(Float32, 7, 3, 4)
        result = TradingStrategy.lstm_trade_actions(decider, model, X)
        @test size(result.probs) == (4, 4)
        @test length(result.actions) == 4
        @test all(ta -> ta.orderlabel == longbuy, result.actions)
    end

    @testset "integration with contract windows" begin
        decider = TradingStrategy.LstmTradeDecider(scorethresholds=(longbuy=0.5f0, longclose=0.5f0, shortbuy=0.5f0, shortclose=0.5f0))
        model = x -> begin
            nclasses = 4
            seqlen = size(x, 2)
            batch = size(x, 3)
            y = zeros(Float32, nclasses, seqlen, batch)
            y[1, :, :] .= 4f0
            return y
        end

        df = DataFrame(
            sampleix=Int32[1, 2, 3, 4, 5, 6],
            rangeid=Int32[10, 10, 10, 11, 11, 11],
            set=["train", "train", "train", "eval", "eval", "eval"],
            target=["longbuy", "longclose", "longclose", "shortbuy", "shortclose", "shortclose"],
            pred_center=Float32[100, 101, 102, 200, 201, 202],
            pred_width=Float32[4, 6, 8, 10, 6, 2],
            longbuy=Float32[0.7, 0.4, 0.2, 0.1, 0.1, 0.1],
            shortbuy=Float32[0.1, 0.2, 0.3, 0.6, 0.4, 0.2],
            allclose=Float32[0.2, 0.4, 0.5, 0.3, 0.5, 0.7],
        )
        contract = Classify.lstm_bounds_trend_features(
            df;
            trendprobcols=[:longbuy, :shortbuy, :allclose],
            centercol=:pred_center,
            widthcol=:pred_width,
            targetcol=:target,
            setcol=:set,
            rangeidcol=:rangeid,
            rixcol=:sampleix,
        )

        result = TradingStrategy.lstm_trade_actions(decider, model, contract; seqlen=3)
        @test size(result.probs, 1) == 4
        @test size(result.probs, 2) == 2
        @test length(result.actions) == 2
        @test length(result.targets) == 2
        @test length(result.sets) == 2
        @test length(result.rangeids) == 2
        @test length(result.endrix) == 2
        @test all(ta -> ta.orderlabel == longbuy, result.actions)
    end
end
