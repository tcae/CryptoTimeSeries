using Classify, DataFrames, Test, Random

@testset "LSTM phase-2 training" begin
    Random.seed!(42)
    
    df = DataFrame(
        sampleix=Int32[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        rangeid=Int32[10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11],
        set=["train", "train", "train", "train", "eval", "eval", "train", "train", "eval", "eval", "eval", "eval"],
        target=["longbuy", "longclose", "longclose", "longbuy", "longclose", "longclose", 
                "shortbuy", "shortclose", "shortbuy", "shortclose", "shortclose", "shortbuy"],
        pred_center=Float32[100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205],
        pred_width=Float32[4, 6, 8, 5, 7, 3, 10, 6, 2, 8, 4, 6],
        longbuy=Float32[0.7, 0.4, 0.2, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        shortbuy=Float32[0.1, 0.2, 0.3, 0.05, 0.4, 0.2, 0.8, 0.3, 0.6, 0.4, 0.2, 0.7],
        allclose=Float32[0.2, 0.4, 0.5, 0.05, 0.5, 0.7, 0.1, 0.6, 0.3, 0.5, 0.7, 0.2],
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
    
    @test size(contract.features) == (7, 12)

    @testset "generic contract and hidden feature extraction" begin
        generic_contract = Classify.lstm_feature_contract(
            select(df, :sampleix, :rangeid, :set, :target, :longbuy, :shortbuy, :allclose);
            featurecols=[:longbuy, :shortbuy, :allclose],
            targetcol=:target,
            setcol=:set,
            rangeidcol=:rangeid,
            rixcol=:sampleix,
        )
        @test size(generic_contract.features) == (3, 12)
        @test generic_contract.feature_names == ["longbuy", "shortbuy", "allclose"]

        nn = Classify.model002(7, ["up", "down", "flat"], "phase_test")
        hidden = Classify.penultimatefeatures(nn, rand(Float32, 7, 5))
        lay1 = 3 * 7
        lay2 = round(Int, lay1 * 2 / 3)
        lay3 = round(Int, (lay2 + 3) / 2)
        @test size(hidden) == (lay3, 5)
        @test all(isfinite, hidden)
    end
    
    @testset "LSTM model initialization" begin
        nfeatures = 7
        seqlen = 3
        hidden_dim = 16
        
        model, labels = Classify.lstm_trade_signal_model(nfeatures, seqlen, hidden_dim)
        @test labels == ["longbuy", "longclose", "shortbuy", "shortclose"]
        
        X_test = randn(Float32, nfeatures, seqlen, 2)
        ŷ = model(X_test)
        @test size(ŷ) == (4, seqlen, 2)
    end
    
    @testset "LSTM training convergence" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=20, batchsize=4)
        @test haskey(result, :model)
        @test haskey(result, :losses)
        @test haskey(result, :eval_losses)
        @test length(result.losses) > 0
        @test length(result.losses) >= 1
        
        if length(result.losses) > 1
            @test result.losses[end] <= result.losses[1] * 1.5
        end
    end
    
    @testset "LSTM prediction" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=10, batchsize=4)
        model = result.model
        
        windows = Classify.lstm_tensor_windows(contract; seqlen=3)
        X_all = windows.X
        
        probs = Classify.predict_lstm_trade_signals(model, X_all)
        
        @test size(probs, 1) == 4
        @test size(probs, 2) == size(X_all, 3)
        
        @test all(probs .>= 0.0f0)
        @test all(probs .<= 1.0f0)
        
        col_sums = sum(probs; dims=1)
        @test all(isapprox.(vec(col_sums), 1.0f0; atol=1e-5))
    end
    
    @testset "LSTM prediction classes" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=10, batchsize=4)
        model = result.model
        
        windows = Classify.lstm_tensor_windows(contract; seqlen=3)
        X_all = windows.X
        probs = Classify.predict_lstm_trade_signals(model, X_all)
        
        pred_classes = argmax(probs; dims=1)
        
        for i in 1:size(X_all, 3)
            pred_idx = pred_classes[1, i][1]  # CartesianIndex → integer class index
            @test 1 <= pred_idx <= 4
        end
    end
    
    @testset "LSTM output structure" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=32, maxepoch=5, batchsize=4)
        @test result.labels == ["longbuy", "longclose", "shortbuy", "shortclose"]
        @test length(result.losses) <= 20
        @test all(result.losses .> 0)
    end

end
