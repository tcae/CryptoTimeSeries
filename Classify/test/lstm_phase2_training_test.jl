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

    uniqueprefix(tag::AbstractString) = tag * "_" * basename(tempname())

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
        featuremat = rand(Float32, 7, 5)
        hidden = Classify.penultimatefeatures(nn, featuremat)
        featuredf = DataFrame([Symbol("f" * string(ix)) => vec(featuremat[ix, :]) for ix in axes(featuremat, 1)])
        hidden_batched = Classify.penultimatefeatures(nn, featuredf, Symbol.(names(featuredf)); batchsize=2)
        hidden_rows = Classify.penultimatefeatures(nn, featuredf, Symbol.(names(featuredf)); batchsize=2, rows=[1, 3, 5])
        lay1 = 3 * 7
        lay2 = round(Int, lay1 * 2 / 3)
        lay3 = round(Int, (lay2 + 3) / 2)
        @test size(hidden) == (lay3, 5)
        @test size(hidden_batched) == size(hidden)
        @test hidden_batched ≈ hidden
        @test size(hidden_rows) == (lay3, 3)
        @test hidden_rows ≈ hidden[:, [1, 3, 5]]
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

    @testset "nnconverged accepts loss vectors" begin
        @test Classify.nnconverged(Float32[1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.50, 0.50, 0.51, 0.52, 0.53])
        @test !Classify.nnconverged(Float32[1.0, 0.9, 0.8, 0.7, 0.6])
    end
    
    @testset "LSTM training convergence" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=20, batchsize=4, fileprefix=uniqueprefix("lstm_convergence"))
        @test haskey(result, :model)
        @test haskey(result, :losses)
        @test haskey(result, :eval_losses)
        @test haskey(result, :checkpointfile)
        @test endswith(result.checkpointfile, ".bson")
        @test isfile(result.checkpointfile)
        @test length(result.losses) > 0
        @test length(result.losses) >= 1
        
        if length(result.losses) > 1
            @test result.losses[end] <= result.losses[1] * 1.5
        end
    end

    @testset "LSTM training resumes from checkpoint" begin
        fileprefix = basename(tempname())
        first = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=2, batchsize=4, fileprefix=fileprefix, resume=false)
        @test length(first.losses) == 2
        @test isfile(first.checkpointfile)

        resumed = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=4, batchsize=4, fileprefix=fileprefix)
        @test resumed.checkpointfile == first.checkpointfile
        @test length(resumed.losses) == 4
        @test length(resumed.eval_losses) == 4
        @test resumed.losses[1:2] == first.losses
        @test resumed.eval_losses[1:2] == first.eval_losses
    end
    
    @testset "LSTM prediction" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=10, batchsize=4, fileprefix=uniqueprefix("lstm_prediction"))
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

    @testset "LSTM streamed contract prediction" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=5, batchsize=2, fileprefix=uniqueprefix("lstm_streamed_prediction"))
        model = result.model

        windows = Classify.lstm_tensor_windows(contract; seqlen=3)
        dense_probs = Classify.predict_lstm_trade_signals(model, windows.X)
        streamed = Classify.predict_lstm_trade_signals(model, contract; seqlen=3, batchsize=2)

        @test size(streamed.probs) == size(dense_probs)
        @test all(isapprox.(streamed.probs, dense_probs; atol=1e-5))
        @test streamed.targets == windows.targets
        @test streamed.sets == windows.sets
        @test streamed.rangeids == windows.rangeids
        @test streamed.endrix == windows.endrix
    end

    @testset "LSTM multi-contract streaming" begin
        contract_a = Classify.lstm_feature_contract(
            contract.features[:, 1:6];
            feature_names=contract.feature_names,
            targets=contract.targets[1:6],
            sets=contract.sets[1:6],
            rangeids=contract.rangeids[1:6],
            rix=contract.rix[1:6],
        )
        contract_b = Classify.lstm_feature_contract(
            contract.features[:, 7:12];
            feature_names=contract.feature_names,
            targets=contract.targets[7:12],
            sets=contract.sets[7:12],
            rangeids=contract.rangeids[7:12],
            rix=contract.rix[7:12],
        )
        contracts = [contract_a, contract_b]

        combined_windows = Classify.lstm_tensor_windows(contract; seqlen=3)
        split_windows = Classify.lstm_tensor_windows(contracts; seqlen=3)

        @test size(split_windows.X) == size(combined_windows.X)
        @test all(isapprox.(split_windows.X, combined_windows.X; atol=1e-5))
        @test split_windows.targets == combined_windows.targets
        @test split_windows.sets == combined_windows.sets
        @test split_windows.rangeids == combined_windows.rangeids
        @test split_windows.endrix == combined_windows.endrix

        result = Classify.train_lstm_trade_signals!(contracts, 3; hidden_dim=16, maxepoch=5, batchsize=2, fileprefix=uniqueprefix("lstm_multi_contract"))
        pred = Classify.predict_lstm_trade_signals(result.model, contracts; seqlen=3, batchsize=2)

        @test size(pred.probs, 2) == length(split_windows.targets)
        @test pred.targets == split_windows.targets
        @test pred.sets == split_windows.sets
        @test pred.rangeids == split_windows.rangeids
        @test pred.endrix == split_windows.endrix
    end
    
    @testset "LSTM prediction classes" begin
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=16, maxepoch=10, batchsize=4, fileprefix=uniqueprefix("lstm_prediction_classes"))
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
        result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=32, maxepoch=5, batchsize=4, fileprefix=uniqueprefix("lstm_output_structure"))
        @test result.labels == ["longbuy", "longclose", "shortbuy", "shortclose"]
        @test length(result.losses) <= 20
        @test all(result.losses .> 0)
    end

end
