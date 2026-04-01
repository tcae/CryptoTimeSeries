using Classify, DataFrames, Test

@testset "LSTM data contract" begin
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

    @test size(contract.features) == (7, 6)
    @test contract.feature_names == [
        "trend_prob_longbuy",
        "trend_prob_shortbuy",
        "trend_prob_allclose",
        "center",
        "width",
        "lower",
        "upper",
    ]

    lower = contract.features[6, :]
    upper = contract.features[7, :]
    @test all(lower .<= upper)

    windows = Classify.lstm_tensor_windows(contract; seqlen=2)
    @test size(windows.X) == (7, 2, 4)
    @test length(windows.targets) == 4
    @test length(windows.endrix) == 4
    @test windows.endrix == Int32[2, 3, 5, 6]
end
