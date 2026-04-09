module BoundsEstimatorTests

using Test
using DataFrames
using Dates
using EnvConfig
using Targets
using Classify

include("../scripts/BoundsEstimator.jl")

@testset "BoundsEstimator ratio targets" begin
    phot = DataFrame(
        high=Float32[110, 220],
        low=Float32[90, 180],
        pivot=Float32[100, 200],
    )

    center_abs = (phot[!, :high] .+ phot[!, :low]) ./ 2
    width_abs = phot[!, :high] .- phot[!, :low]
    center_ratio, width_ratio = BoundsEstimator.normalize_bounds_targets(center_abs, width_abs, phot[!, :pivot])

    @test center_ratio == Float32[0.0, 0.0]
    @test width_ratio == Float32[0.2, 0.2]
end

@testset "BoundsEstimator predicted band reconstruction" begin
    pred_center = Float32[0.0, 0.01, -0.02]
    pred_width = Float32[0.2, 0.1, 0.05]
    pivot = Float32[100, 200, 50]

    lower, upper = BoundsEstimator.denormalize_predicted_bounds(pred_center, pred_width, pivot)

    @test lower ≈ Float32[90.0, 192.0, 47.75]
    @test upper ≈ Float32[110.0, 212.0, 50.25]
end

@testset "BoundsEstimator prediction cache format validation" begin
    currentdf = DataFrame(centerpred=Float32[0.0], widthpred=Float32[0.1])
    legacydf = DataFrame(
        centerpred=Float32[0.0],
        widthpred=Float32[0.1],
        bounds_prediction_format=fill(BoundsEstimator.BOUNDS_RATIO_FORMAT, 1),
    )

    @test BoundsEstimator.has_current_prediction_format(currentdf)
    @test !BoundsEstimator.has_current_prediction_format(legacydf)
end

@testset "BoundsEstimator rejects non-positive pivot" begin
    @test_throws AssertionError BoundsEstimator.denormalize_predicted_bounds(Float32[0.0], Float32[0.1], Float32[0.0])
end

@testset "BoundsEstimator LSTM contract without sampleix" begin
    merged_df = DataFrame(
        rangeid=Int32[10, 10, 10, 11, 11, 11],
        set=["train", "train", "train", "eval", "eval", "eval"],
        target=["longbuy", "longclose", "longclose", "shortbuy", "shortclose", "shortclose"],
        centerpred=Float32[100, 101, 102, 200, 201, 202],
        widthpred=Float32[4, 6, 8, 10, 6, 2],
        longbuy=Float32[0.7, 0.4, 0.2, 0.1, 0.1, 0.1],
        shortbuy=Float32[0.1, 0.2, 0.3, 0.6, 0.4, 0.2],
        allclose=Float32[0.2, 0.4, 0.5, 0.3, 0.5, 0.7],
    )

    contract = BoundsEstimator.build_lstm_contract(merged_df; trendprobcols=[:longbuy, :shortbuy, :allclose])
    windows = Classify.lstm_tensor_windows(contract; seqlen=2)

    @test size(contract.features) == (7, 6)
    @test size(windows.X) == (7, 2, 4)
    @test windows.endrix == Int32[2, 3, 5, 6]
end

@testset "BoundsEstimator bounds quality forward-window metrics" begin
    EnvConfig.init(test)

    cfg = BoundsEstimator.BoundsEstimatorConfig(
        configname="ut-bounds-quality",
        featconfig=BoundsEstimator.boundsf6config01(2),
        targetconfig=Targets.Bounds01(2),
        regressormodel=Classify.boundsregressor001,
        tradingstrategy=BoundsEstimator.tradingstrategy02(),
        startdt=DateTime("2025-01-01T00:00:00"),
        enddt=DateTime("2025-01-01T00:10:00"),
        coins=["SINE"],
    )

    mockpdf = DataFrame(
        centerpred=Float32[0, 0, 0, 0, 0],
        widthpred=Float32[0.1, 0.1, 0.1, 0.1, 0.1],
        centertarget=Float32[0, 0, 0, 0, 0],
        widthtarget=Float32[0.1, 0.1, 0.1, 0.1, 0.1],
        pivot=Float32[100, 100, 100, 100, 100],
        high=Float32[100, 106, 103, 104, 106],
        low=Float32[100, 100, 96, 96, 94],
        close=Float32[100, 100, 100, 110, 102],
        set=["train", "train", "train", "test", "test"],
        coin=["SINE", "SINE", "SINE", "SINE", "SINE"],
        rangeid=Int16[1, 1, 1, 1, 1],
        opentime=[
            DateTime("2025-01-01T00:00:00"),
            DateTime("2025-01-01T00:01:00"),
            DateTime("2025-01-01T00:02:00"),
            DateTime("2025-01-01T00:03:00"),
            DateTime("2025-01-01T00:04:00"),
        ],
    )

    # Inject deterministic prediction/results rows for this unit test.
    @eval BoundsEstimator begin
        function getboundspredictionsdf(cfg::BoundsEstimatorConfig)
            return deepcopy($mockpdf)
        end
    end

    qdf = BoundsEstimator.getboundsqualitydf(cfg)

    @test haskey(qdf, :high)
    @test haskey(qdf, :low)
    @test size(qdf.high, 1) == 2
    @test size(qdf.low, 1) == 2

    highcols = Set(Symbol.(names(qdf.high)))
    @test all(c -> c in highcols, [
        :set,
        :mae_center,
        :mae_width,
        :high_hit_within_window_pct,
        :mean_samples_to_first_high_hit,
        :mean_high_hit_gain_vs_close_pct,
        :mean_samples_to_first_high_exceed_after_window,
        :mean_high_loss_vs_close_pct_after_window,
        :rows,
    ])

    lowcols = Set(Symbol.(names(qdf.low)))
    @test all(c -> c in lowcols, [
        :set,
        :mae_center,
        :mae_width,
        :low_hit_within_window_pct,
        :mean_samples_to_first_low_hit,
        :mean_low_hit_gain_vs_close_pct,
        :mean_samples_to_first_low_exceed_after_window,
        :mean_low_loss_vs_close_pct_after_window,
        :rows,
    ])

    train_high = qdf.high[qdf.high[!, :set] .== "train", :]
    test_high = qdf.high[qdf.high[!, :set] .== "test", :]
    train_low = qdf.low[qdf.low[!, :set] .== "train", :]
    test_low = qdf.low[qdf.low[!, :set] .== "test", :]
    @test size(train_high, 1) == 1
    @test size(test_high, 1) == 1
    @test size(train_low, 1) == 1
    @test size(test_low, 1) == 1

    @test train_high[1, :rows] == 3
    @test test_high[1, :rows] == 2
    @test train_low[1, :rows] == 3
    @test test_low[1, :rows] == 2

    # Targets equal predictions in mock data.
    @test train_high[1, :mae_center] ≈ 0f0 atol=1e-6
    @test train_high[1, :mae_width] ≈ 0f0 atol=1e-6
    @test test_high[1, :mae_center] ≈ 0f0 atol=1e-6
    @test test_high[1, :mae_width] ≈ 0f0 atol=1e-6
    @test train_low[1, :mae_center] ≈ 0f0 atol=1e-6
    @test train_low[1, :mae_width] ≈ 0f0 atol=1e-6
    @test test_low[1, :mae_center] ≈ 0f0 atol=1e-6
    @test test_low[1, :mae_width] ≈ 0f0 atol=1e-6

    # high-side hits within window for window=2:
    # train rows => [true, false, true] => 66.666...
    # test rows  => [true, false]       => 50
    @test train_high[1, :high_hit_within_window_pct] ≈ (2f0 / 3f0 * 100f0) atol=1e-4
    @test test_high[1, :high_hit_within_window_pct] ≈ 50f0 atol=1e-6

    # low-side hits within window:
    # train rows => [false, false, true] => 33.333...
    # test rows  => [true, false]        => 50
    @test train_low[1, :low_hit_within_window_pct] ≈ (1f0 / 3f0 * 100f0) atol=1e-4
    @test test_low[1, :low_hit_within_window_pct] ≈ 50f0 atol=1e-6

    # first high-hit offsets (train): row1 -> 1, row3 -> 2 => mean 1.5
    @test train_high[1, :mean_samples_to_first_high_hit] ≈ 1.5 atol=1e-6
    # first low-hit offsets (train): row3 -> 2
    @test train_low[1, :mean_samples_to_first_low_hit] ≈ 2.0 atol=1e-6

    # high-hit gains (train): rows 1 and 3 both 5%.
    @test train_high[1, :mean_high_hit_gain_vs_close_pct] ≈ 5f0 atol=1e-6
    # low-hit gains (train): row 3 only, 5%.
    @test train_low[1, :mean_low_hit_gain_vs_close_pct] ≈ 5f0 atol=1e-6

    # high late exceed: row2 first exceeds at row5 (offset 3), close change 2%.
    @test train_high[1, :mean_samples_to_first_high_exceed_after_window] ≈ 3.0 atol=1e-6
    @test train_high[1, :mean_high_loss_vs_close_pct_after_window] ≈ 2f0 atol=1e-6

    # low late exceed: row1 first exceeds at row5 (offset 4), row2 at row5 (offset 3), mean 3.5.
    @test train_low[1, :mean_samples_to_first_low_exceed_after_window] ≈ 3.5 atol=1e-6
    @test train_low[1, :mean_low_loss_vs_close_pct_after_window] ≈ 2f0 atol=1e-6
end

@testset "BoundsEstimator accepts cached results without format marker" begin
    EnvConfig.init(test)

    cfg = BoundsEstimator.BoundsEstimatorConfig(
        configname="ut-bounds-cache-no-format-marker",
        featconfig=BoundsEstimator.boundsf6config01(2),
        targetconfig=Targets.Bounds01(2),
        regressormodel=Classify.boundsregressor001,
        tradingstrategy=BoundsEstimator.tradingstrategy02(),
        startdt=DateTime("2025-01-01T00:00:00"),
        enddt=DateTime("2025-01-01T00:10:00"),
        coins=["SINE"],
    )

    cached_results = DataFrame(
        centertarget=Float32[0.0, 0.01],
        widthtarget=Float32[0.1, 0.12],
        bounds_target_format=fill(BoundsEstimator.BOUNDS_RATIO_FORMAT, 2),
        pivot=Float32[100.0, 101.0],
        high=Float32[102.0, 103.0],
        low=Float32[98.0, 99.0],
        close=Float32[100.0, 100.5],
        set=["train", "eval"],
        coin=["SINE", "SINE"],
        rangeid=Int16[1, 1],
        opentime=[
            DateTime("2025-01-01T00:00:00"),
            DateTime("2025-01-01T00:01:00"),
        ],
    )
    cached_features = DataFrame(dummy=Float32[1.0, 2.0])

    EnvConfig.savedf(cached_results, BoundsEstimator.resultsfilename())
    EnvConfig.savedf(cached_features, BoundsEstimator.featuresfilename())

    @eval BoundsEstimator begin
        function getfeaturestargets(cfg::BoundsEstimatorConfig, coinix, rangeid, samplesets)
            error("unexpected cache rebuild for $(cfg.configname)")
        end
    end

    resultsdf, featuresdf = BoundsEstimator.getfeaturestargetsdf(cfg)

    @test size(resultsdf, 1) == 2
    @test size(featuresdf, 1) == 2
    @test :bounds_target_format ∉ propertynames(resultsdf)
    @test resultsdf[!, :centertarget] == cached_results[!, :centertarget]
    @test featuresdf[!, :dummy] == cached_features[!, :dummy]
end

@testset "BoundsEstimator reads Arrow subfolder caches" begin
    oldformat = EnvConfig.dfformat()
    EnvConfig.init(test)
    EnvConfig.setdfformat!(:arrow)

    cfg = BoundsEstimator.BoundsEstimatorConfig(
        configname="ut-bounds-arrow-subfolders",
        featconfig=BoundsEstimator.boundsf6config01(2),
        targetconfig=Targets.Bounds01(2),
        regressormodel=Classify.boundsregressor001,
        tradingstrategy=BoundsEstimator.tradingstrategy02(),
        startdt=DateTime("2025-01-01T00:00:00"),
        enddt=DateTime("2025-01-01T00:10:00"),
        coins=["SINE"],
    )

    try
        cached_results = DataFrame(
            centertarget=Float32[0.0, 0.01],
            widthtarget=Float32[0.1, 0.12],
            pivot=Float32[100.0, 101.0],
            high=Float32[102.0, 103.0],
            low=Float32[98.0, 99.0],
            close=Float32[100.0, 100.5],
            set=["train", "eval"],
            coin=["SINE", "SINE"],
            rangeid=Int16[1, 1],
            opentime=[DateTime("2025-01-01T00:00:00"), DateTime("2025-01-01T00:01:00")],
        )
        cached_features = DataFrame(dummy=Float32[1.0, 2.0])
        cached_predictions = DataFrame(centerpred=Float32[0.0, 0.02], widthpred=Float32[0.1, 0.11])

        EnvConfig.savedf(cached_results, joinpath("results", "all"); format=:arrow)
        EnvConfig.savedf(cached_features, joinpath("features", "all"); format=:arrow)
        EnvConfig.savedf(cached_predictions, joinpath("predictions", "maxpredictions"); format=:arrow)

        @eval BoundsEstimator begin
            function getfeaturestargets(cfg::BoundsEstimatorConfig, coinix, rangeid, samplesets)
                error("unexpected Arrow cache rebuild for $(cfg.configname)")
            end
        end

        resultsdf, featuresdf = BoundsEstimator.getfeaturestargetsdf(cfg)
        @test size(resultsdf, 1) == 2
        @test size(featuresdf, 1) == 2
        @test resultsdf[!, :centertarget] == cached_results[!, :centertarget]
        @test featuresdf[!, :dummy] == cached_features[!, :dummy]

        predictionsdf = EnvConfig.readdf(BoundsEstimator.predictionsfilename())
        @test !isnothing(predictionsdf)
        @test size(predictionsdf, 1) == 2
        @test predictionsdf[!, :centerpred] == cached_predictions[!, :centerpred]
        @test predictionsdf[!, :widthpred] == cached_predictions[!, :widthpred]
    finally
        EnvConfig.setdfformat!(oldformat)
    end
end

end # module
