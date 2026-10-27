using Test
include("../scripts/benchmark_model008.jl")

@testset "model008 benchmark helpers" begin
    Random.seed!(713)
    labels = Targets.uniquelabels(TradingStrategy.targetconfig("T24"))
    nn = Classify.model010(7, labels, "benchmark-test")
    features = randn(Float32, 7, 19)
    targets = Flux.onehotbatch([labels[mod1(ix, length(labels))] for ix in 1:19], labels)
    source = Classify.MatrixSampleSource(features, targets, ones(Float32, 19))
    @test benchmarkepoch!(nn, source, collect(1:19), MersenneTwister(713), 8) === nn
    metrics = benchmarkquality(nn, source, labels)
    @test 0 <= metrics.accuracy <= 1
    @test isfinite(metrics.crossentropy)
    @test Set(keys(metrics.entries)) == Set(string.([Targets.longopen, Targets.shortopen]))
    @test benchmarkinference(nn, features; repeats=2) > 0
    @test_throws AssertionError benchmarkmodel008(epochs=0)
    @test_throws AssertionError benchmarkmodel008(batchnorm=true, batchsize=1)

    dates = [DateTime("2025-01-01") + Minute(ix) for ix in 1:19]
    codes = [Targets.tradelabelcode(labels[mod1(ix, length(labels))]) for ix in 1:19]
    stores = Dict("SINE" => (DataFrame(opentime=dates, x=features[1, :]),
        DataFrame(opentime=dates, label=codes), DataFrame(opentime=dates, set=fill("train", 19))))
    for balancing in (false, true)
        data = benchmarksource(stores, ["SINE"], labels, ["x"], "train"; classbalancing=balancing)
        @test data.X == features[1:1, :]
        @test data.Y == targets
        @test isnothing(data.W) == !balancing
    end
    @test_throws AssertionError benchmarksource(stores, ["SINE"], labels, ["x"], "test"; classbalancing=true)
end
