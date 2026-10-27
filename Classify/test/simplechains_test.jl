using Test, Random, Flux, Classify, EnvConfig

"""Build identically initialized dense models for backend parity tests."""
function densebackendpair()
    Random.seed!(713)
    simple = Classify.model009(7, ["A", "B", "C"], "simplechains-test")
    Random.seed!(713)
    flux = Classify.model010(7, ["A", "B", "C"], "flux-control-test")
    return simple, flux
end

@testset "SimpleChains dense alternative" begin
    Random.seed!(43)
    features = randn(Float32, 7, 19)
    targets = Flux.onehotbatch(repeat(["A", "B", "C"]; outer=7)[1:19], ["A", "B", "C"])
    weights = Float32[isodd(ix) ? 0.5 : 2 for ix in 1:19]
    for batchweights in (nothing, weights)
        simple, flux = densebackendpair()
        @test simple.model isa Classify.SimpleChainsModel
        @test Flux.trainmode!(simple.model) === simple.model
        @test Flux.testmode!(simple.model) === simple.model
        @test simple.model(features) ≈ flux.model(features) rtol=2f-5 atol=2f-6
        before = copy(simple.model.parameters)
        gradients = Flux.gradient(flux.model) do model
            logits = model(features)
            isnothing(batchweights) ? flux.lossfunc(logits, targets) :
                Classify.weightedloss(flux, logits, targets, batchweights)
        end
        for step in 1:4
            simpleloss = Classify.trainbatch!(simple, features, targets, batchweights)
            fluxloss = Classify.trainbatch!(flux, features, targets, batchweights)
            @test simpleloss ≈ fluxloss rtol=2f-5 atol=2f-6
            @test simple.model(features) ≈ flux.model(features) rtol=3f-4 atol=3f-6
            if step == 1
                offset = 0
                for layer in gradients[1].layers, derivative in (layer.weight, layer.bias)
                    count = length(derivative)
                    @test simple.model.gradient[offset+1:offset+count] ≈ vec(derivative) rtol=3f-4 atol=2f-7
                    offset += count
                end
            end
        end
        @test simple.model.parameters != before
        probabilities = Classify.predict(simple, features)
        @test size(probabilities) == (3, 19)
        @test all(isfinite, probabilities)
        @test vec(sum(probabilities; dims=1)) ≈ ones(Float32, 19)
        retained = simple.model(features)
        simple.model(2f0 .* features)
        @test retained ≈ flux.model(features) rtol=3f-4 atol=3f-6
    end
    @test_throws AssertionError Classify.model009(0, ["A"], "invalid")
    @test_throws AssertionError Classify.model010(7, String[], "invalid")

    @testset "weighted loss value and pullback" begin
        logits = Float32[1000 -1000; 999 -999; 998 -998]
        y = Flux.onehotbatch([1, 3], 1:3)
        w = Float32[1, 3]
        loss = Classify.SimpleChainsWeightedLoss(y, w)
        original = copy(logits)
        expected = sum(w .* vec(-sum(y .* Flux.logsoftmax(logits); dims=1))) / sum(w)
        @test Classify._simplechainsloss!(logits, loss, false) ≈ expected
        @test logits == original
        @test Classify._simplechainsloss!(logits, loss, true) ≈ expected
        @test logits ≈ (Flux.softmax(original) .- y) .* reshape(w ./ sum(w), 1, :)
        @test Classify.SimpleChains.target(loss) === y
        value, _, _ = loss(copy(original), Ptr{Float32}(0), Ptr{UInt8}(0))
        @test value ≈ expected
        @test_throws AssertionError Classify._simplechainsloss!(copy(original),
            Classify.SimpleChainsWeightedLoss(y, zeros(Float32, 2)), true)
        simple, _ = densebackendpair()
        @test_throws AssertionError Classify.trainbatch!(simple, features, targets[1:2, :], weights)
        @test_throws AssertionError Classify.trainbatch!(simple, features, targets, weights[1:2])
    end

    @testset "adaptation, reinforcement and checkpoint round trip" begin
        EnvConfig.init(EnvConfig.test)
        folder = mktempdir(mkpath(EnvConfig.defaultdebugfilespath))
        try
            simple, _ = densebackendpair()
            source = Classify.MatrixSampleSource(features, targets, weights)
            initial = Classify.fullsetloss(simple, source)
            Classify.adaptnn!(simple, source; maxepochs=3, batchsize=8,
                reinforce_epochs=1, folderpath=folder)
            @test length(simple.losses) == 3
            @test all(isfinite, simple.losses)
            @test Classify.isadapted(simple)
            @test isfinite(initial)
            restored = Classify.loadnn(simple.fileprefix * "-lastepoch"; folderpath=folder)
            @test restored.model.parameters == simple.model.parameters
            @test Classify.predict(restored, features) == Classify.predict(simple, features)
            @test Classify.trainbatch!(restored, features, targets, weights) ≈
                Classify.trainbatch!(simple, features, targets, weights)
            @test restored.model.parameters == simple.model.parameters
        finally
            rm(folder; recursive=true)
        end
    end
end
