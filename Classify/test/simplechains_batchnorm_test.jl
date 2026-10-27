using Test, Random, Flux, Classify, EnvConfig

"""Build a native and Flux BatchNorm with nontrivial affine parameters and state."""
function batchnormtestpair(channels::Int)
    flux = Flux.BatchNorm(channels)
    flux.β .= range(-0.2f0, 0.3f0; length=channels)
    flux.γ .= range(-0.7f0, 1.3f0; length=channels)
    flux.μ .= 0.4f0
    flux.σ² .= 1.7f0
    native = Classify.SimpleChainsBatchNorm(channels)
    native.mean .= flux.μ
    native.variance .= flux.σ²
    parameters = vcat(flux.β, flux.γ)
    chain = Classify.SimpleChains.SimpleChain(channels, (native,))
    return (; flux, native, parameters, chain)
end

"""Compare BatchNorm parameter and input gradients against Flux, with explicit mode selection."""
function checkbatchnormgradients(training::Bool, observations::Int; constant::Bool=false)
    pair = batchnormtestpair(3)
    pair.native.training = training
    training ? Flux.trainmode!(pair.flux) : Flux.testmode!(pair.flux)
    input = randn(Float32, 3, observations)
    constant && (input[2, :] .= 0.25f0)
    targets = Flux.onehotbatch([mod1(ix, 3) for ix in 1:observations], 1:3)
    weights = ones(Float32, observations)
    weights[1] = 3f0
    loss = Classify.SimpleChainsWeightedLoss(targets, weights)
    expected, gradients = Flux.withgradient(pair.flux, input) do layer, values
        sum(weights .* vec(-sum(targets .* Flux.logsoftmax(layer(values)); dims=1))) / sum(weights)
    end
    parametergradient = similar(pair.parameters)
    inputgradient = similar(input)
    measured = Classify.SimpleChains.valgrad!((inputgradient, parametergradient),
        Classify.SimpleChains.add_loss(pair.chain, loss), input, pair.parameters)
    @test measured ≈ expected rtol=2f-5 atol=2f-6
    @test parametergradient ≈ vcat(gradients[1].β, gradients[1].γ) rtol=3f-4 atol=2f-5
    @test inputgradient ≈ gradients[2] rtol=3f-4 atol=3f-5
    @test pair.native.mean ≈ pair.flux.μ rtol=2f-5 atol=1f-6
    @test pair.native.variance ≈ pair.flux.σ² rtol=2f-5 atol=1f-6
    return pair
end

@testset "Native SimpleChains BatchNorm" begin
    Random.seed!(713)
    for training in (false, true), observations in (2, 17), constant in (false, true)
        checkbatchnormgradients(training, observations; constant)
    end
    checkbatchnormgradients(false, 1)

    @testset "forward, initialization and mode contracts" begin
        pair = batchnormtestpair(3)
        native = pair.native
        @test !Classify.SimpleChains.parameter_free(native)
        @test Classify.SimpleChains.numparam(native, (3, 7)) == (6, (3, 7))
        initialized = Classify.SimpleChains.init_params(pair.chain)
        @test collect(initialized) == Float32[0, 0, 0, 1, 1, 1]
        input = randn(Float32, 3, 17)
        for training in (true, false)
            native.training = training
            training ? Flux.trainmode!(pair.flux) : Flux.testmode!(pair.flux)
            meanbefore = copy(native.mean)
            variancebefore = copy(native.variance)
            @test pair.chain(input, pair.parameters) ≈ pair.flux(input) rtol=2f-5 atol=2f-6
            @test native.mean ≈ pair.flux.μ rtol=2f-5 atol=1f-6
            @test native.variance ≈ pair.flux.σ² rtol=2f-5 atol=1f-6
            if !training
                @test native.mean == meanbefore
                @test native.variance == variancebefore
            end
        end
        @test_throws AssertionError Classify.SimpleChainsBatchNorm(0)
        @test_throws AssertionError Classify.SimpleChainsBatchNorm(3; eps=0f0)
        @test_throws AssertionError Classify.SimpleChainsBatchNorm(3; momentum=2f0)
        @test_throws AssertionError Classify.SimpleChains.numparam(native, (2, 3))
        native.training = true
        @test_throws AssertionError pair.chain(input[:, 1:1], pair.parameters)
    end

    @testset "M011 matches M008 initialization and short training" begin
        features = randn(Float32, 7, 19)
        labels = ["A", "B", "C", "D", "E"]
        targets = Flux.onehotbatch([labels[mod1(ix, 5)] for ix in 1:19], labels)
        for weights in (nothing, Float32[isodd(ix) ? 1 : 3 for ix in 1:19])
            Random.seed!(713)
            native = Classify.model011(7, labels, "native-bn-test")
            Random.seed!(713)
            flux = Classify.model008(7, labels, "flux-bn-test")
            @test length(native.model.parameters) == 785
            Flux.testmode!(flux.model)
            @test native.model(features) ≈ flux.model(features) rtol=2f-5 atol=2f-6
            Flux.trainmode!(native.model)
            Flux.trainmode!(flux.model)
            gradients = Flux.gradient(flux.model) do model
                logits = model(features)
                isnothing(weights) ? flux.lossfunc(logits, targets) : Classify.weightedloss(flux, logits, targets, weights)
            end
            # The gradient reference forward updated Flux statistics once.
            for index in (1, 3, 5, 7)
                native.model.chain.layers[index].mean .= flux.model.layers[index].μ
                native.model.chain.layers[index].variance .= flux.model.layers[index].σ²
            end
            for step in 1:3
                nativeloss = Classify.trainbatch!(native, features, targets, weights)
                fluxloss = Classify.trainbatch!(flux, features, targets, weights)
                @test nativeloss ≈ fluxloss rtol=2f-4 atol=2f-5
                if step == 1
                    offset = 0
                    for (layer, derivative) in zip(flux.model.layers, gradients[1].layers)
                        arrays = layer isa Flux.BatchNorm ? (derivative.β, derivative.γ) : (derivative.weight, derivative.bias)
                        for array in arrays
                            count = length(array)
                            @test native.model.gradient[offset+1:offset+count] ≈ vec(array) rtol=5f-4 atol=3f-6
                            offset += count
                        end
                    end
                end
            end
            Flux.testmode!(native.model)
            Flux.testmode!(flux.model)
            @test native.model(features) ≈ flux.model(features) rtol=5f-3 atol=2f-3
            @test all(layer -> !layer.training, native.model.chain.layers[[1, 3, 5, 7]])
            for index in (1, 3, 5, 7)
                @test native.model.chain.layers[index].mean ≈ flux.model.layers[index].μ rtol=3f-3 atol=3f-4
                @test native.model.chain.layers[index].variance ≈ flux.model.layers[index].σ² rtol=3f-3 atol=3f-4
            end
        end
    end

    @testset "checkpoint statistics and optimizer continuation" begin
        EnvConfig.init(EnvConfig.test)
        folder = mktempdir(mkpath(EnvConfig.defaultdebugfilespath))
        try
            nn = Classify.model011(3, ["A", "B"], "native-bn-checkpoint")
            input = randn(Float32, 3, 16)
            targets = Flux.onehotbatch(repeat(["A", "B"]; outer=8), nn.labels)
            source = Classify.MatrixSampleSource(input, targets)
            Classify.adaptnn!(nn, source; maxepochs=2, batchsize=8, reinforce_epochs=1, folderpath=folder)
            restored = Classify.loadnn(nn.fileprefix * "-lastepoch"; folderpath=folder)
            Flux.testmode!(restored.model)
            @test Classify.predict(restored, input) == Classify.predict(nn, input)
            for index in (1, 3, 5, 7)
                @test restored.model.chain.layers[index].mean == nn.model.chain.layers[index].mean
                @test restored.model.chain.layers[index].variance == nn.model.chain.layers[index].variance
            end
            Flux.trainmode!(nn.model)
            Flux.trainmode!(restored.model)
            @test Classify.trainbatch!(nn, input, targets, nothing) == Classify.trainbatch!(restored, input, targets, nothing)
            @test nn.model.parameters == restored.model.parameters
        finally
            rm(folder; recursive=true)
        end
    end
end
