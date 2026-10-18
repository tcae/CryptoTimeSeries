using Flux
using EnvConfig

function _reinforce_test_nn(name::AbstractString)
    model = Dense(1 => 2)
    model.weight .= 0f0
    model.bias .= Float32[2f0, -2f0] # Always predicts label index 1.
    optim = Flux.setup(Flux.Adam(0f0), model)
    return Classify.NN(model, optim, Flux.logitcrossentropy, ["A", "B"], "reinforce-test", name, name)
end

@testset "Classify adaptnn reinforce_epochs" begin
    tmpdir = mktempdir()
    try
        x = Float32[0f0 1f0 2f0 3f0]
        y = ["A", "A", "B", "B"]

        nn = _reinforce_test_nn("reinforce_idx")
        @test Classify.misclassifiedindices(nn, Classify.MatrixSampleSource(nn, x, y)) == [3, 4]

        nn_no_reinforce = _reinforce_test_nn("no_reinforce")
        Classify.adaptnn!(nn_no_reinforce, Classify.MatrixSampleSource(nn_no_reinforce, x, y); maxepochs=10, reinforce_epochs=0, folderpath=tmpdir)
        @test length(nn_no_reinforce.losses) == 10
        @test isapprox(nn_no_reinforce.losses[1], nn_no_reinforce.losses[2]; atol=1f-6)
        @test isfile(Classify.nnfilename("no_reinforce"; folderpath=tmpdir))

        nn_reinforce = _reinforce_test_nn("with_reinforce")
        Classify.adaptnn!(nn_reinforce, Classify.MatrixSampleSource(nn_reinforce, x, y); maxepochs=10, reinforce_epochs=1, folderpath=tmpdir)
        @test length(nn_reinforce.losses) == 10
        @test isapprox(nn_reinforce.losses[2], nn_reinforce.losses[1]; atol=1f-6)
        @test isapprox(nn_reinforce.losses[2], nn_reinforce.losses[end]; atol=1f-6)

        # weighted full set loss equals the weighted mean of per-sample losses
        w = Float32[1, 1, 3, 3]
        src = Classify.MatrixSampleSource(nn, x, y, w)
        losses = Classify.samplelosses(nn, nn.model(x), Float32.(Flux.onehotbatch(y, nn.labels)))
        @test Classify.fullsetloss(nn, src; chunksize=3) ≈ sum(w .* losses) / sum(w)
        @test Classify.classweights([2, 0, 6]) ≈ Float32[2, 0, 2/3]
    finally
        rm(tmpdir; force=true, recursive=true)
    end
end
