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
    oldfolder = EnvConfig.logfolder()
    oldmode = EnvConfig.configmode
    tmpdir = mktempdir()

    try
        EnvConfig.setlogpath(tmpdir)
        EnvConfig.configmode = EnvConfig.test

        x = Float32[0f0 1f0 2f0 3f0]
        y = ["A", "A", "B", "B"]
        onehot = Float32.(Flux.onehotbatch(y, ["A", "B"]))

        nn = _reinforce_test_nn("reinforce_idx")
        wrongix = Classify._misclassified_indices(nn, x, onehot)
        @test wrongix == [3, 4]

        nn_no_reinforce = _reinforce_test_nn("no_reinforce")
        Classify.adaptnn!(nn_no_reinforce, x, y; reinforce_epochs=0)
        @test length(nn_no_reinforce.losses) == 10
        @test isapprox(nn_no_reinforce.losses[1], nn_no_reinforce.losses[2]; atol=1f-6)

        nn_reinforce = _reinforce_test_nn("with_reinforce")
        Classify.adaptnn!(nn_reinforce, x, y; reinforce_epochs=1)
        @test length(nn_reinforce.losses) == 10
        @test nn_reinforce.losses[2] > nn_reinforce.losses[1] + 1f0
        @test isapprox(nn_reinforce.losses[2], nn_reinforce.losses[end]; atol=1f-6)
    finally
        EnvConfig.configmode = oldmode
        EnvConfig.setlogpath(oldfolder)
        rm(tmpdir; force=true, recursive=true)
    end
end
