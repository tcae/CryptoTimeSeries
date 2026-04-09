using EnvConfig

@testset "Classify storage honors Arrow format" begin
    oldfolder = EnvConfig.logfolder()
    oldformat = EnvConfig.dfformat()
    tmpdir = mktempdir()

    try
        EnvConfig.setlogpath(tmpdir)
        EnvConfig.setdfformat!(:arrow)

        simdf = DataFrame(asset=["BTC"], gain=Float32[0.1f0], opentime=[DateTime(2024, 1, 1)])
        Classify.writesimulation(simdf)
        simpath = EnvConfig.tablepath("ClassifierTradesim"; folderpath=EnvConfig.logfolder(), format=:arrow)
        @test isfile(simpath)

        loadedsim = Classify.readsimulation()
        @test size(loadedsim) == size(simdf)
        @test loadedsim[!, :asset] == simdf[!, :asset]
        @test loadedsim[!, :gain] == simdf[!, :gain]

        predf = DataFrame(
            opentime=[DateTime(2024, 1, 1), DateTime(2024, 1, 1, 0, 1)],
            pivot=Float32[1.0f0, 2.0f0],
            longbuy=Float32[0.7f0, 0.2f0],
            allclose=Float32[0.3f0, 0.8f0],
            targets=categorical(["longbuy", "allclose"]),
        )
        Classify.savepredictions(predf, "TEST_predictions")
        predpath = EnvConfig.tablepath("TEST_predictions"; folderpath=EnvConfig.logfolder(), format=:arrow)
        @test isfile(predpath)

        loadedpred = Classify.loadpredictions("TEST_predictions")
        @test loadedpred[!, :pivot] == predf[!, :pivot]
        @test string.(loadedpred[!, :targets]) == string.(predf[!, :targets])

        nn = Classify.NN(nothing, nothing, nothing, ["center"], "demo", "demo", "demo")
        nn.losses = Float32[1.0f0, 0.5f0]
        Classify.savelosses(nn)
        losspath = EnvConfig.tablepath("losses_demo"; folderpath=EnvConfig.logfolder(), format=:arrow)
        @test isfile(losspath)

        nn.losses = Float32[]
        Classify.loadlosses!(nn)
        @test nn.losses == Float32[1.0f0, 0.5f0]
    finally
        EnvConfig.setdfformat!(oldformat)
        EnvConfig.setlogpath(oldfolder)
        rm(tmpdir; force=true, recursive=true)
    end
end
