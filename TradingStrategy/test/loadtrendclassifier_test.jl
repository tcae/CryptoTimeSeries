@testset "loadtrendclassifier searches the active log folder second" begin
    configname = "loadtrendclassifier-" * string(time_ns())
    modelprefix = Classify.trend_nn_fileprefix(
        configname,
        Classify.trend_runtime_load_phase(EnvConfig.configmode),
    )
    logfolder = EnvConfig.logfolder()
    nn = Classify.NN(nothing, nothing, nothing, Targets.TradeLabel[], "loader test")
    Classify.setmnemonic(nn, modelprefix)
    Classify.savenn(
        nn;
        folderpath=logfolder,
        fileprefix=modelprefix,
        save_lastepoch=false,
        save_result=true,
    )

    try
        cfg = (
            configname=configname,
            featconfig=Features.Features006(),
        )
        classifier = TradingStrategy.loadtrendclassifier(cfg; mode=EnvConfig.configmode)
        @test classifier.nn.fileprefix == modelprefix
        @test classifier.required_minutes == 2
    finally
        rm(Classify.nnfilename(modelprefix; folderpath=logfolder); force=true)
    end
end
