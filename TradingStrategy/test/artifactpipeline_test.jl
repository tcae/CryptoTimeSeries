using Dates, Ohlcv, TestOhlcv

@testset "artifact pipeline reuses stores and classifier" begin
    EnvConfig.init(EnvConfig.test)
    oldcoinspath = EnvConfig.activecoinspath
    testfolder = joinpath(EnvConfig.debugfilesfolder, "artifactpipeline_test")
    rm(normpath(joinpath(dirname(EnvConfig.defaultlogfilespath), testfolder)); force=true, recursive=true)
    EnvConfig.setcoinspath!(testfolder)
    coins = ["SINE", "DOUBLESINE"]
    startdt = DateTime("2025-02-01T00:00:00")
    enddt = startdt + Day(20) - Minute(1)
    TradingStrategy.COINSET_CONFIGS["CSTEST"] = (coins=coins, maxepochs=2)
    TradingStrategy.DATERANGE_CONFIGS["DTEST"] = (startdt=startdt + Day(1), enddt=enddt)
    clcfg = TradingStrategy.classifierconfig("F09-T09-P02-M002-B1")
    name = TradingStrategy.classifiername(clcfg; coinset="CSTEST", daterange="DTEST")
    rm(EnvConfig.classifierfolder(name); force=true, recursive=true)
    try
        for coin in coins
            Ohlcv.write(TestOhlcv.testohlcv(coin, startdt, enddt))
        end
        cl = TradingStrategy.trainclassifier!(clcfg; coinset="CSTEST", daterange="DTEST")
        @test cl isa Classify.TrendClassifier001
        @test length(cl.nn.losses) <= 2
        @test Classify.isclassifierstored(name)
        classifierfolder = EnvConfig.classifierfolder(name)
        @test isfile(joinpath(classifierfolder, Classify.TRAINFEATURES_FILE))
        @test isfile(joinpath(classifierfolder, Classify.TRAINLABELS_FILE))
        @test !isfile(joinpath(classifierfolder, "trainindex.arrow"))
        meta = Classify.readmeta(name)
        @test meta["coins"] == join(coins, ",")
        @test meta["featureshash"] == TradingStrategy.featureshash("F09")
        nnpath = Classify.nnfilename(Classify.NN_FILEPREFIX; folderpath=EnvConfig.classifierfolder(name))
        mtime1 = stat(nnpath).mtime
        TradingStrategy.trainclassifier!(clcfg; coinset="CSTEST", daterange="DTEST")
        @test stat(nnpath).mtime == mtime1

        @test TradingStrategy.predictcoins!(clcfg, coins; coinset="CSTEST", daterange="DTEST") == coins
        fdf = TradingStrategy.coinstores(clcfg, "SINE"; daterange="DTEST")[1]
        @test size(Classify.readpredictions(name, "SINE"), 1) == size(fdf, 1)
        fullpred = DataFrame(Classify.readpredictions(name, "SINE"); copycols=true)
        Classify.savepredictions(name, "SINE", fullpred[1:(end - 500), :])
        TradingStrategy.predictcoins!(clcfg, ["SINE"]; coinset="CSTEST", daterange="DTEST")
        extpred = Classify.readpredictions(name, "SINE")
        @test extpred[!, :opentime] == fullpred[!, :opentime]
        @test extpred[!, :label] == fullpred[!, :label]
        @test extpred[!, :score] ≈ fullpred[!, :score]
        cmdf, xcmdf = TradingStrategy.evaluateclassifier(clcfg, coins; coinset="CSTEST", daterange="DTEST")
        @test size(cmdf, 1) > 0
        @test size(xcmdf, 1) > 0

        replaydf = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CSTEST", daterange="DTEST")
        @test names(replaydf) == ["opentime", "high", "low", "close", "coin", "rangeid", "set", "target", "score", "label"]
        @test size(replaydf, 1) > 0
        @test Set(string.(replaydf[!, :set])) ⊆ Set(TradingStrategy.settypes())
        windowdf = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CSTEST", daterange="DTEST", startdt=startdt + Day(15), enddt=startdt + Day(16))
        @test all(string.(windowdf[!, :set]) .== "all")
        @test windowdf[begin, :opentime] >= startdt + Day(15)

        # a strategy context reuses the trained classifier without recalculation
        strategy = TradingStrategy.strategyconfig(TradingStrategy.tsconfig("047"); coinset="CSTEST", daterange="DTEST")
        @test strategy.classifier isa Classify.TrendClassifier001
        @test strategy.algorithmconfig == TradingStrategy.strategytemplate("S04").algorithmconfig
        @test stat(nnpath).mtime == mtime1

        previouslosses = copy(cl.nn.losses)
        continued = TradingStrategy.trainclassifier!(clcfg; coinset="CSTEST", daterange="DTEST", retrain=true, reinforce_epochs=1)
        @test length(previouslosses) < length(continued.nn.losses) <= length(previouslosses) + 2
        @test continued.nn.losses[1:length(previouslosses)] == previouslosses
        @test isnothing(Classify.readpredictions(name, "SINE"))
        @test isnothing(Classify.readpredictions(name, "DOUBLESINE"))
        reused = TradingStrategy.trainclassifier!(clcfg; coinset="CSTEST", daterange="DTEST")
        @test reused.nn.losses == continued.nn.losses
    finally
        delete!(TradingStrategy.COINSET_CONFIGS, "CSTEST")
        delete!(TradingStrategy.DATERANGE_CONFIGS, "DTEST")
        EnvConfig.setcoinspath!(relpath(oldcoinspath, dirname(EnvConfig.defaultlogfilespath)))
    end
end
