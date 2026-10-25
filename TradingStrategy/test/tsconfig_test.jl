using Dates, Ohlcv, TestOhlcv, Xch

@testset "configuration registries" begin
    EnvConfig.init(EnvConfig.test)
    for (name, cfg) in TradingStrategy.TS_CONFIGS
        @test cfg.configname == name
        @test keys(cfg) == (:configname, :classifier, :strategy)
        @test haskey(TradingStrategy.CLASSIFIER_CONFIGS, cfg.classifier)
        @test TradingStrategy.strategytemplate(cfg.strategy) isa TradingStrategy.StrategyConfig
        @test isnothing(TradingStrategy.strategytemplate(cfg.strategy).classifier)
        @test TradingStrategy.tsconfig(name).configname == name
    end

    @testset "normalized features preserve raw strategy regression gates" begin
        EnvConfig.init(EnvConfig.test)
        startdt = DateTime("2025-02-17T13:30:00")
        ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Hour(8))
        f6 = TradingStrategy.featureconfig("F16")
        Features.setbase!(f6, ohlcv; usecache=false)
        labels = TradingStrategy.classifierlabels(TradingStrategy.classifierconfig(TradingStrategy.tsconfig("056").classifier))
        nn = Classify.model008(Features.featurecount(f6), labels, "nrg-regression-gates")
        cl = Classify.TrendClassifier001(nn; featconfig=TradingStrategy.trendf6config16)
        cl.bc[ohlcv.base] = (ohlcv=ohlcv, featcfg=f6)
        odf = Ohlcv.dataframe(ohlcv)
        dt = odf[end, :opentime]
        price = odf[end, :pivot]
        pair = Xch.tradingpairkey(ohlcv.base, ohlcv.quotecoin)
        df = DataFrame(opentime=[dt], close=[price], pair=[pair])
        for contributor in TSM.tradesdf_all_contributors()
            contributor(df)
        end
        df[1, :opentime] = dt
        df[1, :close] = price
        TSM.setcategorical!(df[!, :pair], 1, pair)
        cols = TSM.TradesColumns(df)
        cfg = TradingStrategy.GainLimitReversalConfig(maxwindow=240)
        regr = Features.regressionat(f6, 240, dt)
        @test TradingStrategy._relative_to_regression(cfg, cl, cols, 1) ≈ (price - regr.regry) / regr.regry
        @test TradingStrategy._regression_slope_gain_per_hour(cfg, cl, cols, 1) ≈ Features.relativegain(regr.regry, regr.grad, 60)
        @test f6.fdfno[end, "nrg+240"] ≈ regr.grad / Features.regressionmeanpivot(regr.regry, regr.grad, 240)
    end
    for (basename, clcfg) in TradingStrategy.CLASSIFIER_CONFIGS
        @test TradingStrategy.classifierbasename(clcfg) == basename
        @test TradingStrategy.featureconfig(clcfg.features) isa Features.Features006
        @test TradingStrategy.modelconfig(clcfg.model) isa Function
        @test !isempty(TradingStrategy.classifierlabels(clcfg))
        @test TradingStrategy.partitionconfig(clcfg.partition).samplesets == TradingStrategy.partitionconfig02().samplesets
    end

    # a content hash identifies exactly one registry id to avoid recalculating identical artifacts
    featurehashes = [TradingStrategy.featureshash(id) for id in keys(TradingStrategy.FEATURE_CONFIGS)]
    targethashes = [TradingStrategy.targetshash(id) for id in keys(TradingStrategy.TARGET_CONFIGS)]
    @test allunique(featurehashes)
    @test allunique(targethashes)
    @test TradingStrategy.featureshash("F09") == TradingStrategy.featureshash("f09")
    @test length(TradingStrategy.featureshash("F09")) == 40
    @test TradingStrategy.setshash("P02", "D00") != TradingStrategy.setshash("P02", "D01")
    @test TradingStrategy.setsdescription("P02", "D00") == "samplesets=train,test,train,train,eval,train;partitionsize=1440;gapsize=480;minpartitionsize=720;maxpartitionsize=2880;startdt=2025-01-17T20:56:00;enddt=2025-08-10T15:00:00"

    ts046 = TradingStrategy.tsconfig("ts046")
    @test ts046.classifier == "F09-T09-P02-M002-B1"
    @test TradingStrategy.tsconfig("046") == ts046
    # strategy variants share the classifier and therefore its artifacts
    @test TradingStrategy.tsconfig("047").classifier == ts046.classifier
    @test TradingStrategy.tsconfig("047").strategy != ts046.strategy
    @test TradingStrategy.tsclassifiername(ts046; coinset="CS01", daterange="D01") == "F09-T09-P02-M002-B1-CS01-D01"
    @test TradingStrategy.defaultcoinset() == "CS00"
    @test TradingStrategy.defaultdaterange() == "D00"
    @test TradingStrategy.tsconfigsource(ts046) == "tradingstrategy:046/F09-T09-P02-M002-B1/S03"
    @test TradingStrategy.classifierbasename(TradingStrategy.clconfig("F01", "T01", "M002"; classbalancing=false, variant="x")) == "F01-T01-P02-M002-B0-x"

    ts051 = TradingStrategy.tsconfig("ts051")
    cl051 = TradingStrategy.classifierconfig(ts051.classifier)
    @test cl051 == TradingStrategy.clconfig("F10", "T25", "M002")
    @test ts051.strategy == "S08"
    @test TradingStrategy.tsclassifiername(ts051; coinset="CS01", daterange="D01") == "F10-T25-P02-M002-B1-CS01-D01"
    target051 = TradingStrategy.targetconfig(cl051.targets)
    @test target051 isa Targets.Trend05
    @test (target051.leadregr, target051.supportregr, target051.trendregr) == (15, 60, 12*60)
    @test !isempty(Targets._trend05_requiredfeatures(TradingStrategy.featureconfig(cl051.features), target051))

    for (newid, oldid, featuresid) in [("054", "038", "F16"), ("055", "046", "F16"),
                                     ("056", "052", "F16"), ("057", "053", "F17")]
        newts = TradingStrategy.tsconfig(newid)
        oldts = TradingStrategy.tsconfig(oldid)
        newcl = TradingStrategy.classifierconfig(newts.classifier)
        oldcl = TradingStrategy.classifierconfig(oldts.classifier)
        @test newcl.features == featuresid
        @test newcl.model == "M008"
        @test newcl.targets == oldcl.targets
        @test newcl.partition == oldcl.partition
        @test newcl.classbalancing == oldcl.classbalancing
        @test newts.strategy == oldts.strategy
        oldcols = Features.requestedcolumns(TradingStrategy.featureconfig(oldcl.features))
        newcols = Features.requestedcolumns(TradingStrategy.featureconfig(newcl.features))
        @test newcols == replace.(oldcols, r"^rg\+" => "nrg+")
        @test TradingStrategy.featureshash(newcl.features) != TradingStrategy.featureshash(oldcl.features)
    end

    meta = TradingStrategy.classifiermeta(TradingStrategy.classifierconfig(ts046.classifier); coinset="CS00", daterange="D00")
    @test meta["featureshash"] == TradingStrategy.featureshash("F09")
    @test meta["labels"] == join(string.(Targets.uniquelabels(TradingStrategy.targetconfig("T09"))), ",")

    @test_throws AssertionError TradingStrategy.tsconfig("doesnotexist")
    @test_throws AssertionError TradingStrategy.featureconfig("F02")
end
