using Test, Dates, DataFrames, EnvConfig, TradingStrategy, Classify, Features, Ohlcv, TestOhlcv, Targets, TSM

@testset "Missing predictions seed per-tick classification" begin
    EnvConfig.init(EnvConfig.test)
    oldcoinspath = EnvConfig.activecoinspath
    daterange = "DREPLAYSEED"
    clcfg = TradingStrategy.classifierconfig("F09-T09-P02-M011-B1")
    name = TradingStrategy.classifiername(clcfg; coinset="CS00", daterange)
    startdt = DateTime(2025, 2, 1)
    enddt = startdt + Day(20)
    TradingStrategy.DATERANGE_CONFIGS[daterange] = (startdt=startdt + Day(1), enddt=enddt)
    try
        mktempdir(EnvConfig.defaultdebugfilespath; prefix="replay-seed-") do folder
            EnvConfig.setcoinspath!(folder)
            ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
            Ohlcv.write(ohlcv)
            @test isnothing(Classify.readpredictions(name, "SINE"))
            @test_throws "missing predictions or targets" TradingStrategy.replayinput(
                clcfg, "SINE"; coinset="CS00", daterange)

            seeded = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CS00", daterange,
                seedmissing=true, usepartitions=false)
            @test names(seeded) == ["opentime", "high", "low", "close", "coin", "rangeid", "set", "target", "score", "label"]
            odf = Ohlcv.dataframe(ohlcv)
            readydt = odf[begin, :opentime] + Minute(Features.requiredminutes(TradingStrategy.featureconfig(clcfg.features)))
            readyrows = odf.opentime .>= readydt
            @test seeded.opentime == odf[readyrows, :opentime]
            @test seeded.close == odf[readyrows, :close]
            @test all(==("all"), seeded.set)
            @test all(==("SINE"), seeded.coin)
            @test all(==(0f0), seeded.score)
            @test all(==(Targets.ignore), seeded.label)
            @test all(==(Targets.ignore), seeded.target)
            @test isnothing(TradingStrategy.coinstores(clcfg, "SINE"; daterange)[3])

            windowed = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CS00", daterange,
                seedmissing=true, startdt=enddt - Minute(2), enddt=enddt)
            @test windowed.opentime == collect(enddt - Minute(2):Minute(1):enddt)
            @test all(==("all"), windowed.set)
            emptywindow = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CS00", daterange,
                seedmissing=true, startdt=enddt + Day(1), enddt=enddt + Day(2))
            @test nrow(emptywindow) == 0

            partitioned = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CS00", daterange,
                seedmissing=true, usepartitions=true)
            sdf = TradingStrategy.coinstores(clcfg, "SINE"; daterange)[3]
            @test nrow(partitioned) > 0
            @test partitioned.opentime == sdf.opentime
            @test partitioned.set == sdf.set
            @test partitioned.rangeid == sdf.rangeid
            @test all(==(0f0), partitioned.score)

            classifier = Classify.newclassifier(name, TradingStrategy.modelconfig("M011"),
                TradingStrategy.featurefactory(clcfg.features), TradingStrategy.classifierlabels(clcfg))
            runtime = TradingStrategy.TsCache(; classifier, strategy=TradingStrategy.strategytemplate("S09"))
            TradingStrategy.addclassifierbase!(runtime, ohlcv; datetime=first(windowed.opentime))
            trades = select(windowed, :opentime, :high, :low, :close, :label, :score)
            trades[!, :pair] = fill("SINE" * EnvConfig.pairquote, nrow(trades))
            for contributor in TSM.tradesdf_all_contributors()
                contributor(trades)
            end
            columns = TSM.TradesColumns(trades)
            for ix in 1:nrow(trades)
                runtime.cfg.algorithm(runtime.cfg.algorithmconfig, classifier, columns, ix)
            end
            @test all(score -> isfinite(score) && 0f0 < score <= 1f0, trades.score)
            @test isnothing(Classify.readpredictions(name, "SINE"))

            targetdf = DataFrame(opentime=odf.opentime,
                label=fill(Targets.tradelabelcode(Targets.longhold), nrow(odf)))
            EnvConfig.savedf(targetdf, clcfg.targets;
                folderpath=Targets.targetsfolder("SINE", ohlcv.quotecoin), format=:arrow,
                metadata=Dict("confighash" => TradingStrategy.targetshash(clcfg.targets)))
            predictions = DataFrame(opentime=seeded.opentime,
                label=fill(Targets.tradelabelcode(Targets.longhold), nrow(seeded)),
                score=fill(0.75f0, nrow(seeded)))
            Classify.savepredictions(name, "SINE", predictions)
            normal = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CS00", daterange)
            reused = TradingStrategy.replayinput(clcfg, "SINE"; coinset="CS00", daterange,
                seedmissing=true, usepartitions=false)
            @test nrow(normal) > 0
            @test isequal(reused, normal)
            @test all(==(0.75f0), reused.score)
        end
    finally
        delete!(TradingStrategy.DATERANGE_CONFIGS, daterange)
        EnvConfig.setcoinspath!(oldcoinspath)
        rm(EnvConfig.classifierfolder(name); recursive=true)
    end
end
