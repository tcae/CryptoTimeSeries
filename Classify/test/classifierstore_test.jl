using Dates, DataFrames, Flux, Test
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv, Classify

function classifierstoretestf6()
    f6 = Features.Features006()
    Features.addgrad!(f6, window=5, offset=0)
    Features.addgrad!(f6, window=15, offset=0)
    Features.addmaxdist!(f6, window=30, offset=0)
    Features.addrelvol!(f6, short=5, long=60, offset=0)
    return f6
end

classifierstoretesttargets() = Targets.Trend04(5, 60, Targets.thresholds((longopen=0.005, longhold=0.005, shorthold=-0.005, shortopen=-0.005)))

const CLASSIFIERSTORE_PARTITION = (samplesets=["train", "test", "train", "eval"], partitionsize=60, gapsize=10, minpartitionsize=30, maxpartitionsize=120)

@testset "runtime strategy-only regressions preserve inference" begin
    EnvConfig.init(test)
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        oldpath = EnvConfig.coinspath()
        EnvConfig.setcoinspath!(folder)
        try
            startdt = DateTime("2025-02-17T13:30:00")
            ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(240))
            labels = Targets.uniquelabels(classifierstoretesttargets())
            nn = Classify.model008(Features.featurecount(classifierstoretestf6()), labels, "internal-regression-test")
            baseline = Classify.TrendClassifier001(nn; featconfig=classifierstoretestf6)
            augmented = Classify.TrendClassifier001(nn; featconfig=classifierstoretestf6)
            original = Features.configdescription(augmented.featconfig())
            Classify.addbase!(baseline, ohlcv)
            @test_logs (:info, r"Calculating missing strategy regression in memory") Classify.addbase!(augmented, ohlcv; regressionwindows=[180, 180])
            f6 = Classify.featconfig(augmented, "SINE")
            @test Features.features(f6) ≈ Features.features(Classify.featconfig(baseline, "SINE"))
            @test Features.configdescription(augmented.featconfig()) == original
            @test Classify.requiredminutes(augmented) == Classify.requiredminutes(baseline)
            @test Features.regressionat(f6, 180, startdt + Minute(240)) isa NamedTuple
            @test Classify._featurematrix(augmented, Features.features(f6)) ≈ Classify._featurematrix(baseline, Features.features(Classify.featconfig(baseline, "SINE")))
            dt = startdt + Minute(240)
            @test Classify.advice(augmented, "SINE", dt).tradelabel == Classify.advice(baseline, "SINE", dt).tradelabel
            @test Classify.advice(augmented, "SINE", dt).probability ≈ Classify.advice(baseline, "SINE", dt).probability
            other = TestOhlcv.testohlcv("DOUBLESINE", startdt, dt)
            Classify.addbase!(augmented, other; regressionwindows=[180])
            @test Classify.featconfig(augmented, "DOUBLESINE") !== f6
            path = joinpath(Features.regressionfolder("SINE", ohlcv.quotecoin), "regr180.arrow")
            @test !isfile(path)
            Features.updateregression!(ohlcv, 180)
            bytes = read(path)
            timestamp = stat(path).mtime
            cached = Classify.TrendClassifier001(nn; featconfig=classifierstoretestf6)
            @test_logs Classify.addbase!(cached, Ohlcv.ohlcvview(ohlcv, 240:240); regressionwindows=[180])
            @test read(path) == bytes
            @test stat(path).mtime == timestamp
            @test Classify.advice(cached, "SINE", dt).probability ≈ Classify.advice(baseline, "SINE", dt).probability
            extended = TestOhlcv.testohlcv("SINE", startdt, dt + Minute(1))
            Ohlcv.setdataframe!(ohlcv, DataFrame(Ohlcv.dataframe(extended)))
            Classify.supplement!(augmented)
            @test Features.regressionat(f6, 180, dt + Minute(1)) isa NamedTuple
            @test read(path) == bytes
            @test stat(path).mtime == timestamp
            incomplete = Classify.TrendClassifier001(nn; featconfig=classifierstoretestf6)
            @test_logs (:info, r"Calculating missing strategy regression in memory") Classify.addbase!(incomplete, ohlcv; regressionwindows=[180])
            @test read(path) == bytes
        finally
            EnvConfig.setcoinspath!(oldpath)
        end
    end
end

@testset "Classify classifier store" begin
    EnvConfig.init(test)
    oldcoinspath = EnvConfig.activecoinspath
    testfolder = joinpath(EnvConfig.debugfilesfolder, "classifierstore_test")
    rm(normpath(joinpath(dirname(EnvConfig.defaultlogfilespath), testfolder)); force=true, recursive=true)
    EnvConfig.setcoinspath!(testfolder)
    classifiername = "classifierstore-test"
    rm(EnvConfig.classifierfolder(classifiername); force=true, recursive=true)
    try
        startdt = DateTime("2025-02-17T13:30:00")
        enddt = startdt + Day(2) - Minute(1)
        coins = ["SINE", "DOUBLESINE"]
        ohlcvs = Dict(coin => TestOhlcv.testohlcv(coin, startdt, enddt) for coin in coins)
        labels = Targets.uniquelabels(classifierstoretesttargets())

        stores = Dict{String, Any}()
        for coin in coins
            ohlcv = ohlcvs[coin]
            fdf = Features.updatefeatures!(ohlcv, "FCS", classifierstoretestf6(); confighash="f")
            tdf = Targets.updatetargets!(ohlcv, "TCS", classifierstoretesttargets(); confighash="t")
            sdf = Classify.updatesets!(ohlcv, "PCS_DCS", CLASSIFIERSTORE_PARTITION, startdt + Hour(2), enddt; confighash="s")
            stores[coin] = (fdf, tdf, sdf)
        end

        # sets store
        sdf = stores["SINE"][3]
        @test size(sdf, 1) > 0
        @test issorted(sdf[!, :opentime])
        @test Set(sdf[!, :set]) ⊆ Set(CLASSIFIERSTORE_PARTITION.samplesets)
        @test all(sdf[!, :rangeid] .>= Classify.RANGEID_SUBRANGE_SPAN)
        @test sdf[begin, :opentime] >= startdt + Hour(2)
        @test size(Classify.updatesets!(ohlcvs["SINE"], "PCS_DCS", CLASSIFIERSTORE_PARTITION, startdt + Hour(2), enddt; confighash="s"), 1) == size(sdf, 1)
        @test_throws AssertionError Classify.readsets("SINE", ohlcvs["SINE"].quotecoin, "PCS_DCS"; confighash="other")

        # training index references feature rows of the train set with matching label index
        loadcoin(coin) = stores[coin]
        indexdf = Classify.buildtrainindex(coins, loadcoin, labels)
        @test size(indexdf, 1) > 0
        @test names(indexdf) == ["coinix", "row", "labelix"]
        for ix in 1:97:size(indexdf, 1)
            fdf, tdf, sdf = stores[coins[indexdf[ix, :coinix]]]
            dt = fdf[indexdf[ix, :row], :opentime]
            sixs = findall(==(dt), sdf[!, :opentime])
            @test (length(sixs) == 1) && (sdf[only(sixs), :set] == "train")
            @test labels[indexdf[ix, :labelix]] == Targets.TradeLabel(tdf[findfirst(==(dt), tdf[!, :opentime]), :label])
        end
        classweight = Classify.classweights([count(==(UInt8(ix)), indexdf[!, :labelix]) for ix in eachindex(labels)])
        classweightsum = [count(==(UInt8(ix)), indexdf[!, :labelix]) * classweight[ix] for ix in eachindex(labels)]
        activesums = filter(>(0f0), classweightsum)
        @test all(isapprox.(activesums, first(activesums); rtol=1f-4))

        # direct and compact memory mapped sources batch identically to the in-memory feature matrix
        featurecols = Features.requestedcolumns(classifierstoretestf6())
        featurestores = [stores[coin][1] for coin in coins]
        directsrc = Classify.ArrowIndexSource(featurestores, featurecols, indexdf; nlabels=length(labels), useweights=true)
        src = Classify.CompactMmapSource(featurestores, featurecols, indexdf;
            folderpath=EnvConfig.classifierfolder(classifiername), nlabels=length(labels), classbalancing=true)
        @test Classify.nsamples(src) == size(indexdf, 1)
        sampleixs = collect(1:Classify.nsamples(src))
        x, y, w = Classify.batch(src, sampleixs)
        expected = reduce(hcat, [Float32[stores[coins[indexdf[ix, :coinix]]][1][indexdf[ix, :row], col] for col in featurecols] for ix in sampleixs])
        @test x == expected
        @test x == first(Classify.batch(directsrc, sampleixs))
        @test vec(map(argmax, eachcol(y))) == indexdf[!, :labelix]
        @test w == classweight[indexdf[!, :labelix]]
        @test src.labelix == indexdf[!, :labelix]
        @test filesize(joinpath(EnvConfig.classifierfolder(classifiername), Classify.TRAINLABELS_FILE)) == Classify.nsamples(src)
        shuffledixs = [7, 3, 11, 1]
        xs, ys, ws = Classify.batch(src, shuffledixs)
        sortedixs = sort(shuffledixs)
        @test xs == expected[:, sortedixs]
        @test vec(map(argmax, eachcol(ys))) == indexdf[sortedixs, :labelix]
        @test xs == first(Classify.batch(directsrc, shuffledixs))

        # one adaptation implementation for memory mapped and in-memory sources
        cl = Classify.newclassifier(classifiername, Classify.model002, classifierstoretestf6, labels)
        @test cl.nn.fileprefix == Classify.NN_FILEPREFIX
        Classify.adaptnn!(cl.nn, src; maxepochs=3, folderpath=EnvConfig.classifierfolder(classifiername))
        @test length(cl.nn.losses) == 3
        @test Classify.isclassifierstored(classifiername)
        @test Classify.fullsetloss(cl.nn, src) ≈ Classify.fullsetloss(cl.nn, Classify.MatrixSampleSource(expected, Float32.(y), w)) rtol=1f-4

        Classify.writemeta(classifiername, Dict("features" => "FCS", "labels" => join(string.(labels), ",")))
        @test Classify.readmeta(classifiername)["features"] == "FCS"
        loaded = Classify.loadclassifier(classifiername; featconfig=classifierstoretestf6)
        @test loaded.nn.labels == labels

        # predictions and evaluation
        fdf, tdf, sdf = stores["SINE"]
        preddf = Classify.predictfeatures(loaded.nn, fdf, featurecols; chunksize=100)
        @test size(preddf, 1) == size(fdf, 1)
        @test all(0f0 .< preddf[!, :score] .<= 1f0)
        Classify.savepredictions(classifiername, "SINE", preddf)
        @test Classify.readpredictions(classifiername, "SINE")[!, :label] == preddf[!, :label]
        evaldf = Classify.evaluationdf(preddf, tdf, sdf, unique(CLASSIFIERSTORE_PARTITION.samplesets))
        @test size(evaldf, 1) > 0
        @test all(evaldf[!, :target] .!= Targets.ignore)
        cm = Classify.aggregateconfusion([Classify.confusionmatrix(evaldf, labels), Classify.confusionmatrix(evaldf, labels)])
        @test sum(cm[!, :allpredicted]) == 2 * size(evaldf, 1)
        xcm = Classify.aggregateextendedconfusion([Classify.extendedconfusionmatrix(evaldf, labels)])
        @test all(in(names(xcm)), ["set", "pred_label", "bin", "tp", "ppv%"])
    finally
        EnvConfig.setcoinspath!(relpath(oldcoinspath, dirname(EnvConfig.defaultlogfilespath)))
    end
end
