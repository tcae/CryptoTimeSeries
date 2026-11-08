module ScriptHelpTests

using Test, Dates, Statistics
using DataFrames
using Classify, Targets, Distributions, EnvConfig, Xch, Features, Ohlcv, TestOhlcv, TradingStrategy

include("../scripts/BoundsEstimator.jl")
include("../scripts/TrendDetector.jl")
include("../scripts/RegimeTraining.jl")
include("../scripts/refresh.jl")

@testset "script help text includes defaults" begin
    bhelp = BoundsEstimator.boundsestimatorhelp()
    @test occursin("help, --help, -h", bhelp)
    @test occursin("config=<configname>", bhelp)
    @test occursin("test startdt: `2025-01-17T20:56:00`", bhelp)

    thelp = TrendDetector.trenddetectorhelp()
    @test occursin("TS_CONFIGS", thelp)
    @test occursin("stage=<stores|train|predict|evaluate|gain>", thelp)
    @test occursin("help, --help, -h", thelp)

end

@testset "RegimeTraining slices complete shared stores" begin
    startdt = DateTime("2025-02-01T00:00:00")
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(1800))
    history = DataFrame(Ohlcv.dataframe(ohlcv)[101:1400,:];copycols=true)
    featurefactory = RegimeTraining.syntheticfeatures
    targetfactory = RegimeTraining.synthetictarget

    features = featurefactory()
    Features.setbase!(features,ohlcv;usecache=false)
    required = Features.requiredminutes(features)
    featurestore = DataFrame(features.fdf[(required-features.maxoffset):end,
        vcat(["opentime"],Features.requestedcolumns(features))];copycols=true)

    target = targetfactory()
    regression = Targets.regressionfeatures(target)
    Features.setbase!(regression,ohlcv;usecache=false)
    Targets.setbase!(target,regression)
    codes = Targets.tradelabelcode.(Targets.labels(target))
    codes[(Targets.finallabelcount(target,length(codes))+1):end] .= Targets.tradelabelcode(Targets.ignore)
    targetstore = DataFrame(opentime=copy(Ohlcv.dataframe(ohlcv).opentime),label=codes)

    selectedfeatures, selectedtargets = TradingStrategy.sharedstorewindow(history,
        featurestore,targetstore,featurefactory,targetfactory,"SINE")
    @test selectedfeatures.opentime == history.opentime[Features.requiredminutes(featurefactory()):end]
    @test selectedtargets.opentime == history.opentime
    lastfinal = nrow(history) - Targets.lookaheadminutes(targetfactory())
    @test all(selectedtargets[(lastfinal+1):end,:label] .== Targets.tradelabelcode(Targets.ignore))
    mktempdir() do folder
        shared = Dict("SINE"=>(selectedfeatures,selectedtargets))
        stores = TradingStrategy.experimentstores!(Dict("SINE"=>history),featurefactory,
            targetfactory,folder;sharedstores=shared)
        @test stores.featurestores[1] == selectedfeatures
        @test stores.targetstores[1] == selectedtargets
        reused = TradingStrategy.experimentstores!(Dict("SINE"=>history),featurefactory,
            targetfactory,folder;sharedstores=shared)
        @test reused.featurestores[1] == selectedfeatures
        @test reused.targetstores[1] == selectedtargets
    end
end

@testset "Refresh config parsing" begin
    @test Refresh.parseoptions(["ts=063"])==Dict("ts"=>"063")
    @test occursin("ts=<id,...>",Refresh.helptext())
    @test Refresh.regimeconfigs("062, 066,067")==["062","066","067"]
    for raw in ("","062,","062,062","062,059","unknown")
        @test_throws AssertionError Refresh.regimeconfigs(raw)
    end
    @test_throws AssertionError Refresh.main(["ts=059"])
    @test_throws AssertionError Refresh.main(["ts=063","xch=KrakenSpot"])
    @test Refresh.parseoptions(["features=F19,F20", "targets=T24", "pairs=BTC-USDT,ETH"]) ==
        Dict("features"=>"F19,F20", "targets"=>"T24", "pairs"=>"BTC-USDT,ETH")
    @test_throws AssertionError Refresh.parseoptions(["pair=BTC-USDT"])
    @test_throws AssertionError Refresh.parseoptions(["unknown=value"])
    @test_throws AssertionError Refresh.parseoptions(["features=F19", "features=F20"])
    @test Refresh.configids("f19,F20", "features") == ["F19", "F20"]
    @test_throws AssertionError Refresh.configids("F19,F19", "features")
    @test Refresh.parseoptions(["partitions=P02_D01, P02_D00"]) ==
        Dict("partitions"=>"P02_D01, P02_D00")
    @test Refresh.configids("p02_d01, P02_D00", "partitions") == ["P02_D01", "P02_D00"]
    for raw in ("", "P02", "P02_D01_extra", "_D01", "P02_", "P99_D01", "P02_D99", "P02_D01,p02_d01", "P02_D01,")
        @test_throws AssertionError Refresh.configids(raw, "partitions")
    end
    @test occursin("partitions=<Pxx_Dxx,...>", Refresh.helptext())
    @test_throws AssertionError Refresh.main(String[])
    @test_throws AssertionError Refresh.main(["partitions=P99_D01"])
    @test Refresh.exchangecfg("krakenspot") == (Xch.EXCHANGE_KRAKENSPOT, "USDC")
    @test Refresh.resolvepairs("btc-usdt,eth", ["BTC-USDT", "ETH-USDT"], "USDT") ==
        ["BTC-USDT", "ETH-USDT"]
    @test_throws AssertionError Refresh.resolvepairs("BTC", ["ETH-USDT"], "USDT")
    @test Refresh.main(["features=F19", "help"]) === nothing
    @test Refresh.main(["--help"]) === nothing
end

@testset "Refresh RegimeTraining eligibility output" begin
    sets = Dict("GOOD"=>(train=1,eval=1,test=1),
        "NO_TEST"=>(train=1,eval=1,test=0))
    io = IOBuffer()
    @test Refresh.reportregimecoins(sets,["GOOD","NO_TEST","CAMP"],"063","D01";io)===nothing
    report = String(take!(io))
    @test occursin("RegimeTraining eligibility: ts=063, daterange=D01",report)
    @test occursin("CAMP: excluded (insufficient window/context history)",report)
    @test occursin("NO_TEST: train=1 eval=1 test=0 excluded",report)
    @test occursin("GOOD: train=1 eval=1 test=1 eligible",report)
    @test occursin("traincoins() = [\"GOOD\"]",report)
    io = IOBuffer()
    @test Refresh.reportregimecoins(empty(sets),["CAMP"],"063","D01";io)===nothing
    @test occursin("traincoins() = String[]",String(take!(io)))
end

@testset "Common eligibility requires every configuration and population" begin
    complete = (train=1,eval=1,test=1)
    partial = (train=1,eval=1,test=0)
    coins = ["COMMON","ONLY_FIRST","ONLY_SECOND","SHORT","NO_EVAL"]
    populations = Dict(
        "062"=>Dict("COMMON"=>complete,"ONLY_FIRST"=>complete,"ONLY_SECOND"=>partial,
            "SHORT"=>complete,"NO_EVAL"=>(train=1,eval=0,test=1)),
        "064"=>Dict("COMMON"=>complete,"ONLY_FIRST"=>partial,"ONLY_SECOND"=>complete,
            "NO_EVAL"=>complete))
    io = IOBuffer()
    @test Refresh.reportcommonregimecoins(populations,["062","064"],coins,"D01";io)===nothing
    report = String(take!(io))
    common = last(split(report,"Common RegimeTraining eligibility:"))
    @test occursin("ts=062,064, daterange=D01",common)
    @test occursin("traincoins() = [\"COMMON\"]",common)
    @test coins==["COMMON","ONLY_FIRST","ONLY_SECOND","SHORT","NO_EVAL"]
    populations["064"] = empty(populations["064"])
    io = IOBuffer()
    @test Refresh.reportcommonregimecoins(populations,["062","064"],coins,"D01";io)===nothing
    @test endswith(String(take!(io)),"traincoins() = String[]\n")
end

@testset "Refresh full RegimeTraining preparation without adaptation" begin
    EnvConfig.init(EnvConfig.test)
    oldcoinspath = EnvConfig.activecoinspath
    oldquote = EnvConfig.pairquote
    startdt = DateTime(2025,1,1)
    TradingStrategy.DATERANGE_CONFIGS["DREGIMETEST"] =
        (startdt=startdt,enddt=startdt+Day(30))
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        try
            EnvConfig.setcoinspath!(folder)
            EnvConfig.setpairquote!("USDT")
            raw = TestOhlcv.testohlcv("SINE",startdt,startdt+Minute(30000))
            Ohlcv.write(raw)
            short = TestOhlcv.testohlcv("DOUBLESINE",startdt,startdt+Minute(26678))
            Ohlcv.write(short)
            EnvConfig.setpairquote!("USD")
            io = IOBuffer()
            progresspath = joinpath(folder,"progress.txt")
            open(progresspath,"w") do progressio
                redirect_stdout(progressio) do
                    @test_logs (:warn,r"skipping coin with insufficient") match_mode=:any begin
                        @test Refresh.regimereport!(["062","066","067"],["DREGIMETEST"],
                            ["SINE-USDT","DOUBLESINE-USDT"];io)===nothing
                    end
                end
            end
            progress = read(progresspath,String)
            @test occursin("ensuring stores F19, T24 of SINE (2/2)",progress)
            @test occursin("ensuring stores F20, T25 of SINE (2/2)",progress)
            @test !occursin("(1/1)",progress)
            report = String(take!(io))
            @test occursin("DOUBLESINE: excluded (insufficient window/context history)",report)
            @test occursin("SINE: train=0 eval=0 test=0 excluded",report)
            for ts in ("062","066","067")
                @test occursin("RegimeTraining eligibility: ts=$ts, daterange=DREGIMETEST",report)
            end
            @test occursin("Common RegimeTraining eligibility: ts=062,066,067",report)
            @test isfile(joinpath(Features.featuresfolder("SINE","USDT"),"F20.arrow"))
            @test isfile(joinpath(Features.featuresfolder("SINE","USDT"),"F19.arrow"))
            @test isfile(joinpath(Features.featuresfolder("SINE","USDT"),"F21.arrow"))
            @test !isdir(joinpath(folder,"DOUBLESINE-USDT","features"))
            @test !isdir(joinpath(folder,"classifiers"))
            clcfg = TradingStrategy.classifierconfig(TradingStrategy.tsconfig("062").classifier)
            @test_throws AssertionError TradingStrategy.ensurefeaturetargetstores!(clcfg,["SINE"];
                progressoffset=-1,progresstotal=2)
            @test_throws AssertionError TradingStrategy.ensurefeaturetargetstores!(clcfg,["SINE"];
                progressoffset=2,progresstotal=2)
        finally
            EnvConfig.setcoinspath!(relpath(oldcoinspath,dirname(EnvConfig.defaultlogfilespath)))
            EnvConfig.setpairquote!(oldquote)
            delete!(TradingStrategy.DATERANGE_CONFIGS,"DREGIMETEST")
        end
    end
end

@testset "Streaming refresh counts equal pooled RegimeTraining memberships" begin
    EnvConfig.init(EnvConfig.test)
    oldcoinspath = EnvConfig.activecoinspath
    oldquote = EnvConfig.pairquote
    startdt = DateTime(2025,1,1)
    histories = Dict("0G"=>TestOhlcv.sinedata(2400,60000,0,1;firstutc=startdt),
        "ZTX"=>TestOhlcv.sinedata(2400,60000,0,10.5;firstutc=startdt))
    dr = (startdt=startdt,enddt=last(histories["0G"].opentime))
    liquidity = merge(Ohlcv.ld,(checkperiod=60,minliquidminutes=120,startdistance=120))
    regime = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,
        stepminutes=30,mincoins=2,weightcap=0.5,threshold=0.0025,strength=0.0,
        confirmation=2,liquidity=liquidity)
    folds = TradingStrategy.FoldConfig(anchor=startdt,foldminutes=60000,
        blockminutes=1200,gapminutes=480,minminutes=120)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        try
            EnvConfig.setcoinspath!(folder)
            EnvConfig.setpairquote!("USDT")
            for (coin,history) in histories
                ohlcv = Ohlcv.defaultohlcv(coin)
                Ohlcv.setdataframe!(ohlcv,history)
                Ohlcv.write(ohlcv)
            end
            clcfg = TradingStrategy.classifierconfig(TradingStrategy.tsconfig("066").classifier)
            featurefactory = TradingStrategy.featurefactory(clcfg.features)
            targetfactory = () -> TradingStrategy.targetconfig(clcfg.targets)
            warmup = TradingStrategy.regimewarmup(featurefactory,targetfactory,regime)
            selected = Dict(coin=>TradingStrategy.regimehistory(history,coin,dr.startdt,dr.enddt,warmup)
                for (coin,history) in histories)
            mktempdir(folder) do pooledfolder
                stores = TradingStrategy.experimentstores!(selected,featurefactory,targetfactory,pooledfolder)
                assignments = TradingStrategy.regimeassignments(selected,stores,regime,folds,
                    dr.startdt,dr.enddt,Targets.lookaheadminutes(targetfactory()))
                expected = Dict(coin=>(train=count(==("train"),sets.set),
                    eval=count(==("eval"),sets.set),test=count(==("test"),sets.set))
                    for (coin,sets) in assignments.sets)
                times = collect(startdt:Minute(regime.stepminutes):dr.enddt)
                mktempdir(folder) do timestampfolder
                    coins = sort!(collect(keys(histories)))
                    persisted = Dict{String,Vector{DateTime}}()
                    observationranges = UnitRange{Int}[]
                    observationrows = 0
                    for (coinix,coin) in enumerate(coins)
                        observation = Refresh.preparecoin!(coin,dr,warmup,clcfg,regime,
                            times,timestampfolder;coinix,ncoins=length(coins))
                        @test !isnothing(observation)
                        observationref = WeakRef(observation)
                        EnvConfig.appenddf(observation,"observations";folderpath=timestampfolder)
                        push!(observationranges,(observationrows+1):(observationrows+nrow(observation)))
                        observationrows += nrow(observation)
                        observation = nothing
                        persisted[coin] = copy(Refresh.readregimetimes(coin,timestampfolder))
                        @test !isempty(persisted[coin])
                        @test issorted(persisted[coin]) && allunique(persisted[coin])
                        GC.gc()
                        @test isnothing(observationref.value)
                        for preparedcoin in coins[1:coinix]
                            @test Refresh.readregimetimes(preparedcoin,timestampfolder) ==
                                persisted[preparedcoin]
                        end
                    end
                    mapped = EnvConfig.readtable("observations";folderpath=timestampfolder,
                        format=:arrow,copycols=false)
                    @test nrow(mapped)==observationrows
                    @test !(mapped.slope isa Vector)
                    market = TradingStrategy.marketcalendar(
                        [view(mapped,rows,:) for rows in observationranges],regime;
                        startdt=dr.startdt,enddt=dr.enddt)
                    @test isequal(market,assignments.market)
                    loader = coin -> Refresh.readregimetimes(coin,timestampfolder)
                    calendar = TradingStrategy.foldcalendar(market,loader,coins,
                        folds;horizonminutes=Targets.lookaheadminutes(targetfactory()),enddt=dr.enddt)
                    @test Dict(coin=>TradingStrategy.calendarcounts(loader(coin),calendar)
                        for coin in coins)==expected
                    EnvConfig.writetable(DataFrame(opentime=DateTime[]),"EMPTY";
                        folderpath=timestampfolder,format=:arrow)
                    emptytimes = Refresh.readregimetimes("EMPTY",timestampfolder)
                    @test isempty(emptytimes)
                    @test eltype(emptytimes)==DateTime
                    @test TradingStrategy.calendarcounts(emptytimes,calendar)==
                        (train=0,eval=0,test=0)
                    @test_throws AssertionError Refresh.readregimetimes("MISSING",timestampfolder)
                end
                streamed = Refresh.regimecounts("066",dr,sort!(collect(keys(histories)));regime,folds)
                @test streamed==expected
                @test all(counts -> all(>(0),values(counts)),values(streamed))
                @test all(counts -> counts isa NamedTuple{(:train,:eval,:test)},values(streamed))
                opts = Dict("ts"=>"066","mode"=>"market",
                    "coins"=>join(stores.coins,","),"startdt"=>string(dr.startdt),
                    "enddt"=>string(dr.enddt))
                input = RegimeTraining.inputs(opts)
                @test isempty(input.histories)
                @test isempty(input.coverage)
                @test input.coins==stores.coins
                EnvConfig.setcoinspath!(folder)
                input = merge(input,(regime=regime,folds=folds,featurefactory=featurefactory,
                    targetfactory=targetfactory))
                mktempdir(folder) do trainingfolder
                    prepared = RegimeTraining.prepare!(input,trainingfolder)
                    @test isempty(prepared.input.histories)
                    @test isempty(prepared.stores.featurestores)
                    @test isempty(prepared.stores.targetstores)
                    @test isempty(prepared.eligible)
                    @test isempty(prepared.sets)
                    @test isequal(prepared.market,assignments.market)
                    @test isequal(prepared.calendar,assignments.calendar)
                    @test !isdir(joinpath(trainingfolder,"stores"))
                    full = [RegimeTraining.inputfullstores(input,coin) for coin in stores.coins]
                    shared = (coins=stores.coins,featurestores=[s.features for s in full],
                        targetstores=[s.targets for s in full])
                    labels = Targets.uniquelabels(targetfactory())
                    cols = Features.requestedcolumns(featurefactory())
                    for coin in stores.coins
                        windowfeatures,windowtargets = RegimeTraining.inputsharedstores(input,coin,selected[coin])
                        windoweligible = TradingStrategy.experimenttimes(Dict(coin=>selected[coin]),
                            (coins=[coin],featurestores=[windowfeatures],targetstores=[windowtargets]),regime)
                        @test RegimeTraining.readpreparedtable(joinpath(trainingfolder,"eligible"),coin).opentime ==
                            windoweligible[coin]
                    end
                    sources = RegimeTraining.trainingsources(prepared,prepared.sets,
                        joinpath(trainingfolder,"balanced");streaming=true)
                    @test sources.compiled
                    @test sources.evalweights==Classify.fixedevalweights(
                        RegimeTraining.loadevalindex(sources).regime,
                        RegimeTraining.loadevalindex(sources).foldid)
                    for (settype,source) in (("train",sources.train),("eval",sources.eval),("test",sources.test))
                        expectedindex = TradingStrategy.experimentindex(shared,assignments.sets,labels;settype)
                        actualindex = RegimeTraining.readpreparedtable(sources.folder,settype*"-index")
                        @test isequal(actualindex,expectedindex)
                        oracle = Classify.CompactMmapSource(shared.featurestores,cols,expectedindex;
                            folderpath=joinpath(trainingfolder,"oracle",settype),nlabels=length(labels),
                            classbalancing=settype=="train" && input.classbalancing)
                        @test source.X==oracle.X
                        @test source.labelix==oracle.labelix
                        @test source.classweight==oracle.classweight
                    end
                    group = first(groupby(RegimeTraining.loadtestindex(sources),:coin))
                    coinix = first(group.coinix)
                    coin = prepared.stores.coins[coinix]
                    selection = RegimeTraining.gainsources(prepared)
                    testrows = RegimeTraining.gaincoin(prepared,selection,coin,coinix,group,[240])
                    @test nrow(testrows)==nrow(group)
                    @test Matrix(testrows[:,cols])==permutedims(sources.test.X[:,findall(==(coinix),
                        RegimeTraining.loadtestindex(sources).coinix)])
                    regression = Features.readregression(coin,"USDT",240)
                    positions = searchsortedfirst.(Ref(regression.opentime),group.opentime)
                    @test testrows.regry240==regression.regry[positions]
                    @test testrows.grad240==regression.grad[positions]
                    shifted = DataFrame(group)
                    shifted.row .+= 1
                    @test_throws AssertionError RegimeTraining.gaincoin(prepared,selection,
                        coin,coinix,shifted,Int[])
                    @test_throws AssertionError RegimeTraining.gaincoin(prepared,selection,
                        coin,coinix,group,[9999])
                    misaligned = DataFrame(regression;copycols=true)
                    misaligned.opentime .+= Second(30)
                    EnvConfig.writetable(misaligned,"regr9998";
                        folderpath=Features.regressionfolder(coin,"USDT"),format=:arrow)
                    @test_throws AssertionError RegimeTraining.gaincoin(prepared,selection,
                        coin,coinix,group,[9998])
                    counts = RegimeTraining.readpreparedtable(trainingfolder,"sample-coverage")
                    expectedcounts = combine(groupby(vcat(values(assignments.sets)...),
                        [:set,:regime,:foldid]),nrow=>:samples)
                    @test isequal(sort(counts,[:set,:regime,:foldid]),
                        sort(expectedcounts,[:set,:regime,:foldid]))
                    loaded = String[]
                    references = WeakRef[]
                    loader = coin -> begin
                        GC.gc()
                        @test all(ref -> isnothing(ref.value),references)
                        push!(loaded,coin)
                        tables = RegimeTraining.inputfullstores(input,coin)
                        push!(references,WeakRef(tables.features),WeakRef(tables.targets))
                        tables
                    end
                    RegimeTraining.compileblocks!(prepared,joinpath(trainingfolder,"direct"),
                        labels,cols,"direct-test";folds=Int[],regimes=String[],storeloader=loader)
                    @test loaded==stores.coins
                    bytes = read(joinpath(sources.folder,"train",Classify.TRAINFEATURES_FILE))
                    @test_throws AssertionError RegimeTraining.compileblocks!(prepared,sources.folder,
                        labels,cols,"changed";folds=Int[],regimes=String[])
                    @test read(joinpath(sources.folder,"train",Classify.TRAINFEATURES_FILE))==bytes
                    weighted = RegimeTraining.trainingsources(prepared,prepared.sets,
                        joinpath(trainingfolder,"weighted");streaming=true,trainweights="balanced")
                    @test weighted.train.weights==TradingStrategy.balancedcellweights(
                        RegimeTraining.readpreparedtable(weighted.folder,"train-index"))
                    @test weighted.train.source.X==sources.train.X
                    for coin in stores.coins
                        prepared.input.histories[coin] = RegimeTraining.loadinputhistory(input,coin)
                    end
                    RegimeTraining.reloadtrainingstores!(prepared)
                    for (coinix,coin) in enumerate(stores.coins)
                        expectedfeatures,expectedtargets = RegimeTraining.inputsharedstores(
                            input,coin,prepared.input.histories[coin])
                        @test isequal(prepared.stores.featurestores[coinix],expectedfeatures)
                        @test isequal(prepared.stores.targetstores[coinix],expectedtargets)
                    end
                    chronological = RegimeTraining.trainingsources(prepared,assignments.sets,
                        joinpath(trainingfolder,"chronological");streaming=true,keephistories=true)
                    @test !chronological.compiled
                    @test chronological.train.X==sources.train.X
                    @test chronological.eval.X==sources.eval.X
                end
            end
        finally
            EnvConfig.setcoinspath!(relpath(oldcoinspath,dirname(EnvConfig.defaultlogfilespath)))
            EnvConfig.setpairquote!(oldquote)
        end
    end
end

@testset "Refresh writes complete shared stores" begin
    EnvConfig.init(EnvConfig.test)
    oldcoinspath = EnvConfig.activecoinspath
    oldquote = EnvConfig.pairquote
    tempfolder = mktempdir()
    try
        EnvConfig.setcoinspath!(tempfolder)
        EnvConfig.setpairquote!("USDT")
        startdt = DateTime("2025-03-01T00:00:00")
        ohlcv = TestOhlcv.testohlcv("SINE",startdt,startdt+Minute(1200))
        Ohlcv.write(ohlcv)
        failures = Pair{String,String}[]
        Refresh.refreshpair!(failures, "SINE-USDT", ["F08", "F19"], ["T24"], ["P02_D00", "P02_D01"])
        @test first.(failures) == ["SINE-USDT F08"]
        @test occursin("insufficient OHLCV history", last(only(failures)))
        Refresh.refreshpair!(failures, "MISSING-USDT", ["F19"], ["T24"], ["P02_D00"])
        @test first.(failures) == ["SINE-USDT F08", "MISSING-USDT OHLCV"]
        @test Refresh.reportfailures(failures) === nothing
        features = Features.readfeatures("SINE","USDT","F19";
            confighash=TradingStrategy.featureshash("F19"))
        targets = Targets.readtargets("SINE","USDT","T24";
            confighash=TradingStrategy.targetshash("T24"))
        @test nrow(features) == nrow(Ohlcv.dataframe(ohlcv))-Features.requiredminutes(TradingStrategy.featureconfig("F19"))+1
        @test features.opentime[end] == Ohlcv.dataframe(ohlcv).opentime[end]
        @test nrow(targets) == nrow(Ohlcv.dataframe(ohlcv))
        @test targets.opentime[end] == Ohlcv.dataframe(ohlcv).opentime[end]
        for id in ("P02_D00", "P02_D01")
            _, daterangeid = split(id, "_")
            hash = TradingStrategy.setshash("P02", daterangeid)
            path = joinpath(Classify.setsfolder("SINE", "USDT"), id * ".arrow")
            @test isfile(path)
            @test Classify.readsets("SINE", "USDT", id; confighash=hash) isa DataFrame
            EnvConfig.savedf(DataFrame(opentime=[startdt], set=["stale"], rangeid=[0]), id;
                folderpath=dirname(path), format=:arrow,
                metadata=Dict("confighash"=>"old-date-range", "coverage"=>"old"))
            @test_throws AssertionError Classify.readsets("SINE", "USDT", id; confighash=hash)
        end
        featurepath = joinpath(Features.featuresfolder("SINE", "USDT"), "F19.arrow")
        targetpath = joinpath(Targets.targetsfolder("SINE", "USDT"), "T24.arrow")
        featurebytes, targetbytes = read(featurepath), read(targetpath)
        partitionfailures = Pair{String,String}[]
        zeroassignments = Pair{String,String}[]
        Refresh.refreshpair!(partitionfailures, "SINE-USDT", String[], String[],
            ["P99_D00", "P02_D00", "P02_D01"]; zeroassignments)
        @test first.(partitionfailures) == ["SINE-USDT P99_D00"]
        @test zeroassignments == ["SINE-USDT"=>"P02_D00", "SINE-USDT"=>"P02_D01"]
        @test_logs (:warn, r"rebuilt sets store has zero assignments") Refresh.refreshpartitions!(ohlcv, "SINE-USDT", "P02_D00")
        for id in ("P02_D00", "P02_D01")
            _, daterangeid = split(id, "_")
            store = Classify.readsets("SINE", "USDT", id;
                confighash=TradingStrategy.setshash("P02", daterangeid))
            @test !("stale" in store.set)
        end
        @test read(featurepath) == featurebytes
        @test read(targetpath) == targetbytes
    finally
        EnvConfig.setcoinspath!(relpath(oldcoinspath,dirname(EnvConfig.defaultlogfilespath)))
        EnvConfig.setpairquote!(oldquote)
        rm(tempfolder;force=true,recursive=true)
    end
end

@testset "Refresh rebuilds nonempty partition assignments" begin
    EnvConfig.init(EnvConfig.test)
    oldpath = EnvConfig.coinspath()
    oldquote = EnvConfig.pairquote
    startdt = DateTime("2025-02-17T13:30:00")
    enddt = startdt + Day(2) - Minute(1)
    TradingStrategy.PARTITION_CONFIGS["PREFRESH"] = () ->
        (samplesets=["train", "test", "train", "eval"], partitionsize=60,
            gapsize=10, minpartitionsize=30, maxpartitionsize=120)
    TradingStrategy.DATERANGE_CONFIGS["DREFRESH"] = (;startdt, enddt)
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        EnvConfig.setcoinspath!(folder)
        EnvConfig.setpairquote!("USDT")
        try
            ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
            Ohlcv.write(ohlcv)
            id = "PREFRESH_DREFRESH"
            hash = TradingStrategy.setshash("PREFRESH", "DREFRESH")
            failures = Pair{String,String}[]
            zeroassignments = Pair{String,String}[]
            @test_logs Refresh.refreshpair!(failures, "SINE-USDT", String[], String[], [id]; zeroassignments)
            @test isempty(failures)
            @test isempty(zeroassignments)
            original = Classify.readsets("SINE", "USDT", id; confighash=hash)
            @test nrow(original) > 0
            @test Set(original.set) == Set(["train", "test", "eval"])

            TradingStrategy.DATERANGE_CONFIGS["DREFRESH"] = (startdt=startdt, enddt=enddt-Hour(2))
            newhash = TradingStrategy.setshash("PREFRESH", "DREFRESH")
            @test hash != newhash
            @test_throws AssertionError Classify.readsets("SINE", "USDT", id; confighash=newhash)
            partitionmemberships = Dict{Tuple{String,String},Set{String}}()
            Refresh.refreshpair!(failures, "SINE-USDT", String[], String[], [id];partitionmemberships)
            @test isempty(failures)
            rebuilt = Classify.readsets("SINE", "USDT", id; confighash=newhash)
            @test 0 < nrow(rebuilt) < nrow(original)
            @test all(dt -> dt <= enddt-Hour(2), rebuilt.opentime)
            io = IOBuffer()
            Refresh.reportpartitioncoins([id],partitionmemberships;io)
            @test occursin("traincoins() = [\"SINE\"]",String(take!(io)))
            @test !isdir(joinpath(folder, "SINE-USDT", "features"))
            @test !isdir(joinpath(folder, "SINE-USDT", "targets"))
        finally
            EnvConfig.setcoinspath!(oldpath)
            EnvConfig.setpairquote!(oldquote)
            delete!(TradingStrategy.PARTITION_CONFIGS, "PREFRESH")
            delete!(TradingStrategy.DATERANGE_CONFIGS, "DREFRESH")
        end
    end
end

@testset "Refresh zero-assignment summary" begin
    io = IOBuffer()
    @test Refresh.reportzeroassignments(Pair{String,String}[]; io) === nothing
    @test isempty(take!(io))
    Refresh.reportzeroassignments(["SINE-USDT"=>"P02_D01", "ADA-USDT"=>"P02_D00",
        "SINE-USDT"=>"P02_D00"]; io)
    @test String(take!(io)) == """

Zero assignments in 3 rebuilt sets stores across 2 coin pairs:
  ADA-USDT: P02_D00
  SINE-USDT: P02_D00, P02_D01
Empty stores were saved; these pairs contribute no samples through the listed stores.
"""
end

@testset "script help flag detection" begin
    @test BoundsEstimator._wants_help(["help"])
    @test BoundsEstimator._wants_help(["--help"])
    @test BoundsEstimator._wants_help(["help=true"])
    @test !BoundsEstimator._wants_help(["help=false"])

    @test TrendDetector._wants_help(["-h"])
    @test !TrendDetector._wants_help(["train"])

    @test isnothing(TrendDetector._assertknownargs(["train", "ts=050", "stage=train", "coins=SINE", "runid=x"]))
    @test_throws AssertionError TrendDetector._assertknownargs(["config=050"])
    @test_throws AssertionError TrendDetector._assertknownargs(["special"])
    @test_throws AssertionError TrendDetector._assertknownargs(["TRAIN"])

end

@testset "script config selection by configname" begin
    startdt = DateTime("2025-01-17T20:56:00")
    enddt = DateTime("2025-08-10T15:00:00")

    EnvConfig.init(test)
    rc = TrendDetector.buildrunconfig(["ts=029", "stage=train", "coins=SINE"])
    @test rc.tscfg.configname == "029"
    @test rc.clcfg.classbalancing == false
    @test rc.coinset == "CS00"
    @test rc.coins == ["SINE"]
    @test TrendDetector.classifiername(rc) == "F01-T01-P02-M002-B0-CS00-D00"
    @test TrendDetector.runsstage(rc, "stores") && !TrendDetector.runsstage(rc, "predict")
    @test_throws AssertionError TrendDetector.buildrunconfig(["stage=unknown"])
    @test_throws AssertionError TrendDetector.buildrunconfig(["startdt=2025-07-01T01:00:00"])

    boundscfg = BoundsEstimator.buildcfg(["config=001"], ["SINE"], startdt, enddt)
    @test boundscfg.configname == "001"

end

@testset "class weights use inverse class frequency" begin
    weights = Classify.classweights([2, 1, 0])
    @test weights[1] < weights[2]
    @test weights[3] == 0f0
    @test isapprox((2 * weights[1] + weights[2]) / 3, 1.0; atol=1e-6)
end

@testset "weighted classifier adaptation accepts sample weights" begin
    features = Float32[0 0 0 1; 0 1 2 3]
    targets = [longopen, longopen, longopen, allclose]
    nn = Classify.model006(size(features, 1), [longopen, allclose], "ut_weighted_trend")
    weights = Classify.classweights([3, 1])[[1, 1, 1, 2]]
    mktempdir() do tmpdir
        adapted = Classify.adaptnn!(nn, Classify.MatrixSampleSource(nn, features, targets, weights); maxepochs=3, folderpath=tmpdir)
        @test length(adapted.losses) == 3
    end
end

end
