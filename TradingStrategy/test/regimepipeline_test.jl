using Test, Dates, DataFrames
import TradingStrategy, TestOhlcv, Ohlcv, Features, Targets, EnvConfig

"""Construct small Features006 fixtures for store and source integration."""
function regimepipelinefeatures()
    f6 = Features.Features006()
    Features.addnormgrad!(f6;window=15)
    Features.addnormgrad!(f6;window=60)
    return f6
end

@testset "Block selection equals timestamp membership joins" begin
    start = DateTime(2025,1,1)
    timestamps = collect(start:Minute(1):start+Minute(10))
    labels = Targets.uniquelabels(Targets.Trend05())
    features = DataFrame(opentime=timestamps,value=ones(Float32,11))
    targets = DataFrame(opentime=timestamps,label=fill(Targets.tradelabelcode(first(labels)),11))
    eligible = timestamps[[1,3,5,8,11]]
    calendar = DataFrame(blockid=[1,2,3],foldid=[1,1,2],regime=["up","flat","down"],
        start=timestamps[[1,5,11]],stop=timestamps[[3,8,11]],
        set=["train","eval","test"])
    stores = (coins=["SINE"],featurestores=[features],targetstores=[targets])
    sets = Dict("SINE"=>TradingStrategy.calendarsets(eligible,calendar))
    for (folds,regimes) in ((Int[],String[]),([1],String[]),(Int[],["up","down"]))
        direct = TradingStrategy.blockexperimentindices(features,targets,eligible,
            calendar,labels,"SINE",1;folds,regimes)
        for (settype,index) in zip(("train","eval","test"),values(direct))
            expected = TradingStrategy._experimentindex()
            TradingStrategy._appendexperimentindex!(expected,features,targets,sets["SINE"],
                labels,"SINE",1;settype,folds,regimes)
            @test isequal(index,expected)
        end
    end
    empty = TradingStrategy.blockexperimentindices(features,targets,DateTime[],
        calendar,labels,"SINE",1)
    @test all(isempty,values(empty))
    @test_throws AssertionError TradingStrategy.blockexperimentindices(features[2:end,:],
        targets,eligible,calendar,labels,"SINE",1)
    @test_throws AssertionError TradingStrategy.blockexperimentindices(features,targets,
        reverse(eligible),calendar,labels,"SINE",1)
    overlapping = copy(calendar)
    overlapping.stop[1] = overlapping.start[2]
    @test_throws AssertionError TradingStrategy.blockexperimentindices(features,targets,
        eligible,overlapping,labels,"SINE",1)
    invalid = copy(targets)
    invalid.label[1] = Targets.tradelabelcode(Targets.ignore)
    @test_throws AssertionError TradingStrategy.blockexperimentindices(features,invalid,
        eligible,calendar,labels,"SINE",1)
end

"""Construct bounded Trend05 targets for store integration."""
regimepipelinetarget() = Targets.Trend05(leadregr=5,supportregr=15,
    trendregr=60,triggerdist=0.001f0,targetgain=0.005f0,targetminutes=60)

"""Construct a feature configuration whose history exceeds the small fixture's detector warmup."""
function regimepipelinelongfeatures()
    features = Features.Features006()
    Features.addnormgrad!(features;window=600)
    return features
end

"""Construct a target configuration whose regression history exceeds feature warmup."""
regimepipelinelongtarget() = Targets.Trend05(leadregr=5,supportregr=15,
    trendregr=800,triggerdist=0.001f0,targetgain=0.005f0,targetminutes=120)

"""Retain the original dictionary join as an exact eligibility regression oracle."""
function dictionaryexperimenttimes(histories::AbstractDict, stores::NamedTuple,
        config::TradingStrategy.RegimeConfig; cutoff::Union{Nothing,DateTime}=nothing)
    result = Dict{String,Vector{DateTime}}()
    for (coinix,coin) in enumerate(stores.coins)
        history = histories[coin]
        lastix = isnothing(cutoff) ? nrow(history) : searchsortedlast(history.opentime,cutoff)
        liquid = falses(nrow(history))
        if lastix > 0
            for range in Ohlcv.liquiditycheck(view(history,1:lastix,:);config.liquidity...)
                liquid[range] .= true
            end
        end
        fdf,tdf = stores.featurestores[coinix],stores.targetstores[coinix]
        features = Dict(time=>ix for (ix,time) in enumerate(fdf.opentime))
        targets = Dict(time=>ix for (ix,time) in enumerate(tdf.opentime))
        times = DateTime[]
        for (ix,time) in enumerate(history.opentime)
            liquid[ix] && haskey(features,time) && haskey(targets,time) || continue
            tdf.label[targets[time]] != Targets.tradelabelcode(Targets.ignore) || continue
            @assert all(isfinite,fdf[features[time],Not(:opentime)]) "invalid feature row for $coin at $time"
            push!(times,time)
        end
        result[coin] = times
    end
    return result
end

"""Keep the previous dictionary-based index join as an independent regression oracle."""
function dictionaryexperimentindex(stores::NamedTuple, sets::AbstractDict,
        labels::AbstractVector; settype::String, folds::Vector{Int}=Int[],
        regimes::Vector{String}=String[])::DataFrame
    index = DataFrame(coinix=Int[],row=Int[],labelix=Int[],coin=String[],
        opentime=DateTime[],set=String[],regime=String[],foldid=Int[],blockid=Int[])
    codes = Dict(Targets.tradelabelcode(label)=>ix for (ix,label) in enumerate(labels))
    for (coinix,coin) in enumerate(stores.coins)
        frows = Dict(time=>ix for (ix,time) in enumerate(stores.featurestores[coinix].opentime))
        trows = Dict(time=>ix for (ix,time) in enumerate(stores.targetstores[coinix].opentime))
        for row in eachrow(sets[coin])
            row.set==settype && row.regime!="unknown" || continue
            isempty(folds) || row.foldid in folds || continue
            isempty(regimes) || row.regime in regimes || continue
            code = stores.targetstores[coinix].label[trows[row.opentime]]
            push!(index,(coinix,frows[row.opentime],codes[code],coin,row.opentime,
                settype,row.regime,row.foldid,row.blockid))
        end
    end
    return index
end

@testset "Sorted source coordinates preserve gaps, filtering and fail-fast joins" begin
    start = DateTime(2025)
    times = [start+Minute(ix) for ix in 0:8]
    features = DataFrame(opentime=times[[1,3,4,6,8]],value=ones(Float32,5))
    labels = Targets.uniquelabels(regimepipelinetarget())
    targets = DataFrame(opentime=times[[2,3,4,6,8,9]],
        label=Targets.tradelabelcode.(labels[[1,2,3,4,5,1]]))
    sets = Dict("SINE"=>DataFrame(opentime=times[[2,3,4,6,8]],
        set=["eval","train","train","train","train"],regime=["unknown","up","flat","down","up"],
        foldid=[1,1,1,2,2],blockid=1:5))
    stores = (coins=["SINE"],featurestores=[features],targetstores=[targets])
    for (folds,regimes) in ((Int[],String[]),([1],String[]),(Int[],["up"]),([2],["up"]))
        actual = TradingStrategy.experimentindex(stores,sets,labels;settype="train",folds,regimes)
        @test isequal(actual,dictionaryexperimentindex(stores,sets,labels;settype="train",folds,regimes))
    end
    missingfeatures = merge(stores,(featurestores=[features[1:end-1,:]],))
    @test_throws AssertionError TradingStrategy.experimentindex(missingfeatures,sets,labels;settype="train")
    missingtargets = merge(stores,(targetstores=[targets[1:2,:]],))
    @test_throws AssertionError TradingStrategy.experimentindex(missingtargets,sets,labels;settype="train")
    invalidtargets = copy(targets)
    invalidtargets.label[2] = Targets.tradelabelcode(Targets.ignore)
    @test_throws AssertionError TradingStrategy.experimentindex(
        merge(stores,(targetstores=[invalidtargets],)),sets,labels;settype="train")
    @test_throws AssertionError TradingStrategy.experimentindex(
        merge(stores,(featurestores=[reverse(features)],)),sets,labels;settype="train")
end

@testset "Sorted eligibility walk preserves exact timestamp joins" begin
    start = DateTime(2025,1,1)
    history = TestOhlcv.sinedata(120,2400,0,1;firstutc=start)
    Ohlcv.addpivot!(history)
    histories = Dict("SINE"=>history)
    liquidity = (minquotevol=5000f0,accumulate=15,checkperiod=60,
        startthreshold=0.05,stopthreshold=0.25,minliquidminutes=120,startdistance=120)
    config = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,
        stepminutes=30,mincoins=2,weightcap=0.5,liquidity=liquidity)
    featuretimes = vcat([start-Minute(2)],history.opentime[5:2390][[ix%7!=0 for ix in 5:2390]],
        [last(history.opentime)+Minute(2)])
    targettimes = history.opentime[[ix%11!=0 for ix in 1:nrow(history)]]
    features = DataFrame(opentime=featuretimes,first=ones(Float32,length(featuretimes)),
        second=ones(Float64,length(featuretimes)))
    targets = DataFrame(opentime=targettimes,
        label=fill(Targets.tradelabelcode(Targets.allclose),length(targettimes)))
    targets.label[1:13:end] .= Targets.tradelabelcode(Targets.ignore)
    stores = (coins=["SINE"],featurestores=[features],targetstores=[targets])
    expected = dictionaryexperimenttimes(histories,stores,config)
    @test !isempty(expected["SINE"])
    @test TradingStrategy.experimenttimes(histories,stores,config)==expected
    liquid = TradingStrategy.regimeliquidity(history,config)
    @test TradingStrategy.experimenttimes(histories,stores,config;liquidity=Dict("SINE"=>liquid))==expected
    @test isempty(TradingStrategy.experimenttimes(histories,stores,config;
        liquidity=Dict("SINE"=>falses(nrow(history))))["SINE"])
    for cutoff in (start-Minute(1),start+Minute(900),last(history.opentime)+Minute(1))
        @test TradingStrategy.experimenttimes(histories,stores,config;cutoff)==
            dictionaryexperimenttimes(histories,stores,config;cutoff)
        mask = TradingStrategy.regimeliquidity(history,config;cutoff)
        @test TradingStrategy.experimenttimes(histories,stores,config;cutoff,
            liquidity=Dict("SINE"=>mask))==dictionaryexperimenttimes(histories,stores,config;cutoff)
    end
    @test_throws AssertionError TradingStrategy.experimenttimes(histories,stores,config;
        cutoff=start,liquidity=Dict("SINE"=>liquid))
    for (fdf,tdf) in ((features[1:0,:],targets),(features,targets[1:0,:]),
            (features[1:3,:],targets),(features,targets[end-2:end,:]))
        partial = (coins=["SINE"],featurestores=[fdf],targetstores=[tdf])
        @test TradingStrategy.experimenttimes(histories,partial,config)==
            dictionaryexperimenttimes(histories,partial,config)
    end
    @test TradingStrategy.experimenttimes(Dict("SINE"=>history[1:0,:]),stores,config)["SINE"]==DateTime[]
    invalid = deepcopy(features)
    validtime = first(expected["SINE"])
    invalid.first[findfirst(==(validtime),invalid.opentime)] = NaN32
    invalidstores = merge(stores,(featurestores=[invalid],))
    @test_throws AssertionError TradingStrategy.experimenttimes(histories,invalidstores,config)
    cutoff = validtime-Minute(1)
    @test TradingStrategy.experimenttimes(histories,invalidstores,config;cutoff)==
        dictionaryexperimenttimes(histories,invalidstores,config;cutoff)
    ignored = deepcopy(targets)
    ignored.label[findfirst(==(validtime),ignored.opentime)] = Targets.tradelabelcode(Targets.ignore)
    ignoredstores = merge(invalidstores,(targetstores=[ignored],))
    @test TradingStrategy.experimenttimes(histories,ignoredstores,config)==
        dictionaryexperimenttimes(histories,ignoredstores,config)
    for timestamps in (DateTime[],[start],[start,start+Minute(2)])
        @test TradingStrategy._strictregimetimes(timestamps)
    end
    @test !TradingStrategy._strictregimetimes([start,start])
    @test !TradingStrategy._strictregimetimes([start+Minute(1),start])
    for table in (history,features,targets)
        original = table.opentime[2]
        table.opentime[2] = table.opentime[1]
        @test_throws AssertionError TradingStrategy.experimenttimes(histories,stores,config)
        table.opentime[2] = original
    end
end

@testset "Regime history eligibility boundaries" begin
    start = DateTime(2025,1,1)
    raw = TestOhlcv.sinedata(120,28801,0,1;firstutc=start)
    warmup = TradingStrategy.regimewarmup(regimepipelinefeatures,regimepipelinetarget,
        TradingStrategy.RegimeConfig())
    @test warmup==28800
    liquidity = merge(Ohlcv.ld,(startdistance=120,minliquidminutes=120))
    config = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,liquidity=liquidity)
    featurewarmup = TradingStrategy.regimewarmup(regimepipelinelongfeatures,
        regimepipelinetarget,config)
    targetwarmup = TradingStrategy.regimewarmup(regimepipelinelongfeatures,
        regimepipelinelongtarget,config)
    @test featurewarmup==Features.requiredminutes(regimepipelinelongfeatures())
    @test targetwarmup==Features.requiredminutes(Targets.regressionfeatures(regimepipelinelongtarget()))
    @test targetwarmup>featurewarmup>config.longminutes
    for rows in (0,26679,28800)
        @test_logs (:warn,r"skipping coin with insufficient window/context history") begin
            @test isnothing(TradingStrategy.regimehistory(raw[1:rows,:],"CAMP",
                start,start+Day(30),warmup))
        end
    end
    history = TradingStrategy.regimehistory(raw,"SINE",start,start+Day(30),warmup)
    @test nrow(history)==28801
    @test :pivot in propertynames(history)
    @test history.opentime==raw.opentime
    @test raw.close==history.close
    @test_logs (:warn,r"skipping coin") begin
        @test isnothing(TradingStrategy.regimehistory(raw,"SINE",start,start+Minute(26678),warmup))
    end
end

@testset "Synthetic experiment coverage and configuration" begin
    EnvConfig.init(EnvConfig.test)
    start = DateTime(2025,1,1)
    histories = Dict("SINE"=>TestOhlcv.sinedata(1200,6000,0,1;firstutc=start),
        "DOUBLESINE"=>TestOhlcv.sinedata(1200,6000,0,10.5;firstutc=start))
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        identity = TradingStrategy.experimentmanifest!(folder,histories,"fixture-v1")
        @test identity == TradingStrategy.experimentmanifest!(folder,histories,"fixture-v1")
        @test_throws AssertionError TradingStrategy.experimentmanifest!(folder,histories,"fixture-v2")
        changed = deepcopy(histories)
        changed["SINE"].close[200] += 0.01f0
        @test identity == TradingStrategy.experimentmanifest!(folder,changed,"fixture-v1")
        shortened = copy(histories)
        shortened["SINE"] = histories["SINE"][1:end-1,:]
        @test_throws AssertionError TradingStrategy.experimentmanifest!(folder,shortened,"fixture-v1")
        shifted = deepcopy(histories)
        shifted["SINE"].opentime .+= Minute(1)
        @test_throws AssertionError TradingStrategy.experimentmanifest!(folder,shifted,"fixture-v1")
        coverage = EnvConfig.readtable("input-identities";folderpath=folder,format=:arrow)
        @test names(coverage)==["coin","rows","start","stop"]
        stores = TradingStrategy.experimentstores!(histories,regimepipelinefeatures,regimepipelinetarget,folder)
        @test stores.coins == ["DOUBLESINE","SINE"]
        @test all(df->nrow(df)>0,stores.featurestores)
        @test all(df->nrow(df)==6000,stores.targetstores)
        reused = TradingStrategy.experimentstores!(histories,regimepipelinefeatures,regimepipelinetarget,folder)
        @test isequal(stores.featurestores,reused.featurestores)
        liquidity = (minquotevol=5000f0,accumulate=15,checkperiod=60,
            startthreshold=0.05,stopthreshold=0.25,minliquidminutes=120,startdistance=120)
        config = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,
            stepminutes=30,mincoins=2,weightcap=0.5,threshold=0.0025,strength=0.0,
            confirmation=2,liquidity=liquidity)
        times = TradingStrategy.experimenttimes(histories,stores,config)
        @test all(times["SINE"] .> start+Minute(120))
        market = TradingStrategy.marketcalendar(histories,config;
            startdt=start,enddt=last(histories["SINE"].opentime))
        folds = TradingStrategy.FoldConfig(anchor=start,foldminutes=3000,
            blockminutes=240,gapminutes=60,minminutes=60)
        calendar = TradingStrategy.foldcalendar(market,times,folds;horizonminutes=60)
        sets = Dict(coin=>TradingStrategy.calendarsets(times[coin],calendar) for coin in stores.coins)
        preparation = TradingStrategy.regimeassignments(histories,stores,config,folds,
            start,last(histories["SINE"].opentime),60)
        @test isequal(preparation.eligible,times)
        @test isequal(preparation.market,market)
        @test isequal(preparation.sets,sets)
        for coin in stores.coins
            features,targets = TradingStrategy.sharedstorewindow(histories[coin],
                stores.featurestores[findfirst(==(coin),stores.coins)],
                stores.targetstores[findfirst(==(coin),stores.coins)],
                regimepipelinefeatures,regimepipelinetarget,coin)
            @test nrow(features)==6000-Features.requiredminutes(regimepipelinefeatures())+1
            @test nrow(targets)==6000
            @test all(==(Targets.tradelabelcode(Targets.ignore)),targets.label[end-59:end])
        end

        labels = Targets.uniquelabels(regimepipelinetarget())
        index = TradingStrategy.experimentindex(stores,sets,labels;settype="train")
        @test isequal(index,dictionaryexperimentindex(stores,sets,labels;settype="train"))
        mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do indexfolder
            references = WeakRef[]
            loader = coin -> begin
                GC.gc()
                @test all(ref -> isnothing(ref.value),references)
                coinix = findfirst(==(coin),stores.coins)
                loaded = (features=copy(stores.featurestores[coinix]),
                    targets=copy(stores.targetstores[coinix]),sets=copy(sets[coin]))
                append!(references,WeakRef.((loaded.features,loaded.targets,loaded.sets)))
                loaded
            end
            streamed = TradingStrategy.experimentindices(indexfolder,stores.coins,loader,labels)
            for (settype,actual) in zip(("train","eval","test"),values(streamed))
                @test isequal(actual,dictionaryexperimentindex(stores,sets,labels;settype))
                @test !(actual.row isa Vector)
            end
            GC.gc()
            @test all(ref -> isnothing(ref.value),references)
            filtered = TradingStrategy.experimentindices(indexfolder,stores.coins,loader,labels;
                folds=[first(index.foldid)],regimes=[first(index.regime)])
            for (settype,actual) in zip(("train","eval","test"),values(filtered))
                @test isequal(actual,dictionaryexperimentindex(stores,sets,labels;
                    settype,folds=[first(index.foldid)],regimes=[first(index.regime)]))
            end
            @test_throws AssertionError TradingStrategy.experimentindices(indexfolder,
                stores.coins,loader,labels;folds=[100])
        end
        original = sets["SINE"].opentime[2]
        sets["SINE"].opentime[2] = sets["SINE"].opentime[1]
        @test_throws AssertionError TradingStrategy.experimentindex(stores,sets,labels;settype="train")
        sets["SINE"].opentime[2] = original
        @test all(!=("unknown"),index.regime)
        @test all(==("train"),index.set)
        weights = TradingStrategy.balancedcellweights(index)
        @test sum(weights) ≈ 1
        for regime in unique(index.regime)
            @test sum(weights[index.regime .== regime]) ≈ 1/length(unique(index.regime))
        end
        @test_throws AssertionError TradingStrategy.experimentindex(stores,sets,labels;
            settype="train",folds=[100])
        cutoff = start+Minute(2000)
        causal = TradingStrategy.experimenttimes(histories,stores,config;cutoff)
        @test all(time->time<=cutoff,causal["SINE"])
        TradingStrategy.writecalendarcoverage!(folder,sets,times,market)
        @test isfile(joinpath(folder,"exclusions.arrow"))
        @test isfile(joinpath(folder,"sample-coverage.arrow"))
        mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do streamedfolder
            references = WeakRef[]
            loader = coin -> begin
                GC.gc()
                @test all(ref -> isnothing(ref.value),references)
                loaded = (sets=copy(sets[coin]),eligible=copy(times[coin]))
                append!(references,WeakRef.((loaded.sets,loaded.eligible)))
                loaded
            end
            TradingStrategy.writecalendarcoverage!(streamedfolder,stores.coins,loader,market)
            for stem in ("exclusions","sample-coverage")
                expected = EnvConfig.readtable(stem;folderpath=folder,format=:arrow,copycols=true)
                actual = EnvConfig.readtable(stem;folderpath=streamedfolder,format=:arrow,copycols=true)
                stem=="sample-coverage" && (sort!(expected,[:set,:regime,:foldid]);sort!(actual,[:set,:regime,:foldid]))
                @test isequal(actual,expected)
            end
            GC.gc()
            @test all(ref -> isnothing(ref.value),references)
        end
    end
end

@testset "Index coverage does not hash membership contents" begin
    EnvConfig.init(EnvConfig.test)
    start = DateTime(2025,1,1)
    index = DataFrame(opentime=start .+ Minute.([10,20,0,30]),labelix=[1,2,1,2])
    @test TradingStrategy.experimentcoverage("train",index)==
        (coin="train",rows=4,start=start,stop=start+Minute(30))
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        EnvConfig.writetable(index,"empty-arrow-tail";folderpath=folder,format=:arrow,
            preserveschema=true)
        EnvConfig.writetable(index[1:0,:],"empty-arrow-tail";folderpath=folder,
            format=:arrow,append=true,preserveschema=true)
        arrowindex = EnvConfig.readtable("empty-arrow-tail";folderpath=folder,format=:arrow)
        @test TradingStrategy.experimentcoverage("train",arrowindex)==
            (coin="train",rows=4,start=start,stop=start+Minute(30))
        identity = TradingStrategy.experimentmanifest!(folder,Dict("train"=>index),"selection")
        changed = copy(index)
        changed.labelix[2] = 1
        @test TradingStrategy.experimentmanifest!(folder,Dict("train"=>changed),"selection")==identity
        changed.opentime[4] += Minute(1)
        @test_throws AssertionError TradingStrategy.experimentmanifest!(folder,Dict("train"=>changed),"selection")
        @test_throws AssertionError TradingStrategy.experimentcoverage("empty",index[1:0,:])
    end
end
