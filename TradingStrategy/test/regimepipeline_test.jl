using Test, Dates, DataFrames
import TradingStrategy, TestOhlcv, Ohlcv, Features, Targets, EnvConfig

"""Construct small Features006 fixtures for store and source integration."""
function regimepipelinefeatures()
    f6 = Features.Features006()
    Features.addnormgrad!(f6;window=15)
    Features.addnormgrad!(f6;window=60)
    return f6
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
    for cutoff in (start-Minute(1),start+Minute(900),last(history.opentime)+Minute(1))
        @test TradingStrategy.experimenttimes(histories,stores,config;cutoff)==
            dictionaryexperimenttimes(histories,stores,config;cutoff)
    end
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

@testset "Immutable synthetic experiment data" begin
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
        @test_throws AssertionError TradingStrategy.experimentmanifest!(folder,changed,"fixture-v1")
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
    end
end
