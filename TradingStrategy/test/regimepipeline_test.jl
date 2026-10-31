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
