using Test, Dates, DataFrames
import TradingStrategy, TestOhlcv, Ohlcv

@testset "Shared regimes and calendar contracts" begin
    liquidity = (minquotevol=5000f0, accumulate=15, checkperiod=60,
        startthreshold=0.05, stopthreshold=0.25, minliquidminutes=120, startdistance=120)
    config = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,
        stepminutes=30,mincoins=2,weightcap=0.5,threshold=0.0025,strength=0.0,
        confirmation=2,liquidity=liquidity)
    start = DateTime(2025,1,1)
    sine = TestOhlcv.sinedata(2400,24000,0,1; firstutc=start)
    double = TestOhlcv.sinedata(2400,24000,0,10.5; firstutc=start)
    histories = Dict("SINE"=>sine,"DOUBLESINE"=>double)
    market = TradingStrategy.marketcalendar(histories,config;
        startdt=start,enddt=last(sine.opentime))
    times = collect(market.opentime)
    observations = [TradingStrategy.regimeobservations(histories[coin],times,config)
        for coin in sort!(collect(keys(histories)))]
    @test isequal(market,TradingStrategy.marketcalendar(observations,config;
        startdt=start,enddt=last(sine.opentime)))
    @test "unknown" in market.regime
    @test "up" in market.regime
    @test "down" in market.regime
    @test "flat" in market.regime
    @test all(==(2),market.coins[market.eligible])
    @test TradingStrategy.regimeweights([100.0,1.0];cap=0.5) ≈ [0.5,0.5]
    @test_throws AssertionError TradingStrategy.regimeweights([1.0];cap=0.5)
    @test_throws AssertionError TradingStrategy.checkregimeconfig(TradingStrategy.RegimeConfig(mincoins=2))
    online = TradingStrategy.marketcalendar(histories,config;
        startdt=start,enddt=last(sine.opentime),retrospective=false)
    @test all(online.confirmed .<= online.opentime)
    @test any(market.confirmed .> market.opentime)
    # Missing coin coverage cannot become flat or survive state confirmation.
    missinghistory = Dict("SINE"=>sine,"DOUBLESINE"=>double[2000:end,:])
    missingmarket = TradingStrategy.marketcalendar(missinghistory,config;
        startdt=start,enddt=last(sine.opentime))
    @test all(==("unknown"),missingmarket.regime[1:60])
    folds = TradingStrategy.FoldConfig(anchor=start,foldminutes=6000,
        blockminutes=360,gapminutes=60,minminutes=120)
    calendar = TradingStrategy.foldcalendar(market,Dict(coin=>df.opentime for (coin,df) in histories),
        folds;horizonminutes=60)
    @test nrow(calendar)>0
    @test all(in(["train","eval","test"]),calendar.set)
    @test length(unique(calendar.foldid))>1
    @test calendar == TradingStrategy.foldcalendar(market,
        Dict(coin=>df.opentime for (coin,df) in histories),folds;horizonminutes=60)
    loaded = String[]
    previous = Ref(WeakRef(nothing))
    loader = coin -> begin
        @test isnothing(previous[].value)
        push!(loaded,coin)
        timestamps = copy(histories[coin].opentime)
        previous[] = WeakRef(timestamps)
        timestamps
    end
    streamed = TradingStrategy.foldcalendar(market,loader,sort!(collect(keys(histories))),
        folds;horizonminutes=60)
    @test isequal(streamed,calendar)
    @test loaded==["DOUBLESINE","SINE"]
    @test isnothing(previous[].value)
    @test_throws AssertionError TradingStrategy.foldcalendar(market,
        Dict("SINE"=>sine.opentime),folds;horizonminutes=61)
    projected = TradingStrategy.calendarsets(sine.opentime,calendar)
    counts = TradingStrategy.calendarcounts(sine.opentime,streamed)
    @test counts==(train=count(==("train"),projected.set),
        eval=count(==("eval"),projected.set),test=count(==("test"),projected.set))
    @test all(>(0),values(counts))
    @test TradingStrategy.calendarcounts(DateTime[],calendar)==(train=0,eval=0,test=0)
    @test_throws AssertionError TradingStrategy.calendarcounts(reverse(sine.opentime),calendar)
    @test all(!=("unknown"),projected.regime)
    @test allunique(projected.opentime)
    @test projected == TradingStrategy.calendarsets(double.opentime,calendar)
    @test all(calendar.start[2:end] .- calendar.stop[1:end-1] .>= Minute(61))
    schedule = TradingStrategy.walkforwardschedule(start,last(sine.opentime);
        trainminutes=6000,evalminutes=3000,testminutes=3000,gapminutes=60)
    @test nrow(schedule)>1
    @test all(schedule.teststart[2:end] .> schedule.teststop[1:end-1])
    selected = TradingStrategy.walkforwardsets(projected,market,first(eachrow(schedule)))
    @test all(in(["train","eval","test"]),selected.set)
    @test all(!=("unknown"),selected.regime)
    for row in eachrow(selected[selected.set .== "train",:])
        ix = searchsortedlast(market.opentime,row.opentime)
        @test market.confirmed[ix] <= schedule.trainstop[1]
    end
    @test TradingStrategy.checkregimeconfig(config) === nothing
end
