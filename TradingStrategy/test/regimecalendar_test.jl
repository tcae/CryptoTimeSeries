using Test, Dates, DataFrames
import TradingStrategy, TestOhlcv, Ohlcv, EnvConfig

"""Retain the original dictionary-based market aggregation as a numerical regression oracle."""
function dictionarymarketcalendar(observations, config; startdt, enddt, retrospective=true)
    times = collect(ceil(startdt,Minute(config.stepminutes)):Minute(config.stepminutes):enddt)
    locations = [Dict(time=>ix for (ix,time) in enumerate(df.opentime)) for df in observations]
    market = DataFrame(opentime=times,slope=zeros(length(times)),short_slope=zeros(length(times)),
        volatility=zeros(length(times)),breadth=zeros(length(times)),coins=zeros(Int,length(times)),
        eligible=falses(length(times)))
    for (ix,time) in enumerate(times)
        rows = [(df,locations[coinix][time]) for (coinix,df) in enumerate(observations)
            if haskey(locations[coinix],time)]
        market.coins[ix] = length(rows)
        length(rows)>=config.mincoins || continue
        volumes = [df.volume[row] for (df,row) in rows]
        all(>(0),volumes) || continue
        weights = TradingStrategy.regimeweights(volumes;cap=config.weightcap)
        market.slope[ix] = sum(weights .* [df.slope[row] for (df,row) in rows])
        market.short_slope[ix] = sum(weights .* [df.short_slope[row] for (df,row) in rows])
        market.volatility[ix] = sum(weights .* [df.volatility[row] for (df,row) in rows])
        market.breadth[ix] = sum(weights .* [df.slope[row]>0 for (df,row) in rows])
        market.eligible[ix] = true
    end
    return TradingStrategy.detectregimes(market,config;retrospective)
end

@testset "Appended hourly observations preserve the shared market" begin
    start = DateTime(2025,1,1)
    config = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,
        stepminutes=60,mincoins=2,weightcap=0.6,confirmation=2)
    empty = DataFrame(opentime=DateTime[],slope=Float64[],short_slope=Float64[],
        volume=Float64[],volatility=Float64[])
    firstcoin = DataFrame(opentime=start .+ Hour.([-1,0,1,3,4,7]),slope=fill(0.01,6),
        short_slope=fill(0.02,6),volume=fill(100.0,6),volatility=fill(0.01,6))
    secondcoin = DataFrame(opentime=start .+ Hour.([0,1,2,3,4,8]),slope=fill(-0.02,6),
        short_slope=fill(-0.03,6),volume=fill(1.0,6),volatility=fill(0.02,6))
    secondcoin.volume[5] = 0.0
    offgrid = DataFrame(opentime=start .+ Minute.([30,60,180,240,360]),slope=fill(0.03,5),
        short_slope=fill(0.04,5),volume=fill(5.0,5),volatility=fill(0.03,5))
    observations = [empty,firstcoin,secondcoin,offgrid,empty]
    enddt = start+Hour(6)
    for retrospective in (false,true)
        expected = dictionarymarketcalendar(observations,config;startdt=start,enddt,retrospective)
        @test isequal(TradingStrategy.marketcalendar(observations,config;
            startdt=start,enddt,retrospective),expected)
        mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
            ranges = UnitRange{Int}[]
            rows = 0
            for observation in observations
                EnvConfig.appenddf(observation,"observations";folderpath=folder)
                push!(ranges,(rows+1):(rows+nrow(observation)))
                rows += nrow(observation)
            end
            mapped = EnvConfig.readtable("observations";folderpath=folder,format=:arrow,copycols=false)
            @test nrow(mapped)==rows
            @test !(mapped.slope isa Vector)
            views = [view(mapped,range,:) for range in ranges]
            @test isequal(TradingStrategy.marketcalendar(views,config;
                startdt=start,enddt,retrospective),expected)
        end
    end
    @test all(!,TradingStrategy.marketcalendar([empty,empty],config;
        startdt=start,enddt).eligible)
    @test_throws AssertionError TradingStrategy.marketcalendar([firstcoin[end:-1:1,:]],config;
        startdt=start,enddt)
    duplicate = vcat(firstcoin[1:1,:],firstcoin)
    @test_throws AssertionError TradingStrategy.marketcalendar([duplicate],config;
        startdt=start,enddt)
end

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
