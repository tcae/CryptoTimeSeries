using Test, Dates, DataFrames
import Targets, Features, Ohlcv, EnvConfig

"""Build an isolated Trend07 fixture with matching causal feature inputs."""
function trend07prepared(prices::AbstractVector;bandratio::Real=0.01,
        openwindow::Int=10,maxwindow::Int=20)
    ohlcv = Ohlcv.defaultohlcv("SINE")
    times = DateTime(2025) .+ Minute.(0:length(prices)-1)
    history = DataFrame(opentime=times,open=copy(prices),high=copy(prices),
        low=copy(prices),close=copy(prices),pivot=copy(prices),
        basevolume=ones(Float32,length(prices)))
    Ohlcv.setdataframe!(ohlcv,history)
    target = Targets.Trend07(;bandratio,openwindow,maxwindow)
    features = Targets.regressionfeatures(target)
    Features.setbase!(features,ohlcv)
    Targets.setbase!(target,features)
    return target,features
end

"""Independent chronological search implementing fixed-anchor first-passage rules."""
function trend07reference(prices::AbstractVector,bandratio::Real,openwindow::Int,maxwindow::Int)
    extreme = Features.extremesfeatures(prices;bandratio,maxwindow)
    _, recent = Features.rollingregression(prices,5)
    labels = fill(Targets.allclose,length(prices))
    gains = zeros(Float32,length(prices))
    for ix in eachindex(prices)
        hits = zeros(Int,2)
        for (side,(slope,age)) in enumerate(((extreme.minnrg[ix],extreme.minage[ix]),
                (extreme.maxnrg[ix],extreme.maxage[ix])))
            direction = side == 1 ? 1 : -1
            direction*slope > 0 || continue
            anchor = findfirst(candidate -> ix-candidate == age,1:ix)
            for future in ix+1:min(length(prices),ix+maxwindow)
                direction*(prices[future]-prices[anchor]) <= 0 && break
                gain = (prices[future]-prices[ix])/prices[ix]
                if direction*gain >= bandratio
                    hits[side] = future
                    break
                end
            end
        end
        longhit,shorthit = hits
        if ix >= 5 && recent[ix] > 0 && longhit != 0 && longhit <= ix+openwindow
            labels[ix] = Targets.longopen
            hit = longhit
        elseif ix >= 5 && recent[ix] < 0 && shorthit != 0 && shorthit <= ix+openwindow
            labels[ix] = Targets.shortopen
            hit = shorthit
        elseif longhit != 0 && (shorthit == 0 || longhit < shorthit)
            labels[ix] = Targets.longhold
            hit = longhit
        elseif shorthit != 0 && (longhit == 0 || shorthit < longhit)
            labels[ix] = Targets.shorthold
            hit = shorthit
        else
            hit = 0
        end
        hit == 0 || (gains[ix] = (prices[hit]-prices[ix])/prices[ix])
    end
    return labels,gains
end

@testset "Trend07 inclusive anchors, boundaries and precedence" begin
    EnvConfig.init(EnvConfig.test)
    for (prices,anchor,direction,expected) in (
            ([100.0,101,112.5],90.0,1,3),
            ([100.0,90,112.5],90.0,1,0),
            ([100.0,89,112.5],90.0,1,0),
            ([100.0,91,112.5],90.0,1,3),
            ([100.0,110,87.5],110.0,-1,0),
            ([100.0,111,87.5],110.0,-1,0),
            ([100.0,109,87.5],110.0,-1,3))
        tree = Features.ExtremeSummaryTree(prices)
        @test Targets._trend07_hit(tree,1,3,anchor,0.125,direction) == expected
    end
    tree = Features.ExtremeSummaryTree([100.0,101,112.5,120])
    @test Targets._trend07_hit(tree,1,2,90.0,0.125,1) == 0
    @test Targets._trend07_hit(tree,1,3,90.0,0.125,1) == 3
    @test Targets._trend07_firstcrossing(tree,2,4,112.5,true) == 3
    @test Targets._trend07_firstcrossing(tree,2,4,101.0,false) == 2
    @test Targets._trend07_anchor(10,4f0) == 6
    @test_throws AssertionError Targets._trend07_anchor(10,4.5)
    @test Targets._trend07_label(3,2,3,1.0) == (Targets.longopen,3)
    @test Targets._trend07_label(2,3,3,-1.0) == (Targets.shortopen,3)
    @test Targets._trend07_label(3,2,3,0.0) == (Targets.shorthold,2)
    @test Targets._trend07_label(2,3,3,0.0) == (Targets.longhold,2)
    @test Targets._trend07_label(0,0,3,1.0) == (Targets.allclose,0)
    @test Targets._trend07_label(4,0,3,1.0) == (Targets.longhold,4)
    @test_throws AssertionError Targets.Trend07(openwindow=21,maxwindow=20)
    @test_throws AssertionError Targets.Trend07(bandratio=0)
    @test_throws AssertionError Targets.Trend07(bandratio=NaN)
    @test_throws AssertionError Targets.Trend07(openwindow=1)
    @test Targets._trend07_label(2,2,3,0.0) == (Targets.allclose,0)
end

@testset "Trend07 labels, exact dependency windows and interfaces" begin
    for prices in ([100.0+ix for ix in 0:99], [200.0-ix for ix in 0:99],
            [100.0+4sin(ix*0.23)+2sin(ix*0.71) for ix in 1:601],
            Float32[100+4sin(ix*0.19) for ix in 1:301],fill(100.0,61))
        target,features = trend07prepared(prices)
        expected,gains = trend07reference(prices,0.01,10,20)
        @test Targets.labels(target) == expected
        @test Targets.relativegain(target) ≈ gains
        @test Targets.lookaheadminutes(target) == 20
        @test Targets.finallabelcount(target,length(prices)) == length(prices)-20
        @test Targets.labels(target,target.df.opentime[5],target.df.opentime[10]) == expected[5:10]
        @test Targets.relativegain(target,5,10) == gains[5:10]
        @test Targets.labelbinarytargets(target,Targets.longopen) == (expected .== Targets.longopen)
        @test Targets.labelrelativegain(target,Targets.longopen) == gains.*(expected .== Targets.longopen)
        @test Targets.labelvalues(target) == Targets.df(target)
        @test Targets.firstrowix(target) == 1
        @test Targets.lastrowix(target) == length(prices)
        @test Targets.uniquelabels(target) == [Targets.longopen,Targets.longhold,
            Targets.shortopen,Targets.shorthold,Targets.allclose]
        @test occursin("Trend07",Targets.describe(target))
        @test occursin("inclusive-invalidation",Targets.configdescription(target))
        @test Targets.Trend07(openwindow=10,maxwindow=20,f6=features).f6 === features
        @test_throws ArgumentError Targets.setbase!(target,target.ohlcv)
        original = copy(features.fdf)
        Targets.supplement!(target)
        @test features.fdf == original
        Targets.removebase!(target)
        @test isempty(Targets.labels(target))
        @test isempty(Targets.relativegain(target))
        @test Targets.lastrowix(target) == 0
    end
    rising,_ = trend07prepared([100.0+ix for ix in 0:99])
    @test Targets.labels(rising)[50] == Targets.longopen
    prefix = [95.0,96,97,98,99,100]
    atopen,_ = trend07prepared(vcat(prefix,[101.0,102,112.5,113,114]);
        bandratio=0.125,openwindow=3,maxwindow=4)
    athold,_ = trend07prepared(vcat(prefix,[101.0,102,103,112.5,113]);
        bandratio=0.125,openwindow=3,maxwindow=4)
    toolate,_ = trend07prepared(vcat(prefix,[101.0,102,103,104,112.5]);
        bandratio=0.125,openwindow=3,maxwindow=4)
    @test Targets.labels(atopen)[6] == Targets.longopen
    @test Targets.labels(athold)[6] == Targets.longhold
    @test Targets.labels(toolate)[6] == Targets.allclose
    prices = [100.0+4sin(ix*0.2) for ix in 1:150]
    target,_ = trend07prepared(prices)
    ix = 70
    changed = copy(prices)
    changed[1:ix-20] .= 1000
    changed[ix+21:end] .= 2000
    other,_ = trend07prepared(changed)
    @test Targets.labels(other)[ix] == Targets.labels(target)[ix]
    @test Targets.relativegain(other)[ix] == Targets.relativegain(target)[ix]
end

@testset "Trend07 gap isolation and reusable store extension" begin
    prices = [100.0+4sin(ix*0.2) for ix in 1:150]
    target,features = trend07prepared(prices)
    odf = Ohlcv.dataframe(target.ohlcv)
    odf[76:end,:opentime] .+= Minute(79)
    Features.setbase!(features,target.ohlcv;usecache=false)
    Targets.setbase!(target,features)
    for rows in (1:75,76:150)
        expected,_ = trend07prepared(prices[rows])
        @test Targets.labels(target)[rows] == Targets.labels(expected)
    end
    @test Targets.finallabelcount(target,150) == 130
    fullcodes = Targets._targetcodes(Targets.Trend07(openwindow=10,maxwindow=20),target.ohlcv)
    @test fullcodes[1:130] == Targets.tradelabelcode.(Targets.labels(target)[1:130])
    @test all(==(Targets.tradelabelcode(Targets.ignore)),fullcodes[131:end])
    for lastix in (10,50,80,110,140)
        short = Ohlcv.ohlcvview(target.ohlcv,1:lastix)
        config = Targets.Trend07(openwindow=10,maxwindow=20)
        codes = Targets._targetcodes(config,short)
        @test Targets._extendedtargetcodes(config,target.ohlcv,codes) == fullcodes
    end
    oldcoinspath = EnvConfig.activecoinspath
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        try
            EnvConfig.setcoinspath!(folder)
            config = Targets.Trend07(openwindow=10,maxwindow=20)
            identity = Targets.configdescription(config)
            prefix = Ohlcv.defaultohlcv("SINE")
            Ohlcv.setdataframe!(prefix,DataFrame(odf[1:110,:];copycols=true))
            Targets.updatetargets!(prefix,"T07TEST",config;confighash=identity)
            stored = Targets.updatetargets!(target.ohlcv,"T07TEST",config;confighash=identity)
            @test stored.label == fullcodes
            @test stored.opentime == odf.opentime
            @test Targets.readtargets("SINE",target.ohlcv.quotecoin,"T07TEST";
                confighash=identity).label == fullcodes
            @test_throws AssertionError Targets.readtargets("SINE",target.ohlcv.quotecoin,
                "T07TEST";confighash="incorrect")
        finally
            EnvConfig.setcoinspath!(oldcoinspath)
        end
    end
end
