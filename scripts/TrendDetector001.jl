using Classify

"""
- start with Testohlcv to check learning works
- find out how many blocks of a coin are formed by using only liquid coins
- use liquid ranges to determine test, eval, train ranges and store them in persistent dataframe
- generate features and targets per coin and store them in test, eval, train coin specific DataFrames
  - advantage: features/tragets can be loaded according to purpose without time consuming split but dapaptation and evaluation can be done for all coins or coin specific
- then check adaptation per coin
- then adapt with all coins and retrain per coin
- one level base classifier (no combi)

hyper parameters:
- trend gain: 1%, 2%, 4%
- trend breaking gain: 0.5%, 1%

classes: binary longbuy yes vs no (=longclose) with hysteresis using likelihood % of longbuy violate

"""
module TrendDetector001
using Test, Dates, Logging, CSV, JDF, DataFrames, Statistics, MLUtils
using CategoricalArrays
using EnvConfig, Classify, CryptoXch, Ohlcv, Features, Targets

#TODO regression from last trend pivot as feature
"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 3

function f6config01()
    featcfg = Features.Features006()
    Features.addstd!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=5, offset=5)
    Features.addgrad!(featcfg, window=5, offset=10)
    Features.addgrad!(featcfg, window=15, offset=15)
    Features.addgrad!(featcfg, window=15, offset=30)
    Features.addgrad!(featcfg, window=15, offset=45)
    Features.addgrad!(featcfg, window=60, offset=60)
    Features.addgrad!(featcfg, window=60*4, offset=120)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60)
    Features.addmindist!(featcfg, window=60*5, offset=60)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end


function calcfeatures!(featcfg::Features.AbstractFeatures, base::AbstractString, startdt::DateTime, enddt::DateTime)
    (verbosity >= 3) && println("$(EnvConfig.now()) loading $base") 
    xc = CryptoXch.XchCache()
    ohlcv = CryptoXch.cryptodownload(xc, base, "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)

    println("$(EnvConfig.now()) feature calculation")
    Features.setbase!(featcfg, ohlcv, usecache=true)
    # Features.write(featcfg)
    features = Features.features(featcfg)

    features = Array(features)  # change from df to array
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    return features
end

"""
returns targets
feature base has to be set before calling because that determines the ohlcv and relevant time range
"""
function calctargets!(trgcfg::Targets.AbstractTargets, featcfg::Features.AbstractFeatures)
    ohlcv = Features.ohlcv(featcfg)
    features = Features.features(featcfg)
    fot = Features.opentime(featcfg)
    (verbosity >= 3) && println("$(EnvConfig.now()) target calculation fromm $(fot[begin]) until $(fot[end])")
    Targets.setbase!(trgcfg, ohlcv)
    targets = Targets.labels(trgcfg, fot[begin], fot[end])
    Targets.labeldistribution(targets)
    targets = Targets.labelbinarytargets(trgcfg, longbuy, fot[begin], fot[end])
    targets = [lb ? longbuy : allclose for lb in targets]
    @assert size(features, 1) == length(targets) "size(features, 1)=$(size(features, 1)) != length(targets)=$(length(targets))"
    # (verbosity >= 3) && println(describe(trgcfg.df, :all))

    return targets
end

function evalclassifier(nn)
    println("evaluating classifier $(nn.fileprefix)")
    Classify.evaluateclassifier(nn)
    # println("$(EnvConfig.now()) load machine from file $(nn.fileprefix) for regressionwindow $regrwindow and predict")
    # nntest = loadnn(nn.fileprefix)
    # println(nntest)
    # predtest = predict(nntest, features)
    # @assert pred ≈ predtest  "NN results differ from loaded NN: pred[:, 1:5] = $(pred[:, begin:begin+5]) predtest[:, 1:5] = $(predtest[:, begin:begin+5])"
    println(nn)
end

"add colums period and gap to previous range in readable canonical form "
function addperiodgap!(rangedf) 
    rangedf.period = canonicalize.(rangedf[!, :enddt] .- rangedf[!, :startdt])
    rangedf.gap = canonicalize.(rangedf[!, :startdt] .- vcat(rangedf[begin:begin, :startdt], rangedf[begin:end-1, :enddt]))
end

"""
The function generates targets files and features files in the current log folder for the set types ["test", "train", "eval"] of the provided base coins ohlcv data.  
The input log dataframe rangedf is supplemented with the index information of each set type range.  
It returns a tuple of featuresdict, targetsdict both with keys ["test", "train", "eval"] and dataframe values.   
"""
function featurestargetsliquidranges!(settypesdf, rangedf, basecoin, featconfig, trgconfig; samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=Features.requiredminutes(featconfig), minpartitionsize=12*60, maxpartitionsize=2*24*60)
    (verbosity >= 3) && println("$(EnvConfig.now()) loading $basecoin") 
    ohlcv = Ohlcv.read(basecoin)
    rv = Ohlcv.liquiditycheck(ohlcv)
    ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    
    samplesets = CategoricalArray(samplesets)
    featuresdict = Dict() # key = settype, value = DataFrame
    targetsdict = Dict() # key = settype, value = DataFrame
    for settype in levels(samplesets)
        featuresdict[settype] = DataFrame()
        targetsdict[settype] = DataFrame()
    end

    for rng in rv # rng indices are related to ohlcv dataframe rows
        if rng[end] - rng[begin] > 0
            (verbosity >= 2) && println("$(EnvConfig.now()) calculating features and targets for $(ohlcv.base) range $rng from $(ot[rng[begin]]) until $(ot[rng[end]]) with $(rng[end] - rng[begin]) samples")
            ohlcvview = Ohlcv.ohlcvview(ohlcv, rng)
            Features.setbase!(featconfig, ohlcvview, usecache=true)
            rngfeatures = Features.features(featconfig)

            @assert size(rngfeatures, 1) > 0 "features data of $(ohlcv.base) range $rng with $(rng[end] - rng[begin] + 1) rows from $(ot[rng[begin]]) until $(ot[rng[end]]) not matching features size $(size(rngfeatures, 1))"
            rngtargets = DataFrame(targets=calctargets!(trgconfig, featconfig))
            @assert size(rngtargets, 1) > 0 "targets data of $(ohlcv.base) range $rng with $(rng[end] - rng[begin] + 1) rows from $(ot[rng[begin]]) until $(ot[rng[end]]) not matching targets size $(size(rngtargets, 1))"
            psets = Classify.setpartitions(1:size(rngfeatures, 1), samplesets, partitionsize=partitionsize, gapsize=gapsize, minpartitionsize=minpartitionsize, maxpartitionsize=maxpartitionsize)

            for settype in levels(samplesets)
                for psrng in psets[settype] # psrng indices are related to rngfeatures rows (not to ohlcv dataframe rows)
                    six = size(featuresdict[settype], 1) + 1
                    eix = six + size(rngfeatures[psrng, :], 1) - 1
                    beforesize = size(featuresdict[settype], 1)
                    featuresdict[settype] = vcat(featuresdict[settype], rngfeatures[psrng, :])
                    @assert size(featuresdict[settype], 1) == beforesize + size(rngfeatures[psrng, :], 1) "size(featuresdict[settype], 1)=$(size(featuresdict[settype], 1)) != beforesize=$beforesize + size(rngfeatures[psrng, :], 1)=$(size(rngfeatures[psrng, :], 1))"
                    beforesize = size(targetsdict[settype], 1)
                    targetsdict[settype] = vcat(targetsdict[settype], rngtargets[psrng, :])
                    @assert size(targetsdict[settype], 1) == beforesize + size(rngtargets[psrng, :], 1) "size(targetsdict[settype], 1)=$(size(targetsdict[settype], 1)) != beforesize=$beforesize + size(rngtargets[psrng, :], 1)=$(size(rngtargets[psrng, :], 1))"
                    sixohlcv = rng[begin] + psrng[begin] - 1
                    eixohlcv = rng[begin] + psrng[end] - 1
                    push!(rangedf, (coin=ohlcv.base, settype=settype, ohlcvrange=sixohlcv:eixohlcv, startdt=ot[sixohlcv], enddt=ot[eixohlcv], dfrange=six:eix, liquidrange=rng))
                end
            end
        else
            @error "unexpected zero length range for " ohlcv.base rng rv
        end
    end
    sort!(rangedf, :ohlcvrange)
    addperiodgap!(rangedf)

    featuresfname = Dict() # key = settype, value = filename without path
    targetsfname = Dict() # key = settype, value = filename without path
    for settype in levels(samplesets)
        featuresfname[settype] = settype * "_" * Features.describe(featconfig) * ".jdf"
        targetsfname[settype] = settype * "_" * Targets.describe(trgconfig) * ".jdf"
        push!(settypesdf, (coin=ohlcv.base, settype=settype, featuresfname=featuresfname[settype], targetsfname=targetsfname[settype]))

        savedflogfolder(featuresdict[settype], featuresfname[settype])
        savedflogfolder(targetsdict[settype], targetsfname[settype])

    end
    return featuresdict, targetsdict
end

rangefilename() = "ranges.jdf"
settypesfilename() = "settypesfiledict.jdf"


"Saves a given dataframe df in the current log folder using the given filename"
function savedflogfolder(df, filename)
    filepath = EnvConfig.logpath(filename)
    try
        JDF.savejdf(filepath, df)
        (verbosity >= 3) && println("$(EnvConfig.now()) saved dataframe to $(filepath)")
    catch e
        Logging.@error "exception $e detected when writing $(filepath)"
    end
end

"Reads and returns a dataframe from filename in the current log folder"
function readdflogfolder(filename)
    df = DataFrame()
    filepath = EnvConfig.logpath(filename)
    try
        if isdir(filepath)
            (verbosity >= 3) && print("$(EnvConfig.now()) loading dataframe from  $(filepath)")
            df = DataFrame(JDF.loadjdf(filepath))
            (verbosity >= 2) && println(" - $(EnvConfig.now()) loaded $(size(df, 1)) rows successfully")
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(filepath)")
        end
    catch e
        Logging.@error "exception $e detected"
    end
    return df
end

trendsineconfig() = Targets.Trend(5, 30, Targets.thresholds((longbuy=0.1, longhold=0.005, shorthold=-0.005, shortbuy=-0.1)))

function sinetest2()
    # EnvConfig.init(production)
    startdt = DateTime("2023-02-17T13:30:00")
    enddt = startdt + Day(10) - Minute(1)
    featcfg = f6config01()
    trgcfg = trendsineconfig()
    rangedf = DataFrame()
    settypesdf = DataFrame()
    fdict, tdict = featurestargetsliquidranges!(settypesdf, rangedf, ohlcv, featconfig, trgconfig; samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=Features.requiredminutes(featconfig), minpartitionsize=12*60, maxpartitionsize=2*24*60)
end

function sinetest()
    # EnvConfig.init(production)
    startdt = DateTime("2023-02-17T13:30:00")
    enddt = startdt + Day(10) - Minute(1)
    featcfg = f6config01()
    trgcfg = trendsineconfig()
    features = calcfeatures!(featcfg, "SINE", startdt, enddt)
    targets = calctargets!(trgcfg, featcfg)
    # println("targets=$(trgcfg.df[1:300, :])")

    println("$(EnvConfig.now()) subset creation")
    setranges = Classify.setpartitions(1:length(targets), Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/13)
    println("setranges: $setranges")
    println("$(EnvConfig.now()) get train features subset")
    trainfeatures = Classify.subsetdim2(features, setranges["train"])
    println("$(EnvConfig.now()) get train targets subset")
    traintargets = Classify.subsetdim2(targets, setranges["train"])

    println("$(EnvConfig.now()) NN model cration")
    nn = Classify.model001(size(trainfeatures, 1), unique(targets), "TrendDetector10pct30minutes")
    nn.featuresdescription = Features.describe(featcfg)
    nn.targetsdescription = Targets.describe(trgcfg)
    println("$(EnvConfig.now()) adapting machine for targets $(nn.targetsdescription) \nusing features: $(nn.featuresdescription)")
    nn = Classify.adaptnn!(nn, trainfeatures, traintargets)
    println("$(EnvConfig.now()) predicting")
    pred = Classify.predict(nn, features)
    println("$(EnvConfig.now()) predictions to dataframe")
    push!(nn.predictions, Classify.predictionsdataframe(nn, setranges, targets, pred, featcfg))
    # predictiondistribution(pred, nn.mnemonic)

    println("saving adapted classifier $(nn.fileprefix)")
    # println(nn)
    Classify.savenn(nn)
    return nn
end

trendccoinonfig(minwindow, maxwindow, buy, hold) = Targets.Trend(minwindow, maxwindow, Targets.thresholds((longbuy=buy, longhold=hold, shorthold=-hold, shortbuy=-buy)))

function BTCtest()
    # EnvConfig.init(production)
    enddt = DateTime("2024-12-20T22:58:00")
    startdt = enddt - Year(10)
    # startdt = enddt - Month(6)
    featcfg = f6config01()
    features = calcfeatures!(featcfg, "BTC", startdt, enddt)
    for (minwindow, maxwindow, buy, hold) in [(10, 4*60, 0.01, 0.005), (10, 4*60, 0.01, 0.01)]
        trgcfg = trendccoinonfig(minwindow, maxwindow, buy, hold)
        targets = calctargets!(trgcfg, featcfg)
        # println("targets=$(trgcfg.df[1:300, :])")

        println("$(EnvConfig.now()) subset creation")
        setranges = Classify.setpartitions(1:length(targets), ["train", "test", "train", "train", "eval", "train"], partitionsize=50*maxwindow, gapsize=maxwindow)
        # setranges = Classify.setpartitions(1:length(targets), Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/13)
        println("setranges: $setranges")
        trainfeatures = Classify.subsetdim2(features, setranges["train"])
        println("$(EnvConfig.now()) got train features subset size=$(size(trainfeatures))")
        traintargets = Classify.subsetdim2(targets, setranges["train"])
        println("$(EnvConfig.now()) got train targets subset size=$(size(traintargets))")
        # trainfeatures, traintargets = oversample(trainfeatures, traintargets)  # all classes are equally trained
        trainfeatures, traintargets = undersample(trainfeatures, traintargets)  # all classes are equally trained
        println("$(EnvConfig.now()) after undersampling train features subset size=$(size(trainfeatures))")
        println("$(EnvConfig.now()) after undersampling train targets subset size=$(size(traintargets)) type=$(typeof(traintargets))")
        print("Training ")
        Targets.labeldistribution(traintargets)

        println("$(EnvConfig.now()) NN model cration")
        nn = Classify.model001(size(trainfeatures, 1), unique(targets), "TrendDetector$(round(Int, lb*1000))permille$(window)minutes")
        nn.featuresdescription = Features.describe(featcfg)
        nn.targetsdescription = Targets.describe(trgcfg)
        println("$(EnvConfig.now()) adapting machine for targets $(nn.targetsdescription) \nusing features: $(nn.featuresdescription)")
        nn = Classify.adaptnn!(nn, trainfeatures, traintargets)
        println("$(EnvConfig.now()) predicting")
        pred = Classify.predict(nn, features)
        println("$(EnvConfig.now()) predictions to dataframe")
        push!(nn.predictions, Classify.predictionsdataframe(nn, setranges, targets, pred, featcfg))
        # predictiondistribution(pred, nn.mnemonic)

        println("saving adapted classifier $(nn.fileprefix)")
        # println(nn)
        Classify.savenn(nn)
        evalclassifier(nn)
    end
    return nn
end

function liquidcoinstest()
    """
    hyper parameters:
    - target time window, extend to incease lieklihood of gain reached but be less focused on short term features
    - target gain, reduce gain to increase lieklihood but be more merged with noise
    - class balancing yes/no/factor 
    - how many situations observed, i.e. train on all coins and then zoom in on one
    """
    startdt = nothing # DateTime("2024-03-01T00:00:00")
    enddt =   nothing # DateTime("2024-06-06T09:00:00")
    # coins = ["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"]
    coins = nothing # ["BTC"]

    coinsdf = Ohlcv.liquidcoins()
    filtered_df = coinsdf # filter(row -> row.basecoin in coins, coinsdf)
    println("evaluating: $coins \n coinsdf=$coinsdf \n filtered_df=$filtered_df")
    df = Classify.evaluateclassifiers([Classify.Classifier005], filtered_df, startdt, enddt)
    # df = Classify.readsimulation()
    kpidf, gdf = Classify.kpioverview(df, Classify.Classifier005)
    sort!(kpidf, [:gain_sum], rev=true)
    println(kpidf)
    # println(gdf[kpidf[1, :groupindex]])
    # println(gdf[kpidf[2, :groupindex]])
end

function AdaTest()
    EnvConfig.setlogpath("2528-TrendDetector001-Ada")
    rangedf = DataFrame()
    settypesdf = DataFrame()
    featconfig = f6config01()
    trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01)
    fdict, tdict = featurestargetsliquidranges!(settypesdf, rangedf, "ADA", featconfig, trgconfig)
    println("ADA rangedf: $rangedf")
    println("ADA settypesdf: $settypesdf")

    savedflogfolder(rangedf[!, Not([:period, :gap])], rangefilename())
    savedflogfolder(settypesdf, settypesfilename())

    for settype in ["train", "test", "eval"]
        println("$settype: featuressize=$(size(fdict[settype])) targetssize=$(size(tdict[settype]))")
    end

    #TODO read settypesdf then read features and targets and compare sizes with fdict, tdict
    rangedf2 = readdflogfolder(rangefilename())
    addperiodgap!(rangedf2)
    println("rangedf equal test = $(rangedf==rangedf2)")
    settypesdf2 = readdflogfolder(settypesfilename())
    println("settypesdf equal test = $(settypesdf==settypesdf2)")
    featuresfname = Dict() # key = settype, value = filename without path
    targetsfname = Dict() # key = settype, value = filename without path
    featuresdict = Dict() # key = settype, value = DataFrame
    targetsdict = Dict() # key = settype, value = DataFrame
    for settype in unique(settypesdf2[!, :settype])
        featuresfname[settype] = settypesdf2[(settypesdf2[!, :coin] .== "ADA") .&& (settypesdf2[!, :settype] .== settype), :featuresfname][begin] # begin to reduce String vector to String
        targetsfname[settype] = settypesdf2[(settypesdf2[!, :coin] .== "ADA") .&& (settypesdf2[!, :settype] .== settype), :targetsfname][begin] # begin to reduce String vector to String
        featuresdict[settype] = readdflogfolder(featuresfname[settype])
        targetsdict[settype] = readdflogfolder(targetsfname[settype])
        println("features[$settype] size equal test = $(size(featuresdict[settype])==size(fdict[settype]))")
        println("targets[$settype] size equal test = $(size(targetsdict[settype])==size(tdict[settype]))")
        println("features[$settype] equal test = $(featuresdict[settype]==fdict[settype])")
        println("targets[$settype] equal test = $(targetsdict[settype]==tdict[settype])")
    end
end


EnvConfig.init(production)
# EnvConfig.init(test)
# EnvConfig.init(training)
Ohlcv.verbosity = 1
CryptoXch.verbosity = 1
Features.verbosity = 1
Targets.verbosity = 2
EnvConfig.verbosity = 1
Classify.verbosity = 1

# EnvConfig.setlogpath("2524-TrendDetector001-Sine")
# nn = sinetest()
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-20_23-03-26_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb")

# nn = BTCtest()

# EnvConfig.setlogpath("2445-TrendDetector001")
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-22_01-48-26_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # class balancing by undersampling - best generalization
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-21_00-27-30_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # class balancing by oversampling
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-21_21-53-16_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # no class balancing
# evalclassifier(nn)

AdaTest()
println("done")
end # of TrendDetector001
