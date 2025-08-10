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
using CategoricalArrays, CategoricalDistributions, Distributions
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
It returns a tuple of featurestargetsdf, targetsdict both with keys ["test", "train", "eval"] and dataframe values.   
"""
function featurestargetsliquidranges!(settypesdf, rangedf, basecoin, featconfig, trgconfig; samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=Features.requiredminutes(featconfig), minpartitionsize=12*60, maxpartitionsize=2*24*60)
    (verbosity >= 3) && println("$(EnvConfig.now()) loading $basecoin") 
    ohlcv = Ohlcv.read(basecoin)
    rv = Ohlcv.liquiditycheck(ohlcv)
    ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    rangeid = size(rangedf, 1) > 0 ? maximum(rangedf[!, :rangeid]) + 1 : Int16(1)
    
    levels = unique(samplesets)
    push!(levels, "noop")
    samplesets = CategoricalArray(samplesets, levels=levels)
    featurestargetsdf = DataFrame()

    for rng in rv # rng indices are related to ohlcv dataframe rows
        if rng[end] - rng[begin] > 0
            (verbosity >= 2) && println("$(EnvConfig.now()) calculating features and targets for $(ohlcv.base) range $rng from $(ot[rng[begin]]) until $(ot[rng[end]]) with $(rng[end] - rng[begin]) samples")
            ohlcvview = Ohlcv.ohlcvview(ohlcv, rng)
            Features.setbase!(featconfig, ohlcvview, usecache=true)
            rngfeaturestargets = copy(Features.features(featconfig))

            @assert size(rngfeaturestargets, 1) > 0 "features data of $(ohlcv.base) range $rng with $(rng[end] - rng[begin] + 1) rows from $(ot[rng[begin]]) until $(ot[rng[end]]) not matching features size $(size(rngfeaturestargets, 1))"
            rngfeaturestargets[:, :targets] = calctargets!(trgconfig, featconfig)
            rngfeaturestargets[:, :rangeid] .= 0
            rngfeaturestargets[:, :rix] .= 0
            rngfeaturestargets[:, :set] = CategoricalVector(fill("noop", size(rngfeaturestargets, 1)), levels=levels)
            psets = Classify.setpartitions(1:size(rngfeaturestargets, 1), samplesets, partitionsize=partitionsize, gapsize=gapsize, minpartitionsize=minpartitionsize, maxpartitionsize=maxpartitionsize)

            for (settype, psrng) in psets # psrng indices are related to rngfeaturestargets rows (not to ohlcv dataframe rows)
                # six = size(featurestargetsdf[settype], 1) + 1
                # eix = six + size(rngfeaturestargets[psrng, :], 1) - 1
                rngfeaturestargets[psrng, :rangeid] .= rangeid
                rngfeaturestargets[psrng, :rix] = collect(Int32.(psrng))
                rngfeaturestargets[psrng, :set] .= settype
                featurestargetsdf = vcat(featurestargetsdf, rngfeaturestargets[psrng, :])
                sixohlcv = rng[begin] + psrng[begin] - 1
                eixohlcv = rng[begin] + psrng[end] - 1
                push!(rangedf, (coin=ohlcv.base, settype=settype, rangeid=rangeid, ohlcvrange=sixohlcv:eixohlcv, startdt=ot[sixohlcv], enddt=ot[eixohlcv], dfrange=psrng, liquidrange=rng))
                rangeid += 1
            end
        else
            @error "unexpected zero length range for " ohlcv.base rng rv
        end
    end
    if size(featurestargetsdf, 1) > 0
        sort!(rangedf, :ohlcvrange)

        featurestargetsfname = "features_targets_" * ohlcv.base * ".jdf"
        savedflogfolder(featurestargetsdf, featurestargetsfname)
        push!(settypesdf, (coin=ohlcv.base, featuresconfig=Features.describe(featconfig), targetsconfig=Targets.describe(trgconfig), featurestargetsfname=featurestargetsfname))
    else
        @info "skipping $basecoin due to no liquid ranges"
    end
    return featurestargetsdf
end

rangefilename() = "ranges.jdf"
settypesfilename() = "settypesfiledict.jdf"
predictionsfilename(coin, classifier) = "predictions_$(coin)_$(classifier).jdf"


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

isdflogfolder(filename) = isdir(EnvConfig.logpath(filename))

trendsineconfig() = Targets.Trend(5, 30, Targets.thresholds((longbuy=0.1, longhold=0.005, shorthold=-0.005, shortbuy=-0.1)))

function sinetest2()
    # EnvConfig.init(production)
    startdt = DateTime("2023-02-17T13:30:00")
    enddt = startdt + Day(10) - Minute(1)
    featcfg = f6config01()
    trgcfg = trendsineconfig()
    rangedf = DataFrame()
    settypesdf = DataFrame()
    ftdf = featurestargetsliquidranges!(settypesdf, rangedf, ohlcv, featconfig, trgconfig; samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=Features.requiredminutes(featconfig), minpartitionsize=12*60, maxpartitionsize=2*24*60)
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
    push!(nn.predictions, Classify.predictionsdataframeold(nn, setranges, targets, pred, featcfg))
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
        push!(nn.predictions, Classify.predictionsdataframeold(nn, setranges, targets, pred, featcfg))
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

settypes() = ["train", "test", "eval"]

function featurestargetscollect(coins, featconfig, trgconfig)
    rangedf = DataFrame()
    settypesdf = DataFrame()
    coincollect = []
    for coin in coins
        ftdf = featurestargetsliquidranges!(settypesdf, rangedf, coin, featconfig, trgconfig)
        if size(ftdf, 1) > 0
            # println("$coin rangedf: $rangedf")
            println("$coin settypesdf: $settypesdf")

            # savedflogfolder(rangedf[!, Not([:period, :gap])], rangefilename())  # with period and gap columns added before
            savedflogfolder(rangedf, rangefilename())  # without period and gap columns added before
            savedflogfolder(settypesdf, settypesfilename())

            push!(coincollect, (coin=coin, ftdf=ftdf))
        end
    end
    # println("len(coins)=$(length(coins)), len(coincollect)=$(length(coincollect))")
    for ct in coincollect
        coin = ct.coin
        rangedf2 = readdflogfolder(rangefilename())
        # addperiodgap!(rangedf2)
        @assert rangedf==rangedf2 "rangedf=$rangedf \n not equal rangedf2 = $rangedf2"
        settypesdf2 = readdflogfolder(settypesfilename())
        @assert settypesdf==settypesdf2 "settypesdf=$settypesdf \n not equal settypesdf2=$settypesdf2"
        @assert size(settypesdf[settypesdf[!, :coin] .== coin, :], 1) == 1 "size(settypesdf[settypesdf[!, :coin] .== coin, :], 1)=$(size(settypesdf[settypesdf[!, :coin] .== coin, :], 1))"
        ftdf2 = readdflogfolder(settypesdf[settypesdf[!, :coin] .== coin, :featurestargetsfname][begin])
        @assert ftdf2 == ct.ftdf
        # println("size(ftdf2)=$(size(ftdf2)) unique(ftdf2[!, :rangeid]$(unique(ftdf2[!, :rangeid]))")
        rdfview = @view rangedf[rangedf[!, :coin] .== coin, :]
        for rrow in eachrow(rdfview)
            ftdfrange = @view ftdf2[ftdf2[!, :rangeid] .== rrow.rangeid, :]
            # println("rrow=$rrow")
            @assert length(rrow.dfrange) == size(ftdfrange, 1) "length(rrow.dfrange)=$(length(rrow.dfrange)) != size(ftdfrange, 1)=$(size(ftdfrange, 1))"
            @assert all(ftdfrange[!, :rangeid] .== rrow.rangeid) "ftdfrange[!, :rangeid]=$(unique(ftdfrange[!, :rangeid])) .!= rrow.rangeid=$(rrow.rangeid)"
            @assert all(ftdfrange[!, :set] .== rrow.settype) "ftdfrange[!, :set]=$(unique(ftdfrange[!, :set])) .!= rrow.settype=$(rrow.settype)"
        end
    end
    # addperiodgap!(rangedf)
end

classifiermenmonic(coins, coinix) = "model001_" * (isnothing(coinix) ? "mix" : coins[coinix])

function getlatestclassifier(coins, coinix, featureconfig, targetconfig)
    nn = nothing
    if isfile(Classify.nnfilename(classifiermenmonic(coins, coinix)))
        nn = Classify.loadnn(classifiermenmonic(coins, coinix))
        println("getlatestclassifier loaded: labels=$(nn.labels)")
    else
        nn = Classify.model001(Features.featurecount(featureconfig), [longbuy, allclose], classifiermenmonic(coins, coinix))
        println("getlatestclassifier new: labels=$(nn.labels)")
    end
    @assert !isnothing(nn)
    return nn
end

function getfeaturestargets(settypesdf, settype, coin)
    ftdffull = readdflogfolder(settypesdf[settypesdf[!, :coin] .== coin, :featurestargetsfname][begin])
    ftdf = @view ftdffull[(ftdffull[!, :set] .== settype), :]
    features = @view ftdf[!, Not([:set, :rix, :rangeid, :targets])]
    targets = @view ftdf[!, :targets]


    # featuresfname = settypesdf[(settypesdf[!, :coin] .== coin) .&& (settypesdf[!, :settype] .== settype), :featuresfname][begin] # begin to reduce String vector to String
    # targetsfname = settypesdf[(settypesdf[!, :coin] .== coin) .&& (settypesdf[!, :settype] .== settype), :targetsfname][begin] # begin to reduce String vector to String
    # features = readdflogfolder(featuresfname)
    features = Array(features)  # change from df to array
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    # targets = readdflogfolder(targetsfname)[!, :targets]
    (verbosity >= 3) && println("typeof(features)=$(typeof(features)), size(features)=$(size(features)), typeof(targets)=$(typeof(targets)), size(targets)=$(size(targets))") 
    return features, targets
end

function showlosses(nn)
    println("$(EnvConfig.now()) evaluating classifier $(nn.mnemonic)")
    packetsize = length(nn.losses) > 20 ? floor(Int, length(nn.losses) / 20) : 1  # only display 20 lines of loss summary
    startp = lastlosses = nothing
    for i in eachindex(nn.losses)
        if i > firstindex(nn.losses)
            if (i % packetsize == 0) || (i == lastindex(nn.losses))
                plosses = mean(nn.losses[startp:i])
                println("epoch $startp-$i loss: $plosses  lossdiff: $((plosses-lastlosses)/lastlosses*100)%")
                startp = i+1
                lastlosses = plosses
            end
        else
            println("loss: $(nn.losses[i])")
            startp = i+1
            lastlosses = nn.losses[i]
        end
    end
end

function adaptclassifierwithcoin!(nn::Classify.NN, settypesdf, coin)
    println("$(EnvConfig.now()) adapting classifier for $coin")
    trainfeatures, traintargets = getfeaturestargets(settypesdf, "train", coin)
    println("before correction: $(Distributions.fit(UnivariateFinite, categorical(string.(traintargets)))))")
    (trainfeatures), traintargets = oversample((trainfeatures), traintargets)  # all classes are equally trained
    # (trainfeatures), traintargets = undersample((trainfeatures), traintargets)  # all classes are equally trained
    println("after oversampling: $(Distributions.fit(UnivariateFinite, categorical(string.(traintargets)))))")
    Classify.adaptnn!(nn, trainfeatures, traintargets)
    (verbosity >= 3) && showlosses(nn)

    # println("$(EnvConfig.now()) predicting")
    # evalfeatures, evaltargets = getfeaturestargets(settypesdf, "eval", coin)
    # pred = Classify.predict(nn, evalfeatures)
    # println("$(EnvConfig.now()) predictions to dataframe")
    # push!(nn.predictions, Classify.predictionsdataframeold(nn, setranges, targets, pred, featcfg))
    # # predictiondistribution(pred, nn.mnemonic)

    # println("saving adapted classifier $(nn.fileprefix)")
    # # println(nn)
    # Classify.savenn(nn)
    # evalclassifier(nn)

    EnvConfig.savebackup(Classify.nnfilename(nn.fileprefix))
    Classify.savenn(nn)
end

function adaptclassifier(coins, coinix, featconfig, trgconfig, settypesdf)
    @assert length(coins) > 0
    coinix = isnothing(coinix) || (coinix < firstindex(coins)) || (coinix > lastindex(coins)) ? nothing : coinix
    coin = isnothing(coinix) ? "mix" : coins[coinix]
    if isfile(Classify.nnfilename(classifiermenmonic(coins, coinix)))
        println("$coin classifier seems to be adapted - found in $(Classify.nnfilename(classifiermenmonic(coins, coinix)))")
        nn = getlatestclassifier(coins, coinix, featconfig, trgconfig)
    else
        println("$(EnvConfig.now()) adapting $coin classifier")
        # if classifier filedoes not exist then create one
        nn = getlatestclassifier(coins, coinix, featconfig, trgconfig)
        dfp = nothing
        if isnothing(coinix) # adapt a mix classifier with all coins
            for coin in coins
                adaptclassifierwithcoin!(nn, settypesdf, coin)
            end
            for coin in coins
                for settype in settypes()
                    evalfeatures, evaltargets = getfeaturestargets(settypesdf, settype, coin)
                    dfp = predictdf(dfp, nn, evalfeatures, evaltargets, settype)
                end
            end
        else  # adapt a coin specific classifier with all coins
            if !Classify.isadapted(nn)  # if single coin classifier is not adapted than take latest mix classifier as baseline
                nn = getlatestclassifier(coins, nothing, featconfig, trgconfig)
                Classify.setmnemonic(nn, classifiermenmonic(coins, coinix))
                adaptclassifierwithcoin!(nn, settypesdf, coins[coinix])
                for settype in settypes()
                    features, targets = getfeaturestargets(settypesdf, settype, coin)
                    dfp = predictdf(dfp, nn, features, targets, settype)
                end
            end
        end
        savedflogfolder(dfp, predictionsfilename(coin, nn.fileprefix))
        cmdf = Classify.confusionmatrix(dfp)
        (verbosity >=3) && println("cm: $cmdf")
        xcmdf = Classify.extendedconfusionmatrix(dfp)
        (verbosity >=3) && println("xcm: $xcmdf")
        println("$(EnvConfig.now()) finished adapting $coin classifier")
    end
    return nn
end

function adaptclassifiers(coins, featconfig, trgconfig, settypesdf)
    res = []
    if isdflogfolder(settypesfilename())
        nn = adaptclassifier(coins, nothing, featconfig, trgconfig, settypesdf)
        push!(res, (coin=nothing, nn=nn))
        for ix in eachindex(coins)
            nn = adaptclassifier(coins, ix, featconfig, trgconfig, settypesdf)
            push!(res, (coin=coins[ix], nn=nn))
        end
    else
        @error "missing dataframe folder $(EnvConfig.logpath(settypesfilename()))"
    end
    return res
end

function inspect(coins)
    if isdflogfolder(settypesfilename())
        settypesdf = readdflogfolder(settypesfilename())
        println(EnvConfig.logpath(settypesfilename()))
        println(settypesdf[begin:begin+2, :])
        # println(describe(settypesdf))
        println("set types: $(unique(settypesdf[!, :settype]))")
        println("coins: $(unique(settypesdf[!, :coin]))")
        ok = true
        lbls = []
        for strow in eachrow(settypesdf)
            fdf = readdflogfolder(strow.featuresfname)
            if size(fdf, 1) == 0
                println("empty dataframe for $(EnvConfig.logpath(strow.featuresfname))")
                ok = false
            end
            tdf = readdflogfolder(strow.targetsfname)
            if size(tdf, 1) == 0
                println("empty dataframe for $(EnvConfig.logpath(strow.targetsfname))")
                ok = false
            else
                lbls2 = sort(unique(tdf[!, :targets]))
                if (length(lbls) > 0)
                    if lbls != lbls2
                        println("different labels: so far $lbls, $(strow.coin)/$(strow.settype) $lbls2")
                    end
                else
                    lbls = lbls2
                    println("labels: $(Tuple(lbls)), label strings: $(string.(collect(lbls)))")
                    println(tdf[begin, :])
                    println(fdf[begin, :])
                end
            end
        end
    else
        @error "missing features/targets dataframe folder $(EnvConfig.logpath(settypesfilename()))"
    end
    rangedf = readdflogfolder(rangefilename())
    println("ranges dataframe file: $(EnvConfig.logpath(rangefilename()))")
    println(rangedf[begin:begin+2, :])
end

function predictdf(df, nn, features, targets, settype)
    pred = Classify.predict(nn, features)
    pred = permutedims(pred, (2, 1))
    df2 = DataFrame(pred, string.(nn.labels))
    df2[:, "targets"] = CategoricalVector(string.(targets); levels=string.(nn.labels))
    # df2[:, "maxix"] = vec(mapslices(argmax, pred, dims=1))  # store column index of max estimate as column
    df2[:, :set] = CategoricalVector(fill(settype, size(df2, 1)); levels=settypes())
    df = isnothing(df) ? df2 : vcat(df, df2)
    return df
end

println("$PROGRAM_FILE ARGS=$ARGS")
testmode = "test" in ARGS
# testmode = true
inspectonly = "inspect" in ARGS
# inspectonly = true

EnvConfig.init(testmode ? test : training)
EnvConfig.setlogpath("2528-TrendDetector001-CollectSets-$(EnvConfig.configmode)")
println("log folder: $(EnvConfig.logfolder())")

verbosity = 3
if testmode 
    Ohlcv.verbosity = 3
    CryptoXch.verbosity = 3
    Features.verbosity = 3
    Targets.verbosity = 2
    EnvConfig.verbosity = 1
    Classify.verbosity = 3
    coins = ["SINE", "DOUBLESINE"]
else
    Ohlcv.verbosity = 1
    CryptoXch.verbosity = 1
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    coins = []
    for ohlcv in Ohlcv.OhlcvFiles()
        push!(coins, ohlcv.base)
    end
    println("length(coins)=$(length(coins))")
end

if inspectonly
    verbosity = 1
    Ohlcv.verbosity = 1
    CryptoXch.verbosity = 1
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    inspect(coins)
else
    featconfig = f6config01()
    trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01)
    println("featuresconfig=$(Features.describe(featconfig))")
    println("targetsconfig=$(Targets.describe(trgconfig))")
    if !isdflogfolder(settypesfilename())
        featurestargetscollect(coins, featconfig, trgconfig)
    else
        println("skipping featurestargetscollect - seems to be already done")
    end
    settypesdf = readdflogfolder(settypesfilename())
    nnvector = adaptclassifiers(coins, featconfig, trgconfig, settypesdf)
    if verbosity >= 3
        for nnt in nnvector
            println("$(nnt.coin) - $(nnt.nn.fileprefix): $(Classify.nnfilename(nnt.nn.fileprefix))")
            showlosses(nnt.nn)
        end
    end
end

# EnvConfig.setlogpath("2524-TrendDetector001-Sine")
# nn = sinetest()
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-20_23-03-26_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb")

# nn = BTCtest()

# EnvConfig.setlogpath("2445-TrendDetector001")
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-22_01-48-26_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # class balancing by undersampling - best generalization
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-21_00-27-30_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # class balancing by oversampling
# nn = Classify.loadnn("NNTrendDetector10pct30minutes_24-12-21_21-53-16_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # no class balancing
# evalclassifier(nn)

println("done")
end # of TrendDetector001
