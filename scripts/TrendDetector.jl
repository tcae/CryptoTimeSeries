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
module TrendDetector
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
    Features.addstd!(featcfg, window=5, offset=0, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=5, offset=0, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=5, offset=5, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=5, offset=10, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=15, offset=15, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=15, offset=30, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=15, offset=45, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=60, offset=60, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=60*4, offset=120, clip=nothing, norm=nothing)
    Features.addmaxdist!(featcfg, window=60, offset=0, clip=nothing, norm=nothing)
    Features.addmindist!(featcfg, window=60, offset=0, clip=nothing, norm=nothing)
    Features.addmaxdist!(featcfg, window=60*5, offset=60, clip=nothing, norm=nothing)
    Features.addmindist!(featcfg, window=60*5, offset=60, clip=nothing, norm=nothing)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0, clip=nothing, norm=nothing)
    return featcfg
end

function f6config02()
    error("don't use clipping due to worse results") # disable accidental use of clipping
    featcfg = Features.Features006()
    Features.addstd!(featcfg, window=5, offset=0, clip=1f0)
    Features.addgrad!(featcfg, window=5, offset=0, clip=1f0)
    Features.addgrad!(featcfg, window=5, offset=5, clip=1f0)
    Features.addgrad!(featcfg, window=5, offset=10, clip=1f0)
    Features.addgrad!(featcfg, window=15, offset=15, clip=1f0)
    Features.addgrad!(featcfg, window=15, offset=30, clip=1f0)
    Features.addgrad!(featcfg, window=15, offset=45, clip=1f0)
    Features.addgrad!(featcfg, window=60, offset=60, clip=1f0)
    Features.addgrad!(featcfg, window=60*4, offset=120, clip=1f0)
    Features.addmaxdist!(featcfg, window=60, offset=0, clip=1f0)
    Features.addmindist!(featcfg, window=60, offset=0, clip=1f0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60, clip=1f0)
    Features.addmindist!(featcfg, window=60*5, offset=60, clip=1f0)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0, clip=10f0)
    return featcfg
end

function f6config03()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=15, offset=5, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=60, offset=20, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=60*4, offset=80, clip=nothing, norm=nothing)
    Features.addmaxdist!(featcfg, window=60, offset=0, clip=nothing, norm=nothing)
    Features.addmindist!(featcfg, window=60, offset=0, clip=nothing, norm=nothing)
    Features.addmaxdist!(featcfg, window=60*5, offset=60, clip=nothing, norm=nothing)
    Features.addmindist!(featcfg, window=60*5, offset=60, clip=nothing, norm=nothing)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0, clip=nothing, norm=nothing)
    return featcfg
end

function f6config04()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=15, offset=0, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=60, offset=15, clip=nothing, norm=nothing)
    Features.addgrad!(featcfg, window=60*4, offset=75, clip=nothing, norm=nothing)
    Features.addmaxdist!(featcfg, window=60, offset=0, clip=nothing, norm=nothing)
    Features.addmindist!(featcfg, window=60, offset=0, clip=nothing, norm=nothing)
    Features.addmaxdist!(featcfg, window=60*5, offset=60, clip=nothing, norm=nothing)
    Features.addmindist!(featcfg, window=60*5, offset=60, clip=nothing, norm=nothing)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0, clip=nothing, norm=nothing)
    return featcfg
end

"""
returns targets
feature base has to be set before calling because that determines the ohlcv and relevant time range
"""
function calctargets!(trgcfg::Targets.AbstractTargets, featcfg::Features.AbstractFeatures)
    ohlcv = Features.ohlcv(featcfg)
    features = Features.features(featcfg)
    fot = Features.opentime(featcfg)
    (verbosity >= 4) && println("$(EnvConfig.now()) target calculation fromm $(fot[begin]) until $(fot[end])")
    Targets.setbase!(trgcfg, ohlcv)
    targets = Targets.labels(trgcfg, fot[begin], fot[end])
    Targets.labeldistribution(targets)
    targets = Targets.labelbinarytargets(trgcfg, longbuy, fot[begin], fot[end])
    targets = [lb ? longbuy : allclose for lb in targets]
    @assert size(features, 1) == length(targets) "size(features, 1)=$(size(features, 1)) != length(targets)=$(length(targets))"
    # (verbosity >= 3) && println(describe(trgcfg.df, :all))
    return targets
end

"Saves a given dataframe df in the current log folder using the given filename"
function savedflogfolder(df, filename)
    filepath = EnvConfig.logpath(filename)
    # try
        # EnvConfig.savebackup(filepath) # switched off until bug fixed
        JDF.savejdf(filepath, df)
        (verbosity >= 2) && println("$(EnvConfig.now()) saved dataframe to $(filepath)")
    # catch e
    #     Logging.@error "exception $e detected when writing $(filepath)"
    # end
end

"Reads and returns a dataframe from filename in the current log folder"
function readdflogfolder(filename)
    df = DataFrame()
    filepath = EnvConfig.logpath(filename)
    # try
        if isdir(filepath)
            (verbosity >= 4) && print("$(EnvConfig.now()) loading dataframe from  $(filepath)")
            df = DataFrame(JDF.loadjdf(filepath))
            (verbosity >= 4) && println("$(EnvConfig.now()) loaded $(size(df, 1)) rows successfully")
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(filepath)")
        end
    # catch e
    #     Logging.@error "exception $e detected"
    # end
    return df
end

isdflogfolder(filename) = isdir(EnvConfig.logpath(filename))

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
function featurestargetsliquidranges!(basecoin, featconfig, trgconfig; startdt=currentconfig().startdt, enddt=currentconfig().enddt, coinfilesdf = coinfilesdf, rangedf = rangedf, samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=Features.requiredminutes(featconfig), minpartitionsize=12*60, maxpartitionsize=2*24*60)
    featurestargetsdf = DataFrame()
    if size(coinfilesdf, 1) > 0
        cdf = @view coinfilesdf[coinfilesdf[!, :coin] .== basecoin,:]
        if (size(cdf, 1) > 0) && ismissing(cdf[begin, :featurestargetsfname])
            return featurestargetsdf
        end
    end
    (verbosity >= 3) && println("$(EnvConfig.now()) loading $basecoin for feature and target calculation") 
    ohlcv = Ohlcv.read(basecoin)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    rv = Ohlcv.liquiditycheck(ohlcv)
    ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    rangeid = size(rangedf, 1) > 0 ? maximum(rangedf[!, :rangeid]) + 1 : Int16(1)
    
    levels = unique(samplesets)
    push!(levels, "noop")
    samplesets = CategoricalArray(samplesets, levels=levels)

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
            (verbosity >= 4) && println("$basecoin length(psets)=$(length(psets)) rng=$rng") #  psets=$psets

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
        sort!(rangedf, [:coin, :ohlcvrange])
        savedflogfolder(rangedf, rangefilename())  # without period and gap columns added before

        featurestargetsfname = featurestargetsfilename(ohlcv.base)  # "features_targets_" * ohlcv.base * ".jdf"
        savedflogfolder(featurestargetsdf, featurestargetsfname)
        push!(coinfilesdf, (coin=ohlcv.base, featuresconfig=Features.describe(featconfig), targetsconfig=Targets.describe(trgconfig), featurestargetsfname=featurestargetsfname))
    else
        push!(coinfilesdf, (coin=ohlcv.base, featuresconfig=Features.describe(featconfig), targetsconfig=Targets.describe(trgconfig), featurestargetsfname=missing), promote=true)
        @info "skipping $basecoin due to no liquid ranges"
    end
    savedflogfolder(coinfilesdf, coinfilesdffilename())
    return featurestargetsdf
end

rangefilename() = "ranges.jdf"
coinfilesdffilename() = "settypesfiledict.jdf"
featurestargetsfilename(coin) = "features_targets_$coin.jdf"

trendccoinonfig(minwindow, maxwindow, buy, hold) = Targets.Trend(minwindow, maxwindow, Targets.thresholds((longbuy=buy, longhold=hold, shorthold=-hold, shortbuy=-buy)))

settypes() = ["train", "test", "eval"]

function getfeaturestargetsdf(coins, settype=nothing; featconfig=currentconfig().featconfig, trgconfig=currentconfig().trgconfig, startdt=currentconfig().startdt, enddt=currentconfig().enddt)
    ftdf = DataFrame()
    if isa(coins, AbstractVector) && (length(coins) > 0)
        for coin in coins
            ftdf = append!(ftdf, getfeaturestargetsdf(coin, settype))
        end
    else
        @assert isa(coins, AbstractString)
        coin = coins
        if isdflogfolder(featurestargetsfilename(coin))
            ftdffull = readdflogfolder(featurestargetsfilename(coin))
        else
            (verbosity >= 3) && println("calculating $coin features and targets")
            ftdffull = featurestargetsliquidranges!(coin, featconfig, trgconfig, startdt=startdt, enddt=enddt)
        end
        ftdf = isnothing(settype) ? ftdffull : @view ftdffull[(ftdffull[!, :set] .== settype), :]
    end
    return ftdf
end

function df2featurestargets(ftdf, settype=nothing)
    if size(ftdf, 1) > 0
        ftdf = isnothing(settype) ? ftdf : @view ftdf[(ftdf[!, :set] .== settype), :]
        features = @view ftdf[!, Not([:set, :rix, :rangeid, :targets])]
        targets = @view ftdf[!, :targets]
        features = Array(features)  # change from df to array
        features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
        (verbosity >= 3) && println("typeof(features)=$(typeof(features)), size(features)=$(size(features)), typeof(targets)=$(typeof(targets)), size(targets)=$(size(targets))") 
        return features, targets
    else
        return nothing, nothing
    end
end

function getfeaturestargets(coins, settype=nothing; featconfig=currentconfig().featconfig, trgconfig=currentconfig().trgconfig, startdt=currentconfig().startdt, enddt=currentconfig().enddt)
    if isa(coins, AbstractVector) && (length(coins) > 0)
        multifeatures = multitargets = nothing
        for coin in coins
            features, targets = getfeaturestargets(coin, settype)
            if !isnothing(features) && !isnothing(targets)
                multifeatures = isnothing(multifeatures) ? features : hcat(multifeatures, features)
                multitargets = isnothing(multitargets) ? targets : vcat(multitargets, targets)
                (verbosity >= 3) && println("multicoin: typeof(multifeatures)=$(typeof(multifeatures)), size(multifeatures)=$(size(multifeatures)), typeof(multitargets)=$(typeof(multitargets)), size(multitargets)=$(size(multitargets))") 
            end
        end
        return multifeatures, multitargets
    else
        @assert isa(coins, AbstractString) "typeof(coins)=$(typeof(coins)), isa(coins, AbstractString)=$(isa(coins, AbstractString))"
        coin = coins
        if isdflogfolder(featurestargetsfilename(coin))
            ftdffull = readdflogfolder(featurestargetsfilename(coin))
        else
            (verbosity >= 3) && println("calculating $coin features and targets")
            ftdffull = featurestargetsliquidranges!(coin, featconfig, trgconfig, startdt=startdt, enddt=enddt)
        end
        if size(ftdffull, 1) == 0
            return nothing, nothing
        end
        ftdf = ftdffull[(ftdffull[!, :set] .== settype), :]
        features = ftdf[!, Not([:set, :rix, :rangeid, :targets])]
        targets = ftdf[!, :targets]
        features = Array(features)  # change from df to array
        features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
        (verbosity >= 3) && println("$coin typeof(features)=$(typeof(features)), size(features)=$(size(features)), typeof(targets)=$(typeof(targets)), size(targets)=$(size(targets))") 
        return features, targets
    end
end

classifiermenmonic(coins, coinix) = "model001_" * (isnothing(coinix) ? "mix" : coins[coinix])
classifiermenmonic(coins) = "model001_" * ((isa(coins, AbstractVector) && (length(coins) > 0)) ? "mix" : coins)

function getlatestclassifier(coin, featconfig=currentconfig().featconfig, trgconfig=currentconfig().trgconfig, classifiermodel=currentconfig().classifiermodel)
    nn = nothing
    if isfile(Classify.nnfilename(classifiermenmonic(coin)))
        nn = Classify.loadnn(classifiermenmonic(coin))
        (verbosity >= 3) && println("getlatestclassifier loaded: nn=$(nn.fileprefix), labels=$(nn.labels)")
    else
        nn = classifiermodel(Features.featurecount(featconfig), [longbuy, allclose], classifiermenmonic(coin))
        (verbosity >= 3) && println("getlatestclassifier new: nn=$(nn.fileprefix), labels=$(nn.labels)")
    end
    @assert !isnothing(nn)
    return nn
end

# getlatestclassifier(coins, coinix, featureconfig, targetconfig) = getlatestclassifier((isnothing(coinix) ? coins : coins[coinix]), featureconfig, targetconfig)

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

function getclassifier(coins; featconfig=currentconfig().featconfig, trgconfig=currentconfig().trgconfig, classifiermix=currentconfig().classifiermix)
    coin = (isa(coins, AbstractVector) && (length(coins) > 0)) || (classifiermix == mixonly) ? "mix" : coins
    requirestraining = true
    nn = nothing
    if isfile(Classify.nnfilename(classifiermenmonic(coin)))
        nn = getlatestclassifier(coin, featconfig, trgconfig)
        (verbosity >= 3) && println("$coin classifier file loaded from $(Classify.nnfilename(classifiermenmonic(coin)))")
        requirestraining = !Classify.nnconverged(nn)
    else
        @assert ((coin == "mix") && isa(coins, AbstractVector)) || ((coin != "mix") && (classifiermix != mixonly)) "coin=$coin, typeof(coins)=$(typeof(coins)), classifiermix=$classifiermix"
        if (coin != "mix") 
            if (classifiermix in [mixonly, specificmix])
                nn = getlatestclassifier("mix", featconfig, trgconfig)
                @assert Classify.isadapted(nn) "missing adapted mix classifier, which is prerequisite for $coin"
                Classify.setmnemonic(nn, classifiermenmonic(coin))
            else # adapt coin specific classifier without mix baseline 
                nn = getlatestclassifier(coin, featconfig, trgconfig)
            end
        else # adapt mix classifier
            nn = getlatestclassifier(coin, featconfig, trgconfig)
        end
    end
    if requirestraining
        println("$(EnvConfig.now()) adapting $coin classifier")
        # if classifier file does not exist then create one
        # either a mix classifier is adapted using a vector of coins or a coin specific classifier is adapted - not possible to adapt a mix classifiers without providing multiple coins data
        features, targets = getfeaturestargets(coins, "train") # use coins, instead of coin, to get all features / targets in case of a mix classifier adaptation
        if isnothing(features) || isnothing(targets)
            return nothing
        end
        (verbosity >= 2) && println("before correction: $(Distributions.fit(UnivariateFinite, categorical(string.(targets)))))")
        (features), targets = oversample((features), targets)  # all classes are equally trained
        # (features), targets = undersample((features), targets)  # all classes are equally trained
        (verbosity >= 2) && println("after oversampling: $(Distributions.fit(UnivariateFinite, categorical(string.(targets)))))")
        Classify.adaptnn!(nn, features, targets)
        (verbosity >= 3) && showlosses(nn)
        if isnothing(nn)
            # no adaptation took place
            return nothing
        else
            # EnvConfig.savebackup(Classify.nnfilename(nn.fileprefix))
            Classify.savenn(nn)
        end
        println("$(EnvConfig.now()) finished adapting $coin classifier")
    end
    return nn
end

predictionsfilename(coins) = "predictions_$((isa(coins, AbstractVector) && (length(coins) > 0)) ? "mix" : coins).jdf"

"""
Returns the prediction for a single coin. classifiermix determines which classifier to use.
"""
function getpredictions(coin; classifiermix=currentconfig().classifiermix)
    dfp = DataFrame()
    # coin = (isa(coins, AbstractVector) && (length(coins) > 0)) ? "mix" : coins
    if isdflogfolder(predictionsfilename(coin))
        dfp = readdflogfolder(predictionsfilename(coin))
    else
        nn = getclassifier(coin, classifiermix=classifiermix)
        if isnothing(nn)
            (verbosity >= 2) && println("No classifier found for $coin - no predictions can be calculated")
        else
            ftdf = getfeaturestargetsdf(coin) # calc only coin specific predictions, otherwise the match to ohlcv is no longer possible
            features, targets = df2featurestargets(ftdf)
            pred = Classify.predict(nn, features)
            pred = permutedims(pred, (2, 1))
            dfp = DataFrame(pred, string.(nn.labels))
            dfp[!, :targets] = CategoricalVector(string.(targets); levels=string.(nn.labels))  # equals ftdf[!, :targets] but those are TradeLabels and no CategoryVector
            dfp[!, :set] = ftdf[!, :set]
            if size(dfp, 1) > 0 
                savedflogfolder(dfp, predictionsfilename(coin))
            else
                (verbosity >= 2) && println("No $coin predictions could be calculated - no result stored")
            end
        end
    end
    return dfp
end

function inspect(coins)
    TrendDetector.verbosity = 2
    Ohlcv.verbosity = 1
    CryptoXch.verbosity = 1
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    if isdflogfolder(coinfilesdffilename())
        coinfilesdf = readdflogfolder(coinfilesdffilename())
        println(EnvConfig.logpath(coinfilesdffilename()))
        # println(coinfilesdf)
        # println("$(length(setdiff(coins, coinfilesdf[!, :coin]))) unconsidered coins that have no features/targets (probably due to low liquidity): $(setdiff(coins, coinfilesdf[!, :coin]))")
        # println("$(length(setdiff(coinfilesdf[!, :coin], coins)))) coins with features/targets but are missing in the requested set of coins: $(setdiff(coinfilesdf[!, :coin], coins))")
        # coins = intersect(coins, coinfilesdf[!, :coin])
        # println(coinfilesdf[begin:begin+2, :])
        # println(describe(coinfilesdf))
        println("$(length(coins)) processable coins")
        ok = true
        lbls = []
        for strow in eachrow(coinfilesdf)
            if ismissing(strow.featurestargetsfname)
                continue
            end
            ftdf = readdflogfolder(strow.featurestargetsfname)
            if size(ftdf, 1) == 0
                println("empty dataframe for $(EnvConfig.logpath(strow.featurestargetsfname))")
                ok = false
            else
                lbls2 = sort(unique(ftdf[!, :targets]))
                if (length(lbls) > 0)
                    if lbls != lbls2
                        println("different labels: so far $lbls, $(strow.coin)/$(strow.settype) $lbls2")
                    end
                else
                    lbls = lbls2
                    println("labels: $(Tuple(lbls)), label strings: $(string.(collect(lbls)))")
                    println(ftdf[begin, :])
                end
            end
        end
        ftdf = getfeaturestargetsdf(coins) # use coins, instead of coin, to get all features / targets in case of a mix classifier adaptation
        if size(ftdf, 1) > 0
            println("describe(ftdf, :all)=$(describe(ftdf, :all))")
        else
            println("No features and targets found")
        end

        allfilenames = readdir(EnvConfig.logfolder())
        predictionfilenames = filter(filename -> contains(filename, "predictions"), allfilenames)
        if length(predictionfilenames) > 0
            # println("prediction filenames: $predictionfilenames")
            println(predictionfilenames[begin])
            df = readdflogfolder(predictionfilenames[begin])
            println("predictions size=$(size(df)): \n$(df[begin:begin+1, :])\n$(describe(df))")
            println("target distribution: $(Distributions.fit(UnivariateFinite, categorical(string.(df[!, :targets]))))")
            println("set distribution: $(Distributions.fit(UnivariateFinite, categorical(string.(df[!, :set]))))")
        else
            println("no prediction dataframe files in $(EnvConfig.logfolder())")
        end
        println()
    else
        @error "missing coin files dataframe folder $(EnvConfig.logpath(coinfilesdffilename()))"
    end
    rangedf = readdflogfolder(rangefilename())
    println("ranges dataframe file: $(EnvConfig.logpath(rangefilename()))")
    # println(rangedf)
    println(rangedf[[(begin:begin+2)...,(end-1:end)...], :])
    println("\ndescribe(rangedf) $(describe(rangedf))")
end

function _oixdelta(ix, ftdf, rangeid, rangedelta, rangedf, ohlcvdf)
    if ftdf[ix, :rangeid] != rangeid
        rangeid = ftdf[ix, :rangeid]
        rangerow = @view rangedf[rangedf[!, :rangeid] .== rangeid, :]
        @assert size(rangerow, 1) == 1 "size(rangerow, 1)=$(size(rangerow, 1)) != 1, rangerow=$rangerow"
        rangerow = rangerow[begin, :]

        if ohlcvdf[rangerow.liquidrange[begin] + rangerow.dfrange[begin] - 1, :opentime] != rangerow.startdt
            (verbosity >= 3) && println("WARNING: unexpected mismatch of startdt: ohlcvdf[rangerow.liquidrange[begin]=$(rangerow.liquidrange[begin]) + rangerow.dfrange[begin]=$( + rangerow.dfrange[begin]) - 1, :opentime]=$(ohlcvdf[rangerow.liquidrange[begin] + rangerow.dfrange[begin] - 1, :opentime]) != rangerow.startdt=$(rangerow.startdt)") 
        #     ix = Ohlcv.rowix(ohlcvdf[!, :opentime], rangedt)
        # else
        #     ix = rangeix
        end

        rangedelta = rangerow.liquidrange[begin] - 1
    end
    oix = ftdf[ix, :rix] + rangedelta
    return oix, rangedelta, rangeid
end

"collects gains of a prediction vector of a specific coin"
function _getgainsdf(scores, ftdf, rangedf, ohlcvdf, openthreshold, closethreshold, inlabel, outlabel)
    gdf = DataFrame() # set=String[], label=String[], samplecount=Int32[], gain=Float32[], startdt=DateTime[], enddt=DateTime[], ftstartix=Int32[], ftendix=Int32[])
    if length(scores) == 0
        return gdf
    end
    startix = startprice = starttime = labelix = nothing
    lastlabel = outlabel
    # delta = ftix2ohlcvixoffset(ftdf[begin, :rix], ftdf, rangedf, ohlcvdf)
    # if delta != 0
    #     ftdf[!, :rix] .= ftdf[!, :rix] .+ delta
    # end
    rangeid = nothing
    rangedelta = nothing
    for ix in eachindex(scores)
        labelix = lastlabel == inlabel ? (scores[ix] > closethreshold ? inlabel : outlabel) : (scores[ix] > openthreshold ? inlabel : outlabel)
        if labelix != lastlabel
            oix, rangedelta, rangeid = _oixdelta(ix, ftdf, rangeid, rangedelta, rangedf, ohlcvdf)
            ixprice = ohlcvdf[oix, :close]
            ixtime = ohlcvdf[oix, :opentime]
            # lastlabel is the correct label to use for the segment because labelix is the first label seen after the label changed
            if lastlabel == inlabel # only add inlabel gains
                gain = (ixprice - startprice) / startprice
                push!(gdf, (set=ftdf[ix, :set], label=lastlabel, samplecount=Minute(ixtime-starttime).value + 1, gain=gain, startdt=starttime, enddt=ixtime, ftstartix=startix, ftendix=ix, openthreshold=openthreshold, closethreshold=closethreshold))
            end
            startix = ix
            startprice = ixprice
            starttime = ixtime
            lastlabel = labelix
        end
    end
    return gdf
end

gainsfilename() = "gains.jdf"

function getgainsdf(coins; rangedf = rangedf)
    gaindf = DataFrame()
    # coin = (isa(coins, AbstractVector) && (length(coins) > 0)) ? "mix" : coins
    if isdflogfolder(EnvConfig.logpath(gainsfilename()))
        gaindf = readdflogfolder(EnvConfig.logpath(gainsfilename()))
        if size(gaindf, 1) > 0
            return gaindf
        end
    end
    coins = isa(coins, AbstractVector) ? coins : [coins]
    for coin in coins
        dfp = getpredictions(coin)
        if (size(dfp, 1) > 0)
            for (openthreshold, closethreshold) in [(0.8, 0.5), (0.7, 0.5), (0.6, 0.5), (0.8, 0.6), (0.7, 0.6), (0.6, 0.55)] # [(0.5, 0.5)]
                ohlcv = Ohlcv.read(coin)
                ohlcvdf = Ohlcv.dataframe(ohlcv)
                # rangedf = readdflogfolder(rangefilename())
                ftdf = getfeaturestargetsdf(coin)
                diff = dfp[!, :set] .!= ftdf[!, :set]
                diffdf = DataFrame(dfp=dfp[diff, :set], ftdf=ftdf[diff, :set])
                @assert dfp[!, :set] == ftdf[!, :set] "dfp[!, :set]=$(length(dfp[!, :set])) != ftdf[!, :set]=$(length(ftdf[!, :set])) diff=$diffdf"
                prednames = CategoricalArray(Classify.predictioncolumns(dfp), ordered=false)
                # assuming binary classification with 2 classes and prednames[1] is the open trade class (longbuy or shortbuy)
                @assert string(prednames[2]) == "allclose" "prednames=$(prednames)"
                @assert length(prednames) == 2 "prednames=$(prednames)"

                scores = dfp[!, string(prednames[1])]
                # predonly = @view dfp[!, string.(prednames)]
                # scores, maxindex = Classify.maxpredictions(Matrix(predonly), 2)
                # predicted = vec([prednames[ix] for ix in maxindex])
                gdf = _getgainsdf(scores, ftdf, rangedf, ohlcvdf, openthreshold, closethreshold, prednames[1], prednames[2])
                gdf[!, :coin] = fill(coin, size(gdf, 1))
                gdf[!, :predicted] = fill(true, size(gdf, 1))
                gaindf = append!(gaindf, gdf)

                gdf = _getgainsdf([(t == prednames[1] ? 1f0 : 0f0) for t in dfp[!, :targets]], ftdf, rangedf, ohlcvdf, openthreshold, closethreshold, prednames[1], prednames[2])
                gdf[!, :coin] = fill(coin, size(gdf, 1))
                gdf[!, :predicted] = fill(false, size(gdf, 1))
                dfp = @view dfp[dfp[!, :set] .!= "noop", :] # exclude gaps between set partitions
                gaindf = append!(gaindf, gdf)
            end
        else
            (verbosity >= 1) && println("skipping gain collection of $(coin) due to missing predictions due to size(dfp)= $(size(dfp))")
        end
    end
    if size(gaindf, 1) > 0
        sort!(gaindf, [:coin, :predicted, :openthreshold, :closethreshold, :startdt])
        savedflogfolder(gaindf, EnvConfig.logpath(gainsfilename()))
    end
    return gaindf
end

distancesfilename() = "distances.jdf"

"""
Provides distance information between neighboring predicted gain segments in form of one data frame row for each precited positive segment with the following columns:  
- :coin and :set as taken over from gains
- :label is the predicted label
- :tpdistnext = in case the predicted segment does overlap with a labeled segment (= true positive) and the next segment is also a true positive segment of the same labeled segment, distance to the next true positive predicted segment
- :fpdistnext = in case the predicted segment does not overlap with a labeled segment (= false positive), distance to the next predicted segment
- :distfirst = in case of the first true positive predicted segment of a labeled segment, distance to the beginning of the labeled  sgement to see how long the prediction needs to detect a true positive situation
- :distlast - in case of the last true positive predcited segment, distance to the end of the labelled segment
- :startdt, :enddt provide the timestamps of the predicted segment
- :truestartdt, :trueenddt provide the timestamps of the labelled segment
"""
function getdistances(coins; rangedf = rangedf)

    function getdist!(distdf, cprow, cpnextrow, ctrow)
        distnext = isnothing(cpnextrow) ? missing : Minute(cpnextrow.startdt - cprow.enddt).value
        @assert isnothing(cpnextrow) || (cprow.enddt < cpnextrow.startdt) "cprow=$cprow \ncpnextrow=$cpnextrow \nctrow=$ctrow"
        distrow = (coin=cprow.coin, set=cprow.set, label=cprow.label, tpdistnext=missing, fpdistnext=missing, distfirst=missing, distlast=missing, startdt=cprow.startdt, enddt=cprow.enddt, truestartdt=missing, trueenddt=missing)
        if isnothing(ctrow) # there is no labeled segement to match with - hence, the predicetd segment is false positive
            distrow = (distrow..., fpdistnext=distnext)
        else
            distrow = (distrow..., truestartdt=ctrow.startdt, trueenddt=ctrow.enddt)
            if cprow.enddt < ctrow.startdt # predicted segment is completely before true segment - hence, the predicted segment is false positive
                distrow = (distrow..., fpdistnext=distnext)
            else
                lastdistrow = size(distdf, 1) > 1 ? distdf[end, :] : nothing
                distfirst = isnothing(lastdistrow) || (lastdistrow.enddt < ctrow.startdt) ? Minute(cprow.startdt - ctrow.startdt).value : missing
                distlast = isnothing(cpnextrow) || (ctrow.enddt < cpnextrow.startdt) ? Minute(cprow.enddt - ctrow.enddt).value : missing
                if !isnothing(cpnextrow) && (ctrow.enddt < cpnextrow.startdt)
                    distrow = (distrow..., fpdistnext=distnext, distfirst=distfirst, distlast=distlast)
                else
                    distrow = (distrow..., tpdistnext=distnext, distfirst=distfirst, distlast=distlast)
                end
            end
        end
        # (verbosity >= 3) && println("distrow=$distrow, cprow=$cprow, cpnextrow=$cpnextrow, ctrow=$ctrow")
        push!(distdf, distrow, promote=true)
    end

    distdf = DataFrame()
    if isdflogfolder(EnvConfig.logpath(distancesfilename()))
        distdf = readdflogfolder(EnvConfig.logpath(distancesfilename()))
        if size(distdf, 1) > 0
            return distdf
        end
    end
    gaindf = getgainsdf(coins; rangedf = rangedf)
    if (size(gaindf, 1) > 0)
        gaindf = @view gaindf[gaindf[!, :openthreshold] .== minimum(gaindf[!, :openthreshold]), :]
        gaindf = @view gaindf[gaindf[!, :closethreshold] .== minimum(gaindf[!, :closethreshold]), :]
        gaindfgrp = groupby(gaindf, [:coin, :predicted])
        coins = isa(coins, AbstractVector) ? coins : [coins]
        for coin in coins
            cpgaindf = get(gaindfgrp, (coin, true), DataFrame())
            ctgaindf = get(gaindfgrp, (coin, false), DataFrame())
            if size(ctgaindf, 1) > 0
                @assert issorted(ctgaindf[!, :startdt])
                ctix = firstindex(ctgaindf, 1)
            else
                ctix = nothing
            end
            if size(cpgaindf, 1) > 0
                @assert issorted(cpgaindf[!, :startdt])
                for cpix in 1:nrow(cpgaindf)
                    cpnix = cpix < lastindex(cpgaindf, 1) ? cpix + 1 : nothing
                    while !isnothing(ctix) && (cpgaindf[cpix, :startdt] > ctgaindf[ctix, :enddt])
                        ctix = ctix < lastindex(ctgaindf, 1) ? ctix + 1 : nothing
                    end
                    getdist!(distdf, cpgaindf[cpix, :], (isnothing(cpnix) ? nothing : cpgaindf[cpnix, :]), (isnothing(ctix) ? nothing : ctgaindf[ctix, :]))
                end
            else
                (verbosity >= 1) && println("skipping distances collection of $(coin) due to missing gain predictions due to size(cpgaindf)= $(size(cpgaindf))")
                # no row if no gain record - even with a truth record 
            end
        end
    else
        (verbosity >= 1) && println("skipping distances collection of $(coin) due to missing gains due to size(gaindf)= $(size(gaindf))")
    end
    if size(distdf, 1) > 0
        savedflogfolder(distdf, EnvConfig.logpath(distancesfilename()))
    end
    return distdf
end

confusionfilename() = "confusion.jdf"
xconfusionfilename() = "xconfusion.jdf"

function getconfusionmatrices(coins)

    function _getconfusionmatrices!(cmdf, xcmdf, coins, coin)
        dfp = getpredictions(coins)
        dfp = @view dfp[dfp[!, :set] .!= "noop", :] # exclude gaps between set partitions
        if (size(dfp, 1) > 0)
            _cmdf = Classify.confusionmatrix(dfp)
            insertcols!(_cmdf, 1, :coin => fill(coin, size(_cmdf, 1)))
            cmdf = isnothing(cmdf) ? _cmdf : append!(cmdf, _cmdf)
            (verbosity >= 4) && println("$coin cm: $_cmdf")

            _xcmdf = Classify.extendedconfusionmatrix(dfp)
            insertcols!(_xcmdf, 1, :coin => fill(coin, size(_xcmdf, 1)))
            xcmdf = isnothing(xcmdf) ? _xcmdf : append!(xcmdf, _xcmdf)
            (verbosity >= 4) && println("$coin xcm: $_xcmdf")
            # gaindf = getgainsdf(coins)
        else
            (verbosity >= 1) && println("skipping evaluation of $(coins) due to missing predictions (size(dfp)= $(size(dfp)))")
        end
    end

    coin = (isa(coins, AbstractVector) && (length(coins) > 0)) ? "mix" : coins
    xcmdf = DataFrame()
    cmdf = DataFrame()
    if isdflogfolder(EnvConfig.logpath(confusionfilename()))
        cmdf = readdflogfolder(EnvConfig.logpath(confusionfilename()))
    end
    if isdflogfolder(EnvConfig.logpath(xconfusionfilename()))
        xcmdf = readdflogfolder(EnvConfig.logpath(xconfusionfilename()))
    end
    if !isnothing(cmdf) && !isnothing(xcmdf) && (size(cmdf, 1) > 0) && (size(xcmdf, 1) > 0)
        return cmdf, xcmdf
    end
    # _getconfusionmatrices!(cmdf, xcmdf, coins, coin) # mix confusion matrix
    for coin in coins
        _getconfusionmatrices!(cmdf, xcmdf, coin, coin)
    end
    if size(cmdf, 1) > 0
        savedflogfolder(cmdf, confusionfilename())
    end
    if size(xcmdf, 1) > 0
        savedflogfolder(xcmdf, xconfusionfilename())
    end
    return cmdf, xcmdf
end

function averageconfusionmatrix(coins)
    cmdf, xcmdf = getconfusionmatrices(coins)
    if size(cmdf, 1) > 0
        # cmdfgrp = groupby(cmdf, [:coin, :set, :prediction])
        cmdf = @view cmdf[cmdf[!, :set] .!= "noop", :] # exclude gaps between set partitions
        cmdfgrp = groupby(cmdf, [:set, :prediction])
        ccmdf = combine(cmdfgrp, [:truth_longbuy, :truth_allclose] => ((lb, ac) -> sum(lb) / (sum(lb) + sum(ac)) * 100) => "longbuy_ppv%")
    else
        (verbosity >= 2) && println("cannot get confusion matrices")
        ccmdf = DataFrame()
    end
    if size(xcmdf, 1) > 0
        # cmdfgrp = groupby(xcmdf, [:coin, :set, :prediction])
        xcmdf = @view xcmdf[xcmdf[!, :set] .!= "noop", :] # exclude gaps between set partitions
        # println("DEBUG xcmdf=$(xcmdf[1:100, :])")
        xcmdfgrp = groupby(xcmdf, [:set, :pred_label, :bin])
        cxcmdf = combine(xcmdfgrp, [:tp, :fp] => ((tp, fp) -> sum(tp) / (sum(tp) + sum(fp)) * 100) => "ppv%")
    else
        (verbosity >= 2) && println("cannot get confusion matrices")
        cxcmdf = DataFrame()
    end
    return ccmdf, cxcmdf
end

function safe(f, v; default=missing)
    v = skipmissing(v)
    isempty(v) ? default : f(v)
end
"""
mixonly = only one mix classifier that is used for any coin  
specificonly = only a dedicated classifier per coin  
specificmix = common mix classifier with a coin specific adaptation on top  
"""
@enum ClassifierMix mixonly specificmix specificonly
# startdt = nothing  # means use all what is stored as canned data
# enddt = nothing  # means use all what is stored as canned data
startdt = DateTime("2017-11-17T20:56:00")
enddt = DateTime("2025-08-10T15:00:00")

"""
mk1 = mix adapted with multiple adaptations per coin with **good results**: ppv(longbuy) = 72%
"""
mk1config() = (folder="2528-TrendDetector001-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=specificmix, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

"""
mk2 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk2config() = (folder="2533-TrendDetector002-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=specificmix, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

"""
mk3 = one iteration adaptation with one merged set (allclose=>0.9, longbuy=>0.1), but target config trendccoinonfig(10, 4*60, 0.05, 0.03) with **poor results**: ppv(longbuy) = 26%
"""
mk3config() = (folder="2534-TrendDetector003-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.05, 0.03), classifiermix=specificmix, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

"""
mk4 = trendccoinonfig(10, 4*60, 0.02, 0.01) with one merged set (allclose=>0.726, longbuy=>0.274)
"""
mk4config() = (folder="2534-TrendDetector004-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.02, 0.01), classifiermix=specificmix, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

"""
mk5 = trendccoinonfig(10, 4*60, 0.007, 0.005) with one merged set (allclose=>?, longbuy=>?)
"""
mk5config() = (folder="2535-TrendDetector005-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.007, 0.005), classifiermix=specificmix, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

"""
mk6 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk6config() = (folder="2534-TrendDetector006-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

"""
same as mk6 butwith copied mix classifier from mk2
mk7 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk7config() = (folder="2534-TrendDetector007-mix-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

"""
mk8 = mix adapted with all coin features/targets in one set, features are clipped, normalized, shifted, and in addition batch norm layer after initial layer with relu activation in model001
equal mean, q25, q75, min, max does not look like healthy feature values - longbuy ppv classification performance is with close to 70% also worse
"""
mk8config() = (folder="2535-TrendDetector008-mixonly-clipnormshift-$(EnvConfig.configmode)", featconfig = f6config02(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model001, startdt=startdt, enddt=enddt)

""" **my favorite**  
mk9 = mix adapted with all coin features/targets in one set, features are not clipped, batch norm layer before and between layers with relu activation in model002
"""
mk9config() = (folder="2535-TrendDetector009-mixonlymodel002noclip-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model002, startdt=startdt, enddt=enddt)

"""
mk10 = mix adapted with all coin features/targets in one set, features are clipped, initial batch norm layer in model002
"""
mk10config() = (folder="2535-TrendDetector010-mixonlymodel002clip-$(EnvConfig.configmode)", featconfig = f6config02(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model002, startdt=startdt, enddt=enddt)

"""
mk11 = mix adapted with all coin features/targets in one set, no clipping, initial batch norm layer but no further internal batch norm layers
"""
mk11config() = (folder="2535-TrendDetector011-model003-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model003, startdt=startdt, enddt=enddt)

"""
mk12 = like mk9 but with an additional layer
"""
mk12config() = (folder="2535-TrendDetector012-model004-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model004, startdt=startdt, enddt=enddt)

"""
mk13 = like mk9 but with 4/3 broader layers
"""
mk13config() = (folder="2535-TrendDetector013-model005-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model005, startdt=startdt, enddt=enddt)

"""
mk14 = like mk11 but removed layer 3
"""
mk14config() = (folder="2535-TrendDetector014-model006-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model006, startdt=startdt, enddt=enddt)

"""
mk15 = like mk11 but reduced number of nodes of layers by reducing factor of layer 1 from 3 to 2
"""
mk15config() = (folder="2535-TrendDetector015-model007-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model007, startdt=startdt, enddt=enddt)

"""
mk16 = no tolerance against target disturbances (minwindow=0), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk16config() = (folder="2537-TrendDetector016-notargetminwindow-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(0, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model002, startdt=startdt, enddt=enddt)

"""
mk17 = short tolerance against target disturbances (minwindow=2), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk17config() = (folder="2537-TrendDetector017-shorttargetminwindow-$(EnvConfig.configmode)", featconfig = f6config01(), trgconfig = trendccoinonfig(2, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model002, startdt=startdt, enddt=enddt)

"""  
mk18 = mk9 but with simplified features
"""
mk18config() = (folder="2538-TrendDetector018-mixonlymodel002noclip-$(EnvConfig.configmode)", featconfig = f6config03(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model002, startdt=startdt, enddt=enddt)

"""  
mk19 = mk9 but with simplified features
"""
mk19config() = (folder="2538-TrendDetector019-mixonlymodel002noclip-$(EnvConfig.configmode)", featconfig = f6config04(), trgconfig = trendccoinonfig(10, 4*60, 0.01, 0.01), classifiermix=mixonly, classifiermodel=Classify.model002, startdt=startdt, enddt=enddt)

currentconfig() = mk19config()

println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$ARGS")
testmode = "test" in ARGS
# testmode = true
inspectonly = "inspect" in ARGS
specialonly = "special" in ARGS
# inspectonly = true
# specialonly = true

EnvConfig.init(testmode ? test : training)
#* folder1 and folder2 are used to compare folder content
#* folder is used to process the content of just a single folder
folder2, _, _ = mk1config() # mk1 = mix adapted with multiple adaptations per coin
folder1, _, _ = mk2config() # mk2 = mix adapted in just one iteration with all coin features/targets in one set (allclose=>0.494, longbuy=>0.506)

# println(currentconfig())
folder, featconfig, trgconfig = currentconfig()
EnvConfig.setlogpath(folder)
println("log folder: $(EnvConfig.logfolder())")

verbosity = 4
if testmode 
    Ohlcv.verbosity = 3
    CryptoXch.verbosity = 3
    Features.verbosity = 3
    Targets.verbosity = 2
    EnvConfig.verbosity = 1
    Classify.verbosity = 3
    coins = ["SINE", "DOUBLESINE"]
else # training or production
    verbosity = 2
    Ohlcv.verbosity = 1
    CryptoXch.verbosity = 1
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    # coins = []
    # for ohlcv in Ohlcv.OhlcvFiles()
    #     push!(coins, ohlcv.base)
    # end
    # println("length(coins)=$(length(coins))")
    # println(coins)

    # long version 
    # coins = ["1INCH", "5IRE", "A8", "AAVE", "ACA", "ACH", "ACS", "ADA", "AEG", "AERO", "AEVO", "AGIX", "AGI", "AGLA", "AGLD", "AI16Z", "AIOZ", "AIXBT", "AKI", "ALCH", "ALGO", "ALT", "AMI", "ANIME", "ANKR", "AO", "APEX", "APE", "APP", "APRS", "APT", "ARB", "ARKM", "AR", "ASRR", "ATH", "ATOM", "AVAIL", "AVAX", "AVA", "AVL", "AXL", "AXS", "A", "B3", "BABYDOGE", "BAN", "BBL", "BBQ", "BB", "BCH", "BCUT", "BDXN", "BEAM", "BEL", "BERA", "BICO", "BLAST", "BLOCK", "BLUR", "BMT", "BNB", "BOBA", "BOB", "BOMB", "BOME", "BONK", "BRETT", "BR", "BTC", "BUBBLE", "C98", "CAKE", "CARV", "CATBNB", "CATI", "CELO", "CEL", "CGPT", "CHILLGUY", "CHRP", "CHZ", "CLOUD", "CMETH", "COA", "COMP", "COM", "COOKIE", "COOK", "COQ", "CORE", "COT", "CPOOL", "CRV", "CSPR", "CTA", "CTC", "CTT", "CUDIS", "CYBER", "DBR", "DECHAT", "DEEP", "DEFI", "DEGEN", "DGB", "DIAM", "DMAIL", "DOGE", "DOGS", "DOLO", "DOP1", "DOT", "DRIFT", "DSRUN", "DUEL", "DYDX", "DYM", "EGLD", "EGO", "EIGEN", "ELDE", "ELX", "ENA", "ENJ", "ENS", "EOS", "EPT", "ERA", "ESE", "ES", "ETC", "ETHFI", "ETHW", "ETH", "EVERY", "EXVG", "FAR", "FB", "FET", "FHE", "FIDA", "FIL", "FIRE", "FITFI", "FLIP", "FLOCK", "FLOKI", "FLOW", "FLR", "FLT", "FLUID", "FMB", "FON", "FORT", "FOXY", "FRAG", "FTM", "FTT", "FUEL", "FXS", "F", "G3", "G7", "GALAXIS", "GALA", "GAL", "GAME", "GLMR", "GMT", "GMX", "GOAT", "GPS", "GPT", "GRAPE", "GRASS", "GRT", "GSTS", "GST", "GTAI", "HAEDAL", "HBAR", "HFT", "HLG", "HMSTR", "HNT", "HOME", "HOOK", "HOT", "HTX", "HUMA", "HVH", "HYPER", "HYPE", "H", "ICNT", "ICP", "ID", "IMX", "INIT", "INJ", "INSP", "IO", "IP", "IRL", "IZI", "JASMY", "JTO", "JUP", "J", "KAIA", "KAS", "KAVA", "KCAL", "KDA", "KLAY", "KMNO", "KSM", "L3", "LADYS", "LAI", "LAYER", "LA", "LDO", "LENDS", "LEVER", "LFT", "LGX", "LINK", "LL", "LMWR", "LOOKS", "LRC", "LTC", "LUNAI", "LUNA", "LUNC", "MAGIC", "MAJOR", "MANA", "MANTA", "MASA", "MASK", "MATIC", "MAVIA", "MBOX", "MEMEFI", "MEME", "MERL", "METH", "MEW", "ME", "MILK", "MINA", "MINU", "MIX", "MKR", "MLK", "MNT", "MOCA", "MOG", "MOJO", "MON", "MORPHO", "MOVE", "MOVR", "MOZ", "MPLX", "MVL", "MYRIA", "MYRO", "NAKA", "NEAR", "NEIRO", "NEON", "NEWT", "NEXT", "NGL", "NIBI", "NLK", "NOT", "NS", "NUTS", "NXPC", "NYAN", "OBOL", "ODOS", "OKG", "OL", "OMG", "OMNI", "OM", "ONDO", "ONE", "OP", "ORDER", "ORDI", "ORT", "PAAL", "PARTI", "PENDLE", "PENGU", "PEOPLE", "PEPE", "PERP", "PFVS", "PLANET", "PLAY", "PLUME", "PNUT", "POKT", "POL", "PONKE", "POPCAT", "PORT3", "PORTAL", "PPT", "PRCL", "PRIME", "PUFFER", "PUFF", "PUMP", "PURSE", "PYTH", "PYUSD", "QNT", "QORPO", "QTUM", "RACA", "RAIN", "RATS", "RDNT", "RED", "RENDER", "RESOLV", "RNDR", "ROAM", "ROOT", "ROSE", "RPK", "RSS3", "RUNE", "RVN", "SAFE", "SALD", "SAND", "SATS", "SCA", "SCRT", "SCR", "SC", "SEI", "SEND", "SEOR", "SERAPH", "SFUND", "SHARK", "SHIB", "SHILL", "SHRAP", "SIDUS", "SIGN", "SIS", "SKATE", "SLG", "SMILE", "SNX", "SOLO", "SOLV", "SOL", "SONIC", "SOSO", "SPEC", "SPELL", "SPK", "SPX", "SQD", "SQR", "SQT", "SSV", "STAR", "STETH", "STG", "STOP", "STREAM", "STRK", "STX", "SUI", "SUNDOG", "SUN", "SUPRA", "SUSHI", "SVL", "SWEAT", "SWELL", "SXT", "S", "TAC", "TAIKO", "TAI", "TAP", "TAVA", "TA", "TENET", "THETA", "THRUST", "TIA", "TNSR", "TOKEN", "TOMI", "TON", "TOSHI", "TOWNS", "TREE", "TRUMP", "TRX", "TWT", "T", "ULTI", "UMA", "UNI", "USTC", "UXLINK", "VANA", "VANRY", "VELAR", "VELO", "VENOM", "VET", "VEXT", "VIRTUAL", "VRA", "VRTX", "VV", "WAL", "WAVES", "WAXP", "WBTC", "WCT", "WELL", "WEMIX", "WEN", "WIF", "WLD", "WLKN", "WOO", "WWY", "W", "XAI", "XAR", "XAUT", "XAVA", "XCAD", "XDC", "XEC", "XION", "XLM", "XO", "XRP3L", "XRP", "XTER", "XTZ", "XWG", "X", "YFI", "ZEND", "ZENT", "ZEN", "ZEREBRO", "ZERO", "ZETA", "ZEX", "ZIG", "ZIL", "ZKF", "ZKJ", "ZKL", "ZK", "ZORA", "ZRC", "ZRO", "ZRX", "ZTX"]

    # test
    # coins =["BTC"]

    # liquid coins
    coins = ["1INCH", "AAVE", "ACH", "ADA", "AI16Z", "ALGO", "ANKR", "APEX", "APE", "APT", "ARB", "AR", "ATOM", "AVAX", "AXS", "BCH", "BNB", "BONK", "BRETT", "BTC", "C98", "CAKE", "CARV", "CELO", "CHILLGUY", "CHZ", "COMP", "CRV", "CTC", "DEEP", "DEGEN", "DGB", "DOGE", "DOGS", "DOT", "DRIFT", "DYDX", "EGLD", "ELX", "ENA", "ENJ", "ENS", "EOS", "ETC", "ETH", "FET", "FIL", "FIRE", "FLOCK", "FLOKI", "FLOW", "FTM", "FTT", "FXS", "GALA", "GAL", "GLMR", "GMT", "GOAT", "GPS", "GRASS", "GRT", "HBAR", "HFT", "HNT", "HOOK", "HOT", "H", "ICP", "ID", "IMX", "INIT", "INJ", "IP", "JASMY", "JUP", "KAS", "KAVA", "KLAY", "KSM", "LDO", "LINK", "LRC", "LTC", "LUNA", "LUNC", "MAGIC", "MANA", "MASK", "MATIC", "MAVIA", "MBOX", "MERL", "MINA", "MKR", "MNT", "MOVE", "MYRO", "NAKA", "NEAR", "NOT", "NXPC", "OMG", "ONDO", "ONE", "OP", "PENGU", "PEOPLE", "PEPE", "PLANET", "PLUME", "POL", "POPCAT", "PUFFER", "PUMP", "PYTH", "QNT", "QTUM", "RDNT", "RENDER", "RNDR", "ROSE", "RUNE", "RVN", "SAND", "SC", "SEI", "SERAPH", "SHIB", "SNX", "SOL", "SPX", "STETH", "STG", "STRK", "STX", "SUI", "SUNDOG", "SUSHI", "TAI", "THETA", "TIA", "TON", "TRUMP", "TRX", "TWT", "UNI", "UXLINK", "VIRTUAL", "WAVES", "WAXP", "WIF", "WLD", "XLM", "XRP", "XTER", "XTZ", "XWG", "X", "YFI", "ZEN", "ZIL", "ZRX"]
end

coinfilesdf = readdflogfolder(coinfilesdffilename())
if size(coinfilesdf, 1) > 0
    coins = dropmissing(coinfilesdf)[!, :coin]
end
# println("Used coins with sufficient liquidity: $coins")
rangedf = readdflogfolder(rangefilename())

if specialonly
    # renamepredictionfiles([mk1config().folder, mk2config().folder, mk3config().folder, mk4config().folder, mk5config().folder])
    println("No special task defined")
elseif inspectonly
    inspect(coins)
else
    (verbosity >= 2) && println("featuresconfig=$(Features.describe(featconfig))")
    (verbosity >= 2) && println("targetsconfig=$(Targets.describe(trgconfig))")
    getclassifier(coins) # ensure preparation of baseline mix classifier 
    ccmdf,cxcmdf = averageconfusionmatrix(coins)
    println(cxcmdf)
    println(ccmdf)
    gaindf = getgainsdf(coins)
    if size(gaindf, 1) > 0
        println(describe(gaindf))
        println(gaindf[1:2,:])
        gaindfgroup = groupby(gaindf, [:set, :label, :predicted, :openthreshold, :closethreshold])
        # cgaindf = combine(gaindfgroup, [:truth_longbuy, :truth_allclose] => ((lb, ac) -> sum(lb) / (sum(lb) + sum(ac)) * 100) => "longbuy_ppv%")
        cgaindf = combine(gaindfgroup, :gain => mean, :samplecount => mean, nrow, :gain => sum)
        println("cgaindf=$cgaindf")
    end

    distdf = getdistances(coins)
    if size(distdf, 1) > 0
        println("size(distdf)=$(size(distdf))")
        println("describe(distdf)=$(describe(distdf))")
        # println(distdf[.!ismissing.(distdf[!, :tpdistnext]),:])
        distdfgroup = groupby(distdf, [:set, :label])
        # println(distdfgroup)
        # diststatdf = combine(distdfgroup, :tpdistnext => (x -> safe(mean, x)) => :tpdistnext_mean, :tpdistnext => (x -> safe(std, x)) => :tpdistnext_std, :tpdistnext => (x -> (safe(count, x; default=0) / nrow)) => :tpdistnext_pct, :fpdistnext => (x -> safe(mean, x)) => :fpdistnext_mean, :fpdistnext => (x -> safe(std, x)) => :fpdistnext_std, :fpdistnext => (x -> (safe(count, x; default=0) / nrow)) => :fpdistnext_pct, :distfirst => (x -> safe(mean, x)) => :distfirst_mean, :distfirst => (x -> safe(std, x)) => :distfirst_std, :distfirst => (x -> (safe(count, x; default=0) / nrow)) => :distfirst_pct, :distlast => (x -> safe(mean, x)) => :distlast_mean, :distlast => (x -> safe(std, x)) => :distlast_std, :distlast => (x -> (safe(count, x; default=0) / nrow)) => :distlast_pct)
        diststatdf = combine(distdfgroup, :tpdistnext => (x -> safe(mean, x)) => :tpdistnext_mean, :tpdistnext => (x -> safe(median, x)) => :tpdistnext_median, :tpdistnext => (x -> safe(std, x)) => :tpdistnext_std, :fpdistnext => (x -> safe(mean, x)) => :fpdistnext_mean, :fpdistnext => (x -> safe(std, x)) => :fpdistnext_std, :distfirst => (x -> safe(mean, x)) => :distfirst_mean, :distfirst => (x -> safe(median, x)) => :distfirst_median, :distfirst => (x -> safe(std, x)) => :distfirst_std, :distlast => (x -> safe(mean, x)) => :distlast_mean, :distlast => (x -> safe(median, x)) => :distlast_median, :distlast => (x -> safe(std, x)) => :distlast_std)
    end
    println(diststatdf)
end


println("$(EnvConfig.now()) done @ $(currentconfig().folder)")
end # of TrendDetector
