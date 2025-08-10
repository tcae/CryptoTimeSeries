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

        featurestargetsfname = "features_targets_" * ohlcv.base * ".jdf"
        savedflogfolder(featurestargetsdf, featurestargetsfname)
        push!(settypesdf, (coin=ohlcv.base, featuresconfig=Features.describe(featconfig), targetsconfig=Targets.describe(trgconfig), featurestargetsfname=featurestargetsfname))
    else
        push!(settypesdf, (coin=ohlcv.base, featuresconfig=Features.describe(featconfig), targetsconfig=Targets.describe(trgconfig), featurestargetsfname=missing), promote=true)
        @info "skipping $basecoin due to no liquid ranges"
    end
    savedflogfolder(settypesdf, settypesfilename())
    return featurestargetsdf
end

rangefilename() = "ranges.jdf"
settypesfilename() = "settypesfiledict.jdf"
predictionsfilename(coin, classifier) = "predictions_$(coin)_$(classifier).jdf"


"Saves a given dataframe df in the current log folder using the given filename"
function savedflogfolder(df, filename)
    filepath = EnvConfig.logpath(filename)
    # try
        # EnvConfig.savebackup(filepath) # switched off until bug fixed
        JDF.savejdf(filepath, df)
        (verbosity >= 3) && println("$(EnvConfig.now()) saved dataframe to $(filepath)")
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
            (verbosity >= 3) && println(" - $(EnvConfig.now()) loaded $(size(df, 1)) rows successfully")
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(filepath)")
        end
    # catch e
    #     Logging.@error "exception $e detected"
    # end
    return df
end

isdflogfolder(filename) = isdir(EnvConfig.logpath(filename))

trendsineconfig() = Targets.Trend(5, 30, Targets.thresholds((longbuy=0.1, longhold=0.005, shorthold=-0.005, shortbuy=-0.1)))

trendccoinonfig(minwindow, maxwindow, buy, hold) = Targets.Trend(minwindow, maxwindow, Targets.thresholds((longbuy=buy, longhold=hold, shorthold=-hold, shortbuy=-buy)))

settypes() = ["train", "test", "eval"]

function featurestargetscollect!(settypesdf, coins, featconfig, trgconfig)
    rangedf = DataFrame()
    coincollect = []
    for coin in coins
        if coin in settypesdf[!, :coin]
            (verbosity >= 4) && println("entry for $coin found - skipping features / targets creation")
            continue
        end
        ftdf = featurestargetsliquidranges!(settypesdf, rangedf, coin, featconfig, trgconfig)
        if size(ftdf, 1) > 0
            push!(coincollect, (coin=coin, ftdf=ftdf))
        end
    end
    # println("len(coins)=$(length(coins)), len(coincollect)=$(length(coincollect))")
    for ct in coincollect
        rangedf2 = readdflogfolder(rangefilename())
        # addperiodgap!(rangedf2)
        @assert rangedf==rangedf2 "rangedf=$rangedf \n not equal rangedf2 = $rangedf2"
        settypesdf2 = readdflogfolder(settypesfilename())
        @assert settypesdf==settypesdf2 "settypesdf=$settypesdf \n not equal settypesdf2=$settypesdf2"
        @assert size(settypesdf[settypesdf[!, :coin] .== ct.coin, :], 1) == 1 "size(settypesdf[settypesdf[!, :coin] .== ct.coin, :], 1)=$(size(settypesdf[settypesdf[!, :coin] .== ct.coin, :], 1))"
        ftdf2 = readdflogfolder(settypesdf[settypesdf[!, :coin] .== ct.coin, :featurestargetsfname][begin])
        @assert ftdf2 == ct.ftdf
        rdfview = @view rangedf[rangedf[!, :coin] .== ct.coin, :]
        for rrow in eachrow(rdfview)
            ftdfrange = @view ftdf2[ftdf2[!, :rangeid] .== rrow.rangeid, :]
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
        (verbosity >= 3) && println("getlatestclassifier loaded: labels=$(nn.labels)")
    else
        nn = Classify.model001(Features.featurecount(featureconfig), [longbuy, allclose], classifiermenmonic(coins, coinix))
        (verbosity >= 3) && println("getlatestclassifier new: labels=$(nn.labels)")
    end
    @assert !isnothing(nn)
    return nn
end

function getfeaturestargets(settypesdf, settype, coin)
    subsetdf = @view settypesdf[settypesdf[!, :coin] .== coin, :featurestargetsfname]
    if size(subsetdf, 1) == 0
        (verbosity >= 3) && println("no features / targets data found for $coin")
        return nothing, nothing
    end
    if size(subsetdf, 1) > 1
        (verbosity >= 1) && println("found more than 1 dataset for $coin: $subsetdf \n using only first")
    end
    if ismissing(subsetdf[begin])
        (verbosity >= 3) && println("no features / targets data were generated for $coin probably due to insufficient liquidity")
        return nothing, nothing
    end
    ftdffull = readdflogfolder(subsetdf[begin])
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
    trainfeatures, traintargets = getfeaturestargets(settypesdf, "train", coin)
    if isnothing(trainfeatures) || isnothing(traintargets)
        return
    end
    (verbosity >= 2) && println("$(EnvConfig.now()) adapting classifier with $coin data")
    (verbosity >= 3) && println("before correction: $(Distributions.fit(UnivariateFinite, categorical(string.(traintargets)))))")
    (trainfeatures), traintargets = oversample((trainfeatures), traintargets)  # all classes are equally trained
    # (trainfeatures), traintargets = undersample((trainfeatures), traintargets)  # all classes are equally trained
    (verbosity >= 3) && println("after oversampling: $(Distributions.fit(UnivariateFinite, categorical(string.(traintargets)))))")
    Classify.adaptnn!(nn, trainfeatures, traintargets)
    (verbosity >= 3) && showlosses(nn)

    # EnvConfig.savebackup(Classify.nnfilename(nn.fileprefix))
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
                    if !isnothing(trainfeatures) && !isnothing(traintargets)
                        dfp = predictdf(dfp, nn, evalfeatures, evaltargets, settype)
                    end
                end
            end
        else  # adapt a coin specific classifier with all coins
            if !Classify.isadapted(nn)  # if single coin classifier is not adapted than take latest mix classifier as baseline
                nn = getlatestclassifier(coins, nothing, featconfig, trgconfig)
                Classify.setmnemonic(nn, classifiermenmonic(coins, coinix))
                adaptclassifierwithcoin!(nn, settypesdf, coins[coinix])
                for settype in settypes()
                    features, targets = getfeaturestargets(settypesdf, settype, coin)
                    if !isnothing(trainfeatures) && !isnothing(traintargets)
                        dfp = predictdf(dfp, nn, features, targets, settype)
                    end
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
        println(settypesdf)
        println("$(length(setdiff(coins, settypesdf[!, :coin]))) unconsidered coins that have no features/targets (probably due to low liquidity): $(setdiff(coins, settypesdf[!, :coin]))")
        println("$(length(setdiff(settypesdf[!, :coin], coins)))) coins with features/targets but are missing in the requested set of coins: $(setdiff(settypesdf[!, :coin], coins))")
        coins = intersect(coins, settypesdf[!, :coin])
        # println(settypesdf[begin:begin+2, :])
        # println(describe(settypesdf))
        println("$(length(coins)) processable coins")
        ok = true
        lbls = []
        for strow in eachrow(settypesdf)
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
    else
        @error "missing features/targets dataframe folder $(EnvConfig.logpath(settypesfilename()))"
    end
    rangedf = readdflogfolder(rangefilename())
    println("ranges dataframe file: $(EnvConfig.logpath(rangefilename()))")
    # println(rangedf)
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

println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$ARGS")
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
    # coins = []
    # for ohlcv in Ohlcv.OhlcvFiles()
    #     push!(coins, ohlcv.base)
    # end
    # println("length(coins)=$(length(coins))")
    # println(coins)
    coins = ["1INCH", "5IRE", "A8", "AAVE", "ACA", "ACH", "ACS", "ADA", "AEG", "AERO", "AEVO", "AGIX", "AGI", "AGLA", "AGLD", "AI16Z", "AIOZ", "AIXBT", "AKI", "ALCH", "ALGO", "ALT", "AMI", "ANIME", "ANKR", "AO", "APEX", "APE", "APP", "APRS", "APT", "ARB", "ARKM", "AR", "ASRR", "ATH", "ATOM", "AVAIL", "AVAX", "AVA", "AVL", "AXL", "AXS", "A", "B3", "BABYDOGE", "BAN", "BBL", "BBQ", "BB", "BCH", "BCUT", "BDXN", "BEAM", "BEL", "BERA", "BICO", "BLAST", "BLOCK", "BLUR", "BMT", "BNB", "BOBA", "BOB", "BOMB", "BOME", "BONK", "BRETT", "BR", "BTC", "BUBBLE", "C98", "CAKE", "CARV", "CATBNB", "CATI", "CELO", "CEL", "CGPT", "CHILLGUY", "CHRP", "CHZ", "CLOUD", "CMETH", "COA", "COMP", "COM", "COOKIE", "COOK", "COQ", "CORE", "COT", "CPOOL", "CRV", "CSPR", "CTA", "CTC", "CTT", "CUDIS", "CYBER", "DBR", "DECHAT", "DEEP", "DEFI", "DEGEN", "DGB", "DIAM", "DMAIL", "DOGE", "DOGS", "DOLO", "DOP1", "DOT", "DRIFT", "DSRUN", "DUEL", "DYDX", "DYM", "EGLD", "EGO", "EIGEN", "ELDE", "ELX", "ENA", "ENJ", "ENS", "EOS", "EPT", "ERA", "ESE", "ES", "ETC", "ETHFI", "ETHW", "ETH", "EVERY", "EXVG", "FAR", "FB", "FET", "FHE", "FIDA", "FIL", "FIRE", "FITFI", "FLIP", "FLOCK", "FLOKI", "FLOW", "FLR", "FLT", "FLUID", "FMB", "FON", "FORT", "FOXY", "FRAG", "FTM", "FTT", "FUEL", "FXS", "F", "G3", "G7", "GALAXIS", "GALA", "GAL", "GAME", "GLMR", "GMT", "GMX", "GOAT", "GPS", "GPT", "GRAPE", "GRASS", "GRT", "GSTS", "GST", "GTAI", "HAEDAL", "HBAR", "HFT", "HLG", "HMSTR", "HNT", "HOME", "HOOK", "HOT", "HTX", "HUMA", "HVH", "HYPER", "HYPE", "H", "ICNT", "ICP", "ID", "IMX", "INIT", "INJ", "INSP", "IO", "IP", "IRL", "IZI", "JASMY", "JTO", "JUP", "J", "KAIA", "KAS", "KAVA", "KCAL", "KDA", "KLAY", "KMNO", "KSM", "L3", "LADYS", "LAI", "LAYER", "LA", "LDO", "LENDS", "LEVER", "LFT", "LGX", "LINK", "LL", "LMWR", "LOOKS", "LRC", "LTC", "LUNAI", "LUNA", "LUNC", "MAGIC", "MAJOR", "MANA", "MANTA", "MASA", "MASK", "MATIC", "MAVIA", "MBOX", "MEMEFI", "MEME", "MERL", "METH", "MEW", "ME", "MILK", "MINA", "MINU", "MIX", "MKR", "MLK", "MNT", "MOCA", "MOG", "MOJO", "MON", "MORPHO", "MOVE", "MOVR", "MOZ", "MPLX", "MVL", "MYRIA", "MYRO", "NAKA", "NEAR", "NEIRO", "NEON", "NEWT", "NEXT", "NGL", "NIBI", "NLK", "NOT", "NS", "NUTS", "NXPC", "NYAN", "OBOL", "ODOS", "OKG", "OL", "OMG", "OMNI", "OM", "ONDO", "ONE", "OP", "ORDER", "ORDI", "ORT", "PAAL", "PARTI", "PENDLE", "PENGU", "PEOPLE", "PEPE", "PERP", "PFVS", "PLANET", "PLAY", "PLUME", "PNUT", "POKT", "POL", "PONKE", "POPCAT", "PORT3", "PORTAL", "PPT", "PRCL", "PRIME", "PUFFER", "PUFF", "PUMP", "PURSE", "PYTH", "PYUSD", "QNT", "QORPO", "QTUM", "RACA", "RAIN", "RATS", "RDNT", "RED", "RENDER", "RESOLV", "RNDR", "ROAM", "ROOT", "ROSE", "RPK", "RSS3", "RUNE", "RVN", "SAFE", "SALD", "SAND", "SATS", "SCA", "SCRT", "SCR", "SC", "SEI", "SEND", "SEOR", "SERAPH", "SFUND", "SHARK", "SHIB", "SHILL", "SHRAP", "SIDUS", "SIGN", "SIS", "SKATE", "SLG", "SMILE", "SNX", "SOLO", "SOLV", "SOL", "SONIC", "SOSO", "SPEC", "SPELL", "SPK", "SPX", "SQD", "SQR", "SQT", "SSV", "STAR", "STETH", "STG", "STOP", "STREAM", "STRK", "STX", "SUI", "SUNDOG", "SUN", "SUPRA", "SUSHI", "SVL", "SWEAT", "SWELL", "SXT", "S", "TAC", "TAIKO", "TAI", "TAP", "TAVA", "TA", "TENET", "THETA", "THRUST", "TIA", "TNSR", "TOKEN", "TOMI", "TON", "TOSHI", "TOWNS", "TREE", "TRUMP", "TRX", "TWT", "T", "ULTI", "UMA", "UNI", "USD1", "USDC", "USDE", "USTC", "UXLINK", "VANA", "VANRY", "VELAR", "VELO", "VENOM", "VET", "VEXT", "VIRTUAL", "VRA", "VRTX", "VV", "WAL", "WAVES", "WAXP", "WBTC", "WCT", "WELL", "WEMIX", "WEN", "WIF", "WLD", "WLKN", "WOO", "WWY", "W", "XAI", "XAR", "XAUT", "XAVA", "XCAD", "XDC", "XEC", "XION", "XLM", "XO", "XRP3L", "XRP", "XTER", "XTZ", "XWG", "X", "YFI", "ZEND", "ZENT", "ZEN", "ZEREBRO", "ZERO", "ZETA", "ZEX", "ZIG", "ZIL", "ZKF", "ZKJ", "ZKL", "ZK", "ZORA", "ZRC", "ZRO", "ZRX", "ZTX"]
    # coins =["BTC"]
end

if inspectonly
    verbosity = 2
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
    (verbosity >= 3) && println("featuresconfig=$(Features.describe(featconfig))")
    (verbosity >= 3) && println("targetsconfig=$(Targets.describe(trgconfig))")
    settypesdf = readdflogfolder(settypesfilename())
    (verbosity >= 3) && println("unconsidered coins that have no features/targets (probably due to low liquidity): $(setdiff(coins, settypesdf[!, :coin]))")
    (verbosity >= 3) && println("coins with features/targets but are missing in the requested set of coins: $(setdiff(settypesdf[!, :coin], coins))")
    # coins = intersect(coins, settypesdf[!, :coin])

    featurestargetscollect!(settypesdf, coins, featconfig, trgconfig)
    nnvector = adaptclassifiers(coins, featconfig, trgconfig, settypesdf)
    if verbosity >= 3
        for nnt in nnvector
            println("$(nnt.coin) - $(nnt.nn.fileprefix): $(Classify.nnfilename(nnt.nn.fileprefix))")
            showlosses(nnt.nn)
        end
    end
end


println("$(EnvConfig.now()) done")
end # of TrendDetector001
