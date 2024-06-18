"""
Train and evaluate the trading signal classifiers
"""
module Classify

using CSV, DataFrames, Logging, Dates
using BSON, JDF, Flux, Statistics, ProgressMeter, StatisticalMeasures, MLUtils
using CategoricalArrays
using CategoricalDistributions
# using Distributions
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv, CryptoXch
export noop, hold, sell, buy, strongsell, strongbuy

@enum InvestProposal noop hold sell buy strongsell strongbuy # noop = no statement possible, e.g. due to error, trading shall go on save side

const PREDICTIONLISTFILE = "predictionlist.csv"
DEBUG = false

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info, e.g. number of steps in rowix
"""
verbosity = 1


#region abstractclassifier
abstract type AbstractBaseClassifier end

abstract type AbstractClassifier <: AbstractConfiguration end
# field bc::Dict{AbstractString, AbstractBaseClassifier} connects base with AbstractBaseClassifier

"Adds ohlcv base with all ohlcv data, which can span the last requiredminutes or the time range of a back test / training.
The required feature/target will be supplemented. "
function addbase!(cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData) end

"Removes base and all associated data from cl. If base === nothing then all bases are removed."
removebase!(cl::AbstractClassifier, base::Union{Nothing, AbstractString}) = (isnothing(base) && hasproperty(cl, :bc) ? cl.bc = Dict() : delete!(cl.bc, base); cl)

"Returns all successfully added bases."
bases(cl::AbstractClassifier)::AbstractVector{AbstractString} = collect(keys(cl.bc))

"Returns the OhlcvData of base taht was previously added to cl."
ohlcv(cl::AbstractClassifier, base::AbstractString) = hasproperty(cl, :bc) ? (hasproperty(cl.bc[base], :ohlcv) ? cl.bc[base].ohlcv  : nothing) : nothing

""" Supplements all with Ohlcv and feature/target data of bases to match with base ohlcv data.
In a trade loop ohlcv data is updated first and then feature/target data is supplement with this function.
This to avoid a Classifier dependency to CryptoXch. """
function supplement!(cl::AbstractClassifier) end

"Write all feature and target data (but not ohlcv) of all bases to canned storage, which overwrites previous canned data."
function writetargetsfeatures(cl::AbstractClassifier) end

"Reduces all associated base data to the specified time range."
function timerangecut!(cl::AbstractClassifier, startdt::DateTime, enddt::DateTime) end

"Returns the required minutes of data history that is required for that classifier to work properly."
function requiredminutes(cl::AbstractClassifier)::Integer
    @error "missing $(typeof(cl)) implementation"
    return 0
end

"""
Returns number of part orders of an investment shall be split on average, e.g. to spread buying over a timerange.
Required time range gaps between the split orders shall be handled via classifier advice time gaps, i.e. the last advice is stored in BaseClassifier and an advice of same type is only issued after the tarheted time gap.
"""
buysplitparts(cl::AbstractClassifier, base::AbstractString) = 1

"Minimum number of minutes between consecutive gaps of the same trade type (e.g. buy)"
tradegapminutes(cl::AbstractClassifier, base::AbstractString) = 60

"Factor used to sell buy parts, i.e. 2 = sell with each order double the volume than bought. It shall be an Integer > 0"
sellvolumefactor(cl::AbstractClassifier, base::AbstractString) = 1

"`nothing` if no take profit request or a relative gain"
takeprofitgain(cl::AbstractClassifier, base::AbstractString) = nothing

"fee to be considered in classifier backtesting"
relativefee(cl::AbstractClassifier, base::AbstractString) = 0.08 / 100  # Bybit VIP1 taker fee: 0.08%

"Returns a trading advice for the specified time. Will return a noop in case of insufficient associated base data."
function advice(cl::AbstractClassifier, base::AbstractString, dt::DateTime)::InvestProposal
    @error "missing $(typeof(cl)) implementation"
    return noop
end

"Returns a trading advice for the current ohlcv index. Will return a noop in case of insufficient associated base data."
function advice(cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix)::InvestProposal
    @error "missing $(typeof(cl)) implementation"
    return noop
end

"Trains the classifier based on previosly provided data and stores the adapted classifier."
function train!(cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData)
    @error "missing $(typeof(cl)) implementation"
end

function configurationid4base(cl::AbstractClassifier, base::AbstractString)::Integer
    if !hasproperty(cl, :cfg)
        return 0
    end
    @error "missing $(typeof(cl)) implementation"
end

"Configure classifer according to configuration identifier"
function configureclassifier!(cl::AbstractClassifier, base::AbstractString, configid::Integer)
    if !hasproperty(cl, :cfg)
        return
    end
    @error "missing $(typeof(cl)) implementation"
end

"Iterates over ohlcv and logs simulation results in df"
function logsim!(df::AbstractDataFrame, cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData)
    otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
    piv = Ohlcv.dataframe(ohlcv)[!, :pivot]
    cfgid = configurationid4base(cl, ohlcv.base)
    lasttp = noop
    tradegap = tradegapminutes(cl, ohlcv.base)
    buyenabledix = sellenabledix = 1
    buyorders = []
    sellmultiple = round(Int, sellvolumefactor(cl, ohlcv.base))

    function pushsim(buydt, buyprice, selldt, sellprice, possiblebuycount, closetrade, fee=relativefee(cl, ohlcv.base))
        @assert buydt <= selldt "buydt=$buydt > selldt=$selldt"
        buyprice = buyprice * (1 + fee)
        sellprice = sellprice * (1 - fee)
        push!(df, (classifier=string(typeof(cl)), basecoin=ohlcv.base, cfgid=cfgid, buydt=buydt, buyprice=buyprice, selldt=selldt, sellprice=sellprice, gain=(sellprice - buyprice)/buyprice, possiblebuycount=possiblebuycount, closetrade=closetrade, fee=fee))
    end

    (verbosity >= 3) && println("$(EnvConfig.now()): logsim! running $(ohlcv.base) from $(otime[begin]) until $(otime[end])")
    # pushsim(otime[begin], piv[begin], otime[begin], piv[begin], buysplitparts(cl, ohlcv.base) - length(buyorders), "dummy", 0f0) # dummy to avoid empty dataframe
    for ix in eachindex(otime)
        Ohlcv.setix!(ohlcv, ix)
        tp = advice(cl, ohlcv)
        if ix >= buyenabledix
            possiblebuycount = buysplitparts(cl, ohlcv.base) - length(buyorders)
            if possiblebuycount > 0
                if tp == buy
                    push!(buyorders, (buydt=otime[ix], buyprice=piv[ix], possiblebuycount=possiblebuycount))
                elseif tp == strongbuy
                    while possiblebuycount > 0
                        push!(buyorders, (buydt=otime[ix], buyprice=piv[ix], possiblebuycount=possiblebuycount))
                        possiblebuycount -= 1
                    end
                end
                buyenabledix = ix + tradegap
                sellenabledix = ix + 1
            end
            lasttp = tp
        end
        takeprofitcheck(bo) = piv[ix] > bo.buyprice * (1 + takeprofitgain(cl, ohlcv.base))
        ixv = findall(takeprofitcheck, buyorders)
        if length(ixv) > 0
            sort!(ixv, rev = true)
            for bix in ixv
                @assert length(buyorders) > 0
                bo = popat!(buyorders, bix)
                pushsim(bo.buydt, bo.buyprice, otime[ix], piv[ix], bo.possiblebuycount, "takeprofit")
            end
            buyenabledix = ix + 1
            sellenabledix = ix + tradegap
            lasttp = tp
        end
        if ix >= sellenabledix
            if (tp in [sell, strongsell])
                for _ in 1:sellmultiple
                    if length(buyorders) > 0
                        bo = popfirst!(buyorders)
                        pushsim(bo.buydt, bo.buyprice, otime[ix], piv[ix], bo.possiblebuycount, "selladvice")
                        buyenabledix = ix + 1
                        sellenabledix = ix + tradegap
                    end
                end
            end
        end
    end
    while length(buyorders) > 0
        bo = popfirst!(buyorders)
        pushsim(bo.buydt, bo.buyprice, otime[end], piv[end], bo.possiblebuycount, "closebacktest")
    end
end

"Configures cl and evaluates cl by calling logsim! to log all trades in df."
function evaluate!(df::AbstractDataFrame, cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData)
    # default implementation with no spcific configuration
    logsim!(df, cl, ohlcv)
end

"Returns the full log file path including filename that is used to log the trade simulation results"
function evalfilename()
    return EnvConfig.logpath("ClassifierTradesim.jdf")
end

"Writes the DataFrame of simulation results to file"
function writesimulation(df)
    if (size(df, 1) > 0)
        fn = evalfilename()
        (verbosity >= 3) && println("$(EnvConfig.now()): saving classifier simulation trades in =$fn")
        JDF.savejdf(fn, parent(df))
    end
end

"Reads and returns the DataFrame of simulation results from file"
function readsimulation()
    df = DataFrame()
    fn = evalfilename()
    if isdir(fn)
        df = DataFrame(JDF.loadjdf(fn))
    end
    if !isnothing(df) && (size(df, 1) > 0 )
        (verbosity >= 2) && println("$(EnvConfig.now()) loaded classifier simulation trades from $fn")
    else
        (verbosity >= 2) && !isnothing(df) && println("$(EnvConfig.now()) Loading $fn failed")
    end
    return df
end

"""
Instantiates and evaluates all provided classifier types and logs all trades in df.
- basecoinsdf is a DataFrame as generated by Ohlcv.liquidcoins with the columns basecoin, startix, endix that are used for evaluation
- startdt and enddt further constraints the data specified by basecoinsdf if they are not disabled by value `nothing`
"""
function evaluateclassifiers(classifiertypevector, basecoinsdf, startdt, enddt)
    df = readsimulation()
    (verbosity >=3) && println("$(EnvConfig.now()): successfully read classifier simulation trades with $(size(df, 1)) entries")
    for clt in classifiertypevector
        cl = clt()
        for row in eachrow(basecoinsdf)
            ohlcv = Ohlcv.defaultohlcv(row.basecoin)
            Ohlcv.read!(ohlcv)
            otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
            sdt = isnothing(startdt) ? otime[row.startix] : startdt < otime[row.startix] ? otime[row.startix] : floor(startdt, Minute(1))
            edt = isnothing(enddt) ? otime[row.endix] : enddt > otime[row.endix] ? otime[row.endix] : floor(enddt, Minute(1))
            if edt < sdt
                verbosity >= 1 && @warn "$(ohlcv.base) requested time range $startdt - $enddt out of liquid range $(row)"
            end
            if sdt <= edt + Minute(requiredminutes(cl))
                Ohlcv.timerangecut!(ohlcv, sdt, edt)
                addbase!(cl, ohlcv)
                (verbosity >=2) && println("$(EnvConfig.now()): evaluating classifier $(string(clt)) for $(ohlcv.base) from $sdt until $edt")
                evaluate!(df, cl, ohlcv)
                writesimulation(df)
            else
                verbosity >= 1 && @warn "$(ohlcv.base) insufficient data coverage for simulation: enddt=$edt - startdt=$sdt = $(Minute(edt-sdt)) < Minute(requiredminutes(cl)=$(Minute(requiredminutes(cl)))"
            end
        end
    end
    return df
end

function kpioverview(df::AbstractDataFrame, classifiertype)
    if size(df, 1) > 0
        df = @view df[df[!, :classifier] .== string(classifiertype), :]
        # gdf = groupby(df, [:classifier, :basecoin, :cfgid])
        gdf = groupby(df, [:basecoin, :cfgid])
        rdf = combine(gdf, :gain => mean, :gain => median, :gain => sum, :gain => minimum, :gain => maximum, nrow, :closetrade => (x -> count(x .== "takeprofit")) => :takeprofit, :closetrade => (x -> count(x .== "selladvice")) => :selladvice, :closetrade => (x -> count(x .== "closebacktest")) => :closebacktest, :possiblebuycount => (x -> count(x .== 1)) => :buyblocked, groupindices => :groupindex)
        cl = classifiertype()
        if hasproperty(cl, :cfg)
            leftjoin!(rdf, cl.cfg, on=:cfgid)
        end
        # println(rdf)
        rdffilename = EnvConfig.logpath(string(classifiertype) * "simulationresult.csv")
        EnvConfig.checkbackup(rdffilename)
        CSV.write(rdffilename, rdf, decimal=',', delim=';')  # decimal as , to consume with European locale
        return rdf, gdf
    else
        println("noresults - empty dataframe will not be saved")
        return df, nothing
    end
end

#endregion abstractclassifier


mutable struct NN
    model
    optim
    lossfunc
    labels::Vector{String}  # in fixed sequence as index == class id
    description
    mnemonic::String
    fileprefix::String  # filename without suffix
    predecessors::Vector{String}  # filename vector without suffix
    featuresdescription::String
    targetsdescription::String
    losses::Vector
    predictions::Vector{String}  # filename vector without suffix of predictions
end

function NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix)
    return NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix, String[], "", "", [], String[])
end

function Base.show(io::IO, nn::NN)
    println(io, "NN: labels=$(nn.labels) #predecessors=$(length(nn.predecessors)) mnemonic=$(nn.mnemonic) fileprefix=$(nn.fileprefix) #losses=$(length(nn.losses))")
    println(io, "NN: featuresdecription=$(nn.featuresdescription) targetsdecription=$(nn.targetsdescription)")
    println(io, "NN: predecessors=$(nn.predecessors)")
    println(io, "NN: predictions=$(nn.predictions)")
    println(io, "NN: description=$(nn.description)")
end


#region DataPrep
# folds(data, nfolds) = partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...)

"""
    - Returns a Dict of setname => vector of row ranges
    - input
        - `rowrange` is the range of rows that are split in subranges, e.g. 2000:5000
        - `samplesets` is a Dict of setname => relative setsize with sum of all setsizes == 1
        - `gapsize` is the number of rows between partitions of different sets that are not included in any partition
        - `relativesubrangesize` is the subrange size relative to `rowrange` of a consecutive range assigned to a setinterval
        - gaps will be removed from a subrange to avoid crosstalk bias between ranges
        - relativesubrangesize * length(rowrange) > 2 * gapsize
        - any relative setsize > 2 * relativesubrangesize
        - a mixture of ranges and individual indices in a vector can be unpacked into a index vector via `[ix for r in rr for ix in r]`
"""
function setpartitions(rowrange, samplesets::Dict, gapsize, relativesubrangesize)
    rowstart = rowrange[1]
    rowend = rowrange[end]
    rows = rowend - rowstart + 1
    gapsize = relativesubrangesize * rows > 2 * gapsize ? gapsize : floor(Int, relativesubrangesize * rows / 3)
    println("$(EnvConfig.now()) setpartitions rowrange=$rowrange, samplesets=$samplesets gapsize=$gapsize relativesubrangesize=$relativesubrangesize")
    # @assert relativesubrangesize * rows > 2 * gapsize
    @assert max(collect(values(samplesets))...) > 2 * relativesubrangesize "max(collect(values(samplesets))...)=$(max(collect(values(samplesets))...)) <= 2 * relativesubrangesize = $(2 * relativesubrangesize)"
    minrelsetsize = min(collect(values(samplesets))...)
    @assert minrelsetsize > 0.0
    sn = [setname for setname in keys(samplesets)]
    snl = length(sn)
    ix = rowstart
    aix = 1
    arr =[[] for _ in sn]
    while ix <= rowend
        subrangesize = round(Int, rows * relativesubrangesize * samplesets[sn[aix]] / minrelsetsize)
        push!(arr[aix], ((ix == rowstart ? ix : ix + gapsize), min(rowend, ix+subrangesize-1)))
        ix = ix + subrangesize
        aix = aix % snl + 1
    end
    res = Dict(sn[aix] => [t[1]:t[2] for t in arr[aix]] for aix in eachindex(arr))
    return res
end

function test_setpartitions()
    # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 0, 1/9)
    # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 1, 1/9)
    res = setpartitions(1:49, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 1, 3/50)
    # 3-element Vector{Vector{Any}}:
    #     [UnitRange{Int64}[1:3, 10:12, 19:21]]
    #     [UnitRange{Int64}[4:6, 13:15, 22:24]]
    #     [UnitRange{Int64}[7:9, 16:18, 25:26]]

    for (k, vec) in res
        # for rangeset in vec  # rangeset not necessary
            for range in vec  # rangeset
                for ix in range
                    # println("$k: $ix")
                end
            end
        # end
    end
    println(res)
end

"prediction list is a csv file with the one line summary of an training/evaluation to compare the different configurations quickly"
function predictionlist()
    df = nothing
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    pf = EnvConfig.logpath(PREDICTIONLISTFILE)
    if isfile(pf)
        df = CSV.read(pf, DataFrame, decimal='.', delim=';')
        # println(describe(df))
        # println(df)
    end
    EnvConfig.setlogpath(sf)
    return df
end

function registerpredictions(filename, evalperf=missing, testperf=missing)
    df = predictionlist()
    if isnothing(df)
        df = DataFrame(abbrev=String[], evalperf=[], testperf=[], comment=String[], subfolder=String[], filename=String[])
    else
        rows = findall(==(filename), df[!, "filename"])
        if length(rows) > 1
            @error "found $rows entries instead of one in $PREDICTIONLISTFILE"
            return
        elseif length(rows) == 1
            # no action - filename already registered
            return
        elseif length(rows) == 0
            # not found - don't return but add filename
        else
            println("found for $filename the follwing in $PREDICTIONLISTFILE: $rows")
        end
    end
    sf = EnvConfig.logsubfolder()
    push!(df, ("abbrev", round(evalperf, digits=3), round(testperf, digits=3), "comment", sf, filename))
    EnvConfig.setlogpath(nothing)
    pf = EnvConfig.logpath(PREDICTIONLISTFILE)
    CSV.write(pf, df, decimal='.', delim=';')
    EnvConfig.setlogpath(sf)
end

function loadpredictions(filename)
    filename = predictionsfilename(filename)
    df = DataFrame()
    try
        df = DataFrame(JDF.loadjdf(EnvConfig.logpath(filename)))
        println("loaded $filename predictions dataframe of size=$(size(df))")
    catch e
        Logging.@warn "exception $e detected"
    end
    if !("pivot" in names(df))
        ix = findfirst("USDT", filename)
        if isnothing(ix)
            @warn "cannot repair pivot because no USDT found in filename"
        else
            # println("ix=$ix typeof(ix)=$(typeof(ix))")  # ix is and index range
            base = SubString(filename, 1, (first(ix)-1))
            ohlcv = Ohlcv.defaultohlcv(base)
            startdt = minimum(df[!, "opentime"])
            if startdt != df[begin, "opentime"]
                @warn "startdt=$startdt != df[begin, opentime]=$(df[begin, "opentime"])"
            end
            enddt = maximum(df[!, "opentime"])
            if enddt != df[end, "opentime"]
                @warn "enddt=$enddt != df[begin, opentime]=$(df[end, "opentime"])"
            end
            Ohlcv.read!(ohlcv)
            subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
            df[:, "pivot"] = Ohlcv.dataframe(ohlcv).pivot
            println("succesful repair of pivot")
        end
    end
    # checktargetsequence(df[!, "targets"])
    checklabeldistribution(df[!, "targets"])
    return df
end

function savepredictions(df, fileprefix)
    filename = predictionsfilename(fileprefix)
    println("saving $filename predictions dataframe of size=$(size(df))")
    try
        JDF.savejdf(EnvConfig.logpath(filename), df)
    catch e
        Logging.@warn "exception $e detected"
    end
    return fileprefix
end

function predictionsdataframe(nn::NN, setranges, targets, predictions, f3::Features.Features003)
    df = DataFrame(permutedims(predictions, (2, 1)), nn.labels)
    # for (ix, label) in enumerate(nn.labels)
    #     df[label, :] = predictions[ix, :]
    # end
    setlabels = fill("unused", size(df, 1))
    for (setname, vec) in setranges
        for range in vec  # rangeset
            for ix in range
                setlabels[ix] = setname
            end
        end
    end
    sc = categorical(setlabels; compress=true)
    df[:, "set"] = sc
    df[:, "targets"] = categorical(targets; levels=nn.labels, compress=true)
    df[:, "opentime"] = Ohlcv.dataframe(f3.f2.ohlcv).opentime[f3.firstix:end]
    df[:, "pivot"] = Ohlcv.dataframe(f3.f2.ohlcv).pivot[f3.firstix:end]
    println("Classify.predictionsdataframe size=$(size(df)) keys=$(names(df))")
    println(describe(df, :all))
    fileprefix = uppercase(Ohlcv.basecoin(f3.f2.ohlcv) * Ohlcv.quotecoin(f3.f2.ohlcv)) * "_" * nn.fileprefix
    savepredictions(df, fileprefix)
    return fileprefix
end

function predictionmatrix(df, labels)
    pred = Array{Float32, 2}(undef, length(labels), size(df, 1))
    pnames = names(df)
    @assert size(pred, 1) == length(labels)
    @assert size(pred, 2) == size(df, 1)
    for (ix, label) in enumerate(labels)
        if label in pnames
            pred[ix, :] = df[:, label]
        else
            pred[ix, :] .= 0.0f0
        end
    end
    return pred
end

"Target consistency check that a hold or close signal can only follow a buy or sell signal"
function checktargetsequence(targets::CategoricalArray)
    labels = levels(targets)
    ignoreix, longbuyix, longholdix, closeix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in ["ignore", "longbuy", "longhold", "close", "shorthold", "shortbuy"])
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    buy = levelcode(first(targets)) in [longholdix, closeix] ? longbuyix : (levelcode(first(targets)) == shortholdix ? shortbuyix : ignoreix)
    for (ix, tl) in enumerate(targets)
        if levelcode(tl) == longbuyix
            buy = longbuyix
        elseif (levelcode(tl) == longholdix)
            if (buy != longbuyix)
                @error "$ix: missed $longbuyix ($(labels[longbuyix])) before $longholdix ($(labels[longholdix]))"
            end
        elseif levelcode(tl) == shortbuyix
            buy = shortbuyix
        elseif (levelcode(tl) == shortholdix)
            if (buy != shortbuyix)
                @error "$ix: missed $shortbuyix ($(labels[shortbuyix])) before $shortholdix ($(labels[shortholdix]))"
            end
        elseif (levelcode(tl) == closeix)
            if (buy == ignoreix)
                @error "$ix: missed either $shortbuyix ($(labels[shortbuyix])) or $longbuyix ($(labels[longbuyix])) before $closeix ($(labels[closeix]))"
            end
        elseif (levelcode(tl) == ignoreix)
            buy = ignoreix
        else
            @error "$ix: unexpected $(levelcode(tl)) ($(labels[closeix]))"
        end
    end
    println("target sequence check ready")

end

function checklabeldistribution(targets::CategoricalArray)
    labels = levels(targets)
    cnt = zeros(Int, length(labels))
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    for tl in targets
        cnt[levelcode(tl)] += 1
    end
    targetcount = size(targets, 1)
    println("target label distribution in %: ", [(labels[i], round(cnt[i] / targetcount*100, digits=1)) for i in eachindex(labels)])

end

"returns features as dataframe and targets as CategorialArray (which is why this function resides in Classify and not in Targets)."
function featurestargets(regrwindow, f3::Features.Features003, pe::Dict)
    @assert regrwindow in keys(Features.featureslookback01) "regrwindow=$regrwindow not in keys(Features.featureslookback01)=$(keys(Features.featureslookback01))"
    @assert regrwindow in keys(pe)
    features, _ = Features.featureslookback01[regrwindow](f3)
    println(describe(features, :all))
    featuresdescription = "featureslookback01[$regrwindow](f3)"
    labels, relativedist, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe[regrwindow])
    targets = labels[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    relativedist = relativedist[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    println(describe(DataFrame(reshape(targets, (length(targets), 1)), ["targets"]), :all))
    targetsdescription = "ohlcvlabels(ohlcv.pivot, PriceExtreme[$regrwindow])"
    features = Array(features)  # change from df to array
    @assert size(targets, 1) == size(features, 1) == size(relativedist, 1)  "size(targets, 1)=$(size(targets, 1)) == size(features, 1)=$(size(features, 1)) == size(relativedist, 1)=$(size(relativedist, 1))"
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    return features, targets, featuresdescription, targetsdescription, relativedist
end

function combifeaturestargets(nnvec::Vector{NN}, f3::Features.Features003, pe::Dict)
    @assert "combi" in keys(pe) "`combi` not in keys(pe)=$(keys(pe))"
    labels, relativedist, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe["combi"])
    targets = labels[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    relativedist = relativedist[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    targetsdescription = "ohlcvlabels(ohlcv.pivot, PriceExtreme[combi])"
    features = nothing
    for nn in nnvec
        df = loadpredictions(nn.predictions[begin])
        pred = predictionmatrix(df, nn.labels)
        features = isnothing(features) ? pred : vcat(features, pred)
    end
    featuresdescription = "basepredictions[$([nn.mnemonic for nn in nnvec])]"
    @assert size(targets, 1) == size(features, 2) == size(relativedist, 1)  "size(targets, 1)=$(size(targets, 1)) == size(features, 2)=$(size(features, 2)) == size(relativedist, 1)=$(size(relativedist, 1))"
    return features, targets, featuresdescription, targetsdescription, relativedist
end

"Returns the column (=samples) subset of featurestargets as given in ranges, which shall be a vector of ranges"
function subsetdim2(featurestargets::AbstractArray, ranges::AbstractVector)
    dim = length(size(featurestargets))
    @assert 0 < dim <= 2 "dim=$dim"

    # working with views
    ixvec = Int32[ix for range in ranges for ix in range]
    res = dim == 1 ? view(featurestargets, ixvec) : view(featurestargets, :, ixvec)
    return res

    # copying ranges - currently disabled
    res = nothing
    for range in ranges
        res = dim == 1 ? (isnothing(res) ? featurestargets[range] : vcat(res, featurestargets[range])) : (isnothing(res) ? featurestargets[:, range] : hcat(res, featurestargets[:, range]))
    end
    return res
end

#endregion DataPrep

#region LearningNetwork Flux

"""```
Neural Net description:
lay_in = featurecount
lay_out = length(labels)
lay1 = 3 * lay_in
lay2 = round(Int, lay1 * 2 / 3)
lay3 = round(Int, (lay2 + lay_out) / 2)
model = Chain(
    Dense(lay_in => lay1, relu),   # activation function inside layer
    BatchNorm(lay1),
    Dense(lay1 => lay2, relu),   # activation function inside layer
    BatchNorm(lay2),
    Dense(lay2 => lay3, relu),   # activation function inside layer
    BatchNorm(lay3),
    Dense(lay3 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
lossfunc = Flux.logitcrossentropy
```
"""
function model001(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 3 * lay_in
    lay2 = round(Int, lay1 * 2 / 3)
    lay3 = round(Int, (lay2 + lay_out) / 2)
    model = Chain(
        Dense(lay_in => lay1, relu),   # activation function inside layer
        BatchNorm(lay1),
        Dense(lay1 => lay2, relu),   # activation function inside layer
        BatchNorm(lay2),
        Dense(lay2 => lay3, relu),   # activation function inside layer
        BatchNorm(lay3),
        Dense(lay3 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.logitcrossentropy

    description = (@doc model001);
    mnemonic = "NN" * (isnothing(mnemonic) ? "" : "$(mnemonic)")
    fileprefix = mnemonic * "_" * EnvConfig.runid()
    nn = NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix)
    return nn
end

"""
creates and adapts a neural network using `features` with ground truth label provided with `targets` that belong to observation samples with index ix within the original sample sequence.
relativedist is a vector
"""
function adaptnn!(nn::NN, features::AbstractMatrix, targets::AbstractVector)
    onehottargets = Flux.onehotbatch(targets, Targets.all_labels)  # onehot class encoding of an observation as one column
    loader = Flux.DataLoader((features, onehottargets), batchsize=64, shuffle=true);

    # Training loop, using the whole data set 1000 times:
    nn.losses = Float32[]
    testmode!(nn, false)
    minloss = maxloss = missing
    breakmsg = ""
    @showprogress for epoch in 1:1000
    # for epoch in 1:200  # 1:1000
        losses = Float32[]
        for (x, y) in loader
            loss, grads = Flux.withgradient(nn.model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                #y_hat is a Float32 matrix , y is a boolean matrix both of size (classes, batchsize). The loss function returns a single Float32 number
                nn.lossfunc(y_hat, y)
            end
            minloss = ismissing(minloss) ? loss : min(minloss, loss)
            maxloss = ismissing(maxloss) ? loss : max(maxloss, loss)
            Flux.update!(nn.optim, nn.model, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
        epochloss = mean(losses)
        push!(nn.losses, epochloss)  # register only mean(losses) over a whole epoch
        if (epoch > 10) && (nn.losses[end-4] <= nn.losses[end-3] <= nn.losses[end-2] <= nn.losses[end-1] <= nn.losses[end])
            breakmsg = "stopping adaptation after epoch $epoch due to no loss reduction ($(nn.losses[end-4:end])) in last 4 epochs"
            break
        end
    end
    testmode!(nn, true)
    nn.optim # parameters, momenta and output have all changed
    println(breakmsg)  # print after showprogress loop to avoid cluttered text output
    println("minloss=$minloss  maxloss=$maxloss")
    return nn
end

" Returns a predictions Float Array of size(classes, observations)"
predict(nn::NN, features) = nn.model(features)  # size(classes, observations)

function predictiondistribution(predictions, classifiertitle)
    maxindex = mapslices(argmax, predictions, dims=1)
    dist = zeros(Int, maximum(unique(maxindex)))  # maximum == length
    for ix in maxindex
        dist[ix] += 1
    end
    println("$(EnvConfig.now()) prediction distribution with $classifiertitle classifier: $dist")
end

function adaptbase(regrwindow, f3::Features.Features003, pe::Dict, setranges::Dict)
    println("$(EnvConfig.now()) preparing features and targets for regressionwindow $regrwindow")
    features, targets, featuresdescription, targetsdescription, relativedist = featurestargets(regrwindow, f3, pe)
    # trainix = subsetdim2(collect(firstindex(targets):lastindex(targets)), setranges["base"])
    trainfeatures = subsetdim2(features, setranges["base"])
    traintargets = subsetdim2(targets, setranges["base"])
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    (trainfeatures), traintargets = oversample((trainfeatures), traintargets)  # all classes are equally trained
    # println("after oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    println("$(EnvConfig.now()) adapting machine for regressionwindow $regrwindow")
    nn = model001(size(trainfeatures, 1), Targets.all_labels, Features.periodlabels(regrwindow))
    nn = adaptnn!(nn, trainfeatures, traintargets)
    nn.featuresdescription = featuresdescription
    nn.targetsdescription = targetsdescription
    println("$(EnvConfig.now()) predicting with machine for regressionwindow $regrwindow")
    pred = predict(nn, features)
    push!(nn.predictions, predictionsdataframe(nn, setranges, targets, pred, f3))
    # predictiondistribution(pred, nn.mnemonic)

    println("saving adapted classifier $(nn.fileprefix)")
    # println(nn)
    savenn(nn)
    # println("$(EnvConfig.now()) load machine from file $(nn.fileprefix) for regressionwindow $regrwindow and predict")
    # nntest = loadnn(nn.fileprefix)
    # println(nntest)
    # predtest = predict(nntest, features)
    # @assert pred ≈ predtest  "NN results differ from loaded NN: pred[:, 1:5] = $(pred[:, begin:begin+5]) predtest[:, 1:5] = $(predtest[:, begin:begin+5])"
    return nn
end

function adaptcombi(nnvec::Vector{NN}, f3::Features.Features003, pe::Dict, setranges::Dict)
    println("$(EnvConfig.now()) preparing features and targets for combi classifier")
    features, targets, featuresdescription, targetsdescription, relativedist = combifeaturestargets(nnvec, f3, pe)
    # trainix = subsetdim2(collect(firstindex(targets):lastindex(targets)), setranges["combi"])
    # println("size(features)=$(size(features)), size(targets)=$(size(targets))")
    trainfeatures = subsetdim2(features, setranges["combi"])
    traintargets = subsetdim2(targets, setranges["combi"])
    # println("before oversample size(trainfeatures)=$(size(trainfeatures)), size(traintargets)=$(size(traintargets))")
    (trainfeatures), traintargets = oversample((trainfeatures), traintargets)  # all classes are equally trained
    # println("after oversample size(trainfeatures)=$(size(trainfeatures)), size(traintargets)=$(size(traintargets))")
    println("$(EnvConfig.now()) adapting machine for combi classifier")
    nn = model001(size(trainfeatures, 1), Targets.all_labels, "combi")
    nn = adaptnn!(nn, trainfeatures, traintargets)
    nn.featuresdescription = featuresdescription
    nn.targetsdescription = targetsdescription
    nn.predecessors = [nn.fileprefix for nn in nnvec]
    println("$(EnvConfig.now()) predicting with combi classifier")
    pred = predict(nn, features)
    push!(nn.predictions, predictionsdataframe(nn, setranges, targets, pred, f3))
    # predictiondistribution(pred, nn.mnemonic)
    @assert size(pred, 2) == size(features, 2)  "size(pred[combi], 2)=$(size(pred, 2)) == size(features, 2)=$(size(features, 2))"
    @assert size(targets, 1) == size(features, 2)  "size(targets[combi], 1)=$(size(targets, 1)) == size(features, 2)=$(size(features, 2))"

    println("saving adapted classifier $(nn.fileprefix)")
    # println(nn)
    savenn(nn)
    # println("$(EnvConfig.now()) load machine from file $(nn.fileprefix) for regressionwindow combi and predict")
    # nntest = loadnn(nn.fileprefix)
    # println(nntest)
    # predtest = predict(nntest, features)
    # @assert pred ≈ predtest  "NN results differ from loaded NN: pred[:, 1:5] = $(pred[:, begin:begin+5]) predtest[:, 1:5] = $(predtest[:, begin:begin+5])"
    return nn
end

function lossesfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return "losses_" * prefix * ".jdf"
end

function loadlosses!(nn)
    filename = lossesfilename(nn.fileprefix)
    df = DataFrame()
    try
        df = DataFrame(JDF.loadjdf(EnvConfig.logpath(filename)))
        nn.losses = df[!, "losses"]
        println("loaded $filename losses dataframe of size=$(size(df))")
    catch e
        Logging.@warn "exception $e detected"
    end
    return nn.losses
end

function savelosses(nn::NN)
    filename = lossesfilename(nn.fileprefix)
    df = DataFrame(reshape(nn.losses, (length(nn.losses), 1)), ["losses"])
    println("saving $filename losses dataframe of size=$(size(df))")
    try
        JDF.savejdf(EnvConfig.logpath(filename), df)
    catch e
        Logging.@warn "exception $e detected"
    end
end

function nnfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return prefix * ".bson"
end

function compresslosses(losses)
    if length(losses) <= 1000
        return losses
    end
    @warn "nn.losses length=$(length(losses))"
    lvec = losses
    gap = floor(Int, length(lvec) / 100)
    start = length(lvec) % 100
    closses = [l for (ix, l) in enumerate(lvec) if ((ix-start) % gap) == 0]
    @assert length(closses) <= 100 "length(loses)=$(length(closses))"
    return closses
end

function savenn(nn::NN)
    # nn.losses = compresslosses(nn.losses)
    BSON.@save EnvConfig.logpath(nnfilename(nn.fileprefix)) nn
    # @error "save machine to be implemented for pure flux" filename
    # smach = serializable(mach)
    # JLSO.save(filename, :machine => smach)
end

function loadnn(filename)
    nn = model001(1, Targets.all_labels, "dummy")  # dummy data struct
    BSON.@load EnvConfig.logpath(nnfilename(filename)) nn
    # loadlosses!(nn)
    return nn
    # @error "load machine to be implemented for pure flux" filename
    # loadedmach = JLSO.load(filename)[:machine]
    # Deserialize and restore learned parameters to useable form:
    # restore!(loadedmach)
    # return loadedmach
end

#endregion LearningNetwork

#region Evaluation

"""
Groups trading pairs and stores them with the corresponding gain in a DataFrame that is returned.
Pairs that cross a set before closure are being forced closed at set border.
"""
function trades(predictions::AbstractDataFrame, thresholds::Vector)
    df = DataFrame(set=CategoricalArray(undef, 0; levels=levels(predictions.set), ordered=false), opentrade=Int32[], openix=Int32[], closetrade=Int32[], closeix=Int32[], gain=Float32[])
    if size(predictions, 1) == 0
        return df
    end
    scores, maxindex = maxpredictions(predictions)
    labels = levels(predictions.targets)
    ignoreix, longbuyix, longholdix, closeix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in ["ignore", "longbuy", "longhold", "close", "shorthold", "shortbuy"])
    buytrade = (tradeix=closeix, predix=0, set=predictions.set[begin])  # tradesignal, predictions index
    holdtrade = (tradeix=closeix, predix=0, set=predictions.set[begin])  # tradesignal, predictions index

    function closetrade!(tradetuple, closetrade, ix)
        gain = (predictions.pivot[ix] - predictions.pivot[tradetuple.predix]) / predictions.pivot[tradetuple.predix] * 100
        gain = tradetuple.tradeix in [longbuyix, longholdix] ? gain : -gain
        push!(df, (tradetuple.set, tradetuple.tradeix, tradetuple.predix, closetrade, ix, gain))
        return (tradeix=closeix, predix=ix, set=predictions.set[ix])
    end

    function closeifneeded!(buychecktrades, holdchecktrades, closetrade, ix)
        buytrade = (buytrade.tradeix == buychecktrades) || ((buytrade.set != predictions.set[ix]) && (buytrade.tradeix in [longbuyix, shortbuyix])) ? closetrade!(buytrade, closetrade, ix) : buytrade
        holdtrade = (holdtrade.tradeix == holdchecktrades) || ((buytrade.set != predictions.set[ix]) && (buytrade.tradeix in [longholdix, shortholdix])) ? closetrade!(holdtrade, closetrade, ix) : holdtrade
        return buytrade, holdtrade
    end

    for ix in eachindex(maxindex)
        labelix = maxindex[ix]
        score = scores[ix]
        if (labelix == longbuyix) && (thresholds[longbuyix] <= score)
            buytrade, holdtrade = closeifneeded!(shortbuyix, shortholdix, longbuyix, ix)
            buytrade = buytrade.tradeix == closeix ? (tradeix=longbuyix, predix=ix, set=predictions.set[ix]) : buytrade
        elseif (labelix == longholdix) && (thresholds[longholdix] <= score)
            buytrade, holdtrade = closeifneeded!(shortbuyix, shortholdix, longholdix, ix)
            holdtrade = holdtrade.tradeix == closeix ? (tradeix=longholdix, predix=ix, set=predictions.set[ix]) : holdtrade
        elseif ((labelix == closeix) && (thresholds[closeix] <= score)) || ((labelix == ignoreix) && (thresholds[ignoreix] <= score))
            buytrade = buytrade.tradeix != closeix ? closetrade!(buytrade, closeix, ix) : buytrade
            holdtrade = holdtrade.tradeix != closeix ? closetrade!(holdtrade, closeix, ix) : holdtrade
        elseif (labelix == shortholdix) && (thresholds[shortholdix] <= score)
            buytrade, holdtrade = closeifneeded!(longbuyix, longholdix, shortholdix, ix)
            holdtrade = holdtrade.tradeix == closeix ? (tradeix=shortholdix, predix=ix, set=predictions.set[ix]) : holdtrade
        elseif (labelix == shortbuyix) && (thresholds[shortbuyix] <= score)
            buytrade, holdtrade = closeifneeded!(longbuyix, longholdix, shortbuyix, ix)
            buytrade = buytrade.tradeix == closeix ? (tradeix=shortbuyix, predix=ix, set=predictions.set[ix]) : buytrade
        end
    end
    return df
end

function tradeperformance(trades::AbstractDataFrame, labels::Vector)
    df = DataFrame(set=CategoricalArray(undef, 0; levels=labels, ordered=false), trade=[], tradeix=[], gainpct=[], cnt=[], gainpctpertrade=[])
    for tset in levels(trades.set)
        for ix in eachindex(labels)
            selectedtrades = filter(row -> (row.set == tset) && (row.opentrade == ix), trades, view=true)
            if size(selectedtrades, 1) > 0
                tcount = count(i-> !ismissing(i), selectedtrades.gain)
                tsum = sum(selectedtrades.gain)
                push!(df, (tset, labels[ix], ix, tsum, tcount, round(tsum/tcount, digits=3)))
            end
        end
        selectedtrades = filter(row -> (row.set == tset), trades, view=true)
        tcount = count(i-> !ismissing(i), selectedtrades.gain)
        tsum = sum(selectedtrades.gain)
        push!(df, (tset, "total", missing, tsum, tcount, tsum/tcount))
    end
    # println(df)
    return df
end

"maps a 0.0 <= score <= 1.0  to one of `thresholdbins` bins"
score2bin(score, thresholdbins) = max(min(floor(Int, score / (1.0/thresholdbins)) + 1, thresholdbins), 1)

"maps the index of one of `thresholdbins` bins to a score"
bin2score(binix, thresholdbins) = round((binix-1)*1.0/thresholdbins; digits = 2), round(binix*1.0/thresholdbins; digits = 2)

"""
generates summary statistics from predictions
"""
function extendedconfusionmatrix(predictions::AbstractDataFrame, thresholdbins=10)
    scores, maxindex = maxpredictions(predictions)
    labels = levels(predictions.targets)
    confcatsyms = [:tp, :tn, :fp, :fn]
    confcat = Dict(zip(confcatsyms, 1:length(confcatsyms)))
    # preallocate collection matrices with columns TP, TN, FP, FN and rows as bins with lower separation value x/thresholdbins per label per set
    setnames = levels(predictions.set)
    cmc = zeros(Int, length(setnames), length(labels), length(confcatsyms), thresholdbins)
    for ix in eachindex(maxindex)
        labelix = maxindex[ix]  # label of maxscore
        binix = score2bin(scores[ix], thresholdbins)
        if labelix == levelcode(predictions.targets[ix])
            cmc[levelcode(predictions.set[ix]), labelix, confcat[:tp], binix] += 1
        else  # labelix != levelcode(predictions.targets[ix])
            cmc[levelcode(predictions.set[ix]), labelix, confcat[:fp], binix] += 1
        end
    end
    cm = zeros(Int, length(setnames), length(labels), length(confcatsyms), thresholdbins)
    for six in eachindex(setnames)
        for lix in eachindex(labels)
            for bix in 1:thresholdbins
                for bix2 in bix:thresholdbins
                    cm[six, lix, confcat[:tp], bix] += cmc[six, lix, confcat[:tp], bix2]
                    cm[six, lix, confcat[:fp], bix] += cmc[six, lix, confcat[:fp], bix2]
                end
                for bix2 in 1:(bix-1)
                    cm[six, lix, confcat[:fn], bix] += cmc[six, lix, confcat[:tp], bix2]
                    cm[six, lix, confcat[:tn], bix] += cmc[six, lix, confcat[:fp], bix2]
                end
            end
        end
    end
    setnamevec = [setnames[six] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    sc = categorical(setnamevec; levels=setnames)
    labelsvec = [labels[lix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    lc = categorical(labelsvec; levels=labels)
    binvec = [(scr = bin2score(bix, thresholdbins); "$bix/[$(scr[1])-$(scr[2])]") for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    bc = categorical(binvec)
    tpvec = [cm[six, lix, confcat[:tp], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    tnvec = [cm[six, lix, confcat[:tn], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    fpvec = [cm[six, lix, confcat[:fp], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    fnvec = [cm[six, lix, confcat[:fn], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    allvec = tpvec + tnvec + fpvec + fnvec
    tpprc = round.(tpvec ./ allvec .* 100.0; digits=2)
    tnprc = round.(tnvec ./ allvec .* 100.0; digits=2)
    fpprc = round.(fpvec ./ allvec .* 100.0; digits=2)
    fnprc = round.(fnvec ./ allvec .* 100.0; digits=2)
    tpr = round.(tpvec ./ (tpvec + fnvec); digits=2)
    fpr = round.(fpvec ./ (fpvec + tnvec); digits=2)
    xcdf = DataFrame("set" => sc, "pred_label" => lc, "bin" => bc, "tp" => tpvec, "tn" => tnvec, "fp" => fpvec, "fn" => fnvec, "tp%" => tpprc, "tn%" => tnprc, "fp%" => fpprc, "fn%" => fnprc, "tpr" => tpr, "fpr" => fpr)
    # println(xcdf)
    return xcdf
    #TODO next step: take only first of an equal trading signal sequence according to threshold -> how often is a sequence missed?
 end

 function confusionmatrix(predictions::AbstractDataFrame)
    scores, maxindex = maxpredictions(predictions)
    labels = levels(predictions.targets)
    confcatsyms = [:tp, :tn, :fp, :fn]
    confcat = Dict(zip(confcatsyms, 1:length(confcatsyms)))
    # preallocate collection matrices with columns TP, TN, FP, FN and rows as bins with lower separation value x/thresholdbins per label per set
    setnames = levels(predictions.set)
    cm = zeros(Int, length(setnames), length(labels), length(labels))
    for ix in eachindex(maxindex)
        cm[levelcode(predictions.set[ix]), maxindex[ix], levelcode(predictions.targets[ix])] += 1
    end
    setnamevec = [setnames[six] for six in eachindex(setnames) for lix in eachindex(labels)]
    sc = categorical(setnamevec; levels=setnames)
    labelsvec = ["pred_"*labels[lix] for six in eachindex(setnames) for lix in eachindex(labels)]
    plc = categorical(labelsvec)
    cdf = DataFrame(:set => sc, :predlabel => plc)
    pred_sum = zeros(Int, length(setnames) * length(labels))
    for (ix, l) in enumerate(labels)
        if l != "pred_invalid"
            lvec = [cm[six, lix, ix] for six in eachindex(setnames) for lix in eachindex(labels)]
            cdf[:, "truth_"*l] = lvec
            pred_sum += lvec
        end
    end
    # cdf[:, "truth_other"] = pred_sum - [cm[six, lix, lix] for six in eachindex(setnames) for lix in eachindex(labels)]
    cdf[:, "truth_all"] = pred_sum
    for (ix, l) in enumerate(labels)
        cdf[:, ("truth_"*l*"_%")] = round.(cdf[:, ("truth_"*l)] ./ pred_sum * 100; digits=2)
    end
    # println(cdf)
    return cdf
 end

function predictionsfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return prefix * ".jdf"
end

labelvec(labelindexvec, labels=Targets.all_labels) = [labels[i] for i in labelindexvec]

labelindexvec(labelvec, labels=Targets.all_labels) = [findfirst(x -> x == focuslabel, labels) for focuslabel in labelvec]

"returns a (scores, labelindices) tuple of best predictions. Without labels the index is the index within levels(df.targets)."
function maxpredictions end
function maxpredictions(predictions::AbstractMatrix, labels=Targets.all_labels)
    @assert length(labels) == size(predictions, 1)
    maxindex = mapslices(argmax, predictions, dims=1)
    scores = [predictions[maxindex[i], i] for i in eachindex(maxindex)]
    return scores, maxindex
end

# function maxpredictions(predictions::AbstractDataFrame)
#     if size(predictions, 1) == 0
#         return [],[]
#     end
#     scores = zeros32(size(predictions, 1))
#     maxindex = zeros(UInt32, size(predictions, 1))
#     for (lix, label) in enumerate(names(predictions))
#         if eltype(predictions[!, label]) <: AbstractFloat
#             if !(label in Targets.all_labels)
#                 @warn "unexpected predictions class: $label"
#             end
#             vec = predictions[!, label]
#             for ix in eachindex(vec)
#                 if scores[ix] < vec[ix]
#                     scores[ix] = vec[ix]
#                     maxindex[ix] = lix
#                 end
#             end
#         end
#     end
#     return scores, maxindex
# end

function maxpredictions(predictions::AbstractDataFrame, labels=Targets.all_labels)
    if size(predictions, 1) == 0
        return [],[]
    end
    # labels = levels(predictions.targets)
    # @assert labels == levels(predictions.targets) "labels=$labels != levels(predictions.targets)=$(levels(predictions.targets))"
    pnames = names(predictions)
    scores = zeros32(size(predictions, 1))
    maxindex = zeros(UInt32, size(predictions, 1))
    missinglabels = []
    for (lix, label) in enumerate(labels)
        if label in pnames
            vec = predictions[!, label]
            for ix in eachindex(vec)
                if scores[ix] < vec[ix]
                    scores[ix] = vec[ix]
                    maxindex[ix] = lix
                end
            end
        else
            push!(missinglabels, label)
        end
    end
    if length(missinglabels) > 0
        println("maxpredictions: no predictions for $missinglabels")
    end
    # ts, tm = maxpredictionsfromcols(predictions)
    # @assert all([pnames[tm[ix]] == labels[maxindex[ix]] for ix in eachindex(maxindex)]) && (ts == scores) "maxindex check = $(tm == maxindex)  scores check = $(ts == scores)"
    return scores, maxindex
end

"returns a (scores, booltargets) tuple of binary predictions of class `label`, i.e. booltargets[ix] == true if bestscore is assigned to focuslabel"
function binarypredictions end

function binarypredictions(predictions::AbstractMatrix, focuslabel::String, labels=Targets.all_labels)
    flix = findfirst(x -> x == focuslabel, labels)
    @assert !isnothing(flix) && (firstindex(labels) <= flix <= lastindex(labels)) "$focuslabel==$(isnothing(flix) ? "nothing" : flix) not found in $labels[$(firstindex(labels)):$(lastindex(labels))]"
    @assert length(labels) == size(predictions, 1) "length(labels)=$(length(labels)) == size(predictions, 1)=$(size(predictions, 1))"
    if size(predictions, 2) == 0
        return [],[]
    end
    maxindex = mapslices(argmax, predictions, dims=1)
    ixvec = [flix == maxindex[i] for i in eachindex(maxindex)]
    # labelvec = [flix == maxindex[i] ? focuslabel : "other" for i in eachindex(maxindex)]
    return (length(ixvec) > 0 ? predictions[flix, :] : []), ixvec
end

function binarypredictions(predictions::AbstractDataFrame, focuslabel::String, labels=Targets.all_labels)
    @assert focuslabel in labels
    if size(predictions, 1) == 0
        return [],[]
    end
    pnames = names(predictions)
    if !(focuslabel in pnames)
        return fill(0.0f0, size(predictions, 1)), fill(false, size(predictions, 1))
    end
    _, maxindex = maxpredictions(predictions, labels)
    flix = 0
    for (lix, label) in enumerate(labels)
        flix = focuslabel == label ? lix : flix
    end
    ixvec = [flix == maxindex[i] for i in eachindex(maxindex)]
    return predictions[:, focuslabel], ixvec
end

function smauc(scores, predlabels)
    y = categorical(predlabels, ordered=true)  # StatisticalMeasures package
    ŷ = UnivariateFinite([false, true], scores, augment=true, pool=y)   # StatisticalMeasures package
    return auc(ŷ, y)  # StatisticalMeasures package
end

function aucscores(pred, labels=Targets.all_labels)
    aucdict = Dict()
    if typeof(pred) == DataFrame ? (size(pred, 1) > 0) : (size(pred, 2) > 0)
        for focuslabel in labels
            scores, predlabels = binarypredictions(pred, focuslabel, labels)
            if length(unique(predlabels)) == 2
                aucdict[focuslabel] = smauc(scores, predlabels)
            else
                # @warn "no max score for $focuslabel => auc = 0"
                aucdict[focuslabel] = 0.0
            end
        end
    end
    return aucdict
end

# aucscores(pred, labels=Targets.all_labels) = Dict(String(focuslabel) => auc(binarypredictions(pred, focuslabel, labels)...) for focuslabel in labels)
    # auc_scores = []
    # for class_label in unique(targets)
    #     class_scores, class_events = binarypredictions(pred, targets, class_label)
    #     auc_score = auc(class_scores, class_events)
    #     push!(auc_scores, auc_score)
    # end
    # return auc_scores
# end

"Returns a Dict of class => roc tuple of vectors for false_positive_rates, true_positive_rates, thresholds"
function roccurves(pred, labels=Targets.all_labels)
    rocdict = Dict()
    if typeof(pred) == DataFrame ? (size(pred, 1) > 0) : (size(pred, 2) > 0)
        for focuslabel in labels
            scores, predlabels = binarypredictions(pred, focuslabel, labels)
            if length(unique(predlabels)) == 2
                y = categorical(predlabels, ordered=true)  # StatisticalMeasures package
                ŷ = UnivariateFinite([false, true], scores, augment=true, pool=y)   # StatisticalMeasures package
                rocdict[focuslabel] = roc_curve(ŷ, y)  # StatisticalMeasures package
            else
                # @warn "no max score for $focuslabel => roc = nothing"
                rocdict[focuslabel] = nothing
            end
        end
    end
    return rocdict
end

# function plotroccurves(rc::Dict, customtitle)
#     plotlyjs()
#     default(legend=true)
#     plt = plot()
#     for (k, v) in rc
#         if !isnothing(v)
#             # println("$k = $(length(v[1]))  xxx $(length(v[2]))")
#             # plot!(v, label=k)  # ROC package
#             plot!(v[1], v[2], label=k)  # StatisticalMeasures package
#         end
#     end
#     # plot!(xlab="false positive rate", ylab="true positive rate")
#     plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black, label="chance")  # StatisticalMeasures package
#     xlabel!("false positive rate")
#     ylabel!("true positive rate")
#     title!("receiver operator characteristic $customtitle")
#     display(plt)
# end

function confusionmatrix(pred, targets, labels=Targets.all_labels)
    _, maxindex = maxpredictions(pred, labels)
    targets = [String(targets[ix]) for ix in eachindex(targets)]
    predlabels = labelvec(maxindex, labels)
    StatisticalMeasures.ConfusionMatrices.confmat(predlabels, targets)
end


function evaluatepredictions(predictions::AbstractDataFrame, fileprefix)
    assetpair, nntitle = split(fileprefix, "_")[1:2]
    title = assetpair * "_" * nntitle
    if EnvConfig.configmode == EnvConfig.test
        cdf = confusionmatrix(predictions)
        xcdf = extendedconfusionmatrix(predictions)
    end
    labels = levels(predictions.targets)
    thresholds = [0.01f0 for l in labels]
    tdf = trades(predictions, thresholds)
    tpdf = tradeperformance(tdf, labels)

    selectedtrades = filter(row -> (row.set == "eval") && ((row.trade == "longbuy") || (row.trade == "shortbuy")), tpdf, view=true)
    if length(selectedtrades.gainpct) > 0
        tcount = sum(selectedtrades.cnt)
        tsum = sum(selectedtrades.gainpct)
        evalperf = round(tsum/tcount, digits=3)
    else
        evalperf = 0.0
    end
    selectedtrades = filter(row -> (row.set == "test") && ((row.trade == "longbuy") || (row.trade == "shortbuy")), tpdf, view=true)
    if length(selectedtrades.gainpct) > 0
        tcount = sum(selectedtrades.cnt)
        tsum = sum(selectedtrades.gainpct)
        testperf = round(tsum/tcount, digits=3)
    else
        testperf = 0.0
    end
    if EnvConfig.configmode == EnvConfig.production
        registerpredictions(fileprefix, evalperf, testperf)
    end

    for s in levels(predictions.set)
        if s == "unused"
            continue
        end
        sdf = filter(row -> row.set == s, predictions, view=true)
        if size(sdf, 1) > 0
            if EnvConfig.configmode == EnvConfig.test
                # println(title)
                # aucscores = Classify.aucscores(sdf)
                # println("auc[$s, $title]=$(aucscores)")
                # rc = Classify.roccurves(sdf)
                # Classify.plotroccurves(rc, "$s / $title")
                println(title)
                show(stdout, MIME"text/plain"(), Classify.confusionmatrix(sdf, sdf.targets))  # prints the table
                println(title)
                println(filter(row -> row.set == s, cdf, view=true))
                println(title)
                println(filter(row -> row.set == s, xcdf, view=true))
            end
            println(title)
            println(filter(row -> row.set == s, tpdf, view=true))
        else
            @warn "no auc or roc data for [$s, $title] due to missing predictions"
        end
    end
end

function evaluatepredictions(filename)
    println("$(EnvConfig.now()) load predictions from file $(filename)")
    df = loadpredictions(filename)
    evaluatepredictions(df,filename)
end

function evaluateclassifier(nn::NN)
    title = nn.mnemonic
    if length(nn.predecessors) > 0
        println("$(EnvConfig.now()) evaluating $(length(nn.predecessors)) predecessors of $title")
    end
    for nnfileprefix in nn.predecessors
        evaluateclassifier(nnfileprefix)
    end
    println("$(EnvConfig.now()) evaluating classifier $title")
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
    for fileprefix in nn.predictions
        df = loadpredictions(fileprefix)
        evaluatepredictions(df, fileprefix)
    end
end

function evaluateclassifier(fileprefix::String)
    nn = loadnn(fileprefix)
    evaluateclassifier(nn)
end

function evaluate(ohlcv::Ohlcv.OhlcvData, labelthresholds; select=nothing)
    nnvec = NN[]
    # push!(nnvec, loadnn("NN1d_23-12-18_15-58-18_gitSHA-b02abf01b3a714054ea6dd92d5b683648878b079"))
    f3, pe = Targets.loaddata(ohlcv, labelthresholds)
    # println(f3)
    len = length(Ohlcv.dataframe(f3.f2.ohlcv).pivot) - Features.ohlcvix(f3, 1) + 1
    # println("$(EnvConfig.now()) len=$len  length(Ohlcv.dataframe(f3.f2.ohlcv).pivot)=$(length(Ohlcv.dataframe(f3.f2.ohlcv).pivot)) Features.ohlcvix(f3, 1)=$(Features.ohlcvix(f3, 1))")
    setranges = setpartitions(1:len, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/80)
    for (s,v) in setranges
        println("$s: length=$(length(v))")
    end
    # Threads.@threads for regrwindow in f3.regrwindow
    for regrwindow in f3.regrwindow
        if isnothing(select) || (regrwindow in select)
            push!(nnvec, adaptbase(regrwindow, f3, pe, setranges))
        else
            println("skipping $regrwindow classifier due to not selected")
        end
    end
    if isnothing(select) || ("combi" in select)
        nncombi = adaptcombi(nnvec, f3, pe, setranges)
        for nn in nncombi.predecessors
            evaluateclassifier(nn)
        end
        evaluateclassifier(nncombi)
    else
        for nn in nnvec
            evaluateclassifier(nn)
        end
        # println("skipping combi classifier due to not selected")
    end
    println("$(EnvConfig.now()) ready with adapting and evaluating classifier stack")
end

function evaluate(base::String, startdt::Dates.DateTime=DateTime("2017-07-02T22:54:00"), period=Dates.Year(10); select=nothing)
    ohlcv = Ohlcv.defaultohlcv(base)
    enddt = startdt + period
    Ohlcv.read!(ohlcv)
    subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
    println("loaded $ohlcv")
    labelthresholds = Targets.defaultlabelthresholds
    evaluate(ohlcv, labelthresholds, select=select);
end

# function evaluate(base::String; select=nothing)
#     ohlcv = Ohlcv.defaultohlcv(base)
#     Ohlcv.read!(ohlcv)
#     println("loaded $ohlcv")
#     println(describe(Ohlcv.dataframe(ohlcv)))
#     labelthresholds = Targets.defaultlabelthresholds
#     evaluate(ohlcv, labelthresholds, select=select);
# end

function evaluatetest(startdt=DateTime("2022-01-02T22:54:00")::Dates.DateTime, period=Dates.Day(40); select=nothing)
    enddt = startdt + period
    ohlcv = TestOhlcv.testohlcv("SINEUSDT", startdt, enddt)
    labelthresholds = Targets.defaultlabelthresholds
    evaluate( ohlcv, labelthresholds, select=select)
end

#endregion Evaluation

#region Classifier001


STDREGRWINDOW = 1440
STDREGRWINDOWSET = [12*60, 24*60, 3*24*60, 10*24*60]  # [rw for rw in Features.regressionwindows004 if  (12*60) <= rw <= (3*24*60)]
STDHEADWINDOW = Dict(12*60 => [0, 60], 24*60 => [0, 60], 3*24*60 => [0, 4*60], 10*24*60 => [0, 12*60])
STDTRENDWINDOW = Dict(12*60 => [0, 3*24*60], 24*60 => [0, 10*24*60], 3*24*60 => [0, 10*24*60], 10*24*60 => [0, 10*24*60])
STDGAINTHRSHLD = 0.01f0
STDGAINTHRSHLDSET = [0.01f0, 0.02f0, 0.04f0]  # excluding 0.005f0,
FEE = 0.1f0 / 100f0  # 0.1%
# FEE = 0f0  # no fee
MINSIMGAINPCT = 5  # 5% is standard minimum simulation gain
MINGAINLOSSRATIO = 2 # minimum gain versus max loss ratio
SIMSTART = 10000f0  # simulation budget
SIMTRADEFRACTION = 1/1000  # fraction of total simulation budget for one trade
STDMODEL = "baseline"
STDMODELSET = [STDMODEL]

mutable struct BaseClassifier001
    ohlcv::Union{Nothing, Ohlcv.OhlcvData}    # ohlcv cache
    f4::Union{Nothing, Features.Features004}  # f4 cache that is shared with all regressionwindows
    bestix # nothing if cfg is empty or not yet determined else row index of best config within cfg
    cfg::AbstractDataFrame
    dbgdf
end

function BaseClassifier001(ohlcv::Ohlcv.OhlcvData, f4=Features.Features004(ohlcv, usecache=true))
    cl = isnothing(f4) ? nothing : BaseClassifier001(ohlcv, f4, nothing, emptyconfigdf(), nothing)
    return cl
end

function Base.show(io::IO, cl::BaseClassifier001)
    println(io, "BaseClassifier001[$(cl.ohlcv.base)]: ohlcv.ix=$(cl.ohlcv.ix),  ohlcv length=$(size(cl.ohlcv.df,1)), has f4=$(!isnothing(cl.f4)), bestix=$(cl.bestix), size(cfg)=$(size(cl.cfg))")  # \n$(cl.ohlcv) \n$(cl.f4)")
end

function timerangecut!(cl::BaseClassifier001, startdt::DateTime, enddt::DateTime)
    if !isnothing(cl.f4)
        ohlcvstartdt = startdt - Minute(Classify.requiredminutes(cl)-1)
        Ohlcv.timerangecut!(cl.ohlcv, ohlcvstartdt, enddt)
        Features.timerangecut!(cl.f4, startdt, enddt)
        Features.f4offset!(cl.f4, cl.ohlcv)
    end
end

function writetargetsfeatures(cl::BaseClassifier001)
    Ohlcv.write(cl.ohlcv)
    if !isnothing(cl.f4)
        Features.write(cl.f4)
    end
end

# function baseconfiguration(cl::BaseClassifier001)
#     if !isnothing(cl.bestix) && (size(cl.cfg, 1) > 0)
#         return merge(configuration(Classifier001(), cl.cfg[cl.bestix, :cfgid]), (basecoin=cl.cfg[cl.bestix, :basecoin], simgain=cl.cfg[cl.bestix, :simgain], minsimgain=cl.cfg[cl.bestix, :minsimgain],))
#     else
#         return nothing
#     end
# end

supplement!(cl::BaseClassifier001) = Features.supplement!(cl.f4, cl.ohlcv)


# function _cfgfilename(timestamp::Union{Nothing, DateTime}, ext)
#     cfgfilename = EnvConfig.logpath(CLASSIFIER001_CONFIGFILE)
#     if isnothing(timestamp)
#         cfgfilename = join([cfgfilename, ext], ".")
#     else
#         cfgfilename = join([cfgfilename, Dates.format(timestamp, "yy-mm-dd_HH-MM"), ext], "_", ".")
#     end
#     return cfgfilename
# end

requiredminutes() = maximum(Features.regressionwindows004)
requiredminutes(cl::BaseClassifier001) = requiredminutes()

"config DataFrame with columns: basecoin, cfgid, minqteqty, startdt, enddt, totalcnt, sellcnt, buycnt, maxccbuycnt, medianccbuycnt, unresolvedcnt, totalgain, mediangain, meangain, cumgain, maxcumgain, mincumgain, maxgap, mediangap, simgain, minsimgain, maxsimgain"
emptyconfigdf() = DataFrame(basecoin=String[], cfgid=Int16[], minqteqty=Float32[], startdt=DateTime[], enddt=DateTime[], totalcnt=Int32[], sellcnt=Int32[], buycnt=Int32[], maxccbuycnt=Int32[], medianccbuycnt=Float32[], unresolvedcnt=Int32[], totalgain=Float32[], mediangain=Float32[], meangain=Float32[], cumgain=Float32[], maxcumgain=Float32[], mincumgain=Float32[], maxgap=Int32[], mediangap=Float32[], simgain=Float32[], minsimgain=Float32[], maxsimgain=Float32[])
dummyconfig = (basecoin="dummy", cfgid=0, minqteqty=0f0, startdt=DateTime("2020-01-01T01:00:00"), enddt=DateTime("2020-01-01T01:00:00"), totalcnt=0, sellcnt=0, buycnt=0, maxccbuycnt=0, medianccbuycnt=0f0, unresolvedcnt=0, totalgain=0f0, mediangain=0f0, meangain=0f0, cumgain=0f0, maxcumgain=0f0, mincumgain=0f0, maxgap=0, mediangap=0f0, simgain=0f0, minsimgain=0f0, maxsimgain=0f0)

"Add a new BaseClassifier configuration and isenties it as the current/best one. The configuration is returned as DataFrameRow."
function addreplaceconfig!(cl::BaseClassifier001, base, regrwindow, gainthreshold, headwindow, trendwindow)
    @assert regrwindow in Features.regressionwindows004
    cfgid = configurationid(Classifier001(), (regrwindow=regrwindow, gainthreshold=gainthreshold, headwindow=headwindow, trendwindow=trendwindow))
    cfgix = findall((cl.cfg[!, :basecoin] .== base) .&& (cl.cfg[!, :cfgid] .== cfgid))
    cfg = nothing
    if length(cfgix) == 0
        push!(cl.cfg, (dummyconfig..., basecoin = base, cfgid = cfgid))
        cl.bestix = lastindex(cl.cfg, 1)
        cfg = cl.cfg[end, :]
        # sort!(cl.cfg, [:basecoin, :regrwindow, :gainthreshold])
    else
        for ix in reverse(cfgix) # use reverse to maintain correct index in loop
            if ix == first(cfgix)
                cl.bestix = ix
                # leave first entry
                cfg = cl.cfg[ix, :]
            else
                #remove all other entries
                (verbosity >= 2) && @warn "$(length(cfgix)) classifier001 entries instead of 1 $ix: $(cl.cfg[ix, :]) will be removed"
                deleteat!(cl.cfg, ix)
            end
        end
    end
    return cfg
end

function removeconfig!(cl::BaseClassifier001, base, regrwindow, gainthreshold, headwindow, trendwindow)
    cfgid = configurationid(Classifier001(), (regrwindow=regrwindow, gainthreshold=gainthreshold, headwindow=headwindow, trendwindow=trendwindow))
    cfgix = findall((cl.cfg[!, :basecoin] .== base) .&& (cl.cfg[!, :cfgid] .== cfgid))
    if length(cfgix) > 0
        for ix in reverse(cfgix) # use reverse to maintain correct index in loop
            deleteat!(cl.cfg, ix)
        end
    end
end

function baseadvice(cl::BaseClassifier001, ohlcvix, regrwindow, gainthreshold, headwindow, trendwindow)
    ohlcvdf = Ohlcv.dataframe(cl.ohlcv)
    fix = Features.f4ix(cl.f4, ohlcvix)
    dt = ohlcvdf[ohlcvix, :opentime]
    if fix <= 0
        return noop
    end
    if cl.f4.rw[regrwindow][fix, :opentime] == dt
        if ohlcvdf[ohlcvix, :pivot] > cl.f4.rw[regrwindow][fix, :regry] * (1 + gainthreshold)
            if (headwindow == 0) || (cl.f4.rw[headwindow][fix, :grad] < 0f0)
                return sell
            end
        elseif ohlcvdf[ohlcvix, :pivot] < cl.f4.rw[regrwindow][fix, :regry] * (1 - gainthreshold)
            if (trendwindow == 0) || (cl.f4.rw[trendwindow][fix, :grad] >= 0f0)
                if (headwindow == 0) || (cl.f4.rw[headwindow][fix, :grad] > 0f0)
                    return buy
                end
            end
        end
    else  # if cl.f4.rw[regrwindow][begin, :opentime] < dt
        @warn "expected $(cl.ohlcv.base) ohlcv opentime[ohlcvix]=$dt not matching $(cl.f4.rw[regrwindow][fix, :opentime]) of f4[$regrwindow] with start=$(cl.f4.rw[regrwindow][begin, :opentime]) end=$(cl.f4.rw[regrwindow][end, :opentime])"
    end
    return hold
end

"""
Classifier001 is handled as singleton due to the BaseClassifier implementation that calls for every new config Classifier001(), which is then not in sync with other instances concerning their cfg.
"""
_cls001 = nothing  # Classifier001

mutable struct Classifier001 <: AbstractClassifier
    cfg  # DataFrame of classifier configurations
    bc::Dict{AbstractString, BaseClassifier001}  # base => BaseClassifier001
    function Classifier001() # implement persistent singleton
        global _cls001
        if isnothing(_cls001) || isnothing(_cls001.cfg)
            _cls001 = new(nothing, Dict())
            readconfigurations!(_cls001)
            @assert !isnothing(_cls001.cfg)
        end
        return _cls001
    end
end

requiredminutes(cl::Classifier001) = requiredminutes()

function EnvConfig.emptyconfigdf(cfgset::Classifier001)
    return  DataFrame(
        cfgid=Int16[],               # classifier identificator
        regrwindow=Int16[],         # regression window of decisive regression line in minutes
        gainthreshold=Float32[],    # gain threshold ratio of minimum gain/price of regression line for trade signal
        headwindow=Int16[],         # regression window of heading regression line in minutes (0 = switched off) ; grad < 0 to sell; grad > 0 to buy
        trendwindow=Int16[],        # regression window of trend regression line in minutes (0 = switched off) ; grad > 0 to buy
    )
end

function addbase!(cl::Classifier001, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier001(ohlcv)
    if !isnothing(bc)
        cl.bc[ohlcv.base] = bc
    end
end

addreplaceconfig!(cl::Classifier001, base, regrwindow, gainthreshold, headwindow, trendwindow) = base in keys(cl.bc) ? addreplaceconfig!(cl.bc[base], base, regrwindow, gainthreshold, headwindow, trendwindow) : nothing

removebase!(cl::Classifier001, base::Union{Nothing, AbstractString}) = isnothing(base) ? cl.bc = Dict() : delete!(cl.bc, base);

bases(cl::Classifier001)::AbstractVector{AbstractString} = collect(keys(cl.bc))

ohlcv(cl::Classifier001, base::AbstractString) = cl.bc[base].ohlcv

function supplement!(cl::Classifier001)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier001)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

function timerangecut!(cl::Classifier001, startdt::DateTime, enddt::DateTime)
    for bcl in values(cl.bc)
        timerangecut!(bcl, startdt, enddt)
    end
end

function buysplitparts(cl::Classifier001, base::AbstractString)
    bc = cl.bc[base]
    config = configuration(cl,bc.cfg[bc.bestix, :cfgid])
    return floor(Int, config.regrwindow / 2)
end

function advice(cl::Classifier001, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix)::InvestProposal
    bc = cl.bc[ohlcv.base]
    cfg = configuration(cl, bc.cfg[bc.bestix, :cfgid])
    return baseadvice(bc, ohlcvix, cfg.regrwindow, cfg.gainthreshold, cfg.headwindow, cfg.trendwindow)
end

function advice(cl::Classifier001, base::AbstractString, dt::DateTime)::InvestProposal
    bc = base in keys(cl.bc) ? cl.bc[base] : nothing
    if isnothing(bc)
        return noop
    end
    ohlcvix = Ohlcv.rowix(bc.ohlcv.df[!, :opentime], dt)
    if bc.ohlcv.df[ohlcvix, :opentime] == dt
        cfg = configuration(cl, bc.cfg[bc.bestix, :cfgid])
        return baseadvice(bc, ohlcvix, cfg.regrwindow, cfg.gainthreshold, cfg.headwindow, cfg.trendwindow)
    else
        return noop
    end
end

function evaluate!(df::AbstractDataFrame, cl::Classifier001, ohlcv::Ohlcv.OhlcvData)
    for regrwindow in STDREGRWINDOWSET
        for headwindow in STDHEADWINDOW[regrwindow]
            for trendwindow in STDTRENDWINDOW[regrwindow]
                for gainthreshold in STDGAINTHRSHLDSET
                    addreplaceconfig!(cl, ohlcv.base, regrwindow, gainthreshold, headwindow, trendwindow)
                    logsim!(df, cl, ohlcv)
                end
            end
        end
    end
end


function configurationid4base(cl::Classifier001, base::AbstractString)::Integer
    bc = base in keys(cl.bc) ? cl.bc[base] : nothing
    if isnothing(bc)
        return 0
    else
        return bc.cfg[bc.bestix, :cfgid]
    end
end



#endregion Classifier001

#region Classifier002

mutable struct BaseClassifier002
    ohlcv::Ohlcv.OhlcvData
    meanwindowix::Integer
end

"""
Classification is based on MA (moving average) price change of a time range (=window) length that is determined by the largest window that can catch all relative noise price changes
- close when pivot price <= mean
- buy when
  - last price relative to mean > rbt to ensure hysteresis avoiding quick fallback sell
  - mean relative to last mean per day > rbt
  - mean trendwindow price change >= 0
- one window change step per time (=per advice call)
  - change to next smaller window if abs(last price relative to mean) >= rbt
  - change to next larger window if abs(last price relative to mean) < rbt for that larger window

"""
mutable struct Classifier002 <: AbstractClassifier
    "bd: base date maps the base string onto ohlcv data"
    bc::Dict
    "window vector of considered MA window in minutes"
    window::AbstractVector{Integer}  # have to be sorted in increasing order
    "rbt: relative buy threshold of mean price increase / day"
    rbt::Float32
    "trendwindow: mean(trendwindow) - mean(trendwindow @ trendwindow distance) shall be > 0 to enable buy trades. Is switched off if trendwindow==0."
    trendwindow::Integer
    dbgdf
    function Classifier002(;window=[15, 30, 60, 2*60, 4*60, 8*60, 16*60], rbt=0.01f0, trendwindow=4*60)
        window = sort(window)
        cl = new(Dict(), window, rbt, trendwindow, DataFrame())
        return cl
    end
end

function addbase!(cl::Classifier002, ohlcv::Ohlcv.OhlcvData)
    cl.bc[ohlcv.base] = BaseClassifier002(ohlcv, lastindex(cl.window))
end

removebase!(cl::Classifier002, base::Union{Nothing, AbstractString}) = isnothing(base) ? cl.bc = Dict() : delete!(cl.bc, base);

bases(cl::Classifier002)::AbstractVector{AbstractString} = collect(keys(cl.bc))

ohlcv(cl::Classifier002, base::AbstractString) = cl.bc[base].ohlcv

function timerangecut!(cl::Classifier002, startdt::DateTime, enddt::DateTime)
    for bc in values(cl.bc)
        ohlcvstartdt = startdt - Minute(Classify.requiredminutes(cl)-1)
        Ohlcv.timerangecut!(bc.ohlcv, ohlcvstartdt, enddt)
    end
end

requiredminutes(cl::Classifier002)::Integer = 2 * max(cl.trendwindow, maximum(cl.window))

function buysplitparts(cl::Classifier002, base::AbstractString)
    bc = cl.bc[base]
    return floor(Int, cl.window[bc.meanwindowix] / 2)
end

"returns a tuple of (relative price of 2 consecutive mean pivot price windows, current mean pivot price, last mean pivot price)"
function relativemeanchange(cl::Classifier002, piv, oix, win)
    meanpiv = mean(piv[(oix-win+1):oix])
    if oix-2*win+1 < 0
        lastmeanpiv = meanpiv
    else
        lastmeanpiv = mean(piv[(oix-2*win+1):(oix-win+1)])
    end
    return ((meanpiv - lastmeanpiv) / lastmeanpiv), meanpiv, lastmeanpiv
end

function positiverelativetrendwindowchange(cl::Classifier002, bc::BaseClassifier002, piv, oix)
    if cl.trendwindow > meanwindow(cl, bc)
        return relativemeanchange(cl, piv, oix, cl.trendwindow)[1] > 0f0
    else
        return true
    end
end

relativelastprice(cl::Classifier002, piv, oix, meanpiv) = (piv[oix] - meanpiv) / meanpiv
meanwindow(cl::Classifier002, bc::BaseClassifier002, ix=bc.meanwindowix) = cl.window[ix]

function meanchangeperday(cl::Classifier002, bc::BaseClassifier002, otime, oix, relmeanchange)
    rpcpd = relmeanchange * 24f0 * 60f0 * Float32((otime[oix] - otime[oix-meanwindow(cl, bc)]) / Minute(1)) # relative pivot price change per day
    return rpcpd
end

function largerwindow(cl::Classifier002, bc::BaseClassifier002, piv, oix)
    if bc.meanwindowix < lastindex(cl.window)
        mwix = bc.meanwindowix + 1
        meanpiv = mean(piv[(oix - 2 * meanwindow(cl, bc, mwix)):(oix - meanwindow(cl, bc, mwix))])
        if abs(relativelastprice(cl, piv, oix, meanpiv)) < cl.rbt
            return mwix
        end
    end
    return bc.meanwindowix
end

smallerwindow(cl::Classifier002, bc::BaseClassifier002) = bc.meanwindowix > firstindex(cl.window) ? bc.meanwindowix - 1 : bc.meanwindowix

function advice(cl::Classifier002, base::AbstractString, dt::DateTime)::InvestProposal
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier002"
        return noop
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix)
end

function advice(cl::Classifier002, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix)::InvestProposal
    base = ohlcv.base
    otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
    if ohlcvix < requiredminutes(cl)
        return noop
    end
    bc = cl.bc[base]
    piv = Ohlcv.pivot!(ohlcv)
    relmeanchange, meanpiv, _ = relativemeanchange(cl, piv, ohlcvix, meanwindow(cl, bc))
    rlp = relativelastprice(cl, piv, ohlcvix, meanpiv)
    if rlp < 0
        if abs(rlp) >= cl.rbt
            bc.meanwindowix = smallerwindow(cl, bc)
        else
            bc.meanwindowix = largerwindow(cl, bc, piv, ohlcvix)
        end
        # (verbosity >= 3) && push!(cl.dbgdf, (opentime=otime[ohlcvix], piv=piv[ohlcvix], meanpiv=meanpiv, rlp=rlp, lpcok=false, mcpd=0f0, mcpdok=false, postrend=false))
        return sell
    elseif rlp >= cl.rbt
        bc.meanwindowix = smallerwindow(cl,bc)
        mcpd = meanchangeperday(cl, bc, otime, ohlcvix, relmeanchange)
        if (mcpd > cl.rbt)
            if positiverelativetrendwindowchange(cl, bc, piv, ohlcvix)
                (verbosity >= 3) && push!(cl.dbgdf, (opentime=otime[ohlcvix], piv=piv[ohlcvix], meanpiv=meanpiv, rlp=rlp, lpcok=true, mcpd=mcpd, mcpdok=true, postrend=true))
                return buy
            else
                (verbosity >= 3) && push!(cl.dbgdf, (opentime=otime[ohlcvix], piv=piv[ohlcvix], meanpiv=meanpiv, rlp=rlp, lpcok=true, mcpd=mcpd, mcpdok=true, postrend=false))
                return hold
            end
        else
            (verbosity >= 3) && push!(cl.dbgdf, (opentime=otime[ohlcvix], piv=piv[ohlcvix], meanpiv=meanpiv, rlp=rlp, lpcok=true, mcpd=mcpd, mcpdok=false, postrend=false))
            return hold
        end
    else # (cl.rbt / 2f0) > rlp >= 0
        # (verbosity >= 3) && push!(cl.dbgdf, (opentime=otime[ohlcvix], piv=piv[ohlcvix], meanpiv=meanpiv, rlp=rlp, lpcok=false, mcpd=0f0, mcpdok=false, postrend=false))
        bc.meanwindowix = largerwindow(cl, bc,piv, ohlcvix)
        return hold
    end
end


#endregion Classifier002

#region Classifier003

mutable struct BaseClassifier003
    ohlcv::Ohlcv.OhlcvData
    shortwindowix::Integer
    longwindowix::Integer
end

"""
Classification is based on MA (moving average) price change of a time range (=window) length that is determined by the largest window that can catch all relative noise price changes
Follows a long term mean and a short head mean: long term window = short term window * 16
short term adapts to catch noise amplitudes of rnt.
- close when
  - last price relative to long term mean > 0
  - short term mean change < 0f0
- buy when
  - last price relative to long term mean < -rbt
  - short term mean change > 0f0
  - long term mean relative to last mean per day > rnt
- one window change step per time (=per advice call)
  - change to next smaller window if abs(any price relative to short term mean) >= rbt
  - change to next larger window if abs(any price relative to short term mean) < rbt for that larger window
"""
mutable struct Classifier003 <: AbstractClassifier
    "bd: base date maps the base string onto ohlcv data"
    bc::Dict{AbstractString, BaseClassifier003}
    "window vector of considered mean window in minutes"
    shortwindow::AbstractVector{Integer}  # have to be sorted in increasing order
    longwindow::AbstractVector{Integer}  # have to be sorted in increasing order
    "rbt: relative buy threshold to long mean price => buy when below -rbt"
    rbt::Float32
    "srnt: relative noise threshold of short window mean price change, also used for long window meanchange per day as positive trend indicator"
    srnt::Float32
    "lrnt: relative noise threshold of long window mean price change"
    lrnt::Float32
    dbgdf
    function Classifier003(;shortwindow=[15, 30, 60, 2*60, 4*60], longwindow=[12*60, 24*60, 2*24*60, 4*24*60, 8*24*60], rbt=0.01f0, srnt=0.005f0, lrnt=0.03f0)
        shortwindow = sort(shortwindow)
        longwindow = sort(longwindow)
        return new(Dict(), shortwindow, longwindow, rbt, srnt, lrnt, DataFrame())
    end
end

function addbase!(cl::Classifier003, ohlcv::Ohlcv.OhlcvData)
    cl.bc[ohlcv.base] = BaseClassifier003(ohlcv, lastindex(cl.shortwindow), lastindex(cl.longwindow))
end

removebase!(cl::Classifier003, base::Union{Nothing, AbstractString}) = isnothing(base) ? cl.bc = Dict() : delete!(cl.bc, base);

bases(cl::Classifier003)::AbstractVector{AbstractString} = collect(keys(cl.bc))

ohlcv(cl::Classifier003, base::AbstractString) = cl.bc[base].ohlcv

function timerangecut!(cl::Classifier003, startdt::DateTime, enddt::DateTime)
    for bc in values(cl.bc)
        ohlcvstartdt = startdt - Minute(Classify.requiredminutes(cl)-1)
        Ohlcv.timerangecut!(bc.ohlcv, ohlcvstartdt, enddt)
    end
end

requiredminutes(cl::Classifier003)::Integer = max(maximum(cl.shortwindow), maximum(cl.longwindow)) * 2

function buysplitparts(cl::Classifier003, base::AbstractString)
    bc = cl.bc[base]
    return floor(Int, cl.shortwindow[bc.shortwindowix] / 2)
end


"""Returns a tuple (windowix, meanpivot, relativebreakoutprice) of the largest window index with all noise in +-cl.rnt bounds
or - in case of a breakout - the relative break out pivotprice , which is otherwise `nothing`"""
function noiseoutofbounds(cl::Classifier003, piv, oix, meanpiv, win, rt)
    for ix in oix-win:oix
        if relativelastprice(cl, piv, ix, meanpiv) > rt
            return true
        end
    end
    return false
end

"returns a tuple of (relative price of 2 consecutive mean pivot price windows, current mean pivot price, last mean pivot price)"
function relativemeanchange(cl::Classifier003, piv, oix, win)
    meanpiv = mean(piv[(oix-win+1):oix])
    lastmeanpiv = mean(piv[(oix-2*win+1):(oix-win+1)])
    return ((meanpiv - lastmeanpiv) / lastmeanpiv), meanpiv, lastmeanpiv
end

relativelastprice(cl::Classifier003, piv, oix, meanpiv) = (piv[oix] - meanpiv) / meanpiv
meanwindow(cl::Classifier003, bc::BaseClassifier003, ix=bc.shortwindowix) = cl.shortwindow[ix]

function meanchangeperday(cl::Classifier003, bc::BaseClassifier003, otime, oix, relmeanchange)
    rpcpd = relmeanchange * 24f0 * 60f0 * Float32((otime[oix] - otime[oix-cl.longwindow[bc.longwindowix]]) / Minute(1)) # relative pivot price change per day
    return rpcpd
end

function adjustwindow(cl::Classifier003, piv, oix, winmean, windows, wix, rt)
    if noiseoutofbounds(cl, piv, oix, winmean, windows[wix], rt)
        # shorten window to catch noise range
        newwix = wix > firstindex(windows) ? wix - 1 : wix
        return newwix
    else
        # enlarge window if noise still can be caught
        newwix = wix < lastindex(windows) ? wix + 1 : wix
        if (newwix != wix)
            newwin = windows[newwix]
            newmean = mean(piv[(oix-newwin):oix])
            if !noiseoutofbounds(cl, piv, oix, newmean, newwin, rt)
                return newwix
            end
        end
    end
    return wix
end

function advice(cl::Classifier003, base::AbstractString, dt::DateTime)::InvestProposal
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier003"
        return noop
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix)
end

function advice(cl::Classifier003, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix)::InvestProposal
    base = ohlcv.base
    otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
    if ohlcvix < requiredminutes(cl)
        # (verbosity >= 3) && println("$(otime[ohlcvix]) $base ($(ohlcv.df[begin,:opentime])-$(ohlcv.df[end,:opentime])) ohlcvix=$ohlcvix < requiredminutes(Classifier003)=$(requiredminutes(cl)) => noop")
        return noop
    end
    bc = cl.bc[base]
    piv = Ohlcv.pivot!(ohlcv)
    otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
    shortmeanchange, shortmean, _ = relativemeanchange(cl, piv, ohlcvix, cl.shortwindow[bc.shortwindowix])
    longmeanchange, longmean, _ = relativemeanchange(cl, piv, ohlcvix, cl.longwindow[bc.longwindowix])
    rlp = relativelastprice(cl, piv, ohlcvix, longmean)
    adjustwindow(cl, piv, ohlcvix, shortmean, cl.shortwindow, bc.shortwindowix, cl.srnt)
    adjustwindow(cl, piv, ohlcvix, shortmean, cl.longwindow, bc.longwindowix, cl.lrnt)
    if rlp > 0 # sell above long window average
        if shortmeanchange < 0f0 # sell only if short window average is already falling assuming we are close to peak
            return sell
        else
            return hold
        end
    elseif rlp <= -cl.rbt # buy at local dip
        mcpd = meanchangeperday(cl, bc, otime, ohlcvix, longmeanchange)
        if (mcpd > cl.srnt) # positive trend
            if shortmeanchange > 0f0 # positive short term average change
                return buy
            else
                return hold
            end
        else
            return hold
        end
    else # -cl.rbt < rlp <= 0
        return hold
    end
end

#endregion Classifier003


#region Classifier004

"""
Classifier004 idea
- don't use a reference (like a mean or regression line) but look back over various time ranges and determine high and lows
- buy around the anticipated low, sell at anticipated high with some head room in case the high is not reached.
- goal is frequent trading with small wins, which makes losses acceptable as long as the significant majority of trades is positve

specific
- choose a time window and look at 3 or 4 concatenations
- for each of these windows identify maximum and minimum pivot peak
- take the smallest as indication of expected upward amplitude
- take smallest max to min amplitude as expecation to estimate next minimum
- buy at expected minimum if expected upward amplitude > gainthreshold (gth)
- sell at expected maximum
"""

mutable struct Amplitudes
    "mean amplitude over the last `lookback` time range windows that overlap each other by half"
    meanamp::Float32
    "mean high change over the last `lookback` time range windows that overlap each other by half"
    meanhighchange::Float32
    "mean low change over the last `lookback` time range windows that overlap each other by half"
    meanlowchange::Float32
    "relative highest pivot change within current time range window compared window before"
    highchange::Float32
    lowchange::Float32
    "highest pivot within current and most recent before current time range window"
    lasthigh::Float32
    high::Float32
    "lowest pivot within current and most recent before current time range window"
    lastlow::Float32
    low::Float32
    cfg::DataFrame
    Amplitudes() = new(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, DataFrame())
end

mutable struct BaseClassifier004
    ohlcv::Ohlcv.OhlcvData
    "Dict of Amplitudes per window"
    amplitudes::Dict{Integer, Amplitudes}
    BaseClassifier004(ohlcv::Ohlcv.OhlcvData) = new(ohlcv, Dict())
end

"Classification is based on MA (moving average) price change of a time range (=window) length that is determined by the largest window that can catch all relative noise price changes"
mutable struct Classifier004 <: AbstractClassifier
    "bd: base date maps the base string onto ohlcv data"
    bc::Dict{AbstractString, BaseClassifier004}
    "window vector of considered timerange windows in minutes"
    window::AbstractVector{Integer}  # have to be sorted in increasing order
    cfg::Union{Nothing, DataFrame}  # configurations
    "cfgid: id to retrieve configuration parameters of Classifier004; uninitialized == 0"
    cfgid::Int16
    "anchor DateTime to calculate window extremes only every half window"
    anchor::Union{Nothing, DateTime} # nothing if not yet set
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    function Classifier004(;window=WINDOWSET)
        window = sort(window)
        cl = new(Dict(), window, nothing, 0, nothing, nothing)
        readconfigurations!(cl)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

const WINDOWSET = [0, 15, 30, 60, 2*60, 4*60]         # 0 = dynamic use of all window sizes
const LOOKBACKSET = [5, 10, 20, 40]
const BREAKOUTFACTORSET = [0.5f0, 1.0f0, 2.0f0, 10000.0f0]      # 10000.0 == de facto switched off
const TRENDFACTORSET = [0.015f0, 0.03f0, 10000.0f0]             # 10000.0 == de facto switched off
const GAINTHRESHOLDSET = [0.01f0, 0.02f0, 0.04f0, 10000.0f0]    # 10000.0 == de facto switched off

EnvConfig.emptyconfigdf(cl::Classifier004) = DataFrame(
    cfgid=Int16[],               # classifier identificator
    window=Int16[],             # window for amplitude measurement in minutes
    lookback=Int16[],           # number of windows (in half window steps) back to calculate mean values
    breakoutfactor=Float32[],   # highchange / meanchange to consider it as breakout worth strongbuy
    trendfactor=Float32[],      # gain threshold of highchange and lowchange to consider trend worth buy
    gainthreshold=Float32[]     # gain threshold ratio of minimum gain/price of mean amplitude to consider volatility trading
)


function addbase!(cl::Classifier004, ohlcv::Ohlcv.OhlcvData)
    cl.bc[ohlcv.base] = BaseClassifier004(ohlcv)
end

function timerangecut!(cl::Classifier004, startdt::DateTime, enddt::DateTime)
    for bc in values(cl.bc)
        ohlcvstartdt = startdt - Minute(Classify.requiredminutes(cl)-1)
        Ohlcv.timerangecut!(bc.ohlcv, ohlcvstartdt, enddt)
    end
end

requiredminutes(cl::Classifier004)::Integer =  cl.cfgid == 0 ? maximum(cl.window) * 20 : (cl.cfg[cl.cfgid, :lookback] + 1) * floor(Int, maximum(cl.window) / 2)

"(new - base) / base"
reldiff(new, base) = (new - base) / abs(base)

"Updates `BaseClassifier004` `amplitudes` for every `cl.window` elements and their corresponding `Amplitudes`"
function updateamplitudes!(cl::Classifier004, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix::Integer)
    otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
    piv = Ohlcv.pivot!(ohlcv)
    bc = cl.bc[ohlcv.base]
    lookback = cl.cfg[cl.cfgid, :lookback]
    windows = cl.cfg[cl.cfgid, :window] == 0 ? cl.window : [cl.cfg[cl.cfgid, :window]]
    for win in windows
        if win == 0 continue end
        if isnothing(cl.anchor)
            bc.amplitudes[win] = Amplitudes()
            for ix in lookback:-1:1
                endix = ohlcvix - (ix - 1) * floor(Int, win / 2)
                startix = endix - win + 1
                high = maximum(piv[startix:endix])
                low = minimum(piv[startix:endix])
                bc.amplitudes[win].meanamp += reldiff(high, low)
                if ix < lookback
                    bc.amplitudes[win].meanhighchange += reldiff(high, bc.amplitudes[win].high)
                    bc.amplitudes[win].meanlowchange += reldiff(low, bc.amplitudes[win].low)
                end
                bc.amplitudes[win].lasthigh = bc.amplitudes[win].high
                bc.amplitudes[win].lastlow = bc.amplitudes[win].low
                bc.amplitudes[win].high = high
                bc.amplitudes[win].low = low
                bc.amplitudes[win].highchange = reldiff(bc.amplitudes[win].high, bc.amplitudes[win].lasthigh)
                bc.amplitudes[win].lowchange = reldiff(bc.amplitudes[win].low, bc.amplitudes[win].lastlow)
            end
            bc.amplitudes[win].meanamp = bc.amplitudes[win].meanamp / lookback
            bc.amplitudes[win].meanhighchange = bc.amplitudes[win].meanhighchange / (lookback - 1)
            bc.amplitudes[win].meanlowchange = bc.amplitudes[win].meanlowchange / (lookback - 1)
        else
            endix = ohlcvix
            startix = endix - win + 1
            high = maximum(piv[startix:endix])
            low = minimum(piv[startix:endix])
            if floor(Int, ((otime[ohlcvix] - cl.anchor) / Minute(1)) % floor(Int, win / 2)) == 0
                bc.amplitudes[win].meanamp = (bc.amplitudes[win].meanamp * (lookback - 1) + reldiff(bc.amplitudes[win].high, bc.amplitudes[win].low)) / lookback
                bc.amplitudes[win].meanhighchange = (bc.amplitudes[win].meanhighchange * (lookback - 2) + reldiff(high, bc.amplitudes[win].high)) / (lookback - 1)
                bc.amplitudes[win].meanlowchange = (bc.amplitudes[win].meanlowchange * (lookback - 2) + reldiff(low, bc.amplitudes[win].low)) / (lookback - 1)
                bc.amplitudes[win].lasthigh = bc.amplitudes[win].high
                bc.amplitudes[win].lastlow = bc.amplitudes[win].low
            end
            bc.amplitudes[win].high = high
            bc.amplitudes[win].low = low
            bc.amplitudes[win].highchange = reldiff(bc.amplitudes[win].high, bc.amplitudes[win].lasthigh)
            bc.amplitudes[win].lowchange = reldiff(bc.amplitudes[win].low, bc.amplitudes[win].lastlow)
    end
    end
    if isnothing(cl.anchor) cl.anchor = otime[ohlcvix] end
end

function advice(cl::Classifier004, base::AbstractString, dt::DateTime)::InvestProposal
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier004"
        return noop
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix)
end

function advice(cl::Classifier004, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix)::InvestProposal
    base = ohlcv.base
    if ohlcvix < requiredminutes(cl)
        return noop
    end
    updateamplitudes!(cl, ohlcv, ohlcvix)
    bc = cl.bc[base]
    piv = Ohlcv.pivot!(ohlcv)
    windows = cl.cfg[cl.cfgid, :window] == 0 ? cl.window : [cl.cfg[cl.cfgid, :window]]
    for win in windows # from short to long windows
        if win == 0 continue end
        # first check breakout
        if reldiff(bc.amplitudes[win].highchange, bc.amplitudes[win].meanhighchange) >= cl.cfg[cl.cfgid, :breakoutfactor] # rel change of 2 rel chanegs
            return strongbuy
        elseif reldiff(bc.amplitudes[win].lowchange, bc.amplitudes[win].meanlowchange) >= -cl.cfg[cl.cfgid, :breakoutfactor] # rel change of 2 rel chanegs
            return strongsell
        # then check trend
        elseif (bc.amplitudes[win].meanhighchange >= cl.cfg[cl.cfgid, :trendfactor]) && (bc.amplitudes[win].highchange >= bc.amplitudes[win].meanhighchange)
            return buy
        elseif (bc.amplitudes[win].meanlowchange <= -cl.cfg[cl.cfgid, :trendfactor]) && (bc.amplitudes[win].highchange <= bc.amplitudes[win].meanhighchange)
            return sell
        # then check volatility trade
        elseif bc.amplitudes[win].meanamp >= cl.cfg[cl.cfgid, :gainthreshold]
            if abs(reldiff(piv[ohlcvix], bc.amplitudes[win].low)) <= 0.2 * bc.amplitudes[win].meanamp
                return buy
            elseif abs(reldiff(piv[ohlcvix], bc.amplitudes[win].high)) <= 0.2 * bc.amplitudes[win].meanamp
                return sell
            end
        end
    end
    return hold
end

function evaluate!(df::AbstractDataFrame, cl::Classifier004, ohlcv::Ohlcv.OhlcvData)
    for window in WINDOWSET
        for lookback in LOOKBACKSET
            for trendfactor in TRENDFACTORSET
                for breakoutfactor in BREAKOUTFACTORSET
                    for gainthreshold in STDGAINTHRSHLDSET
                        cl.cfgid = configurationid(cl, (window=window, lookback=lookback, breakoutfactor=breakoutfactor, trendfactor=trendfactor, gainthreshold=gainthreshold))
                        logsim!(df, cl, ohlcv)
                    end
                end
            end
        end
    end
end

configurationid4base(cl::Classifier004, base::AbstractString)::Integer = cl.cfgid


#endregion Classifier004

#region Classifier005

mutable struct BaseClassifier005
    ohlcv::Ohlcv.OhlcvData
    f4::Union{Nothing, Features.Features004}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier005(ohlcv::Ohlcv.OhlcvData, f4=Features.Features004(ohlcv, usecache=true))
        cl = isnothing(f4) ? nothing : new(ohlcv, f4, 0)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier005)
    println(io, "BaseClassifier005[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f4=$(!isnothing(bc.f4)), cfgid=$(bc.cfgid)")
end

function timerangecut!(bc::BaseClassifier005, startdt::DateTime, enddt::DateTime)
    if !isnothing(bc.f4)
        ohlcvstartdt = startdt - Minute(Classify.requiredminutes(bc)-1)
        Ohlcv.timerangecut!(bc.ohlcv, ohlcvstartdt, enddt)
        Features.timerangecut!(bc.f4, startdt, enddt)
        Features.f4offset!(bc.f4, bc.ohlcv)
    end
end

function writetargetsfeatures(bc::BaseClassifier005)
    Ohlcv.write(bc.ohlcv)
    if !isnothing(bc.f4)
        Features.write(bc.f4)
    end
end

supplement!(bc::BaseClassifier005) = Features.supplement!(bc.f4, bc.ohlcv)


"""
Classifier005 idea
- use regression to identify downward peaks buy at downward peak if regression is positive and trend is positive
- sell if gain is realized or when trend turns too much negative

specific
- check down peak against a set of regression windows
  - shorter windows can follow a breakout
- specify buythreshold of relative loss compared to regression line for buy trigger
  - buy when price comes back from below the buythreshold
- specify gainthreshold to take profit
- specify trendwindow of trend regression line with buygrad in %/d to enable buy and selgrad in %/d to close open positions
"""
mutable struct Classifier005 <: AbstractClassifier
    "bd: base date maps the base string onto ohlcv data"
    bc::Dict{AbstractString, BaseClassifier005}
    cfg::Union{Nothing, DataFrame}  # configurations
    "cfgid: id to retrieve configuration parameters of Classifier005; uninitialized == 0"
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    function Classifier005()
        cl = new(Dict(), DataFrame(), nothing)
        readconfigurations!(cl)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

const REGRWINDOWSET005 = [24*60, 3*24*60]
const TRENDWINDOWSET005 = [10*24*60]
const TAKEPROFITGAINSET005 = [0.02f0, 0.04f0]
const BUYTHRESHOLDSET005 = [0.01f0, 0.02f0]
const TRENDBUYGRADSET005 = [0.01f0] # -1f0 = equal to switched off
const TRENDSELLGRADSET005 = [-0.03f0, -0.06f0] # -1f0 = equal to switched off
const BUYSPLITPARTSSET005 = [10]
const SELLVOLUMEFACTORET005 = [1]
const TRADEGAPMINUTES005 = [10]

EnvConfig.emptyconfigdf(cl::Classifier005) = DataFrame(
    cfgid=Int16[],              # classifier identificator
    regrwindow=Int16[],         # regression window as reference to detect downward peaks
    trendwindow=Int16[],        # regression window to identify trend gradient; 0= don't use trend info
    takeprofitgain=Float32[],   # take profit gain threshold ratio to meet or exceed before selling
    buythreshold=Float32[],     # buy threshold ratio below regression window; price to enter from below before buying
    trendbuygrad=Float32[],     # relative price change/day; buy only if trend regression line >= trendbuygrad
    trendsellgrad=Float32[],    # relative price change/day; sell as loss limit if trend regression line <= trendsellgrad
    buysplitparts=Int16[],
    sellvolumefactor=Int16[],
    tradegapminutes=Int16[]
)


function addbase!(cl::Classifier005, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier005(ohlcv)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier005)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier005)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

function timerangecut!(cl::Classifier005, startdt::DateTime, enddt::DateTime)
    for bc in values(cl.bc)
        timerangecut!(bc, startdt, enddt)
    end
end

# requiredminutes(cl::Classifier005)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier005)::Integer =  maximum(Features.regressionwindows004)

function buysplitparts(cl::Classifier005, base::AbstractString)
    return configuration(cl, configurationid4base(cl, base)).buysplitparts
end

function tradegapminutes(cl::Classifier005, base::AbstractString)
    return configuration(cl, configurationid4base(cl, base)).tradegapminutes
end

function sellvolumefactor(cl::Classifier005, base::AbstractString)
    return configuration(cl, configurationid4base(cl, base)).sellvolumefactor
end

function takeprofitgain(cl::Classifier005, base::AbstractString)
    return configuration(cl, configurationid4base(cl, base)).takeprofitgain
end

function advice(cl::Classifier005, base::AbstractString, dt::DateTime)::InvestProposal
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier005"
        return noop
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix)
end

function advice(cl::Classifier005, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix)::InvestProposal
    if ohlcvix < requiredminutes(cl)
        return noop
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.f4ix(bc.f4, ohlcvix)
    regry = Features.regry(bc.f4, cfg.regrwindow)[fix]
    buyprice = regry * (1 - cfg.buythreshold)
    trenddaygain = cfg.trendwindow == 0 ? Inf32 : Features.relativedaygain(bc.f4, cfg.trendwindow, fix)
    if (piv[ohlcvix] >= buyprice) && (piv[ohlcvix-1] < buyprice) # price swings back to regression after peak down
        if cfg.trendbuygrad <= trenddaygain
            return buy
        end
    end
    if (cfg.trendwindow > 0) && (cfg.trendsellgrad >= trenddaygain)
        return strongsell  # stop loss
    end
    return hold
end

function evaluate!(df::AbstractDataFrame, cl::Classifier005, ohlcv::Ohlcv.OhlcvData)
    (verbosity >=2) && println("$(EnvConfig.now()): evaluating Classifier005")
    addbase!(cl, ohlcv)
    if !(ohlcv.base in keys(cl.bc))
        @error "addbase! failed"
        return df
    end
    bc = cl.bc[ohlcv.base]
    for regrwindow in REGRWINDOWSET005
        for trendwindow in TRENDWINDOWSET005
            for takeprofitgain in TAKEPROFITGAINSET005
                for buythreshold in BUYTHRESHOLDSET005
                    for buysplitparts in BUYSPLITPARTSSET005
                        for sellvolumefactor in SELLVOLUMEFACTORET005
                            for tradegapminutes in TRADEGAPMINUTES005
                                for trendbuygrad in TRENDBUYGRADSET005
                                    for trendsellgrad in TRENDSELLGRADSET005
                                        if (trendwindow == 0) && !((trendbuygrad == first(TRENDBUYGRADSET005)) && (trendsellgrad == first(TRENDSELLGRADSET005)))
                                            # execute only once because trendbuygrad and trendsellgrad are ignored with trendwindow == 0
                                            continue
                                        end

                                        cfgid = configurationid(cl, (regrwindow=regrwindow, trendwindow=trendwindow, takeprofitgain=takeprofitgain, buythreshold=buythreshold, trendbuygrad=trendbuygrad, trendsellgrad=trendsellgrad, buysplitparts=buysplitparts, sellvolumefactor=sellvolumefactor, tradegapminutes=tradegapminutes))
                                        configureclassifier!(cl, ohlcv.base, cfgid)
                                        (verbosity >=2) && println("$(EnvConfig.now()): evaluating Classifier005 config $(NamedTuple(configuration(cl, cfgid)))")
                                        logsim!(df, cl, ohlcv)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

configurationid4base(cl::Classifier005, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier005, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
    else
        @error "cannot find $base in Classifier005"
    end
end

#endregion Classifier005

end  # module
