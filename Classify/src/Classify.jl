"""
Train and evaluate the trading signal classifiers
"""
module Classify

using CSV, DataFrames, Logging, Dates
using BSON, JDF, Flux, Statistics, ProgressMeter, StatisticalMeasures, MLUtils
using CategoricalArrays
using CategoricalDistributions
using Distributions
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv, CryptoXch

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
"""
- `addbase!` adds bases you want to work with, get a list of them with bases or remove them with `removebase!`
- save any parameters for features, targets, classifiers as a `EnvConfig.AbstractConfiguration` and
  apply to to a classifier/base combination with `configureclassifier!` (retrieve the applied configuration id with `configurationid4base`)
- use ohlcv to calculate the required features by implementing `supplement!`, then save them with `writetargetsfeatures`
- features and targets can only be used after the initial `requiredminutes`
- train the classfier by implementing `train!`
    - uses `setpartitions` with a default split into learn, eval and test sets
- in order to supplement new feature/target data of an incoming data stream implement `supplement!` like for the initial calculation as mentioned above
- `advice` is implementing the classification verdict
- a classifier can be simulation tested over a set of configuration given in the classifier property `optparams` by using
    - `evaluate!` for a single classifier and the to be evaluated ohlcv data
    - `evaluateclassifiers` for a set of classifiers and a set of bases on a given time range
    - implement a binary answer of `configureclassifier!` to skip specific configurations
    - simulation results are saved by `writesimulation` and can be retrieved by `readsimulation` into a `DataFrame`
    - from the data frame of all simulations the simulation of a specific classifier can be isolated by `kpioverview`
"""
abstract type AbstractClassifier <: AbstractConfiguration end

mutable struct TradeAdvice
    classifier::AbstractClassifier
    configid
    tradelabel::TradeLabel
    relativeamount  # relative investment amount to be spent from the maximum amount considered for classifier/base combination
    base
    price  # limit price of this trade advice
    datetime  # exchange datetime (== opentime of OHLCV) of this trade advice 
    hourlygain  # used as criteria to give coin investments priority
    probability  # 0 <= probability <= 1; used to calculate risk as probability * hourlygain
    investmentid  # nothing until filled by Trade based on a transaction
end

"""
classifier usage approach: classifier that serves one or more bases but can be trained with arbitrary bases
- preparation
    - specify features
    - specify targets
    - specify set partitions
    - specify NN
    - add bases including logic what part to use, e.g. #* --> Ohlcv.volumeohlcv as filter for logic below
        - add only parts with a minimal daily volume #* done
        - stop when volume is 50% of minimal daily volume #* done
        - consecutive minimum time range = 10 * gap * #partitions #* done

- apply in arbitrary order with arbitrary base coins
    - training
    - evaluation
    - test
- evaluate classification result
"""

"Adds ohlcv base with all ohlcv data, which can span the last requiredminutes or the time range of a back test / training.
The required feature/target will be supplemented. "
function addbase!(cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData) end

"Removes base and all associated data from cl. If base === nothing then all bases are removed."
removebase!(cl::AbstractClassifier, base::Union{Nothing, AbstractString}) = (isnothing(base) && hasproperty(cl, :bc) ? cl.bc = Dict() : delete!(cl.bc, base); cl)

"Returns all successfully added bases as Vector of Strings."
bases(cl::AbstractClassifier)::AbstractVector{} = hasproperty(cl, :bc) ? collect(keys(cl.bc)) : []

"Returns the OhlcvData of base that was previously added to cl."
ohlcv(cl::AbstractClassifier, base::AbstractString) = hasproperty(cl, :bc) ? (hasproperty(cl.bc[base], :ohlcv) ? cl.bc[base].ohlcv  : nothing) : nothing

""" Supplements all with Ohlcv and feature/target data of bases to match with base ohlcv data.
In a trade loop ohlcv data is updated first and then feature/target data is supplement with this function.
This to avoid a Classifier dependency to CryptoXch. """
function supplement!(cl::AbstractClassifier) end

"""
Write all ohlcv, feature and target data (but not ohlcv) of all bases to canned storage, which overwrites previous canned data.
The default implementation only write ohlcv.
"""
function writetargetsfeatures(cl::AbstractClassifier)
    if hasproperty(cl, :bc)
        for bc in values(cl.bc)
            if hasproperty(bc, :ohlcv)
                # Ohlcv.write(bc.ohlcv)  --> don't write ohlcv within writetargetsfeatures because then it is only written when features are valid
                # ohlcv should also written when data length is too short to calculate features
            end
        end
    end
end

"Returns the required minutes of data history that is required for that classifier to work properly."
function requiredminutes(cl::AbstractClassifier)::Integer
    @error "missing $(typeof(cl)) implementation"
    return 0
end

"fee to be considered in classifier backtesting"
relativefee(cl::AbstractClassifier, base::AbstractString) = 0.08 / 100  # Bybit VIP1 taker fee: 0.08%

"Returns a trading advice for the specified time either for a running investment or the basecoin (investment=nothing). Will return a ignore in case of insufficient associated base data."
function advice(cl::AbstractClassifier, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    @error "missing $(typeof(cl)) implementation"
    return nothing
end

function advice(cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    @error "missing $(typeof(cl)) implementation"
    return nothing
end

"Returns a Dict of setname => vector of row ranges"
function setpartitions(cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData)
    len = length(Ohlcv.dataframe(ohlcv).pivot)
    return setpartitions(1:len, Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/80)
end

"Trains the classifier based on previosly provided data and stores the adapted classifier."
function train!(cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData)
    #! requires the definition of train, eval, test sets - that may be a configuration of target/features combi including requiredminutes of its own?
    @error "missing $(typeof(cl)) implementation"
end

function configurationid4base(cl::AbstractClassifier, base::AbstractString)::Integer
    if hasproperty(cl, :bc) && isa(cl.bc, Dict)
        return base in keys(cl.bc) ? cl.bc[base].cfgid : 0
    end
    @error "missing $(typeof(cl)) implementation" cl_has_bc=hasproperty(cl, :bc) bc_as_Dict=isa(cl.bc, Dict) cl_type=typeof(cl)
end

"Configure classifer according to configuration identifier. Returns true in case of a valid config or false otherwise"
function configureclassifier!(cl::AbstractClassifier, base::AbstractString, configid::Integer)
    if hasproperty(cl, :cfg)
        if configid in cl.cfg[!, :cfgid]
            if hasproperty(cl, :bc) && isa(cl.bc, Dict)
                if (base in keys(cl.bc)) && hasproperty(cl.bc[base], :cfgid)
                    cl.bc[base].cfgid = configid
                    return true
                else
                    (verbosity >= 3) && println("false because base in keys(cl.bc) = $(base in keys(cl.bc)) or hasproperty(cl.bc[base], :cfgid)=$(hasproperty(cl.bc[base], :cfgid)) in $(typeof(cl))")
                    return false
                end
            else
                (verbosity >= 3) && println("false because hasproperty(cl, :bc) = $(hasproperty(cl, :bc)) or isa(cl.bc, Dict)=$(isa(cl.bc, Dict)) in $(typeof(cl))")
                return false
            end
        else
            (verbosity >= 3) && println("false because configid in cl.cfg[!, :cfgid] = $(configid in cl.cfg[!, :cfgid]) in $(typeof(cl))")
            return false
        end
    end
    @error "missing $(typeof(cl)) implementation and found no cfg property in $(typeof(cl))"
    return false
end

"Iterates over ohlcv and logs simulation results in df"
function logsim!(df::AbstractDataFrame, cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData)
    otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
    piv = Ohlcv.dataframe(ohlcv)[!, :pivot]
    cfgid = configurationid4base(cl, ohlcv.base)
    openorder = nothing
    rows = 0

    function pushlongsim(tradepair, transactionfee=relativefee(cl, ohlcv.base))
        #TODO add lendingfee(cl, base, startdt, enddt)
        @assert tradepair.buydt <= tradepair.selldt "buydt=$buydt > selldt=$selldt"
        gain = fee = nothing
        if tradepair.trade == "long"
            buyprice = tradepair.buyprice * (1 + transactionfee)
            sellprice = tradepair.sellprice * (1 - transactionfee)
            gain = (sellprice - buyprice)/buyprice
        else
            buyprice = tradepair.buyprice * (1 - transactionfee)
            sellprice = tradepair.sellprice * (1 + transactionfee)
            gain = (buyprice - sellprice)/buyprice
        end
        fee = (tradepair.buyprice + tradepair.sellprice) * transactionfee # * amount of invest == implicit 1 USDT
        # println(tradepair)
        push!(df, (classifier=string(typeof(cl)), basecoin=ohlcv.base, cfgid=cfgid, buydt=tradepair.buydt, selldt=tradepair.selldt, minutes=Minute(tradepair.selldt-tradepair.buydt).value, buyprice=tradepair.buyprice, sellprice=tradepair.sellprice, gain=gain, trade=tradepair.trade, fee=fee))
        rows += 1
        tradepair.ta.investmentid = rows
    end

    # pushlongsim(otime[begin], piv[begin], otime[begin], piv[begin], "dummy", 0f0) # dummy to avoid empty dataframe
    for ix in eachindex(otime)
        Ohlcv.setix!(ohlcv, ix)
        ta = advice(cl, ohlcv, investment=(isnothing(openorder) ? nothing : openorder.ta))
        if !isnothing(ta)
            if  (ta.tradelabel in [longbuy, longstrongbuy]) && isnothing(openorder)
                openorder = (buydt=otime[ix], buyprice=piv[ix], trade="long", ta=ta)
            end
            if (ta.tradelabel in [longclose, longstrongclose]) && !isnothing(openorder) && (openorder.trade == "long")
                pushlongsim((openorder..., selldt=otime[ix], sellprice=piv[ix]))
                openorder = nothing
            end

            if (ta.tradelabel in [shortbuy, shortstrongbuy]) && isnothing(openorder)
                openorder = (buydt=otime[ix], buyprice=piv[ix], trade="short", ta=ta)
            end
            if (ta.tradelabel in [shortclose, shortstrongclose]) && !isnothing(openorder) && (openorder.trade == "short")
                pushlongsim((openorder..., selldt=otime[ix], sellprice=piv[ix]))
                openorder = nothing
            end
        end
    end
    if !isnothing(openorder)
        pushlongsim((openorder..., selldt=otime[end], sellprice=piv[end]))
    end
    return rows
end

function _collectconfig!(df::AbstractDataFrame, cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData, opsymbols, config)
    if length(opsymbols) > 0
        opsym = pop!(opsymbols)
        for opval in values(cl.optparams[String(opsym)])
            config[opsym] = opval
            if length(opsymbols) > 0
                _collectconfig!(df, cl, ohlcv, copy(opsymbols), config)
            else
                cfgid = configurationid(cl, (;config...))
                if configureclassifier!(cl, ohlcv.base, cfgid)
                    rows = logsim!(df, cl, ohlcv)
                    otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
                    (verbosity > 2) && print("\r$(EnvConfig.now()): ran $(ohlcv.base) from $(otime[begin]) until $(otime[end]) with $rows investments by $(typeof(cl)) config $(NamedTuple(configuration(cl, cfgid)))")
                    (verbosity > 3) && println("$(EnvConfig.now()): ran $(ohlcv.base) from $(otime[begin]) until $(otime[end]) with $rows investments by $(typeof(cl)) config $(NamedTuple(configuration(cl, cfgid)))")
                end
            end
        end
    end
end

"""
Configures cl and evaluates cl by calling logsim! to log all trades in df.
A property optparams may contain a Dict with String keys as parameter name and a Vector of to be evaluated values.
"""
function evaluate!(df::AbstractDataFrame, cl::AbstractClassifier, ohlcv::Ohlcv.OhlcvData, startdt, enddt)
    # default implementation with no spcific configuration
    (verbosity >=2) && println("$(EnvConfig.now()): evaluating $(typeof(cl))")
    addbase!(cl, ohlcv)
    if !(ohlcv.base in keys(cl.bc))
        @error "addbase! failed"
        return
    end
    if hasproperty(cl, :optparams)
        opsymbols = [Symbol(k) for k in keys(cl.optparams)]
        _collectconfig!(df, cl, ohlcv, opsymbols, Dict())
    else
        logsim!(df, cl, ohlcv)
    end
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
- basecoins is a String vector of basecoin names
- startdt and enddt further constraints the timerange
"""
function evaluateclassifiers(classifiertypevector, basecoins, startdt, enddt)
    df = readsimulation()
    (verbosity >=3) && println("$(EnvConfig.now()): successfully read classifier simulation trades with $(size(df, 1)) entries")
    xc = CryptoXch.XchCache(startdt=startdt, enddt=enddt)
    CryptoXch.addbases!(xc, basecoins, startdt, enddt)
    for clt in classifiertypevector
        cl = clt()
        for base in basecoins
            ohlcv = CryptoXch.ohlcv(xc, base)
            # ohlcv = Ohlcv.defaultohlcv(row.basecoin)
            # Ohlcv.read!(ohlcv)
            otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
            sdt = isnothing(startdt) ? otime[begin] : floor(startdt, Minute(1))
            edt = isnothing(enddt) ? otime[end] : floor(enddt, Minute(1))
            if edt < sdt
                verbosity >= 1 && @warn "$(ohlcv.base) requested time range $startdt - $enddt out of liquid range $(row)"
            end
            if sdt <= edt
                Ohlcv.timerangecut!(ohlcv, sdt - Minute(requiredminutes(cl)+1), edt)
                addbase!(cl, ohlcv)
                if ohlcv.base in Classify.bases(cl)
                    (verbosity >=2) && println("$(EnvConfig.now()): evaluating classifier $(string(clt)) for $(ohlcv.base) from $sdt until $edt")
                    evaluate!(df, cl, ohlcv, sdt, edt)
                    writesimulation(df)
                else
                    verbosity >= 1 && @warn "evaluation of $(string(clt)) failed because classifier did not accept $ohlcv"
                end
            else
                verbosity >= 1 && @warn "evaluation of $(string(clt))/$(ohlcv.base) failed due to enddt=$edt < startdt=$sdt"
            end
        end
        # println("cl=$(typeof(cl)) cl.dgbdf=$(typeof(cl.dbgdf))")
        # println(describe(cl.dbgdf))
        # println("rw=(24h:$(count(cl.dbgdf[!, :rw] .== 24*60)), 12h:$(count(cl.dbgdf[!, :rw] .== 12*60)), 4h:$(count(cl.dbgdf[!, :rw] .== 4*60)))")    
    end
    return df
end

function kpioverview(df::AbstractDataFrame, classifiertype)
    if size(df, 1) > 0
        df = @view df[df[!, :classifier] .== string(classifiertype), :]
        gdf = groupby(df, [:basecoin, :cfgid])
        # gdf = groupby(df, [:classifier, :basecoin, :cfgid])
        rdf = combine(gdf, :gain => mean, :gain => median, :gain => sum, :gain => minimum, :gain => maximum, nrow, :minutes => mean, :minutes => median, :minutes => std, :trade => (x -> count(x .== "long")) => :long, :trade => (x -> count(x .== "short")) => :short, groupindices => :groupindex)
        cl = classifiertype()
        if hasproperty(cl, :cfg)
            leftjoin!(rdf, cl.cfg, on=:cfgid)
        end
        # println(rdf)
        rdffilename = EnvConfig.logpath(string(classifiertype) * "simulationresult.csv")
        EnvConfig.savebackup(rdffilename)
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
    labels::AbstractVector  # in fixed sequence as index == class id
    description
    mnemonic::String
    fileprefix::String  # filename without suffix
    predecessors::Vector{String}  # filename vector without suffix
    featuresdescription::String
    targetsdescription::String
    losses::Vector
    predictions::Vector{String}  # filename vector without suffix of predictions
end

function NN(model, optim, lossfunc, labels, description, mnemonic="", fileprefix="")
    return NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix, String[], "", "", [], String[])
end

isadapted(nn::NN) = length(nn.losses) > 0

function setmnemonic(nn::NN, mnemonic)
    nn.mnemonic = "NN" * (isnothing(mnemonic) ? "" : "$(mnemonic)")
    nn.fileprefix = mnemonic
end

function Base.show(io::IO, nn::NN)
    println(io, "NN: labels=$(nn.labels) #predecessors=$(length(nn.predecessors)) mnemonic=$(nn.mnemonic) fileprefix=$(nn.fileprefix) #losses=$(length(nn.losses))")
    println(io, "NN: featuresdecription=$(nn.featuresdescription) targetsdecription=$(nn.targetsdescription)")
    println(io, "NN: predecessors=$(nn.predecessors)")
    println(io, "NN: predictions=$(nn.predictions)")
    println(io, "NN: description=$(nn.description)")
end


#region DataPrep
mutable struct SetPartitions
    samplesets::Dict
    gapsize
    relativesubrangesize
end

# folds(data, nfolds) = partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...)

function setpartitions(rowrange, sp::SetPartitions)
    setpartitions(rowrange, sp.samplesets, sp.gapsize, sp.relativesubrangesize)
    @assert isapprox(sum(values(sp.samplesets)), 1, atol=0.001) "sum($(sp.samplesets))=$(sum(values(sp.samplesets))) != 1"
    @assert sp.relativesubrangesize <= minimum(values(sp.samplesets))
end

function _realpartitionsize(rowrange, samplesets, gapsize, partitionsize, minpartitionsize, maxpartitionsize)
    gapix = [ix for ix in eachindex(samplesets) if ix < lastindex(samplesets) ? !(samplesets[ix] == samplesets[(ix + 1)]) : !(samplesets[ix] == samplesets[1])]
    gapcount = length(gapix)
    remainder = length(rowrange) % (partitionsize * length(samplesets) + gapcount)
    ps = round(Int, remainder / length(samplesets))
    # ps = round(Int, (remainder - gapcount) / length(samplesets))
    if (length(rowrange) / (partitionsize * length(samplesets) + gapcount)) < 1
        res = max(ps - 1, minpartitionsize)
    else
        res = min(partitionsize + ps, maxpartitionsize)
    end
    (verbosity >= 3) && println("res=$res, rowrange=$rowrange, samplesets=$samplesets, ps=$ps, remainder=$remainder, gapcount=$gapcount, gapix=$gapix")
    return res
end

"""
    - Returns a Dict of setname => vector of row ranges
    - input
        - `rowrange` is the range of rows that are split in subranges, e.g. 2000:5000
        - `samplesets` is a string vector that denotes the set sequence comprising of samples of partitionsize, e.g. ["train", "test", "train", "train", "eval", "train"]
        - `gapsize` is the number of rows between partitions of different sets that are not included in any partition
        - `partitionsize` is the number of rows of the smallest allowed partition.
    - gaps will be removed from a subrange to avoid crosstalk bias between ranges
    - a mixture of ranges and individual indices in a vector can be unpacked into a index vector via `[ix for r in rr for ix in r]`
"""
function setpartitions(rowrange, samplesets::AbstractVector; gapsize=24*60, partitionsize=20*24*60, minpartitionsize=min(2*length(samplesets)*gapsize, partitionsize), maxpartitionsize=partitionsize*2)
    @assert length(samplesets) > 0 "length(samplesets)=$(length(samplesets))"
    @assert partitionsize > 0 "partitionsize=$(partitionsize)"
    # @assert length(rowrange) > (partitionsize * length(samplesets)) "length(rowrange)=$(length(rowrange)) > (partitionsize=$(partitionsize) * length(samplesets)=$(length(samplesets)))"
    @assert gapsize >= 0 "gapsize=$(gapsize)"
    #TODO gapsize not always equal (in ca 1% not - see ldf result of TrendDetector001.featurestargetsliquidranges!()) - fix is low prio
    if length(rowrange) < (partitionsize * length(samplesets))
        # rowrange too short 
        return []
    end
    sv = CategoricalArray(samplesets)
    psize = _realpartitionsize(rowrange, sv, gapsize, partitionsize, minpartitionsize, maxpartitionsize)
    rix = rowrange[begin]
    six = 1
    arr = []
    p = nothing
    while rix < rowrange[end]
        if !isnothing(p) && (p.setname != sv[six])
            push!(arr, p)
            p = nothing
            rix += gapsize
            if rix > rowrange[end]
                break
            end
        end
        if isnothing(p)
            p = (setname=sv[six], range=rix:min(rix+psize-1, rowrange[end]))  # new partition
        else
            p = (setname=p.setname, range=p.range[begin]:min(rix+psize-1, rowrange[end])) # extend partition range
        end
        rix += psize
        six = six % length(sv) + 1
    end
    if !isnothing(p)
        push!(arr, (p.setname, p.range))
    end
    return arr
    # res = Dict(sn => [] for sn in sv)
    # for p in arr
    #         res[p.setname] = push!(res[p.setname], p.range)
    # end
    # result = Dict(String(sn) => rv for (sn, rv) in res)
    # (verbosity >= 3) && println("$([kv for kv in result])")
    # return result
end

"""
    - Returns a Dict of setname => vector of row ranges
    - input
        - `rowrange` is the range of rows that are split in subranges, e.g. 2000:5000
        - `samplesets` is a Dict of setname => relative setsize with sum of all setsizes == 1
        - `gapsize` is the number of rows between partitions of different sets that are not included in any partition
        - `relativesubrangesize` is the minimum subrange size relative to `rowrange` of a consecutive range assigned to a setinterval
    - gaps will be removed from a subrange to avoid crosstalk bias between ranges
    - relativesubrangesize * length(rowrange) > 2 * gapsize
    - any relative setsize > 2 * relativesubrangesize
    - a mixture of ranges and individual indices in a vector can be unpacked into a index vector via `[ix for r in rr for ix in r]`
"""
function setpartitions(rowrange, samplesets::Dict, gapsize, relativesubrangesize) #* DEPRECATED  
    @assert isapprox(sum(values(samplesets)), 1, atol=0.001) "sum($(samplesets))=$(sum(values(samplesets))) != 1"
    @assert relativesubrangesize <= minimum(values(samplesets))
    rowstart = rowrange[1]
    rowend = rowrange[end]
    rows = rowend - rowstart + 1
    gapsize = relativesubrangesize * rows > 2 * gapsize ? gapsize : floor(Int, relativesubrangesize * rows / 3)
    (verbosity >= 3) && println("$(EnvConfig.now()) setpartitions rowrange=$rowrange, samplesets=$samplesets gapsize=$gapsize relativesubrangesize=$relativesubrangesize")
    @assert relativesubrangesize * rows > 2 * gapsize
    @assert max(collect(values(samplesets))...) > 2 * relativesubrangesize "max(collect(values(samplesets)=$(collect(values(samplesets))...))...)=$(max(collect(values(samplesets))...)) <= 2 * relativesubrangesize = $(2 * relativesubrangesize)"
    minrelsetsize = min(collect(values(samplesets))...)
    @assert minrelsetsize > 0.0
    sn = [setname for setname in keys(samplesets)]
    snl = length(sn)
    ix = rowstart
    aix = 1
    arr =[[] for _ in sn]
    scale = [relativesubrangesize * samplesets[sn[aix]] / minrelsetsize for aix in 1:snl]
    subsetcount = floor(1.0 / mean(scale))
    setrows = rows - subsetcount * gapsize
    actualgapsize = 0
    while ix <= rowend
        subrangesize = max(1, round(Int, setrows * scale[aix]))
        nextrange = (ix, min(rowend, ix+subrangesize-1))
        (verbosity >= 3) && println("\rsubrangesize=$subrangesize ix=$ix, snl=$snl aix=$aix rowend=$rowend nextrange=$nextrange actual range length=$(nextrange[2] - nextrange[1] + 1) actual gap size=$actualgapsize")
        push!(arr[aix], nextrange)
        ix = ix + subrangesize + gapsize
        actualgapsize = ix - nextrange[2] - 1
        aix = aix % snl + 1
    end
    res = Dict(sn[aix] => [t[1]:t[2] for t in arr[aix]] for aix in eachindex(arr))
    return res
end

function _test_setpartitions(samples=129601, samplesets=Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), gapsize=24*60, relativesubrangesize=1/13)
    # function _test_setpartitions(samples=49, samplesets=Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), gapsize=1, relativesubrangesize=3/50)
        # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 0, 1/9)
    # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 1, 1/9)
    println("samples=$samples, samplesets=$samplesets, gapsize=$gapsize, relativesubrangesize=$relativesubrangesize")
    res = setpartitions(1:samples, samplesets, gapsize, relativesubrangesize)
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
    println("\nres=$res")
end

"summary prediction list is a csv file with the one line summary of an training/evaluation to compare the different configurations quickly"
function summarypredictionlist()
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

"adds a summary prediction line with overall performance of evaluation / test sets and the filename of details to the prediction list"
function registersummaryprediction(filename, evalperf=missing, testperf=missing)
    df = summarypredictionlist()
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

"loads and returns the predictions for every ohlcv time of a classifier froma jdf file into a DataFrame"
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
    # checklabeldistribution(df[!, "targets"])
    Targets.labeldistribution(df[!, "targets"])
    return df
end

"saves the predictions for every ohlcv time of a classifier given in df"
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

"creates a DataFrame of predictions for every ohlcv time of a classifier"
function predictionsdataframeold(nn::NN, setranges, targets, predictions, features)
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
    # df[:, "targets"] = categorical(targets; levels=nn.labels, compress=true)
    df[:, "targets"] = targets
    df[:, "opentime"] = Features.ohlcvdfview(features)[!, :opentime]
    df[:, "pivot"] = Features.ohlcvdfview(features)[!, :pivot]
    println("Classify.predictionsdataframeold size=$(size(df)) keys=$(names(df))")
    println(describe(df, :all))
    println("diagnose: features=$features")
    fileprefix = uppercase(Ohlcv.basecoin(Features.ohlcv(features)) * Ohlcv.quotecoin(Features.ohlcv(features))) * "_" * nn.fileprefix
    savepredictions(df, fileprefix)
    return fileprefix
end

"creates a feature matrix from the dataframe results of a base classifier"
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

"Target consistency check that a longhold or close signal can only follow a longbuy or longclose signal"
function checktargetsequence(targets::CategoricalArray)
    labels = levels(targets)
    ignoreix, longbuyix, longholdix, allcloseix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in Targets.tradelabels())
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    longbuy = levelcode(first(targets)) in [longholdix, allcloseix] ? longbuyix : (levelcode(first(targets)) == shortholdix ? shortbuyix : ignoreix)
    for (ix, tl) in enumerate(targets)
        if levelcode(tl) == longbuyix
            longbuy = longbuyix
        elseif (levelcode(tl) == longholdix)
            if (longbuy != longbuyix)
                @error "$ix: missed $longbuyix ($(labels[longbuyix])) before $longholdix ($(labels[longholdix]))"
            end
        elseif levelcode(tl) == shortbuyix
            longbuy = shortbuyix
        elseif (levelcode(tl) == shortholdix)
            if (longbuy != shortbuyix)
                @error "$ix: missed $shortbuyix ($(labels[shortbuyix])) before $shortholdix ($(labels[shortholdix]))"
            end
        elseif (levelcode(tl) == allcloseix)
            if (longbuy == ignoreix)
                @error "$ix: missed either $shortbuyix ($(labels[shortbuyix])) or $longbuyix ($(labels[longbuyix])) before $allcloseix ($(labels[allcloseix]))"
            end
        elseif (levelcode(tl) == ignoreix)
            longbuy = ignoreix
        else
            @error "$ix: unexpected $(levelcode(tl)) ($(labels[allcloseix]))"
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
    (verbosity >= 3) && println("target label distribution in %: ", [(labels[i], round(cnt[i] / targetcount*100, digits=1)) for i in eachindex(labels)])

end

"returns features as Array and targets as label vector (for classifier) and as gain vector (for regressor)."
function featurestargets(base::AbstractString, features::Features.AbstractFeatures, targets::Targets.AbstractTargets)
    features = Features.features(features, base)
    odf = Features.ohlcvdfview(features)
    println(Features.describe(features))
    featuresdescription = Features.describe(features)
    labels = Targets.labels(targets, odf[begin, :opentime], odf[end, :opentime])
    @assert size(odf, 1) == length(labels) "base=$base: size(odf, 1)=$(size(odf, 1)) == length(labels)=$(length(labels))"
    relativegain = Targets.relativegain(targets, odf[begin, :opentime], odf[end, :opentime])
    @assert size(odf, 1) == length(relativegain) "base=$base: size(odf, 1)=$(size(odf, 1)) == length(relativegain)=$(length(relativegain))"
    println(describe(DataFrame(reshape(labels, (length(labels), 1)), ["labels"]), :all))
    targetsdescription = Targets.describe(targets)
    features = Array(features)  # change from df to array
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    return features, labels, featuresdescription, targetsdescription, relativegain
end

"Under construction "
function combifeaturestargets(nnvec::Vector{NN}, features::Features.AbstractFeatures, targets::Targets.AbstractTargets)
    @assert false "under construction - does not work"
    #TODO targets to be provided - open how to generate
    features = Features.features(features)
    odf = Features.ohlcvdfview(features)
    println(Features.describe(features))
    featuresdescription = Features.describe(features)
    labels = Targets.labels(targets, odf[begin, :opentime], odf[end, :opentime])
    @assert size(odf, 1) == length(labels) "size(odf, 1)=$(size(odf, 1)) == length(labels)=$(length(labels))"
    relativegain = Targets.relativegain(targets, odf[begin, :opentime], odf[end, :opentime])
    @assert size(odf, 1) == length(relativegain) "size(odf, 1)=$(size(odf, 1)) == length(relativegain)=$(length(relativegain))"

    #TODO what follows is the old stuff
    labels, relativedist, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(features.f2.ohlcv).pivot, nothing) #pe["combi"])
    targets = labels[Features.ohlcvix(features, 1):end]  # cut beginning from ohlcv observations to feature observations
    relativedist = relativedist[Features.ohlcvix(features, 1):end]  # cut beginning from ohlcv observations to feature observations
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

    description = "Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay_out) relu)" # (@doc model001);
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, mnemonic)
    return nn
end

"""```
Neural Net description:
lay_in = featurecount
lay_out = length(labels)
lay1 = 3 * lay_in
lay2 = round(Int, lay1 * 2 / 3)
lay3 = round(Int, (lay2 + lay_out) / 2)
model = Chain(
    BatchNorm(lay_in),
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
function model002(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 3 * lay_in
    lay2 = round(Int, lay1 * 2 / 3)
    lay3 = round(Int, (lay2 + lay_out) / 2)
    model = Chain(
        BatchNorm(lay_in),             #* this initial normalization is the only difference to model001
        Dense(lay_in => lay1, relu),   # activation function inside layer
        BatchNorm(lay1),
        Dense(lay1 => lay2, relu),   # activation function inside layer
        BatchNorm(lay2),
        Dense(lay2 => lay3, relu),   # activation function inside layer
        BatchNorm(lay3),
        Dense(lay3 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.logitcrossentropy

    description = "Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay_out) relu)" # (@doc model001);
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, mnemonic)
    return nn
end

"""
creates and adapts a neural network using `features` with ground truth label provided with `targets` that belong to observation samples with index ix within the original sample sequence.
relativedist is a vector
"""
function adaptnn!(nn::NN, features::AbstractMatrix, targets::AbstractVector)
    # onehottargets = Flux.onehotbatch(targets, unique(targets))  # onehot class encoding of an observation as one column
    onehottargets = Flux.onehotbatch(targets, nn.labels)  # onehot class encoding of an observation as one column
    loader = Flux.DataLoader((features, onehottargets), batchsize=64, shuffle=true);

    # Training loop, using the whole data set 1000 times:
    nn.losses = Float32[]
    trainmode!(nn)
    minloss = maxloss = missing
    breakmsg = "epoch loop finished without convergence"
    maxepoch = EnvConfig.configmode == test ? 10 : 1000
    @showprogress for epoch in 1:maxepoch
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
    testmode!(nn)
    nn.optim # parameters, momenta and output have all changed
    println(breakmsg)  # print after showprogress loop to avoid cluttered text output
    println("minloss=$minloss  maxloss=$maxloss")
    return nn
end

" Returns a predictions Float Array of size(classes, observations)"
predict(nn::NN, features) = Flux.softmax(nn.model(features))  # size(classes, observations)

function predictiondistribution(predictions, classifiertitle)
    maxindex = mapslices(argmax, predictions, dims=1)
    dist = zeros(Int, maximum(unique(maxindex)))  # maximum == length
    for ix in maxindex
        dist[ix] += 1
    end
    println("$(EnvConfig.now()) prediction distribution with $classifiertitle classifier: $dist")
end

function adaptbase(regrwindow, features::Features.AbstractFeatures, pe::Dict, setranges::Dict)
    println("$(EnvConfig.now()) preparing features and targets for regressionwindow $regrwindow")
    features, targets, featuresdescription, targetsdescription, relativedist = featurestargets(regrwindow, features, pe)
    # trainix = subsetdim2(collect(firstindex(targets):lastindex(targets)), setranges["base"])
    trainfeatures = subsetdim2(features, setranges["base"])
    traintargets = subsetdim2(targets, setranges["base"])
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    # (trainfeatures), traintargets = oversample((trainfeatures), traintargets)  # all classes are equally trained
    (trainfeatures), traintargets = undersample((trainfeatures), traintargets)  # all classes are equally trained
    # println("after oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    println("$(EnvConfig.now()) adapting machine for regressionwindow $regrwindow")
    nn = model001(size(trainfeatures, 1), Targets.tradelabels(), Features.periodlabels(regrwindow))
    nn = adaptnn!(nn, trainfeatures, traintargets)
    nn.featuresdescription = featuresdescription
    nn.targetsdescription = targetsdescription
    println("$(EnvConfig.now()) predicting with machine for regressionwindow $regrwindow")
    pred = predict(nn, features)
    push!(nn.predictions, predictionsdataframeold(nn, setranges, targets, pred, features))
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

function adaptcombi(nnvec::Vector{NN}, features::Features.AbstractFeatures, pe::Dict, setranges::Dict)
    @error "under reconsideration - does not work"
    return

    println("$(EnvConfig.now()) preparing features and targets for combi classifier")
    features, targets, featuresdescription, targetsdescription, relativedist = combifeaturestargets(nnvec, features, pe)
    # trainix = subsetdim2(collect(firstindex(targets):lastindex(targets)), setranges["combi"])
    # println("size(features)=$(size(features)), size(targets)=$(size(targets))")
    trainfeatures = subsetdim2(features, setranges["combi"])
    traintargets = subsetdim2(targets, setranges["combi"])
    # println("before oversample size(trainfeatures)=$(size(trainfeatures)), size(traintargets)=$(size(traintargets))")
    (trainfeatures), traintargets = undersample((trainfeatures), traintargets)  # all classes are equally trained
    # println("after oversample size(trainfeatures)=$(size(trainfeatures)), size(traintargets)=$(size(traintargets))")
    println("$(EnvConfig.now()) adapting machine for combi classifier")
    nn = model001(size(trainfeatures, 1), Targets.tradelabels(), "combi")
    nn = adaptnn!(nn, trainfeatures, traintargets)
    nn.featuresdescription = featuresdescription
    nn.targetsdescription = targetsdescription
    nn.predecessors = [nn.fileprefix for nn in nnvec]
    println("$(EnvConfig.now()) predicting with combi classifier")
    pred = predict(nn, features)
    push!(nn.predictions, predictionsdataframeold(nn, setranges, targets, pred, features))
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
    return EnvConfig.logpath(prefix * ".bson")
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
    (verbosity >= 2) && println("saving classifier $(nn.fileprefix) to $(nnfilename(nn.fileprefix))")
    # nn.losses = compresslosses(nn.losses)
    BSON.@save nnfilename(nn.fileprefix) nn
    # @error "save machine to be implemented for pure flux" filename
    # smach = serializable(mach)
    # JLSO.save(filename, :machine => smach)
end

function loadnn(filename)
    (verbosity >= 2) && println("loading classifier $filename from $(nnfilename(filename))")
    nn = model001(1, ["dummy1", "dummy2"], "dummy menmonic")  # dummy data struct
    BSON.@load nnfilename(filename) nn
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
    df = DataFrame(set=CategoricalArray(undef, 0; levels=levels(predictions.set), ordered=false), opentrade=Int32[], openix=Int32[], trade=Int32[], allcloseix=Int32[], gain=Float32[])
    if size(predictions, 1) == 0
        return df
    end
    predonly = predictions[!, predictioncolumns(predictions)]
    scores, maxindex = maxpredictions(Matrix(predonly), 2)
    labels = levels(predictions.targets)
    ignoreix, longbuyix, longholdix, allcloseix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in Targets.tradelabels())
    buytrade = (tradeix=allcloseix, predix=0, set=predictions[begin, :set])  # tradesignal, predictions index
    holdtrade = (tradeix=allcloseix, predix=0, set=predictions[begin, :set])  # tradesignal, predictions index

    function closetrade!(tradetuple, closetrade, ix)
        gain = (predictions.pivot[ix] - predictions.pivot[tradetuple.predix]) / predictions.pivot[tradetuple.predix] * 100
        gain = tradetuple.tradeix in [longbuyix, longholdix] ? gain : -gain
        push!(df, (tradetuple.set, tradetuple.tradeix, tradetuple.predix, closetrade, ix, gain))
        return (tradeix=allcloseix, predix=ix, set=predictions.set[ix])
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
            buytrade = buytrade.tradeix == allcloseix ? (tradeix=longbuyix, predix=ix, set=predictions.set[ix]) : buytrade
        elseif (labelix == longholdix) && (thresholds[longholdix] <= score)
            buytrade, holdtrade = closeifneeded!(shortbuyix, shortholdix, longholdix, ix)
            holdtrade = holdtrade.tradeix == allcloseix ? (tradeix=longholdix, predix=ix, set=predictions.set[ix]) : holdtrade
        elseif ((labelix == allcloseix) && (thresholds[allcloseix] <= score)) || ((labelix == ignoreix) && (thresholds[ignoreix] <= score))
            buytrade = buytrade.tradeix != allcloseix ? closetrade!(buytrade, allcloseix, ix) : buytrade
            holdtrade = holdtrade.tradeix != allcloseix ? closetrade!(holdtrade, allcloseix, ix) : holdtrade
        elseif (labelix == shortholdix) && (thresholds[shortholdix] <= score)
            buytrade, holdtrade = closeifneeded!(longbuyix, longholdix, shortholdix, ix)
            holdtrade = holdtrade.tradeix == allcloseix ? (tradeix=shortholdix, predix=ix, set=predictions.set[ix]) : holdtrade
        elseif (labelix == shortbuyix) && (thresholds[shortbuyix] <= score)
            buytrade, holdtrade = closeifneeded!(longbuyix, longholdix, shortbuyix, ix)
            buytrade = buytrade.tradeix == allcloseix ? (tradeix=shortbuyix, predix=ix, set=predictions.set[ix]) : buytrade
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
    predonly = predictions[!, predictioncolumns(predictions)]
    scores, maxindex = maxpredictions(Matrix(predonly), 2)
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
    # (verbosity >= 3) && println(xcdf)
    return xcdf
    #TODO next step: take only first of an equal trading signal sequence according to threshold -> how often is a sequence missed?
 end

function predictioncolumns(predictionsdf::AbstractDataFrame)
    nms = names(predictionsdf)
    tl = string.(Targets.tradelabels())
    [nms[nmix] for nmix in eachindex(nms) if nms[nmix] in tl]
end

newtargetsdict(predictions) = Dict(zip(levels(predictions[!, :targets]), fill(0, length(levels(predictions[!, :targets])))))
newclassifydict(predictions, classified) = Dict(zip(levels(classified), [newtargetsdict(predictions) for _ in levels(classified)]))

function confusionmatrix(predictions::AbstractDataFrame)
    # maxindex -> classified label column
    # combi -> classified label vs target label -> tp, fp, tn, fn label = cm label
    # %cm label = per set specific cm label / all cm label 
    prednames = predictioncolumns(predictions)
    predonly = @view predictions[!, prednames]
    scores, maxindex = maxpredictions(Matrix(predonly), 2)
    predstr = vec([string.(prednames[ix]) for ix in maxindex])
    # predstr = vec([prednames[ix] for ix in maxindex])
    # classified = CategoricalVector(predstr, levels=string.(prednames), ordered=false)
    classified = CategoricalVector(predstr, levels=prednames, ordered=false)
    setnames = levels(predictions.set)
    # create cmdict as a dict(key=setname, value=Dict(key=classified label, value=Dict(key=target label, value=count)))
    cmdict = Dict(zip(setnames, [newclassifydict(predictions, classified) for _ in setnames]))
    for ix in eachindex(classified)
        cmdict[predictions[ix, :set]][classified[ix]][predictions[ix, :targets]] += 1
    end
    setcount = Dict([(setname, count(predictions[!, :set] .== setname)) for setname in setnames])

    cmdf = DataFrame()
    for setname in keys(cmdict) # build up data frame column names
        cmdf[!, "set"] = String[]
        for cl in keys(cmdict[setname])
            cmdf[!, "prediction"] = String[]
            for trg in keys(cmdict[setname][cl])
                cmdf[!, "truth_" * trg] = Int32[]
            end
            # cmdf[!, "truth_all"] = Int32[]
            # cmdf[!, "set_all"] = Int32[]
            #!TODO adding Positive predictive value (PPV) related to count
            # for trg in keys(cmdict[setname][cl])
            #     cmdf[!, "truth_" * trg * "_%"] = Float32[]
            # end
        end
    end
    for setname in keys(cmdict) # fill data frame column data
        for cl in keys(cmdict[setname])
            row = Any[]
            push!(row, setname)
            push!(row, "pred_" * string(cl))
            count = 0
            for trg in keys(cmdict[setname][cl])
                push!(row, cmdict[setname][cl][trg])
                count += cmdict[setname][cl][trg]
            end
            # push!(row, count)
            # push!(row, setcount[setname])
            # for trg in keys(cmdict[setname][cl])
            #     push!(row, round(cmdict[setname][cl][trg] / setcount[setname] * 100; digits=1))
            # end
            push!(cmdf, row)
        end
    end
    return sort!(cmdf, [order(:set,rev=true), :prediction])
end

function confusionmatrix_deprecated(predictions::AbstractDataFrame)
    predonly = predictions[!, predictioncolumns(predictions)]
    # return confusionmatrix(Matrix(predonly), predictions.targets, names(predonly))

    scores, maxindex = maxpredictions(Matrix(predonly), 2)
    labels = names(predonly)
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
    return sort!(cdf, [:set, :predlabel])
end

function predictionsfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return prefix * ".jdf"
end

labelvec(labelindexvec, labels=Targets.tradelabels()) = [labels[i] for i in labelindexvec]

labelindexvec(labelvec, labels=Targets.tradelabels()) = [findfirst(x -> x == focuslabel, labels) for focuslabel in labelvec]

"returns a (scores, labelindices) tuple of best predictions. Without labels the index is the index within levels(df.targets).
dim=1 indicates predictions of the same column are compared while dim=2 indicates that predictions of the same row are compared."
function maxpredictions(predictions::AbstractMatrix, dim=1)
    if dim == 1
        maxindex = mapslices(argmax, predictions, dims=1)
        scores = [predictions[maxindex[i], i] for i in eachindex(maxindex)]
    elseif dim == 2
        maxindex = mapslices(argmax, predictions, dims=2)
        scores = [predictions[i, maxindex[i]] for i in eachindex(maxindex)]
    else
        @error "unexpected dim=$dim for size(predictions)=$(size(predictions))"
    end
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
#             if !(label in Targets.tradelabels())
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

function maxpredictions(predictions::AbstractDataFrame)
    return maxpredictions(Matrix(predictions), 2)


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

function binarypredictions(predictions::AbstractMatrix, focuslabel::String, labels=Targets.tradelabels())
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

function binarypredictions(predictions::AbstractDataFrame, focuslabel::String, labels=Targets.tradelabels())
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

function aucscores(pred, labels=Targets.tradelabels())
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

# aucscores(pred, labels=Targets.tradelabels()) = Dict(String(focuslabel) => auc(binarypredictions(pred, focuslabel, labels)...) for focuslabel in labels)
    # auc_scores = []
    # for class_label in unique(targets)
    #     class_scores, class_events = binarypredictions(pred, targets, class_label)
    #     auc_score = auc(class_scores, class_events)
    #     push!(auc_scores, auc_score)
    # end
    # return auc_scores
# end

"Returns a Dict of class => roc tuple of vectors for false_positive_rates, true_positive_rates, thresholds"
function roccurves(pred, labels=Targets.tradelabels())
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

function confusionmatrix(pred, targets, labels=Targets.tradelabels())
    predonly = pred[!, predictioncolumns(pred)]

    dim = length(targets) == size(pred, 1) ? 2 : 1
    _, maxindex = maxpredictions(Matrix(predonly), dim)
    targets = [String(targets[ix]) for ix in eachindex(targets)] # convert CategoricalVector to String Vector
    predlabels = labelvec(maxindex, levels(targets))
    StatisticalMeasures.ConfusionMatrices.confmat(predlabels, targets)
end

function _sumperf(tpdf, setname)
    selectedtrades = filter(row -> (row.set == setname) && ((row.trade == "longbuy") || (row.trade == "shortbuy")), tpdf, view=true)
    if length(selectedtrades.gainpct) > 0
        tcount = sum(selectedtrades.cnt)
        tsum = sum(selectedtrades.gainpct)
        sumperf = round(tsum/tcount, digits=3)
    else
        sumperf = 0.0
    end
    return sumperf
end

function evaluatepredictions(predictions::AbstractDataFrame, fileprefix)
    println("evaluatepredictions: size=$(size(predictions)) \n$(describe(predictions))")
    assetpair, nntitle = split(fileprefix, "_")[1:2]
    title = assetpair * "_" * nntitle
    # if EnvConfig.configmode == EnvConfig.test
        cdf = confusionmatrix(predictions)
        xcdf = extendedconfusionmatrix(predictions)
    # end
    labels = levels(predictions.targets)
    thresholds = [0.01f0 for l in labels]
    tdf = trades(predictions, thresholds)
    println("evaluatepredictions: size(tdf)=$(size(tdf)) \n$(describe(tdf))")
    tpdf = tradeperformance(tdf, labels)
    println("evaluatepredictions: size(tpdf)=$(size(tpdf)) \n$(describe(tpdf))")

    evalperf = _sumperf(tpdf, "eval")
    testperf = _sumperf(tpdf, "test")
    # if EnvConfig.configmode == EnvConfig.production
        registersummaryprediction(fileprefix, evalperf, testperf)
    # end

    for s in levels(predictions.set)
        if s == "unused"
            continue
        end
        sdf = filter(row -> row.set == s, predictions, view=true)
        if size(sdf, 1) > 0
            # if EnvConfig.configmode == EnvConfig.test
                # println(title)
                # aucscores = Classify.aucscores(sdf)
                # println("auc[$s, $title]=$(aucscores)")
                # rc = Classify.roccurves(sdf)
                # Classify.plotroccurves(rc, "$s / $title")
                println(title)
                show(stdout, MIME"text/plain"(), confusionmatrix(sdf, sdf.targets))  # prints the table
                println(title)
                println(filter(row -> row.set == s, cdf, view=true))
                println(title)
                println(filter(row -> row.set == s, xcdf, view=true))
            # end
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
    features, pe = Targets.loaddata(ohlcv, labelthresholds)
    # println(features)
    len = length(Ohlcv.dataframe(features.f2.ohlcv).pivot) - Features.ohlcvix(features, 1) + 1
    # println("$(EnvConfig.now()) len=$len  length(Ohlcv.dataframe(features.f2.ohlcv).pivot)=$(length(Ohlcv.dataframe(features.f2.ohlcv).pivot)) Features.ohlcvix(features, 1)=$(Features.ohlcvix(features, 1))")
    setranges = setpartitions(1:len, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/80)
    for (s,v) in setranges
        println("$s: length=$(length(v))")
    end
    # Threads.@threads for regrwindow in features.regrwindow
    for regrwindow in features.regrwindow
        if isnothing(select) || (regrwindow in select)
            push!(nnvec, adaptbase(regrwindow, features, pe, setranges))
        else
            println("skipping $regrwindow classifier due to not selected")
        end
    end
    if isnothing(select) || ("combi" in select)
        nncombi = adaptcombi(nnvec, features, pe, setranges)
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

#region Classifier013

mutable struct BaseClassifier013
    ohlcv::Ohlcv.OhlcvData
    f5::Union{Nothing, Features.Features005}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier013(ohlcv::Ohlcv.OhlcvData, f5=Features.Features005(Features.featurespecification005(Features.regressionfeaturespec005(),[],[])))
        Features.setbase!(f5, ohlcv, usecache=true)
        cl = isnothing(f5) ? nothing : new(ohlcv, f5, 0)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier013)
    println(io, "BaseClassifier013[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f5=$(!isnothing(bc.f5)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier013)
    if !isnothing(bc.f5)
        Features.write(bc.f5)
    end
end

supplement!(bc::BaseClassifier013) = Features.supplement!(bc.f5)

const REGRESSIONWINDOW013 = Int32[rw for rw in Features.regressionwindows005 if 12*60 <= rw <= 24*60]
const LONGGAINTHRESHOLD013 = Float32[0.02f0, 0.04f0, 1f0]
const SHORTGAINTHRESHOLD013 = Float32[-0.02f0, -0.04f0, -1f0]
const LONGSELLTHRESHOLD013 = Float32[0.01f0]
const SHORTSELLTHRESHOLD013 = Float32[-0.01f0]
const SELLTHRESHOLDFACTOR013 = Float32[0.25f0]
const OPTPARAMS013 = Dict(
    "regrwindow" => REGRESSIONWINDOW013,
    "longgainthreshold" => LONGGAINTHRESHOLD013,
    "shortgainthreshold" => SHORTGAINTHRESHOLD013,
    "longsellthreshold" => LONGSELLTHRESHOLD013,
    "shortsellthreshold" => SHORTSELLTHRESHOLD013,
    "sellthresholdfactor" => SELLTHRESHOLDFACTOR013
)

"""
Classifier013 idea
- focus regression line is the one that exceeds a gain threshold over its regression line length and the gain of the next smaller (over focus length) is higher than focus gain ==> longbuy
- longclose if gain is decreased to gain threshold / x or pivot crosses regression line 
- longclose criteria in case of portfolio assets and new start (i.e. no known focus regression): calc focus regression -> if none then longclose otherwise follow longclose criteria above

"""
mutable struct Classifier013 <: AbstractClassifier
    bc::Dict{AbstractString, BaseClassifier013}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    function Classifier013(optparams=OPTPARAMS013)
        cl = new(Dict(), DataFrame(), optparams, nothing)
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier013, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier013(ohlcv)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier013)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier013)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier013)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier013)::Integer =  maximum(Features.regressionwindows005)


function advice(cl::Classifier013, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier013"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end

function advice(cl::Classifier013, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f5, ohlcvix)
    regrwindow = nothing
    if !isnothing(investment)
        if configureclassifier!(cl, base, investment.configid)
            regrwindow = cfg.regrwindow
        else
            @error "cannot configure string(cl) with configid from $investment"
        end
    end
    lastgrad = nothing
    ta = TradeAdvice(cl, bc.cfgid, allclose, 1f0, base, piv[ohlcvix], Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], 0f0, 1f0, nothing)
    if !isnothing(regrwindow) # check longclose condition
        regry = Features.regry(bc.f5, regrwindow)[fix]
        grad = Features.grad(bc.f5, regrwindow)[fix]
        gain = Targets.relativegain(regry, grad, regrwindow, forward=false)
        ta.hourlygain = Targets.relativegain(regry, grad, 60, forward=false)
        if investment.tradelabel in [longhold, longbuy, longstrongbuy]
            if (gain < cfg.longgainthreshold * cfg.sellthresholdfactor) && (piv[ohlcvix] >= regry * (1 + cfg.longsellthreshold))
                ta.tradelabel = longclose
            else
                ta.tradelabel = longhold
            end
        elseif investment.tradelabel in [shortstrongbuy, shortbuy, shorthold]
            if (gain > cfg.shortgainthreshold * cfg.sellthresholdfactor) && (piv[ohlcvix] <= regry * (1 + cfg.shortsellthreshold))
                ta.tradelabel = shortclose
            else
                ta.tradelabel = shorthold
            end
        end
    end
    longbuyenabled = shortbuyenabled = true
    for rw in Features.regressionwindows005
        regry = Features.regry(bc.f5, rw)[fix]
        grad = Features.grad(bc.f5, rw)[fix]
        gain = Targets.relativegain(regry, grad, rw, forward=false)
        if gain < 0f0
            longbuyenabled = false  # no long buy if a smaller regression window has negative gradient
        elseif gain > 0f0
            shortbuyenabled = false  # no short buy if a smaller regression window has positive gradient
        end
        if !isnothing(regrwindow) && (rw > regrwindow) # no focus on longer term windows
            break
        elseif gain >= cfg.longgainthreshold
            # if longbuyenabled && (isnothing(lastgrad) || (lastgrad >= grad))  # check that trend is still there and not a short term peak that is flatten out
                # longbuy condition in place - ay be with shorter rw as established regrwindow
                bc.cfgid = configurationid(cl, (cfg..., regrwindow=rw))
                regrwindow = rw
                ta.hourlygain = Targets.relativegain(regry, grad, 60, forward=false)
                ta.tradelabel = longbuy
                break
            # end
        elseif gain <= cfg.shortgainthreshold
            # if shortbuyenabled && (isnothing(lastgrad) || (lastgrad <= grad))  # check that trend is still there and not a short term peak that is flatten out
                # shortbuy condition in place - may be with shorter rw as established regrwindow
                bc.cfgid = configurationid(cl, (cfg..., regrwindow=rw))
                regrwindow = rw
                ta.hourlygain = Targets.relativegain(regry, grad, 60, forward=false)
                ta.tradelabel = shortbuy
                break
            # end
        end
        lastgrad = grad
    end
    return ta
end

configurationid4base(cl::Classifier013, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier013, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier013"
        return false
    end
end

#endregion Classifier013

#region Classifier011

mutable struct BaseClassifier011
    ohlcv::Ohlcv.OhlcvData
    f4::Union{Nothing, Features.Features004}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier011(ohlcv::Ohlcv.OhlcvData, cfgid, f4=Features.Features004(ohlcv, usecache=true))
        cl = isnothing(f4) ? nothing : new(ohlcv, f4, cfgid)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier011)
    println(io, "BaseClassifier011[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f4=$(!isnothing(bc.f4)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier011)
    if !isnothing(bc.f4)
        Features.write(bc.f4)
    end
end

supplement!(bc::BaseClassifier011) = Features.supplement!(bc.f4, bc.ohlcv)

const REGRWINDOW011 = Int16[60]
const LONGTRENDTHRESHOLD011 = Float32[1f0] # , 0.04f0, 0.06f0, 1f0]  # 1f0 == switch off long trend following
const SHORTTRENDTHRESHOLD011 = Float32[-1f0] # , -0.04f0, -0.06f0, -1f0]  # -1f0 == switch off short trend following
const VOLATILITYBUYTHRESHOLD011 = Float32[-0.005f0]
const VOLATILITYSELLTHRESHOLD011 = Float32[0.005f0]
const VOLATILITYSHORTTHRESHOLD011 = Float32[-1f0] # 0f0, -1f0]  # -1f0 == switch off volatility short investments
const VOLATILITYLONGTHRESHOLD011 = Float32[-0.005f0]  # 1f0 == switch off volatility long investments
const OPTPARAMS011 = Dict(
    "regrwindow" => REGRWINDOW011,
    "longtrendthreshold" => LONGTRENDTHRESHOLD011,
    "shorttrendthreshold" => SHORTTRENDTHRESHOLD011,
    "volatilitybuythreshold" => VOLATILITYBUYTHRESHOLD011,   # symetric for long and short
    "volatilitysellthreshold" => VOLATILITYSELLTHRESHOLD011, # symetric for long and short
    "volatilityshortthreshold" => VOLATILITYSHORTTHRESHOLD011,
    "volatilitylongthreshold" => VOLATILITYLONGTHRESHOLD011
)

"""
Classifier011 idea
- leverage long and short volatility trades in times of flat slopes
- follow the regression long and short if their gradient exceeds thresholds
- use fixed regression window
"""
mutable struct Classifier011 <: AbstractClassifier  #* is also used as ohlcv and f4 anchor in cryptocockpit
    bc::Dict{AbstractString, BaseClassifier011}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    defaultcfgid
    function Classifier011(optparams=OPTPARAMS011)
        cl = new(Dict(), DataFrame(), optparams, nothing, 1)
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier011, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier011(ohlcv, cl.defaultcfgid)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier011)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier011)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier011)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier011)::Integer =  maximum(Features.regressionwindows004)


function advice(cl::Classifier011, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier011"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end


#TODO implement batchadvice - see Trade
#TODO implement self evaluation on regular basis - transparent for Trade?
function advice(cl::Classifier011, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f4, ohlcvix)
    regry = Features.regry(bc.f4, cfg.regrwindow)[fix]
    grad = Features.grad(bc.f4, cfg.regrwindow)[fix]
    ra = Targets.relativegain(regry, grad, cfg.regrwindow, forward=false)
    volatilitydownprice = regry * (1 + cfg.volatilitybuythreshold)
    volatilityupprice = regry * (1 + cfg.volatilitysellthreshold)
    ta = TradeAdvice(cl, bc.cfgid, allclose, 1f0, base, piv[ohlcvix], Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], Targets.relativegain(regry, grad, 60, forward=false), 1f0, nothing)

    if ((piv[ohlcvix] < volatilitydownprice) && (ra >= cfg.volatilitylongthreshold)) || (ra >= cfg.longtrendthreshold)
        ta.tradelabel = longbuy
    elseif (piv[ohlcvix-1] >= volatilityupprice) && (ra < cfg.longtrendthreshold)
        ta.tradelabel = longclose
    elseif ((piv[ohlcvix] > volatilityupprice) && (ra <= cfg.volatilityshortthreshold)) || (ra <= cfg.shorttrendthreshold)
        ta.tradelabel = shortbuy
    elseif (piv[ohlcvix-1] <= volatilitydownprice) && (ra > cfg.shorttrendthreshold)
        ta.tradelabel = shortclose
    elseif (ra >= cfg.volatilitylongthreshold)
        ta.tradelabel = longhold
    elseif (ra <= cfg.volatilityshortthreshold)
        ta.tradelabel = shorthold
    else
        ta.tradelabel = allclose
    end
    return ta
end

configurationid4base(cl::Classifier011, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier011, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier011"
        return false
    end
end

function configureclassifier!(cl::Classifier011, configid::Integer, updatedbases::Bool)
    cl.defaultcfgid = configid
    if updatedbases
        for base in keys(cl.bc)
            cl.bc[base].cfgid = configid
        end
    end
end

#endregion Classifier011

#region Classifier014
"""
idea: stay close volatility crossing to leverage overall swing as well as volatility, considering that a swing is often followed buy volatile ending

- focus regression is the largest one with touchpoints in first and second half of the regression line
- each rergression line has its own volatility buy and sell amplitudes
- buy when 
    - buy volatility amplitude is reached
    - gain over focus regression length exceeds threshold AND next smaller regression gradient is equal or greater, which prevents buy when smaller regression flattens out
- sell when
  - volatility is reached and gain over focus regression length does not exceed threshold
"""
mutable struct BaseClassifier014
    ohlcv::Ohlcv.OhlcvData
    f5::Union{Nothing, Features.Features005}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier014(ohlcv::Ohlcv.OhlcvData, f5=Features.Features005(Features.featurespecification005(Features.regressionfeaturespec005(),[],[])))
        Features.setbase!(f5, ohlcv, usecache=true)
        cl = isnothing(f5) ? nothing : new(ohlcv, f5, 0)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier014)
    println(io, "BaseClassifier014[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f5=$(!isnothing(bc.f5)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier014)
    if !isnothing(bc.f5)
        Features.write(bc.f5)
    end
end

supplement!(bc::BaseClassifier014) = Features.supplement!(bc.f5)

const REGRESSIONWINDOW014 = Int32[rw for rw in Features.regressionwindows005 if 4*60 <= rw <= 24*60]
const LONGTRENDTHRESHOLD014 = Float32[0.02f0]  # 1f0 == switch off long trend following
const SHORTTRENDTHRESHOLD014 = Float32[-0.02f0]  # -1f0 == switch off short trend following
const VOLATILITYBUYTHRESHOLD014 = Float32[-0.01f0, -0.02f0]
const VOLATILITYSELLTHRESHOLD014 = Float32[0.01f0, 0.02f0]
const VOLATILITYSHORTTHRESHOLD014 = Float32[0f0]  # -1f0 == switch off volatility short investments
const VOLATILITYLONGTHRESHOLD014 = Float32[0f0]  # 1f0 == switch off volatility long investments
const AUTOSWITCH014 = Int32[1, 2, 3]  # swithc between regrwindows 1=stay at 24h, 2= switch between 24h and 12h, 3= switch between 24h, 12h, 4h 
const OPTPARAMS014 = Dict(
    "regrwindow" => REGRESSIONWINDOW014,
    "longtrendthreshold" => LONGTRENDTHRESHOLD014,
    "shorttrendthreshold" => SHORTTRENDTHRESHOLD014,
    "volatilitybuythreshold" => VOLATILITYBUYTHRESHOLD014,   # symetric for long and short
    "volatilitysellthreshold" => VOLATILITYSELLTHRESHOLD014, # symetric for long and short
    "volatilityshortthreshold" => VOLATILITYSHORTTHRESHOLD014,
    "volatilitylongthreshold" => VOLATILITYLONGTHRESHOLD014,
    "autoswitch" => AUTOSWITCH014
)

"""
Classifier014 idea
- focus regression line is the one that exceeds a gain threshold over its regression line length and the gain of the next smaller (over focus length) is higher than focus gain ==> longbuy
- longclose if gain is decreased to gain threshold / x or pivot crosses regression line 
- longclose criteria in case of portfolio assets and new start (i.e. no known focus regression): calc focus regression -> if none then longclose otherwise follow longclose criteria above

"""
mutable struct Classifier014 <: AbstractClassifier
    bc::Dict{AbstractString, BaseClassifier014}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    defaultcfgid
    function Classifier014(optparams=OPTPARAMS014)
        cl = new(Dict(), DataFrame(), optparams, DataFrame(), 1)
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier014, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier014(ohlcv)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier014)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier014)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier014)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier014)::Integer =  maximum(Features.regressionwindows005)


function advice(cl::Classifier014, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier014"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end

"checks whether all regression lines in rws were touched or crossed in the first half as well as the second half of the regression line"
function regressiontouched(piv, pix, f5, fix, rws)
    res = Dict()
    checkminutes = 24*60
    halftime = round(Int, checkminutes / 2)
    if (pix < checkminutes) || (fix < checkminutes)
        [res[rw] = (rw == rws[end]) for rw in rws]  # no history => stay with longest regression window
    else
        for rw in rws
            regry = Features.regry(f5, rw)
            c1 = c2 = 0
            dl = nothing

            function _check(c, dl, ix)
                d = piv[pix-ix] - regry[fix-ix]
                if isnothing(dl) ? (d == 0) : (dl * d <= 0)
                    # change and on different sides of regression line
                    c += 1
                    dl = d
                end
                return c, dl
            end

            for ix in 0:halftime-1
                c2, dl = _check(c2, dl, ix)
            end
            for ix in halftime:checkminutes-1
                c1, dl = _check(c1, dl, ix)
            end
            res[rw] = ((c1>0) && (c2>0))
        end
    end
    return res
end

function advice(cl::Classifier014, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f5, ohlcvix)
    touched = regressiontouched(piv, ohlcvix, bc.f5, fix, REGRESSIONWINDOW014)
    regrwindow = REGRESSIONWINDOW014[end]
    for rw in reverse(REGRESSIONWINDOW014)
        if touched[rw]
            regrwindow = rw
            break
        end
    end
    regry = Features.regry(bc.f5, regrwindow)[fix]
    grad = Features.grad(bc.f5, regrwindow)[fix]
    ra = Targets.relativegain(regry, grad, regrwindow, forward=false)
    volatilitydownprice = regry * (1 + cfg.volatilitybuythreshold)
    volatilityupprice = regry * (1 + cfg.volatilitysellthreshold)
    ta = TradeAdvice(cl, bc.cfgid, allclose, 1f0, base, piv[ohlcvix], Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], Targets.relativegain(regry, grad, 60, forward=false), 1f0, nothing)

    if ((piv[ohlcvix] < volatilitydownprice) && (ra >= cfg.volatilitylongthreshold)) || (ra >= cfg.longtrendthreshold)
        ta.tradelabel = longbuy
    elseif (piv[ohlcvix-1] >= volatilityupprice) && (ra < cfg.longtrendthreshold)
        ta.tradelabel = longclose
    elseif ((piv[ohlcvix] > volatilityupprice) && (ra <= cfg.volatilityshortthreshold)) || (ra <= cfg.shorttrendthreshold)
        ta.tradelabel = shortbuy
    elseif (piv[ohlcvix-1] <= volatilitydownprice) && (ra > cfg.shorttrendthreshold)
        ta.tradelabel = shortclose
    else
        ta.tradelabel = shorthold
    end
    push!(cl.dbgdf, (otime=Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], tradelabel=ta.tradelabel, cfgid=bc.cfgid, rw=regrwindow, cfg...))
    return ta
end

configurationid4base(cl::Classifier014, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier014, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier014"
        return false
    end
end

function configureclassifier!(cl::Classifier014, configid::Integer, updatedbases::Bool)
    cl.defaultcfgid = configid
    if updatedbases
        for base in keys(cl.bc)
            cl.bc[base].cfgid = configid
        end
    end
end

#endregion Classifier014


#region Classifier015

mutable struct BaseClassifier015
    ohlcv::Ohlcv.OhlcvData
    f4::Union{Nothing, Features.Features004}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier015(ohlcv::Ohlcv.OhlcvData, cfgid, f4=Features.Features004(ohlcv, usecache=true))
        cl = isnothing(f4) ? nothing : new(ohlcv, f4, cfgid)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier015)
    println(io, "BaseClassifier015[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f4=$(!isnothing(bc.f4)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier015)
    if !isnothing(bc.f4)
        Features.write(bc.f4)
    end
end

supplement!(bc::BaseClassifier015) = Features.supplement!(bc.f4, bc.ohlcv)

const REGRWINDOW015 = Int16[24*60]
const LONGTRENDTHRESHOLD015 = Float32[0.02f0] # , 0.04f0, 0.06f0, 1f0]  # 1f0 == switch off long trend following
const SHORTTRENDTHRESHOLD015 = Float32[-0.02f0] # , -0.04f0, -0.06f0, -1f0]  # -1f0 == switch off short trend following
const VOLATILITYBUYTHRESHOLD015 = Float32[-0.01f0, -0.02f0]
const VOLATILITYSELLTHRESHOLD015 = Float32[0.01f0, 0.02f0]
const VOLATILITYSHORTTHRESHOLD015 = Float32[0f0] # , -1f0]  # -1f0 == switch off volatility short investments
const VOLATILITYLONGTHRESHOLD015 = Float32[0f0] # , 1f0]  # 1f0 == switch off volatility long investments
const OPTPARAMS015 = Dict(
    "regrwindow" => REGRWINDOW015,
    "longtrendthreshold" => LONGTRENDTHRESHOLD015,
    "shorttrendthreshold" => SHORTTRENDTHRESHOLD015,
    "volatilitybuythreshold" => VOLATILITYBUYTHRESHOLD015,   # symetric for long and short
    "volatilitysellthreshold" => VOLATILITYSELLTHRESHOLD015, # symetric for long and short
    "volatilityshortthreshold" => VOLATILITYSHORTTHRESHOLD015,
    "volatilitylongthreshold" => VOLATILITYLONGTHRESHOLD015
)

"""
Classifier015 idea
- leverage long and short volatility trades in times of flat slopes
- follow the regression long and short if their gradient exceeds thresholds
- use fixed regression window
"""
mutable struct Classifier015 <: AbstractClassifier
    bc::Dict{AbstractString, BaseClassifier015}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    defaultcfgid
    function Classifier015(optparams=OPTPARAMS015)
        cl = new(Dict(), DataFrame(), optparams, nothing, 1)
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier015, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier015(ohlcv, cl.defaultcfgid)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier015)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier015)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier015)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier015)::Integer =  maximum(Features.regressionwindows004)


function advice(cl::Classifier015, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier015"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end


#TODO implement batchadvice - see Trade
#TODO implement self evaluation on regular basis - transparent for Trade?
function advice(cl::Classifier015, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f4, ohlcvix)
    regry = Features.regry(bc.f4, cfg.regrwindow)[fix]
    grad = Features.grad(bc.f4, cfg.regrwindow)[fix]
    ra = Targets.relativegain(regry, grad, cfg.regrwindow, forward=false)
    volatilitydownprice = regry * (1 + cfg.volatilitybuythreshold)
    volatilityupprice = regry * (1 + cfg.volatilitysellthreshold)
    ta = TradeAdvice(cl, bc.cfgid, allclose, 1f0, base, piv[ohlcvix], Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], Targets.relativegain(regry, grad, 60, forward=false), 1f0, nothing)

    if ((piv[ohlcvix] < volatilitydownprice) && (ra >= cfg.volatilitylongthreshold)) || (ra >= cfg.longtrendthreshold)
        ta.tradelabel = longbuy
    elseif (piv[ohlcvix-1] >= volatilityupprice) && (ra < cfg.longtrendthreshold)
        ta.tradelabel = longclose
    elseif ((piv[ohlcvix] > volatilityupprice) && (ra <= cfg.volatilityshortthreshold)) || (ra <= cfg.shorttrendthreshold)
        ta.tradelabel = shortbuy
    elseif (piv[ohlcvix-1] <= volatilitydownprice) && (ra > cfg.shorttrendthreshold)
        ta.tradelabel = shortclose
    else
        ta.tradelabel = shorthold
    end
    return ta
end

configurationid4base(cl::Classifier015, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier015, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier015"
        return false
    end
end

function configureclassifier!(cl::Classifier015, configid::Integer, updatedbases::Bool)
    cl.defaultcfgid = configid
    if updatedbases
        for base in keys(cl.bc)
            cl.bc[base].cfgid = configid
        end
    end
end

#endregion Classifier015


#region Classifier016

const REGRESSIONWINDOW016 = Int32[rw for rw in Features.regressionwindows005 if rw <= 4*60]

"per base and pre regression variables"
mutable struct BaseRegr016 #TODO should move to Features
    lastndiff  # vector with the last N extreme differences
    lastxpiv  # price of last extreme
    BaseRegr016(lastn) = new(zeros(Float32, lastn), nothing)
end

mutable struct BaseClassifier016
    ohlcv::Ohlcv.OhlcvData
    f5::Union{Nothing, Features.Features005}
    reprpar::Dict  # key = regression window, value = BaseRegr016
    function BaseClassifier016(ohlcv::Ohlcv.OhlcvData, lastn, f5=Features.Features005(Features.featurespecification005(Features.regressionfeaturespec005(REGRESSIONWINDOW016, ["grad", "regry"]),[],[])))
        Features.setbase!(f5, ohlcv, usecache=true)
        cl = isnothing(f5) ? nothing : new(ohlcv, f5, Dict([rw => BaseRegr016(lastn) for rw in REGRESSIONWINDOW016]))
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier016)
    println(io, "BaseClassifier016[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f5=$(!isnothing(bc.f5)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier016)
    if !isnothing(bc.f5)
        Features.write(bc.f5)
    end
end

supplement!(bc::BaseClassifier016) = Features.supplement!(bc.f5)

const BUYTHRESHOLD016 = Float32[0.02f0]  # onlybuy if one of the last N differences exceeded that threshold
const SEPARATELONGSHORT016 = Bool[true, false]  # separate long from short differences for buy decision 
const LASTN016 = Int32[2, 4]
const SHORTESTREGRESSION016 = REGRESSIONWINDOW016[begin:end-1]
const LONGESTREGRESSION016 = REGRESSIONWINDOW016[end]
const OPTPARAMS016 = Dict(
    "regrwindow" => REGRESSIONWINDOW016,
    "buythreshold" => BUYTHRESHOLD016,
    "separatelongshort" => SEPARATELONGSHORT016,
    "lastn" => LASTN016, 
    "shortestregression" => SHORTESTREGRESSION016
)

"""
Classifier016 idea
- a) the smallest regression is always best following the real price line and is the start to buy and sell at extremes
- b) if the N (e.g. N=2 to consider the last uphill as well as last downhill) differences in extremes don't exceed a minimumm threshold then consider the next longer regession 
and repeat step a) with it
"""
mutable struct Classifier016 <: AbstractClassifier
    bc::Dict{AbstractString, BaseClassifier016}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    function Classifier016(optparams=OPTPARAMS016)
        cl = new(Dict(), DataFrame(), optparams, 1, DataFrame())
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier016, ohlcv::Ohlcv.OhlcvData)
    cfg = configuration(cl, bc.cfgid)
    bc = BaseClassifier016(ohlcv, cfg.lastn)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier016)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier016)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier016)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier016)::Integer =  maximum(Features.regressionwindows005)


function advice(cl::Classifier016, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier016"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end

function advice(cl::Classifier016, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f5, ohlcvix)




    #TODO advice not yet implemented
    error("advice not yet implemented")

    touched = regressiontouched(piv, ohlcvix, bc.f5, fix, REGRESSIONWINDOW016)
    regrwindow = REGRESSIONWINDOW016[end]
    for rw in reverse(REGRESSIONWINDOW016)
        if touched[rw]
            regrwindow = rw
            break
        end
    end
    regry = Features.regry(bc.f5, regrwindow)[fix]
    grad = Features.grad(bc.f5, regrwindow)[fix]
    ra = Targets.relativegain(regry, grad, regrwindow, forward=false)
    volatilitydownprice = regry * (1 + cfg.volatilitybuythreshold)
    volatilityupprice = regry * (1 + cfg.volatilitysellthreshold)
    ta = TradeAdvice(cl, bc.cfgid, allclose, 1f0, base, piv[ohlcvix], Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], Targets.relativegain(regry, grad, 60, forward=false), 1f0, nothing)

    if ((piv[ohlcvix] < volatilitydownprice) && (ra >= cfg.volatilitylongthreshold)) || (ra >= cfg.longtrendthreshold)
        ta.tradelabel = longbuy
    elseif (piv[ohlcvix-1] >= volatilityupprice) && (ra < cfg.longtrendthreshold)
        ta.tradelabel = longclose
    elseif ((piv[ohlcvix] > volatilityupprice) && (ra <= cfg.volatilityshortthreshold)) || (ra <= cfg.shorttrendthreshold)
        ta.tradelabel = shortbuy
    elseif (piv[ohlcvix-1] <= volatilitydownprice) && (ra > cfg.shorttrendthreshold)
        ta.tradelabel = shortclose
    else
        ta.tradelabel = shorthold
    end
    push!(cl.dbgdf, (otime=Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], tradelabel=ta.tradelabel, cfgid=bc.cfgid, rw=regrwindow, cfg...))
    return ta
end

configurationid4base(cl::Classifier016, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier016, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier016"
        return false
    end
end

function configureclassifier!(cl::Classifier016, configid::Integer, updatedbases::Bool)
    cl.defaultcfgid = configid
    if updatedbases
        for base in keys(cl.bc)
            cl.bc[base].cfgid = configid
        end
    end
end

#endregion Classifier016

"""
idea:
- follow the most suitable regression line but what is a suitable gradient(regr) to follow?
- evaluate distribution per regression line over large population to determine the grad deviation and correlate those as being part of a x% peak of only that regr or not
- compare those distributions across liquidity bins (median/d) and coins to see whether those are similar or need to be adapted
"""


end  # module
