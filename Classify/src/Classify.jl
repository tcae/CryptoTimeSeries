"""
Train and evaluate the trading signal classifiers
"""
module Classify

using CSV, DataFrames, Logging, Dates, Random
using BSON, JDF, Flux, Statistics, ProgressMeter, StatisticalMeasures, MLUtils, Tables
using CategoricalArrays
using CategoricalDistributions
using Distributions
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv

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


export lstm_bounds_trend_features, lstm_feature_contract, lstm_tensor_windows
export lstm_trade_signal_model, train_lstm_trade_signals!, predict_lstm_trade_signals, penultimatefeatures

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

"Returns the storage key used to persist classifier trade simulation results."
function evalfilename()
    return "ClassifierTradesim"
end

"Writes the DataFrame of simulation results to file."
function writesimulation(df)
    if size(df, 1) > 0
        filepath = EnvConfig.savedf(df, evalfilename())
        (verbosity >= 3) && println("$(EnvConfig.now()): saved classifier simulation trades in $filepath")
    end
end

"Reads and returns the DataFrame of simulation results from file."
function readsimulation()
    df = EnvConfig.readdf(evalfilename(); copycols=true)
    df = isnothing(df) ? DataFrame() : df
    filepath = EnvConfig.tablepath(evalfilename(); format=:auto)
    if size(df, 1) > 0
        (verbosity >= 2) && println("$(EnvConfig.now()) loaded classifier simulation trades from $filepath")
    else
        (verbosity >= 2) && println("$(EnvConfig.now()) Loading $filepath failed")
    end
    return df
end

"""
Instantiates and evaluates all provided classifier types and logs all trades in df.
- basecoins is a String vector of basecoin names
- startdt and enddt further constraints the timerange
Evaluates using cached OHLCV data via Ohlcv.read() (following TrendDetector pattern)
"""
function evaluateclassifiers(classifiertypevector, basecoins, startdt, enddt)
    df = readsimulation()
    (verbosity >=3) && println("$(EnvConfig.now()): successfully read classifier simulation trades with $(size(df, 1)) entries")
    for clt in classifiertypevector
        cl = clt()
        for base in basecoins
            ohlcv = Ohlcv.read(base)
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
    valuecorrection::Function
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

_identitycorrection(y_raw::AbstractMatrix) = y_raw

# Regressor output layout is [center; width], with width constrained to non-negative.
function _centerwidthclamp(y_raw::AbstractMatrix)
    return vcat(@view(y_raw[1:1, :]), clamp.(@view(y_raw[2:2, :]), 0f0, Inf32))
end

function NN(model, optim, lossfunc, labels, description, mnemonic="", fileprefix=""; valuecorrection::Function=_identitycorrection)
    return NN(model, optim, lossfunc, valuecorrection, labels, description, mnemonic, fileprefix, String[], "", "", Float32[], String[])
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

# Composite wrapper: feedforward base NN + LSTM that learns sequences of bucketed
# classifier outputs. The LSTM operates on averaged classifier probability vectors
# across buckets of `bucketlen` minutes and learns windows of length `seqlen` buckets.
mutable struct CompositeNN
    base::NN
    lstmmodel::Any
    lstmoptim::Any
    lstmloss::Function
    labels::AbstractVector
    mnemonic::String
    fileprefix::String
    losses::Vector{Float32}
    seqlen::Int
    bucketlen::Int
end

function CompositeNN(base::NN, lstmmodel, lstmoptim, lstmloss, seqlen::Int, bucketlen::Int)
    return CompositeNN(base, lstmmodel, lstmoptim, lstmloss, base.labels, base.mnemonic * "_LSTM", base.fileprefix * "_lstm", Float32[], seqlen, bucketlen)
end

compositefilename(fileprefix::String) = EnvConfig.logpath(splitext(fileprefix)[1] * "_composite.bson")

function savecomposite(cnn::CompositeNN)
    try
        BSON.@save compositefilename(cnn.fileprefix) cnn
    catch e
        Logging.@warn "failed to save composite $(compositefilename(fileprefix)): $e"
    end
end

function loadcomposite(fileprefix::String)
    cnn = nothing
    try
        BSON.@load compositefilename(fileprefix) cnn
    catch e
        Logging.@warn "failed to load composite $(compositefilename(fileprefix)): $e"
    end
    return cnn
end

"Train an LSTM on per-minute classifier predictions stored in `pred_df`.
 pred_df: DataFrame with one column per class (string names matching `base.labels`),
 and columns `:target`, `:set`, `:rangeid`, `:rix` (as produced by TrendDetector.getfeaturestargetsdf).
 seqlen: number of buckets in a sequence window. bucketlen: bucket size in minutes.
 Returns a CompositeNN with trained LSTM attached.
"
function train_lstm_on_predictions(base::NN, pred_df::DataFrame, ftdf::DataFrame; seqlen::Int=3, bucketlen::Int=5, hidden::Int=32, epochs::Int=10, batchsize::Int=64)

    labels = base.labels
    # ensure label columns present
    classcols = string.(labels)
    for c in classcols
        @assert c in names(pred_df) "missing class column $c in pred_df"
    end

    # group by rangeid to prepare sequences per independent time series
    grp = groupby(pred_df, :rangeid)

    inputs = Vector{Array{Float32,2}}() # list of (classes seqlen buckets) matrices for each sequence sliding window
    targets = String[]
    sets = String[]
    maps_back = Vector{Tuple{Int,Int,Int}}() # (rangeid, bucket_end_index, bucket_start_rowix) for mapping back

    for g in grp
        nrows = nrow(g)
        # number of full buckets
        nbuckets = fld(nrows, bucketlen)
        if nbuckets < seqlen # not enough buckets for one window
            continue
        end
        # build bucketed average matrix: classes seqlen nbuckets
        bucketmat = zeros(Float32, length(classcols), nbuckets)
        for b in 1:nbuckets #! mistake: the seqlen buckets - each of bucketlen averaged prediction results - need to be calculated for every minute
            rstart = (b-1)*bucketlen + 1
            rend = b*bucketlen
            for (ci, col) in enumerate(classcols)
                bucketmat[ci, b] = mean(Float32.(g[rstart:rend, Symbol(col)]))
            end
        end
        # sliding windows
        for be in seqlen:nbuckets
            window = bucketmat[:, (be-seqlen+1):be] # classes x seqlen
            push!(inputs, copy(window))
            # choose target = label at last row of this bucket window
            last_row = (be-1)*bucketlen + bucketlen
            targ = String(g[last_row, :target])
            push!(targets, targ)
            push!(sets, String(g[last_row, :set]))
            push!(maps_back, (first(g[!, :rangeid]), be, last_row))
        end
    end

    if length(inputs) == 0
        @warn "no training samples for LSTM (maybe sequences too short)"
        return nothing
    end

    # restrict to training samples
    train_ix = findall(==("train"), sets)
    if length(train_ix) == 0
        @warn "no train set windows found for LSTM"
        return nothing
    end

    # prepare batches
    function make_batch(idxs)
        b = length(idxs)
        X = Array{Float32,3}(undef, length(classcols), size(inputs[1], 2), b) # classes x seq x batch
        for (bi, ix) in enumerate(idxs)
            X[:, :, bi] = inputs[ix]
        end
        ys = targets[idxs]
        Y = onehotbatch(ys, string.(labels))
        return X, Y
    end

    # define LSTM model
    lstm = LSTM(length(classcols) => hidden)
    dense = Dense(hidden => length(classcols)) #! shouldn't the relu not part of dense?
    model = Chain(lstm, x -> Flux.relu.(x), dense)
    optim = Flux.setup(Flux.Adam(), params(model))
    lossfn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)

    # training loop
    train_idxs = train_ix
    nepoch = epochs
    for epoch in 1:nepoch
        shuffle!(train_idxs)
        for i in 1:batchsize:length(train_idxs)
            batch_ids = train_idxs[i:min(i+batchsize-1, end)]
            X, Y = make_batch(batch_ids)
            Flux.reset!(lstm)
            gs = Flux.gradient(params(model)) do
                seq_len = size(X, 2)
                for t in 1:seq_len
                    xt = X[:, t, :]
                    lstm(xt)
                end
                h = lstm.h
                ŷ = dense(h)
                lossfn(ŷ, Y)
            end
            Flux.update!(optim, params(model), gs)
        end
    end

    cnn = CompositeNN(base, model, optim, lossfn, seqlen, bucketlen)
    savecomposite(cnn)
    return cnn
end

"Predict with a CompositeNN given base predictions: returns a DataFrame aligned to base_pred_df rows with class probability columns.
 The LSTM refines the base predictions by computing bucket averages and applying the trained LSTM on windows of length `seqlen`.
 The LSTM output for each bucket is broadcast to the rows of that bucket to return per-minute predictions.
 
 base_pred_df: DataFrame with class label columns (one per label in cnn.labels), plus :target, :set, :rangeid, :rix columns
"
function predict(cnn::CompositeNN, base_pred_df::DataFrame)
    labels = cnn.labels
    classcols = string.(labels)
    
    # Ensure required columns present
    for c in classcols
        @assert c in names(base_pred_df) "missing class column \$c in base_pred_df"
    end
    for c in [:target, :set, :rangeid, :rix]
        @assert c in names(base_pred_df) "missing required column \$c in base_pred_df"
    end
    
    # Initialize output with base predictions
    result = select(base_pred_df, vcat(:rix, classcols, [:target, :set, :rangeid]))
    result_ords = sort(result, :rix)[:, 1:end-3]  # drop trailing index columns for sorting
    
    # Get bucket-averaged predictions and bucket-to-row mappings
    bucketed_preds_result = _bucket_predictions_with_indices(base_pred_df, cnn.bucketlen, classcols)
    
    if isnothing(bucketed_preds_result)
        # Not enough data to apply LSTM, return base predictions
        return select(result, vcat(classcols, [:target, :set]))
    end
    
    bucketed_preds, bucket_to_rows, bucket_to_rangeid = bucketed_preds_result
    
    # Run LSTM inference on bucketed predictions
    lstm_refined_preds = _apply_lstm_inference(cnn, bucketed_preds, classcols)
    
    if isnothing(lstm_refined_preds)
        return select(result, vcat(classcols, [:target, :set]))
    end
    
    # Expand LSTM refined predictions back to per-minute level
    for (bucket_idx, row_ixs) in enumerate(bucket_to_rows)
        if bucket_idx <= length(lstm_refined_preds)
            lstm_pred_vec = lstm_refined_preds[bucket_idx]
            for row_ix in row_ixs
                for (ci, col) in enumerate(classcols)
                    result[row_ix, Symbol(col)] = lstm_pred_vec[ci]
                end
            end
        end
    end
    
    # Return with original row order preserved
    sort_ix = sortperm(result_ords[!, :rix])
    return select(result[sort_ix, :], vcat(classcols, [:target, :set]))
end

"Internal: Bucket per-minute predictions into averaged bucket predictions while tracking row indices.
 Returns: (bucketed_inputs, bucket_to_row_indices, bucket_to_rangeid) or nothing if insufficient data
  - bucketed_inputs: Vector of class-probability vectors (one per bucket)
  - bucket_to_row_indices: Vector mapping bucket index to row indices in original DataFrame
  - bucket_to_rangeid: Vector mapping bucket index to its rangeid
"
function _bucket_predictions_with_indices(pred_df::DataFrame, bucketlen::Int, classcols::Vector{String})
    grp = groupby(pred_df, :rangeid)
    
    bucketed_preds = Vector{Vector{Float32}}()
    bucket_to_row_idx = Vector{Vector{Int}}()
    bucket_to_rangeid = Vector{Int}()
    
    for g in grp
        nrows = nrow(g)
        nbuckets = fld(nrows, bucketlen)
        
        if nbuckets == 0
            continue # skip rangeids with too few samples
        end
        
        rangeid = first(g[!, :rangeid])
        row_ixs_in_group = 1:nrows  # indices within this group
        
        # Create bucketed average vectors
        for b in 1:nbuckets
            rstart = (b-1)*bucketlen + 1
            rend = b*bucketlen
            
            # Compute average prediction vector for this bucket
            bucket_avg = Float32[mean(g[rstart:rend, Symbol(col)]) for col in classcols]
            
            push!(bucketed_preds, bucket_avg)
            push!(bucket_to_rangeid, rangeid)
            push!(bucket_to_row_idx, collect(rstart:rend))
        end
    end
    
    if length(bucketed_preds) == 0
        return nothing
    end
    
    return (bucketed_preds, bucket_to_row_idx, bucket_to_rangeid)
end

"Internal: Apply LSTM inference on bucketed predictions.
 Returns a vector of refined class probability predictions (one vector per bucket) or nothing on error.
"
function _apply_lstm_inference(cnn::CompositeNN, bucketed_preds::Vector, classcols::Vector{String})
    try
        outputs = Vector{Vector{Float32}}()
        
        # Re-run the LSTM architecture (same structure as training)
        lstm_layer = cnn.lstmmodel[1]  # extract LSTM
        dense_layer = cnn.lstmmodel[3]  # extract final dense layer
        
        # Process each bucketed prediction through LSTM
        for bucket_pred in bucketed_preds
            # Reset LSTM state
            Flux.reset!(lstm_layer)
            
            # Convert to matrix format: classes x 1 (single time step)
            x = reshape(bucket_pred, length(bucket_pred), 1)
            
            # Pass through LSTM to get hidden state
            h = lstm_layer(x[:, 1])
            
            # Pass hidden state through dense output layer
            ŷ = dense_layer(h)
            
            # Apply softmax to get probability distribution
            pred_probs = Flux.softmax(ŷ)
            
            push!(outputs, pred_probs)
        end
        
        return outputs
        
    catch e
        Logging.@warn "LSTM inference failed: \$e"
        return nothing
    end
end

"""
Contract object for LSTM input preparation that combines bounds regressor outputs
with trend classifier probabilities.

`features` must have shape `(nfeatures, nsamples)` where rows are ordered according
to `feature_names`. The default feature order is:
`trend_prob_*`, `center`, `width`, `lower`, `upper`.
"""
struct LstmBoundsTrendFeatures
    features::Matrix{Float32}
    feature_names::Vector{String}
    targets::Vector{String}
    sets::Vector{String}
    rangeids::Vector{Int32}
    rix::Vector{Int32}
end

"""
Build `LstmBoundsTrendFeatures` from a row-aligned dataframe.

Required columns:
- trend probabilities from `trendprobcols`
- `centercol`, `widthcol`
- `targetcol`, `setcol`, `rangeidcol`, `rixcol`

Derived bounds columns are computed as:
`lower = center - width/2`, `upper = center + width/2`.
"""
function lstm_bounds_trend_features(df::AbstractDataFrame;
    trendprobcols::Vector{Symbol},
    centercol::Symbol=:pred_center,
    widthcol::Symbol=:pred_width,
    targetcol::Symbol=:target,
    setcol::Symbol=:set,
    rangeidcol::Symbol=:rangeid,
    rixcol::Symbol=:sampleix)

    @assert length(trendprobcols) > 0 "trendprobcols must not be empty"

    required = vcat(trendprobcols, [centercol, widthcol, targetcol, setcol, rangeidcol, rixcol])
    required_str = string.(required)
    for col in required_str
        @assert col in names(df) "missing required column $(col); names(df)=$(names(df))"
    end

    center = Float32.(df[!, centercol])
    width = Float32.(df[!, widthcol])
    @assert all(width .>= 0f0) "width must be >= 0; minimum(width)=$(minimum(width))"

    lower = center .- width ./ 2f0
    upper = center .+ width ./ 2f0
    @assert all(lower .<= upper) "expected lower <= upper for all rows"

    nsamples = nrow(df)
    ntrend = length(trendprobcols)
    nfeatures = ntrend + 4 # trend probs + center + width + lower + upper
    feats = Matrix{Float32}(undef, nfeatures, nsamples)

    fidx = 1
    fnames = String[]
    for col in trendprobcols
        feats[fidx, :] = Float32.(df[!, col])
        push!(fnames, "trend_prob_$(String(col))")
        fidx += 1
    end
    feats[fidx, :] = center
    push!(fnames, "center")
    fidx += 1
    feats[fidx, :] = width
    push!(fnames, "width")
    fidx += 1
    feats[fidx, :] = lower
    push!(fnames, "lower")
    fidx += 1
    feats[fidx, :] = upper
    push!(fnames, "upper")

    targets = string.(df[!, targetcol])
    sets = string.(df[!, setcol])
    rangeids = Int32.(df[!, rangeidcol])
    rix = Int32.(df[!, rixcol])

    @assert size(feats, 2) == length(targets) == length(sets) == length(rangeids) == length(rix) "contract length mismatch: size(feats,2)=$(size(feats,2)) length(targets)=$(length(targets)) length(sets)=$(length(sets)) length(rangeids)=$(length(rangeids)) length(rix)=$(length(rix))"

    return LstmBoundsTrendFeatures(feats, fnames, targets, sets, rangeids, rix)
end

"""
Materialize selected table feature columns into a `Float32` matrix of shape
`(nfeatures, nsamples)` without creating a large intermediate wide matrix.
"""
function _table_feature_matrix(columns, featurecols::AbstractVector; rows)
    nfeatures = length(featurecols)
    nsamples = length(rows)
    feats = Matrix{Float32}(undef, nfeatures, nsamples)

    for (fidx, col) in enumerate(featurecols)
        colname = col isa Symbol ? col : Symbol(col)
        coldata = Tables.getcolumn(columns, colname)
        for (j, srcix) in enumerate(rows)
            feats[fidx, j] = Float32(coldata[srcix])
        end
    end

    return feats
end

function _dataframe_feature_matrix(df::AbstractDataFrame, featurecols::AbstractVector; rows=axes(df, 1))
    return _table_feature_matrix(Tables.columns(df), featurecols; rows=rows)
end

"""
Build a generic `LstmBoundsTrendFeatures` contract directly from a precomputed
feature matrix and aligned metadata vectors.
"""
function lstm_feature_contract(features::AbstractMatrix;
    feature_names::Vector{String},
    targets::AbstractVector,
    sets::AbstractVector,
    rangeids::AbstractVector,
    rix::AbstractVector)

    feats = features isa Matrix{Float32} ? features : Matrix{Float32}(features)
    targets_str = string.(targets)
    sets_str = string.(sets)
    rangeids_i32 = Int32.(rangeids)
    rix_i32 = Int32.(rix)

    @assert size(feats, 1) == length(feature_names) "feature_names length mismatch: size(feats,1)=$(size(feats,1)) length(feature_names)=$(length(feature_names))"
    @assert size(feats, 2) == length(targets_str) == length(sets_str) == length(rangeids_i32) == length(rix_i32) "contract length mismatch: size(feats,2)=$(size(feats,2)) length(targets)=$(length(targets_str)) length(sets)=$(length(sets_str)) length(rangeids)=$(length(rangeids_i32)) length(rix)=$(length(rix_i32))"

    return LstmBoundsTrendFeatures(feats, copy(feature_names), targets_str, sets_str, rangeids_i32, rix_i32)
end

"""
Build a generic `LstmBoundsTrendFeatures` contract from arbitrary row-aligned
feature columns.

Despite the historical type name, this contract is generic and can be reused for
any sequential LSTM feature set, including hidden activations exported from a
TrendDetector classifier.
"""
function lstm_feature_contract(df::AbstractDataFrame;
    featurecols::AbstractVector,
    targetcol::Symbol=:target,
    setcol::Symbol=:set,
    rangeidcol::Symbol=:rangeid,
    rixcol::Symbol=:sampleix)

    @assert !isempty(featurecols) "featurecols must not be empty"
    required = vcat(featurecols, [targetcol, setcol, rangeidcol, rixcol])
    required_str = string.(required)
    for col in required_str
        @assert col in names(df) "missing required column $(col); names(df)=$(names(df))"
    end

    feats = _dataframe_feature_matrix(df, featurecols)
    return lstm_feature_contract(
        feats;
        feature_names=string.(featurecols),
        targets=df[!, targetcol],
        sets=df[!, setcol],
        rangeids=df[!, rangeidcol],
        rix=df[!, rixcol],
    )
end

"""
Count how many sliding windows can be formed inside contiguous `rangeids` blocks.
"""
function _count_lstm_windows(rangeids::AbstractVector{<:Integer}, seqlen::Int)::Int
    total = 0
    i = 1
    nsamples = length(rangeids)
    while i <= nsamples
        rid = rangeids[i]
        j = i
        while (j <= nsamples) && (rangeids[j] == rid)
            j += 1
        end
        seglen = j - i
        total += max(seglen - seqlen + 1, 0)
        i = j
    end
    return total
end

"""
Build compact metadata for LSTM windows without materializing the full 3D tensor.
"""
function _lstm_window_index(contract::LstmBoundsTrendFeatures; seqlen::Int=3)
    @assert seqlen > 0 "seqlen=$(seqlen) must be > 0"

    nfeatures, nsamples = size(contract.features)
    order = sortperm(contract.rix)
    if nsamples < seqlen
        return (nfeatures=nfeatures, seqlen=seqlen, order=order, endpos=Int[], targets=String[], sets=String[], rangeids=Int32[], endrix=Int32[])
    end

    targets = contract.targets[order]
    sets = contract.sets[order]
    rangeids = contract.rangeids[order]
    rix = contract.rix[order]

    nwindows = _count_lstm_windows(rangeids, seqlen)
    endpos = Vector{Int}(undef, nwindows)
    yvec = Vector{String}(undef, nwindows)
    svec = Vector{String}(undef, nwindows)
    ridvec = Vector{Int32}(undef, nwindows)
    endrix = Vector{Int32}(undef, nwindows)

    wi = 1
    i = 1
    while i <= nsamples
        rid = rangeids[i]
        j = i
        while (j <= nsamples) && (rangeids[j] == rid)
            j += 1
        end
        lastidx = j - 1
        seglen = lastidx - i + 1
        if seglen >= seqlen
            for wend in (i + seqlen - 1):lastidx
                endpos[wi] = wend
                yvec[wi] = targets[wend]
                svec[wi] = sets[wend]
                ridvec[wi] = rid
                endrix[wi] = rix[wend]
                wi += 1
            end
        end
        i = j
    end

    @assert wi == nwindows + 1 "window indexing mismatch: wi=$(wi) nwindows=$(nwindows)"
    return (nfeatures=nfeatures, seqlen=seqlen, order=order, endpos=endpos, targets=yvec, sets=svec, rangeids=ridvec, endrix=endrix)
end

"""
Build compact metadata for LSTM windows across multiple contracts without
concatenating their feature matrices in memory.
"""
function _lstm_window_index(contracts::AbstractVector{<:LstmBoundsTrendFeatures}; seqlen::Int=3)
    @assert seqlen > 0 "seqlen=$(seqlen) must be > 0"
    isempty(contracts) && return (nfeatures=0, seqlen=seqlen, order=nothing, endpos=Int[], targets=String[], sets=String[], rangeids=Int32[], endrix=Int32[], contractix=Int[], local_indexes=Any[])

    nfeatures = size(first(contracts).features, 1)
    local_indexes = Vector{Any}(undef, length(contracts))
    totalwindows = 0

    for (cix, contract) in pairs(contracts)
        @assert size(contract.features, 1) == nfeatures "all contracts must share the same feature count; contract $(cix) has $(size(contract.features, 1)) features but expected $(nfeatures)"
        localindex = _lstm_window_index(contract; seqlen=seqlen)
        local_indexes[cix] = localindex
        totalwindows += length(localindex.endpos)
    end

    endpos = Vector{Int}(undef, totalwindows)
    yvec = Vector{String}(undef, totalwindows)
    svec = Vector{String}(undef, totalwindows)
    ridvec = Vector{Int32}(undef, totalwindows)
    endrix = Vector{Int32}(undef, totalwindows)
    contractix = Vector{Int}(undef, totalwindows)

    wi = 1
    for (cix, localindex) in pairs(local_indexes)
        nlocal = length(localindex.endpos)
        if nlocal > 0
            rng = wi:(wi + nlocal - 1)
            endpos[rng] .= localindex.endpos
            yvec[rng] .= localindex.targets
            svec[rng] .= localindex.sets
            ridvec[rng] .= localindex.rangeids
            endrix[rng] .= localindex.endrix
            contractix[rng] .= cix
            wi += nlocal
        end
    end

    @assert wi == totalwindows + 1 "multi-contract window indexing mismatch: wi=$(wi) totalwindows=$(totalwindows)"
    return (nfeatures=nfeatures, seqlen=seqlen, order=nothing, endpos=endpos, targets=yvec, sets=svec, rangeids=ridvec, endrix=endrix, contractix=contractix, local_indexes=local_indexes)
end

"""
Materialize only the requested LSTM windows into a dense tensor.
"""
function _lstm_window_tensor(contract::LstmBoundsTrendFeatures, windowindex, windowix::AbstractVector{<:Integer})
    nwindows = length(windowix)
    X = Array{Float32, 3}(undef, windowindex.nfeatures, windowindex.seqlen, nwindows)
    feats = contract.features
    order = windowindex.order

    for (bi, wi) in enumerate(windowix)
        wend = windowindex.endpos[wi]
        wstart = wend - windowindex.seqlen + 1
        for seqix in 1:windowindex.seqlen
            srcix = order[wstart + seqix - 1]
            @views X[:, seqix, bi] .= feats[:, srcix]
        end
    end

    return X
end

function _lstm_window_tensor(contracts::AbstractVector{<:LstmBoundsTrendFeatures}, windowindex, windowix::AbstractVector{<:Integer})
    nwindows = length(windowix)
    X = Array{Float32, 3}(undef, windowindex.nfeatures, windowindex.seqlen, nwindows)

    for (bi, wi) in enumerate(windowix)
        cix = windowindex.contractix[wi]
        contract = contracts[cix]
        localindex = windowindex.local_indexes[cix]
        wend = windowindex.endpos[wi]
        wstart = wend - windowindex.seqlen + 1
        for seqix in 1:windowindex.seqlen
            srcix = localindex.order[wstart + seqix - 1]
            @views X[:, seqix, bi] .= contract.features[:, srcix]
        end
    end

    return X
end

"""
Create sliding LSTM windows with output tensor shape `(nfeatures, seqlen, nbatch)`.

Windows are built per `rangeids` sequence to avoid crossing independent ranges.
Returns `(X, targets, sets, rangeids, endrix)` where metadata vectors are aligned
to the batch axis of `X`.
"""
function lstm_tensor_windows(contract::LstmBoundsTrendFeatures; seqlen::Int=3)
    windowindex = _lstm_window_index(contract; seqlen=seqlen)
    if isempty(windowindex.endpos)
        xempty = Array{Float32, 3}(undef, windowindex.nfeatures, seqlen, 0)
        return (X=xempty, targets=String[], sets=String[], rangeids=Int32[], endrix=Int32[])
    end

    X = _lstm_window_tensor(contract, windowindex, 1:length(windowindex.endpos))
    return (X=X, targets=windowindex.targets, sets=windowindex.sets, rangeids=windowindex.rangeids, endrix=windowindex.endrix)
end

function lstm_tensor_windows(contracts::AbstractVector{<:LstmBoundsTrendFeatures}; seqlen::Int=3)
    windowindex = _lstm_window_index(contracts; seqlen=seqlen)
    if isempty(windowindex.endpos)
        xempty = Array{Float32, 3}(undef, windowindex.nfeatures, seqlen, 0)
        return (X=xempty, targets=String[], sets=String[], rangeids=Int32[], endrix=Int32[])
    end

    X = _lstm_window_tensor(contracts, windowindex, 1:length(windowindex.endpos))
    return (X=X, targets=windowindex.targets, sets=windowindex.sets, rangeids=windowindex.rangeids, endrix=windowindex.endrix)
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
        - `gapsize` is the number of rows between partitions of different sets that are not included in any partition (should be fixed and not a f(requiredminutes) for stable comparison with different features/targets)
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

"Loads and returns the predictions for every OHLCV time of a classifier into a DataFrame."
function loadpredictions(filename)
    filename = predictionsfilename(filename)
    df = DataFrame()
    try
        loaded = EnvConfig.readdf(filename; copycols=true)
        if !isnothing(loaded)
            df = loaded
            println("loaded $filename predictions dataframe of size=$(size(df))")
        end
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
    # Targets.labeldistribution(df[!, "targets"])
    return df
end

"Saves the predictions for every OHLCV time of a classifier given in `df`."
function savepredictions(df, fileprefix)
    filename = predictionsfilename(fileprefix)
    println("saving $filename predictions dataframe of size=$(size(df))")
    try
        EnvConfig.savedf(df, filename)
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
    ignoreix, longbuyix, longholdix, allcloseix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in Targets.uniquelabels())
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

    description = "Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay_out))"
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, "model001_" * mnemonic)
    return nn
end

"""
Compared to model001, added initial BatchNorm before input layer  
This added performance
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

    description = "BatchNorm($(lay_in))-Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay_out))"
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, "model002_" * mnemonic)
    return nn
end

"""
Compared to model002, removed BatchNorm between layers.  
This reduced performace by 1% but was significatly faster.
"""
function model003(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 3 * lay_in
    lay2 = round(Int, lay1 * 2 / 3)
    lay3 = round(Int, (lay2 + lay_out) / 2)
    model = Chain(
        BatchNorm(lay_in),             # delta to model002: remove all but initial BatchNorm
        Dense(lay_in => lay1, relu),   # activation function inside layer
        Dense(lay1 => lay2, relu),   # activation function inside layer
        Dense(lay2 => lay3, relu),   # activation function inside layer
        Dense(lay3 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.logitcrossentropy

    description = "BatchNorm($(lay_in))-Dense($(lay_in)->$(lay1) relu)-Dense($(lay1)->$(lay2) relu)-Dense($(lay2)->$(lay3) relu)-Dense($(lay3)->$(lay_out))"
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, "model003_" * mnemonic)
    return nn
end

"""
Compared to model002, added an additional layer.  
Performace was worse than model002 - surprise
"""
function model004(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 3 * lay_in
    lay2 = round(Int, lay1 * 2 / 3)
    lay3 = round(Int, lay2 * 2 / 3)
    lay4 = round(Int, (lay3 + lay_out) / 2)
    model = Chain(
        BatchNorm(lay_in),
        Dense(lay_in => lay1, relu),   # activation function inside layer
        BatchNorm(lay1),
        Dense(lay1 => lay2, relu),   # activation function inside layer
        BatchNorm(lay2),
        Dense(lay2 => lay3, relu),   # activation function inside layer
        BatchNorm(lay3),
        Dense(lay3 => lay4, relu),   # activation function inside layer
        BatchNorm(lay4),            #* the additiona layer is the difference to model002
        Dense(lay4 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.logitcrossentropy

    description = "BatchNorm($(lay_in))-Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay4) relu)-BatchNorm($(lay4))-Dense($(lay4)->$(lay_out))"
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, "model004_" * mnemonic)
    return nn
end

"""
Compared to model002, extended number of layer nodes by changing factor of layer 1 from 3 to 4.  
Performace was worse than model002 - surprise
"""
function model005(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 4 * lay_in               #* this initial normalization is the only difference to model002
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

    description = "BatchNorm($(lay_in))-Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay_out))"
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, "model005_" * mnemonic)
    return nn
end

"""
Compared to model003, removed layer 3. 
performance impact? 
"""
function model006(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 3 * lay_in
    lay2 = round(Int, lay1 * 2 / 3)
    lay3 = round(Int, (lay2 + lay_out) / 2)
    model = Chain(
        BatchNorm(lay_in),             # delta to model002: remove all but initial BatchNorm
        Dense(lay_in => lay1, relu),   # activation function inside layer
        Dense(lay1 => lay2, relu),   # activation function inside layer
        Dense(lay2 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.logitcrossentropy

    description = "BatchNorm($(lay_in))-Dense($(lay_in)->$(lay1) relu)-Dense($(lay1)->$(lay2) relu)-Dense($(lay2)->$(lay_out))"
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, "model006_" * mnemonic)
    return nn
end

"""
Compared to model003, reduced number of nodes of layers by reducing factor of layer 1 from 3 to 2.  
performance impact?
"""
function model007(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 2 * lay_in
    lay2 = round(Int, lay1 * 2 / 3)
    lay3 = round(Int, (lay2 + lay_out) / 2)
    model = Chain(
        BatchNorm(lay_in),             # delta to model002: remove all but initial BatchNorm
        Dense(lay_in => lay1, relu),   # activation function inside layer
        Dense(lay1 => lay2, relu),   # activation function inside layer
        Dense(lay2 => lay3, relu),   # activation function inside layer
        Dense(lay3 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.logitcrossentropy

    description = "Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay_out) relu)" # (@doc model001);
    nn = NN(model, optim, lossfunc, labels, description)
    setmnemonic(nn, "model007_" * mnemonic)
    return nn
end

"""
Regressor model for a center and width estimation. The clamping makes it an application specific regressor.
"""
function boundsregressor001(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    @assert lay_out == 2 "Unexpected (instead of 2) number ($lay_out) of regressor output variables [$labels] to estimate"
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
        Dense(lay3 => lay_out))   # no activation function inside layer
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.mse  # loss(ŷ, y) convention [1](https://fluxml.ai/Flux.jl/v0.12/models/losses/)[2](https://fluxml.ai/Flux.jl/stable/reference/models/losses/)
    # If your data has outliers, consider Flux.Losses.huber_loss instead of pure MSE.

    description = "Regressor for center/width output: BatchNorm($(lay_in))-Dense($(lay_in)->$(lay1) relu)-BatchNorm($(lay1))-Dense($(lay1)->$(lay2) relu)-BatchNorm($(lay2))-Dense($(lay2)->$(lay3) relu)-BatchNorm($(lay3))-Dense($(lay3)->$(lay_out))"
    nn = NN(model, optim, lossfunc, labels, description; valuecorrection=_centerwidthclamp)
    setmnemonic(nn, "regressor001_" * mnemonic)
    return nn
end

"Return whether a loss history has converged by showing no improvement in the last 5 recorded epochs after at least 11 epochs."
nnconverged(losses::AbstractVector) = (length(losses) > 10) && (losses[end-4] <= losses[end-3] <= losses[end-2] <= losses[end-1] <= losses[end])
nnconverged(nn) = nnconverged(nn.losses)

# using Flux
# using NNlib: softplus

function adaptboundsregressor!(nn::NN, features::AbstractMatrix, Y::AbstractMatrix)
    # Bounds regressor expects [center; width] in Y and uses nn.valuecorrection for width range correction.
    @assert size(features, 2) == size(Y, 2)   # same number of observations
    @assert size(Y, 1) == 2            # centerwidth
    return adaptnn!(nn, features, Y)
end

"""
creates and adapts a neural network using `features` with ground truth label provided with `targets` that belong to observation samples with index ix within the original sample sequence.
relativedist is a vector
"""
function adaptnn!(nn::NN, features::AbstractMatrix, targets::AbstractVector)
    # onehottargets = Flux.onehotbatch(targets, unique(targets))  # onehot class encoding of an observation as one column
    onehottargets = Flux.onehotbatch(targets, nn.labels)  # onehot class encoding of an observation as one column
    return adaptnn!(nn, features, onehottargets)
end

"""
Common adaptation loop for matrix targets.
Uses nn.valuecorrection to transform raw model outputs before loss evaluation.
"""
function adaptnn!(nn::NN, features::AbstractMatrix, Y::AbstractMatrix)
    X = Float32.(features)
    Yf = Float32.(Y)
    @assert size(X, 2) == size(Yf, 2) "size(X,2)=$(size(X,2)) must equal size(Y,2)=$(size(Yf,2))"
    loader = Flux.DataLoader((X, Yf), batchsize=64, shuffle=true)

    # Training loop, using the whole data set 1000 times:
    trainmode!(nn)
    minloss = maxloss = missing
    breakmsg = "epoch loop finished without convergence"
    maxepoch = EnvConfig.configmode == test ? 10 : 1000
    @showprogress for epoch in 1:maxepoch
        losses = Float32[]
        for (x, y) in loader
            loss, grads = Flux.withgradient(nn.model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = nn.valuecorrection(m(x))
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
        savenn(nn)
        if nnconverged(nn)
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

"""
Return the hidden activations before the final output layer of `nn`.

For `model002`, the default behavior evaluates the full network up to the
normalized `lay3` representation and omits only the final classifier head.
"""
function penultimatefeatures(nn::NN, features::AbstractMatrix; stoplayer::Int=length(nn.model.layers) - 1)
    X = features isa Matrix{Float32} ? features : Float32.(features)
    @assert hasproperty(nn.model, :layers) "nn.model must expose a layers field; got $(typeof(nn.model))"
    nlayers = length(nn.model.layers)
    @assert 1 <= stoplayer < nlayers "stoplayer=$(stoplayer) must satisfy 1 <= stoplayer < number of layers $(nlayers)"

    Flux.testmode!(nn.model)
    hidden = X
    for lix in 1:stoplayer
        hidden = nn.model.layers[lix](hidden)
    end
    return Float32.(hidden)
end

"""
Return the penultimate activations for table-backed feature columns while
materializing only one batch at a time.

When `rows` is provided, only those source rows are evaluated. This allows
callers to iterate coin-by-coin without concatenating the full hidden matrix in
memory.
"""
function penultimatefeatures(nn::NN, table, featurecols::AbstractVector;
    stoplayer::Int=length(nn.model.layers) - 1, batchsize::Int=4096, rows=nothing)
    @assert !isempty(featurecols) "featurecols must not be empty"
    @assert batchsize > 0 "batchsize=$(batchsize) must be > 0"

    columns = Tables.columns(table)
    firstcol = featurecols[1] isa Symbol ? featurecols[1] : Symbol(featurecols[1])
    nobs_total = length(Tables.getcolumn(columns, firstcol))
    selectedrows = isnothing(rows) ? collect(1:nobs_total) : collect(rows)
    nobs = length(selectedrows)
    if nobs == 0
        return Matrix{Float32}(undef, 0, 0)
    end

    hidden = Matrix{Float32}(undef, 0, 0)
    for start in 1:batchsize:nobs
        stop = min(start + batchsize - 1, nobs)
        batchrows = @view selectedrows[start:stop]
        x_batch = _table_feature_matrix(columns, featurecols; rows=batchrows)
        hidden_batch = penultimatefeatures(nn, x_batch; stoplayer=stoplayer)
        if size(hidden, 1) == 0
            hidden = Matrix{Float32}(undef, size(hidden_batch, 1), nobs)
        end
        hidden[:, start:stop] .= hidden_batch
    end

    return hidden
end

"Returns a DataFrame of predictions of size(observations, classes) with class labels as column names"
predictdf(nn::NN, features) = DataFrame(permutedims(predict(nn, features), (2, 1)), string.(nn.labels))

"Returns the scores::Float32 of the maximum label::TradeLabel prediction for each observation as a vector of size(observations) as 2 vectors"
function maxpredict(nn::NN, features)
    predonly = predict(nn, features)
    scores, maxindex = maxpredictions(Matrix(predonly), 1)
    labels = vec([nn.labels[ix] for ix in maxindex])
    return scores, labels
end

function maxpredictdf(nn::NN, features)
    score, label = maxpredict(nn, features)
    # println("typeof(score)=$(typeof(score)) size(score)=$(size(score)) type(label)=$(typeof(label)) size(label)=$(size(label))")
    return DataFrame(score=score, label=label)
end

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
    nn = model001(size(trainfeatures, 1), Targets.uniquelabels(), Features.periodlabels(regrwindow))
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
    nn = model001(size(trainfeatures, 1), Targets.uniquelabels(), "combi")
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
    return "losses_" * prefix
end

function loadlosses!(nn)
    filename = lossesfilename(nn.fileprefix)
    df = DataFrame()
    try
        loaded = EnvConfig.readdf(filename)
        if !isnothing(loaded)
            df = loaded
            nn.losses = Vector(df[!, "losses"])
            println("loaded $filename losses dataframe of size=$(size(df))")
        end
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
        EnvConfig.savedf(df, filename)
    catch e
        Logging.@warn "exception $e detected"
    end
end

nnfilename(fileprefix::String) = EnvConfig.logpath(splitext(fileprefix)[1] * ".bson")

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
    (verbosity >= 4) && println("saving classifier $(nn.fileprefix) to $(nnfilename(nn.fileprefix))")
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
    ignoreix, longbuyix, longholdix, allcloseix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in Targets.uniquelabels())
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

"maps a 0.0 <= score <= 1.0  to 1:`thresholdbins` bins - or 0 if score range is invalid or NaN if score is not a number"
function score2bin(score, thresholdbins) 
    if isnan(score)
        return NaN
    end
    @assert thresholdbins > 0 "thresholdbins=$thresholdbins must be > 0"
    return max(min(floor(Int, score / (1.0/thresholdbins)) + 1, thresholdbins), 1)
end

"maps the index of one of `thresholdbins` bins to a score"
bin2score(binix, thresholdbins) = round((binix-1)*1.0/thresholdbins; digits = 2), round(binix*1.0/thresholdbins; digits = 2)

"""
generates summary statistics from predictions
"""
function extendedconfusionmatrix(predictions::AbstractDataFrame, alllabels, thresholdbins=10)
    @assert isinteger(thresholdbins) && (thresholdbins > 0) "thresholdbins=$thresholdbins must be a positive integer"
    nancount = outofrangecount = 0
    confcatsyms = [:tp, :tn, :fp, :fn]
    confcat = Dict(zip(confcatsyms, 1:length(confcatsyms)))
    # preallocate collection matrices with columns TP, TN, FP, FN and rows as bins with lower separation value x/thresholdbins per label per set
    setnames = levels(predictions.set)
    cmc = zeros(Int, length(setnames), length(alllabels), length(confcatsyms), thresholdbins)
    for ix in eachindex(predictions[!, :label])
        labelix = Targets.tradelabelix(predictions[ix, :label], alllabels)
        @assert 0.0 <= predictions[ix, :score] <= 1.0 "prediction[$ix]==$(predictions[ix, :])"
        binix = score2bin(predictions[ix, :score], thresholdbins)
        if isnan(binix)
            nancount += 1
        elseif binix == 0
            outofrangecount += 1
        else
            if predictions[ix, :label] == predictions[ix, :target]
                cmc[levelcode(predictions.set[ix]), labelix, confcat[:tp], binix] += 1
            else  
                cmc[levelcode(predictions.set[ix]), labelix, confcat[:fp], binix] += 1
            end
        end
    end
    if (verbosity >= 1) && (nancount > 0 || outofrangecount > 0)
        println("extended confusion matrix: nancount=$nancount outofrangecount=$outofrangecount")
    end
    cm = zeros(Int, length(setnames), length(alllabels), length(confcatsyms), thresholdbins)
    for six in eachindex(setnames)
        for lix in eachindex(alllabels)
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
    setnamevec = [setnames[six] for six in eachindex(setnames) for lix in eachindex(alllabels) for bix in 1:thresholdbins]
    sc = categorical(setnamevec; levels=setnames)
    labelsvec = [l for six in eachindex(setnames) for l in alllabels for bix in 1:thresholdbins]
    binvec = [(scr = bin2score(bix, thresholdbins); "$bix/[$(scr[1])-$(scr[2])]") for six in eachindex(setnames) for lix in eachindex(alllabels) for bix in 1:thresholdbins]
    bc = categorical(binvec)
    tpvec = [cm[six, lix, confcat[:tp], bix] for six in eachindex(setnames) for lix in eachindex(alllabels) for bix in 1:thresholdbins]
    tnvec = [cm[six, lix, confcat[:tn], bix] for six in eachindex(setnames) for lix in eachindex(alllabels) for bix in 1:thresholdbins]
    fpvec = [cm[six, lix, confcat[:fp], bix] for six in eachindex(setnames) for lix in eachindex(alllabels) for bix in 1:thresholdbins]
    fnvec = [cm[six, lix, confcat[:fn], bix] for six in eachindex(setnames) for lix in eachindex(alllabels) for bix in 1:thresholdbins]
    allvec = tpvec + tnvec + fpvec + fnvec
    tpprc = round.(tpvec ./ allvec .* 100.0; digits=0)
    tnprc = round.(tnvec ./ allvec .* 100.0; digits=0)
    fpprc = round.(fpvec ./ allvec .* 100.0; digits=0)
    fnprc = round.(fnvec ./ allvec .* 100.0; digits=0)
    tpr = round.(tpvec ./ (tpvec + fnvec); digits=2)
    fpr = round.(fpvec ./ (fpvec + tnvec); digits=2)
    ppvprc = round.(tpvec ./ (tpvec + fpvec) .* 100.0; digits=0)
    npvprc = round.(tnvec ./ (tnvec + fnvec) .* 100.0; digits=0)
    xcdf = DataFrame("set" => sc, "pred_label" => labelsvec, "bin" => bc, "tp" => tpvec, "tn" => tnvec, "fp" => fpvec, "fn" => fnvec, "tp%" => tpprc, "tn%" => tnprc, "fp%" => fpprc, "fn%" => fnprc, "tpr" => tpr, "fpr" => fpr, "ppv%" => ppvprc, "npv%" => npvprc)
    # (verbosity >= 3) && println(xcdf)
    return xcdf
    #TODO next step: take only first of an equal trading signal sequence according to threshold -> how often is a sequence missed?
 end

function predictioncolumns(predictionsdf::AbstractDataFrame)
    nms = names(predictionsdf)
    tl = string.(Targets.uniquelabels())
    [nms[nmix] for nmix in eachindex(nms) if nms[nmix] in tl]
end

newtargetsdict(alllabels) = Dict(zip(alllabels, fill(0, length(alllabels))))
newclassifydict(alllabels) = Dict(zip(alllabels, [newtargetsdict(alllabels) for _ in alllabels]))

function confusionmatrix(predictions::AbstractDataFrame, alllabels::AbstractVector)
    # maxindex -> classified label column
    # combi -> classified label vs target label -> tp, fp, tn, fn label = cm label
    # %cm label = per set specific cm label / all cm label 
    classified = predictions.label
    setnames = levels(predictions.set)
    # create cmdict as a dict(key=setname, value=Dict(key=classified label, value=Dict(key=target label, value=count)))
    cmdict = Dict(zip(setnames, [newclassifydict(alllabels) for _ in setnames]))
    for ix in eachindex(classified)
        cmdict[predictions[ix, :set]][classified[ix]][predictions[ix, :target]] += 1
    end
    # setcount = Dict([(setname, count(predictions[!, :set] .== setname)) for setname in setnames])

    cmdf = DataFrame()
    for setname in keys(cmdict) # build up data frame column names
        cmdf[!, "set"] = String[]
        for cl in keys(cmdict[setname])
            cmdf[!, "prediction"] = String[]
            for trg in keys(cmdict[setname][cl])
                cmdf[!, "truth_" * string(trg)] = Int32[]
            end
            cmdf[!, "allpredicted"] = Int32[]
            cmdf[!, "truepositive"] = Int32[]
            cmdf[!, "ppv%"] = Float32[]
        end
    end
    for setname in keys(cmdict) # fill data frame column data
        for cl in keys(cmdict[setname])
            row = Any[]
            push!(row, setname)
            push!(row, "pred_" * string(cl))
            count = tpcount = 0
            for trg in keys(cmdict[setname][cl])
                push!(row, cmdict[setname][cl][trg])
                count += cmdict[setname][cl][trg]
                if cl == trg
                    tpcount = cmdict[setname][cl][trg]
                end
            end
            push!(row, count)
            push!(row, tpcount)
            push!(row, count > 0 ? round(tpcount / count * 100; digits=0) : 0.0)
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
    return splitext(fileprefix)[1]
end

labelvec(labelindexvec, labels=Targets.uniquelabels()) = [labels[i] for i in labelindexvec]

labelindexvec(labelvec, labels=Targets.uniquelabels()) = [findfirst(x -> x == focuslabel, labels) for focuslabel in labelvec]

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
#             if !(label in Targets.uniquelabels())
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

function binarypredictions(predictions::AbstractMatrix, focuslabel::String, labels=Targets.uniquelabels())
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

function binarypredictions(predictions::AbstractDataFrame, focuslabel::String, labels=Targets.uniquelabels())
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

function aucscores(pred, labels=Targets.uniquelabels())
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

# aucscores(pred, labels=Targets.uniquelabels()) = Dict(String(focuslabel) => auc(binarypredictions(pred, focuslabel, labels)...) for focuslabel in labels)
    # auc_scores = []
    # for class_label in unique(targets)
    #     class_scores, class_events = binarypredictions(pred, targets, class_label)
    #     auc_score = auc(class_scores, class_events)
    #     push!(auc_scores, auc_score)
    # end
    # return auc_scores
# end

"Returns a Dict of class => roc tuple of vectors for false_positive_rates, true_positive_rates, thresholds"
function roccurves(pred, labels=Targets.uniquelabels())
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

"Provides a confusion matrix. all 3 vectors shall be of CategoricalVectors. alllabels is a vector of all unique labels, e.g. levels(targets)"
function confusionmatrix(pred, targets, alllabels)
    predonly = pred[!, predictioncolumns(pred)]

    dim = length(targets) == size(pred, 1) ? 2 : 1
    _, maxindex = maxpredictions(Matrix(predonly), dim)
    predlabels = categorical(labelvec(maxindex, levels(targets)))
    return StatisticalMeasures.ConfusionMatrices.confmat(predlabels, targets, levels=alllabels)
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
        cdf = confusionmatrix(predictions, unique(predictions.targets))
        xcdf = extendedconfusionmatrix(predictions, unique(predictions.targets))
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
                t = isa(te, CategoricalVector) ? sdf.targets : categorical(sdf.targets)
                println(title)
                show(stdout, MIME"text/plain"(), confusionmatrix(sdf, t, levels(t)))  # prints the table
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

include("Classifier011.jl")
include("Classifier013.jl")
include("Classifier014.jl")
include("Classifier015.jl")
include("Classifier016.jl")

"""
idea:
- follow the most suitable regression line but what is a suitable gradient(regr) to follow?
- evaluate distribution per regression line over large population to determine the grad deviation and correlate those as being part of a x% peak of only that regr or not
- compare those distributions across liquidity bins (median/d) and coins to see whether those are similar or need to be adapted
"""


end  # module
