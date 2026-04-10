
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
end


"""
    lstm_trade_signal_model(nfeatures::Int, seqlen::Int, hidden_dim::Int=32; labels=[...])

Create an LSTM neural network for trade-signal classification.

Architecture:
- LSTM layer: `LSTM(nfeatures → hidden_dim)` processes sequences
- Dense layer: `Dense(hidden_dim → nclasses)` outputs logits for the supplied trade classes
- **Output: raw logits (no activation)** for use with logitcrossentropy loss
"""
function lstm_trade_signal_model(nfeatures::Int, seqlen::Int, hidden_dim::Int=32; 
    labels=["longbuy", "longclose", "shortbuy", "shortclose"])
    
    @assert nfeatures > 0 "nfeatures must be > 0; got $nfeatures"
    @assert seqlen > 0 "seqlen must be > 0; got $seqlen"
    @assert hidden_dim > 0 "hidden_dim must be > 0; got $hidden_dim"
    @assert length(labels) >= 2 "labels must contain at least 2 classes; got $(length(labels))"
    
    nclasses = length(labels)
    
    model = Flux.Chain(
        Flux.LSTM(nfeatures => hidden_dim),  # Sequence processing: (nfeatures, seqlen, batch) → (hidden_dim, seqlen, batch)
        Flux.Dense(hidden_dim => nclasses)   # Output: (hidden_dim, seqlen, batch) → (nclasses, seqlen, batch)
    )
    
    return model, labels
end

"""
    lstm_checkpoint_filename(fileprefix::AbstractString="LstmTradeSignalModel")

Return the BSON checkpoint path for a persisted LSTM trade-signal model inside the
currently active `EnvConfig` log folder.
"""
lstm_checkpoint_filename(fileprefix::AbstractString="LstmTradeSignalModel") = EnvConfig.logpath(splitext(fileprefix)[1] * ".bson")

"""
    save_lstm_checkpoint(model, optim, losses, eval_losses, labels, seqlen, hidden_dim, epoch;
                         fileprefix::AbstractString="LstmTradeSignalModel")

Persist the current LSTM training state to a BSON checkpoint file and return the
full checkpoint path.
"""
function save_lstm_checkpoint(model, optim, losses, eval_losses, labels, seqlen::Int, hidden_dim::Int, epoch::Int;
    fileprefix::AbstractString="LstmTradeSignalModel", nfeatures::Int)
    checkpointfile = lstm_checkpoint_filename(fileprefix)
    checkpoint = Dict{Symbol, Any}(
        :checkpoint_version => 2,
        :model_state => Flux.state(model),
        :losses => Float32.(losses),
        :eval_losses => Float32.(eval_losses),
        :labels => String.(labels),
        :nfeatures => nfeatures,
        :seqlen => seqlen,
        :hidden_dim => hidden_dim,
        :epoch => epoch,
        :savedat => Dates.now(),
    )
    (verbosity >= 2) && println("$(EnvConfig.now()) saving LSTM checkpoint epoch=$(epoch) to $(checkpointfile)")
    BSON.@save checkpointfile checkpoint
    return checkpointfile
end

"Return the normalized contents of a safe versioned LSTM checkpoint dictionary."
function _normalize_lstm_checkpoint(checkpoint)
    checkpoint isa AbstractDict || return nothing
    required = (:checkpoint_version, :model_state, :losses, :eval_losses, :labels, :nfeatures, :seqlen, :hidden_dim, :epoch)
    all(key -> haskey(checkpoint, key), required) || return nothing
    return (
        checkpoint_version = Int(checkpoint[:checkpoint_version]),
        model_state = checkpoint[:model_state],
        losses = Float32.(checkpoint[:losses]),
        eval_losses = Float32.(checkpoint[:eval_losses]),
        labels = String.(checkpoint[:labels]),
        nfeatures = Int(checkpoint[:nfeatures]),
        seqlen = Int(checkpoint[:seqlen]),
        hidden_dim = Int(checkpoint[:hidden_dim]),
        epoch = Int(checkpoint[:epoch]),
        savedat = get(checkpoint, :savedat, missing),
    )
end

"""
    load_lstm_checkpoint(fileprefix::AbstractString="LstmTradeSignalModel")

Load the saved LSTM training checkpoint from the active log folder and return
it as a normalized named tuple, or `nothing` if no checkpoint exists yet.
Legacy BSON object-graph checkpoints are ignored rather than deserialized,
because they can crash newer Julia/Flux/BSON combinations.
"""
function load_lstm_checkpoint(fileprefix::AbstractString="LstmTradeSignalModel")
    checkpointfile = lstm_checkpoint_filename(fileprefix)
    isfile(checkpointfile) || return nothing

    raw = nothing
    try
        raw = BSON.parse(checkpointfile)
    catch e
        @warn "Ignoring unreadable LSTM checkpoint at $(checkpointfile); starting fresh" exception=e
        return nothing
    end

    rawcheckpoint = get(raw, :checkpoint, nothing)
    if isnothing(rawcheckpoint)
        @warn "Ignoring malformed LSTM checkpoint at $(checkpointfile); missing :checkpoint payload"
        return nothing
    end
    if (rawcheckpoint isa AbstractDict) && !haskey(rawcheckpoint, :checkpoint_version)
        @warn "Ignoring legacy LSTM checkpoint at $(checkpointfile); it uses the older BSON object-graph format that is unsafe in the current environment"
        return nothing
    end

    checkpoint = nothing
    try
        BSON.@load checkpointfile checkpoint
    catch e
        @warn "Ignoring unreadable LSTM checkpoint at $(checkpointfile); starting fresh" exception=e
        return nothing
    end

    normalized = _normalize_lstm_checkpoint(checkpoint)
    if isnothing(normalized)
        @warn "Ignoring malformed LSTM checkpoint at $(checkpointfile); required fields are missing"
        return nothing
    end
    return normalized
end

"""
Compute the mean LSTM loss over a set of windows while materializing only one
batch at a time.
"""
function _mean_lstm_trade_signal_loss(model, lossfunc, contract, windowindex,
    sampleix::AbstractVector{<:Integer}, trade_labels::Vector{String}, batchsize::Int)::Float32
    isempty(sampleix) && return NaN32

    total_loss = 0f0
    total_count = 0
    for start in 1:batchsize:length(sampleix)
        stop = min(start + batchsize - 1, length(sampleix))
        batch_ix = sampleix[start:stop]
        x_batch = _lstm_window_tensor(contract, windowindex, batch_ix)
        y_batch = Flux.onehotbatch(windowindex.targets[batch_ix], trade_labels)
        Flux.testmode!(model)
        # Flux.reset!(model) # reset!(m) is deprecated. You can remove this call as it is no more needed.
        ŷ = model(x_batch)
        ŷ_final = ŷ[:, end, :]
        batch_loss = Float32(lossfunc(ŷ_final, y_batch))
        count = length(batch_ix)
        total_loss += batch_loss * count
        total_count += count
    end

    return total_count == 0 ? NaN32 : total_loss / total_count
end

"""
    train_lstm_trade_signals!(contract, seqlen::Int;
                              hidden_dim::Int=32, maxepoch::Int=1000, batchsize::Int=64,
                              labels=nothing, fileprefix::AbstractString="LstmTradeSignalModel",
                              resume::Bool=true)

Train an LSTM trade signal classifier on one contract or a vector of per-coin contracts.

The training loop keeps memory bounded by materializing only the current batch of
sliding windows instead of building duplicated full-dataset tensors. When
`resume=true`, an existing checkpoint with the same `fileprefix` is reused so a
stopped run can continue from its last completed epoch.

# Arguments
- `contract`: Either one `LstmBoundsTrendFeatures` or a vector of them
- `seqlen::Int`: Sliding window sequence length
- `hidden_dim::Int`: LSTM hidden dimension (default: 32)
- `maxepoch::Int`: Maximum training epochs (default: 1000)
- `batchsize::Int`: Training batch size (default: 64)
- `labels`: Optional explicit class label order
- `fileprefix::AbstractString`: Checkpoint filename prefix in the current log folder
- `resume::Bool`: Resume from an existing compatible checkpoint when available

# Returns
- `model_state::NamedTuple`: `(model=trained_model, optim=optimizer_state, losses=loss_history, eval_losses=eval_loss_history, labels=classes, checkpointfile=path)`

# Example
```julia
windows = Classify.lstm_tensor_windows(contract; seqlen=3)
result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=32)
# result.model is trained for prediction
# result.losses is vector of epoch mean losses
# result.checkpointfile points to the saved BSON checkpoint
```
"""
function train_lstm_trade_signals!(contract, seqlen::Int;
    hidden_dim::Int=32, maxepoch::Int=1000, batchsize::Int=64,
    labels::Union{Nothing,Vector{String}}=nothing, fileprefix::AbstractString="LstmTradeSignalModel",
    resume::Bool=true)

    windowindex = _lstm_window_index(contract; seqlen=seqlen)
    targets = windowindex.targets
    sets = windowindex.sets

    @assert !isempty(windowindex.endpos) "No training windows generated; check contract and seqlen"

    nfeatures = windowindex.nfeatures
    labelorder = Dict(
        "longbuy" => 1,
        "longhold" => 2,
        "longclose" => 3,
        "shortbuy" => 4,
        "shorthold" => 5,
        "shortclose" => 6,
        "allclose" => 7,
    )
    trade_labels = isnothing(labels) ? sort(unique(String.(targets)); by=label -> get(labelorder, label, length(labelorder) + 1)) : String.(labels)
    @assert length(trade_labels) >= 2 "trade_labels must contain at least 2 classes; got trade_labels=$(trade_labels)"

    checkpointfile = lstm_checkpoint_filename(fileprefix)
    checkpoint = resume ? load_lstm_checkpoint(fileprefix) : nothing
    checkpoint_compatible = !isnothing(checkpoint) &&
        Int(checkpoint.nfeatures) == nfeatures &&
        Int(checkpoint.seqlen) == seqlen &&
        Int(checkpoint.hidden_dim) == hidden_dim &&
        String.(checkpoint.labels) == trade_labels

    if checkpoint_compatible
        model, _ = lstm_trade_signal_model(nfeatures, seqlen, hidden_dim; labels=trade_labels)
        try
            Flux.loadmodel!(model, checkpoint.model_state)
            losses = Float32.(checkpoint.losses)
            eval_losses = Float32.(checkpoint.eval_losses)
            start_epoch = Int(checkpoint.epoch)
            optim = Flux.setup(Flux.Adam(0.001, (0.9, 0.999)), model)
            (verbosity >= 2) && println("$(EnvConfig.now()) resuming LSTM checkpoint from epoch=$(start_epoch) at $(checkpointfile)")
        catch e
            checkpoint_compatible = false
            @warn "Ignoring incompatible LSTM checkpoint state at $(checkpointfile); starting fresh" exception=e
        end
    end

    if !checkpoint_compatible
        if !isnothing(checkpoint)
            @warn "Ignoring incompatible LSTM checkpoint at $(checkpointfile); requested nfeatures=$(nfeatures), seqlen=$(seqlen), hidden_dim=$(hidden_dim), labels=$(trade_labels), checkpoint nfeatures=$(checkpoint.nfeatures), seqlen=$(checkpoint.seqlen), hidden_dim=$(checkpoint.hidden_dim), labels=$(String.(checkpoint.labels))"
        end
        model, _ = lstm_trade_signal_model(nfeatures, seqlen, hidden_dim; labels=trade_labels)
        optim = Flux.setup(Flux.Adam(0.001, (0.9, 0.999)), model)
        losses = Float32[]
        eval_losses = Float32[]
        start_epoch = 0
    end

    train_ix = findall(==("train"), sets)
    eval_ix = findall(==("eval"), sets)

    @assert !isempty(train_ix) "No training samples in contract; check set partitioning"
    (verbosity >= 2) && println("$(EnvConfig.now()) LSTM training with $(length(train_ix)) training samples, $(length(eval_ix)) eval samples")

    batchsize = max(1, batchsize)
    lossfunc = Flux.logitcrossentropy
    shuffled_train_ix = copy(train_ix)

    Flux.trainmode!(model)

    @showprogress for epoch in (start_epoch + 1):maxepoch
        Random.shuffle!(shuffled_train_ix)
        epoch_loss_sum = 0f0
        epoch_count = 0

        for start in 1:batchsize:length(shuffled_train_ix)
            stop = min(start + batchsize - 1, length(shuffled_train_ix))
            batch_ix = shuffled_train_ix[start:stop]
            x_batch = _lstm_window_tensor(contract, windowindex, batch_ix)
            y_batch = Flux.onehotbatch(targets[batch_ix], trade_labels)

            # Flux.reset!(model) # reset!(m) is deprecated. You can remove this call as it is no more needed.
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x_batch)
                ŷ_final = ŷ[:, end, :]
                lossfunc(ŷ_final, y_batch)
            end
            Flux.update!(optim, model, grads[1])

            count = length(batch_ix)
            epoch_loss_sum += Float32(loss) * count
            epoch_count += count
        end

        epoch_loss = epoch_count == 0 ? NaN32 : epoch_loss_sum / epoch_count
        push!(losses, epoch_loss)

        eval_loss = isempty(eval_ix) ? epoch_loss : _mean_lstm_trade_signal_loss(model, lossfunc, contract, windowindex, eval_ix, trade_labels, batchsize)
        push!(eval_losses, eval_loss)
        Flux.trainmode!(model)

        if (verbosity >= 3)
            println("Epoch $epoch: train_loss=$(round(epoch_loss; digits=4)) eval_loss=$(round(eval_loss; digits=4))")
        end

        checkpointfile = save_lstm_checkpoint(model, optim, losses, eval_losses, trade_labels, seqlen, hidden_dim, epoch; fileprefix=fileprefix, nfeatures=nfeatures)

        if nnconverged(losses)
            (verbosity >= 2) && println("Converged at epoch $epoch according to nnconverged(losses); recent train losses=$(losses[end-4:end])")
            break
        end
    end

    Flux.testmode!(model)
    (verbosity >= 2) && println("$(EnvConfig.now()) LSTM training complete with $(length(losses)) epochs; checkpoint=$(checkpointfile)")

    return (model=model, optim=optim, losses=losses, eval_losses=eval_losses, labels=trade_labels, checkpointfile=checkpointfile)
end

"""
    predict_lstm_trade_signals(model, contract; seqlen::Int,
                               batchsize::Int=1024)

Predict trade-signal probabilities for all windows in one contract or a vector
of per-coin contracts while only materializing one batch at a time.

Returns `(probs, targets, sets, rangeids, endrix)` with metadata aligned to the
columns of `probs`.
"""
function predict_lstm_trade_signals(model, contract; seqlen::Int, batchsize::Int=1024)
    windowindex = _lstm_window_index(contract; seqlen=seqlen)
    nwindows = length(windowindex.endpos)
    if nwindows == 0
        return (probs=Array{Float32, 2}(undef, 0, 0), targets=String[], sets=String[], rangeids=Int32[], endrix=Int32[])
    end

    batchsize = max(1, batchsize)
    probs = Array{Float32, 2}(undef, 0, 0)
    for start in 1:batchsize:nwindows
        stop = min(start + batchsize - 1, nwindows)
        x_batch = _lstm_window_tensor(contract, windowindex, start:stop)
        batch_probs = predict_lstm_trade_signals(model, x_batch)
        if size(probs, 1) == 0
            probs = Array{Float32, 2}(undef, size(batch_probs, 1), nwindows)
        end
        probs[:, start:stop] .= batch_probs
    end

    return (probs=probs, targets=windowindex.targets, sets=windowindex.sets, rangeids=windowindex.rangeids, endrix=windowindex.endrix)
end

"""
    predict_lstm_trade_signals(model, X::Array{Float32,3})::Array{Float32,2}

Inference function for trained LSTM trade signal classifier.

Takes a tensor of sequences and returns softmax probabilities for 4 trade classes.

# Arguments
- `model`: Trained LSTM Flux.Chain from train_lstm_trade_signals!
- `X::Array{Float32,3}`: Input tensor shape (nfeatures, seqlen, batch_size)

# Returns
- `probs::Array{Float32,2}`: Softmax probabilities shape (4, batch_size)
  - Row 1: P(longbuy)
  - Row 2: P(longclose)
  - Row 3: P(shortbuy)
  - Row 4: P(shortclose)

# Example
```julia
X = randn(Float32, 7, 3, 32)  # 7 features, seqlen=3, batch=32
probs = predict_lstm_trade_signals(model, X)
argmax(probs; dims=1)  # Predicted class per sample
```
"""
function predict_lstm_trade_signals(model, X::Array{Float32,3})::Array{Float32,2}
    @assert size(X, 1) > 0 && size(X, 2) > 0 && size(X, 3) > 0 "Invalid input shape: $X"

    Flux.testmode!(model)
    # Flux.reset!(model) # reset!(m) is deprecated. You can remove this call as it is no more needed.
    ŷ = model(X)  # (nclasses, seqlen, batch)
    ŷ_final = ŷ[:, end, :]  # Take final timestep: (nclasses, batch)
    probs = softmax(ŷ_final; dims=1)  # Apply softmax per sample

    return probs
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
