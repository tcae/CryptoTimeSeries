
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
    train_lstm_trade_signals!(contract::LstmBoundsTrendFeatures, seqlen::Int; 
                             hidden_dim::Int=32, maxepoch::Int=1000, batchsize::Int=64)

Train an LSTM trade signal classifier on contract data.

Follows the same training pattern as `adaptnn!()` in Classify:
- Uses Flux.DataLoader for batching with automatic shuffling
- Loss function: logitcrossentropy (expects raw logits, one-hot targets)
- Optimizer: Adam(0.001, (0.9, 0.999)) with stateful optimizer state
- Convergence: Stops if 5 consecutive epochs show non-decreasing loss
- **Saves model after every epoch** to BSON for recovery

# Arguments
- `contract::LstmBoundsTrendFeatures`: Contract with features, targets, sets
- `seqlen::Int`: Sliding window sequence length
- `hidden_dim::Int`: LSTM hidden dimension (default: 32)
- `maxepoch::Int`: Maximum training epochs (default: 1000)
- `batchsize::Int`: Training batch size (default: 64)

# Returns
- `model_state::NamedTuple`: `(model=trained_model, optim=optimizer_state, losses=loss_history, labels=classes)`

# Example
```julia
windows = Classify.lstm_tensor_windows(contract; seqlen=3)
result = Classify.train_lstm_trade_signals!(contract, 3; hidden_dim=32)
# result.model is trained for prediction
# result.losses is vector of epoch mean losses
```
"""
function train_lstm_trade_signals!(contract::LstmBoundsTrendFeatures, seqlen::Int; 
    hidden_dim::Int=32, maxepoch::Int=1000, batchsize::Int=64, labels::Union{Nothing,Vector{String}}=nothing)
    
    # Build sliding windows
    windows = lstm_tensor_windows(contract; seqlen=seqlen)
    X = windows.X  # (nfeatures, seqlen, nbatch)
    targets = windows.targets  # String labels per window
    sets = windows.sets  # "train", "eval", "test"
    
    @assert size(X, 3) > 0 "No training windows generated; check contract and seqlen"
    
    nfeatures = size(X, 1)
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
    
    # Initialize model and optimizer
    model, _ = lstm_trade_signal_model(nfeatures, seqlen, hidden_dim; labels=trade_labels)
    optim = Flux.setup(Flux.Adam(0.001, (0.9, 0.999)), model)
    
    # Partition into train/eval sets
    train_mask = sets .== "train"
    eval_mask = sets .== "eval"
    
    X_train = X[:, :, train_mask]  # (nfeatures, seqlen, ntrain)
    y_train = targets[train_mask]   # Vector of train labels
    
    X_eval = X[:, :, eval_mask]
    y_eval = targets[eval_mask]
    
    @assert size(X_train, 3) > 0 "No training samples in contract; check set partitioning"
    (verbosity >= 2) && println("$(EnvConfig.now()) LSTM training with $(size(X_train, 3)) training samples, $(size(X_eval, 3)) eval samples")
    
    # Convert targets to one-hot
    y_train_onehot = Flux.onehotbatch(y_train, trade_labels)  # (nclasses, ntrain)
    y_eval_onehot = Flux.onehotbatch(y_eval, trade_labels)    # (nclasses, neval)
    
    # Create DataLoader for training
    loader = Flux.DataLoader((X_train, y_train_onehot), batchsize=batchsize, shuffle=true)
    
    lossfunc = Flux.logitcrossentropy
    losses = Float32[]
    eval_losses = Float32[]
    
    Flux.trainmode!(model)
    testmode_original = false  # Track if we were in test mode initially
    
    @showprogress for epoch in 1:maxepoch
        # Training loop
        epoch_losses = Float32[]
        for (x_batch, y_batch) in loader
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x_batch)  # (nclasses, seqlen, batch) → we take last timestep
                ŷ_final = ŷ[:, end, :]  # Take final timestep: (nclasses, batch)
                lossfunc(ŷ_final, y_batch)
            end
            Flux.update!(optim, model, grads[1])
            push!(epoch_losses, loss)
        end
        
        # Compute epoch loss
        epoch_loss = mean(epoch_losses)
        push!(losses, epoch_loss)
        
        # Evaluate on eval set
        Flux.testmode!(model)
        ŷ_eval = model(X_eval)  # (nclasses, seqlen, neval)
        ŷ_eval_final = ŷ_eval[:, end, :]  # (nclasses, neval)
        eval_loss = lossfunc(ŷ_eval_final, y_eval_onehot)
        push!(eval_losses, eval_loss)
        Flux.trainmode!(model)
        
        if (verbosity >= 3)
            println("Epoch $epoch: train_loss=$(round(epoch_loss; digits=4)) eval_loss=$(round(eval_loss; digits=4))")
        end
        
        # Convergence check: 5 consecutive epochs of non-decreasing loss
        if length(losses) > 5 && 
           losses[end-4] <= losses[end-3] <= losses[end-2] <= losses[end-1] <= losses[end]
            (verbosity >= 2) && println("Converged at epoch $epoch (5 consecutive non-decreasing loss epochs)")
            break
        end
    end
    
    Flux.testmode!(model)
    (verbosity >= 2) && println("$(EnvConfig.now()) LSTM training complete with $(length(losses)) epochs")
    
    return (model=model, optim=optim, losses=losses, eval_losses=eval_losses, labels=trade_labels)
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
