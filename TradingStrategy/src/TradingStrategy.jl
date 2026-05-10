"""
Provides strategies to translate classifier predictions into trading actions.

- classifier shall provide
  - score acceptance thresholds for each label
  - mingain and maxgain per sample as limit inidications
  - maxwindow of predictions
  - maxpredictionlabel, maxscore, set, symbol (coin) per sample
- 
"""
module TradingStrategy

using DataFrames, Dates
using EnvConfig, Ohlcv, Features, Targets, Classify

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

"Defines the single symbol trading interface that shall be provided by all trading strategy implementations."
abstract type AbstractSingleSymbolTrading <: EnvConfig.AbstractConfiguration end

"Defines the multi symbol trading interface that shall be provided by all trading strategy implementations."
abstract type AbstractMultiSymbolTrading <: EnvConfig.AbstractConfiguration end

"Return the normalized config-scoped subfolder used for persisted trade artifacts."
function tradesfolder(stem::AbstractString="gains")::String
    normalized = replace(normpath(splitext(String(stem))[1]), '\\' => '/')
    return startswith(normalized, "trades/") || (normalized == "trades") ? normalized : joinpath("trades", normalized)
end

"Return the aggregate storage key used for one persisted trade artifact."
tradesaggregate(stem::AbstractString="gains") = joinpath("trades", splitext(basename(String(stem)))[1] * "_all")

"Return the per-coin storage key used for one persisted trade artifact."
tradefilename(coin::AbstractString; stem::AbstractString="gains") = joinpath(tradesfolder(stem), uppercase(strip(String(coin))))

"Persist a trades dataframe into the config-scoped `trades/` folder and also emit per-coin copies."
function savetrades(tradedf::AbstractDataFrame; stem::AbstractString="gains", include_aggregate::Bool=true)
    if size(tradedf, 1) == 0
        return String[]
    end
    @assert :coin in propertynames(tradedf) "tradedf must contain a :coin column; names=$(names(tradedf))"

    paths = String[]
    coins = unique(String.(tradedf[!, :coin]))
    for coin in coins
        coindf = DataFrame(tradedf[String.(tradedf[!, :coin]) .== coin, :])
        if size(coindf, 1) > 0
            push!(paths, EnvConfig.savedf(coindf, tradefilename(coin; stem=stem)))
        end
    end
    if include_aggregate
        push!(paths, EnvConfig.savedf(DataFrame(tradedf), tradesaggregate(stem)))
    end
    return paths
end

"Load persisted trade artifacts from the config-scoped `trades/` folder with aggregate-first fallback behavior."
function loadtrades(; stem::AbstractString="gains")
    aggregate = EnvConfig.readdf(tradesaggregate(stem))
    if !isnothing(aggregate) && (size(aggregate, 1) > 0)
        return DataFrame(aggregate)
    end

    folderpath = normpath(joinpath(EnvConfig.logfolder(), tradesfolder(stem)))
    isdir(folderpath) || return DataFrame()

    parts = DataFrame[]
    for entry in readdir(folderpath; join=false, sort=true)
        name = splitext(entry)[1]
        piece = EnvConfig.readdf(name; folderpath=folderpath)
        if !isnothing(piece) && (size(piece, 1) > 0)
            push!(parts, DataFrame(piece))
        end
    end
    return isempty(parts) ? DataFrame() : reduce(vcat, parts; cols=:union)
end

"Load the persisted trades for one specific coin from the config-scoped `trades/` folder."
function loadtrades(coin::AbstractString; stem::AbstractString="gains")
    tradedf = EnvConfig.readdf(tradefilename(coin; stem=stem))
    return isnothing(tradedf) ? DataFrame() : DataFrame(tradedf)
end

"Maps label strings to the trade label symbols used in Targets."
const _LSTM_LABEL_MAP = Dict(
    "longbuy" => longbuy,
    "longhold" => longhold,
    "longclose" => longclose,
    "shortbuy" => shortbuy,
    "shorthold" => shorthold,
    "shortclose" => shortclose,
    "allclose" => allclose,
)

"Maps coarse trend-phase class names to the corresponding `TrendPhase`."
const _LSTM_PHASE_MAP = Dict(
    "up" => up,
    "down" => down,
    "flat" => flat,
)

_mapsignal(label::TradeLabel) = label
_mapsignal(label::TrendPhase) = label
function _mapsignal(label)
    key = lowercase(strip(string(label)))
    if key in keys(_LSTM_LABEL_MAP)
        return _LSTM_LABEL_MAP[key]
    elseif key in keys(_LSTM_PHASE_MAP)
        return _LSTM_PHASE_MAP[key]
    end
    @assert false "unsupported LSTM label=$(label); expected one of $(sort!(vcat(collect(keys(_LSTM_LABEL_MAP)), collect(keys(_LSTM_PHASE_MAP)))))"
end

"Adds a coin with OhlcvData to the target generation. Each coin can only have 1 associated data set."
function setbase!(targets::AbstractSingleSymbolTrading, ohlcv::Ohlcv.OhlcvData) error("not implemented") end

function _mapleft!(d, tl, tlix, tll)
    for ix in tlix:-1:firstindex(tll)
        if tll[ix] in keys(d)
            d[tl] = d[tll[ix]] # mapped
            break
        end
    end
end

function _mapright!(d, tl, tlix, tll)
    for ix in tlix:lastindex(tll)
        if tll[ix] in keys(d)
            d[tl] = d[tll[ix]] # mapped
            break
        end
    end
end

"""
Transforms a named tuple of trade label names with score thresholds and returns a dict of it, which is subsequently used to look up a corresponding score threshold.  
Any missing trade label that is found in the predictions is mapped to a neighbor trade label.
"""
function unused_scoreacceptnt2dict(scorethreshold)::Dict
    d = Dict() # Dict([(tl, nothing) for tl in Targets.uniquelabels()])
    (verbosity >= 3) && println(scorethreshold)
    tls = keys(scorethreshold) 
    for tlsix in eachindex(tls)
        d[Targets.tradelabel(string(tls[tlsix]))] = scorethreshold[tls[tlsix]]
        (verbosity >= 3) && println("d[Targets.tradelabel(string(tls[tlsix]))=$(Targets.tradelabel(string(tls[tlsix])))] = scorethreshold[tls[tlsix]]=$(scorethreshold[tls[tlsix]])")
    end
    (verbosity >= 3) && println("dict of named tuple: $d")
    # map missing values to provided values
    tll = Targets.uniquelabels()
    tli = Int.(tll)
    for tl in tll
        if tl == ignore
            continue
        end
        if !(tl in keys(d))
            tlix = findfirst(x -> x == tl, tll)
            if Int(tl) < 0 # short
                _mapright!(d, tl, tlix, tll)
                if !(tl in keys(d))
                    _mapleft!(d, tl, tlix, tll)
                end
            else # long
                _mapleft!(d, tl, tlix, tll)
                if !(tl in keys(d))
                    _mapright!(d, tl, tlix, tll)
                end
            end
        end
    end
    (verbosity >= 3) && println("extended dict of named tuple including missing trade label score thresholds: $d")
    return d
end

"""
Transforms a named tuple of trade label names with score thresholds and returns a dict of it, which is subsequently used to look up a corresponding score threshold.  
Any missing trade label that is found in the predictions will result in a dict key error.
"""
function scorethresholdnt2dict(scorethreshold)::Dict
    d = Dict() # Dict([(tl, nothing) for tl in Targets.uniquelabels()])
    (verbosity >= 4) && println(scorethreshold)
    tls = keys(scorethreshold) 
    for tlsix in eachindex(tls)
        d[Targets.tradelabel(string(tls[tlsix]))] = scorethreshold[tls[tlsix]]
        (verbosity >= 4) && println("d[Targets.tradelabel(string(tls[tlsix]))=$(Targets.tradelabel(string(tls[tlsix])))] = scorethreshold[tls[tlsix]]=$(scorethreshold[tls[tlsix]])")
    end
    (verbosity >= 4) && println(d)
    return d
end

@inline islongopenlabel(label::TradeLabel) = (label == longbuy) || (label == longstrongbuy)
@inline isshortopenlabel(label::TradeLabel) = (label == shortbuy) || (label == shortstrongbuy)
@inline islongholdoropenlabel(label::TradeLabel) = (label == longhold) || islongopenlabel(label)
@inline isshortholdoropenlabel(label::TradeLabel) = (label == shorthold) || isshortopenlabel(label)
@inline islongcloselabel(label::TradeLabel) = (label == allclose) || (label == longstrongclose) || (label == longclose)
@inline isshortcloselabel(label::TradeLabel) = (label == shortclose) || (label == shortstrongclose) || (label == allclose)

# println("scorethresholdnt2dict((longbuy=0.8, allclose=0.5, shortbuy=0.7)) = $(scorethresholdnt2dict((longbuy=0.8,  allclose=0.5, shortbuy=0.7)))")

"""
This struct is used to convey input data to the selected trading startegy and to receive trade action instructions.  
Considers the price development of one symbol at a specific minute.  
The absolute amount is not determined by trading strategy but a factor of an unknown amount.

  - as input a short/long buy/close order can be open
    - if within limit price range it is being executed
    - it may be canceled
    - the limit may be changed
  - as output
    - a new short/long buy/close limit order may be created
    - a new short/long buy/close market order may be created
  
"""
mutable struct TradeAction
    orderlabel::Union{TradeLabel, Nothing} # used: longbuy, longclose, shortclose, shortbuy or ignore as default
    orderlimit::Float32 # order limit or 0f0 if no limit order placed
    buyprice::Float32 # price at which the position was opened; used for gain calculation and limit adjustments; 0f0 if no position open
    buyix::Integer # index of the bar at which the position was opened; used for gain calculation and limit adjustments; 0 if no position open
    function TradeAction(orderlabel::Union{TradeLabel, Nothing}=ignore, orderlimit=0f0, buyprice=0f0, buyix=0)
        ta = new(orderlabel, orderlimit, buyprice, buyix)
        isopen(ta)
        return ta
    end
end

"Backward-compatible constructor used by LSTM trade mapping."
function TradeAction(cancelrunningorder::Bool, orderlabel::Union{TradeLabel, Nothing}, orderlimit::Union{Float32, Nothing}, amountfactor::Float32)
    _ = cancelrunningorder
    _ = amountfactor
    limit = isnothing(orderlimit) ? 0f0 : orderlimit
    return TradeAction(orderlabel, limit, 0f0, 0)
end

"Compatibility shim for legacy code paths that still read removed TradeAction fields."
function Base.getproperty(ta::TradeAction, name::Symbol)
    if name === :cancelrunningorder
        return false
    elseif name === :amountfactor
        return 1f0
    end
    return getfield(ta, name)
end

"Compatibility shim for legacy code paths that still write removed TradeAction fields."
function Base.setproperty!(ta::TradeAction, name::Symbol, value)
    if (name === :cancelrunningorder) || (name === :amountfactor)
        return value
    elseif (name === :orderlimit) && isnothing(value)
        setfield!(ta, :orderlimit, 0f0)
        return 0f0
    end
    setfield!(ta, name, value)
    return value
end

function isopen(ta::TradeAction)
    if ta.orderlimit > 0f0
        @assert (ta.orderlabel != ignore) && (ta.orderlimit > 0f0) && (ta.buyprice > 0f0) "(ta.orderlabel != ignore) && (ta.orderlimit > 0f0) && (ta.buyprice > 0f0); got orderlabel=$(ta.orderlabel), orderlimit=$(ta.orderlimit), buyprice=$(ta.buyprice)"
    end
    return ta.orderlimit > 0f0
end

function removeorder!(ta::TradeAction)
    ta.orderlabel = ignore
    ta.orderlimit = 0f0
    ta.buyprice = 0f0
    ta.buyix = 0
    return ta
end

"""
LSTM-based action decider that maps class probabilities to trade actions.

The expected default class order is:
`["longbuy", "longclose", "shortbuy", "shortclose"]`.
"""
mutable struct LstmTradeDecider <: AbstractSingleSymbolTrading
    labels::Vector{Any}
    scorethresholds::Dict{Any, Float32}
    fallbacklabel::Union{TradeLabel, TrendPhase}
    ohlcv::Union{Nothing, Ohlcv.OhlcvData}
end

"""
Create a new `LstmTradeDecider`.

# Arguments
- `labels`: model output labels as strings, aligned with probability vector order;
  supports both legacy trade labels and coarse phase labels `up/down/flat`
- `scorethresholds`: per-label acceptance score thresholds
- `fallbacklabel`: label or phase used when confidence is below threshold
"""
function LstmTradeDecider(; labels=["longbuy", "longclose", "shortbuy", "shortclose"], scorethresholds=(longbuy=0.5f0, longclose=0.5f0, shortbuy=0.5f0, shortclose=0.5f0), fallbacklabel::Union{TradeLabel, TrendPhase}=allclose)
    @assert length(labels) > 0 "labels length must be > 0; got $(length(labels))"
    mapped = Any[]
    for label in labels
        push!(mapped, _mapsignal(label))
    end
    thresholds = Dict{Any, Float32}()
    for (k, v) in pairs(scorethresholds)
        thresholds[_mapsignal(String(k))] = Float32(v)
    end
    return LstmTradeDecider(mapped, thresholds, fallbacklabel, nothing)
end

"Stores OHLCV reference for interface compatibility with single-symbol strategies."
function setbase!(decider::LstmTradeDecider, ohlcv::Ohlcv.OhlcvData)
    decider.ohlcv = ohlcv
    return decider
end

"Resolves the final order label to be emitted for a selected trade label."
function _label2orderlabel(label::TradeLabel, assettype::TrendPhase)
    if label in [longbuy, longclose, shortbuy, shortclose]
        return label
    elseif label == allclose
        if assettype == up
            return longclose
        elseif assettype == down
            return shortclose
        else
            return nothing
        end
    else
        return nothing
    end
end

"Resolves a coarse phase prediction to the corresponding trade action for the current asset state."
function _label2orderlabel(phase::TrendPhase, assettype::TrendPhase)
    if phase == up
        if assettype == down
            return shortclose
        elseif assettype == flat
            return longbuy
        else
            return nothing
        end
    elseif phase == down
        if assettype == up
            return longclose
        elseif assettype == flat
            return shortbuy
        else
            return nothing
        end
    elseif phase == flat
        if assettype == up
            return longclose
        elseif assettype == down
            return shortclose
        else
            return nothing
        end
    else
        return nothing
    end
end

"""
Maps a single probability vector to one trade action.

The max-probability class is selected first. If its probability is below the
configured class threshold, `fallbacklabel` is used.
"""
function lstm_trade_action(decider::LstmTradeDecider, probs::AbstractVector{<:Real}; assettype::TrendPhase=flat)
    @assert length(probs) == length(decider.labels) "length(probs)=$(length(probs)) must equal number of decider labels=$(length(decider.labels))"
    predix = argmax(probs)
    predlabel = decider.labels[predix]
    predscore = Float32(probs[predix])
    threshold = get(decider.scorethresholds, predlabel, 0.5f0)
    selected = predscore >= threshold ? predlabel : decider.fallbacklabel
    orderlabel = _label2orderlabel(selected, assettype)
    return TradeAction(false, orderlabel, nothing, 1f0)
end

"Maps all probability columns `(nclasses, nsamples)` to one `TradeAction` per sample."
function lstm_trade_actions(decider::LstmTradeDecider, probsmat::AbstractMatrix{<:Real}; assettype::TrendPhase=flat)
    @assert size(probsmat, 1) == length(decider.labels) "size(probsmat, 1)=$(size(probsmat, 1)) must equal number of decider labels=$(length(decider.labels))"
    actions = Vector{TradeAction}(undef, size(probsmat, 2))
    for ix in 1:size(probsmat, 2)
        actions[ix] = lstm_trade_action(decider, view(probsmat, :, ix), assettype=assettype)
    end
    return actions
end

"""
Runs the trained Classify LSTM model on tensor windows and maps probabilities to
trade actions.

Returns a named tuple with `actions` and `probs`.
"""
function lstm_trade_actions(decider::LstmTradeDecider, model, X::Array{Float32,3}; assettype::TrendPhase=flat)
    probs = Classify.predict_lstm_trade_signals(model, X)
    actions = lstm_trade_actions(decider, probs; assettype=assettype)
    return (actions=actions, probs=probs)
end

"""
Builds windows from an LSTM contract, predicts probabilities using a
trained Classify LSTM model, and maps probabilities to trade actions.

Returns a named tuple with `actions`, `probs`, and aligned window metadata
(`targets`, `sets`, `rangeids`, `endrix`).
"""
function lstm_trade_actions(decider::LstmTradeDecider, model, contract::Classify.LstmBoundsTrendFeatures; seqlen::Int, assettype::TrendPhase=flat)
    predres = Classify.predict_lstm_trade_signals(model, contract; seqlen=seqlen)
    if size(predres.probs, 2) == 0
        return (
            actions=TradeAction[],
            probs=Array{Float32}(undef, length(decider.labels), 0),
            targets=String[],
            sets=String[],
            rangeids=Int32[],
            endrix=Int32[],
        )
    end
    actions = lstm_trade_actions(decider, predres.probs; assettype=assettype)
    return (
        actions=actions,
        probs=predres.probs,
        targets=predres.targets,
        sets=predres.sets,
        rangeids=predres.rangeids,
        endrix=predres.endrix,
    )
end

"Returns the predicted lower/upper band for one row of a predictions dataframe."
function _predicted_band(predictionsdf::AbstractDataFrame, ix::Integer)
    centercol = :centerpred in propertynames(predictionsdf) ? :centerpred : (:pred_center in propertynames(predictionsdf) ? :pred_center : nothing)
    widthcol = :widthpred in propertynames(predictionsdf) ? :widthpred : (:pred_width in propertynames(predictionsdf) ? :pred_width : nothing)
    @assert !isnothing(centercol) && !isnothing(widthcol) "predictionsdf must contain centerpred/widthpred or pred_center/pred_width; names=$(names(predictionsdf))"
    center = Float32(predictionsdf[ix, centercol])
    width = max(Float32(predictionsdf[ix, widthcol]), 0f0)
    halfwidth = width / 2f0
    return (center - halfwidth, center + halfwidth)
end

@inline _price_in_bar(price::Float32, low::Real, high::Real) = (Float32(low) <= price) && (price <= Float32(high))

"""
Convert a phase sequence (`up`, `down`, `flat`) into lifecycle trade labels.

The first directional sample emits a buy label, continued directional samples emit
hold labels, and returning to `flat` emits the corresponding close label.
Direct reversals emit the opposite open signal, which the simulator interprets as
close-and-reverse logic at the same close price.
"""
function phase_sequence_trade_labels(phases::AbstractVector)
    tradelabels = fill(allclose, length(phases))
    prevphase = flat
    for ix in eachindex(phases)
        phase = _mapsignal(phases[ix])
        if phase isa TradeLabel
            tradelabels[ix] = phase
            if islongopenlabel(phase) || (phase == longhold)
                prevphase = up
            elseif isshortopenlabel(phase) || (phase == shorthold)
                prevphase = down
            elseif islongcloselabel(phase) || isshortcloselabel(phase)
                prevphase = flat
            end
            continue
        end

        if phase == up
            tradelabels[ix] = prevphase == up ? longhold : longbuy
        elseif phase == down
            tradelabels[ix] = prevphase == down ? shorthold : shortbuy
        else
            tradelabels[ix] = prevphase == up ? longclose : (prevphase == down ? shortclose : allclose)
        end
        prevphase = phase
    end
    return tradelabels
end

_normalize_limit_labels(labels::AbstractVector) = all(label -> label isa TradeLabel, labels) ? collect(labels) : phase_sequence_trade_labels(labels)

"Returns an empty dataframe for limit-aware entry/exit trade pairs."
function emptytradepairdf()::DataFrame
    return DataFrame(
        trend=TrendPhase[],
        samplecount=Int[],
        minutes=Int[],
        gain=Float32[],
        gainfee=Float32[],
        startdt=DateTime[],
        enddt=DateTime[],
        startix=Int[],
        endix=Int[],
        entryprice=Float32[],
        exitprice=Float32[],
        entrylimit=Float32[],
        exitlimit=Float32[],
        entryfilled=Bool[],
        exitfilled=Bool[],
        missedexit=Bool[],
        entryreason=String[],
        exitreason=String[],
    )
end

"Appends one realized limit-aware trade pair to the simulation dataframe."
function _pushtradepair!(tradedf::DataFrame, trend::TrendPhase, predictionsdf::AbstractDataFrame, startix::Integer, endix::Integer, entryprice::Float32, exitprice::Float32, entrylimit::Float32, exitlimit::Float32, entryfee::Float32, exitfee::Float32, entryfilled::Bool, exitfilled::Bool, missedexit::Bool, entryreason::AbstractString, exitreason::AbstractString)
    @assert (trend == up) || (trend == down) "trend must be up or down; got trend=$(trend)"
    startdt = predictionsdf[startix, :opentime]
    enddt = predictionsdf[endix, :opentime]
    rawgain = trend == up ? (exitprice - entryprice) / entryprice : -(exitprice - entryprice) / entryprice
    minutes = Int(div(Dates.value(enddt - startdt), 60000)) + 1
    push!(tradedf, (
        trend,
        Int(endix - startix + 1),
        minutes,
        rawgain,
        rawgain - entryfee - exitfee,
        startdt,
        enddt,
        Int(startix),
        Int(endix),
        entryprice,
        exitprice,
        entrylimit,
        exitlimit,
        entryfilled,
        exitfilled,
        missedexit,
        String(entryreason),
        String(exitreason),
    ))
    return tradedf
end

"Returns the initial state used by `simulate_limit_trade_pairs`."
function _initial_limit_state()
    return (
        assettype=flat,
        buyix=0,
        buyprice=0f0,
        entrylimit=Float32(NaN),
        entryfee=0f0,
        pendingentry=false,
        pendingentryix=0,
        pendingentrytrend=flat,
        pendingentrylimit=Float32(NaN),
        pendingexit=false,
        pendingexitix=0,
        pendingexitlimit=Float32(NaN),
    )
end

"""
    simulate_limit_trade_pairs(predictionsdf, scores, labels; ...)

Simulate explicit entry/exit trade pairs from row-aligned LSTM predictions and
bounds estimates. Entry limits are placed at the predicted lower band for long
trades and upper band for short trades. Exit limits use the opposite band. If an
exit limit is not hit within `exittimeout` bars, the position is force-closed at
market, so missed exits can realize losses.

`labels` may either be direct trade labels (`longbuy`, `shortclose`, ...) or a
coarse trend-phase sequence (`up`, `down`, `flat`). Phase sequences are converted
internally so that actions are only emitted when a trend starts or ends.
"""
function simulate_limit_trade_pairs(predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector; openthreshold::AbstractFloat=0.6f0, closethreshold::AbstractFloat=0.5f0, entrytimeout::Int=2, exittimeout::Int=2, makerfee::AbstractFloat=0.0015f0, takerfee::AbstractFloat=0.002f0, exitstrategy::Symbol=:opposite_signal_market, forceclose::Bool=true)
    nrows = size(predictionsdf, 1)
    normlabels = _normalize_limit_labels(labels)
    @assert nrows == length(scores) == length(normlabels) "size(predictionsdf, 1)=$nrows must match length(scores)=$(length(scores)) and length(labels)=$(length(normlabels))"
    @assert entrytimeout >= 0 "entrytimeout=$(entrytimeout) must be >= 0"
    @assert exittimeout >= 0 "exittimeout=$(exittimeout) must be >= 0"
    @assert exitstrategy in (:opposite_signal_market,) "unsupported exitstrategy=$exitstrategy; expected :opposite_signal_market"
    @assert all(c -> c in propertynames(predictionsdf), [:opentime, :high, :low, :close]) "predictionsdf must contain opentime/high/low/close; names=$(names(predictionsdf))"

    tradedf = emptytradepairdf()
    if nrows == 0
        return tradedf
    end

    state = _initial_limit_state()

    function open_position!(ix::Int, trend::TrendPhase, price::Float32, limit::Float32)
        state = merge(state, (assettype=trend, buyix=ix, buyprice=price, entrylimit=limit, entryfee=Float32(makerfee), pendingentry=false, pendingentryix=0, pendingentrytrend=flat, pendingentrylimit=Float32(NaN), pendingexit=false, pendingexitix=0, pendingexitlimit=Float32(NaN)))
        return state
    end

    function close_position!(ix::Int, exitprice::Float32, exitlimit::Float32, exitfilled::Bool, missedexit::Bool, exitreason::String)
        if state.assettype in (up, down)
            exitfee = exitfilled ? Float32(makerfee) : Float32(takerfee)
            _pushtradepair!(tradedf, state.assettype, predictionsdf, state.buyix, ix, state.buyprice, exitprice, state.entrylimit, exitlimit, state.entryfee, exitfee, true, exitfilled, missedexit, "limit_fill", exitreason)
        end
        state = merge(state, (assettype=flat, buyix=0, buyprice=0f0, entrylimit=Float32(NaN), entryfee=0f0, pendingexit=false, pendingexitix=0, pendingexitlimit=Float32(NaN)))
        return state
    end

    @inbounds for ix in 1:nrows
        low = Float32(predictionsdf[ix, :low])
        high = Float32(predictionsdf[ix, :high])
        close = Float32(predictionsdf[ix, :close])
        label = normlabels[ix]
        score = Float32(scores[ix])
        lower, upper = _predicted_band(predictionsdf, ix)

        longopen = islongopenlabel(label) && (score >= openthreshold)
        shortopen = isshortopenlabel(label) && (score >= openthreshold)
        longclose = islongcloselabel(label) && (score >= closethreshold)
        shortclose = isshortcloselabel(label) && (score >= closethreshold)

        if state.pendingentry
            if _price_in_bar(state.pendingentrylimit, low, high)
                state = open_position!(ix, state.pendingentrytrend, state.pendingentrylimit, state.pendingentrylimit)
            elseif (ix - state.pendingentryix) >= entrytimeout
                state = merge(state, (pendingentry=false, pendingentryix=0, pendingentrytrend=flat, pendingentrylimit=Float32(NaN)))
            elseif ((state.pendingentrytrend == up) && shortopen) || ((state.pendingentrytrend == down) && longopen)
                state = merge(state, (pendingentry=false, pendingentryix=0, pendingentrytrend=flat, pendingentrylimit=Float32(NaN)))
            end
        end

        if (state.assettype in (up, down)) && state.pendingexit
            if _price_in_bar(state.pendingexitlimit, low, high)
                state = close_position!(ix, state.pendingexitlimit, state.pendingexitlimit, true, false, "limit_fill")
            elseif (ix - state.pendingexitix) >= exittimeout
                state = close_position!(ix, close, state.pendingexitlimit, false, true, "timeout_market")
            end
        end

        if state.assettype == flat
            if !state.pendingentry
                if longopen
                    state = merge(state, (pendingentry=true, pendingentryix=ix, pendingentrytrend=up, pendingentrylimit=lower))
                    if _price_in_bar(state.pendingentrylimit, low, high)
                        state = open_position!(ix, up, state.pendingentrylimit, state.pendingentrylimit)
                    end
                elseif shortopen
                    state = merge(state, (pendingentry=true, pendingentryix=ix, pendingentrytrend=down, pendingentrylimit=upper))
                    if _price_in_bar(state.pendingentrylimit, low, high)
                        state = open_position!(ix, down, state.pendingentrylimit, state.pendingentrylimit)
                    end
                end
            end
        elseif state.assettype == up
            if longclose
                state = merge(state, (pendingexit=true, pendingexitix=ix, pendingexitlimit=upper))
                if _price_in_bar(state.pendingexitlimit, low, high)
                    state = close_position!(ix, state.pendingexitlimit, state.pendingexitlimit, true, false, "limit_fill")
                end
            elseif shortopen
                missedexit = state.pendingexit
                exitlimit = state.pendingexit ? state.pendingexitlimit : Float32(NaN)
                state = close_position!(ix, close, exitlimit, false, missedexit, "opposite_signal_market")
                state = merge(state, (pendingentry=true, pendingentryix=ix, pendingentrytrend=down, pendingentrylimit=upper))
                if _price_in_bar(state.pendingentrylimit, low, high)
                    state = open_position!(ix, down, state.pendingentrylimit, state.pendingentrylimit)
                end
            end
        else
            if shortclose
                state = merge(state, (pendingexit=true, pendingexitix=ix, pendingexitlimit=lower))
                if _price_in_bar(state.pendingexitlimit, low, high)
                    state = close_position!(ix, state.pendingexitlimit, state.pendingexitlimit, true, false, "limit_fill")
                end
            elseif longopen
                missedexit = state.pendingexit
                exitlimit = state.pendingexit ? state.pendingexitlimit : Float32(NaN)
                state = close_position!(ix, close, exitlimit, false, missedexit, "opposite_signal_market")
                state = merge(state, (pendingentry=true, pendingentryix=ix, pendingentrytrend=up, pendingentrylimit=lower))
                if _price_in_bar(state.pendingentrylimit, low, high)
                    state = open_position!(ix, up, state.pendingentrylimit, state.pendingentrylimit)
                end
            end
        end
    end

    if forceclose && (state.assettype in (up, down))
        lastix = nrows
        lastclose = Float32(predictionsdf[lastix, :close])
        missedexit = state.pendingexit
        exitlimit = state.pendingexit ? state.pendingexitlimit : Float32(NaN)
        reason = missedexit ? "final_market_after_missed_limit" : "final_market"
        close_position!(lastix, lastclose, exitlimit, false, missedexit, reason)
    end

    return tradedf
end

"""
    simulate_market_trade_pairs(predictionsdf, scores, labels; ...)

Simulate trade pairs using the row `:close` price whenever an open or close
signal is issued. `labels` may be direct trade labels or the coarse phase labels
`up`, `down`, and `flat`.
"""
function simulate_market_trade_pairs(predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector; openthreshold::AbstractFloat=0.6f0, closethreshold::AbstractFloat=0.5f0, makerfee::AbstractFloat=0.0015f0, takerfee::AbstractFloat=0.002f0, forceclose::Bool=true)
    nrows = size(predictionsdf, 1)
    normlabels = _normalize_limit_labels(labels)
    @assert nrows == length(scores) == length(normlabels) "size(predictionsdf, 1)=$nrows must match length(scores)=$(length(scores)) and length(labels)=$(length(normlabels))"
    @assert all(c -> c in propertynames(predictionsdf), [:opentime, :close]) "predictionsdf must contain opentime/close; names=$(names(predictionsdf))"

    if nrows == 0
        return emptygaindf()
    end

    gs = GainSegment(
        ;
        maxwindow=max(1, nrows),
        openthreshold=Float32(openthreshold),
        closethreshold=Float32(closethreshold),
        algorithm=algorithm02!,
        makerfee=Float32(makerfee),
        takerfee=Float32(takerfee),
    )
    gdf = getgains(gs, predictionsdf, Float32.(scores), normlabels, forceclose; lastix=nrows, openthreshold=Float32(openthreshold), closethreshold=Float32(closethreshold))
    return copy(gdf)
end

function emptygaindf()::DataFrame
    return DataFrame(
        trend=TrendPhase[],
        samplecount=Int[],
        minutes=Int[],
        gain=Float32[],
        gainfee=Float32[],
        startdt=DateTime[],
        enddt=DateTime[],
        startix=Int[],
        endix=Int[],
    )
end

mutable struct GainSegment
    algorithm  # algorithm function to be applied
    openthreshold::Float32 # score threshold
    closethreshold::Float32 # score threshold
    # scorethreshold::Dict # score threshold dict
    maxwindow::Integer # maxwindow of sample window considered for the prediction
    endix::Integer # is the index of the last analyzed row of scores/labels/predictionsdf
    gaindf::DataFrame # dataframe to save the identified gain segments with their start/end index, start/end time, gain, gain - fee, sample count, minutes, trend
    makerfee::Float32
    takerfee::Float32
    scores::Union{AbstractVector, Nothing}
    labels::Union{AbstractVector, Nothing}
    predictionsdf::Union{AbstractDataFrame, Nothing}
    lastix::Integer # last inspected row index of last open gain segment that is not yet saved in gainsdf
    buygain::Float32 # relative gain compared to current price for limit order
    sellgain::Float32 # relative gain compared to current price for limit order
    limitreduction::Float32 # factor to reduce limit price in case of unfilled order for every trend sample with a label that does not support the trend (e.g. allclose label for an open long gain segment)
    buyta::TradeAction # buy of either long or short
    sellta::TradeAction # sell of either long or short
    function GainSegment(;maxwindow::Integer=4*60, openthreshold=0.6, closethreshold=0.5, algorithm=algorithm02!, makerfee::AbstractFloat=0f0, takerfee::AbstractFloat=0f0, limitreduction::AbstractFloat=0f0)
        return new(algorithm, openthreshold, closethreshold, maxwindow, 0, emptygaindf(), makerfee, takerfee, nothing, nothing, nothing, 0, 0.001f0, 0.01f0, limitreduction, TradeAction(), TradeAction())
    end
end

function reset!(gs::GainSegment)
    gs.predictionsdf = nothing
    gs.scores = nothing
    gs.labels = nothing
    gs.endix = gs.lastix = 0
    gs.gaindf = emptygaindf()
    removeorder!(gs.sellta)
    removeorder!(gs.buyta)
    return gs
end

isopensegment(gs::GainSegment) = isopen(gs.sellta)
assettrend(ta::TradeAction) = ta.orderlimit > ta.buyprice ? up : (ta.orderlimit < ta.buyprice ? down : flat)

"Calculates gain for an open segment and closes it. An open segment must have buyix > 0, buyprice != nothing"
function calcgain!(gs::GainSegment, sellix::Integer, sellprice::Float32=gs.predictionsdf[sellix, :close])
    if isopen(gs.sellta)
        starttime = gs.predictionsdf[gs.sellta.buyix, :opentime]
        ixtime = gs.predictionsdf[sellix, :opentime]
        trend = assettrend(gs.sellta)
        if trend == up  # long sell
            gain = (sellprice - gs.sellta.buyprice) / gs.sellta.buyprice
        else  # short sell
            gain = -(sellprice - gs.sellta.buyprice) / gs.sellta.buyprice  # down trend -> negative price diff -> gain shall be positive for a short trade
        end
        minutes = Int(div(Dates.value(ixtime - starttime), 60000)) + 1 # from milliseconds to minutes, add 1 to count the current minute as well
        push!(gs.gaindf, (trend, (sellix - gs.sellta.buyix + 1), minutes, gain, (gain - 2f0 * gs.makerfee), starttime, ixtime, gs.sellta.buyix, sellix))
        removeorder!(gs.sellta)
    end
    return gs
end

"""
Consumes the prediction results of a range (1 coin, 1 settype) and calculates the corresponding gain segments that are stored in gaindf.  
Hold as long as trend is supported above closethreshold by hold, buy, strongbuy, otherwise close
"""
function algorithm01!(gs::GainSegment, lastix)
    labels = gs.labels
    scores = gs.scores
    closecol = gs.predictionsdf[!, :close]
    lasttrend = flat
    @inbounds for ix in (gs.endix + 1):lastix
        #TODO first check whether the limit order is executed, if so, set trend and buyix and selllimit
        #TODO decide how long the limit order should stay and what limit corrections are required
        label = labels[ix]
        score = scores[ix]
        thistrend = lasttrend
        thisbuyix = 0
        if (lasttrend == up) && islongholdoropenlabel(label) && (score > gs.closethreshold)
            thistrend = up
        elseif islongopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif (lasttrend == down) && isshortholdoropenlabel(label) && (score > gs.closethreshold)
            thistrend = down
        elseif isshortopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end
        if lasttrend != thistrend
            calcgain!(gs, ix)
            if thistrend == up
                gs.sellta = TradeAction(longclose, closecol[ix] * (1f0 + gs.sellgain), closecol[ix], thisbuyix)
            elseif thistrend == down
                gs.sellta = TradeAction(shortclose, closecol[ix] * (1f0 - gs.sellgain), closecol[ix], thisbuyix)
            else
                removeorder!(gs.sellta)
            end
            lasttrend = thistrend
        end
    end
    gs.endix = lastix
    return gs
end

"""
Consumes the prediction results of a range (1 coin, 1 settype) and calculates the corresponding gain segments that are stored in gaindf.  
Hold as long as trend is not broken by opposite buy or close above openthreshold.
"""
function algorithm02!(gs::GainSegment, lastix)
    labels = gs.labels
    scores = gs.scores
    closecol = gs.predictionsdf[!, :close]
    lasttrend = flat
    @inbounds for ix in (gs.endix + 1):lastix
        #TODO first check whether the limit order is executed, if so, set trend and buyix and selllimit
        #TODO decide how long the limit order should stay and what limit corrections are required
        label = labels[ix]
        score = scores[ix]
        thistrend = lasttrend
        thisbuyix = 0
        if (lasttrend == up) && islongcloselabel(label) && (score > gs.openthreshold)
            thistrend = flat
        elseif (lasttrend == down) && isshortcloselabel(label) && (score > gs.openthreshold)
            thistrend = flat
        elseif islongopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif isshortopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end
        if lasttrend != thistrend
            calcgain!(gs, ix)
            if thistrend == up
                gs.sellta = TradeAction(longclose, closecol[ix] * (1f0 + gs.sellgain), closecol[ix], thisbuyix)
            elseif thistrend == down
                gs.sellta = TradeAction(shortclose, closecol[ix] * (1f0 - gs.sellgain), closecol[ix], thisbuyix)
            else
                removeorder!(gs.sellta)
            end
            lasttrend = thistrend
        end
    end
    gs.endix = lastix
    return gs
end

"check price ranges whether an open order is closed and calculates the gain"
function processclosedorder!(gs::GainSegment, ix::Integer, ta::TradeAction)
    if isopen(ta) 
        if (ta.orderlabel in [longbuy, longstrongbuy]) 
            if (gs.predictionsdf[ix, :low] <= ta.buyprice)
                # assume trade took place
                ta.orderlabel = longclose
                ta.buyix = ix
            end
        elseif (ta.orderlabel in [shortbuy, shortstrongbuy]) 
            if (gs.predictionsdf[ix, :high] >= ta.buyprice)
                # assume trade took place
                ta.orderlabel = shortclose
                ta.buyix = ix
            end
        elseif (ta.orderlabel in [longclose, longstrongclose]) 
            if (ta.orderlimit <= gs.predictionsdf[ix, :high])                
                # assume trade took place
                calcgain!(gs, ix, ta.orderlimit) # if segment close then also resets buyix and buyprice
                removeorder!(ta)
            end
        elseif (ta.orderlabel in [shortclose, shortstrongclose]) 
            if (gs.predictionsdf[ix, :low] <= ta.orderlimit)
                # assume trade took place
                calcgain!(gs, ix, ta.orderlimit) # if segment close then also resets buyix and buyprice
                removeorder!(ta)
            end
        end
    end
end

islongclose(label::TradeLabel) = label in [allclose, longclose, longstrongclose]
isshortclose(label::TradeLabel) = label in [allclose, shortclose, shortstrongclose]

islongclose(ta::TradeAction) = islongclose(ta.orderlabel)
isshortclose(ta::TradeAction) = isshortclose(ta.orderlabel)

"""
Receives a sell and buy TradeAction and the current label + score and checks whether the current label supports the trend of the active TradeAction. 
If not, it reduces the limit price of sell trade action with a factor of limitreduction for every non trend supporting sample. 
If the current label signals an opposite trend then a reversal is signalled and an opposite buy action is proposed.
If the current label supports the current trend with a buy then the limit is adjusted as gain of the current close.

configuration parameters are:
- openthreshold: minimum score to consider a buy signal as valid
- buygain: initial gain for buy limit price compared to current close price
- sellgain: initial gain for sell limit price compared to current close price
- limitreduction: factor to reduce limit price in case of unfilled order for every trend sample with a label that does not support the trend (e.g. allclose label for a long sell action)
"""
function reachgainuntilreversal!(sellta::TradeAction, buyta::TradeAction, label::TradeLabel, score, high, low, close, openthreshold, buygain, sellgain, limitreduction)
    if (label in [longbuy, longstrongbuy]) && (score >= openthreshold)
        if isopen(sellta) 
            if isshortclose(sellta.orderlabel)
                sellta.orderlimit = close * (1f0 - buygain) # adapt to buy limit of opposite order due to trend reversal
            elseif islongclose(sellta.orderlabel)
                sellta.orderlimit = close * (1f0 + sellgain)
            end
        else  # no sell order but if buy order present then it did not reach buy limit and need to be reestablihed with new buy limit
            buyta.buyprice = close * (1f0 - buygain)
            buyta.orderlabel = longbuy
            buyta.orderlimit = close * (1f0 + sellgain)
        end
    elseif (label in [shortbuy, shortstrongbuy]) && (score >= openthreshold)
        if isopen(sellta) 
            if islongclose(sellta.orderlabel)
                sellta.orderlimit = close * (1f0 + buygain) # adapt to buy limit of opposite order due to trend reversal
            elseif isshortclose(sellta.orderlabel)
                sellta.orderlimit = close * (1f0 - sellgain)
            end
        else  # no sell order but if buy order present then it did not reach buy limit and need to be reestablihed with new buy limit
            buyta.buyprice = close * (1f0 + buygain)
            buyta.orderlabel = shortbuy
            buyta.orderlimit = close * (1f0 - sellgain)
        end
    elseif isopen(sellta)
        if (sellta.orderlimit > sellta.buyprice) && (sellta.orderlabel != longhold) # long order
            # limit order in place, incrementally reduce limit due to missing label support until it reaches the current close price
            sellta.orderlimit = max(close, sellta.orderlimit * (1f0 - sellgain * limitreduction))
        elseif (sellta.orderlimit < sellta.buyprice) && (sellta.orderlabel != shorthold) # short order
            # limit order in place, incrementally reduce limit due to missing label support until it reaches the current close price
            sellta.orderlimit = min(close, sellta.orderlimit * (1f0 + sellgain * limitreduction))
        end
    end
end

"""
Consumes the next prediction results after endix for all remaining predictions and calculates the corresponding gain segments that are stored in gaindf.  
limitreduction is reducing the expected gain for each non trend supporting sample, which is used to adjust the limit price for buy and sell orders.
Buy will buy at current price +- buygain, which is reduced with a limitreduction rate for non trend supporting samples. This limit is newly set when a buy signal for that trend is observed.
A bought sample receives a sell limit at current price +- sellgain, which is reduced with a limitreduction rate for non trend supporting samples.  
Hold as long as trend is not broken by opposite buy or close above openthreshold. If limit is hit then close gain segment and open new one if buy signal is present.  
Implementation details:  
- labels are indicating a trend change, but the gain materializes after the corresponding order is closed
- buygain and sellgain are defining the initial limit distance to the current price, which is reduced with a limitreduction rate for non trend supporting samples utilizing the price volatility.
"""
function algorithm03!(gs::GainSegment, lastix)
    # @assert length(scores) == length(labels) == size(predictionsdf, 1) > 0 "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1)) > 0"
    for ix in (gs.endix+1):lastix
        # first check whether the limit order is executed, if so, set trend and buyix and selllimit
        processclosedorder!(gs, ix, gs.sellta)
        processclosedorder!(gs, ix, gs.buyta)
        if gs.buyta.orderlabel in [longclose, shortclose] # buy trade was executed
            @assert gs.sellta.orderlabel == ignore "inconsistency: if buyta.orderlabel=$(gs.buyta.orderlabel) is a close order then sellta.orderlabel=$(gs.sellta.orderlabel) must be ignore"
            gs.sellta = gs.buyta
            gs.buyta = TradeAction()
        end
        reachgainuntilreversal!(gs.sellta, gs.buyta, gs.labels[ix], gs.scores[ix], gs.predictionsdf[ix, :high], gs.predictionsdf[ix, :low], gs.predictionsdf[ix, :close], gs.openthreshold, gs.buygain, gs.sellgain, gs.limitreduction)
    end
    gs.endix = lastix
end

"""
Returns a dataframe with gains of buy/sell pair actions until lastix. If `forcegain` == true then add a row that forces to calculate the last open gain segment.
The returned dataframe has the following columns:
- trend (up, flat, down)  
- samplecount of gain segment  
- minutes of gain segment  
- gain of gain segment
- startdt and enddt :: DateTime of gain segment
- startix, endix within predictionsdf rows of gain segment

Input is 
-  predictionsdf::AbstractDataFrame of a consecutive range with the following colums
    - label as TradeLabel
    - score as Float32 prediction score
    - target as TradeLabel
    - mingain as expected % below pivot price to set a trade limit
    - maxgain as expected % above pivot price to set a trade limit
    - downminutes as expected down trend time in minutes
    - upminutes as expected up trend time in minutes
    - lastextrememinutes as minutes since last extreme
    - rangeid as indicator which samples belong to a sequence
    - opentime, high, low, close, pivot (samples of the same range are expected to be in opentime sorting order)
    - set as CategoricalVector with levels train, test, eval
    - coin as CategoricalVector
"""
function getgains(gs::GainSegment, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector, forcegain::Bool; lastix=lastindex(scores), openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
    @assert length(scores) == length(labels) == size(predictionsdf, 1) "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1))"
    gs.openthreshold = openthreshold
    gs.closethreshold = closethreshold
    gs.predictionsdf = predictionsdf
    gs.scores = scores
    gs.labels = labels
    gs.algorithm(gs, lastix)
    if forcegain
        calcgain!(gs, lastix)
    end
    return gs.gaindf
end

#region deprecated-legacy-copy
function addgainadmin!(gdf, coin, sampleset, predicted, rangeid, openthreshold, closethreshold)
    gdf[!, :coin] = fill(coin, size(gdf, 1))
    gdf[!, :set] = fill(sampleset, size(gdf, 1))
    gdf[!, :predicted] = fill(predicted, size(gdf, 1))
    gdf[!, :rangeid] = fill(rangeid, size(gdf, 1))
    gdf[!, :openthreshold] = fill(openthreshold, size(gdf, 1))
    gdf[!, :closethreshold] = fill(closethreshold, size(gdf, 1))
end

"""
Returns a dataframe with gains of buy/sell pair actions. At the end of a range a sell is enforced.
The returned dataframe has the following columns:
  - trend (up, flat, down)
  - samplecount of gain segment
  - minutes of gain segment
  - gain of gain segment
  - startdt and enddt :: DateTime of gain segment
  - startix, endix within predictionsdf rows of gain segment

Input is 
  -  predictionsdf::AbstractDataFrame with the following colums
    - label as TradeLabel
    - score as Float32 prediction score
    - target as TradeLabel
    - mingain as expected % below pivot price to set a trade limit
    - maxgain as expected % above pivot price to set a trade limit
    - downminutes as expected down trend time in minutes
    - upminutes as expected up trend time in minutes
    - lastextrememinutes as minutes since last extreme
    - rangeid as indicator which samples belong to a sequence
    - opentime, high, low, close, pivot (samples of the same range are expected to be in opentime sorting order)
    - set as CategoricalVector with levels train, test, eval
    - coin as CategoricalVector
  - maxwindow of sample window considered for the prediction
  - scorethresholds as named tuple with acceptance score; trade label names that are not included are ignored
"""
function getgainsdf(predictionsdf::AbstractDataFrame; maxwindow, scorethresholds=(longbuy=0.6, allclose=0.6, shortbuy=0.6))
    if isnothing(predictionsdf) || (size(predictionsdf, 1) == 0)
        return nothing
    end
    @assert maxwindow > 0
    ranges = unique(predictionsdf[!, :rangeid])
    for rngix in eachindex(ranges)
        rng = ranges[rngix]
        resultsview = @view predictionsdf[predictionsdf[!, :rangeid] .== rng, :]
        (verbosity >= 2) && print("$(EnvConfig.now()) calculating gains for range ($rngix/$(length(ranges))) $rng                             \r")
        (verbosity >= 3) && println()
        if size(resultsview, 1) == 0 
            (verbosity >= 2) && println("\n$(EnvConfig.now()) WARNING: unexpected empty resultsview for rangeid $rng")
            continue
        end
        # @assert issorted(resultsview[!, :opentime]) "unexpected unsorted opentime in resultsview for rangeid $rng"
        # @assert all(resultsview[begin, :set] .== resultsview[!, :set]) "Unexpected different sets $(unique(resultsview[!, :set])) in same range $rng"

        for scorethreshold in scorethresholds
            gdf = _getgainsdf(resultsview, resultsview[!, :score], resultsview[!, :label], scorethreshold)
            if size(gdf, 1) > 0
                addgainadmin!(gdf, resultsview[begin, :coin], resultsview[begin, :set], true, rng, openthreshold, closethreshold)
                gaindf = isnothing(gaindf) ? gdf : append!(gaindf, gdf)
            end
        end
        gdf = _getgainsdf(resultsview, fill(1f0, size(resultsview, 1)), resultsview[!, :target], 0.9f0, 0.9f0)
        addgainadmin!(gdf, resultsview[begin, :coin], resultsview[begin, :set], false, rng, 0.9f0, 0.9f0)
        gaindf = isnothing(gaindf) ? gdf : vcat(gaindf, gdf)

    end
    # println("describe(gaindf)=$(describe(gaindf)), size(gaindf)=$(size(gaindf))")
    gaindf = gaindf[.!ismissing.(gaindf[!, :set]), :] # exclude gaps between set partitions
    if size(gaindf, 1) > 0
        sort!(gaindf, [:coin, :predicted, :openthreshold, :closethreshold, :startdt])
        EnvConfig.savedf(gaindf, gainsfilename())
    end
    (verbosity >= 2) && println("$(EnvConfig.now()) calculated gains for $(length(ranges)) ranges")
    return gaindf
end
#endregion deprecated-legacy-copy

end # module
