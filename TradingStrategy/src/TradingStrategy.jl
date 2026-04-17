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
    cancelrunningorder::Bool
    orderlabel::Union{TradeLabel, Nothing} # used: longbuy, longclose, shortclose, shortbuy
    orderlimit::Union{Float32, Nothing} # order limit or nothing if no limit order placed
    amountfactor::Float32
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
    gaindf::DataFrame
    makerfee::Float32
    takerfee::Float32
    scores::Union{AbstractVector, Nothing}
    labels::Union{AbstractVector, Nothing}
    predictionsdf::Union{AbstractDataFrame, Nothing}
    buyix::Integer # buy index of last open gain segment that is not yet saved in gainsdf
    lastix::Integer # last inspected row index of last open gain segment that is not yet saved in gainsdf
    trend::TrendPhase # trend of last 
    limit::Union{Float32, Nothing} # order limit or nothing if no limit order placed
    buygain::Float32 # relative gain compared to current price for limit order
    sellgain::Float32 # relative gain compared to current price for limit order
    buyprice::Union{Float32, Nothing} # buy price of open gain segment
    ordertype::TradeLabel # used: longbuy, longclose, allclose (default), shortclose, shortbuy
    assettype::TrendPhase # indicates whether current assets are not present (flat), short (down), long (up)
    function GainSegment(;maxwindow::Integer, openthreshold, closethreshold, algorithm=algorithm02!, makerfee::AbstractFloat=0f0, takerfee::AbstractFloat=0f0)
        return new(algorithm, openthreshold, closethreshold, maxwindow, 0, emptygaindf(), makerfee, takerfee, nothing, nothing, nothing, 0, 0, flat, nothing, 0.01f0, 0.005f0, nothing, allclose, flat)
    end
end

function reset!(gs::GainSegment)
    gs.predictionsdf = nothing
    gs.scores = nothing
    gs.labels = nothing
    gs.endix = gs.lastix = gs.buyix = 0
    gs.trend = flat
    gs.limit = nothing
    gs.buyprice = nothing
    gs.ordertype = allclose
    gs.assettype = flat
    gs.gaindf = emptygaindf()
    return gs
end

function isopengainsegment(gs::GainSegment)
    return (gs.buyix > 0)
end

"Calculates gain for an open segment and closes it. An open segment must have buyix > 0, buyprice != nothing, assettype in [up, down]"
function calcgain!(gs::GainSegment, sellix::Integer, sellprice::Float32=gs.predictionsdf[sellix, :close])
    if isopengainsegment(gs)
        @assert !isnothing(gs.buyprice)  "inconsistency: gs.buyix=$(gs.buyix), gs.buyprice=$(gs.buyprice)"
        starttime = gs.predictionsdf[gs.buyix, :opentime]
        ixtime = gs.predictionsdf[sellix, :opentime]
        @assert (gs.assettype == up) || (gs.assettype == down) "if gs.trend=$(gs.assettype) == flat then buyix=$(gs.buyix) == 0"
        if gs.assettype == up
            # long sell
            startprice = gs.buyprice # * (1 + gs.makerfee)
            ixprice = sellprice # * (1 - gs.makerfee)
            gain = (ixprice - startprice) / startprice
        else
            # short sell
            startprice = gs.buyprice # * (1 - gs.makerfee)
            ixprice = sellprice # * (1 + gs.makerfee)
            gain = -(ixprice - startprice) / startprice  # down trend -> negative price diff -> gain shall be positive for a short trade
        end
        minutes = Int(div(Dates.value(ixtime - starttime), 60000)) + 1
        push!(gs.gaindf, (gs.trend, (sellix - gs.buyix + 1), minutes, gain, (gain - 2f0 * gs.makerfee), starttime, ixtime, gs.buyix, sellix))
        gs.buyix = 0  # indicates closure of gain segment
        gs.buyprice = nothing
        gs.limit = nothing
        gs.assettype = flat
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
    @inbounds for ix in (gs.endix + 1):lastix
        #TODO first check whether the limit order is executed, if so, set trend and buyix and selllimit
        #TODO decide how long the limit order should stay and what limit corrections are required
        label = labels[ix]
        score = scores[ix]
        thistrend = gs.trend
        thisbuyix = 0
        if (gs.trend == up) && islongholdoropenlabel(label) && (score > gs.closethreshold)
            thistrend = up
        elseif islongopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif (gs.trend == down) && isshortholdoropenlabel(label) && (score > gs.closethreshold)
            thistrend = down
        elseif isshortopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end
        if gs.trend != thistrend
            calcgain!(gs, ix)
            gs.buyix = thisbuyix
            gs.trend = thistrend
            if isopengainsegment(gs)
                gs.buyprice = closecol[gs.buyix]
                gs.assettype = thistrend
            else
                gs.buyprice = nothing
                gs.assettype = flat
            end
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
    @inbounds for ix in (gs.endix + 1):lastix
        #TODO first check whether the limit order is executed, if so, set trend and buyix and selllimit
        #TODO decide how long the limit order should stay and what limit corrections are required
        label = labels[ix]
        score = scores[ix]
        thistrend = gs.trend
        thisbuyix = 0
        if (gs.trend == up) && islongcloselabel(label) && (score > gs.openthreshold)
            thistrend = flat
        elseif (gs.trend == down) && isshortcloselabel(label) && (score > gs.openthreshold)
            thistrend = flat
        elseif islongopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif isshortopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end
        if gs.trend != thistrend
            calcgain!(gs, ix)
            gs.buyix = thisbuyix
            gs.trend = thistrend
            if isopengainsegment(gs)
                gs.buyprice = closecol[gs.buyix]
                gs.assettype = thistrend
            else
                gs.buyprice = nothing
                gs.assettype = flat
            end
        end
    end
    gs.endix = lastix
    return gs
end

"check price ranges whether an open order is closed and calculates the gain"
function processclosedorder!(gs::GainSegment, ix::Integer)
    if !isnothing(gs.limit) && (gs.predictionsdf[gs.ix, :low] <= gs.limit <= gs.predictionsdf[gs.ix, :high])
        # assume trade took place
        if gs.assettype in [up, down]
            # opposite trend asset was sold together with asset buy trade
            calcgain!(gs, ix, gs.limit) # if segment close then also resets buyix and buyprice
        end
        if gs.ordertype in [longbuy, shortbuy]
            if gs.ordertype == longbuy
                # short trend asset was sold together with long asset buy trade
                gs.assettype = up
            elseif gs.ordertype == shortbuy
                # long trend asset was sold together with short asset buy trade
                gs.assettype = down
            end
            gs.buyprice = gs.limit
            gs.buyix = ix
        else
            gs.assettype = flat
            gs.buyprice = nothing
            gs.buyix = 0 # 
        end
        gs.limit = nothing # order is closed
    end
end

"""
Consumes the next prediction results after endix for all remaining predictions and calculates the corresponding gain segments that are stored in gaindf.  
Buy will buy at current price +- buygain, which is reduced with a limitreduction rate for non trend supporting samples. This limit is newly set when a buy signal for that trend is observed.
A bought sample receives a sell limit at current price +- sellgain, which is reduced with a limitreduction rate for non trend supporting samples.  
Hold as long as trend is not broken by opposite buy or close above openthreshold. If limit is hit then close gain segment and open new one if buy signal is present.  
Implementation details:  
- labels are indicating a trend change, but the gain materializes after the corresponding order is closed
- that requires distinguishing between the assettype and the trend
- buygain and sellgain are defining the initial limit distance to the current price, which is reduced with a limitreduction rate for non trend supporting samples utilizing the price volatility.
"""
function algorithm03!(gs::GainSegment, lastix)
    limitreduction = 1//20 # reduce limit by 5% with each non trend supporting sample
    # @assert length(scores) == length(labels) == size(predictionsdf, 1) > 0 "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1)) > 0"
    for ix in (gs.endix+1):lastix
        if gs.labels[ix] in [shortstrongbuy, shortbuy, shorthold] gs.trend = down
        elseif gs.labels[ix] in [longstrongbuy, longbuy, longhold] gs.trend = up
        else gs.trend = flat end
        ta = TradeAction(false, nothing, nothing, 1f0)
        # first check whether the limit order is executed, if so, set trend and buyix and selllimit
        processclosedorder!(gs, ix)
        """
        (short/long) types of limitorder:  
        - buy due to buy signal
        - continuelimitorder due to previous buy signal but not now (and no other buy signal), i.e. adjust limit
        - sell due to sell signal
        """
        if (gs.labels[ix] in [longbuy, longstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            if !isnothing(gs.limit) && (gs.assettype == down)
                # cancel shortclose order and add amount to longbuy order
                ta.cancelrunningorder = true
                ta.amountfactor = 2 # amount to sell long assets + amount to buy short assets
            end
            ta.orderlimit = gs.limit = gs.predictionsdf[gs.buyix, :close] * (1f0 - gs.buygain)
            ta.orderlabel = gs.ordertype = longbuy
            gs.trend = up
        elseif (gs.labels[ix] in [shortbuy, shortstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            if !isnothing(gs.limit) && (gs.assettype == up)
                # cancel longclose order and add amount to shortbuy order
                ta.cancelrunningorder = true
                ta.amountfactor = 2 # amount to sell long assets + amount to buy short assets
            end
            ta.orderlimit = gs.limit = gs.predictionsdf[gs.buyix, :close] * (1f0 + gs.buygain)
            ta.orderlabel = gs.ordertype = shortbuy
            gs.trend = down
        else
            if gs.assettype == up
                if isnothing(gs.limit)
                    # no limit order in place, set initial close limit
                     ta.orderlimit = gs.limit = gs.predictionsdf[ix, :close] * (1f0 + gs.sellgain)
                     ta.orderlabel = gs.ordertype = longclose
                elseif gs.labels[ix] != longhold
                    # limit order in place, adjust limit
                     ta.orderlimit = gs.limit = max(gs.predictionsdf[ix, :close], gs.limit * (1f0 - gs.sellgain * limitreduction))
                end
            elseif gs.assettype == down
                if isnothing(gs.limit)
                    # no limit order in place, set initial close limit
                     ta.orderlimit = gs.limit = gs.predictionsdf[ix, :close] * (1f0 - gs.sellgain)
                     ta.orderlabel = gs.ordertype = shortclose
                elseif gs.labels[ix] != shorthold
                    # limit order in place, adjust limit
                     ta.orderlimit = gs.limit = min(gs.predictionsdf[ix, :close], gs.limit * (1f0 + gs.sellgain * limitreduction))
                end
            end
        end
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
