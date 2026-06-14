"""
Minimal TradingStrategy module retained for Trade integration and TrendDetector workflows.

This module intentionally keeps only the API surface required by:
- Trade/src/Trade.jl
- scripts/TrendDetector.jl
- scripts/tradereal.jl
- scripts/tradesim.jl
"""
module TradingStrategy

using DataFrames, Dates
using EnvConfig, Targets, Classify, Xch, Ohlcv

"""Return the normalized config-scoped subfolder used for persisted trade artifacts."""
function tradesfolder(stem::AbstractString="gains")::String
    normalized = replace(normpath(splitext(String(stem))[1]), '\\' => '/')
    return startswith(normalized, "trades/") || (normalized == "trades") ? normalized : joinpath("trades", normalized)
end

"""Return the aggregate storage key used for one persisted trade artifact."""
tradesaggregate(stem::AbstractString="gains") = joinpath("trades", splitext(basename(String(stem)))[1] * "_all")

"""Return the per-coin storage key used for one persisted trade artifact."""
tradefilename(coin::AbstractString; stem::AbstractString="gains") = joinpath(tradesfolder(stem), uppercase(strip(String(coin))))

"""Persist a trades dataframe into config-scoped storage, plus optional aggregate copy."""
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

"""Load persisted trades, preferring aggregate cache over per-coin fragments."""
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

"""Load persisted trades for one specific coin."""
function loadtrades(coin::AbstractString; stem::AbstractString="gains")
    tradedf = EnvConfig.readdf(tradefilename(coin; stem=stem))
    return isnothing(tradedf) ? DataFrame() : DataFrame(tradedf)
end

@inline islongopenlabel(label::TradeLabel) = (label == longopen) || (label == longstrongopen)
@inline isshortopenlabel(label::TradeLabel) = (label == shortopen) || (label == shortstrongopen)
@inline islongholdoropenlabel(label::TradeLabel) = (label == longhold) || islongopenlabel(label)
@inline isshortholdoropenlabel(label::TradeLabel) = (label == shorthold) || isshortopenlabel(label)
@inline islongcloselabel(label::TradeLabel) = (label == allclose) || (label == longstrongclose) || (label == longclose)
@inline isshortcloselabel(label::TradeLabel) = (label == shortclose) || (label == shortstrongclose) || (label == allclose)

"""
Lane state for open/close intent and realized-entry bookkeeping.

- `label`: directional open/close intent or `ignore`
- `closeprice`: active close limit (or 0f0 when absent)
- `openprice`: intended/realized entry anchor
- `openix`: realized entry bar index (0 means not realized)
"""
mutable struct TradeAction
    label::Union{TradeLabel, Nothing}
    closeprice::Float32
    openprice::Float32
    openix::Integer
    function TradeAction(label::Union{TradeLabel, Nothing}=ignore, closeprice=0f0, openprice=0f0, openix=0)
        ta = new(label, closeprice, openprice, openix)
        isopen(ta)
        return ta
    end
end

"""Return whether a trade action currently carries an active limit order."""
function isopen(ta::TradeAction)
    if ta.closeprice > 0f0
        @assert (ta.label != ignore) && (ta.closeprice > 0f0) && (ta.openprice > 0f0) "(ta.label != ignore) && (ta.closeprice > 0f0) && (ta.openprice > 0f0); got label=$(ta.label), closeprice=$(ta.closeprice), openprice=$(ta.openprice)"
    end
    return ta.closeprice > 0f0
end

"""Clear all order state from a trade action."""
function removeorder!(ta::TradeAction)
    ta.label = ignore
    ta.closeprice = 0f0
    ta.openprice = 0f0
    ta.openix = 0
    return ta
end

@inline _islongopenaction(label) = label in [longopen, longstrongopen]
@inline _isshortopenaction(label) = label in [shortopen, shortstrongopen]

"""Clear one lane state completely."""
function _clearactionlane!(ta::TradeAction)
    ta.label = ignore
    ta.closeprice = 0f0
    ta.openprice = 0f0
    ta.openix = 0
    return ta
end

"""Clear only open-intent labels while preserving close guidance."""
function _clearopenintent!(ta::TradeAction)
    if ta.label in [longopen, longstrongopen, shortopen, shortstrongopen]
        ta.label = ignore
    end
    return ta
end

@inline _lanehascloseguidance(ta::TradeAction) = (ta.openprice > 0f0) && (ta.closeprice > 0f0)
@inline _price_in_bar(price::Float32, low::Real, high::Real) = (Float32(low) <= price) && (price <= Float32(high))
@inline _price_in_bar(price::Float32, low::Real, high::Real, boundary::Symbol) = boundary === :high ? (price <= Float32(high)) : (Float32(low) <= price)
@inline _relpricedelta(a::Real, b::Real) = abs(Float32(a) - Float32(b)) / max(abs(Float32(b)), 1f-6)

@inline function _should_update_price(current::Real, candidate::Real, minpricedelta::Float32)
    currentf = Float32(current)
    candidatef = Float32(candidate)
    if currentf <= 0f0 || minpricedelta <= 0f0
        return true
    end
    return _relpricedelta(candidatef, currentf) >= minpricedelta
end

abstract type AbstractStrategyRuntime end

"""
Immutable strategy configuration independent of GainSegment mutable runtime state.
"""
Base.@kwdef struct StrategySpec
    algorithm::Function = gain_reversal!
    maxwindow::Int = 4 * 60
    openthreshold::Float32 = 0.6f0
    closethreshold::Float32 = 0.5f0
    makerfee::Float32 = 0f0
    takerfee::Float32 = 0f0
    buygain::Float32 = 0.001f0
    sellgain::Float32 = 0.01f0
    limitreduction::Float32 = 0f0
    minpricedelta::Float32 = 0.001f0
    max_classify_staleness_minutes::Int = 5
end

"Build one immutable strategy specification payload."
function makestrategy(; algorithm=gain_reversal!, maxwindow::Integer=4*60, openthreshold=0.6f0, closethreshold=0.5f0, makerfee::Real=0f0, takerfee::Real=0f0, buygain::Real=0.001f0, sellgain::Real=0.01f0, limitreduction::Real=0f0, minpricedelta::Real=0.001f0, max_classify_staleness_minutes::Integer=5)::StrategySpec
    return StrategySpec(
        algorithm=algorithm,
        maxwindow=Int(maxwindow),
        openthreshold=Float32(openthreshold),
        closethreshold=Float32(closethreshold),
        makerfee=Float32(makerfee),
        takerfee=Float32(takerfee),
        buygain=Float32(buygain),
        sellgain=Float32(sellgain),
        limitreduction=Float32(limitreduction),
        minpricedelta=Float32(minpricedelta),
        max_classify_staleness_minutes=Int(max_classify_staleness_minutes),
    )
end

"Resolve a classifier type token to a concrete `Classify.AbstractClassifier` subtype."
function _classifiertype(token)::Type{<:Classify.AbstractClassifier}
    if token isa Type
        token <: Classify.AbstractClassifier || throw(ArgumentError("classifier_type must subtype Classify.AbstractClassifier, got $(token)"))
        return token
    end
    name = String(token)
    isdefined(Classify, Symbol(name)) || throw(ArgumentError("unknown classifier_type=$(name); expected one exported by Classify"))
    typ = getproperty(Classify, Symbol(name))
    (typ isa Type) && (typ <: Classify.AbstractClassifier) || throw(ArgumentError("classifier_type=$(name) is not a Classify.AbstractClassifier subtype"))
    return typ
end

"Load classifier from runtime configuration using one supported classifier configuration path."
function _loadclassifierfromconfig(configuration::AbstractDict; mode=EnvConfig.configmode)::Classify.AbstractClassifier
    if haskey(configuration, :classifier)
        cl = configuration[:classifier]
        cl isa Classify.AbstractClassifier || throw(ArgumentError("configuration[:classifier] must be Classify.AbstractClassifier, got $(typeof(cl))"))
        return cl
    end

    if haskey(configuration, :classifier_factory)
        factory = configuration[:classifier_factory]
        factory isa Function || throw(ArgumentError("configuration[:classifier_factory] must be a zero-arg function"))
        cl = factory()
        cl isa Classify.AbstractClassifier || throw(ArgumentError("classifier_factory must return Classify.AbstractClassifier, got $(typeof(cl))"))
        return cl
    end

    has_type = haskey(configuration, :classifier_type)
    has_spec = haskey(configuration, :classifier_spec)
    if has_type && has_spec
        typ = _classifiertype(configuration[:classifier_type])
        spec = configuration[:classifier_spec]
        return Classify.load(typ, spec; mode=mode)
    elseif has_type
        typ = _classifiertype(configuration[:classifier_type])
        return typ()
    end

    throw(ArgumentError("classifier configuration missing: provide one of [:classifier], [:classifier_factory], [:classifier_type], or [:classifier_type,:classifier_spec]"))
end

mutable struct GainSegmentRuntime <: AbstractStrategyRuntime
    classifier::Classify.AbstractClassifier
    strategy_template::Any
    strategy_state::Dict{String, Any}
    strategy_history::Dict{String, NamedTuple{(:predictionsdf, :scores, :labels), Tuple{DataFrame, Vector{Float32}, Vector{Targets.TradeLabel}}}}
    classifier_gate_state::Dict{String, NamedTuple{(:last_advice, :last_classify_dt, :last_classify_close), Tuple{Any, Union{Nothing, DateTime}, Float32}}}
    accepted::Set{String}
    source::String
    function GainSegmentRuntime(; classifier::Union{Nothing, Classify.AbstractClassifier}=nothing, classifier_type=nothing, classifier_spec=nothing, strategy::Any=nothing, source::AbstractString="manual", mode=EnvConfig.configmode)
        resolved_classifier = if !isnothing(classifier)
            classifier
        elseif !isnothing(classifier_type) && !isnothing(classifier_spec)
            Classify.load(_classifiertype(classifier_type), classifier_spec; mode=mode)
        elseif !isnothing(classifier_type)
            _classifiertype(classifier_type)()
        else
            throw(ArgumentError("GainSegmentRuntime classifier is required; provide classifier or classifier_type (and optional classifier_spec)"))
        end
        return new(resolved_classifier, deepcopy(strategy), Dict{String, Any}(), Dict{String, NamedTuple{(:predictionsdf, :scores, :labels), Tuple{DataFrame, Vector{Float32}, Vector{Targets.TradeLabel}}}}(), Dict{String, NamedTuple{(:last_advice, :last_classify_dt, :last_classify_close), Tuple{Any, Union{Nothing, DateTime}, Float32}}}(), Set{String}(), String(source))
    end
end

"""Per-trading-pair runtime state holder used by `TsCache`."""
Base.@kwdef mutable struct TsTp
    pair::String
    tradesdf::DataFrame = DataFrame()
    last_update_dt::Union{Nothing, DateTime} = nothing
end

"""
Internal runtime cache for the Phase 2 Trades DataFrame architecture.

`TsCache` keeps pair-scoped runtime references while `Xch` remains owner of the
mutable per-pair Trades DataFrames.
"""
mutable struct TsCache <: AbstractStrategyRuntime
    configuration::Dict{Symbol, Any}
    classifier::Classify.AbstractClassifier
    pairs::Dict{String, TsTp}
    source::String
end

"Build TsCache with explicit classifier wiring from argument or configuration."
function TsCache(; configuration::Dict{Symbol, Any}=Dict{Symbol, Any}(), classifier::Union{Nothing, Classify.AbstractClassifier}=nothing, source::AbstractString="manual", mode=EnvConfig.configmode)
    resolved_classifier = isnothing(classifier) ? _loadclassifierfromconfig(configuration; mode=mode) : classifier
    return TsCache(configuration, resolved_classifier, Dict{String, TsTp}(), String(source))
end

"Return canonical trading-pair key for TsCache pair state lookups."
function tspairkey(base::AbstractString, quotecoin::AbstractString=EnvConfig.pairquote)::String
    return uppercase(String(base)) * uppercase(String(quotecoin))
end

"Return TsCache pair-state entry for one pair, creating an empty entry when missing."
function getpairstate!(ts::TsCache, pair::AbstractString)::TsTp
    key = uppercase(String(pair))
    return get!(ts.pairs, key) do
        TsTp(pair=key)
    end
end

"Return TsCache pair-state entry for one `(base, quotecoin)` pair."
function getpairstate!(ts::TsCache, base::AbstractString, quotecoin::AbstractString)::TsTp
    return getpairstate!(ts, tspairkey(base, quotecoin))
end

"Return currently tracked pair keys in deterministic sorted order."
function pairkeys(ts::TsCache)::Vector{String}
    return sort!(collect(keys(ts.pairs)))
end

"Drop one pair from TsCache pair-state map."
function droppair!(ts::TsCache, pair::AbstractString)::Nothing
    delete!(ts.pairs, uppercase(String(pair)))
    return nothing
end

"Synchronize one TsCache pair entry to the Xch-owned mutable Trades DataFrame."
function syncpairtrades!(ts::TsCache, xc::Xch.XchCache, pair::AbstractString; datetime::Union{Nothing, DateTime}=nothing)::TsTp
    tp = getpairstate!(ts, pair)
    tp.tradesdf = Xch.trades(xc, pair)
    tp.last_update_dt = datetime
    return tp
end

"Synchronize one TsCache pair entry to the Xch-owned mutable Trades DataFrame."
function syncpairtrades!(ts::TsCache, xc::Xch.XchCache, base::AbstractString, quotecoin::AbstractString; datetime::Union{Nothing, DateTime}=nothing)::TsTp
    pair = tspairkey(base, quotecoin)
    tp = getpairstate!(ts, pair)
    tp.tradesdf = Xch.trades(xc, base, quotecoin)
    tp.last_update_dt = datetime
    return tp
end

"Return true when TsCache currently tracks one pair state entry."
function haspairstate(ts::TsCache, pair::AbstractString)::Bool
    return haskey(ts.pairs, uppercase(String(pair)))
end

"""Return an empty gain dataframe with the canonical Trade-consumed schema."""
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

"""
State container for gain-based strategy execution over one symbol.

The lane model keeps long and short guidance independently in `longta` and `shortta`.
"""
mutable struct GainSegment
    algorithm
    openthreshold::Float32
    closethreshold::Float32 # candidate for removal
    maxwindow::Integer # not yet used but only after maxwindow incremental limitreduction should start
    endix::Integer # last processed ix in the predictionsdf
    gaindf::DataFrame # not used by Trade only used by TrendDetector
    makerfee::Float32
    takerfee::Float32
    scores::Union{AbstractVector, Nothing}
    labels::Union{AbstractVector, Nothing}
    predictionsdf::Union{AbstractDataFrame, Nothing}
    lastix::Integer
    buygain::Float32
    sellgain::Float32
    limitreduction::Float32
    minpricedelta::Float32
    max_classify_staleness_minutes::Int
    longta::TradeAction
    shortta::TradeAction
    trace::Union{Nothing, DataFrame}
    tracecontext::Union{Nothing, String}
    lane_reconciliation::Union{Nothing, NamedTuple}
    function GainSegment(;maxwindow::Integer=4*60, openthreshold=0.6, closethreshold=0.5, algorithm=gain_reversal!, makerfee::AbstractFloat=0f0, takerfee::AbstractFloat=0f0, buygain::AbstractFloat=0.001f0, sellgain::AbstractFloat=0.01f0, limitreduction::AbstractFloat=0f0, minpricedelta::AbstractFloat=0.001f0, max_classify_staleness_minutes::Integer=5)
        classify_staleness_minutes = Int(max_classify_staleness_minutes)
        return new(algorithm, openthreshold, closethreshold, maxwindow, 0, emptygaindf(), makerfee, takerfee, nothing, nothing, nothing, 0, buygain, sellgain, limitreduction, Float32(minpricedelta), Int(max(0, classify_staleness_minutes)), TradeAction(), TradeAction(), nothing, nothing, nothing)
    end
end

@inline max_classify_staleness_minutes(gs::GainSegment) = gs.max_classify_staleness_minutes

Base.@kwdef mutable struct StrategySnapshot
    base::String
    datetime::DateTime
    label::Targets.TradeLabel = Targets.ignore
    long_openprice::Float32 = 0f0
    long_closeprice::Float32 = 0f0
    long_openix::Int = 0
    short_openprice::Float32 = 0f0
    short_closeprice::Float32 = 0f0
    short_openix::Int = 0
    probability::Float32 = 0f0
    configid::Int = 0
    source::Symbol = :tradingstrategy
end

Base.@kwdef struct StrategyReconciliationInput
    has_long_open::Bool = false
    long_avg_entry::Float32 = 0f0
    long_open_ix::Int = 0
    has_short_open::Bool = false
    short_avg_entry::Float32 = 0f0
    short_open_ix::Int = 0
end

"Return the minimum history window required by this runtime implementation."
requiredhistoryminutes(rt::AbstractStrategyRuntime)::Int = 0

"Raise a clear contract error when a mandatory runtime method was not overridden."
function _runtime_contract_error(methodname::AbstractString, rt::AbstractStrategyRuntime)
    throw(ArgumentError("$(methodname) must be implemented for runtime type $(typeof(rt)); provide a concrete $(methodname) method for your AbstractStrategyRuntime subtype"))
end

"Return the minimum history window required by the gain-segment runtime classifier."
function requiredhistoryminutes(rt::GainSegmentRuntime)::Int
    try
        return max(0, Int(Classify.requiredminutes(rt.classifier)))
    catch
        return 0
    end
end

"Return the bases currently accepted by this runtime implementation."
acceptedbases(rt::AbstractStrategyRuntime)::Set{String} = Set{String}()
acceptedbases(rt::GainSegmentRuntime)::Set{String} = copy(rt.accepted)

"Drop one base from runtime state and any cached strategy data."
function dropbase!(rt::AbstractStrategyRuntime, base::AbstractString)::Nothing
    _ = rt
    _ = base
    return nothing
end

"Drop one base from the gain-segment runtime, including classifier and cached lane state."
function dropbase!(rt::GainSegmentRuntime, base::AbstractString)::Nothing
    basekey = uppercase(String(base))
    try
        Classify.removebase!(rt.classifier, basekey)
    catch
    end
    delete!(rt.strategy_state, basekey)
    delete!(rt.strategy_history, basekey)
    delete!(rt.classifier_gate_state, basekey)
    delete!(rt.accepted, basekey)
    return nothing
end

"Reset runtime state to its default empty configuration."
function reset!(rt::AbstractStrategyRuntime)::Nothing
    _ = rt
    return nothing
end

"Reset the gain-segment runtime, clearing accepted bases and cached classifier state."
function reset!(rt::GainSegmentRuntime)::Nothing
    empty!(rt.strategy_state)
    empty!(rt.strategy_history)
    empty!(rt.classifier_gate_state)
    empty!(rt.accepted)
    try
        Classify.removebase!(rt.classifier, nothing)
    catch
    end
    return nothing
end

"Apply a strategy template to the runtime implementation."
function apply_strategy!(rt::AbstractStrategyRuntime, gs::GainSegment; source::AbstractString="manual")::Nothing
    _ = rt
    _ = gs
    _ = source
    return nothing
end

"Apply a gain-segment template to the runtime and clear any derived cached state."
function apply_strategy!(rt::GainSegmentRuntime, gs::GainSegment; source::AbstractString="manual")::Nothing
    rt.strategy_template = deepcopy(gs)
    rt.source = String(source)
    empty!(rt.strategy_state)
    empty!(rt.strategy_history)
    empty!(rt.classifier_gate_state)
    empty!(rt.accepted)
    try
        Classify.removebase!(rt.classifier, nothing)
    catch
    end
    return nothing
end

"Return the per-base runtime history cache, creating an empty one when needed."
function _runtimehistory!(rt::GainSegmentRuntime, base::AbstractString)
    basekey = uppercase(String(base))
    return get!(rt.strategy_history, basekey) do
        (
            predictionsdf=DataFrame(opentime=DateTime[], high=Float32[], low=Float32[], close=Float32[]),
            scores=Float32[],
            labels=Targets.TradeLabel[],
        )
    end
end

"Return the per-base gain-segment runtime state, cloning the current template when needed."
function _runtimestate!(rt::GainSegmentRuntime, base::AbstractString)
    basekey = uppercase(String(base))
    return get!(rt.strategy_state, basekey) do
        deepcopy(rt.strategy_template)
    end
end

"Return the per-base classifier gate state, creating an empty one when needed."
function _runtimegatestate!(rt::GainSegmentRuntime, base::AbstractString)
    basekey = uppercase(String(base))
    return get!(rt.classifier_gate_state, basekey) do
        (last_advice=nothing, last_classify_dt=nothing, last_classify_close=0f0)
    end
end

function _set_runtimegatestate!(rt::GainSegmentRuntime, base::AbstractString; last_advice, last_classify_dt::Union{Nothing, DateTime}, last_classify_close::Real)
    basekey = uppercase(String(base))
    rt.classifier_gate_state[basekey] = (
        last_advice=last_advice,
        last_classify_dt=last_classify_dt,
        last_classify_close=Float32(last_classify_close),
    )
    return rt
end

@inline _classifier_gating_enabled(gs::GainSegment) = gs.algorithm === gain_limit_reversal_pricedelta!

@inline function _classification_triggered(gs::GainSegment, interval_ok::Bool, delta_ok::Bool)::Bool
    interval_enabled = gs.max_classify_staleness_minutes > 0
    delta_enabled = gs.minpricedelta > 0f0
    !(interval_enabled || delta_enabled) && return true
    return (interval_enabled && interval_ok) || (delta_enabled && delta_ok)
end

function _should_skip_classifier(gs::GainSegment, gate, datetime::DateTime, closeprice::Float32, reconciliation::StrategyReconciliationInput)::Bool
    _classifier_gating_enabled(gs) || return false
    isnothing(gate.last_advice) && return false

    interval_ok = true
    if gs.max_classify_staleness_minutes > 0
        isnothing(gate.last_classify_dt) && return false
        elapsed_minutes = Int(div(Dates.value(datetime - gate.last_classify_dt), 60000))
        interval_ok = elapsed_minutes >= gs.max_classify_staleness_minutes
    end

    delta_ok = true
    if gs.minpricedelta > 0f0
        gate.last_classify_close > 0f0 || return false
        delta_ok = _relpricedelta(closeprice, gate.last_classify_close) >= gs.minpricedelta
    end

    return !_classification_triggered(gs, interval_ok, delta_ok)
end

"Append or update one runtime sample derived from the current OHLCV bar and classifier output."
function _upsert_runtime_sample!(history, ohlcv::Ohlcv.OhlcvData, label::Targets.TradeLabel, score)
    rowix = ohlcv.ix
    odf = Ohlcv.dataframe(ohlcv)
    @assert (1 <= rowix <= size(odf, 1)) "rowix=$(rowix) out of bounds for ohlcv rows=$(size(odf, 1))"
    opentime = odf[rowix, :opentime]
    high = Float32(odf[rowix, :high])
    low = Float32(odf[rowix, :low])
    close = Float32(odf[rowix, :close])
    sc = Float32(score)

    if size(history.predictionsdf, 1) > 0 && (history.predictionsdf[end, :opentime] == opentime)
        history.predictionsdf[end, :high] = high
        history.predictionsdf[end, :low] = low
        history.predictionsdf[end, :close] = close
        history.scores[end] = sc
        history.labels[end] = label
    else
        push!(history.predictionsdf, (opentime=opentime, high=high, low=low, close=close))
        push!(history.scores, sc)
        push!(history.labels, label)
    end
    return history
end

"Apply reconciliation input to one gain segment before running gain calculations."
function _apply_runtime_reconciliation!(gs::GainSegment, reconciliation::StrategyReconciliationInput)
    setreconciliation!(
        gs;
        long_open_qty=reconciliation.has_long_open ? 1f0 : 0f0,
        long_avg_entry=reconciliation.long_avg_entry,
        long_open_ix=reconciliation.long_open_ix,
        short_open_qty=reconciliation.has_short_open ? 1f0 : 0f0,
        short_avg_entry=reconciliation.short_avg_entry,
        short_open_ix=reconciliation.short_open_ix,
    )
    return gs
end

"Prepare a runtime implementation for the requested bases. Concrete runtimes must implement this method."
function preparebases!(rt::AbstractStrategyRuntime, xc::Xch.XchCache, bases::AbstractVector{<:AbstractString}; history_startdt::DateTime, datetime::DateTime, updatecache::Bool=false)::Nothing
    _ = rt
    _ = xc
    _ = bases
    _ = history_startdt
    _ = datetime
    _ = updatecache
    _runtime_contract_error("preparebases!", rt)
end

"Prepare the gain-segment runtime for the requested bases using available OHLCV data and update its accepted set."
function preparebases!(rt::GainSegmentRuntime, xc::Xch.XchCache, bases::AbstractVector{<:AbstractString}; history_startdt::DateTime, datetime::DateTime, updatecache::Bool=false)::Nothing
    _ = history_startdt
    _ = datetime
    wanted = Set{String}(uppercase.(String.(bases)))

    loaded = Set{String}(uppercase.(String.(Classify.bases(rt.classifier))))
    for stale in sort!(collect(setdiff(union(rt.accepted, loaded), wanted)))
        dropbase!(rt, stale)
    end

    loaded = Set{String}(uppercase.(String.(Classify.bases(rt.classifier))))
    any_loaded = false
    for base in sort!(collect(wanted))
        try
            ohlcv = Xch.ohlcv(xc, base)
            if !(base in loaded)
                Classify.addbase!(rt.classifier, ohlcv)
                push!(loaded, base)
            end
            any_loaded = true
        catch
            continue
        end
    end

    if any_loaded
        Classify.supplement!(rt.classifier)
        updatecache && Classify.writetargetsfeatures(rt.classifier)
    end

    accepted = Set{String}(uppercase.(String.(Classify.bases(rt.classifier))))
    intersect!(accepted, wanted)
    rt.accepted = accepted
    return nothing
end

"Return one strategy snapshot for a base. Concrete runtimes must implement this method."
function getsnapshot!(rt::AbstractStrategyRuntime, xc::Xch.XchCache, base::AbstractString, datetime::DateTime; reconciliation::StrategyReconciliationInput=StrategyReconciliationInput())::Union{Nothing, StrategySnapshot}
    _ = rt
    _ = xc
    _ = base
    _ = datetime
    _ = reconciliation
    _runtime_contract_error("getsnapshot!", rt)
end

"Return one strategy snapshot for a base using gain-segment runtime state and classifier output."
function getsnapshot!(rt::GainSegmentRuntime, xc::Xch.XchCache, base::AbstractString, datetime::DateTime; reconciliation::StrategyReconciliationInput=StrategyReconciliationInput())::Union{Nothing, StrategySnapshot}
    basekey = uppercase(String(base))
    if !haskey(xc.bases, basekey)
        (EnvConfig.verbosity >= 1) && @warn "base OHLCV unavailable in exchange cache; skipping getsnapshot" base=basekey
        return nothing
    end

    ohlcv = Xch.ohlcv(xc, basekey)
    gs = _runtimestate!(rt, basekey)
    _apply_runtime_reconciliation!(gs, reconciliation)
    gate = _runtimegatestate!(rt, basekey)

    rowix = ohlcv.ix
    odf = Ohlcv.dataframe(ohlcv)
    @assert (1 <= rowix <= size(odf, 1)) "rowix=$(rowix) out of bounds for ohlcv rows=$(size(odf, 1))"
    closeprice = Float32(odf[rowix, :close])

    advice = nothing
    if _should_skip_classifier(gs, gate, datetime, closeprice, reconciliation)
        advice = gate.last_advice
    else
        advice = Classify.advice(rt.classifier, basekey, datetime, investment=nothing)
        if !isnothing(advice)
            _set_runtimegatestate!(rt, basekey; last_advice=advice, last_classify_dt=datetime, last_classify_close=closeprice)
        end
    end

    isnothing(advice) && return nothing

    history = _runtimehistory!(rt, basekey)
    _upsert_runtime_sample!(history, ohlcv, advice.tradelabel, advice.probability)

    lastix = length(history.scores)
    if lastix > 0
        getgains(gs, history.predictionsdf, history.scores, history.labels, false; lastix=lastix, openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
    end

    cfgid = try
        Int(advice.configid)
    catch
        0
    end

    return StrategySnapshot(
        base=basekey,
        datetime=datetime,
        label=advice.tradelabel,
        long_openprice=Float32(gs.longta.openprice),
        long_closeprice=Float32(gs.longta.closeprice),
        long_openix=Int(gs.longta.openix),
        short_openprice=Float32(gs.shortta.openprice),
        short_closeprice=Float32(gs.shortta.closeprice),
        short_openix=Int(gs.shortta.openix),
        probability=Float32(advice.probability),
        configid=cfgid,
        source=:tradingstrategy,
    )
end

"Return one strategy snapshot per requested base. Concrete runtimes must implement this method."
function getsnapshots!(rt::AbstractStrategyRuntime, xc::Xch.XchCache, bases::AbstractVector{<:AbstractString}, datetime::DateTime; reconciliation_by_base::AbstractDict{String, StrategyReconciliationInput}=Dict{String, StrategyReconciliationInput}())::Vector{StrategySnapshot}
    _ = rt
    _ = xc
    _ = bases
    _ = datetime
    _ = reconciliation_by_base
    _runtime_contract_error("getsnapshots!", rt)
end

"Return one strategy snapshot per requested base using gain-segment runtime state."
function getsnapshots!(rt::GainSegmentRuntime, xc::Xch.XchCache, bases::AbstractVector{<:AbstractString}, datetime::DateTime; reconciliation_by_base::AbstractDict{String, StrategyReconciliationInput}=Dict{String, StrategyReconciliationInput}())::Vector{StrategySnapshot}
    snaps = StrategySnapshot[]
    for base in bases
        basekey = uppercase(String(base))
        recon = get(reconciliation_by_base, basekey, StrategyReconciliationInput())
        snap = getsnapshot!(rt, xc, basekey, datetime; reconciliation=recon)
        isnothing(snap) || push!(snaps, snap)
    end
    return snaps
end

"Set execution reconciliation hints for lane synchronization."
function setreconciliation!(gs::GainSegment;
    long_open_qty::Real=0f0,
    long_avg_entry::Real=0f0,
    long_open_ix::Integer=0,
    short_open_qty::Real=0f0,
    short_avg_entry::Real=0f0,
    short_open_ix::Integer=0,
)
    gs.lane_reconciliation = (
        long_open_qty=Float32(max(0f0, Float32(long_open_qty))),
        long_avg_entry=Float32(max(0f0, Float32(long_avg_entry))),
        long_open_ix=Int(long_open_ix),
        short_open_qty=Float32(max(0f0, Float32(short_open_qty))),
        short_avg_entry=Float32(max(0f0, Float32(short_avg_entry))),
        short_open_ix=Int(short_open_ix),
    )
    return gs
end

"Clear previously configured reconciliation hints."
function clearreconciliation!(gs::GainSegment)
    gs.lane_reconciliation = nothing
    return gs
end

"Apply stored reconciliation hints to lane state without changing directional intent."
function _apply_reconciliation_to_lanes!(gs::GainSegment)
    isnothing(gs.lane_reconciliation) && return gs
    rc = gs.lane_reconciliation

    if rc.long_open_qty > 0f0
        entry = rc.long_avg_entry > 0f0 ? rc.long_avg_entry : max(gs.longta.openprice, 0f0)
        if entry > 0f0
            gs.longta.openprice = entry
            if rc.long_open_ix > 0
                gs.longta.openix = rc.long_open_ix
            end
            if gs.longta.closeprice <= 0f0
                gs.longta.closeprice = entry * (1f0 + gs.sellgain)
            end
        end
    end

    if rc.short_open_qty > 0f0
        entry = rc.short_avg_entry > 0f0 ? rc.short_avg_entry : max(gs.shortta.openprice, 0f0)
        if entry > 0f0
            gs.shortta.openprice = entry
            if rc.short_open_ix > 0
                gs.shortta.openix = rc.short_open_ix
            end
            if gs.shortta.closeprice <= 0f0
                gs.shortta.closeprice = entry * (1f0 - gs.sellgain)
            end
        end
    end

    return gs
end

"Compatibility alias that applies stored reconciliation hints to the lane state."
synclanes!(gs::GainSegment) = _apply_reconciliation_to_lanes!(gs)

"Reset one gain segment instance to initial runtime state."
function reset!(gs::GainSegment)
    gs.predictionsdf = nothing
    gs.scores = nothing
    gs.labels = nothing
    gs.endix = gs.lastix = 0
    gs.gaindf = emptygaindf()
    removeorder!(gs.longta)
    removeorder!(gs.shortta)
    clearreconciliation!(gs)
    if !isnothing(gs.trace)
        empty!(gs.trace)
    end
    return gs
end

"Reset one strategy payload when mutable runtime state exists."
resetstrategy!(::StrategySpec) = nothing
resetstrategy!(gs::GainSegment) = reset!(gs)

"Convert immutable StrategySpec to a fresh GainSegment runtime template."
function _togainsegment(spec::StrategySpec)::GainSegment
    return GainSegment(
        maxwindow=spec.maxwindow,
        openthreshold=spec.openthreshold,
        closethreshold=spec.closethreshold,
        algorithm=spec.algorithm,
        makerfee=spec.makerfee,
        takerfee=spec.takerfee,
        buygain=spec.buygain,
        sellgain=spec.sellgain,
        limitreduction=spec.limitreduction,
        minpricedelta=spec.minpricedelta,
        max_classify_staleness_minutes=spec.max_classify_staleness_minutes,
    )
end

"""Return whether any lane currently carries close guidance."""
isopensegment(gs::GainSegment) = _lanehascloseguidance(gs.longta) || _lanehascloseguidance(gs.shortta)

"""Infer trend direction from lane price relation."""
assettrend(ta::TradeAction) = ta.closeprice > ta.openprice ? up : (ta.closeprice < ta.openprice ? down : flat)

@inline function _actiontrend(label::TradeLabel)
    if label in [longopen, longstrongopen, longclose, longstrongclose]
        return up
    elseif label in [shortopen, shortstrongopen, shortclose, shortstrongclose]
        return down
    end
    return flat
end

"""Close all open lanes at one sell index and optional price."""
function calcgain!(gs::GainSegment, sellix::Integer, sellprice::Float32=gs.predictionsdf[sellix, :close])
    if _lanehascloseguidance(gs.longta) && (gs.longta.openix > 0)
        calcgain!(gs, gs.longta, sellix, sellprice)
        _clearactionlane!(gs.longta)
    end
    if _lanehascloseguidance(gs.shortta) && (gs.shortta.openix > 0)
        calcgain!(gs, gs.shortta, sellix, sellprice)
        _clearactionlane!(gs.shortta)
    end
    return gs
end

"""Record one realized gain segment from one lane action."""
function calcgain!(gs::GainSegment, ta::TradeAction, sellix::Integer, sellprice::Float32)
    if (ta.openix > 0) && (ta.openprice > 0f0)
        starttime = gs.predictionsdf[ta.openix, :opentime]
        ixtime = gs.predictionsdf[sellix, :opentime]
        trend = _actiontrend(ta.label)
        if trend == flat
            trend = assettrend(ta)
        end

        if trend == up
            gain = (sellprice - ta.openprice) / ta.openprice
        elseif trend == down
            gain = -(sellprice - ta.openprice) / ta.openprice
        else
            return gs
        end

        minutes = Int(div(Dates.value(ixtime - starttime), 60000)) + 1
        push!(gs.gaindf, (trend, (sellix - ta.openix + 1), minutes, gain, (gain - 2f0 * gs.makerfee), starttime, ixtime, ta.openix, sellix))
        ta.openix = 0
    end
    return gs
end

"""Open/hold/close strategy that closes when hold support drops below threshold."""
function gain_open_close!(gs::GainSegment, lastix)
    labels = gs.labels
    scores = gs.scores
    closecol = gs.predictionsdf[!, :close]
    lasttrend = flat

    @inbounds for ix in (gs.endix + 1):lastix
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
                gs.longta.label = ignore
                gs.longta.closeprice = closecol[ix] * (1f0 + gs.sellgain)
                gs.longta.openprice = closecol[ix]
                gs.longta.openix = thisbuyix
                _clearactionlane!(gs.shortta)
            elseif thistrend == down
                gs.shortta.label = ignore
                gs.shortta.closeprice = closecol[ix] * (1f0 - gs.sellgain)
                gs.shortta.openprice = closecol[ix]
                gs.shortta.openix = thisbuyix
                _clearactionlane!(gs.longta)
            else
                _clearactionlane!(gs.longta)
                _clearactionlane!(gs.shortta)
            end
            lasttrend = thistrend
        end
    end

    gs.endix = lastix
    _apply_reconciliation_to_lanes!(gs)
    return gs
end

"""Reversal strategy that closes on explicit opposite/close labels above threshold."""
function gain_reversal!(gs::GainSegment, lastix)
    labels = gs.labels
    scores = gs.scores
    closecol = gs.predictionsdf[!, :close]
    lasttrend = flat

    @inbounds for ix in (gs.endix + 1):lastix
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
                gs.longta.label = ignore
                gs.longta.closeprice = closecol[ix] * (1f0 + gs.sellgain)
                gs.longta.openprice = closecol[ix]
                gs.longta.openix = thisbuyix
                _clearactionlane!(gs.shortta)
            elseif thistrend == down
                gs.shortta.label = ignore
                gs.shortta.closeprice = closecol[ix] * (1f0 - gs.sellgain)
                gs.shortta.openprice = closecol[ix]
                gs.shortta.openix = thisbuyix
                _clearactionlane!(gs.longta)
            else
                _clearactionlane!(gs.longta)
                _clearactionlane!(gs.shortta)
            end
            lasttrend = thistrend
        end
    end

    gs.endix = lastix
    _apply_reconciliation_to_lanes!(gs)
    return gs
end

"""Handle lane fills for one candle and update realized gain state."""
function processclosedorder!(gs::GainSegment, ix::Integer, ta::TradeAction)
    if (ta.openprice > 0f0) && (ta.closeprice > 0f0)
        if (ta.label in [longopen, longstrongopen, shortopen, shortstrongopen])
            if (ta.openix <= 0) && _price_in_bar(Float32(ta.openprice), gs.predictionsdf[ix, :low], gs.predictionsdf[ix, :high])
                ta.openix = ix
            end
            if (ta.openix > 0) && _price_in_bar(Float32(ta.closeprice), gs.predictionsdf[ix, :low], gs.predictionsdf[ix, :high])
                calcgain!(gs, ta, ix, ta.closeprice)
            end
        elseif ta.label in [longclose, longstrongclose]
            if ta.closeprice <= gs.predictionsdf[ix, :high]
                calcgain!(gs, ta, ix, ta.closeprice)
                _clearactionlane!(ta)
            end
        elseif ta.label in [shortclose, shortstrongclose]
            if gs.predictionsdf[ix, :low] <= ta.closeprice
                calcgain!(gs, ta, ix, ta.closeprice)
                _clearactionlane!(ta)
            end
        end
    end
end

"""Adjust lane open/close intents by one prediction candle for the limit-reversal strategy."""
function reachgainuntilreversal!(longta::TradeAction, shortta::TradeAction, label::TradeLabel, score, high, low, close, openthreshold, buygain, sellgain, limitreduction)
    _ = high
    _ = low
    branch = "none"

    longsignal = (label in [longopen, longstrongopen]) && (score >= openthreshold)
    shortsignal = (label in [shortopen, shortstrongopen]) && (score >= openthreshold)

    !longsignal && _clearopenintent!(longta)
    !shortsignal && _clearopenintent!(shortta)

    if longsignal
        branch = "long_signal"
        if _lanehascloseguidance(shortta)
            shortta.closeprice = close * (1f0 - buygain)
        end
        longta.openprice = close * (1f0 - buygain)
        longta.label = label == longstrongopen ? longstrongopen : longopen
        longta.closeprice = close * (1f0 + sellgain)
    elseif shortsignal
        branch = "short_signal"
        if _lanehascloseguidance(longta)
            longta.closeprice = close * (1f0 + buygain)
        end
        shortta.openprice = close * (1f0 + buygain)
        shortta.label = label == shortstrongopen ? shortstrongopen : shortopen
        shortta.closeprice = close * (1f0 - sellgain)
    else
        branch = "reduce_limit"
        if _lanehascloseguidance(longta) && (longta.closeprice > longta.openprice)
            longta.closeprice = max(close, longta.closeprice * (1f0 - sellgain * limitreduction))
        end
        if _lanehascloseguidance(shortta) && (shortta.closeprice < shortta.openprice)
            shortta.closeprice = min(close, shortta.closeprice * (1f0 + sellgain * limitreduction))
        end
    end

    return branch
end

@inline _missingf32(x) = ismissing(x) ? 0f0 : Float32(x)
@inline _hasguidance(openlimit, closelimit) = (!ismissing(openlimit)) && (!ismissing(closelimit)) && (Float32(openlimit) > 0f0) && (Float32(closelimit) > 0f0)
@inline _openlimitactive(openlimit) = (!ismissing(openlimit)) && (Float32(openlimit) > 0f0)

@inline function _setlastopentrade!(tradesdf::DataFrame, ix::Integer, dt)
    tradesdf[ix, :lastopentrade] = dt
    return nothing
end

"""
    _ensuretradesstrategycolumns!(tradesdf)

Ensure the row-oriented Trades DataFrame has all strategy state columns required
by the DataFrame-based limit-reversal variants.
"""
function _ensuretradesstrategycolumns!(tradesdf::DataFrame)::DataFrame
    n = nrow(tradesdf)
    if :longopenlimit ∉ propertynames(tradesdf)
        tradesdf[!, :longopenlimit] = Vector{Union{Missing, Float32}}(missing, n)
    end
    if :longcloselimit ∉ propertynames(tradesdf)
        tradesdf[!, :longcloselimit] = Vector{Union{Missing, Float32}}(missing, n)
    end
    if :shortopenlimit ∉ propertynames(tradesdf)
        tradesdf[!, :shortopenlimit] = Vector{Union{Missing, Float32}}(missing, n)
    end
    if :shortcloselimit ∉ propertynames(tradesdf)
        tradesdf[!, :shortcloselimit] = Vector{Union{Missing, Float32}}(missing, n)
    end
    if :tradelabel ∉ propertynames(tradesdf)
        tradesdf[!, :tradelabel] = fill(ignore, n)
    end
    if :labelscore ∉ propertynames(tradesdf)
        tradesdf[!, :labelscore] = zeros(Float32, n)
    end
    if :lastopentrade ∉ propertynames(tradesdf)
        tradesdf[!, :lastopentrade] = Vector{Union{Missing, DateTime}}(missing, n)
    end
    return tradesdf
end

"""
    reachgainuntilreversal!(tradesdf, ix, ...; minpricedelta=0f0)

DataFrame-based variant of limit-reversal lane updates that writes one sample row
state (`longopenlimit`, `longcloselimit`, `shortopenlimit`, `shortcloselimit`,
`tradelabel`, `labelscore`) derived from previous row state and the
current classifier `:label/:score`.
"""
function reachgainuntilreversal!(tradesdf::DataFrame, ix::Integer, openthreshold, buygain, sellgain, limitreduction; minpricedelta::Real=0f0)
    _ensuretradesstrategycolumns!(tradesdf)
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"
    @assert (:label in propertynames(tradesdf)) && (:score in propertynames(tradesdf)) && (:close in propertynames(tradesdf)) "tradesdf must contain :label, :score, :close; names=$(names(tradesdf))"

    label = tradesdf[ix, :label]
    score = Float32(tradesdf[ix, :score])
    close = Float32(tradesdf[ix, :close])
    minpd = Float32(minpricedelta)

    previx = ix - 1
    prev_longopen = previx >= 1 ? tradesdf[previx, :longopenlimit] : missing
    prev_longclose = previx >= 1 ? tradesdf[previx, :longcloselimit] : missing
    prev_shortopen = previx >= 1 ? tradesdf[previx, :shortopenlimit] : missing
    prev_shortclose = previx >= 1 ? tradesdf[previx, :shortcloselimit] : missing

    tradesdf[ix, :longopenlimit] = prev_longopen
    tradesdf[ix, :longcloselimit] = prev_longclose
    tradesdf[ix, :shortopenlimit] = prev_shortopen
    tradesdf[ix, :shortcloselimit] = prev_shortclose
    tradesdf[ix, :tradelabel] = ignore
    tradesdf[ix, :labelscore] = score

    branch = "none"
    longsignal = (label in (longopen, longstrongopen)) && (score >= Float32(openthreshold))
    shortsignal = (label in (shortopen, shortstrongopen)) && (score >= Float32(openthreshold))

    if longsignal
        branch = "long_signal"
        short_candidate = close * (1f0 - Float32(buygain))
        long_open_candidate = close * (1f0 - Float32(buygain))
        long_close_candidate = close * (1f0 + Float32(sellgain))

        if _hasguidance(prev_shortopen, prev_shortclose) && _should_update_price(_missingf32(prev_shortclose), short_candidate, minpd)
            tradesdf[ix, :shortcloselimit] = short_candidate
        end
        if _should_update_price(_missingf32(prev_longopen), long_open_candidate, minpd)
            tradesdf[ix, :longopenlimit] = long_open_candidate
        end
        if _should_update_price(_missingf32(prev_longclose), long_close_candidate, minpd)
            tradesdf[ix, :longcloselimit] = long_close_candidate
        end
        # Long-direction rows are encoded by a zero short open limit.
        tradesdf[ix, :shortopenlimit] = 0f0
        tradesdf[ix, :tradelabel] = (label == longstrongopen ? longstrongopen : longopen)
    elseif shortsignal
        branch = "short_signal"
        long_candidate = close * (1f0 + Float32(buygain))
        short_open_candidate = close * (1f0 + Float32(buygain))
        short_close_candidate = close * (1f0 - Float32(sellgain))

        if _hasguidance(prev_longopen, prev_longclose) && _should_update_price(_missingf32(prev_longclose), long_candidate, minpd)
            tradesdf[ix, :longcloselimit] = long_candidate
        end
        if _should_update_price(_missingf32(prev_shortopen), short_open_candidate, minpd)
            tradesdf[ix, :shortopenlimit] = short_open_candidate
        end
        if _should_update_price(_missingf32(prev_shortclose), short_close_candidate, minpd)
            tradesdf[ix, :shortcloselimit] = short_close_candidate
        end
        # Short-direction rows are encoded by a zero long open limit.
        tradesdf[ix, :longopenlimit] = 0f0
        tradesdf[ix, :tradelabel] = (label == shortstrongopen ? shortstrongopen : shortopen)
    else
        branch = "reduce_limit"
        if _hasguidance(prev_longopen, prev_longclose) && (Float32(prev_longclose) > Float32(prev_longopen))
            candidate = max(close, Float32(prev_longclose) * (1f0 - Float32(sellgain) * Float32(limitreduction)))
            if _should_update_price(_missingf32(prev_longclose), candidate, minpd)
                tradesdf[ix, :longcloselimit] = candidate
            end
        end
        if _hasguidance(prev_shortopen, prev_shortclose) && (Float32(prev_shortclose) < Float32(prev_shortopen))
            candidate = min(close, Float32(prev_shortclose) * (1f0 + Float32(sellgain) * Float32(limitreduction)))
            if _should_update_price(_missingf32(prev_shortclose), candidate, minpd)
                tradesdf[ix, :shortcloselimit] = candidate
            end
        end
    end

    return branch
end

"Adjust lane intents with minimum relative price-delta gating for all updates."
function reachgainuntilreversal_pricedelta!(longta::TradeAction, shortta::TradeAction, label::TradeLabel, score, high, low, close, openthreshold, buygain, sellgain, limitreduction, minpricedelta)
    _ = high
    _ = low
    branch = "none"

    longsignal = (label in [longopen, longstrongopen]) && (score >= openthreshold)
    shortsignal = (label in [shortopen, shortstrongopen]) && (score >= openthreshold)

    !longsignal && _clearopenintent!(longta)
    !shortsignal && _clearopenintent!(shortta)

    if longsignal
        branch = "long_signal"
        short_candidate = close * (1f0 - buygain)
        long_open_candidate = close * (1f0 - buygain)
        long_close_candidate = close * (1f0 + sellgain)

        if _lanehascloseguidance(shortta) && _should_update_price(shortta.closeprice, short_candidate, minpricedelta)
            shortta.closeprice = short_candidate
        end
        if _should_update_price(longta.openprice, long_open_candidate, minpricedelta)
            longta.openprice = long_open_candidate
        end
        longta.label = label == longstrongopen ? longstrongopen : longopen
        if _should_update_price(longta.closeprice, long_close_candidate, minpricedelta)
            longta.closeprice = long_close_candidate
        end
    elseif shortsignal
        branch = "short_signal"
        long_candidate = close * (1f0 + buygain)
        short_open_candidate = close * (1f0 + buygain)
        short_close_candidate = close * (1f0 - sellgain)

        if _lanehascloseguidance(longta) && _should_update_price(longta.closeprice, long_candidate, minpricedelta)
            longta.closeprice = long_candidate
        end
        if _should_update_price(shortta.openprice, short_open_candidate, minpricedelta)
            shortta.openprice = short_open_candidate
        end
        shortta.label = label == shortstrongopen ? shortstrongopen : shortopen
        if _should_update_price(shortta.closeprice, short_close_candidate, minpricedelta)
            shortta.closeprice = short_close_candidate
        end
    else
        branch = "reduce_limit"
        if _lanehascloseguidance(longta) && (longta.closeprice > longta.openprice)
            candidate = max(close, longta.closeprice * (1f0 - sellgain * limitreduction))
            if _should_update_price(longta.closeprice, candidate, minpricedelta)
                longta.closeprice = candidate
            end
        end
        if _lanehascloseguidance(shortta) && (shortta.closeprice < shortta.openprice)
            candidate = min(close, shortta.closeprice * (1f0 + sellgain * limitreduction))
            if _should_update_price(shortta.closeprice, candidate, minpricedelta)
                shortta.closeprice = candidate
            end
        end
    end

    return branch
end

"""Limit-reversal strategy with lane-based independent open/close guidance."""
function gain_limit_reversal!(gs::GainSegment, lastix)
    for ix in (gs.endix + 1):lastix
        processclosedorder!(gs, ix, gs.longta)
        processclosedorder!(gs, ix, gs.shortta)
        reachgainuntilreversal!(
            gs.longta,
            gs.shortta,
            gs.labels[ix],
            gs.scores[ix],
            gs.predictionsdf[ix, :high],
            gs.predictionsdf[ix, :low],
            gs.predictionsdf[ix, :close],
            gs.openthreshold,
            gs.buygain,
            gs.sellgain,
            gs.limitreduction,
        )
    end
    gs.endix = lastix
    _apply_reconciliation_to_lanes!(gs)
    return gs
end

"""
    gain_limit_reversal!(tradesdf, lastix; ...)

Row-oriented Trades DataFrame variant of limit-reversal processing.
"""
function gain_limit_reversal!(tradesdf::DataFrame, lastix::Integer;
    openthreshold::Real=0.6f0,
    buygain::Real=0.001f0,
    sellgain::Real=0.01f0,
    limitreduction::Real=0f0,
    minpricedelta::Real=0f0,
)
    _ensuretradesstrategycolumns!(tradesdf)
    upto = min(Int(lastix), nrow(tradesdf))
    upto <= 0 && return tradesdf
    for ix in 1:upto
        reachgainuntilreversal!(tradesdf, ix, openthreshold, buygain, sellgain, limitreduction; minpricedelta=minpricedelta)
    end
    return tradesdf
end

"""
    gain_limit_reversal!(tp, lastix; ...)

TsTp wrapper around the row-oriented Trades DataFrame variant.
"""
function gain_limit_reversal!(tp::TsTp, lastix::Integer;
    openthreshold::Real=0.6f0,
    buygain::Real=0.001f0,
    sellgain::Real=0.01f0,
    limitreduction::Real=0f0,
    minpricedelta::Real=0f0,
    gaindf::Union{Nothing, DataFrame}=nothing,
    makerfee::Real=0f0,
    takerfee::Real=0f0,
)
    _ = takerfee
    _ensuretradesstrategycolumns!(tp.tradesdf)
    upto = min(Int(lastix), nrow(tp.tradesdf))
    last_openix = 0
    for ix in 1:upto
        reachgainuntilreversal!(tp.tradesdf, ix, openthreshold, buygain, sellgain, limitreduction; minpricedelta=minpricedelta)
        if !isnothing(gaindf)
            last_openix = _materialize_gains_sample_from_trades!(gaindf, tp.tradesdf, ix, last_openix; makerfee=makerfee)
        end
    end
    if nrow(tp.tradesdf) > 0 && (:opentime in propertynames(tp.tradesdf))
        tp.last_update_dt = tp.tradesdf[min(Int(lastix), nrow(tp.tradesdf)), :opentime]
    end
    return tp
end

function _materialize_gains_sample_from_trades!(result::DataFrame, tradesdf::DataFrame, ix::Integer, last_openix::Int; makerfee::Real=0f0)::Int
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"

    time = tradesdf[ix, :opentime]
    high = tradesdf[ix, :high]
    low = tradesdf[ix, :low]
    label = tradesdf[ix, :tradelabel]
    lo = tradesdf[ix, :longopenlimit]
    lc = tradesdf[ix, :longcloselimit]
    so = tradesdf[ix, :shortopenlimit]
    sc = tradesdf[ix, :shortcloselimit]

    longsignal = islongopenlabel(label)
    shortsignal = isshortopenlabel(label)

    if longsignal
        @assert _hasguidance(lo, lc) "Missing long guidance for long open signal at ix=$(ix): lo=$(lo), lc=$(lc)"
        @assert !_openlimitactive(so) "long signal requires shortopenlimit == 0f0 at ix=$(ix): shortopenlimit=$(so)"
    end
    if shortsignal
        @assert _hasguidance(so, sc) "Missing short guidance for short open signal at ix=$(ix): so=$(so), sc=$(sc)"
        @assert !_openlimitactive(lo) "short signal requires longopenlimit == 0f0 at ix=$(ix): longopenlimit=$(lo)"
    end

    if last_openix == 0
        #* last_openix is only set the first time a gain segment hits the open bar and not any later hits even with an open signal
        #* that shall be changed if a multi open approach is considered at different entry levels
        long_open_hit = _openlimitactive(lo) && _price_in_bar(Float32(lo), low, high, :high)
        short_open_hit = _openlimitactive(so) && _price_in_bar(Float32(so), low, high, :low)
        @assert !(long_open_hit && short_open_hit) "Both long and short open limits matched same bar at ix=$(ix): lo=$(lo), so=$(so), low=$(low), high=$(high)"
        if long_open_hit || short_open_hit
            last_openix = ix
        end
    end

    if last_openix > 0
        start_lo = tradesdf[last_openix, :longopenlimit]
        start_so = tradesdf[last_openix, :shortopenlimit]
        start_is_long = _openlimitactive(start_lo)
        start_is_short = _openlimitactive(start_so)
        @assert start_is_long != start_is_short "Open direction cannot be inferred at last_openix=$(last_openix): start_lo=$(start_lo), start_so=$(start_so)"

        if start_is_long
            @assert !ismissing(lc) && (Float32(lc) > 0f0) "Missing long close guidance while position is open at ix=$(ix), last_openix=$(last_openix), lc=$(lc)"
            if _price_in_bar(Float32(lc), low, high, :low)
                startix = last_openix
                starttime = tradesdf[startix, :opentime]
                openprice = Float32(tradesdf[startix, :longopenlimit])
                @assert openprice > 0f0 "Invalid long openprice at startix=$(startix): openprice=$(openprice)"
                minutes = Int(div(Dates.value(time - starttime), 60000)) + 1
                gain = (Float32(lc) - openprice) / openprice
                push!(result, (up, (ix - startix + 1), minutes, gain, (gain - 2f0 * Float32(makerfee)), starttime, time, startix, ix))
                last_openix = 0
            end
        else
            @assert !ismissing(sc) && (Float32(sc) > 0f0) "Missing short close guidance while position is open at ix=$(ix), last_openix=$(last_openix), sc=$(sc)"
            if _price_in_bar(Float32(sc), low, high, :high)
                startix = last_openix
                starttime = tradesdf[startix, :opentime]
                openprice = Float32(tradesdf[startix, :shortopenlimit])
                @assert openprice > 0f0 "Invalid short openprice at startix=$(startix): openprice=$(openprice)"
                minutes = Int(div(Dates.value(time - starttime), 60000)) + 1
                gain = -(Float32(sc) - openprice) / openprice
                push!(result, (down, (ix - startix + 1), minutes, gain, (gain - 2f0 * Float32(makerfee)), starttime, time, startix, ix))
                last_openix = 0
            end
        end
    end

    if last_openix > 0
        _setlastopentrade!(tradesdf, ix, tradesdf[last_openix, :opentime])
    else
        _setlastopentrade!(tradesdf, ix, missing)
    end

    return last_openix
end

"Limit-reversal strategy variant with minimum relative price-delta update gating."
function gain_limit_reversal_pricedelta!(gs::GainSegment, lastix)
    for ix in (gs.endix + 1):lastix
        processclosedorder!(gs, ix, gs.longta)
        processclosedorder!(gs, ix, gs.shortta)
        reachgainuntilreversal_pricedelta!(
            gs.longta,
            gs.shortta,
            gs.labels[ix],
            gs.scores[ix],
            gs.predictionsdf[ix, :high],
            gs.predictionsdf[ix, :low],
            gs.predictionsdf[ix, :close],
            gs.openthreshold,
            gs.buygain,
            gs.sellgain,
            gs.limitreduction,
            gs.minpricedelta,
        )
    end
    gs.endix = lastix
    _apply_reconciliation_to_lanes!(gs)
    return gs
end

"Replay runtime-equivalent classification gating over precomputed score/label vectors."
function replay_classification_gating(gs::GainSegment, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector)
    n = length(scores)
    @assert n == length(labels) == size(predictionsdf, 1) "n=$(n) must match labels and predictionsdf rows"

    out_scores = Float32.(scores)
    out_labels = Targets.TradeLabel[labels[i] for i in eachindex(labels)]
    if n <= 1 || !_classifier_gating_enabled(gs)
        return out_scores, out_labels
    end

    last_keep_ix = 1
    for ix in 2:n
        interval_ok = true
        if gs.max_classify_staleness_minutes > 0
            elapsed_minutes = Int(div(Dates.value(predictionsdf[ix, :opentime] - predictionsdf[last_keep_ix, :opentime]), 60000))
            interval_ok = elapsed_minutes >= gs.max_classify_staleness_minutes
        end

        delta_ok = true
        if gs.minpricedelta > 0f0
            delta_ok = _relpricedelta(predictionsdf[ix, :close], predictionsdf[last_keep_ix, :close]) >= gs.minpricedelta
        end

        if _classification_triggered(gs, interval_ok, delta_ok)
            last_keep_ix = ix
        else
            out_scores[ix] = out_scores[last_keep_ix]
            out_labels[ix] = out_labels[last_keep_ix]
        end
    end

    return out_scores, out_labels
end

"Synchronize one pair-scoped Trades DataFrame from prediction inputs and metadata."
function _synctradesframe!(ts::TsCache, xc::Xch.XchCache, base::AbstractString, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector;
    quotecoin::AbstractString=EnvConfig.pairquote,
    metadata::AbstractDict{Symbol, Any}=Dict{Symbol, Any}(),
    datetime::Union{Nothing, DateTime}=nothing,
)::TsTp
    n = size(predictionsdf, 1)
    @assert n == length(scores) == length(labels) "size(predictionsdf, 1)=$(n) must match scores=$(length(scores)) and labels=$(length(labels))"

    tradesdf = DataFrame(predictionsdf; copycols=true)
    tradesdf[!, :score] = Float32.(scores)
    tradesdf[!, :label] = Targets.TradeLabel[labels[ix] for ix in eachindex(labels)]
    tradesdf[!, :coin] = fill(uppercase(String(base)), n)
    tradesdf[!, :pair] = fill(tspairkey(base, quotecoin), n)

    for (k, v) in metadata
        tradesdf[!, k] = fill(v, n)
    end

    Xch.settrades!(xc, base, quotecoin, tradesdf)
    return syncpairtrades!(ts, xc, base, quotecoin; datetime=datetime)
end

"""
Run one gain algorithm pass and optionally force-close open lanes at `lastix`.

The input `predictionsdf` must provide columns `:opentime`, `:high`, `:low`, `:close`.
"""
function getgains(gs::GainSegment, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector, forcegain::Bool; lastix=lastindex(scores), openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
    @assert length(scores) == length(labels) == size(predictionsdf, 1) "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1))"

    gs.openthreshold = Float32(openthreshold)
    gs.closethreshold = Float32(closethreshold)
    gs.predictionsdf = predictionsdf
    gs.scores = scores
    gs.labels = labels
    gs.algorithm(gs, lastix)
    _apply_reconciliation_to_lanes!(gs)

    if forcegain
        calcgain!(gs, lastix)
        _apply_reconciliation_to_lanes!(gs)
    end
    return gs.gaindf
end

"""
Materialize gain summary DataFrame directly from Trades DataFrame row-state.

This phase 2 adaptation reads tradelabel and limit price columns to extract
gain segments without falling back to legacy GainSegment computation.
"""
function materialize_gains_from_trades(tradesdf::DataFrame, predictionsdf::AbstractDataFrame; makerfee::Real=0f0, takerfee::Real=0f0)::DataFrame
    result = emptygaindf()
    _ = takerfee
    if nrow(tradesdf) == 0 || !(:tradelabel in propertynames(tradesdf))
        return result
    end

    @assert nrow(tradesdf) == nrow(predictionsdf) "Trades and predictions must have same row count"
    @assert (:opentime in propertynames(tradesdf)) && (:high in propertynames(tradesdf)) && (:low in propertynames(tradesdf)) "Required columns missing"
    @assert (:longopenlimit in propertynames(tradesdf)) && (:longcloselimit in propertynames(tradesdf)) && (:shortopenlimit in propertynames(tradesdf)) && (:shortcloselimit in propertynames(tradesdf)) "Required strategy columns missing"

    _ensuretradesstrategycolumns!(tradesdf)
    last_openix = 0
    for ix in 1:nrow(tradesdf)
        last_openix = _materialize_gains_sample_from_trades!(result, tradesdf, ix, last_openix; makerfee=makerfee)
    end

    if nrow(result) == 0
        labels = tradesdf[!, :tradelabel]
        open_signal_rows = count(l -> islongopenlabel(l) || isshortopenlabel(l), labels)
        close_signal_rows = count(l -> islongcloselabel(l) || isshortcloselabel(l), labels)

        long_guidance_rows = 0
        short_guidance_rows = 0
        long_open_touch_rows = 0
        long_close_touch_rows = 0
        short_open_touch_rows = 0
        short_close_touch_rows = 0
        for ix in 1:nrow(tradesdf)
            lo = tradesdf[ix, :longopenlimit]
            lc = tradesdf[ix, :longcloselimit]
            so = tradesdf[ix, :shortopenlimit]
            sc = tradesdf[ix, :shortcloselimit]
            high = tradesdf[ix, :high]
            low = tradesdf[ix, :low]

            if _hasguidance(lo, lc)
                long_guidance_rows += 1
                _price_in_bar(Float32(lo), low, high, :high) && (long_open_touch_rows += 1)
                _price_in_bar(Float32(lc), low, high, :low) && (long_close_touch_rows += 1)
            end
            if _hasguidance(so, sc)
                short_guidance_rows += 1
                _price_in_bar(Float32(so), low, high, :low) && (short_open_touch_rows += 1)
                _price_in_bar(Float32(sc), low, high, :high) && (short_close_touch_rows += 1)
            end
        end

        coin = (:coin in propertynames(tradesdf) && nrow(tradesdf) > 0) ? tradesdf[1, :coin] : missing
        predicted = (:predicted in propertynames(tradesdf) && nrow(tradesdf) > 0) ? tradesdf[1, :predicted] : missing
        rangeid = (:rangeid in propertynames(tradesdf) && nrow(tradesdf) > 0) ? tradesdf[1, :rangeid] : missing
        openthreshold = (:openthreshold in propertynames(tradesdf) && nrow(tradesdf) > 0) ? tradesdf[1, :openthreshold] : missing
        closethreshold = (:closethreshold in propertynames(tradesdf) && nrow(tradesdf) > 0) ? tradesdf[1, :closethreshold] : missing

        @warn "materialize_gains_from_trades produced empty output" coin=coin predicted=predicted rangeid=rangeid openthreshold=openthreshold closethreshold=closethreshold rows=nrow(tradesdf) open_signal_rows=open_signal_rows close_signal_rows=close_signal_rows long_guidance_rows=long_guidance_rows short_guidance_rows=short_guidance_rows long_open_touch_rows=long_open_touch_rows long_close_touch_rows=long_close_touch_rows short_open_touch_rows=short_open_touch_rows short_close_touch_rows=short_close_touch_rows
    end

    return result
end

"""
Phase 2 gain evaluation path using `TsCache` and Xch pair-scoped Trades DataFrames.

This keeps `Xch` as owner of mutable pair tables and materializes gains directly
from the Trades DataFrame row-state (tradelabel and limit prices) without fallback
to legacy GainSegment computation.
"""
function getgains(ts::TsCache, xc::Xch.XchCache, base::AbstractString, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector, forcegain::Bool;
    strategy::Union{Nothing, GainSegment, StrategySpec}=nothing,
    lastix::Integer=lastindex(scores),
    openthreshold=0.6f0,
    closethreshold=0.1f0,
    quotecoin::AbstractString=EnvConfig.pairquote,
    metadata::AbstractDict{Symbol, Any}=Dict{Symbol, Any}(),
    datetime::Union{Nothing, DateTime}=nothing,
)
    raw_template = if !isnothing(strategy)
        strategy
    elseif haskey(ts.configuration, :strategy_template)
        ts.configuration[:strategy_template]
    else
        GainSegment()
    end
    gs_template = if raw_template isa GainSegment
        raw_template
    elseif raw_template isa StrategySpec
        _togainsegment(raw_template)
    else
        throw(ArgumentError("strategy template must be GainSegment or StrategySpec, got $(typeof(raw_template))"))
    end

    tp = _synctradesframe!(ts, xc, base, predictionsdf, scores, labels;
        quotecoin=quotecoin,
        metadata=metadata,
        datetime=datetime,
    )

    gaindf = emptygaindf()

    # Phase 2 row-oriented state update and per-sample materialization.
    row_lastix = min(Int(lastix), nrow(tp.tradesdf))
    if row_lastix > 0
        if gs_template.algorithm === gain_limit_reversal!
            gain_limit_reversal!(tp, row_lastix;
                openthreshold=Float32(openthreshold),
                buygain=gs_template.buygain,
                sellgain=gs_template.sellgain,
                limitreduction=gs_template.limitreduction,
                minpricedelta=0f0,
                gaindf=gaindf,
                makerfee=gs_template.makerfee,
                takerfee=gs_template.takerfee,
            )
        elseif gs_template.algorithm === gain_limit_reversal_pricedelta!
            gain_limit_reversal!(tp, row_lastix;
                openthreshold=Float32(openthreshold),
                buygain=gs_template.buygain,
                sellgain=gs_template.sellgain,
                limitreduction=gs_template.limitreduction,
                minpricedelta=gs_template.minpricedelta,
                gaindf=gaindf,
                makerfee=gs_template.makerfee,
                takerfee=gs_template.takerfee,
            )
        end
    end

    return gaindf
end

end # module
