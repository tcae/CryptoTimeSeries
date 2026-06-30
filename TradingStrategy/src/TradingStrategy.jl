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
using EnvConfig, Features, Targets, Classify, Xch, Ohlcv

"""Ensure Trades column `lastopentrade` exists. Owner: TradingStrategy. Eltype: `Union{Missing,DateTime}`."""
function tradesdf_lastopentrade(tradesdf::DataFrame)::DataFrame
    if :lastopentrade ∉ propertynames(tradesdf)
        tradesdf[!, :lastopentrade] = Vector{Union{Missing, DateTime}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `tradelabel` exists. Owner: TradingStrategy. Eltype: `TradeLabel` semantics."""
function tradesdf_tradelabel(tradesdf::DataFrame)::DataFrame
    if :tradelabel ∉ propertynames(tradesdf)
        tradesdf[!, :tradelabel] = Vector{Union{Missing, TradeLabel}}(missing, nrow(tradesdf))
    else
        tradesdf[!, :tradelabel] = Union{Missing, TradeLabel}[
            ismissing(v) ? missing : (v isa TradeLabel ? v : Targets.tradelabel(String(v)))
            for v in tradesdf[!, :tradelabel]
        ]
    end
    return tradesdf
end

"""Ensure Trades column `labelscore` exists. Owner: TradingStrategy. Eltype: `Float32`."""
function tradesdf_labelscore(tradesdf::DataFrame)::DataFrame
    if :labelscore ∉ propertynames(tradesdf)
        tradesdf[!, :labelscore] = zeros(Float32, nrow(tradesdf))
    end
    return tradesdf
end


"""Ensure Trades column `longopenlimit` exists. Owner: TradingStrategy. Eltype: `Union{Missing,Float32}`."""
function tradesdf_longopenlimit(tradesdf::DataFrame)::DataFrame
    if :longopenlimit ∉ propertynames(tradesdf)
        tradesdf[!, :longopenlimit] = Vector{Union{Missing, Float32}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `longcloselimit` exists. Owner: TradingStrategy. Eltype: `Union{Missing,Float32}`."""
function tradesdf_longcloselimit(tradesdf::DataFrame)::DataFrame
    if :longcloselimit ∉ propertynames(tradesdf)
        tradesdf[!, :longcloselimit] = Vector{Union{Missing, Float32}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `shortopenlimit` exists. Owner: TradingStrategy. Eltype: `Union{Missing,Float32}`."""
function tradesdf_shortopenlimit(tradesdf::DataFrame)::DataFrame
    if :shortopenlimit ∉ propertynames(tradesdf)
        tradesdf[!, :shortopenlimit] = Vector{Union{Missing, Float32}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `shortcloselimit` exists. Owner: TradingStrategy. Eltype: `Union{Missing,Float32}`."""
function tradesdf_shortcloselimit(tradesdf::DataFrame)::DataFrame
    if :shortcloselimit ∉ propertynames(tradesdf)
        tradesdf[!, :shortcloselimit] = Vector{Union{Missing, Float32}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Return TradingStrategy/Trade-contributed Trades schema initializer functions."""
function tradesdf_contributors()::Vector{Function}
    return Function[
        tradesdf_lastopentrade,
        tradesdf_tradelabel,
        tradesdf_labelscore,
        tradesdf_longopenlimit,
        tradesdf_longcloselimit,
        tradesdf_shortopenlimit,
        tradesdf_shortcloselimit,
    ]
end

"""Return the normalized config-scoped subfolder used for persisted trade artifacts."""
function tradesfolder(stem::AbstractString="gains")::String
    normalized = replace(normpath(splitext(String(stem))[1]), '\\' => '/')
    return startswith(normalized, "trades/") || (normalized == "trades") ? normalized : joinpath("trades", normalized)
end

"""Return the aggregate storage key used for one persisted trade artifact."""
tradesaggregate(stem::AbstractString="gains") = joinpath("trades", splitext(basename(String(stem)))[1] * "_all")

"""Return the per-coin storage key used for one persisted trade artifact."""
tradefilename(coin::AbstractString; stem::AbstractString="gains") = joinpath(tradesfolder(stem), uppercase(strip(String(coin))))

function _tradebasekey(tradedf::AbstractDataFrame, ix::Integer)::Union{Nothing, String}
    if :coin in propertynames(tradedf)
        coin = tradedf[ix, :coin]
        if !ismissing(coin)
            normalized = uppercase(strip(String(coin)))
            isempty(normalized) || return normalized
        end
    end
    if :pair in propertynames(tradedf)
        pair = tradedf[ix, :pair]
        if !ismissing(pair)
            bq = try
                Xch.basequote(String(pair))
            catch
                nothing
            end
            isnothing(bq) || return uppercase(String(bq.basecoin))
        end
    end
    return nothing
end

"""Persist a trades dataframe into config-scoped storage, plus optional aggregate copy."""
function savetrades(tradedf::AbstractDataFrame; stem::AbstractString="gains", include_aggregate::Bool=true)
    if size(tradedf, 1) == 0
        return String[]
    end
    @assert (:coin in propertynames(tradedf)) || (:pair in propertynames(tradedf)) "tradedf must contain a :coin or :pair column; names=$(names(tradedf))"

    paths = String[]
    basekeys = [_tradebasekey(tradedf, ix) for ix in 1:nrow(tradedf)]
    @assert all(!isnothing, basekeys) "tradedf must provide a resolvable base via :coin or :pair for every row; names=$(names(tradedf))"
    coins = unique(String[key for key in basekeys if !isnothing(key)])
    for coin in coins
        selectrows = [!isnothing(key) && (key == coin) for key in basekeys]
        coindf = DataFrame(tradedf[selectrows, :])
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

"""
Immutable strategy configuration payload for runtime strategy execution.
"""
Base.@kwdef struct StrategyConfig
    algorithm::Function = gain_limit_reversal!
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

"""Per-trading-pair runtime state holder used by `TsCache`."""
Base.@kwdef mutable struct TsTp
    pair::String
    tradesdf::DataFrame = DataFrame()
    closeprices::Vector{Float32} = Float32[]
    last_update_dt::Union{Nothing, DateTime} = nothing
end

"""
Internal runtime cache for the Phase 2 Trades DataFrame architecture.

`TsCache` keeps pair-scoped runtime references while `Xch` remains owner of the
mutable per-pair Trades DataFrames.
"""
mutable struct TsCache
    configuration::Dict{Symbol, Any}
    classifier::Classify.AbstractClassifier
    pairs::Dict{String, TsTp}
    classifier_gate_state::Dict{String, NamedTuple{(:last_advice, :last_classify_close), Tuple{Any, Float32}}}
    accepted::Set{String}
    strategy_config::Any
    source::String
end

"Build TsCache with explicit classifier wiring from argument or configuration."
function TsCache(; configuration::Dict{Symbol, Any}=Dict{Symbol, Any}(), classifier::Union{Nothing, Classify.AbstractClassifier}=nothing, strategy::Any=nothing, source::AbstractString="manual", mode=EnvConfig.configmode)
    resolved_classifier = if !isnothing(classifier)
        classifier
    else
        Classify.resolveclassifier(
            classifier=get(configuration, :classifier, nothing),
            classifier_factory=get(configuration, :classifier_factory, nothing),
            classifier_type=get(configuration, :classifier_type, nothing),
            classifier_spec=get(configuration, :classifier_spec, nothing),
            mode=mode,
        )
    end
    raw_template = if !isnothing(strategy)
        strategy
    elseif haskey(configuration, :strategy_template)
        configuration[:strategy_template]
    else
        StrategyConfig()
    end
    resolved_template = raw_template isa StrategyConfig ? raw_template : throw(ArgumentError("strategy template must be TradingStrategy.StrategyConfig, got $(typeof(raw_template))"))
    return TsCache(configuration, resolved_classifier, Dict{String, TsTp}(), Dict{String, NamedTuple{(:last_advice, :last_classify_close), Tuple{Any, Float32}}}(), Set{String}(), resolved_template, String(source))
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
    isempty(tp.closeprices) || empty!(tp.closeprices)
    tp.last_update_dt = datetime
    return tp
end

"Synchronize one TsCache pair entry to the Xch-owned mutable Trades DataFrame."
function syncpairtrades!(ts::TsCache, xc::Xch.XchCache, base::AbstractString, quotecoin::AbstractString; datetime::Union{Nothing, DateTime}=nothing)::TsTp
    pair = tspairkey(base, quotecoin)
    tp = getpairstate!(ts, pair)
    tp.tradesdf = Xch.trades(xc, base, quotecoin)
    isempty(tp.closeprices) || empty!(tp.closeprices)
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

@inline max_classify_staleness_minutes(spec::StrategyConfig) = spec.max_classify_staleness_minutes

"Return default execution-state reconciliation payload used by runtime strategy evaluation."
function defaultreconciliationinput()
    return (
        has_long_open=false,
        long_avg_entry=0f0,
        long_open_ix=0,
        has_short_open=false,
        short_avg_entry=0f0,
        short_open_ix=0,
    )
end

function _normalizereconciliationinput(reconciliation)
    if isnothing(reconciliation)
        return defaultreconciliationinput()
    end
    return (
        has_long_open=Bool(getproperty(reconciliation, :has_long_open)),
        long_avg_entry=Float32(getproperty(reconciliation, :long_avg_entry)),
        long_open_ix=Int(getproperty(reconciliation, :long_open_ix)),
        has_short_open=Bool(getproperty(reconciliation, :has_short_open)),
        short_avg_entry=Float32(getproperty(reconciliation, :short_avg_entry)),
        short_open_ix=Int(getproperty(reconciliation, :short_open_ix)),
    )
end

acceptedbases(rt::TsCache)::Set{String} = copy(rt.accepted)

"Drop one base from TsCache, including classifier and cached pair state."
function dropbase!(rt::TsCache, base::AbstractString)::Nothing
    basekey = uppercase(String(base))
    try
        Classify.removebase!(rt.classifier, basekey)
    catch
    end
    droppair!(rt, tspairkey(basekey, EnvConfig.pairquote))
    delete!(rt.classifier_gate_state, basekey)
    delete!(rt.accepted, basekey)
    return nothing
end

"Reset TsCache runtime, clearing accepted bases and cached classifier/gate state."
function reset!(rt::TsCache)::Nothing
    empty!(rt.pairs)
    empty!(rt.classifier_gate_state)
    empty!(rt.accepted)
    try
        Classify.removebase!(rt.classifier, nothing)
    catch
    end
    return nothing
end

"Apply a strategy-spec template to TsCache and clear derived cached state."
function apply_strategy!(rt::TsCache, strategy::StrategyConfig; source::AbstractString="manual")::Nothing
    rt.strategy_config = strategy
    rt.source = String(source)
    empty!(rt.pairs)
    empty!(rt.classifier_gate_state)
    empty!(rt.accepted)
    try
        Classify.removebase!(rt.classifier, nothing)
    catch
    end
    return nothing
end

"Return the per-base classifier gate state, creating an empty one when needed."
function _runtimegatestate!(rt::TsCache, base::AbstractString)
    basekey = uppercase(String(base))
    return get!(rt.classifier_gate_state, basekey) do
        (last_advice=nothing, last_classify_close=0f0)
    end
end

function _set_runtimegatestate!(rt::TsCache, base::AbstractString; last_advice, last_classify_close::Real)
    basekey = uppercase(String(base))
    rt.classifier_gate_state[basekey] = (
        last_advice=last_advice,
        last_classify_close=Float32(last_classify_close),
    )
    return rt
end

@inline function _classification_triggered(spec::StrategyConfig, interval_ok::Bool, delta_ok::Bool)::Bool
    interval_enabled = spec.max_classify_staleness_minutes > 0
    delta_enabled = spec.minpricedelta > 0f0
    !(interval_enabled || delta_enabled) && return true
    return (interval_enabled && interval_ok) || (delta_enabled && delta_ok)
end

function _should_skip_classifier(spec::StrategyConfig, gate, datetime::DateTime, closeprice::Float32, last_open_dt::Union{Nothing, DateTime})::Bool
    isnothing(gate.last_advice) && return false

    interval_ok = true
    if spec.max_classify_staleness_minutes > 0
        isnothing(last_open_dt) && return false
        elapsed_minutes = Int(div(Dates.value(datetime - last_open_dt), 60000))
        interval_ok = elapsed_minutes >= spec.max_classify_staleness_minutes
    end

    delta_ok = true
    if spec.minpricedelta > 0f0
        gate.last_classify_close > 0f0 || return false
        delta_ok = _relpricedelta(closeprice, gate.last_classify_close) >= spec.minpricedelta
    end

    return !_classification_triggered(spec, interval_ok, delta_ok)
end

function _lastopentrade_dt(tradesdf::AbstractDataFrame)::Union{Nothing, DateTime}
    (:lastopentrade in propertynames(tradesdf)) || return nothing
    for ix in nrow(tradesdf):-1:1
        dt = tradesdf[ix, :lastopentrade]
        ismissing(dt) || return dt
    end
    return nothing
end

"Prepare TsCache for requested bases using available OHLCV data and update accepted set."
function preparebases!(rt::TsCache, xc::Xch.XchCache, bases::AbstractVector{<:AbstractString}; datetime::DateTime, updatecache::Bool=false)::Nothing
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
    for base in sort!(collect(rt.accepted))
        syncpairtrades!(rt, xc, base, EnvConfig.pairquote; datetime=datetime)
    end
    return nothing
end

"""
Return cached or freshly computed classifier advice together with the current bar context.

This is the read-oriented phase of live runtime processing. It resolves the
current OHLCV row, applies classifier gating, and returns the advice payload
needed by the row-application phase.
"""
function _classify_base_advice!(rt::TsCache, xc::Xch.XchCache, base::AbstractString, datetime::DateTime)::Union{Nothing, NamedTuple}
    basekey = uppercase(String(base))
    if !haskey(xc.bases, basekey)
        (EnvConfig.verbosity >= 1) && @warn "base OHLCV unavailable in exchange cache; skipping gettradesrow!" base=basekey
        return nothing
    end

    ohlcv = Xch.ohlcv(xc, basekey)
    spec = rt.strategy_config
    gate = _runtimegatestate!(rt, basekey)
    tdf = Xch.trades(xc, basekey, EnvConfig.pairquote)
    last_open_dt = _lastopentrade_dt(tdf)

    rowix = ohlcv.ix
    odf = Ohlcv.dataframe(ohlcv)
    @assert (1 <= rowix <= size(odf, 1)) "rowix=$(rowix) out of bounds for ohlcv rows=$(size(odf, 1))"
    closeprice = Float32(odf[rowix, :close])

    advice = if _should_skip_classifier(spec, gate, datetime, closeprice, last_open_dt)
        gate.last_advice
    else
        fresh = Classify.advice(rt.classifier, basekey, datetime, investment=nothing)
        if !isnothing(fresh)
            _set_runtimegatestate!(rt, basekey; last_advice=fresh, last_classify_close=closeprice)
        end
        fresh
    end

    isnothing(advice) && return nothing
    return (
        base=basekey,
        datetime=datetime,
        ohlcv=ohlcv,
        closeprice=closeprice,
        advice=advice,
        spec=spec,
    )
end

"""
Apply one classifier advice payload to the Xch-owned trades row and return row metadata.

This is the write-oriented phase of live runtime processing. It mutates the
pair Trades DataFrame through the configured strategy algorithm and enriches the
current row with reconciliation state when present.
"""
function _apply_base_advice_row!(rt::TsCache, xc::Xch.XchCache, advicectx::NamedTuple; reconciliation=nothing)::NamedTuple
    basekey = String(advicectx.base)
    datetime = advicectx.datetime
    ohlcv = advicectx.ohlcv
    closeprice = Float32(advicectx.closeprice)
    advice = advicectx.advice
    spec = advicectx.spec

    syncpairtrades!(rt, xc, basekey, EnvConfig.pairquote; datetime=datetime)
    odf = Ohlcv.dataframe(ohlcv)
    rowix = ohlcv.ix
    opentime = odf[rowix, :opentime]
    row = Xch.ensuretradesrow!(xc, basekey, EnvConfig.pairquote, opentime)
    tdf = row.tradesdf
    trow = row.rowix

    spec.algorithm(
        tdf,
        trow,
        advice.tradelabel,
        Float32(advice.probability),
        closeprice;
        openthreshold=spec.openthreshold,
        buygain=spec.buygain,
        sellgain=spec.sellgain,
        limitreduction=spec.limitreduction,
        minpricedelta=spec.minpricedelta,
    )

    recon = _normalizereconciliationinput(reconciliation)
    if recon.has_long_open
        if ismissing(tdf[trow, :longopenlimit]) || (Float32(tdf[trow, :longopenlimit]) <= 0f0)
            tdf[trow, :longopenlimit] = recon.long_avg_entry
        end
        if ismissing(tdf[trow, :longcloselimit]) || (Float32(tdf[trow, :longcloselimit]) <= 0f0)
            tdf[trow, :longcloselimit] = recon.long_avg_entry * (1f0 + spec.sellgain)
        end
        tdf[trow, :lastopentrade] = tdf[trow, :opentime]
    end
    if recon.has_short_open
        if ismissing(tdf[trow, :shortopenlimit]) || (Float32(tdf[trow, :shortopenlimit]) <= 0f0)
            tdf[trow, :shortopenlimit] = recon.short_avg_entry
        end
        if ismissing(tdf[trow, :shortcloselimit]) || (Float32(tdf[trow, :shortcloselimit]) <= 0f0)
            tdf[trow, :shortcloselimit] = recon.short_avg_entry * (1f0 - spec.sellgain)
        end
        tdf[trow, :lastopentrade] = tdf[trow, :opentime]
    end

    cfgid = try
        Int(advice.configid)
    catch
        0
    end

    return (
        base=basekey,
        datetime=datetime,
        tradesdf=tdf,
        rowix=Int(trow),
        probability=Float32(advice.probability),
        configid=cfgid,
        source=:tradingstrategy,
    )
end

"Update one base row in the Xch-owned trades dataframe using TsCache runtime state."
function gettradesrow!(rt::TsCache, xc::Xch.XchCache, base::AbstractString, datetime::DateTime; reconciliation=nothing)::Union{Nothing, NamedTuple}
    advicectx = _classify_base_advice!(rt, xc, base, datetime)
    isnothing(advicectx) && return nothing
    return _apply_base_advice_row!(rt, xc, advicectx; reconciliation=reconciliation)
end

"Update requested bases in Xch-owned trades dataframes using TsCache runtime state."
function gettradesrows!(rt::TsCache, xc::Xch.XchCache, bases::AbstractVector{<:AbstractString}, datetime::DateTime; reconciliation_by_base::AbstractDict=Dict{String, Any}())::Vector{NamedTuple}
    rows = NamedTuple[]
    for base in bases
        basekey = uppercase(String(base))
        recon = get(reconciliation_by_base, basekey, defaultreconciliationinput())
        rowmeta = gettradesrow!(rt, xc, basekey, datetime; reconciliation=recon)
        isnothing(rowmeta) || push!(rows, rowmeta)
    end
    return rows
end

@inline _missingf32(x) = ismissing(x) ? 0f0 : Float32(x)
@inline _hasguidance(openlimit, closelimit) = (!ismissing(openlimit)) && (!ismissing(closelimit)) && (Float32(openlimit) > 0f0) && (Float32(closelimit) > 0f0)
@inline _openlimitactive(openlimit) = (!ismissing(openlimit)) && (Float32(openlimit) > 0f0)
@inline _normalizedtradelabel(x) = ismissing(x) ? ignore : (x isa TradeLabel ? x : Targets.tradelabel(String(x)))

@inline function _setlastopentrade!(tradesdf::DataFrame, ix::Integer, dt)
    tradesdf[ix, :lastopentrade] = dt
    return nothing
end

"""
    gain_limit_reversal!(tradesdf, ix, label, score, close; ...)

DataFrame-based limit-reversal lane update that writes one sample row state
(`longopenlimit`, `longcloselimit`, `shortopenlimit`, `shortcloselimit`,
`tradelabel`, `labelscore`) derived from previous row state and the
current classifier label/score input.
"""
function gain_limit_reversal!(tradesdf::DataFrame, ix::Integer, label::TradeLabel, score::Real, close::Real;
    openthreshold::Real=0.6f0,
    buygain::Real=0.001f0,
    sellgain::Real=0.01f0,
    limitreduction::Real=0f0,
    minpricedelta::Real=0f0,
)
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"

    scoref = Float32(score)
    closef = Float32(close)
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
    tradesdf[ix, :labelscore] = scoref

    branch = "none"
    longsignal = (label in (longopen, longstrongopen)) && (scoref >= Float32(openthreshold))
    shortsignal = (label in (shortopen, shortstrongopen)) && (scoref >= Float32(openthreshold))

    if longsignal
        branch = "long_signal"
        short_candidate = closef * (1f0 - Float32(buygain))
        long_open_candidate = closef * (1f0 - Float32(buygain))
        long_close_candidate = closef * (1f0 + Float32(sellgain))

        if _should_update_price(_missingf32(prev_shortclose), short_candidate, minpd)
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
        tradesdf[ix, :tradelabel] = label
    elseif shortsignal
        branch = "short_signal"
        long_candidate = closef * (1f0 + Float32(buygain))
        short_open_candidate = closef * (1f0 + Float32(buygain))
        short_close_candidate = closef * (1f0 - Float32(sellgain))

        if _should_update_price(_missingf32(prev_longclose), long_candidate, minpd)
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
        tradesdf[ix, :tradelabel] = label
    else
        branch = "reduce_limit"
        if _hasguidance(prev_longopen, prev_longclose) && (Float32(prev_longclose) > Float32(prev_longopen))
            long_candidate = max(closef, Float32(prev_longclose) * (1f0 - Float32(sellgain) * Float32(limitreduction)))
            if _should_update_price(_missingf32(prev_longclose), long_candidate, minpd)
                tradesdf[ix, :longcloselimit] = long_candidate
            end
        end
        if _hasguidance(prev_shortopen, prev_shortclose) && (Float32(prev_shortclose) < Float32(prev_shortopen))
            short_candidate = min(closef, Float32(prev_shortclose) * (1f0 + Float32(sellgain) * Float32(limitreduction)))
            if _should_update_price(_missingf32(prev_shortclose), short_candidate, minpd)
                tradesdf[ix, :shortcloselimit] = short_candidate
            end
        end
    end

    return branch
end

@inline function _validate_lastix(lastix::Integer, nrows::Integer)::Nothing
    @assert lastix >= 0 "lastix must be >= 0, got lastix=$(lastix), nrows=$(nrows)"
    @assert lastix <= nrows "lastix must be <= nrows, got lastix=$(lastix), nrows=$(nrows)"
    return nothing
end

function gain_limit_reversal! end

"""Execute per-sample strategy updates and optional gain materialization outside the algorithm implementation."""
function simulate_gains!(tp::TsTp, lastix::Integer;
    algorithm::Function=gain_limit_reversal!,
    openthreshold::Real=0.6f0,
    buygain::Real=0.001f0,
    sellgain::Real=0.01f0,
    limitreduction::Real=0f0,
    minpricedelta::Real=0f0,
    gaindf::Union{Nothing, DataFrame}=nothing,
    makerfee::Real=0f0,
    takerfee::Real=0f0,
    closeprices::Union{Nothing, AbstractVector}=nothing,
)
    _ = takerfee
    lastix = Int(lastix)
    _validate_lastix(lastix, nrow(tp.tradesdf))
    lastix <= 0 && return tp

    closes = if !isnothing(closeprices)
        @assert length(closeprices) >= lastix "closeprices length=$(length(closeprices)) must cover lastix=$(lastix)"
        Float32[Float32(closeprices[ix]) for ix in 1:lastix]
    elseif length(tp.closeprices) >= lastix
        tp.closeprices[1:lastix]
    elseif :close in propertynames(tp.tradesdf)
        Float32[Float32(tp.tradesdf[ix, :close]) for ix in 1:lastix]
    else
        throw(ArgumentError("closeprices unavailable for TsTp pair=$(tp.pair); provide closeprices or seed tp.closeprices"))
    end

    # Keep a stable read view of labels/scores while mutating strategy limits row-by-row.
    rawlabels = TradeLabel[_normalizedtradelabel(tp.tradesdf[ix, :tradelabel]) for ix in 1:lastix]
    rawscores = Float32[Float32(tp.tradesdf[ix, :labelscore]) for ix in 1:lastix]

    local_minpd = Float32(minpricedelta)
    last_openix = 0
    for ix in 1:lastix
        algorithm(
            tp.tradesdf,
            ix,
            rawlabels[ix],
            rawscores[ix],
            closes[ix];
            openthreshold=Float32(openthreshold),
            buygain=buygain,
            sellgain=sellgain,
            limitreduction=limitreduction,
            minpricedelta=local_minpd,
        )
        if !isnothing(gaindf)
            last_openix = _materialize_gains_sample_from_trades!(gaindf, tp.tradesdf, tp.tradesdf, ix, last_openix; makerfee=makerfee)
        end
    end
    if (lastix > 0) && (:opentime in propertynames(tp.tradesdf))
        tp.last_update_dt = tp.tradesdf[lastix, :opentime]
    end
    return tp
end

function _materialize_gains_sample_from_trades!(result::DataFrame, tradesdf::DataFrame, predictionsdf::AbstractDataFrame, ix::Integer, last_openix::Int; makerfee::Real=0f0)::Int
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"
    @assert 1 <= ix <= nrow(predictionsdf) "ix=$(ix) out of bounds for predictions rows=$(nrow(predictionsdf))"

    time = tradesdf[ix, :opentime]
    high = predictionsdf[ix, :high]
    low = predictionsdf[ix, :low]
    label = _normalizedtradelabel(tradesdf[ix, :tradelabel])
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

"""
Synchronize one pair-scoped Trades DataFrame from prediction inputs and metadata.

- Receives a resultview DataFrame with columns: target, opentime, high, low, close, pivot, coin, rangeid, set, score, label
- Drops close and adds labelscore and tradelabel as copies of the already present columns score and labelto produce a Trades DataFrame.
- adds from metadata set, predicted, rangeid, openthreshold, closethreshold
- adds via Xch.settrade->Xch.tradesdf_contributors all Xch columns: (opentime which is already in), pair, 
    - longid, longstatus, longunfilled, longpriceavg, longmsgid, 
    - shortid, shortstatus, shortunfilled, shortpriceavg, shortmsgid
    - postype, posleverage, posamount, quoteprice, maintmargin, equity, balance, freemargin, freequota 
- adds close to TpTs as separate vector  
"""
function _synctradesframe!(ts::TsCache, xc::Xch.XchCache, base::AbstractString, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector;
    quotecoin::AbstractString=EnvConfig.pairquote,
    metadata::AbstractDict{Symbol, Any}=Dict{Symbol, Any}(),
    datetime::Union{Nothing, DateTime}=nothing,
)::TsTp
    n = size(predictionsdf, 1)
    @assert n == length(scores) == length(labels) "size(predictionsdf, 1)=$(n) must match scores=$(length(scores)) and labels=$(length(labels))"
    @assert :opentime in propertynames(predictionsdf) "predictionsdf must contain :opentime; names=$(names(predictionsdf))"
    @assert :close in propertynames(predictionsdf) "predictionsdf must contain :close; names=$(names(predictionsdf))"

    pair = tspairkey(base, quotecoin)
    tp = haspairstate(ts, pair) ? getpairstate!(ts, pair) : TsTp(pair=pair)

    rebuild = true
    if nrow(tp.tradesdf) == n && (:opentime in propertynames(tp.tradesdf))
        if n == 0
            rebuild = false
        else
            rebuild = (tp.tradesdf[1, :opentime] != predictionsdf[1, :opentime]) || (tp.tradesdf[n, :opentime] != predictionsdf[n, :opentime])
        end
    end

    if rebuild
        tradesdf = DataFrame(predictionsdf; copycols=true)
        closeprices = Float32.(tradesdf[!, :close])
        select!(tradesdf, Not(:close))
        Xch.settrades!(xc, base, quotecoin, tradesdf)
        tp = syncpairtrades!(ts, xc, base, quotecoin; datetime=datetime)
        tp.closeprices = closeprices
    else
        tp = syncpairtrades!(ts, xc, base, quotecoin; datetime=datetime)
        tp.closeprices = Float32.(predictionsdf[!, :close])
    end

    if n > 0
        fill!(tp.tradesdf[!, :lastopentrade], missing)
        fill!(tp.tradesdf[!, :longopenlimit], missing)
        fill!(tp.tradesdf[!, :longcloselimit], missing)
        fill!(tp.tradesdf[!, :shortopenlimit], missing)
        fill!(tp.tradesdf[!, :shortcloselimit], missing)
    end

    tp.tradesdf[!, :labelscore] = Float32.(scores)
    tp.tradesdf[!, :tradelabel] = Targets.TradeLabel[labels[ix] for ix in eachindex(labels)]

    for (k, v) in metadata
        tp.tradesdf[!, k] = fill(v, n)
    end
    return tp
end

"""Prepare one replay pair-scoped Trades DataFrame from prediction inputs."""
function preparereplaytrades!(ts::TsCache, xc::Xch.XchCache, base::AbstractString, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector;
    quotecoin::AbstractString=EnvConfig.pairquote,
    metadata::AbstractDict{Symbol, Any}=Dict{Symbol, Any}(),
    datetime::Union{Nothing, DateTime}=nothing,
)::TsTp
    return _synctradesframe!(ts, xc, base, predictionsdf, scores, labels;
        quotecoin=quotecoin,
        metadata=metadata,
        datetime=datetime,
    )
end

function _validatereplayprepared!(tp::TsTp, lastix::Integer)::Nothing
    @assert !isempty(tp.pair) "Replay pair identifier must be non-empty"

    required_cols = (
        :opentime,
        :high,
        :low,
        :tradelabel,
        :labelscore,
        :lastopentrade,
        :longopenlimit,
        :longcloselimit,
        :shortopenlimit,
        :shortcloselimit,
    )
    missing_cols = Symbol[col for col in required_cols if !(col in propertynames(tp.tradesdf))]
    @assert isempty(missing_cols) "Replay state for pair=$(tp.pair) is not prepared; missing columns=$(missing_cols)"

    if lastix > 0
        @assert length(tp.closeprices) >= lastix "Replay state for pair=$(tp.pair) is not prepared with closeprices covering lastix=$(lastix); closeprices length=$(length(tp.closeprices))"
        if :pair in propertynames(tp.tradesdf)
            rowpair = uppercase(String(tp.tradesdf[1, :pair]))
            @assert rowpair == uppercase(tp.pair) "Replay state pair mismatch for pair=$(tp.pair): tradesdf pair=$(rowpair)"
        end
    end
    return nothing
end

"""Process gains for one replay pair after its Trades DataFrame has been prepared explicitly."""
function processreplaygains!(tp::TsTp;
    strategy::StrategyConfig,
    lastix::Integer=nrow(tp.tradesdf),
    forcegain::Bool=true,
    openthreshold=0.6f0,
    closethreshold=0.1f0,
)::DataFrame
    _ = forcegain
    _ = closethreshold
    _validate_lastix(lastix, nrow(tp.tradesdf))
    _validatereplayprepared!(tp, lastix)
    gaindf = emptygaindf()

    if lastix > 0
        minpd = strategy.minpricedelta
        try
            simulate_gains!(tp, lastix;
                algorithm=strategy.algorithm,
                openthreshold=Float32(openthreshold),
                buygain=strategy.buygain,
                sellgain=strategy.sellgain,
                limitreduction=strategy.limitreduction,
                minpricedelta=minpd,
                gaindf=gaindf,
                makerfee=strategy.makerfee,
                takerfee=strategy.takerfee,
            )
        catch err
            if err isa MethodError
                throw(ArgumentError("strategy algorithm $(strategy.algorithm) does not support required signature in replay gain processing. Expected call shape: algorithm(tradesdf::DataFrame, ix::Integer, label::TradeLabel, score::Real, close::Real; openthreshold, buygain, sellgain, limitreduction, minpricedelta)."))
            end
            rethrow(err)
        end
    end

    return gaindf
end

include("tradingstrategyconfig.jl")

end # module
