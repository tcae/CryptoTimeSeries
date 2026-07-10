"""
Minimal TradingStrategy module retained for Trade integration and TrendDetector workflows.

This module intentionally keeps only the API surface required by:
- Trade/src/Trade.jl
- scripts/TrendDetector.jl
- scripts/tradereal.jl
- scripts/tradesim.jl
"""
module TradingStrategy

using DataFrames, Dates, CategoricalArrays
using EnvConfig, Features, Targets, Classify, Xch, Ohlcv

"""Ensure Trades column `label` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: TradingStrategy writes enum labels; Xch consumes them to map open/close actions."""
function tradesdf_label(tradesdf::DataFrame)::DataFrame
    if :label ∉ propertynames(tradesdf)
        tradesdf[!, :label] = fill(ignore, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `score` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: Strategy confidence/score of trade label."""
function tradesdf_score(tradesdf::DataFrame)::DataFrame
    if :score ∉ propertynames(tradesdf)
        tradesdf[!, :score] = zeros(Float32, nrow(tradesdf))
    end
    return tradesdf
end


"""Ensure Trades column `lo_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (long-open limit) consumed by Xch as requested limit per action."""
function tradesdf_lo_limit(tradesdf::DataFrame)::DataFrame
    if :lo_limit ∉ propertynames(tradesdf)
        tradesdf[!, :lo_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `lc_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (long-close limit) consumed by Xch as requested limit per action."""
function tradesdf_lc_limit(tradesdf::DataFrame)::DataFrame
    if :lc_limit ∉ propertynames(tradesdf)
        tradesdf[!, :lc_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `so_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (short-open limit) consumed by Xch as requested limit per action."""
function tradesdf_so_limit(tradesdf::DataFrame)::DataFrame
    if :so_limit ∉ propertynames(tradesdf)
        tradesdf[!, :so_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `sc_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (short-close limit) consumed by Xch as requested limit per action."""
function tradesdf_sc_limit(tradesdf::DataFrame)::DataFrame
    if :sc_limit ∉ propertynames(tradesdf)
        tradesdf[!, :sc_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Return TradingStrategy-contributed Trades schema initializer functions. Note: Trade contributes lo_amount/lc_amount/so_amount/sc_amount columns via Trade.tradesdf_contributors()."""
function tradesdf_contributors()::Vector{Function}
    return Function[
        tradesdf_label,
        tradesdf_score,
        tradesdf_lo_limit,
        tradesdf_lc_limit,
        tradesdf_so_limit,
        tradesdf_sc_limit,
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
    if :pair in propertynames(tradedf)
        pair = tradedf[ix, :pair]
        if !ismissing(pair)
            bq = Xch.basequote(String(pair))
            return uppercase(String(bq.basecoin))
        end
    end
    return nothing
end

"""Persist a trades dataframe into config-scoped storage, plus optional aggregate copy."""
function savetrades(tradedf::AbstractDataFrame; stem::AbstractString="gains", include_aggregate::Bool=true)
    if size(tradedf, 1) == 0
        return String[]
    end
    @assert :pair in propertynames(tradedf) "tradedf must contain a :pair column; names=$(names(tradedf))"

    paths = String[]
    basekeys = [_tradebasekey(tradedf, ix) for ix in 1:nrow(tradedf)]
    @assert all(!isnothing, basekeys) "tradedf must provide a resolvable base via :pair for every row; names=$(names(tradedf))"
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

" true if candidate > 0 and either current <= 0 or relative price delta exceeds minpricedelta"
@inline function _should_update_price(current::Real, candidate::Real, minpricedelta::Float32)
    if candidate <= 0f0
        return false
    end
    if current <= 0f0 || minpricedelta <= 0f0
        return true
    end
    return _relpricedelta(candidate, current) > minpricedelta
end

"""
Immutable strategy configuration payload for runtime strategy execution.
"""
Base.@kwdef struct StrategyConfig
    classifier::Union{Nothing, Classify.AbstractClassifier} = nothing
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

"""Per-trading-pair runtime state holder used by `TsCache`.

OHLCV prices (`close`, `high`, `low`) are stored directly in `tradesdf` as Xch-owned
columns and are not duplicated here."""
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
mutable struct TsCache
    pairs::Dict{String, TsTp}
    classifier_gate_state::Dict{String, NamedTuple{(:last_advice, :last_classify_close), Tuple{Any, Float32}}}
    accepted::Set{String}
    cfg::StrategyConfig
    source::String
end

@inline function _strategy_with_classifier(spec::StrategyConfig, classifier::Classify.AbstractClassifier)::StrategyConfig
    return StrategyConfig(
        classifier=classifier,
        algorithm=spec.algorithm,
        maxwindow=spec.maxwindow,
        openthreshold=spec.openthreshold,
        closethreshold=spec.closethreshold,
        makerfee=spec.makerfee,
        takerfee=spec.takerfee,
        buygain=spec.buygain,
        sellgain=spec.sellgain,
        limitreduction=spec.limitreduction,
        minpricedelta=spec.minpricedelta,
        max_classify_staleness_minutes=spec.max_classify_staleness_minutes,
    )
end

@inline function _strategyclassifier(rt::TsCache)::Classify.AbstractClassifier
    classifier = rt.cfg.classifier
    @assert !isnothing(classifier) "StrategyConfig.classifier must be configured for TsCache runtime"
    return classifier
end

"Build TsCache with explicit classifier wiring from argument or strategy config."
function TsCache(; classifier::Union{Nothing, Classify.AbstractClassifier}=nothing, strategy::Any=nothing, source::AbstractString="manual")
    raw_template = isnothing(strategy) ? StrategyConfig() : strategy
    resolved_template = raw_template isa StrategyConfig ? raw_template : throw(ArgumentError("strategy template must be TradingStrategy.StrategyConfig, got $(typeof(raw_template))"))
    resolved_classifier = !isnothing(classifier) ? classifier : resolved_template.classifier
    !isnothing(resolved_classifier) || throw(ArgumentError("TsCache requires a classifier via classifier keyword or strategy.classifier"))
    configured_strategy = _strategy_with_classifier(resolved_template, resolved_classifier)
    return TsCache(Dict{String, TsTp}(), Dict{String, NamedTuple{(:last_advice, :last_classify_close), Tuple{Any, Float32}}}(), Set{String}(), configured_strategy, String(source))
end

"Build TsCache from a TrendDetector config reference, loading and compiling the strategy under the hood."
function TsCache(configref::AbstractString; mnemonic::AbstractString="mix", mode=EnvConfig.configmode, source::AbstractString="manual")
    strategy = strategyconfig(configref; mnemonic=mnemonic, mode=mode)
    return TsCache(strategy=strategy, source=source)
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

"Return the classifier history requirement in minutes for runtime compatibility callers."
function requiredhistoryminutes(rt::TsCache)::Int
    return Int(max(0, Classify.requiredminutes(_strategyclassifier(rt))))
end

"Drop one base from TsCache, including classifier and cached pair state."
function dropbase!(rt::TsCache, base::AbstractString)::Nothing
    basekey = uppercase(String(base))
    classifier = _strategyclassifier(rt)
    try
        Classify.removebase!(classifier, basekey)
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
    classifier = _strategyclassifier(rt)
    try
        Classify.removebase!(classifier, nothing)
    catch
    end
    return nothing
end

"Apply a strategy-spec template to TsCache and clear derived cached state."
function apply_strategy!(rt::TsCache, strategy::StrategyConfig; source::AbstractString="manual")::Nothing
    classifier = !isnothing(strategy.classifier) ? strategy.classifier : _strategyclassifier(rt)
    rt.cfg = _strategy_with_classifier(strategy, classifier)
    rt.source = String(source)
    empty!(rt.pairs)
    empty!(rt.classifier_gate_state)
    empty!(rt.accepted)
    try
        Classify.removebase!(classifier, nothing)
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

"Return the earliest usable start date for one base, or `nothing` when the base is not acceptable for the current runtime."
function acceptbase!(rt::TsCache, xc::Xch.XchCache, base::AbstractString; datetime::DateTime, updatecache::Bool=false)::Union{Nothing, DateTime}
    basekey = uppercase(String(base))
    haskey(xc.bases, basekey) || return nothing

    ohlcv = Xch.getohlcv(xc, basekey)
    odf = Ohlcv.dataframe(ohlcv)
    rowcount = size(odf, 1)
    rowcount > 0 || return nothing

    required_minutes = requiredhistoryminutes(rt)
    startdt = datetime - Minute(required_minutes)
    if odf[1, :opentime] > startdt
        return nothing
    end

    classifier = _strategyclassifier(rt)
    loaded = Set{String}(uppercase.(String.(Classify.bases(classifier))))
    if !(basekey in loaded)
        Classify.addbase!(classifier, ohlcv)
        push!(loaded, basekey)
    end

    Classify.supplement!(classifier)
    updatecache && Classify.writetargetsfeatures(classifier)

    accepted = Set{String}(uppercase.(String.(Classify.bases(classifier))))
    if !(basekey in accepted)
        dropbase!(rt, basekey)
        return nothing
    end

    push!(rt.accepted, basekey)
    syncpairtrades!(rt, xc, basekey, EnvConfig.pairquote; datetime=datetime)
    return startdt
end

"Prepare TsCache for requested bases using available OHLCV data and update accepted set."
function preparebases!(rt::TsCache, xc::Xch.XchCache, bases::AbstractVector{<:AbstractString}; datetime::DateTime, updatecache::Bool=false)::Nothing
    wanted = Set{String}(uppercase.(String.(bases)))
    classifier = _strategyclassifier(rt)

    loaded = Set{String}(uppercase.(String.(Classify.bases(classifier))))
    for stale in sort!(collect(setdiff(union(rt.accepted, loaded), wanted)))
        dropbase!(rt, stale)
    end

    accepted = Set{String}()
    for base in sort!(collect(wanted))
        startdt = acceptbase!(rt, xc, base; datetime=datetime, updatecache=updatecache)
        isnothing(startdt) && continue
        push!(accepted, base)
    end

    rt.accepted = accepted
    for base in sort!(collect(rt.accepted))
        syncpairtrades!(rt, xc, base, EnvConfig.pairquote; datetime=datetime)
    end
    return nothing
end

"Update one base row in the Xch-owned trades dataframe using TsCache runtime state."
function gettradesrow!(rt::TsCache, xc::Xch.XchCache, base::AbstractString, datetime::DateTime; reconciliation=nothing)::Union{Nothing, NamedTuple}
    basekey = uppercase(String(base))
    if !haskey(xc.bases, basekey)
        (EnvConfig.verbosity >= 1) && @warn "base OHLCV unavailable in exchange cache; skipping gettradesrow!" base=basekey
        return nothing
    end

    startdt = acceptbase!(rt, xc, basekey; datetime=datetime, updatecache=false)
    isnothing(startdt) && return nothing

    ohlcv = Xch.getohlcv(xc, basekey)
    spec = rt.cfg

    rowix = ohlcv.ix
    odf = Ohlcv.dataframe(ohlcv)
    @assert (1 <= rowix <= size(odf, 1)) "rowix=$(rowix) out of bounds for ohlcv rows=$(size(odf, 1))"
    closeprice = odf[rowix, :close]
    opentime = odf[rowix, :opentime]

    _ = reconciliation

    syncpairtrades!(rt, xc, basekey, EnvConfig.pairquote; datetime=datetime)
    row = Xch.ensuretradesrow!(xc, basekey, EnvConfig.pairquote, opentime)
    tdf = row.tradesdf
    trowix = Int(row.rowix)

    # Classification is performed in gain_limit_reversal! when row score is zero.
    tdf[trowix, :label] = ignore
    tdf[trowix, :score] = 0f0
    tdf[trowix, :close] = closeprice

    spec.algorithm(spec, tdf, trowix)

    return (
        base=basekey,
        datetime=datetime,
        tradesdf=tdf,
        rowix=trowix,
        probability=Float32(tdf[trowix, :score]),
        configid=0,
        source=:tradingstrategy,
    )
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

@inline _hasguidance(openlimit, closelimit) = (openlimit > 0) && (closelimit > 0)
@inline _openlimitactive(openlimit) = openlimit > 0

" number of minutes exceeding cfg.maxwindow since lastopentrade; 0 if within maxwindow or no lastopentrade"
function _limitreductionminutes(cfg::StrategyConfig, tradesdf::DataFrame, ix::Integer)
    if (tradesdf[ix, :label] in (longopen, longstrongopen, shortopen, shortstrongopen)) || ismissing(tradesdf[ix, :lastopentrade])
        return 0
    else
        elapsed_minutes = Int(div(Dates.value(DateTime(tradesdf[ix, :opentime]) - DateTime(tradesdf[ix, :lastopentrade])), 60000)) # DateTime difference is in milliseconds
        return elapsed_minutes - Int(cfg.maxwindow)
    end
end

" closeprice relative to reference price, reduced by limitreductionminutes * cfg.limitreduction"
function _closeprice(cfg::StrategyConfig, limitreductionminutes::Int, refprice::Float32, updown::Targets.TrendPhase)
    closelimit = 0f0
    if updown == up
        closelimit = refprice * (1f0 + Float32(cfg.sellgain))
    elseif updown == down
        closelimit = refprice * (1f0 - Float32(cfg.sellgain))
    end
    if limitreductionminutes <= 0
        return closelimit
    end
    reduction_factor = cfg.limitreduction * limitreductionminutes
    if updown == up
        return closelimit * (1f0 - reduction_factor)
    elseif updown == down
        return closelimit * (1f0 + reduction_factor)
    else
        return closelimit
    end
end

function _get_classifier_result!(cfg::StrategyConfig, tradesdf::DataFrame, ix::Integer)
    classifier = cfg.classifier
    @assert !isnothing(classifier) "StrategyConfig.classifier must be configured for classifier fallback at ix=$(ix)"
    @assert :pair in propertynames(tradesdf) "tradesdf must contain :pair for classifier fallback; names=$(names(tradesdf))"
    @assert :opentime in propertynames(tradesdf) "tradesdf must contain :opentime for classifier fallback; names=$(names(tradesdf))"

    pair = tradesdf[ix, :pair]
    @assert !ismissing(pair) "tradesdf[ix=$ix, :pair] must be non-missing for classifier fallback"
    bq = Xch.basequote(String(pair))
    basekey = uppercase(String(bq.basecoin))
    datetime = DateTime(tradesdf[ix, :opentime])

    advice = Classify.advice(classifier, basekey, datetime, investment=nothing)
    if isnothing(advice)
        tradesdf[ix, :label] = ignore
        tradesdf[ix, :score] = 0f0
        return nothing
    end

    tradesdf[ix, :label] = advice.tradelabel
    tradesdf[ix, :score] = advice.probability
    return advice
end

"""
    gain_limit_reversal!(strategy, tradesdf, ix)

DataFrame-based limit-reversal lane update that writes one sample row state.
"""
function gain_limit_reversal!(cfg::StrategyConfig, tradesdf::DataFrame, ix::Integer)
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"
    if tradesdf[ix, :score] == 0f0
        _get_classifier_result!(cfg, tradesdf, ix)
    end

    tradesdf[ix, :lo_limit] = ix > 1 ? tradesdf[ix-1, :lo_limit] : 0f0
    tradesdf[ix, :lc_limit] = ix > 1 ? tradesdf[ix-1, :lc_limit] : 0f0
    tradesdf[ix, :so_limit] = ix > 1 ? tradesdf[ix-1, :so_limit] : 0f0
    tradesdf[ix, :sc_limit] = ix > 1 ? tradesdf[ix-1, :sc_limit] : 0f0

    if (tradesdf[ix, :label] in (longopen, longstrongopen)) 
        if (tradesdf[ix, :score] >= cfg.openthreshold)
            if _should_update_price(tradesdf[ix, :lo_limit], tradesdf[ix, :close] * (1f0 - Float32(cfg.buygain)), cfg.minpricedelta)
                tradesdf[ix, :lo_limit] = tradesdf[ix, :close] * (1f0 - Float32(cfg.buygain))
                tradesdf[ix, :lc_limit] = _closeprice(cfg, 0, tradesdf[ix, :close], up)
            elseif _should_update_price(tradesdf[ix, :lc_limit], tradesdf[ix, :lo_pavg] * (1f0 + Float32(cfg.sellgain)), cfg.minpricedelta)
                # refresh lc_limit in case it was reduced
                tradesdf[ix, :lc_limit] = _closeprice(cfg, 0, tradesdf[ix, :close], up)
            end
            tradesdf[ix, :so_limit] = 0f0
            tradesdf[ix, :sc_limit] = tradesdf[ix, :sp_amount] > 0f0 ? tradesdf[ix, :lo_limit] : 0f0
        else # label below threshold
            tradesdf[ix, :label] = longhold
        end
    elseif (tradesdf[ix, :label] in (shortopen, shortstrongopen)) 
        if (tradesdf[ix, :score] >= cfg.openthreshold)
            if _should_update_price(tradesdf[ix, :so_limit], tradesdf[ix, :close] * (1f0 - Float32(cfg.buygain)), cfg.minpricedelta)
                tradesdf[ix, :so_limit] = tradesdf[ix, :close] * (1f0 + Float32(cfg.buygain))
                tradesdf[ix, :sc_limit] = _closeprice(cfg, 0, tradesdf[ix, :close], down)
            elseif _should_update_price(tradesdf[ix, :sc_limit], tradesdf[ix, :so_pavg] * (1f0 - Float32(cfg.sellgain)), cfg.minpricedelta)
                # refresh sc_limit in case it was reduced
                tradesdf[ix, :sc_limit] = _closeprice(cfg, 0, tradesdf[ix, :close], down)
            end
            tradesdf[ix, :lo_limit] = 0f0
            tradesdf[ix, :lc_limit] = tradesdf[ix, :lp_amount] > 0f0 ? tradesdf[ix, :so_limit] : 0f0
        else # label below threshold
            tradesdf[ix, :label] = shorthold
        end
    else
        lrm = _limitreductionminutes(cfg, tradesdf, ix)
        if lrm > 0
            if (tradesdf[ix, :lc_limit] > 0f0) && (tradesdf[ix, :lp_amount] > 0f0)
                lc_candidate = _closeprice(cfg, lrm, tradesdf[ix, :lo_pavg], up)
                if _should_update_price(tradesdf[ix, :lc_limit], lc_candidate, cfg.minpricedelta)
                    tradesdf[ix, :lc_limit] = lc_candidate
                end
                coupled_short_reversal = (tradesdf[ix, :lp_amount] > 0f0) && (tradesdf[ix, :so_amount] > 0f0) && (tradesdf[ix, :so_limit] > 0f0)
                if coupled_short_reversal
                    # Preserve close-before-open ordering for long->short reversal while still allowing reduction.
                    tradesdf[ix, :lc_limit] = min(tradesdf[ix, :lc_limit], tradesdf[ix, :so_limit])
                end
            # else lc_limit is waiting for lo_limit to fill
            end
            if (tradesdf[ix, :sc_limit] > 0f0) && (tradesdf[ix, :sp_amount] > 0f0)
                sc_candidate = _closeprice(cfg, lrm, tradesdf[ix, :so_pavg], down)
                if _should_update_price(tradesdf[ix, :sc_limit], sc_candidate, cfg.minpricedelta)
                    tradesdf[ix, :sc_limit] = sc_candidate
                end
                coupled_long_reversal = (tradesdf[ix, :sp_amount] > 0f0) && (tradesdf[ix, :lo_amount] > 0f0) && (tradesdf[ix, :lo_limit] > 0f0)
                if coupled_long_reversal
                    # Symmetric ordering guard for short->long reversal.
                    tradesdf[ix, :sc_limit] = max(tradesdf[ix, :sc_limit], tradesdf[ix, :lo_limit])
                end
            # else sc_limit is waiting for so_limit to fill
            end
        end
    end

    return
end

function _resetorder(tradesdf::DataFrame, ix::Integer, ordertype::String; reset_pavg::Bool)
    function _ro(tradesdf::DataFrame, ix::Integer, ordertype::String)
        tradesdf[ix, Symbol(ordertype * "_limit")] = 0f0
        tradesdf[ix, Symbol(ordertype * "_amount")] = 0f0
        tradesdf[ix, Symbol(ordertype * "_status")] = "closed"
        tradesdf[ix, Symbol(ordertype * "_filled")] = 0f0
        tradesdf[ix, Symbol(ordertype * "_id")] = "none"
        if reset_pavg
            tradesdf[ix, Symbol(ordertype * "_pavg")] = 0f0
        end
    end

    _ro(tradesdf, ix, ordertype)
end

function _executed_open_hit_dt(tradesdf::DataFrame, ix::Integer)
    label = tradesdf[ix, :label]
    lo_limit = tradesdf[ix, :lo_limit]
    lo_amount = tradesdf[ix, :lo_amount]
    so_limit = tradesdf[ix, :so_limit]
    so_amount = tradesdf[ix, :so_amount]
    high = tradesdf[ix, :high]
    low = tradesdf[ix, :low]
    long_open_hit = islongopenlabel(label) && (lo_amount > 0f0) && _openlimitactive(lo_limit) && _price_in_bar(Float32(lo_limit), low, high, :low)
    short_open_hit = isshortopenlabel(label) && (so_amount > 0f0) && _openlimitactive(so_limit) && _price_in_bar(Float32(so_limit), low, high, :high)
    @assert !(long_open_hit && short_open_hit) "Both long and short open limits matched same bar at ix=$(ix): lo=$(lo_limit), so=$(so_limit), low=$(low), high=$(high)"
    if long_open_hit
        @assert tradesdf[ix, :sp_amount] == 0f0 "Long open hit at ix=$(ix) but sp_amount=$(tradesdf[ix, :sp_amount]) is not zero"
        tradesdf[ix, :lastopentrade] = tradesdf[ix, :opentime]
        tradesdf[ix, :lp_amount] = tradesdf[ix, :lo_amount]
        tradesdf[ix, :lo_pavg] = tradesdf[ix, :lo_limit]
        _resetorder(tradesdf, ix, "lo", reset_pavg=false)
        tradesdf[ix, :lc_amount] = tradesdf[ix, :lp_amount]
        tradesdf[ix, :lc_pavg] = 0f0
        tradesdf[ix, :lc_filled] = 0f0
        tradesdf[ix, :lc_status] = "submitted"
    elseif short_open_hit
        @assert tradesdf[ix, :lp_amount] == 0f0 "Short open hit at ix=$(ix) but lp_amount=$(tradesdf[ix, :lp_amount]) is not zero"
        tradesdf[ix, :lastopentrade] = tradesdf[ix, :opentime]
        tradesdf[ix, :sp_amount] = tradesdf[ix, :so_amount]
        tradesdf[ix, :so_pavg] = tradesdf[ix, :so_limit]
        _resetorder(tradesdf, ix, "so", reset_pavg=false)
        tradesdf[ix, :sc_amount] = tradesdf[ix, :sp_amount]
        tradesdf[ix, :sc_pavg] = 0f0
        tradesdf[ix, :sc_filled] = 0f0
        tradesdf[ix, :sc_status] = "submitted"
    end
    return (long_open_hit || short_open_hit) ? tradesdf[ix, :opentime] : missing
end

function _rowtakeover!(tdf::DataFrame, ix::Integer)
    if ix > 1
        if tdf[ix, :score] == 0f0
            tdf[ix, :score] = tdf[ix-1, :score]
            tdf[ix, :label] = tdf[ix-1, :label]
        end
        for col in [:lo_limit, :lc_limit, :so_limit, :sc_limit, 
                    :lo_amount, :lc_amount, :so_amount, :sc_amount, 
                    :lo_status, :lc_status, :so_status, :sc_status, 
                    :lo_filled, :lc_filled, :so_filled, :sc_filled, 
                    :lo_id, :lc_id, :so_id, :sc_id, 
                    :lo_pavg, :lc_pavg, :so_pavg, :sc_pavg, 
                    :lastopentrade, :lp_amount, :sp_amount]
            tdf[ix, col] = tdf[ix-1, col]
        end
    end
end

"""Execute per-sample strategy updates and gain materialization outside the algorithm implementation."""
function simulate_gains!(cfg::StrategyConfig, tp::TsTp, lastix::Integer, gaindf::DataFrame=emptygaindf())
    lastix <= 0 && return tp
    @assert !isnothing(gaindf) "expeting gaindf to be a DataFrame when gain materialization is requested"

    # Read-only views of labels/scores — each rawlabels[ix]/rawscores[ix] is read before the algorithm mutates row ix.
    rawlabels = @view tp.tradesdf[!, :label][1:lastix]
    rawscores = @view tp.tradesdf[!, :score][1:lastix]

    last_openix = 0
    for ix in 1:lastix
        try
            _rowtakeover!(tp.tradesdf, ix)
            last_openix = _materialize_gains_sample_from_trades!(gaindf, tp.tradesdf, ix, last_openix; makerfee=cfg.makerfee, lastix=lastix)
            # first close (=materialize) then open if both are applicable shall be also enforced in real by Xch
            last_openix = ismissing(_executed_open_hit_dt(tp.tradesdf, ix)) ? last_openix : ix
            # algorithm after materialization because label and order limits are set on the basis of the last minute (ix is the last complete minute sample in the past) and prices can hit these limits from the next ix+1 minute
            cfg.algorithm(cfg, tp.tradesdf, ix)
            _process_advice_row!(cfg, tp.tradesdf, ix)
            _validate_row_consistency(tp.tradesdf, ix)
        catch err
            if err isa AssertionError
                lo = max(1, ix - 1)
                println(stderr, "Assertion in simulate_gains! at ix=$(ix). Last tradesdf rows $(lo):$(ix):")
                show(stderr, MIME("text/plain"), tp.tradesdf[lo:ix, :]; allrows=true, allcols=true)
                println(stderr)
            end
            rethrow(err)
        end
    end

    if (lastix > 0)
        tp.last_update_dt = tp.tradesdf[lastix, :opentime]
    end
    return tp
end

"Input is :label, :score, :lo_limit, :lc_limit, :so_limit, :sc_limit. Output is :lo_status, :so_status, :lo_amount, :so_amount."
function _process_advice_row!(strategy::StrategyConfig, tradesdf::DataFrame, ix::Integer)
    if islongopenlabel(tradesdf[ix, :label]) && (tradesdf[ix, :lp_amount] == 0f0)
        tradesdf[ix, :lo_status] = "submitted"
        tradesdf[ix, :lo_amount] = 100f0
        tradesdf[ix, :lo_pavg] = 0f0
        tradesdf[ix, :lo_filled] = 0f0
    end
    if isshortopenlabel(tradesdf[ix, :label]) && (tradesdf[ix, :sp_amount] == 0f0)
        tradesdf[ix, :so_status] = "submitted"
        tradesdf[ix, :so_amount] = 100f0
        tradesdf[ix, :so_pavg] = 0f0
        tradesdf[ix, :so_filled] = 0f0
    end
end

function _validate_row_consistency(tradesdf::DataFrame, ix::Integer)::Nothing
    @assert !((tradesdf[ix, :lp_amount] > 0f0) && (tradesdf[ix, :sp_amount] > 0f0)) "Invalid overlap at ix=$(ix): lp_amount=$(tradesdf[ix, :lp_amount]), sp_amount=$(tradesdf[ix, :sp_amount]). A row cannot hold long and short position amounts at the same time."

    if islongopenlabel(tradesdf[ix, :label])
        @assert _hasguidance(tradesdf[ix, :lo_limit], tradesdf[ix, :lc_limit]) "Missing long guidance for long open signal at ix=$(ix): lo=$(tradesdf[ix, :lo_limit]), lc=$(tradesdf[ix, :lc_limit])"
        @assert tradesdf[ix, :so_limit] == 0f0 "Expected zero so_limit for long open orders at ix=$(ix):, so_limit=$(tradesdf[ix, :so_limit])"
    end
    if isshortopenlabel(tradesdf[ix, :label])
        @assert _hasguidance(tradesdf[ix, :so_limit], tradesdf[ix, :sc_limit]) "Missing short guidance for short open signal at ix=$(ix): so=$(tradesdf[ix, :so_limit]), sc=$(tradesdf[ix, :sc_limit])"
        @assert tradesdf[ix, :lo_limit] == 0f0 "Expected zero lo_limit for short open orders at ix=$(ix):, lo_limit=$(tradesdf[ix, :lo_limit])"
    end
    @assert (ismissing(tradesdf[ix, :lastopentrade])  || (tradesdf[1, :opentime] <= tradesdf[ix, :lastopentrade] <= tradesdf[ix, :opentime])) "$(tradesdf[1, :opentime]) <= lastopentrade[ix=$ix]=$(tradesdf[ix, :lastopentrade]) <= $(tradesdf[ix, :opentime])"
    if (tradesdf[ix, :lp_amount] > 0f0)
        @assert tradesdf[ix, :sp_amount] == 0f0 "Expected zero sp_amount for long positions at ix=$(ix): lp_amount=$(tradesdf[ix, :lp_amount]), sp_amount=$(tradesdf[ix, :sp_amount])"
        @assert tradesdf[ix, :lo_pavg] > 0f0 "Expected positive lo_pavg for long positions at ix=$(ix): lp_amount=$(tradesdf[ix, :lp_amount]), lo_pavg=$(tradesdf[ix, :lo_pavg])"
        @assert tradesdf[ix, :lc_limit] > 0f0 "Expected positive lc_limit for long positions at ix=$(ix): lp_amount=$(tradesdf[ix, :lp_amount]), lc_limit=$(tradesdf[ix, :lc_limit])"
        @assert !ismissing(tradesdf[ix, :lastopentrade]) "Expected non-missing lastopentrade for long positions at ix=$(ix): lp_amount=$(tradesdf[ix, :lp_amount]), lastopentrade=$(tradesdf[ix, :lastopentrade])"
    elseif (tradesdf[ix, :sp_amount] > 0f0)
        @assert tradesdf[ix, :lp_amount] == 0f0 "Expected zero lp_amount for short positions at ix=$(ix): lp_amount=$(tradesdf[ix, :lp_amount]), sp_amount=$(tradesdf[ix, :sp_amount])"
        @assert tradesdf[ix, :so_pavg] > 0f0 "Expected positive so_pavg for short positions at ix=$(ix): sp_amount=$(tradesdf[ix, :sp_amount]), so_pavg=$(tradesdf[ix, :so_pavg])"
        @assert tradesdf[ix, :sc_limit] > 0f0 "Expected positive sc_limit for short positions at ix=$(ix): sp_amount=$(tradesdf[ix, :sp_amount]), sc_limit=$(tradesdf[ix, :sc_limit])"
        @assert !ismissing(tradesdf[ix, :lastopentrade]) "Expected non-missing lastopentrade for short positions at ix=$(ix): sp_amount=$(tradesdf[ix, :sp_amount]), lastopentrade=$(tradesdf[ix, :lastopentrade])"
    else
        @assert tradesdf[ix, :lp_amount] == 0f0 "Expected zero lp_amount for flat positions at ix=$(ix): lp_amount=$(tradesdf[ix, :lp_amount])"
        @assert tradesdf[ix, :sp_amount] == 0f0 "Expected zero sp_amount for flat positions at ix=$(ix): sp_amount=$(tradesdf[ix, :sp_amount])"
        @assert tradesdf[ix, :so_pavg] == 0f0 "Expected zero so_pavg for flat positions at ix=$(ix): so_pavg=$(tradesdf[ix, :so_pavg])"
        @assert tradesdf[ix, :lo_pavg] == 0f0 "Expected zero lo_pavg for flat positions at ix=$(ix): lo_pavg=$(tradesdf[ix, :lo_pavg])"
        @assert ismissing(tradesdf[ix, :lastopentrade]) "Expected missing lastopentrade for flat positions at ix=$(ix): lastopentrade=$(tradesdf[ix, :lastopentrade])"
    end
    return nothing
end

function _materialize_gains_sample_from_trades!(result::Union{Nothing, DataFrame}, tradesdf::DataFrame, ix::Integer, last_openix::Int; makerfee::Float32=0f0, lastix::Integer=0)::Int

    if tradesdf[ix, :lp_amount] > 0f0
        openprice = tradesdf[last_openix, :lo_limit]
        minutes = Int(div(Dates.value(tradesdf[ix, :opentime] - tradesdf[ix, :lastopentrade]), 60000)) + 1
        if _price_in_bar(tradesdf[ix, :lc_limit], tradesdf[ix, :low], tradesdf[ix, :high], :high)
            gain = (tradesdf[ix, :lc_limit] - openprice) / openprice
            push!(result, (up, (ix - last_openix + 1), minutes, gain, (gain - 2f0 * makerfee), tradesdf[ix, :lastopentrade], tradesdf[ix, :opentime], last_openix, ix))
            tradesdf[ix, :lc_pavg] = tradesdf[ix, :lc_limit]
            _resetorder(tradesdf, ix, "lc", reset_pavg=false)
            last_openix = 0
        elseif (ix == lastix) && (last_openix > 0)
            # Force-close any open gain segment at range boundary using close price at lastix
            gain = (tradesdf[lastix, :close] - openprice) / openprice
            push!(result, (up, (ix - last_openix + 1), minutes, gain, (gain - 2f0 * makerfee), tradesdf[lastix, :lastopentrade], tradesdf[lastix, :opentime], last_openix, lastix))
            tradesdf[ix, :lc_pavg] = tradesdf[ix, :close]
            _resetorder(tradesdf, ix, "lc", reset_pavg=false)
            last_openix = 0
        end
    elseif tradesdf[ix, :sp_amount] > 0f0
        openprice = tradesdf[last_openix, :so_limit]
        minutes = Int(div(Dates.value(tradesdf[ix, :opentime] - tradesdf[ix, :lastopentrade]), 60000)) + 1
        if _price_in_bar(tradesdf[ix, :sc_limit], tradesdf[ix, :low], tradesdf[ix, :high], :low)
            gain = -(tradesdf[ix, :sc_limit] - openprice) / openprice
            push!(result, (down, (ix - last_openix + 1), minutes, gain, (gain - 2f0 * makerfee), tradesdf[ix, :lastopentrade], tradesdf[ix, :opentime], last_openix, ix))
            tradesdf[ix, :sc_pavg] = tradesdf[ix, :sc_limit]
            _resetorder(tradesdf, ix, "sc", reset_pavg=false)
            last_openix = 0
        elseif (ix == lastix) && (last_openix > 0)
            # Force-close any open gain segment at range boundary using close price at lastix
            gain = -(tradesdf[lastix, :close] - openprice) / openprice
            push!(result, (down, (ix - last_openix + 1), minutes, gain, (gain - 2f0 * makerfee), tradesdf[lastix, :lastopentrade], tradesdf[lastix, :opentime], last_openix, lastix))
            tradesdf[ix, :sc_pavg] = tradesdf[ix, :close]
            _resetorder(tradesdf, ix, "sc", reset_pavg=false)
            last_openix = 0
        end
    end
    if last_openix == 0
        tradesdf[ix, :lastopentrade] = missing
        tradesdf[ix, :lp_amount] = 0f0
        tradesdf[ix, :sp_amount] = 0f0
        tradesdf[ix, :lo_pavg] = 0f0
        tradesdf[ix, :so_pavg] = 0f0
    end

    return last_openix
end

"""Prepare one replay pair-scoped Trades DataFrame from prediction inputs.

- Receives a resultview DataFrame with columns: target, opentime, high, low, close, pivot, coin, rangeid, set, score, label
- Adds via Xch.settrades! all Xch columns: opentime, pair, lo_id/lc_id/so_id/sc_id, lo_status/lc_status/so_status/sc_status, etc.
- Stores optional metadata columns from the metadata dict.
"""
function preparereplaytrades!(ts::TsCache, xc::Xch.XchCache, base::AbstractString, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector;
    quotecoin::AbstractString=EnvConfig.pairquote,
    metadata::AbstractDict{Symbol, Any}=Dict{Symbol, Any}(),
    datetime::Union{Nothing, DateTime}=nothing,
)::TsTp
    n = size(predictionsdf, 1)
    @assert n == length(scores) == length(labels) "size(predictionsdf, 1)=$(n) must match scores=$(length(scores)) and labels=$(length(labels))"
    @assert :opentime in propertynames(predictionsdf) "predictionsdf must contain :opentime; names=$(names(predictionsdf))"
    @assert :close in propertynames(predictionsdf) "predictionsdf must contain :close; names=$(names(predictionsdf))"

    pair = tspairkey(base, quotecoin)
    tp = getpairstate!(ts, pair)

    rebuild = true
    if nrow(tp.tradesdf) == n && (:opentime in propertynames(tp.tradesdf))
        if n == 0
            rebuild = false
        else
            rebuild = (tp.tradesdf[1, :opentime] != predictionsdf[1, :opentime]) || (tp.tradesdf[n, :opentime] != predictionsdf[n, :opentime])
        end
    end

    if rebuild
        tradesdf = DataFrame(predictionsdf; copycols=false)
        ohlcv_cols = filter(col -> col in propertynames(tradesdf), [:open, :basevolume, :pivot, :coin, :target])
        !isempty(ohlcv_cols) && select!(tradesdf, Not(ohlcv_cols))
        Xch.settrades!(xc, base, quotecoin, tradesdf)
        tp = syncpairtrades!(ts, xc, base, quotecoin; datetime=datetime)
    else
        tp = syncpairtrades!(ts, xc, base, quotecoin; datetime=datetime)
        tp.tradesdf[!, :close] .= predictionsdf[!, :close]
        tp.tradesdf[!, :high] .= predictionsdf[!, :high]
        tp.tradesdf[!, :low] .= predictionsdf[!, :low]
    end

    if n > 0
        fill!(@view(tp.tradesdf[:, :lo_limit]), 0f0)
        fill!(@view(tp.tradesdf[:, :lc_limit]), 0f0)
        fill!(@view(tp.tradesdf[:, :so_limit]), 0f0)
        fill!(@view(tp.tradesdf[:, :sc_limit]), 0f0)
    end

    tp.tradesdf[!, :score] = scores
    tp.tradesdf[!, :label] = labels

    for (k, v) in metadata
        tp.tradesdf[!, k] = fill(v, n)
    end
    return tp
end

function _validatereplayprepared!(tp::TsTp, lastix::Integer)::Nothing
    @assert !isempty(tp.pair) "Replay pair identifier must be non-empty"

    required_cols = (
        :opentime,
        :label,
        :score,
        :lastopentrade,
        :close,
        :high,
        :low,
        :lo_limit,
        :lc_limit,
        :so_limit,
        :sc_limit,
    )
    missing_cols = Symbol[col for col in required_cols if !(col in propertynames(tp.tradesdf))]
    @assert isempty(missing_cols) "Replay state for pair=$(tp.pair) is not prepared; missing columns=$(missing_cols)"

    if lastix > 0
        @assert nrow(tp.tradesdf) >= lastix "Replay state for pair=$(tp.pair) has fewer rows=$(nrow(tp.tradesdf)) than lastix=$(lastix)"
        if :pair in propertynames(tp.tradesdf)
            rowpair = uppercase(String(tp.tradesdf[1, :pair]))
            @assert rowpair == uppercase(tp.pair) "Replay state pair mismatch for pair=$(tp.pair): tradesdf pair=$(rowpair)"
        end
    end
    return nothing
end

"""Process gains for one replay pair after its Trades DataFrame has been prepared explicitly.

Thresholds are taken from the strategy config, not from parameters. The strategy.openthreshold
and strategy.closethreshold determine which trades pass the confidence filter during gain materialization.
Any open gain segment at lastix is force-closed using the close price at that row.
"""
function processreplaygains!(tp::TsTp;
    strategy::StrategyConfig,
    lastix::Integer=nrow(tp.tradesdf),
)::DataFrame
    _validatereplayprepared!(tp, lastix)
    gaindf = emptygaindf()

    if lastix > 0
        try
            simulate_gains!(strategy, tp, lastix, gaindf)
        catch err
            if (err isa MethodError) && (getfield(err, :f) === strategy.algorithm)
                throw(ArgumentError("strategy algorithm $(strategy.algorithm) does not support required signature in replay gain processing. Expected call shape: algorithm(strategy::StrategyConfig, tradesdf::DataFrame, ix::Integer)."))
            end
            rethrow(err)
        end
    end

    return gaindf
end

include("tradingstrategyconfig.jl")

end # module
