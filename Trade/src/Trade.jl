# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, longbuy when an uptrend starts and longclose when it ends.
It generates the OHLCV data, executes the trades in a loop and selects the basecoins to trade.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, Statistics
using EnvConfig, Ohlcv, CryptoXch, Classify, Features, Targets, TradeLog, TradingStrategy

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

"""
- buysell is the normal trade mode
- closeonly disables opening trades and only closes existing long/short positions
- quickexit sells all assets as soon as possible
- notrade for testing
"""
@enum TradeMode buysell closeonly quickexit notrade

# Backward compatibility alias (deprecated): `sellonly` == `closeonly`.
const sellonly = closeonly

"""
Loop lifecycle states stored in `TradeCache.mc[:loop_state]`.
- `loop_idle`: loop has not been started yet
- `loop_running`: loop is executing ticks
- `loop_paused`: loop is suspended between ticks
- `loop_stopping`: stop has been requested; loop will exit after current tick
- `loop_stopped`: loop has finished (either normally or after stop request)
"""
@enum LoopState loop_idle loop_running loop_paused loop_stopping loop_stopped


"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: essential status messages, e.g. load and save messages, are reported
- 3: print debug info
"""
verbosity = 2

# Extra minute buffer for liquidity lookback window to absorb minute-boundary rounding
# and small OHLCV gaps without underfetching the required continuity check horizon.
const LIQUIDITY_LOOKBACK_MARGIN_MINUTES = 5
const UNMANAGED_CANCEL_MAX_PER_STEP = 3
const UNMANAGED_CANCEL_COOLDOWN_FALLBACK = Dates.Second(45)
const UNMANAGED_BACKLOG_DRAIN_THRESHOLD = 20
const ORDER_AMEND_PRICE_REL_THRESHOLD_DEFAULT = 1f-4
const ASYNC_SHADOW_PRICE_TOLERANCE = 1f-5
const ASYNC_SHADOW_QTY_TOLERANCE = 1f-6

function _envtrue(name::AbstractString, default::Bool=false)::Bool
    raw = lowercase(strip(get(ENV, String(name), default ? "true" : "false")))
    return raw in ("1", "true", "yes", "on")
end

function _envboolmaybe(name::AbstractString)::Union{Nothing, Bool}
    key = String(name)
    haskey(ENV, key) || return nothing
    raw = lowercase(strip(String(ENV[key])))
    if raw in ("1", "true", "yes", "on")
        return true
    elseif raw in ("0", "false", "no", "off")
        return false
    end
    throw(ArgumentError("invalid boolean env $key=$(repr(raw)); expected one of 1/0,true/false,yes/no,on/off"))
end

function _envstr(name::AbstractString, default::AbstractString="")::String
    return haskey(ENV, String(name)) ? String(ENV[String(name)]) : String(default)
end

function _init_objective4_flags!(mc::AbstractDict)
    mc[:async_engine_enabled] = _envtrue("CTS_ASYNC_ENGINE_ENABLED", get(mc, :async_engine_enabled, false))
    mc[:async_shadow_mode] = _envtrue("CTS_ASYNC_SHADOW_MODE", get(mc, :async_shadow_mode, true))
    mc[:ws_marketdata_enabled] = _envtrue("CTS_WS_MARKETDATA_ENABLED", get(mc, :ws_marketdata_enabled, false))
    mc[:ws_orders_enabled] = _envtrue("CTS_WS_ORDERS_ENABLED", get(mc, :ws_orders_enabled, false))
    mc[:ws_balances_enabled] = _envtrue("CTS_WS_BALANCES_ENABLED", get(mc, :ws_balances_enabled, false))
    mc[:ws_shadow_mode] = _envtrue("CTS_WS_SHADOW_MODE", get(mc, :ws_shadow_mode, true))
    mc[:ws_primary_mode] = _envtrue("CTS_WS_PRIMARY_MODE", get(mc, :ws_primary_mode, false))
    mc[:ws_primary_autofallback_on_mismatch] = _envtrue("CTS_WS_PRIMARY_AUTOFALLBACK_ON_MISMATCH", get(mc, :ws_primary_autofallback_on_mismatch, true))
    mc[:ws_shadow_last_compare] = get(mc, :ws_shadow_last_compare, nothing)
    mc[:ws_primary_fallbacks] = Int(get(mc, :ws_primary_fallbacks, 0))
    mc[:tradelog_migration_worker_probe_enabled] = _envtrue("CTS_TRADELOG_MIGRATION_WORKER_PROBE_ENABLED", get(mc, :tradelog_migration_worker_probe_enabled, false))
    mc[:tradelog_migration_worker_probe_last_signature] = get(mc, :tradelog_migration_worker_probe_last_signature, Dict{Symbol, String}())
    mc[:ohlcv_gap_backfill_on_tradable] = _envtrue("CTS_OHLCV_GAP_BACKFILL_ON_TRADABLE", get(mc, :ohlcv_gap_backfill_on_tradable, false))
    mc[:async_shadow_autodisabled] = get(mc, :async_shadow_autodisabled, false)
    mc[:async_shadow_autodisable_reason] = get(mc, :async_shadow_autodisable_reason, nothing)
    mc[:async_shadow_last_compare] = get(mc, :async_shadow_last_compare, nothing)
    mc[:async_worker_channel_capacity] = Int(get(mc, :async_worker_channel_capacity, 4))
    mc[:async_worker_watchdog_timeout] = get(mc, :async_worker_watchdog_timeout, Dates.Second(120))
    mc[:async_worker_channels] = get(mc, :async_worker_channels, Dict{Symbol, Channel{Any}}())
    mc[:async_worker_heartbeats] = get(mc, :async_worker_heartbeats, Dict{Symbol, DateTime}())
    mc[:async_worker_last_latency_ms] = get(mc, :async_worker_last_latency_ms, Dict{Symbol, Float64}())
    mc[:async_worker_watchdog_breaches] = get(mc, :async_worker_watchdog_breaches, Dict{Symbol, Int}())
    mc[:async_worker_topology_started] = get(mc, :async_worker_topology_started, false)
    mc[:marketdata_ws_freshness_sla] = get(mc, :marketdata_ws_freshness_sla, Dates.Second(30))
    mc[:marketdata_ws_last_update_dt] = get(mc, :marketdata_ws_last_update_dt, nothing)
    mc[:marketdata_source] = get(mc, :marketdata_source, :http)
    mc[:marketdata_ws_fallback_active] = get(mc, :marketdata_ws_fallback_active, false)
    mc[:marketdata_ws_fallback_reason] = get(mc, :marketdata_ws_fallback_reason, nothing)
    mc[:marketdata_ws_fallback_switches] = Int(get(mc, :marketdata_ws_fallback_switches, 0))
    mc[:marketdata_last_policy_eval_dt] = get(mc, :marketdata_last_policy_eval_dt, nothing)
    mc[:marketdata_ws_last_update_by_symbol] = get(mc, :marketdata_ws_last_update_by_symbol, Dict{String, DateTime}())
    mc[:marketdata_ws_stale_symbols] = get(mc, :marketdata_ws_stale_symbols, String[])
    mc[:tradable_ohlcv_state_by_base] = get(mc, :tradable_ohlcv_state_by_base, Dict{String, Symbol}())
    mc[:tradable_ohlcv_state_dt_by_base] = get(mc, :tradable_ohlcv_state_dt_by_base, Dict{String, DateTime}())
    mc[:strategy_last_closed_candle_dt] = get(mc, :strategy_last_closed_candle_dt, nothing)
    mc[:strategy_closed_candle_pending_reason] = get(mc, :strategy_closed_candle_pending_reason, nothing)
    mc[:objective4_cycle_count] = Int(get(mc, :objective4_cycle_count, 0))
    mc[:objective4_order_rejects] = Int(get(mc, :objective4_order_rejects, 0))
    mc[:objective4_permission_rejects] = Int(get(mc, :objective4_permission_rejects, 0))
    mc[:objective4_privatecooldown_skips] = Int(get(mc, :objective4_privatecooldown_skips, 0))
    mc[:objective4_marketdata_fallback_activations] = Int(get(mc, :objective4_marketdata_fallback_activations, 0))
    mc[:objective4_watchdog_breaches_total] = Int(get(mc, :objective4_watchdog_breaches_total, 0))
    mc[:objective4_last_worker_latency_ms] = get(mc, :objective4_last_worker_latency_ms, Dict{Symbol, Float64}())
    return mc
end

function _set_ws_runtime_flags!(cache)
    cache.xc.mc[:ws_orders_enabled] = Bool(get(cache.mc, :ws_orders_enabled, false))
    cache.xc.mc[:ws_balances_enabled] = Bool(get(cache.mc, :ws_balances_enabled, false))
    cache.xc.mc[:ws_primary_mode] = Bool(get(cache.mc, :ws_primary_mode, false))
    return nothing
end

function _objective4_inc!(cache, key::Symbol, delta::Integer=1)
    cache.mc[key] = Int(get(cache.mc, key, 0)) + Int(delta)
    return Int(cache.mc[key])
end

"Return the next closed-candle datetime when progression is available; otherwise return nothing."
function _next_closed_candle_dt!(cache)::Union{Nothing, DateTime}
    currentdt = cache.xc.currentdt
    if isnothing(currentdt)
        cache.mc[:strategy_closed_candle_pending_reason] = :no_currentdt
        return nothing
    end
    closeddt = floor(DateTime(currentdt), Minute(1)) - Minute(1)
    lastdt = get(cache.mc, :strategy_last_closed_candle_dt, nothing)
    if !isnothing(lastdt) && (closeddt <= DateTime(lastdt))
        cache.mc[:strategy_closed_candle_pending_reason] = :no_new_closed_candle
        return nothing
    end
    cache.mc[:strategy_closed_candle_pending_reason] = nothing
    return closeddt
end

"Persist the most recent closed-candle datetime consumed by strategy/classifier updates."
function _mark_closed_candle_consumed!(cache, closeddt::DateTime)
    cache.mc[:strategy_last_closed_candle_dt] = DateTime(closeddt)
    cache.mc[:strategy_closed_candle_pending_reason] = nothing
    return cache
end

"Record one websocket marketdata update timestamp for freshness evaluation."
function _mark_marketdata_ws_update!(cache; datetime=nothing)
    dt = isnothing(datetime) ? cache.xc.currentdt : datetime
    if isnothing(dt)
        dt = floor(Dates.now(Dates.UTC), Minute(1))
    end
    cache.mc[:marketdata_ws_last_update_dt] = dt
    return dt
end

"Sync Trade-local websocket heartbeat state from canonical exchange-owned heartbeat when available."
function _sync_marketdata_ws_heartbeat_from_exchange!(cache)
    wsdt = try
        CryptoXch.marketdataheartbeat(cache.xc)
    catch
        nothing
    end
    if !isnothing(wsdt)
        localdt = get(cache.mc, :marketdata_ws_last_update_dt, nothing)
        if isnothing(localdt) || (DateTime(wsdt) > DateTime(localdt))
            cache.mc[:marketdata_ws_last_update_dt] = DateTime(wsdt)
        end
    end
    symbolmap = get(cache.mc, :marketdata_ws_last_update_by_symbol, Dict{String, DateTime}())
    trymap = try
        CryptoXch.marketdataheartbeats(cache.xc)
    catch
        Dict{String, DateTime}()
    end
    for (sym, dt) in trymap
        key = uppercase(String(sym))
        prev = get(symbolmap, key, nothing)
        if isnothing(prev) || (DateTime(dt) > DateTime(prev))
            symbolmap[key] = DateTime(dt)
        end
    end
    cache.mc[:marketdata_ws_last_update_by_symbol] = symbolmap
    return get(cache.mc, :marketdata_ws_last_update_dt, nothing)
end

function _marketdata_symbols_from_advices(cache, sync_advices)
    symbols = String[]
    for ta in sync_advices
        base = uppercase(String(ta.base))
        push!(symbols, CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.data_exchange))
    end
    return unique(symbols)
end

"Evaluate websocket freshness SLA and auto-switch between websocket and HTTP marketdata sources."
function _update_marketdata_freshness_policy!(cache; symbols::Union{Nothing, AbstractVector}=nothing)
    nowdt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : cache.xc.currentdt
    _sync_marketdata_ws_heartbeat_from_exchange!(cache)
    ws_enabled = Bool(get(cache.mc, :ws_marketdata_enabled, false))
    sla = get(cache.mc, :marketdata_ws_freshness_sla, Dates.Second(30))
    ws_last = get(cache.mc, :marketdata_ws_last_update_dt, nothing)
    prev_source = Symbol(get(cache.mc, :marketdata_source, :http))
    prev_fallback_active = Bool(get(cache.mc, :marketdata_ws_fallback_active, false))

    source = :http
    fallback_active = false
    fallback_reason = nothing

    if ws_enabled
        if isnothing(ws_last)
            fallback_active = true
            fallback_reason = :ws_no_updates
        else
            ws_age = nowdt - DateTime(ws_last)
            if ws_age <= sla
                source = :ws
            else
                fallback_active = true
                fallback_reason = :ws_stale
            end
        end
        if source == :ws && !isnothing(symbols)
            stale = String[]
            symbolmap = get(cache.mc, :marketdata_ws_last_update_by_symbol, Dict{String, DateTime}())
            for sym in symbols
                key = uppercase(String(sym))
                sdt = get(symbolmap, key, nothing)
                if isnothing(sdt) || ((nowdt - DateTime(sdt)) > sla)
                    push!(stale, key)
                end
            end
            cache.mc[:marketdata_ws_stale_symbols] = stale
            if !isempty(stale)
                source = :http
                fallback_active = true
                fallback_reason = :ws_symbol_stale
            end
        else
            cache.mc[:marketdata_ws_stale_symbols] = String[]
        end
    else
        fallback_reason = :ws_disabled
    end

    if ws_enabled && (prev_source != source) && (source == :http)
        cache.mc[:marketdata_ws_fallback_switches] = Int(get(cache.mc, :marketdata_ws_fallback_switches, 0)) + 1
    end

    cache.mc[:marketdata_source] = source
    cache.mc[:marketdata_ws_fallback_active] = fallback_active
    cache.mc[:marketdata_ws_fallback_reason] = fallback_reason
    cache.mc[:marketdata_last_policy_eval_dt] = nowdt
    if fallback_active && !prev_fallback_active
        cache.mc[:objective4_marketdata_fallback_activations] = Int(get(cache.mc, :objective4_marketdata_fallback_activations, 0)) + 1
    end

    ws_age_seconds = isnothing(ws_last) ? nothing : Float64(Dates.value(nowdt - DateTime(ws_last))) / 1000.0
    return (
        source=source,
        ws_enabled=ws_enabled,
        fallback_active=fallback_active,
        fallback_reason=fallback_reason,
        ws_last_update=ws_last,
        ws_age_seconds=ws_age_seconds,
        stale_symbols=copy(get(cache.mc, :marketdata_ws_stale_symbols, String[])),
        fallback_switches=Int(get(cache.mc, :marketdata_ws_fallback_switches, 0)),
        evaluated_at=nowdt,
    )
end

"Return Objective 4.2 worker names in canonical processing order."
_async_worker_names() = (:marketdata, :balance_sync, :order_management)

"Ensure bounded worker channels and per-worker metrics dictionaries are initialized."
function _ensure_async_worker_topology!(cache)
    channels = get(cache.mc, :async_worker_channels, Dict{Symbol, Channel{Any}}())
    heartbeats = get(cache.mc, :async_worker_heartbeats, Dict{Symbol, DateTime}())
    latencies = get(cache.mc, :async_worker_last_latency_ms, Dict{Symbol, Float64}())
    breaches = get(cache.mc, :async_worker_watchdog_breaches, Dict{Symbol, Int}())
    capacity = max(1, Int(get(cache.mc, :async_worker_channel_capacity, 4)))
    nowdt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : cache.xc.currentdt
    for worker in _async_worker_names()
        if !haskey(channels, worker)
            channels[worker] = Channel{Any}(capacity)
        end
        if !haskey(heartbeats, worker)
            heartbeats[worker] = nowdt
        end
        if !haskey(latencies, worker)
            latencies[worker] = 0.0
        end
        if !haskey(breaches, worker)
            breaches[worker] = 0
        end
    end
    cache.mc[:async_worker_channels] = channels
    cache.mc[:async_worker_heartbeats] = heartbeats
    cache.mc[:async_worker_last_latency_ms] = latencies
    cache.mc[:async_worker_watchdog_breaches] = breaches
    cache.mc[:async_worker_topology_started] = true
    return nothing
end

"Record one worker heartbeat and latency metric after processing one shadow payload."
function _record_async_worker_heartbeat!(cache, worker::Symbol, started_at::DateTime)
    nowdt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : cache.xc.currentdt
    elapsed_ms = max(0.0, Float64(Dates.value(nowdt - started_at)) / 1_000.0)
    cache.mc[:async_worker_heartbeats][worker] = nowdt
    cache.mc[:async_worker_last_latency_ms][worker] = elapsed_ms
    return nothing
end

function _should_write_async_worker_probe(cache, worker::Symbol)::Bool
    Bool(get(cache.mc, :tradelog_migration_worker_probe_enabled, false)) || return false
    return worker in (:balance_sync, :order_management)
end

function _async_worker_probe_signature(payload)::String
    return sprint(show, payload)
end

function _async_worker_probe_notes(worker::Symbol, payload)::String
    parts = ["migration_probe=async_worker_observation_v1", "worker=$(worker)"]
    for key in propertynames(payload)
        push!(parts, "$(key)=$(getproperty(payload, key))")
    end
    return join(parts, ";")
end

function _maybe_write_async_worker_probe!(cache, worker::Symbol, payload)
    _should_write_async_worker_probe(cache, worker) || return nothing
    signatures = cache.mc[:tradelog_migration_worker_probe_last_signature]
    signature = _async_worker_probe_signature(payload)
    if get(signatures, worker, "") == signature
        return nothing
    end

    event_time = Dates.now(Dates.UTC)
    exchange_name = CryptoXch._routeexchange(cache.xc.routing, CryptoXch.trade_exchange_spot, CryptoXch.exchange(cache.xc))
    position_after = if hasproperty(payload, :snapshot_rows)
        Float64(getproperty(payload, :snapshot_rows))
    elseif hasproperty(payload, :openorders_rows)
        Float64(getproperty(payload, :openorders_rows))
    else
        missing
    end
    cash_after = hasproperty(payload, :quote_total) ? Float64(getproperty(payload, :quote_total)) : missing
    event = TradeLog.AuditEventRow(
        event_type=TradeLog.POSITION_SNAPSHOT,
        event_time_utc=event_time,
        created_at_utc=event_time,
        source_module="Trade",
        environment=string(Symbol(EnvConfig.configmode)),
        run_mode=CryptoXch.tradelogrunmode(cache.xc),
        run_id=CryptoXch.tradelogrunid(cache.xc),
        exchange=exchange_name,
        account_alias=exchange_name,
        routing_role=TradeLog.routing_trade_exchange_spot,
        market_type=TradeLog.market_unknown,
        asset_class=TradeLog.crypto,
        instrument_type=TradeLog.instrument_unknown,
        symbol="ASYNC_WORKER_$(uppercase(String(worker)))",
        status="worker_payload_changed",
        status_reason="source=objective4_async_worker_probe",
        position_qty_after=position_after,
        cash_after=cash_after,
        notes=_async_worker_probe_notes(worker, payload),
    )
    try
        TradeLog.writeeventwithhash(event)
        signatures[worker] = signature
    catch tradelog_error
        (verbosity >= 1) && @warn "failed to persist async worker probe snapshot" worker exception=(tradelog_error, catch_backtrace())
    end
    return nothing
end

"Drain stale queue items and process one latest payload through the worker channel."
function _run_async_worker_shadow_step!(cache, worker::Symbol, payload)
    channels = cache.mc[:async_worker_channels]
    channel = channels[worker]
    while isready(channel)
        take!(channel)
    end
    put!(channel, payload)
    started_at = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : cache.xc.currentdt
    current = take!(channel)
    _maybe_write_async_worker_probe!(cache, worker, current)
    _record_async_worker_heartbeat!(cache, worker, started_at)
    return current
end

"Update watchdog breach counters when a worker heartbeat is older than timeout."
function _update_async_worker_watchdog!(cache)
    heartbeats = cache.mc[:async_worker_heartbeats]
    breaches = cache.mc[:async_worker_watchdog_breaches]
    timeout = get(cache.mc, :async_worker_watchdog_timeout, Dates.Second(120))
    nowdt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : cache.xc.currentdt
    for worker in _async_worker_names()
        hb = get(heartbeats, worker, nowdt)
        if (nowdt - hb) > timeout
            breaches[worker] = get(breaches, worker, 0) + 1
            cache.mc[:objective4_watchdog_breaches_total] = Int(get(cache.mc, :objective4_watchdog_breaches_total, 0)) + 1
        end
    end
    cache.mc[:async_worker_watchdog_breaches] = breaches
    return nothing
end

"Run Objective 4.2 shadow worker pipeline while leaving synchronous execution authoritative."
function _run_async_shadow_topology!(cache, assets::AbstractDataFrame, openorders::AbstractDataFrame, sync_advices; seed_closed_dt=nothing)
    _ensure_async_worker_topology!(cache)
    mdsymbols = _marketdata_symbols_from_advices(cache, sync_advices)
    md_state = _update_marketdata_freshness_policy!(cache; symbols=mdsymbols)
    snapshot = CryptoXch.balancessnapshot(cache.xc; force_refresh=false, ignoresmallvolume=false).snapshot
    md_payload = (
        datetime=cache.xc.currentdt,
        assets_rows=size(assets, 1),
        openorders_rows=size(openorders, 1),
        source=md_state.source,
        ws_enabled=md_state.ws_enabled,
        fallback_active=md_state.fallback_active,
        fallback_reason=md_state.fallback_reason,
        ws_age_seconds=md_state.ws_age_seconds,
        stale_symbols=md_state.stale_symbols,
    )
    _run_async_worker_shadow_step!(cache, :marketdata, md_payload)

    async_advices = _collect_async_candidate_advices(cache, assets, sync_advices; seed_closed_dt=seed_closed_dt)
    snapshot_state = CryptoXch.balancessnapshot(cache.xc; force_refresh=false, ignoresmallvolume=false)
    balance_sync_payload = (
        datetime=cache.xc.currentdt,
        snapshot_dt=snapshot_state.datetime,
        snapshot_rows=size(snapshot, 1),
        quote_total=(size(snapshot, 1) == 0 ? 0.0 : _portfolioquotevalue(snapshot)),
    )
    _run_async_worker_shadow_step!(cache, :balance_sync, balance_sync_payload)

    order_management_payload = (
        datetime=cache.xc.currentdt,
        authoritative=:sync,
        shadow_mode=Bool(get(cache.mc, :async_shadow_mode, true)),
        sync_count=length(sync_advices),
        candidate_count=length(async_advices),
        openorders_rows=size(openorders, 1),
        stage=:execution_and_reconcile,
    )
    _run_async_worker_shadow_step!(cache, :order_management, order_management_payload)

    _update_async_worker_watchdog!(cache)
    cache.mc[:objective4_last_worker_latency_ms] = deepcopy(get(cache.mc, :async_worker_last_latency_ms, Dict{Symbol, Float64}()))
    return async_advices
end

function _setstrategyruntimefromsegment!(mc::AbstractDict, gs::TradingStrategy.GainSegment, source::AbstractString)
    mc[:strategy_template] = deepcopy(gs)
    mc[:strategy_algorithm] = gs.algorithm
    mc[:strategy_openthreshold] = Float32(gs.openthreshold)   # compatibility mirror
    mc[:strategy_closethreshold] = Float32(gs.closethreshold) # compatibility mirror
    mc[:strategy_buygain] = Float32(gs.buygain)               # compatibility mirror
    mc[:strategy_sellgain] = Float32(gs.sellgain)             # compatibility mirror
    mc[:strategy_limitreduction] = Float32(gs.limitreduction) # compatibility mirror
    mc[:strategy_maxwindow] = Int(gs.maxwindow)               # compatibility mirror
    mc[:strategy_source] = String(source)
    return mc
end

"Return true when runtime strategy API should be enabled by default for this cache."
function _default_use_strategy_runtime_api(xc::CryptoXch.XchCache)::Bool
    _ = xc
    env_override = _envboolmaybe("CTS_USE_STRATEGY_RUNTIME_API")
    !isnothing(env_override) && return Bool(env_override)
    return EnvConfig.configmode == EnvConfig.test
end

function _portfoliototal(assets::AbstractDataFrame)::Float64
    return size(assets, 1) == 0 ? 0.0 : Float64(sum(assets[!, :usdtvalue]))
end

"Return the effective trading budget in quote currency, capped by `mc[:maxbudgetquote]` and reduced by safety margin when configured."
function _effectivebudgetquote(cache, assets::AbstractDataFrame)::Float64
    _ = assets
    capacity = CryptoXch.accountcapacity(cache.xc)
    available_opening_quote = Float64(get(capacity, :available_opening_quote, 0.0))
    safetymargin = Float64(get(cache.mc, :budgetsafetymargin, 0.0))
    safetymargin = clamp(safetymargin, 0.0, 0.99)
    budgetwithsafety = max(0.0, available_opening_quote * (1.0 - safetymargin))
    maxbudget = get(cache.mc, :maxbudgetquote, get(cache.mc, :maxbudgetusdt, nothing))
    if isnothing(maxbudget)
        return budgetwithsafety
    end
    cap = Float64(maxbudget)
    if !isfinite(cap) || (cap <= 0.0)
        return budgetwithsafety
    end
    return min(budgetwithsafety, cap)
end

"Return allocated budget as gross non-quote exposure from assets/positions (long + short)."
function _allocatedbudgetquote(assets::AbstractDataFrame)::Float64
    required = ("coin", "free", "locked", "borrowed", "usdtprice")
    assetnames = Set(String.(names(assets)))
    if any(name -> !(name in assetnames), required)
        return 0.0
    end
    quotetoken = uppercase(String(EnvConfig.cryptoquote))
    allocated = 0.0
    for row in eachrow(assets)
        coin = uppercase(String(row.coin))
        coin == quotetoken && continue
        price = Float64(row.usdtprice)
        (!isfinite(price) || (price <= 0.0)) && continue
        grossbase = abs(Float64(row.free) + Float64(row.locked)) + abs(Float64(row.borrowed))
        allocated += grossbase * price
    end
    return allocated
end

"Return the explicit limit price used for order creation in simulation mode."
function _orderlimitprice(cache, price::Real)
    return cache.xc.mc[:simmode] == CryptoXch.bybitsim ? price : nothing
end

function _portfolioquotevalue(assets::AbstractDataFrame)::Union{Missing, Float64}
    if size(assets, 1) == 0 || !any(name -> name == "coin", names(assets))
        return missing
    end
    quoteix = findfirst(==(EnvConfig.cryptoquote), assets[!, :coin])
    if isnothing(quoteix)
        return missing
    end
    return Float64((assets[quoteix, :free] + assets[quoteix, :locked]) - assets[quoteix, :borrowed])
end

function _update_account_equity_snapshot!(cache)::NamedTuple{(:equity_quote, :equity_delta), Tuple{Float64, Union{Nothing, Float64}}}
    equity_quote = max(0.0, Float64(get(CryptoXch.accountcapacity(cache.xc), :equity_quote, 0.0)))
    prev_equity = get(cache.mc, :last_account_equity_quote, nothing)
    equity_delta = isnothing(prev_equity) ? nothing : (equity_quote - Float64(prev_equity))
    cache.mc[:last_account_equity_quote] = equity_quote
    cache.mc[:last_account_equity_delta] = equity_delta
    return (equity_quote=equity_quote, equity_delta=equity_delta)
end

function _writeportfoliosnapshot!(cache, assets::AbstractDataFrame; source_module::AbstractString="Trade")
    rowcount = size(assets, 1)
    simmode = String(Symbol(cache.xc.mc[:simmode]))
    event_time = Dates.now(Dates.UTC)
    portfolio_total = _portfoliototal(assets)
    cash_after = _portfolioquotevalue(assets)
    equity_quote = max(0.0, Float64(get(cache.mc, :last_account_equity_quote, 0.0)))
    equity_delta = get(cache.mc, :last_account_equity_delta, nothing)
    equity_delta_text = isnothing(equity_delta) ? "NA" : string(round(Int, equity_delta))
    exchange_name = CryptoXch._routeexchange(cache.xc.routing, CryptoXch.trade_exchange_spot, CryptoXch.exchange(cache.xc))
    account_alias = exchange_name
    try
        if rowcount == 0
            event = TradeLog.AuditEventRow(
                event_type=TradeLog.PORTFOLIO_SNAPSHOT,
                event_time_utc=event_time,
                created_at_utc=event_time,
                source_module=String(source_module),
                environment=string(Symbol(EnvConfig.configmode)),
                run_mode=CryptoXch.tradelogrunmode(cache.xc),
                run_id=CryptoXch.tradelogrunid(cache.xc),
                exchange=exchange_name,
                account_alias=account_alias,
                routing_role=TradeLog.routing_trade_exchange_spot,
                market_type=TradeLog.market_unknown,
                asset_class=TradeLog.crypto,
                instrument_type=TradeLog.instrument_unknown,
                symbol="PORTFOLIO",
                cash_after=cash_after,
                portfolio_value_after=portfolio_total,
                notes="rows=0; simmode=$(simmode); equity_quote=$(round(Int, equity_quote)); equity_delta=$(equity_delta_text)"
            )
            TradeLog.writeeventwithhash(event)
            return nothing
        end

        hascoin = "coin" in names(assets)
        hasfree = "free" in names(assets)
        haslocked = "locked" in names(assets)
        hasborrowed = "borrowed" in names(assets)
        hasusdtvalue = "usdtvalue" in names(assets)
        for row in eachrow(assets)
            coin = hascoin ? String(row[:coin]) : "UNKNOWN"
            freeqty = hasfree ? Float64(row[:free]) : 0.0
            lockedqty = haslocked ? Float64(row[:locked]) : 0.0
            borrowedqty = hasborrowed ? Float64(row[:borrowed]) : 0.0
            positionqty = freeqty + lockedqty - borrowedqty
            positionvalue = hasusdtvalue ? Float64(row[:usdtvalue]) : missing
            event = TradeLog.AuditEventRow(
                event_type=TradeLog.PORTFOLIO_SNAPSHOT,
                event_time_utc=event_time,
                created_at_utc=event_time,
                source_module=String(source_module),
                environment=string(Symbol(EnvConfig.configmode)),
                run_mode=CryptoXch.tradelogrunmode(cache.xc),
                run_id=CryptoXch.tradelogrunid(cache.xc),
                exchange=exchange_name,
                account_alias=account_alias,
                routing_role=TradeLog.routing_trade_exchange_spot,
                market_type=TradeLog.market_unknown,
                asset_class=TradeLog.crypto,
                instrument_type=TradeLog.spot_pair,
                symbol=coin,
                baseasset=coin,
                quoteasset=EnvConfig.cryptoquote,
                settlement_asset=EnvConfig.cryptoquote,
                position_qty_after=positionqty,
                cash_after=(coin == EnvConfig.cryptoquote ? positionqty : cash_after),
                portfolio_value_after=portfolio_total,
                fill_notional=positionvalue,
                notes="asset=$(coin); rows=$(rowcount); simmode=$(simmode); equity_quote=$(round(Int, equity_quote)); equity_delta=$(equity_delta_text)"
            )
            TradeLog.writeeventwithhash(event)
        end
    catch tradelog_error
        (verbosity >= 1) && @warn "failed to persist portfolio snapshot" exception=(tradelog_error, catch_backtrace())
    end
    return nothing
end

"Write portfolio tradelog snapshots according to `cache.mc[:tradelog_portfolio_snapshot_mode]`."
function _maybe_writeportfoliosnapshot!(cache, assets::AbstractDataFrame)
    mode = get(cache.mc, :tradelog_portfolio_snapshot_mode, get(cache.mc, :audit_portfolio_snapshot_mode, :all))
    if mode == :none
        return nothing
    elseif mode == :session_start
        if !get(cache.mc, :tradelog_portfolio_snapshot_written, get(cache.mc, :audit_portfolio_snapshot_written, false))
            _writeportfoliosnapshot!(cache, assets)
            cache.mc[:tradelog_portfolio_snapshot_written] = true
        end
        return nothing
    elseif mode == :all
        _writeportfoliosnapshot!(cache, assets)
        return nothing
    end
    @warn "unknown tradelog portfolio snapshot mode=$(mode); expected :all, :session_start or :none"
    return nothing
end

"""
*TradeCache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *TradeCache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct TradeCache
    xc::CryptoXch.XchCache  # required to connect to exchange
    cfg::AbstractDataFrame    # maintains the bases to trade and their classifiers
    cl::Classify.AbstractClassifier
    mc::Dict # MC = module constants
    dbgdf
    looplock::ReentrantLock
    loopcond::Threads.Condition
    function TradeCache(; xc=CryptoXch.XchCache(), cl=Classify.Classifier011(), trademode=notrade)
        looplock = ReentrantLock()
        cache = new(xc, DataFrame(), cl, Dict(), DataFrame(), looplock, Threads.Condition(looplock))
        cache.mc[:exitcoins] = [] # exit specific coins
        cache.mc[:longopencoins] = []  # force open long
        cache.mc[:shortopencoins] = [] # force open short
        cache.mc[:restrictedcoins] = String[] # coins excluded from the robot universe (e.g. account-region restrictions)
        cache.mc[:whitelistcoins] = ["ADA", "AI16Z", "APEX", "AAVE", "BNB", "BTC", "CAKE", "DOGE", "ELX", "ENA", "ETH", "HBAR", "HFT", "JUP", "LINK", "LTC", "MNT", "ONDO", "PEPE", "POPCAT", "S", "SOL", "SUI", "TON", "TRX", "VIRTUAL", "W", "WAL", "WIF", "WLD", "X", "XLM", "XRP"] 
        # not whitelisted: "ANIME", "B3", "BERA", "CMETH", "LDO", "PLUME", "SOSO", "TRUMP"
        cache.mc[:hourlygainlimit] = 0.1f0 # limit hourly gain to a realistic 10% max
        cache.mc[:maxassetfraction] = 0.1f0 # defines the maximum ratio of (a specific asset) / ( total assets) - only close trades, if this is exceeded
        cache.mc[:maxbudgetquote] = nothing # optional overall quote-currency budget cap; if set, trading uses min(totalusdt, maxbudgetquote)
        cache.mc[:maxbudgetusdt] = nothing # deprecated alias for backward compatibility
        cache.mc[:budgetsafetymargin] = 0.05 # budget limit uses sum(balance) * (1 - budgetsafetymargin)
        cache.mc[:reloadtimes] = [Time("04:00:00")]
        cache.mc[:last_traderefresh_dt] = nothing
        cache.mc[:trademode] = trademode  # see TradeMode definition above
        cache.mc[:usenewtrade] = false # implementation switch between old and new trade! method
        cache.mc[:strategy_engine] = :classifier  # :classifier (legacy) or :getgainsalgo
        cache.mc[:use_strategy_runtime_api] = _default_use_strategy_runtime_api(xc) # when true, collect strategy snapshots via TradingStrategy runtime API
        cache.mc[:strategy_state] = Dict{String, Any}()  # per-base TradingStrategy.GainSegment
        cache.mc[:strategy_history] = Dict{String, Any}()  # per-base rolling price+signal history
        cache.mc[:managed_close_orders] = Dict{String, Dict{Symbol, Any}}()  # per-base reconstructed/managed close orders
        cache.mc[:openorders_snapshot] = DataFrame()
        _setstrategyruntimefromsegment!(cache.mc, TradingStrategy.GainSegment(), "default")
        cache.mc[:strategy_runtime] = try
            TradingStrategy.GainSegmentRuntime(classifier=cl, strategy=deepcopy(cache.mc[:strategy_template]), source="default")
        catch
            nothing
        end
        cache.mc[:tradelog_portfolio_snapshot_mode] = :all  # :all, :session_start, :none
        cache.mc[:tradelog_portfolio_snapshot_written] = false
        cache.mc[:loop_state] = loop_idle
        _init_objective4_flags!(cache.mc)
        (verbosity >= 4) && println("TradeCache trademode = $(cache.mc[:trademode]), maxassetfraction = $(cache.mc[:maxassetfraction]), maxbudgetquote = $(cache.mc[:maxbudgetquote]), reloadtimes = $(cache.mc[:reloadtimes]), exitcoins = $(cache.mc[:exitcoins]), whitelistcoins = $(cache.mc[:whitelistcoins]), longopencoins = $(cache.mc[:longopencoins]), shortopencoins = $(cache.mc[:shortopencoins])")
        return cache
    end
end

function _openorderssnapshot(cache::TradeCache)::DataFrame
    oo = get(cache.mc, :openorders_snapshot, DataFrame())
    return oo isa DataFrame ? oo : DataFrame()
end

"Return currently reserved open non-leverage sell quantity for one symbol from the latest open-order snapshot."
function _reservedspotsellqty(cache::TradeCache, symbol::AbstractString; exclude_orderid::Union{Nothing, AbstractString}=nothing)::Float32
    oo = _openorderssnapshot(cache)
    size(oo, 1) == 0 && return 0f0
    excluded = isnothing(exclude_orderid) ? "" : String(exclude_orderid)
    total = 0f0
    for orow in eachrow(oo)
        CryptoXch.openstatus(String(orow.status)) || continue
        String(orow.symbol) == String(symbol) || continue
        uppercase(String(orow.side)) == "SELL" || continue
        _orderisleverage(orow) && continue
        oid = hasproperty(orow, :orderid) ? String(orow.orderid) : ""
        (!isempty(excluded) && (oid == excluded)) && continue
        baseqty = hasproperty(orow, :baseqty) ? Float32(orow.baseqty) : 0f0
        executed = hasproperty(orow, :executedqty) ? Float32(orow.executedqty) : 0f0
        total += max(0f0, baseqty - executed)
    end
    return max(0f0, total)
end

function _activeopenbuysymbols!(cache::TradeCache)::Set{String}
    if !haskey(cache.mc, :active_open_buy_symbols)
        cache.mc[:active_open_buy_symbols] = Set{String}()
    end
    return cache.mc[:active_open_buy_symbols]
end

function _refreshactiveopenbuysymbols!(cache::TradeCache, oo::AbstractDataFrame)
    active = _activeopenbuysymbols!(cache)
    empty!(active)
    for orow in eachrow(oo)
        CryptoXch.openstatus(String(orow.status)) || continue
        lowercase(String(orow.side)) == "buy" || continue
        if _orderisleverage(orow)
            continue
        end
        push!(active, uppercase(String(orow.symbol)))
    end
    return active
end

function _activeopensellsymbols!(cache::TradeCache)::Set{String}
    if !haskey(cache.mc, :active_open_sell_symbols)
        cache.mc[:active_open_sell_symbols] = Set{String}()
    end
    return cache.mc[:active_open_sell_symbols]
end

function _refreshactiveopensellsymbols!(cache::TradeCache, oo::AbstractDataFrame)
    active = _activeopensellsymbols!(cache)
    empty!(active)
    for orow in eachrow(oo)
        CryptoXch.openstatus(String(orow.status)) || continue
        lowercase(String(orow.side)) == "sell" || continue
        if !_orderisleverage(orow)
            continue
        end
        push!(active, uppercase(String(orow.symbol)))
    end
    return active
end

"Return whether one order row is a leverage/margin order across adapter schemas."
function _orderisleverage(orow)::Bool
    if hasproperty(orow, :isLeverage)
        return Bool(getproperty(orow, :isLeverage))
    elseif hasproperty(orow, :marginleverage)
        try
            return Int(getproperty(orow, :marginleverage)) > 0
        catch
            return false
        end
    end
    return false
end

function _rememberactiveopenbuy!(cache::TradeCache, symbol::AbstractString)
    push!(_activeopenbuysymbols!(cache), uppercase(String(symbol)))
    return cache
end

function _rememberactiveopensell!(cache::TradeCache, symbol::AbstractString)
    push!(_activeopensellsymbols!(cache), uppercase(String(symbol)))
    return cache
end

function _hasactiveopenbuy(cache::TradeCache, symbol::AbstractString)::Bool
    return uppercase(String(symbol)) in _activeopenbuysymbols!(cache)
end

function _hasactiveopensell(cache::TradeCache, symbol::AbstractString)::Bool
    return uppercase(String(symbol)) in _activeopensellsymbols!(cache)
end

"Return true when an active open order already exists for symbol/side/leverage class in snapshot."
function _hasopenorderside(cache::TradeCache, symbol::AbstractString; side::AbstractString, require_leverage::Union{Nothing, Bool}=nothing)::Bool
    wanted_symbol = uppercase(String(symbol))
    wanted_side = lowercase(String(side))
    oo = _openorderssnapshot(cache)
    for orow in eachrow(oo)
        CryptoXch.openstatus(String(orow.status)) || continue
        uppercase(String(orow.symbol)) == wanted_symbol || continue
        lowercase(String(orow.side)) == wanted_side || continue
        if !isnothing(require_leverage)
            (_orderisleverage(orow) == require_leverage) || continue
        end
        return true
    end
    return false
end

function _strategyruntime(cache::TradeCache)::Union{Nothing, TradingStrategy.AbstractStrategyRuntime}
    rt = get(cache.mc, :strategy_runtime, nothing)
    return rt isa TradingStrategy.AbstractStrategyRuntime ? rt : nothing
end

function _runtime_api_enabled(cache::TradeCache)::Bool
    return Bool(get(cache.mc, :use_strategy_runtime_api, false)) && !isnothing(_strategyruntime(cache))
end

function _tradeselection_history_minutes(tc::TradeCache)::Int
    classifier_minutes = if _runtime_api_enabled(tc)
        rt = _strategyruntime(tc)
        isnothing(rt) ? 0 : max(0, Int(TradingStrategy.requiredhistoryminutes(rt)))
    else
        try
            Int(Classify.requiredminutes(tc.cl))
        catch
            0
        end
    end
    liquidity_minutes = Int(Ohlcv.ld.checkperiod + Ohlcv.ld.accumulate + LIQUIDITY_LOOKBACK_MARGIN_MINUTES)
    return max(classifier_minutes + 1, liquidity_minutes, 24 * 60)
end

function _wait_for_live_usdtmarket!(tc::TradeCache, datetime::DateTime; requestedbases::Union{Nothing, AbstractVector{<:AbstractString}}=nothing)
    down_start = Dates.now(Dates.UTC)
    attempts = 0
    quotecoin = uppercase(String(EnvConfig.cryptoquote))
    requested = isnothing(requestedbases) ? String[] : unique([uppercase(String(b)) for b in requestedbases if !isempty(String(b)) && (uppercase(String(b)) != quotecoin)])
    while true
        marketdf = CryptoXch.screeningUSDTmarket(tc.xc; dt=datetime)
        if size(marketdf, 1) > 0
            if attempts > 0
                downtime = Dates.now(Dates.UTC) - down_start
                @warn "$(quotecoin) market snapshot restored after downtime" datetime attempts downtime
            end
            return marketdf
        end

        # Fallback: query only requested bases to reduce load and isolate pair-specific failures.
        if !isempty(requested)
            scoped = CryptoXch.valuationUSDTmarket(tc.xc, requested; dt=datetime)
            if size(scoped, 1) > 0
                if attempts > 0
                    downtime = Dates.now(Dates.UTC) - down_start
                    @warn "$(quotecoin) market snapshot restored from scoped fallback" datetime attempts downtime symbols=length(requested)
                end
                return scoped
            end
        end

        attempts += 1
        if attempts == 1
            @warn "$(quotecoin) market snapshot unavailable; polling every second until restored" datetime
        elseif attempts % 60 == 0
            @warn "$(quotecoin) market snapshot still unavailable" datetime attempts
        end
        sleep(1)
    end
end

"Use OHLCV-derived marketview in replay/simulation modes instead of persisted trade config snapshots."
function _uses_simulated_marketview(tc::TradeCache)::Bool
    return !isnothing(tc.xc.enddt) || (tc.xc.mc[:simmode] != CryptoXch.nosimulation)
end

@inline function _rowix_at_or_before(opentimes, datetime::DateTime)::Int
    return searchsortedlast(opentimes, datetime)
end

function _rolling_quotevolume24h(df::AbstractDataFrame, endix::Int, enddt::DateTime)::Float64
    startdt = enddt - Day(1)
    ot = df[!, :opentime]
    stopix = min(endix, searchsortedlast(ot, enddt))
    startix = searchsortedlast(ot, startdt) + 1  # strictly greater than startdt
    if (startix > stopix) || (stopix < 1)
        return 0.0
    end
    if :quotevolume in propertynames(df)
        qv = @view df[startix:stopix, :quotevolume]
        return Float64(sum(qv))
    end
    @assert (:basevolume in propertynames(df)) && (:close in propertynames(df)) "OHLCV dataframe must include quotevolume or basevolume+close; names=$(names(df))"
    basevol = @view df[startix:stopix, :basevolume]
    closes = @view df[startix:stopix, :close]
    s = 0.0
    @inbounds for ix in eachindex(basevol)
        s += Float64(basevol[ix]) * Float64(closes[ix])
    end
    return s
end

function _rolling_pricechangepercent24h(df::AbstractDataFrame, endix::Int, enddt::DateTime)::Float32
    startdt = enddt - Day(1)
    ot = df[!, :opentime]
    startix = searchsortedfirst(ot, startdt)
    if !(1 <= startix <= endix)
        return 0f0
    end
    firstclose = Float64(df[startix, :close])
    lastclose = Float64(df[endix, :close])
    if firstclose <= 0.0
        return 0f0
    end
    return Float32(((lastclose / firstclose) - 1.0) * 100.0)
end

"""
Fast liquidity gate for trade selection at `datetime`.

The objective is to admit coins that are liquid overall (24h quote volume gate)
and currently liquid continuously over the recent `checkperiod` window.
"""
function _continuous_liquidity_now(df::AbstractDataFrame, datetime::DateTime;
    minquotevol::Float32=Ohlcv.ld.minquotevol,
    accumulate::Int=Int(Ohlcv.ld.accumulate),
    checkperiod::Int=Int(Ohlcv.ld.checkperiod),
    threshold::Float64=Float64(Ohlcv.ld.startthreshold))::Bool
    rows = size(df, 1)
    rows == 0 && return false
    endix = min(_rowix_at_or_before(df[!, :opentime], datetime), rows)
    endix <= 0 && return false

    required = checkperiod + accumulate - 1
    endix < required && return false

    startix = endix - required + 1
    qv = if :quotevolume in propertynames(df)
        Float32.(df[startix:endix, :quotevolume])
    else
        @assert (:pivot in propertynames(df)) && (:basevolume in propertynames(df)) "OHLCV dataframe must include quotevolume or pivot+basevolume; names=$(names(df))"
        Float32.(df[startix:endix, :pivot] .* df[startix:endix, :basevolume])
    end
    accqv = 0.0f0
    insufficient = 0
    for ix in eachindex(qv)
        accqv += qv[ix]
        if ix > accumulate
            accqv -= qv[ix - accumulate]
        end
        if ix >= accumulate
            insufficient += accqv < minquotevol ? 1 : 0
        end
    end

    startnok = round(Int, checkperiod * threshold)
    return insufficient < startnok
end

function _ensure_marketview_ohlcv!(tc::TradeCache, base::AbstractString, startdt::DateTime, enddt::DateTime, loaded::Set)
    basekey = String(base)
    if basekey in loaded
        ohlcv = CryptoXch.ohlcv(tc.xc, base)
        CryptoXch.cryptoupdate!(tc.xc, ohlcv, startdt, enddt)
        return ohlcv
    end
    ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", startdt, enddt)
    push!(loaded, basekey)
    return ohlcv
end

function _ensure_marketview_ohlcv!(tc::TradeCache, base::AbstractString, startdt::DateTime, enddt::DateTime)
    loaded = Set{String}(String.(CryptoXch.bases(tc.xc)))
    return _ensure_marketview_ohlcv!(tc, base, startdt, enddt, loaded)
end

"Build a synthetic USDT market snapshot from OHLCV at `datetime` for simulation/backtest selection."
function _simulated_usdtmarketview(tc::TradeCache, datetime::DateTime, bases::Set{String}, history_startdt::DateTime)::DataFrame
    bases_sorted = sort!(collect(bases))
    loaded = Set{String}(String.(CryptoXch.bases(tc.xc)))
    basecoins = String[]
    quotevolumes = Float64[]
    pricechanges = Float32[]
    lastprices = Float32[]
    sizehint!(basecoins, length(bases_sorted))
    sizehint!(quotevolumes, length(bases_sorted))
    sizehint!(pricechanges, length(bases_sorted))
    sizehint!(lastprices, length(bases_sorted))

    for base in bases_sorted
        isempty(base) && continue
        CryptoXch.validbase(tc.xc, base) || continue
        ohlcv = _ensure_marketview_ohlcv!(tc, base, history_startdt, datetime, loaded)
        df = Ohlcv.dataframe(ohlcv)
        if size(df, 1) == 0
            continue
        end
        rowix = _rowix_at_or_before(df[!, :opentime], datetime)
        if rowix < 1
            continue
        end
        lastprice = Float32(df[rowix, :close])
        quotevolume24h = _rolling_quotevolume24h(df, rowix, datetime)
        pricechangepercent = _rolling_pricechangepercent24h(df, rowix, datetime)
        push!(basecoins, String(base))
        push!(quotevolumes, Float64(quotevolume24h))
        push!(pricechanges, Float32(pricechangepercent))
        push!(lastprices, Float32(lastprice))
    end

    if isempty(basecoins)
        return DataFrame(basecoin=String[], quotevolume24h=Float64[], pricechangepercent=Float32[], lastprice=Float32[])
    end
    return DataFrame(basecoin=basecoins, quotevolume24h=quotevolumes, pricechangepercent=pricechanges, lastprice=lastprices)
end

"""
Return the canonical tradelog partition root for the currently routed spot-trading venue.

The returned folder is scoped to the current environment, run mode, exchange,
account alias, asset class, and instrument type so ownership reconstruction does
not mix fills from different exchanges or market types.
"""
function _spot_tradelog_partition_root(cache::TradeCache)::String
    exchange_name = CryptoXch._routeexchange(cache.xc.routing, CryptoXch.trade_exchange_spot, CryptoXch.exchange(cache.xc))
    account_alias = exchange_name
    return joinpath(
        TradeLog.auditroot(),
        "environment=$(string(Symbol(EnvConfig.configmode)))",
        "run_mode=$(CryptoXch.tradelogrunmode(cache.xc))",
        "exchange=$(exchange_name)",
        "account=$(account_alias)",
        "asset_class=$(String(Symbol(TradeLog.crypto)))",
        "instrument_type=$(String(Symbol(TradeLog.spot_pair)))",
    )
end

"""Return one tradelog payload field as a normalized string."""
function _tradelogstring(event::AbstractDict, key::AbstractString)::String
    value = get(event, key, "")
    return (ismissing(value) || isnothing(value)) ? "" : String(value)
end

"""Return one tradelog payload field as `Float64`, defaulting to `0.0` when absent."""
function _tradelogfloat(event::AbstractDict, key::AbstractString)::Float64
    value = get(event, key, 0.0)
    if ismissing(value) || isnothing(value)
        return 0.0
    elseif value isa Real
        return Float64(value)
    end
    try
        return parse(Float64, String(value))
    catch
        return 0.0
    end
end

"""
Accumulate one filled tradelog event into directional robot-owned exposure.

- non-leveraged `Buy` increases long ownership and `Sell` decreases it
- leveraged `Sell` increases short ownership and leveraged `Buy` decreases it
"""
function _apply_robotowned_fill!(owned::Dict{String, NamedTuple{(:longqty, :shortqty), Tuple{Float32, Float32}}}, base::AbstractString, side::AbstractString, leverage::Real, fillqty::Real)
    fillqty <= 0 && return owned
    current = get(owned, uppercase(String(base)), (longqty=0f0, shortqty=0f0))
    longqty = Float32(current.longqty)
    shortqty = Float32(current.shortqty)
    qty = Float32(fillqty)
    sidekey = lowercase(String(side))
    if leverage > 0
        if sidekey == "sell"
            shortqty += qty
        elseif sidekey == "buy"
            shortqty = max(0f0, shortqty - qty)
        end
    else
        if sidekey == "buy"
            longqty += qty
        elseif sidekey == "sell"
            longqty = max(0f0, longqty - qty)
        end
    end
    owned[uppercase(String(base))] = (longqty=longqty, shortqty=shortqty)
    return owned
end

"""
Reconstruct directional robot-owned quantities per base from tradelog fills.

Only fills from the currently routed spot-trading tradelog partition are considered,
which keeps ownership separated by exchange/account scope and prevents cross-venue
mixing. Long and short quantities are tracked independently.
"""
function _robotownedqtymap(cache::TradeCache, bases)::Dict{String, NamedTuple{(:longqty, :shortqty), Tuple{Float32, Float32}}}
    wanted = Set(uppercase.(String.(bases)))
    owned = Dict{String, NamedTuple{(:longqty, :shortqty), Tuple{Float32, Float32}}}()
    isempty(wanted) && return owned

    partition_root = _spot_tradelog_partition_root(cache)
    isdir(partition_root) || return owned

    for (root, _, files) in walkdir(partition_root)
        "events.jsonl" in files || continue
        for event in TradeLog.readjsonlauditevents(joinpath(root, "events.jsonl"))
            event_type = uppercase(_tradelogstring(event, "event_type"))
            event_type in ["ORDER_PARTIAL_FILL", "ORDER_FILLED"] || continue
            _tradelogstring(event, "routing_role") == String(Symbol(TradeLog.routing_trade_exchange_spot)) || continue

            base = uppercase(_tradelogstring(event, "baseasset"))
            if isempty(base)
                pair = CryptoXch.basequote(_tradelogstring(event, "symbol"))
                base = isnothing(pair) ? "" : uppercase(String(pair.basecoin))
            end
            isempty(base) && continue
            base in wanted || continue

            side = _tradelogstring(event, "side")
            fillqty = _tradelogfloat(event, "fill_base_qty")
            leverage = _tradelogfloat(event, "leverage")
            _apply_robotowned_fill!(owned, base, side, leverage, fillqty)
        end
    end
    return owned
end

"""Ensure directional ownership columns exist in the current trade-selection DataFrame."""
function _ensure_robotownership_columns!(tc::TradeCache)
    if !hasproperty(tc.cfg, :robotownedlongqty)
        tc.cfg[:, :robotownedlongqty] = fill(0f0, size(tc.cfg, 1))
    end
    if !hasproperty(tc.cfg, :robotownedshortqty)
        tc.cfg[:, :robotownedshortqty] = fill(0f0, size(tc.cfg, 1))
    end
    return tc
end

"""Populate directional robot-owned quantities for the bases currently present in `tc.cfg`."""
function _annotate_robotownership!(tc::TradeCache)
    _ensure_robotownership_columns!(tc)
    bases = String.(tc.cfg[!, :basecoin])
    owned = _robotownedqtymap(tc, bases)
    tc.cfg[:, :robotownedlongqty] = [get(owned, uppercase(String(base)), (longqty=0f0, shortqty=0f0)).longqty for base in bases]
    tc.cfg[:, :robotownedshortqty] = [get(owned, uppercase(String(base)), (longqty=0f0, shortqty=0f0)).shortqty for base in bases]
    return tc
end

"""Return a mask indicating which trade-selection rows have robot-owned exposure."""
function _robotownedmask(cfg::AbstractDataFrame)
    return (cfg[!, :robotownedlongqty] .> 0f0) .|| (cfg[!, :robotownedshortqty] .> 0f0)
end

"""Synchronize `buyenabled` and `sellenabled` flags from the currently computed criteria columns."""
function _sync_tradeflags!(tc::TradeCache; assetonly::Bool=false)
    _ensure_robotownership_columns!(tc)
    if assetonly
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted]
        tc.cfg[:, :sellenabled] .= tc.cfg[!, :inportfolio]
    else
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :classifieraccepted] .&& tc.cfg[!, :minquotevol] .&& tc.cfg[!, :continuousminvol] .&& tc.cfg[!, :whitelisted]
        tc.cfg[:, :sellenabled] .= tc.cfg[:, :buyenabled] .|| tc.cfg[!, :inportfolio]
    end
    return tc
end

"""Return one optional numeric trade-selection field from a `DataFrameRow`."""
function _cfgfloat(row::DataFrameRow, field::Symbol, default::Float32=0f0)::Float32
    if !hasproperty(row, field)
        return default
    end
    value = getproperty(row, field)
    return (ismissing(value) || isnothing(value)) ? default : Float32(value)
end

"Return one optional boolean trade-selection field from a `DataFrameRow`."
function _cfgbool(row::DataFrameRow, field::Symbol, default::Bool=false)::Bool
    if !hasproperty(row, field)
        return default
    end
    value = getproperty(row, field)
    return (ismissing(value) || isnothing(value)) ? default : Bool(value)
end

"Log enriched diagnostics for Kraken margin order failures with expected vs available margin." 
function _log_margin_order_diagnostics(cache::TradeCache, basecfg::DataFrameRow, ta, base::AbstractString, side::AbstractString, requested_leverage::Signed, requested_limitprice::Union{Nothing, Real}, basequantity::Real, freebase::Real, borrowedbase::Real, freeusdt::Real, totalborrowedusdt::Real, effectivebudgetquote::Real, err)
    symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    additional_base = max(0.0, Float64(basequantity) - Float64(freebase))
    requested_limitprice_value = isnothing(requested_limitprice) ? missing : Float64(requested_limitprice)
    expected_margin_quote = isnothing(requested_limitprice) ? missing : (additional_base * Float64(requested_limitprice))
    limits = CryptoXch.marginlimits(cache.xc, symbol; role=CryptoXch.trade_exchange_spot)
    @error "margin order submission failed" exchange=CryptoXch.exchange(cache.xc) base=String(base) symbol=String(symbol) side=String(side) tradelabel=String(Symbol(ta.tradelabel)) requested_leverage=requested_leverage requested_baseqty=Float64(basequantity) requested_limitprice=requested_limitprice_value expected_margin_quote=expected_margin_quote available_free_quote=Float64(freeusdt) freebase=Float64(freebase) borrowedbase=Float64(borrowedbase) totalborrowedquote=Float64(totalborrowedusdt) effectivebudgetquote=Float64(effectivebudgetquote) buyenabled=_cfgbool(basecfg, :buyenabled, false) sellenabled=_cfgbool(basecfg, :sellenabled, false) inportfolio=_cfgbool(basecfg, :inportfolio, false) maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell error_message=sprint(showerror, err)
end

"Return true when an order error indicates exchange/account permission restrictions for the symbol."
function _ispermissionrestrictederror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("invalid permissions", msg) || occursin("trading restricted", msg) || occursin("permission denied", msg)
end

"Return true when an order error indicates temporary/per-position funding insufficiency."
function _isinsufficientfundserror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("insufficient funds", msg)
end

"Return true when a reduce-only close is rejected because no open position exists anymore."
function _isreduceonlynopositionerror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("reduce only", msg) && occursin("no position exists", msg)
end

"Return true when Kraken private-read cooldown/rate-limit transiently blocks order flow." 
function _isprivatecooldownerror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("private read cooldown", msg) || occursin("rate limit", msg)
end

"Extract cooldown-until timestamp from Kraken error text when present."
function _extractcooldownuntil(err)::Union{Nothing, DateTime}
    msg = sprint(showerror, err)
    m = match(r"until\s+([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?)", msg)
    if isnothing(m)
        return nothing
    end
    try
        return DateTime(m.captures[1])
    catch
        return nothing
    end
end

"Return true when managed close amend/recreate should be paused due to stale/cooldown/backlog conditions."
function _managed_maintenance_blocked(cache::TradeCache)::Bool
    get(cache.mc, :openorders_snapshot_stale, false) && return true
    until = get(cache.mc, :unmanaged_cancel_cooldown_until, nothing)
    if !isnothing(until) && (Dates.now(Dates.UTC) < until)
        return true
    end
    return get(cache.mc, :backlog_drain_mode, false)
end

"Return true when an order error indicates the target order no longer exists (race with fill/cancel)."
function _isunknownordererror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("unknown order", msg) || occursin("order not found", msg)
end

"Disable trading flags for one base in the current runtime config to avoid repeated restricted-order attempts."
function _disablerestrictedbase!(cache::TradeCache, base::AbstractString, reason::AbstractString)::Nothing
    base_upper = uppercase(String(base))
    restricted = get!(cache.mc, :restrictedcoins, String[])
    !(base_upper in restricted) && push!(restricted, base_upper)

    if !hasproperty(cache.cfg, :basecoin)
        (verbosity >= 1) && @warn "permission-restricted base cannot be removed from runtime config because :basecoin column is missing" base=base_upper reason=String(reason)
        return nothing
    end

    rowix = findfirst(==(base_upper), cache.cfg[!, :basecoin])
    if isnothing(rowix)
        (verbosity >= 1) && @warn "permission-restricted base not found in runtime config" base=base_upper reason=String(reason)
        return nothing
    end
    cache.cfg = cache.cfg[cache.cfg[!, :basecoin] .!= base_upper, :]
    try
        CryptoXch.removebase!(cache.xc, base_upper)
    catch err
        (verbosity >= 1) && @warn "failed removing restricted base from exchange cache" base=base_upper error=sprint(showerror, err)
    end
    try
        Classify.removebase!(cache.cl, base_upper)
    catch err
        (verbosity >= 1) && @warn "failed removing restricted base from classifier cache" base=base_upper error=sprint(showerror, err)
    end
    if _runtime_api_enabled(cache)
        rt = _strategyruntime(cache)
        if !isnothing(rt)
            try
                TradingStrategy.dropbase!(rt, base_upper)
            catch err
                (verbosity >= 1) && @warn "failed removing restricted base from strategy runtime" base=base_upper error=sprint(showerror, err)
            end
        end
    end
    (verbosity >= 1) && @warn "removed restricted base from trading universe" base=base_upper reason=String(reason)
    return nothing
end

"Return normalized set of base coins excluded from trading by runtime restrictions."
function _restrictedbaseset(tc::TradeCache, quotecoin::AbstractString)::Set{String}
    tokens = get(tc.mc, :restrictedcoins, String[])
    normalized = [_normalize_basecoin_token(x, quotecoin) for x in tokens]
    return Set(String.(filter(!isnothing, normalized)))
end

"Normalize a base/pair token to a base coin symbol for the configured quote coin."
function _normalize_basecoin_token(token, quotecoin::AbstractString)::Union{Nothing, String}
    q = uppercase(String(quotecoin))
    t = uppercase(strip(String(token)))
    isempty(t) && return nothing
    t == q && return nothing
    if occursin('/', t)
        parts = split(t, '/'; limit=2)
        length(parts) == 2 || return nothing
        base, quotetoken = parts
        quotetoken == q || return nothing
        base == q && return nothing
        return base
    end
    if endswith(t, q)
        base = t[1:(end - length(q))]
        isempty(base) && return nothing
        base == q && return nothing
        return base
    end
    return t
end

"Return a readiness state for tradable OHLCV data, optionally requesting exchange backfill up to `datetime`."
function _prepare_tradable_ohlcv!(tc::TradeCache, ohlcv::Ohlcv.OhlcvData; datetime::DateTime)
    if !Bool(get(tc.mc, :ohlcv_gap_backfill_on_tradable, false))
        return (state=:tradable, ready=true)
    end
    df = Ohlcv.dataframe(ohlcv)
    if size(df, 1) == 0
        return (state=:backfill_required, ready=false)
    end

    targetdt = floor(datetime, Minute(1))
    lastdt = floor(DateTime(df[end, :opentime]), Minute(1))
    needs_backfill = lastdt < targetdt
    if needs_backfill
        try
            CryptoXch.cryptoupdate!(tc.xc, ohlcv, lastdt, targetdt)
        catch err
            (verbosity >= 1) && @warn "tradable OHLCV backfill request failed" base=ohlcv.base lastdt=lastdt targetdt=targetdt error=sprint(showerror, err)
            return (state=:backfill_in_progress, ready=false)
        end
        df = Ohlcv.dataframe(ohlcv)
    end

    ready = (size(df, 1) > 0) && (floor(DateTime(df[end, :opentime]), Minute(1)) >= targetdt)
    if ready
        return (state=:data_ready, ready=true)
    end
    return (state=:backfill_in_progress, ready=false)
end

"Return current tracked tradable OHLCV state for `base` with restart-safe default."
function _tradable_ohlcv_state(tc::TradeCache, base::AbstractString)::Symbol
    statedict = get(tc.mc, :tradable_ohlcv_state_by_base, Dict{String, Symbol}())
    return get(statedict, uppercase(String(base)), :discovered)
end

"Persist current tradable OHLCV state transition for `base` and record transition timestamp."
function _set_tradable_ohlcv_state!(tc::TradeCache, base::AbstractString, state::Symbol; datetime::DateTime)
    key = uppercase(String(base))
    statedict = get(tc.mc, :tradable_ohlcv_state_by_base, Dict{String, Symbol}())
    dtdict = get(tc.mc, :tradable_ohlcv_state_dt_by_base, Dict{String, DateTime}())
    statedict[key] = state
    dtdict[key] = floor(DateTime(datetime), Minute(1))
    tc.mc[:tradable_ohlcv_state_by_base] = statedict
    tc.mc[:tradable_ohlcv_state_dt_by_base] = dtdict
    return state
end

"""
Loads all USDT coins, checks liquidity volume criteria, removes risk coins.
If isnothing(datetime) or datetime > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored.
`assetonly` is an input parameter to limit coins for backtesting.
"""
function tradeselection!(tc::TradeCache, assetbases::Vector; datetime=tc.xc.startdt, assetonly=false, updatecache=false)
    datetime = floor(datetime, Minute(1))
    quotecoin = uppercase(EnvConfig.cryptoquote)
    assetbase_tokens = [_normalize_basecoin_token(x, quotecoin) for x in assetbases]
    whitelist_tokens = [_normalize_basecoin_token(x, quotecoin) for x in tc.mc[:whitelistcoins]]
    assetbaseset = Set{String}(String.(filter(!isnothing, assetbase_tokens)))
    portfolioassetbaseset = copy(assetbaseset)
    whitelistset = Set{String}(String.(filter(!isnothing, whitelist_tokens)))
    restrictedset = _restrictedbaseset(tc, quotecoin)
    assetbaseset = setdiff(assetbaseset, restrictedset)
    whitelistset = setdiff(whitelistset, restrictedset)
    if !assetonly
        balancesdf = CryptoXch.balances(tc.xc; ignoresmallvolume=false)
        if size(balancesdf, 1) > 0
            hasfree = :free in names(balancesdf)
            haslocked = :locked in names(balancesdf)
            hasborrowed = :borrowed in names(balancesdf)
            for row in eachrow(balancesdf)
                base = _normalize_basecoin_token(row.coin, quotecoin)
                isnothing(base) && continue
                freeqty = hasfree ? Float64(row.free) : 0.0
                lockedqty = haslocked ? Float64(row.locked) : 0.0
                borrowedqty = hasborrowed ? Float64(row.borrowed) : 0.0
                if (abs(freeqty) + abs(lockedqty) + abs(borrowedqty)) > 0.0
                    push!(portfolioassetbaseset, String(base))
                end
            end
        end
        # Keep restricted held bases in inportfolio to allow close/monitor flows.
        # Restricted filtering is still applied to non-portfolio candidate expansion.
    end
    history_minutes = _tradeselection_history_minutes(tc)
    history_startdt = datetime - Minute(history_minutes)

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    # CryptoXch.removeallbases(tc.xc)  #* reuse what is in cache
    # Classify.removebase!(tc.cl, nothing)  #* reuse what is in cache

    marketbases = assetonly ? Set(String.(collect(portfolioassetbaseset))) : Set(String.(collect(union(portfolioassetbaseset, whitelistset, Set(String.(CryptoXch.bases(tc.xc)))))))
    marketbases = union(portfolioassetbaseset, setdiff(marketbases, restrictedset))
    if _uses_simulated_marketview(tc)
        usdtdf = _simulated_usdtmarketview(tc, datetime, marketbases, history_startdt)
        if size(usdtdf, 1) == 0
            requestedbases = filter(!=(quotecoin), collect(marketbases))
            error("empty simulated marketview at datetime=$(datetime), requestedbases=$(requestedbases). Check OHLCV availability for configured bases.")
        end
    else
        usdtdf = CryptoXch.screeningUSDTmarket(tc.xc; dt=datetime)  # superset of coins with 24h volume price change and last price
        if size(usdtdf, 1) == 0
            usdtdf = _wait_for_live_usdtmarket!(tc, datetime; requestedbases=collect(marketbases))
        end
        if assetonly
            usdtdf = filter(row -> row.basecoin in assetbaseset, usdtdf)
        end
    end
    if !isempty(restrictedset) && (size(usdtdf, 1) > 0)
        usdtdf = filter(row -> !((String(row.basecoin) in restrictedset) && !(String(row.basecoin) in portfolioassetbaseset)), usdtdf)
    end
    if !assetonly
        knownbases = Set(String.(usdtdf[!, :basecoin]))
        missingportfoliobases = setdiff(portfolioassetbaseset, knownbases)
        if !isempty(missingportfoliobases)
            valuationdf = CryptoXch.valuationUSDTmarket(tc.xc, collect(missingportfoliobases); dt=datetime)
            for row in eachrow(valuationdf)
                base = String(row.basecoin)
                if ((base in restrictedset) && !(base in portfolioassetbaseset)) || (base in knownbases)
                    continue
                end
                push!(usdtdf, (
                    basecoin=base,
                    quotevolume24h=Float32(row.quotevolume24h),
                    pricechangepercent=Float32(row.pricechangepercent),
                    lastprice=Float32(row.lastprice),
                    askprice=Float32(row.askprice),
                    bidprice=Float32(row.bidprice),
                ))
                push!(knownbases, base)
            end
        end
    end
    (verbosity >= 3) && println("USDT market of size=$(size(usdtdf, 1)) at $datetime")
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent, :lastprice)
    if size(tc.cfg, 1) == 0
        tc.cfg[:, :datetime] = DateTime[]
        tc.cfg[:, :minquotevol] = Bool[]
        tc.cfg[:, :continuousminvol] = Bool[]
        tc.cfg[:, :ohlcvstate] = Symbol[]
        tc.cfg[:, :ohlcvready] = Bool[]
        tc.cfg[:, :inportfolio] = Bool[]
        tc.cfg[:, :classifieraccepted] = Bool[]
        tc.cfg[:, :robotownedlongqty] = Float32[]
        tc.cfg[:, :robotownedshortqty] = Float32[]
        tc.cfg[:, :buyenabled] = Bool[]
        tc.cfg[:, :sellenabled] = Bool[]
        tc.cfg[:, :whitelisted] = Bool[]
        (verbosity >= 1) && @warn "no basecoins selected - empty result tc.cfg=$(tc.cfg)"
        return tc
    end
    tc.cfg[:, :datetime] .= datetime
    # tc.cfg[:, :validbase] = [CryptoXch.validbase(tc.xc, base) for base in tc.cfg[!, :basecoin]] # is already filtered by getUSDTmarket
    minimumdayquotevolumemillion = round(Ohlcv.liquiddailyminimumquotevolume() / 1000000, digits=0) # ignore allcoins with less than liquiddailyminimumquotevolume
    tc.cfg[:, :minquotevol] = tc.cfg[:, :quotevolume24h_M] .>= minimumdayquotevolumemillion
    tc.cfg[:, :continuousminvol] .= false
    tc.cfg[:, :ohlcvstate] = [_tradable_ohlcv_state(tc, base) for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :ohlcvready] = falses(size(tc.cfg, 1))
    tc.cfg[:, :inportfolio] = [base in portfolioassetbaseset for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :classifieraccepted] .= false
    tc.cfg[:, :robotownedlongqty] = fill(0f0, size(tc.cfg, 1))
    tc.cfg[:, :robotownedshortqty] = fill(0f0, size(tc.cfg, 1))
    tc.cfg[:, :buyenabled] .= false
    tc.cfg[:, :sellenabled] .= false
    tc.cfg[:, :whitelisted] = [base in whitelistset for base in tc.cfg[!, :basecoin]]
    _annotate_robotownership!(tc)

    # download latest OHLCV and classifier features
    tc.cfg = tc.cfg[tc.cfg[:, :minquotevol] .|| tc.cfg[:, :inportfolio], :]
    (verbosity >= 3) && println("#minquotevol=$(sum(tc.cfg[:, :minquotevol])) #inportfolio=$(sum(tc.cfg[:, :inportfolio]))")
    count = size(tc.cfg, 1)
    xcbases = CryptoXch.bases(tc.xc)
    removebases = setdiff(xcbases, tc.cfg[!, :basecoin])
    for rb in removebases  # remove coins that were loaded but are no longer part of the new configuration
        CryptoXch.removebase!(tc.xc, rb)
        if _runtime_api_enabled(tc)
            rt = _strategyruntime(tc)
            !isnothing(rt) && TradingStrategy.dropbase!(rt, rb)
        else
            Classify.removebase!(tc.cl, rb)
        end
    end
    xcbaseset = Set(CryptoXch.bases(tc.xc))
    candidatebaseset = Set{String}()
    (verbosity >= 3) && println("trade selection history window=$(history_minutes) minutes from $(history_startdt) to $(datetime)")
    for (ix, row) in enumerate(eachrow(tc.cfg))
        (verbosity >= 2) && updatecache &&  print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) including cache update                           ")
        (verbosity >= 2) && !updatecache && print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) without cache update                             ")
        if row.basecoin in xcbaseset
            ohlcv = CryptoXch.ohlcv(tc.xc, row.basecoin)
            CryptoXch.cryptoupdate!(tc.xc, ohlcv, history_startdt, datetime)
        else
            ohlcv = CryptoXch.cryptodownload(tc.xc, row.basecoin, "1m", history_startdt, datetime)
        end
        tradable_state = _prepare_tradable_ohlcv!(tc, ohlcv; datetime=datetime)
        if updatecache
            Ohlcv.write(ohlcv) # write ohlcv even if data length is too short to calculate features
        end
        row.ohlcvstate = tradable_state.state
        row.ohlcvready = tradable_state.ready
        _set_tradable_ohlcv_state!(tc, row.basecoin, tradable_state.state; datetime=datetime)
        row.continuousminvol = row.ohlcvready # TODO re-enable continuous-liquidity check after gap readiness is stable
        if row.ohlcvready && (row.inportfolio || (row.whitelisted && row.minquotevol))
            push!(candidatebaseset, String(row.basecoin))
        end
    end

    # Keep classifier/feature workload limited to liquidity candidates and portfolio holdings.
    for rb in setdiff(Set(CryptoXch.bases(tc.xc)), candidatebaseset)
        CryptoXch.removebase!(tc.xc, rb)
        if _runtime_api_enabled(tc)
            rt = _strategyruntime(tc)
            !isnothing(rt) && TradingStrategy.dropbase!(rt, rb)
        else
            Classify.removebase!(tc.cl, rb)
        end
    end

    selectedbases = String[]
    if _runtime_api_enabled(tc)
        rt = _strategyruntime(tc)
        if !isnothing(rt)
            TradingStrategy.preparebases!(rt, tc.xc, collect(candidatebaseset); history_startdt=history_startdt, datetime=datetime, updatecache=updatecache)
            selectedset = Set(String.(TradingStrategy.acceptedbases(rt)))

            xcbases = CryptoXch.bases(tc.xc)
            remove_xc_bases = setdiff(xcbases, selectedset)
            for rb in remove_xc_bases  # remove coins not accepted by strategy runtime (e.g. insufficient requiredminutes)
                CryptoXch.removebase!(tc.xc, rb)
            end

            remove_runtime_bases = setdiff(selectedset, Set(String.(CryptoXch.bases(tc.xc))))
            for rb in remove_runtime_bases
                TradingStrategy.dropbase!(rt, rb)
            end

            xcbases = CryptoXch.bases(tc.xc)
            selectedset = Set(String.(TradingStrategy.acceptedbases(rt)))
            @assert Set(xcbases) == selectedset "Set(xcbases)=$(xcbases) != Set(selectedbases)=$(collect(selectedset))"
            selectedbases = sort!(collect(selectedset))
            tc.cfg[:, :classifieraccepted] = [base in selectedset for base in tc.cfg[!, :basecoin]]
            for row in eachrow(tc.cfg)
                if Bool(row.classifieraccepted)
                    row.ohlcvstate = :tradable
                    row.ohlcvready = true
                    _set_tradable_ohlcv_state!(tc, row.basecoin, :tradable; datetime=datetime)
                end
            end
        end
    else
        classifierloadedset = Set(String.(Classify.bases(tc.cl)))
        for row in eachrow(tc.cfg)
            base = String(row.basecoin)
            if (base in candidatebaseset) && !(base in classifierloadedset)
                Classify.addbase!(tc.cl, CryptoXch.ohlcv(tc.xc, base))
                push!(classifierloadedset, base)
            end
        end

        if !isempty(classifierloadedset)
            Classify.supplement!(tc.cl)
            if updatecache
                Classify.writetargetsfeatures(tc.cl)
            end
        end
        xcbases = CryptoXch.bases(tc.xc)
        classifierbases = Classify.bases(tc.cl)
        remove_xc_bases = setdiff(xcbases, classifierbases)
        for rb in remove_xc_bases  # remove coins not accepted by classifier (e.g. insufficient requiredminutes)
            CryptoXch.removebase!(tc.xc, rb)
        end
        remove_classifier_bases = setdiff(classifierbases, xcbases)
        for rb in remove_classifier_bases  # drop stale classifier-only bases that are no longer in the exchange cache
            Classify.removebase!(tc.cl, rb)
        end
        xcbases = CryptoXch.bases(tc.xc)
        classifierbases = Classify.bases(tc.cl)
        classifierbaseset = Set(classifierbases)
        @assert Set(xcbases) == classifierbaseset "Set(xcbases)=$(xcbases) != Set(classifierbases)=$(classifierbases)"

        selectedbases = String.(classifierbases)
        tc.cfg[:, :classifieraccepted] = [base in classifierbaseset for base in tc.cfg[!, :basecoin]]
        for row in eachrow(tc.cfg)
            if Bool(row.classifieraccepted)
                row.ohlcvstate = :tradable
                row.ohlcvready = true
                _set_tradable_ohlcv_state!(tc, row.basecoin, :tradable; datetime=datetime)
            end
        end
    end
    _sync_tradeflags!(tc; assetonly=assetonly)
    # (verbosity >= 2) && _log_cachecfg_summary!(tc, "tradeselection")
    (verbosity >= 2) && println("$(CryptoXch.ttstr(tc.xc)) result of tradeselection! $(tc.cfg)")
    # tc.cfg = tc.cfg[(tc.cfg[!, :buyenabled] .|| tc.cfg[:, :sellenabled]), :]
    (verbosity >= 2) && println("$(EnvConfig.now()) #tc.cfg=$(size(tc.cfg, 1)) sum(classifieraccepted)=$(sum(tc.cfg[!, :classifieraccepted])) selectedbases($(length(selectedbases)))=$(selectedbases) ")

    if !assetonly
        (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) trained trade config on the fly including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    end
    return tc
end

"Adds usdtprice and usdtvalue added as well as the portfolio dataframe to trade config and returns trade config and portfolio as tuple"
function addassetsconfig!(tc::TradeCache, assets=CryptoXch.portfolio!(tc.xc))
    sort!(assets, [:coin])  # for readability only

    prev_inportfolio = hasproperty(tc.cfg, :inportfolio) ? Vector{Bool}(tc.cfg[!, :inportfolio]) : fill(false, size(tc.cfg, 1))
    tc.cfg = leftjoin(tc.cfg, assets, on = :basecoin => :coin)
    tc.cfg = tc.cfg[!, Not([:borrowed, :accruedinterest, :locked, :free])]
    joined_inportfolio = .!ismissing.(tc.cfg[:, :usdtvalue])
    tc.cfg[:, :inportfolio] = prev_inportfolio .|| joined_inportfolio
    _annotate_robotownership!(tc)
    _sync_tradeflags!(tc)
    sort!(tc.cfg, [:basecoin])  # for readability only
    sort!(tc.cfg, rev=true, [:buyenabled])  # for readability only
    return tc.cfg, assets
end

"Returns the current TradeConfig dataframe with usdtprice and usdtvalue added as well as the portfolio dataframe as a tuple."
function assetsconfig!(tc::TradeCache, datetime=nothing)
    dt = isnothing(datetime) ? Dates.now(UTC) : floor(datetime, Minute(1))
    assets = CryptoXch.portfolio!(tc.xc)
    tradeselection!(tc, assets[!, :coin]; datetime=dt)
    return addassetsconfig!(tc)
end

significantsellpricechange(tc, orderprice) = abs(tc.sellprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2
significantbuypricechange(tc, orderprice) = abs(tc.buyprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2


currenttime(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :opentime]
currentprice(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :close]
closelongset = [shortstrongbuy, shortbuy, shorthold, allclose, longstrongclose, longclose]
closeshortset = [shortclose, shortstrongclose, allclose, longhold, longbuy, longstrongbuy]

mutable struct Investment  #TODO bookkeeping for consistency checks
    investmentid
    tradeadvice::Vector{Classify.TradeAdvice} # vector of all used trade advices
    orderid::Vector # vector of all used orders
    classifiername
    configid
end

"Trade execution advice emitted by strategy handling inside Trade."
Base.@kwdef mutable struct StrategyAdvice
    classifier::Classify.AbstractClassifier
    configid = 0
    tradelabel::Targets.TradeLabel = ignore
    relativeamount::Float32 = 1f0
    base::String
    price::Union{Nothing, Float32} = nothing
    datetime::DateTime
    hourlygain::Float32 = 0f0
    probability::Float32 = 0f0
    investmentid = nothing
    source::Symbol = :classifier
    allowreversal::Bool = true
end

function _strategyadvice(ta::Classify.TradeAdvice; tradelabel::Targets.TradeLabel=ta.tradelabel, limitprice::Union{Nothing, Real}=nothing, source::Symbol=:classifier, allowreversal::Bool=true)
    lp = isnothing(limitprice) ? nothing : Float32(limitprice)
    return StrategyAdvice(
        classifier=ta.classifier,
        configid=ta.configid,
        tradelabel=tradelabel,
        relativeamount=Float32(ta.relativeamount),
        base=String(ta.base),
        price=lp,
        datetime=ta.datetime,
        hourlygain=Float32(ta.hourlygain),
        probability=Float32(ta.probability),
        investmentid=ta.investmentid,
        source=source,
        allowreversal=allowreversal,
    )
end

function _strategyadvice(ta::StrategyAdvice; tradelabel::Targets.TradeLabel=ta.tradelabel, limitprice::Union{Nothing, Real}=ta.price, source::Symbol=ta.source, allowreversal::Bool=ta.allowreversal)
    lp = isnothing(limitprice) ? nothing : Float32(limitprice)
    return StrategyAdvice(
        classifier=ta.classifier,
        configid=ta.configid,
        tradelabel=tradelabel,
        relativeamount=Float32(ta.relativeamount),
        base=String(ta.base),
        price=lp,
        datetime=ta.datetime,
        hourlygain=Float32(ta.hourlygain),
        probability=Float32(ta.probability),
        investmentid=ta.investmentid,
        source=source,
        allowreversal=allowreversal,
    )
end

function dbgrow(cache::TradeCache, ta)
    return (
        taconfigid = ta.configid,
        tatradelabel = ta.tradelabel,
        tabase = ta.base,
        tahourlygain = ta.hourlygain,
        baseqty = missing,
        minimumbasequantity = missing,
        freebase = missing,
        totalborrowedusdt = missing,
        freeusdt = missing,
        quoteqty = missing,
    )
end

"Adds a dataframe trade advice row "
function _traderow!(df, cache; basecoin="XXX", tradelabel=allclose, hourlygain=0f0, probability=1f0, relativeamount=1f0, investmentid="XXX", price=0f0, datetime=cache.xc.currentdt, classifier=cache.cl, configid=0, oid="", enforced="n/a")
    hourlygain = min(hourlygain, cache.mc[:hourlygainlimit])
    push!(df, (basecoin=basecoin, tradelabel=tradelabel, hourlygain=hourlygain, probability=probability, relativeamount=relativeamount, investmentid=investmentid, price=price, datetime=datetime, classifier=classifier, configid=configid, oid=oid, enforced=enforced))
    return last(df)
end

"Adds a dataframe trade advice row based ona trade advice input"
_tradeadvice2df!(df, cache::TradeCache, ta) = _traderow!(df, cache, basecoin=ta.base, tradelabel=ta.tradelabel, hourlygain=ta.hourlygain, probability=ta.probability, relativeamount=ta.relativeamount, investmentid=ta.investmentid, price=ta.price, datetime=ta.datetime, classifier=ta.classifier, configid=ta.configid)

"Adds enforced trade advics according to black lists and enforced trades constraints"
function _forcetradelabel!(df::DataFrame, cache::TradeCache, coins, tradelabel, hourlygain, enforced)
    for base in coins
        rowix = findfirst(x -> x == base, df[!, :basecoin])
        if isnothing(rowix)
            if base in CryptoXch.bases(cache.xc)
                _traderow!(df, cache, basecoin=base, tradelabel=tradelabel, probability=1f0, relativeamount=1f0, hourlygain=hourlygain, enforced=enforced)
            end
        else
            df[rowix, :tradelabel] = tradelabel
            df[rowix, :investmentid] = trade_enforced
        end
    end
end

_isclosetrade(tl) = tl in [shortclose, shortstrongclose, allclose, longstrongclose, longclose]
_isopentrade(tl) = tl in [shortstrongbuy, shortbuy, longbuy, longstrongbuye]
_isopenshorttrade(tl) = tl in [shortstrongbuy, shortbuy]

"""
Creates dataframe from trade advice vector plus corresponding asset info and adds/changes rows to enforce trades,  
i.e. add trades for enforced long open and short open and long/short exits, removes black listed coins
"""
function policyenforcement(cache::TradeCache, tavec::Vector{Classify.TradeAdvice}, assets::AbstractDataFrame)
    df = DataFrame()
    _traderow!(df, cache)  # create columns
    pop!(df)  # remove dummy row
    if cache.mc[:trademode] == quickexit
        for base in assets[!, :coin]
            uppercase(String(base)) == uppercase(EnvConfig.cryptoquote) && continue
            _traderow!(df, cache, basecoin=base, tradelabel=allclose, enforced="quickexit")
        end
    else # no quick exit
        # don't check against other trade modes to enable debugging of tradeamount()
        for ta in tavec
            if !(ta.base in CryptoXch.baseignore)
                #TODO baseignore -> invalid symbol
                #TODO noinvest => buyenabled = false
                rowix = findfirst(row -> row.basecoin == ta.base, cache.cfg)
                if !isnothing(rowix) 
                    if cache.cfg[rowix, :sellenabled] && _isclosetrade(ta.tradelabel)
                        _tradeadvice2df!(df, cache, ta)
                    elseif cache.cfg[rowix, :buyenabled] && _isopentrade(ta.tradelabel)
                        _tradeadvice2df!(df, cache, ta)
                    # else allhold tradeadvices are skipped
                    end
                end
            end
        end
        _forcetradelabel!(df, cache, cache.mc[:longopencoins], longstrongbuy, 1f0, "longopen")
        _forcetradelabel!(df, cache, cache.mc[:shortopencoins], shortstrongbuy, -1x0, "shortopen")
        _forcetradelabel!(df, cache, cache.mc[:exitcoins], allclose, 0f0, "exit")
    end
    if size(df, 1) > 0
        leftjoin!(df, assets, on = :basecoin => :coin)
    end
    return df
end

"Distributes the available quote assets across all open trades and returns result in quoteamount"
function _tradeamounts!(tadf)
    tadf.quoteamount[_isclosetrade(tadf.tradelabel)] .= abs.(tadf.free .* tadf.usdtprice) # add close amounts (which are not constrained by free quote)

    freequote = sum(tadf[tadf[!, :basecoin] .== EnvConfig.cryptoquote, :free])
    maxassetquote = sum(tadf[!, :usdtvalue]) * cache.mc[:maxassetfraction]
    opentradeweights = abs(tadf[!, :hourlygain] .* tadf[!, :probability] .* tadf[!, :relativeamount])
    opentradeweights = opentradeweights ./ sum(opentradeweights)  # normalize weights that they add up to 1
    tadf.quoteamount[_isopentrade(tadf.tradelabel)] .= opentradeweights .* freequote   # open trades have positive (long) or negative (short) amount, close and hold trades have zero amount
    tadf.quoteamount[_isopentrade(tadf.tradelabel)] .= min.(tadf[!, :quoteamount], maxassetquote)  # limit max amount of trade
    tadf.quoteamount[_isopentrade(tadf.tradelabel)] .= tadf[!, :quoteamount] .- abs.(tadf[!, :free]) .* tadf[!, :usdtprice]  # deduct already opened amount
    reduceamount = _isopentrade(tadf.tradelabel) .&& tadf[!, :quoteamount] .< (-0.1 .* abs.(tadf[!, :free]) .* tadf[!, :usdtprice])  # reduce if current open position is >10% larger than target
    if any(reduceamount)  # instead of open -> change to close with the amount that is above target volume
        tadf.tradelabel[reduceamount] .= allclose
        tadf.quoteamount[reduceamount] .= abs.(trade.quoteamount)
    end
    tadf.quoteamount[_isopentrade(tadf.tradelabel) .&& (tadf[!, :quoteamount] .< 0f0)] .= 0f0  # those who have a reduce amount less than 10% will be ignored

    subset!(tadf, [:quoteamount, :minquoteqty] => (amt, minq) -> abs.(amt) .> minq) # remove alltrades below level threshold
end

_traderank(tl) = _isclosetrade(tl) ? 1 : _isopentrade(tl) ? 2 : 3

function _tradetolabeltext(label)
    return String(Symbol(label))
end

function _withtradelogcontext(f::Function, cache::TradeCache, ta)
    signal_score = try
        Float64(ta.probability)
    catch
        missing
    end
    strategy_engine = String(Symbol(_strategyengine(cache)))
    strategy_ref = string(get(cache.mc, :strategy_source, "default"))
    CryptoXch.settradelogcontext!(
        cache.xc;
        strategy_engine=strategy_engine,
        strategy_config_ref=strategy_ref,
        signal_label=_tradetolabeltext(ta.tradelabel),
        signal_score=signal_score,
    )
    try
        return f()
    finally
        CryptoXch.cleartradelogcontext!(cache.xc)
    end
end

"""
Provides the amount that should be used for the tradeadvice including all considerations in a dataframe.

  - take out all not tradable coins from the trade advice set
    - quotecoins
    - black listed coins
  - calculate the overall amount that can be spend now based on free available portfolio budget
  - sort remaining trade advices according to expected hourly gain
  - determine overall investment for that basecoin 
    - it should not dominate due to risk and hold back a head room part (e.g. 5%) of free usdt
    - consider the hourly gain when calculating the basecoin specific amount
    - reduce investments according to overall investment amount if delta is more than 10% (hysteresis) if advice is buy or hold
  - close investments if 
    - hourly gain of current investments are x% (e.g. 50%) less gain than best investments and those invested coins above that limit are not at dominating limit
    - coin is above dominating investment limit
    - quickexit for specific or all coins
  - split investment in chunks of reasoable size
  - if one chunk is smaller that minimum limits of exchange then merge them if possible
  - too small amounts cannot be traded if they are below exchange limits
  - insuffient free coins will also prevent trading
  - margin trades should only be done if borrowed amount is covered by free quotecoin
  - same amounts are applicable for margin and spot trading, i.e. buy amount also applies to a margin sell without basecoin assets (short buy)

  Returned dataframe should include (all amounts in usdt to better compare magnitudes)

  - basecoin, currentcloseprice, totalwallet, totalbasecoin, minquoteqty, minbaseqty, maxtotalbasecoin, maxbuyamount, maxsellamount, targetchunksize, buyamount, sellamount

  ## Short trades:

  - marginfree and marginlocked can be negative
  - the total absolute value is sum(abs.(marginfree, marginlocked), free, locked, - borrowed per coin)
  - there should be no case of free / locked positive and marginfree / marginlocked negative because there are balanced with each other -> save to add all absolute amounts by abs.(df)
"""
function tradeamount(cache::TradeCache, tavec::Vector{Classify.TradeAdvice}, assets::AbstractDataFrame) #TODO consider negative short amounts
    tadf = policyenforcement(cache, tavec, assets) # returns a dataframe with tradeadvice per line plus corresponding asset info
    if sze(tadf, 1) > 0
        tadf.minquoteqty = [minimumquotequantity(xc, base) for base in tadf[!, :basecoin]]
        _tradeamounts!(tadf)
        sort!(tadf, [order(:tradelabel, by=_traderank), order(:hourlygain, rev=true)])  # order such that close before open and high before low hourlygain
        ta.basetradeqty .=0f0
        ta.vol1hmedian .= 0f0
        ta.baseamount .= 0f0
        for tarow in tadf
            base = String(getproperty(tarow, :basecoin))
            ohlcv = _cachedohlcv(cache, base)
            isnothing(ohlcv) && continue
            price = currentprice(ohlcv)
            tarow.baseamount = tarow.quoteamount / price
        
            tarow.basevol1hmedian = Ohlcv.dataframe(ohlcv)[Ohlcv.rowix(ohlcv, cache.xc.currentdt-Hour(1)):Ohlcv.rowix(ohlcv, cache.xc.currentdt), :basevolume]
            tarow.basetradeqty = max(tarow.baseamount, median(tarow.basevol1hmedian)/2) # don't trade more than 50% of the houry minute median
        end
    end
    return tadf
end


"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, tadf::DataFrameRow)
    for ta in eachrow(tadf)
        if (ta.tradelabel in [longbuy, longstrongbuy]) && (cache.mc[:trademode] == buysell)
            oid = (cache.mc[:trademode] == notrade) ? "BuySpotSim" : CryptoXch.createbuyorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=ta.basetradeqty, maker=true, marginleverage=0)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        elseif (ta.tradelabel in [longstrongclose, longclose]) && (cache.mc[:trademode] in [buysell, closeonly, quickexit])
            oid = (cache.mc[:trademode] == notrade) ? "SellSpotSim" : CryptoXch.createsellorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=basequta.basetradeqtyantity, maker=true, marginleverage=0)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        elseif (ta.tradelabel in [shortclose, shortstrongclose]) && (cache.mc[:trademode] in [buysell, closeonly, quickexit]) && basecfg.sellenabled
            oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : CryptoXch.createbuyorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=ta.basetradeqty, maker=true, marginleverage=2, reduceonly=true)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        elseif (ta.tradelabel in [shortstrongbuy, shortbuy]) && (cache.mc[:trademode] == buysell)
            oid = (cache.mc[:trademode] == notrade) ? "SellMarginSim" : CryptoXch.createsellorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=ta.basetradeqty, maker=true, marginleverage=2)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        end
        push!(cache.dbgdf, ta, promote=true)
    end
end

"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, basecfg::DataFrameRow, ta::Classify.TradeAdvice, assets::AbstractDataFrame)
    return trade!(cache, basecfg, _strategyadvice(ta; source=:classifier), assets)
end

function _requested_limitprice(cache::TradeCache, ta::StrategyAdvice, fallback_price::Real)
    if ta.tradelabel in [longstrongbuy, shortstrongbuy, longstrongclose, shortstrongclose]
        return nothing
    end
    if (ta.tradelabel in [longbuy, shortbuy]) && (_strategyengine(cache) == :getgainsalgo)
        strategy_buygain = Float32(get(cache.mc, :strategy_buygain, 0f0))
        if abs(strategy_buygain) <= eps(Float32)
            return nothing
        end
    end
    return isnothing(ta.price) ? _orderlimitprice(cache, fallback_price) : Float32(ta.price)
end

"Return relative threshold for amending limit prices; default is 0.01% (1e-4)."
function _order_amend_price_rel_threshold(cache::TradeCache)::Float32
    raw = get(cache.mc, :orderamendpricerelthreshold, ORDER_AMEND_PRICE_REL_THRESHOLD_DEFAULT)
    th = Float32(raw)
    return (isfinite(th) && th >= 0f0) ? th : ORDER_AMEND_PRICE_REL_THRESHOLD_DEFAULT
end

"Return true when limit price or base quantity changed enough to justify amend/recreate."
function _material_order_change(existing_limit, requested_limit, existing_qty::Real, requested_qty::Real; price_reltol::Float32=ORDER_AMEND_PRICE_REL_THRESHOLD_DEFAULT, qty_reltol::Float32=1f-3, abstol::Float32=1f-6)
    if isnothing(existing_limit) != isnothing(requested_limit)
        return true
    end
    if !isnothing(existing_limit)
        oldp = Float32(existing_limit)
        newp = Float32(requested_limit)
        pdiff = abs(oldp - newp)
        pscale = max(abs(oldp), abs(newp), 1f0)
        if pdiff > max(abstol, price_reltol * pscale)
            return true
        end
    end
    oldq = Float32(existing_qty)
    newq = Float32(requested_qty)
    qdiff = abs(oldq - newq)
    qscale = max(abs(oldq), abs(newq), 1f0)
    return qdiff > max(abstol, qty_reltol * qscale)
end

function trade!(cache::TradeCache, basecfg::DataFrameRow, ta::StrategyAdvice, assets::AbstractDataFrame)
    sellbuyqtyratio = 2 # longclose qty / longbuy qty per order, if > 1 longclose quicker than buying it
    qtyacceleration = 4 # if > 1 then increase longbuy and longclose order qty by this factor
    short_margin_leverage = 2
    result = nothing
    base = ta.base
    accountcap = CryptoXch.accountcapacity(cache.xc)
    equityquote = Float64(get(accountcap, :equity_quote, 0.0))
    if equityquote <= 0
        totalusdt = sum(assets.usdtvalue)
        @warn "equityquote=$equityquote is insufficient, totalusdt=$totalusdt, assets=$assets"
        return nothing
    end
    effectivebudgetquote = _effectivebudgetquote(cache, assets)
    if effectivebudgetquote <= 0
        (verbosity > 2) && println("$(tradetime(cache)) skip $base: effectivebudgetquote=$effectivebudgetquote is insufficient")
        return nothing
    end
    allocatedbudgetquote = _allocatedbudgetquote(assets)
    remainingbudgetquote = max(0.0, effectivebudgetquote - allocatedbudgetquote)
    overallocatedbudgetquote = max(0.0, allocatedbudgetquote - effectivebudgetquote)
    basequantity = missing
    freeusdtfractionmargin = 0.05
    totalborrowedusdt = sum(assets[!, :borrowed] .* assets[!, :usdtprice])
    freeusdt = max(0.0, Float64(get(accountcap, :available_long_quote, 0.0)))
    freeshortquote = max(0.0, Float64(get(accountcap, :available_short_quote, freeusdt)))
    freebase = sum(assets[assets[!, :coin] .== base, :free]) *(1-eps(Float32))
    lockedbase = sum(assets[assets[!, :coin] .== base, :locked])
    borrowedbase = sum(assets[assets[!, :coin] .== base, :borrowed])
    maxassetquote = cache.mc[:maxassetfraction] * effectivebudgetquote
    quotequantity = cache.mc[:maxassetfraction] * effectivebudgetquote / 10  # distribute over 10 trades within effective budget
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    price = currentprice(ohlcv)
    longexposurequote = max(0f0, (freebase + lockedbase) * price)
    shortexposurequote = max(0f0, borrowedbase * price)
    symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    @assert base == ohlcv.base == ta.base
    minimumbasequantity = CryptoXch.minimumbasequantity(cache.xc, base, price)
    if isnothing(minimumbasequantity)
        (verbosity > 2) && println("$(tradetime(cache)) skip $base due to missing minimum base quantity at price=$price")
        return nothing
    end
    # (verbosity > 2) && println("$(tradetime(cache)) entry $base , $(ta.tradelabel)")
    # CryptoXch.portfolio subtracts the borrowed amount from usdtvalue of each base
    if (cache.mc[:trademode] == quickexit) || (base in cache.mc[:exitcoins])
        ta.tradelabel = allclose
    end
    if (ta.tradelabel in [allclose, longhold]) && (borrowedbase > 0)
        ta.tradelabel = shortclose
    end
    if (ta.tradelabel in [allclose, shorthold]) && (freebase > 0)
        ta.tradelabel = longclose
    end
    if (ta.tradelabel in [allclose, shorthold, longhold])
        return nothing
    end
    if (ta.tradelabel in [longstrongclose, longclose]) && (cache.mc[:trademode] in [buysell, closeonly, quickexit]) && basecfg.sellenabled
        existing = _managedcloseget(cache, base, longclose)
        existing_orderid = isnothing(existing) ? nothing : String(existing[:orderid])
        if !isnothing(existing)
            inherited = Bool(get(existing, :inherited, false))
            existing_tif = uppercase(String(get(existing, :timeinforce, "")))
            if (ta.tradelabel == longstrongclose) && inherited && (existing_tif != "POSTONLY")
                try
                    CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
                catch err
                    (verbosity >= 1) && @warn "failed to cancel inherited non-PostOnly close order before strongclose upgrade" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                end
                _managedcloseclear!(cache, base, longclose)
                existing = nothing
            end
        end
        reservedsell = _reservedspotsellqty(cache, symbol; exclude_orderid=existing_orderid)
        closeablelong = max(0f0, freebase - borrowedbase - reservedsell)
        requiredlongreductionquote = max(0f0, longexposurequote - maxassetquote)
        if (overallocatedbudgetquote > 0.0) && (allocatedbudgetquote > 0.0)
            globalsharequote = overallocatedbudgetquote * (Float64(longexposurequote) / allocatedbudgetquote)
            requiredlongreductionquote = max(requiredlongreductionquote, Float32(globalsharequote))
        end
        targetclosequote = (ta.tradelabel == longstrongclose) ? requiredlongreductionquote : max(sellbuyqtyratio * qtyacceleration * quotequantity, requiredlongreductionquote)
        if (ta.tradelabel == longstrongclose) && (targetclosequote <= 0f0)
            return nothing
        end
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = closeablelong <= 2 * minimumbasequantity ? (closeablelong >= minimumbasequantity ? closeablelong : minimumbasequantity) : minimumbasequantity
        basequantity = min(max(targetclosequote / price, minimumbasequantity), closeablelong)
        sufficientsellbalance = (basequantity <= closeablelong) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if sufficientsellbalance && exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = nothing
            if !isnothing(existing)
                if _managed_maintenance_blocked(cache)
                    oid = String(existing[:orderid])
                else
                    existing_limit = get(existing, :limitprice, nothing)
                    existing_qty = Float32(get(existing, :baseqty, 0f0))
                    if !_material_order_change(existing_limit, requested_limitprice, existing_qty, basequantity; price_reltol=_order_amend_price_rel_threshold(cache))
                        oid = String(existing[:orderid])
                    else
                        amended = try
                            (cache.mc[:trademode] == notrade) ? String(existing[:orderid]) : _withtradelogcontext(cache, ta) do
                                CryptoXch.changeorder(cache.xc, symbol, String(existing[:orderid]); limitprice=requested_limitprice, basequantity=basequantity)
                            end
                        catch err
                            if _isunknownordererror(err)
                                (verbosity >= 1) && @warn "managed longclose amend skipped because order is no longer present" base=base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                                recovered = _recover_managed_close_from_snapshot!(cache, base, longclose)
                                if isnothing(recovered)
                                    _managedcloseclear!(cache, base, longclose)
                                end
                                recovered
                            else
                                rethrow(err)
                            end
                        end
                        if !isnothing(amended)
                            oid = amended
                        else
                            if !isnothing(_managedcloseget(cache, base, longclose))
                                try
                                    CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
                                catch err
                                    (verbosity >= 1) && @warn "failed to cancel managed longclose before recreate" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                                end
                                _managedcloseclear!(cache, base, longclose)
                            end
                        end
                    end
                end
            end
            if isnothing(oid)
                oid = (cache.mc[:trademode] == notrade) ? "SellSpotSim" : _withtradelogcontext(cache, ta) do
                    CryptoXch.closeorder(cache.xc, base; positionside=:long, limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=0, reduceonly=false)
                end
            end
            if !isnothing(oid)
                _managedcloseset!(cache, base, oid, longclose; limitprice=requested_limitprice, baseqty=basequantity)
                result = (trade=longclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longclose order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longclose order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no longclose $base due to sufficientsellbalance=$sufficientsellbalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    elseif (ta.tradelabel in [longbuy, longstrongbuy]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        remaininglongcapacityquote = max(0f0, min(maxassetquote - longexposurequote, Float32(remainingbudgetquote)))
        targetopenquote = min(qtyacceleration * quotequantity, remaininglongcapacityquote)
        basequantity = max(0f0, min(max(targetopenquote / price, minimumbasequantity) * price, freeusdt - freeusdtfractionmargin * effectivebudgetquote) / price)
        sufficientbuybalance = (basequantity * price < freeusdt) && ((basequantity + borrowedbase) > 0.0)
        # basequantity += borrowedbase # buy all short as well when switching to long
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        basefraction = (longexposurequote + basequantity * price) / effectivebudgetquote
    
        if remainingbudgetquote <= 0.0
            (verbosity > 2) && println("$(tradetime(cache)) skip $base longbuy: allocated budget exhausted allocated=$(allocatedbudgetquote) limit=$(effectivebudgetquote)")
        elseif remaininglongcapacityquote <= 0f0
            (verbosity > 2) && println("$(tradetime(cache)) skip $base longbuy: max asset fraction reached longexposurequote=$(longexposurequote) maxassetquote=$(maxassetquote)")
        elseif basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 3) && println("$(tradetime(cache)) skip $base longbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif _hasopenorderside(cache, symbol; side="buy", require_leverage=false) || _hasactiveopenbuy(cache, symbol)
            (verbosity >= 2) && println("$(tradetime(cache)) skip $base longbuy: existing open longbuy order is still active")
        elseif sufficientbuybalance && exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = (cache.mc[:trademode] == notrade) ? "BuySpotSim" : _withtradelogcontext(cache, ta) do
                CryptoXch.createbuyorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=0)
            end
            if !isnothing(oid)
                result = (trade=longbuy, oid=oid)
                _rememberactiveopenbuy!(cache, symbol)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longbuy order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longbuy order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base longbuy due to sufficientbuybalance=$sufficientbuybalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
        elseif (ta.tradelabel in [shortstrongbuy, shortbuy]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        remainingshortcapacityquote = max(0f0, min(maxassetquote - shortexposurequote, Float32(remainingbudgetquote)))
        targetshortopenquote = min(qtyacceleration * quotequantity, remainingshortcapacityquote)
        basequantity = max(targetshortopenquote / price, minimumbasequantity)
        sufficientbuybalance = ((basequantity - freebase) * price < freeshortquote) && (basequantity > 0.0)
        basefraction = (shortexposurequote + basequantity * price) / effectivebudgetquote
        marginok = CryptoXch.marginpermitted(cache.xc, symbol, "Sell", short_margin_leverage; role=CryptoXch.trade_exchange_spot)
        if remainingbudgetquote <= 0.0
            (verbosity > 2) && println("$(tradetime(cache)) skip $base shortbuy: allocated budget exhausted allocated=$(allocatedbudgetquote) limit=$(effectivebudgetquote)")
        elseif remainingshortcapacityquote <= 0f0
            (verbosity > 2) && println("$(tradetime(cache)) skip $base shortbuy: max asset fraction reached shortexposurequote=$(shortexposurequote) maxassetquote=$(maxassetquote)")
        elseif basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 2) && println("$(tradetime(cache)) skip $base shortbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif _hasopenorderside(cache, symbol; side="sell", require_leverage=true) || _hasactiveopensell(cache, symbol)
            (verbosity >= 2) && println("$(tradetime(cache)) skip $base shortbuy: existing open short-entry sell order is still active")
        elseif !marginok
            limits = CryptoXch.marginlimits(cache.xc, symbol; role=CryptoXch.trade_exchange_spot)
            (verbosity >= 1) && @warn "skip $base shortbuy due to Kraken margin metadata limits" symbol=symbol requested_leverage=short_margin_leverage maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell
        elseif sufficientbuybalance
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = (cache.mc[:trademode] == notrade) ? "SellMarginSim" : _withtradelogcontext(cache, ta) do
                try
                    CryptoXch.createsellorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=short_margin_leverage)
                catch err
                    _log_margin_order_diagnostics(cache, basecfg, ta, base, "Sell", short_margin_leverage, requested_limitprice, basequantity, freebase, borrowedbase, freeshortquote, totalborrowedusdt, effectivebudgetquote, err)
                    rethrow(err)
                end
            end
            if !isnothing(oid)
                result = (trade=shortbuy, oid=oid)
                _rememberactiveopensell!(cache, symbol)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortbuy order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeshortquote)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortbuy order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeshortquote)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base shortbuy due to sufficientbuybalance=$sufficientbuybalance")
        end
    elseif (ta.tradelabel in [shortclose, shortstrongclose]) && (cache.mc[:trademode] in [buysell, closeonly, quickexit]) && basecfg.sellenabled
        existing = _managedcloseget(cache, base, shortclose)
        if !isnothing(existing)
            inherited = Bool(get(existing, :inherited, false))
            existing_tif = uppercase(String(get(existing, :timeinforce, "")))
            if (ta.tradelabel == shortstrongclose) && inherited && (existing_tif != "POSTONLY")
                try
                    CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
                catch err
                    (verbosity >= 1) && @warn "failed to cancel inherited non-PostOnly close order before strongclose upgrade" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                end
                _managedcloseclear!(cache, base, shortclose)
                existing = nothing
            end
        end
        closeableshort = borrowedbase
        requiredshortreductionquote = max(0f0, shortexposurequote - maxassetquote)
        if (overallocatedbudgetquote > 0.0) && (allocatedbudgetquote > 0.0)
            globalsharequote = overallocatedbudgetquote * (Float64(shortexposurequote) / allocatedbudgetquote)
            requiredshortreductionquote = max(requiredshortreductionquote, Float32(globalsharequote))
        end
        targetshortclosequote = (ta.tradelabel == shortstrongclose) ? requiredshortreductionquote : max(sellbuyqtyratio * qtyacceleration * quotequantity, requiredshortreductionquote)
        if (ta.tradelabel == shortstrongclose) && (targetshortclosequote <= 0f0)
            return nothing
        end
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = closeableshort <= 2 * minimumbasequantity ? (closeableshort >= minimumbasequantity ? closeableshort : minimumbasequantity) : minimumbasequantity # increase minimumbasequantity if otherwise a too small base amount remains that cannot be sold
        basequantity = max(0f0, min(max(targetshortclosequote / price, minimumbasequantity), closeableshort))
        # Cap short-close buys by currently free quote balance to avoid repeated insufficient-funds rejects.
        maxcloseaffordable = price > 0f0 ? max(0f0, freeusdt / price) : 0f0
        basequantity = min(basequantity, maxcloseaffordable)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = nothing
            if !isnothing(existing)
                if _managed_maintenance_blocked(cache)
                    oid = String(existing[:orderid])
                else
                    existing_limit = get(existing, :limitprice, nothing)
                    existing_qty = Float32(get(existing, :baseqty, 0f0))
                    if !_material_order_change(existing_limit, requested_limitprice, existing_qty, basequantity; price_reltol=_order_amend_price_rel_threshold(cache))
                        oid = String(existing[:orderid])
                    else
                        amended = try
                            (cache.mc[:trademode] == notrade) ? String(existing[:orderid]) : _withtradelogcontext(cache, ta) do
                                CryptoXch.changeorder(cache.xc, symbol, String(existing[:orderid]); limitprice=requested_limitprice, basequantity=basequantity)
                            end
                        catch err
                            if _isunknownordererror(err)
                                (verbosity >= 1) && @warn "managed shortclose amend skipped because order is no longer present" base=base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                                recovered = _recover_managed_close_from_snapshot!(cache, base, shortclose)
                                if isnothing(recovered)
                                    _managedcloseclear!(cache, base, shortclose)
                                end
                                recovered
                            else
                                rethrow(err)
                            end
                        end
                        if !isnothing(amended)
                            oid = amended
                        else
                            if !isnothing(_managedcloseget(cache, base, shortclose))
                                try
                                    CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
                                catch err
                                    (verbosity >= 1) && @warn "failed to cancel managed shortclose before recreate" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                                end
                                _managedcloseclear!(cache, base, shortclose)
                            end
                        end
                    end
                end
            end
            if isnothing(oid)
                oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : _withtradelogcontext(cache, ta) do
                    try
                        CryptoXch.closeorder(cache.xc, base; positionside=:short, limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=short_margin_leverage, reduceonly=true)
                    catch err
                        _log_margin_order_diagnostics(cache, basecfg, ta, base, "Buy", short_margin_leverage, requested_limitprice, basequantity, freebase, borrowedbase, freeusdt, totalborrowedusdt, effectivebudgetquote, err)
                        rethrow(err)
                    end
                end
            end
            if !isnothing(oid)
                _managedcloseset!(cache, base, oid, shortclose; limitprice=requested_limitprice, baseqty=basequantity)
                result = (trade=shortclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortclose order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortclose order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(cache, assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no shortclose $base due to exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    end
    push!(cache.dbgdf, (
        taconfigid = isnothing(ta) ? missing : ta.configid,
        tatradelabel = isnothing(ta) ? missing : ta.tradelabel,
        tabase = isnothing(ta) ? missing : ta.base,
        tahourlygain = isnothing(ta) ? missing : ta.hourlygain,
        oid = isnothing(result) ? missing : result.oid,
        baseqty = basequantity,
        minimumbasequantity = minimumbasequantity,
        freebase = freebase,
        totalborrowedusdt = totalborrowedusdt,
        freeusdt = freeusdt,
        quoteqty = quotequantity,
        price = price,
        opentime=currenttime(ohlcv)
    ), promote=true)
    if !isnothing(result)

    end 
    return result
end

tradetime(cache::TradeCache) = CryptoXch.ttstr(cache.xc)
function USDTmsg(cache::TradeCache, assets)
    totalusdt = sum(assets.usdtvalue)
    totalborrowedusdt = sum(assets[!, :borrowed] .* assets[!, :usdtprice])
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free]) - totalborrowedusdt
    equityquote = max(0.0, Float64(get(CryptoXch.accountcapacity(cache.xc), :equity_quote, totalusdt)))
    freepct = equityquote > 0f0 ? min(100, round(Int, freeusdt / equityquote * 100)) : 0
    return string("$(EnvConfig.cryptoquote): equity=$(round(Int, equityquote)), exposure=$(round(Int, totalusdt)), quotefree=$(freepct)%")
end

function tradeadvicelessthan(ta1, ta2)
    closeset = [shortclose, shortstrongclose, allclose, longstrongclose, longclose]
    buyset = [shortstrongbuy, shortbuy, longbuy, longstrongbuy]
    holdset = [shorthold, longhold]
    if (ta1.tradelabel in closeset) && !(ta2.tradelabel in closeset)
        return true
    elseif (ta1.tradelabel in buyset) && (ta2.tradelabel in buyset)
        if ta1.hourlygain < ta2.hourlygain
            return true
        end
    end
    return false
end

_strategyengine(cache::TradeCache) = Symbol(get(cache.mc, :strategy_engine, :classifier))

function _snapshot_strategy_advices(tradeadvices::Vector{StrategyAdvice})::Vector{NamedTuple{(:base, :tradelabel, :price, :relativeamount), Tuple{String, String, Union{Nothing, Float32}, Float32}}}
    rows = NamedTuple{(:base, :tradelabel, :price, :relativeamount), Tuple{String, String, Union{Nothing, Float32}, Float32}}[]
    for ta in tradeadvices
        push!(rows, (
            base=uppercase(String(ta.base)),
            tradelabel=String(Symbol(ta.tradelabel)),
            price=isnothing(ta.price) ? nothing : Float32(ta.price),
            relativeamount=Float32(ta.relativeamount),
        ))
    end
    sort!(rows; by=x -> (x.base, x.tradelabel, isnothing(x.price) ? -1f0 : x.price, x.relativeamount))
    return rows
end

function _shadow_compare_strategy_advices(sync_advices::Vector{StrategyAdvice}, async_advices::Vector{StrategyAdvice})
    sync_rows = _snapshot_strategy_advices(sync_advices)
    async_rows = _snapshot_strategy_advices(async_advices)
    diffs = String[]
    critical = false

    if length(sync_rows) != length(async_rows)
        critical = true
        push!(diffs, "count_mismatch sync=$(length(sync_rows)) async=$(length(async_rows))")
    end

    shared = min(length(sync_rows), length(async_rows))
    for i in 1:shared
        s = sync_rows[i]
        a = async_rows[i]
        if (s.base != a.base) || (s.tradelabel != a.tradelabel)
            critical = true
            push!(diffs, "label_mismatch ix=$(i) sync=$(s.base):$(s.tradelabel) async=$(a.base):$(a.tradelabel)")
        end
        if isnothing(s.price) != isnothing(a.price)
            critical = true
            push!(diffs, "price_mode_mismatch ix=$(i) sync=$(s.price) async=$(a.price)")
        elseif !isnothing(s.price)
            pricediff = abs(Float32(s.price) - Float32(a.price))
            if pricediff > ASYNC_SHADOW_PRICE_TOLERANCE
                critical = true
                push!(diffs, "price_mismatch ix=$(i) sync=$(s.price) async=$(a.price) diff=$(pricediff)")
            end
        end
        qtydiff = abs(Float32(s.relativeamount) - Float32(a.relativeamount))
        if qtydiff > ASYNC_SHADOW_QTY_TOLERANCE
            critical = true
            push!(diffs, "qty_mismatch ix=$(i) sync=$(s.relativeamount) async=$(a.relativeamount) diff=$(qtydiff)")
        end
    end

    return (ok=!critical, critical=critical, sync_count=length(sync_rows), async_count=length(async_rows), diffs=diffs)
end

function _async_worker_tradecache!(cache::TradeCache; seed_closed_dt=nothing)::TradeCache
    if !haskey(cache.mc, :async_worker_trade_cache)
        worker = TradeCache(xc=cache.xc, cl=cache.cl, trademode=cache.mc[:trademode])
        worker.mc[:tradelog_portfolio_snapshot_mode] = :none
        worker.mc[:tradelog_migration_worker_probe_enabled] = false
        worker.mc[:async_engine_enabled] = false
        worker.mc[:async_shadow_mode] = true
        worker.mc[:use_strategy_runtime_api] = Bool(get(cache.mc, :use_strategy_runtime_api, false))
        apply_tradingstrategy!(worker, deepcopy(cache.mc[:strategy_template]); strategy_engine=_strategyengine(cache), source=String(get(cache.mc, :strategy_source, "async_worker")))
        if !isnothing(seed_closed_dt)
            worker.mc[:strategy_last_closed_candle_dt] = DateTime(seed_closed_dt)
        end
        cache.mc[:async_worker_trade_cache] = worker
    end
    worker = cache.mc[:async_worker_trade_cache]
    worker.cfg = deepcopy(cache.cfg)
    worker.mc[:trademode] = cache.mc[:trademode]
    worker.mc[:usenewtrade] = get(cache.mc, :usenewtrade, false)
    worker.mc[:use_strategy_runtime_api] = Bool(get(cache.mc, :use_strategy_runtime_api, false))
    return worker
end

"Compute async canary advice from dedicated worker state while keeping sync state untouched."
function _collect_async_candidate_advices(cache::TradeCache, assets::AbstractDataFrame, sync_advices::Vector{StrategyAdvice}; seed_closed_dt=nothing)::Vector{StrategyAdvice}
    isempty(sync_advices) && return StrategyAdvice[]
    size(cache.cfg, 1) == 0 && return deepcopy(sync_advices)
    worker = _async_worker_tradecache!(cache; seed_closed_dt=seed_closed_dt)
    worker_advices = _collect_strategy_advices(worker, deepcopy(assets))
    canary_bases = Set(uppercase(String(ta.base)) for ta in sync_advices)
    return [ta for ta in worker_advices if uppercase(String(ta.base)) in canary_bases]
end

function _run_async_shadow_compare!(cache::TradeCache, sync_advices::Vector{StrategyAdvice}, async_advices::Vector{StrategyAdvice})::Bool
    result = _shadow_compare_strategy_advices(sync_advices, async_advices)
    cache.mc[:async_shadow_last_compare] = merge((
        datetime=cache.xc.currentdt,
        async_shadow_mode=Bool(get(cache.mc, :async_shadow_mode, true)),
    ), result)
    if result.critical
        cache.mc[:async_shadow_autodisabled] = true
        cache.mc[:async_shadow_autodisable_reason] = isempty(result.diffs) ? "critical divergence" : result.diffs[1]
        cache.mc[:async_engine_enabled] = false
        (verbosity >= 1) && @warn "async shadow divergence detected; auto-disabling async engine" reason=cache.mc[:async_shadow_autodisable_reason] details=result.diffs
        return false
    end
    return true
end

function _orders_idset(df::AbstractDataFrame)::Set{String}
    if !any(name -> String(name) == "orderid", names(df))
        return Set{String}()
    end
    return Set(String.(df[!, :orderid]))
end

function _balances_by_coin(df::AbstractDataFrame)::Dict{String, NamedTuple{(:free, :locked, :borrowed), Tuple{Float64, Float64, Float64}}}
    out = Dict{String, NamedTuple{(:free, :locked, :borrowed), Tuple{Float64, Float64, Float64}}}()
    required = Set(["coin", "free", "locked", "borrowed"])
    if !all(x -> x in Set(String.(names(df))), required)
        return out
    end
    for row in eachrow(df)
        coin = uppercase(String(row.coin))
        out[coin] = (
            free=Float64(row.free),
            locked=Float64(row.locked),
            borrowed=Float64(row.borrowed),
        )
    end
    return out
end

function _run_ws_shadow_compare!(cache::TradeCache, openorders::AbstractDataFrame, assets::AbstractDataFrame)::Bool
    if !Bool(get(cache.mc, :ws_shadow_mode, true))
        return true
    end
    diffs = String[]
    orders_ok = true
    balances_ok = true

    if Bool(get(cache.mc, :ws_orders_enabled, false))
        ws_orders = try
            CryptoXch.wsordersnapshot(cache.xc)
        catch
            DataFrame()
        end
        sync_ids = _orders_idset(openorders)
        ws_ids = _orders_idset(ws_orders)
        if !isempty(sync_ids) || !isempty(ws_ids)
            only_sync = setdiff(sync_ids, ws_ids)
            only_ws = setdiff(ws_ids, sync_ids)
            if !isempty(only_sync) || !isempty(only_ws)
                orders_ok = false
                push!(diffs, "orders id mismatch sync_only=$(length(only_sync)) ws_only=$(length(only_ws))")
            end
        end
    end

    if Bool(get(cache.mc, :ws_balances_enabled, false))
        ws_balances = try
            CryptoXch.wsbalancessnapshot(cache.xc)
        catch
            DataFrame()
        end
        sync_map = _balances_by_coin(assets)
        ws_map = _balances_by_coin(ws_balances)
        if !isempty(sync_map) || !isempty(ws_map)
            tol = 1e-4
            allcoins = union(Set(keys(sync_map)), Set(keys(ws_map)))
            for coin in allcoins
                sv = get(sync_map, coin, (free=0.0, locked=0.0, borrowed=0.0))
                wv = get(ws_map, coin, (free=0.0, locked=0.0, borrowed=0.0))
                if (abs(sv.free - wv.free) > tol) || (abs(sv.locked - wv.locked) > tol) || (abs(sv.borrowed - wv.borrowed) > tol)
                    balances_ok = false
                    push!(diffs, "balance mismatch coin=$(coin) sync=$(sv) ws=$(wv)")
                    break
                end
            end
        end
    end

    ok = orders_ok && balances_ok
    cache.mc[:ws_shadow_last_compare] = (
        datetime=cache.xc.currentdt,
        orders_ok=orders_ok,
        balances_ok=balances_ok,
        ws_orders_enabled=Bool(get(cache.mc, :ws_orders_enabled, false)),
        ws_balances_enabled=Bool(get(cache.mc, :ws_balances_enabled, false)),
        diffs=diffs,
    )

    if !ok
        (verbosity >= 1) && @warn "ws shadow compare mismatch" diffs=diffs
        if Bool(get(cache.mc, :ws_primary_mode, false)) && Bool(get(cache.mc, :ws_primary_autofallback_on_mismatch, true))
            cache.mc[:ws_primary_mode] = false
            cache.xc.mc[:ws_primary_mode] = false
            _objective4_inc!(cache, :ws_primary_fallbacks)
            (verbosity >= 1) && @warn "ws primary mode auto-fallback activated due to shadow mismatch"
        end
        return false
    end
    return true
end

function _sync_exchange_balances_snapshot!(cache::TradeCache, assets::AbstractDataFrame)
    snap = try
        CryptoXch.refreshbalancessnapshot!(cache.xc; ignoresmallvolume=false)
    catch err
        (verbosity >= 1) && @warn "failed to refresh exchange-owned balances snapshot" error=sprint(showerror, err)
        return nothing
    end
    return nothing
end

# ── Loop control ────────────────────────────────────────────────────────────

"Returns the current loop lifecycle state."
_loopstate_nolock(cache::TradeCache) = LoopState(Int(cache.mc[:loop_state]))
_setloopstate_nolock!(cache::TradeCache, s::LoopState) = (cache.mc[:loop_state] = s; nothing)

function _setloopstate!(cache::TradeCache, s::LoopState)
    lock(cache.looplock)
    try
        _setloopstate_nolock!(cache, s)
        notify(cache.loopcond; all=true)
    finally
        unlock(cache.looplock)
    end
    return nothing
end

function _waitforactive_loopstate!(cache::TradeCache)
    lock(cache.looplock)
    try
        while _loopstate_nolock(cache) == loop_paused
            wait(cache.loopcond)
        end
        return _loopstate_nolock(cache)
    finally
        unlock(cache.looplock)
    end
end

# ── Strategy config ─────────────────────────────────────────────────────────

function _validatestrategyconfig!(mc::AbstractDict)
    openthreshold = Float32(mc[:strategy_openthreshold])
    closethreshold = Float32(mc[:strategy_closethreshold])
    buygain = Float32(mc[:strategy_buygain])
    sellgain = Float32(mc[:strategy_sellgain])
    limitreduction = Float32(mc[:strategy_limitreduction])
    maxwindow = Int(mc[:strategy_maxwindow])

    @assert 0f0 <= openthreshold <= 1f0 "strategy_openthreshold must be in [0, 1], got $(openthreshold)"
    @assert 0f0 <= closethreshold <= 1f0 "strategy_closethreshold must be in [0, 1], got $(closethreshold)"
    @assert 0f0 <= buygain <= 1f0 "strategy_buygain must be in [0, 1], got $(buygain)"
    @assert 0f0 <= sellgain <= 1f0 "strategy_sellgain must be in [0, 1], got $(sellgain)"
    @assert 0f0 <= limitreduction <= 1f0 "strategy_limitreduction must be in [0, 1], got $(limitreduction)"
    @assert maxwindow > 0 "strategy_maxwindow must be > 0, got $(maxwindow)"
    return mc
end

"Validate strategy runtime parameters stored in `TradeCache.mc`."
function _validatestrategyconfig!(cache::TradeCache)
    _validatestrategyconfig!(cache.mc)
    return cache
end

"Apply strategy runtime settings from a `TradingStrategy.GainSegment` and reset derived per-base state."
function apply_tradingstrategy!(mc::AbstractDict, gs::TradingStrategy.GainSegment; strategy_engine::Symbol=:getgainsalgo, source::AbstractString="manual")
    mc[:strategy_engine] = strategy_engine
    _setstrategyruntimefromsegment!(mc, gs, source)

    strategy_state = get!(mc, :strategy_state, Dict{String, Any}())
    strategy_history = get!(mc, :strategy_history, Dict{String, Any}())
    managed_close_orders = get!(mc, :managed_close_orders, Dict{String, Dict{Symbol, Any}}())
    empty!(strategy_state)
    empty!(strategy_history)
    empty!(managed_close_orders)
    return _validatestrategyconfig!(mc)
end

function apply_tradingstrategy!(cache::TradeCache, gs::TradingStrategy.GainSegment; strategy_engine::Symbol=:getgainsalgo, source::AbstractString="manual")
    apply_tradingstrategy!(cache.mc, gs; strategy_engine=strategy_engine, source=source)
    rt = _strategyruntime(cache)
    !isnothing(rt) && TradingStrategy.apply_strategy!(rt, gs; source=source)
    return cache
end

function _strategyhistory!(cache::TradeCache, base::AbstractString)
    return get!(cache.mc[:strategy_history], String(base)) do
        (
            predictionsdf=DataFrame(opentime=DateTime[], high=Float32[], low=Float32[], close=Float32[]),
            scores=Float32[],
            labels=Targets.TradeLabel[],
        )
    end
end

function _strategystate!(cache::TradeCache, base::AbstractString)
    return get!(cache.mc[:strategy_state], String(base)) do
        _validatestrategyconfig!(cache)
        gs = TradingStrategy.GainSegment(
            ;
            maxwindow=Int(cache.mc[:strategy_maxwindow]),
            openthreshold=Float32(cache.mc[:strategy_openthreshold]),
            closethreshold=Float32(cache.mc[:strategy_closethreshold]),
            algorithm=get(cache.mc, :strategy_algorithm, TradingStrategy.gain_reversal!),
            limitreduction=Float32(cache.mc[:strategy_limitreduction]),
        )
        gs.buygain = Float32(cache.mc[:strategy_buygain])
        gs.sellgain = Float32(cache.mc[:strategy_sellgain])
        gs
    end
end

"update if the record already exists, otherwise insert it."
function _upsert_getgainsalgo_sample!(history, ohlcv::Ohlcv.OhlcvData, label::Targets.TradeLabel, score)
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

function _assets_base_mask(assets::AbstractDataFrame, base::AbstractString)
    hasproperty(assets, :coin) || return falses(size(assets, 1))
    basekey = uppercase(String(base))
    return uppercase.(String.(assets[!, :coin])) .== basekey
end

function _asset_price_hint(assets::AbstractDataFrame, mask)::Float32
    if !hasproperty(assets, :usdtprice)
        return 0f0
    end
    vals = Float32.(assets[mask, :usdtprice])
    vals = vals[vals .> 0f0]
    return isempty(vals) ? 0f0 : vals[1]
end

function _set_gainsalgo_reconciliation!(gs::TradingStrategy.GainSegment, base::AbstractString, assets::AbstractDataFrame)
    mask = _assets_base_mask(assets, base)
    freebase = hasproperty(assets, :free) ? Float32(sum(assets[mask, :free])) : 0f0
    borrowedbase = hasproperty(assets, :borrowed) ? Float32(sum(assets[mask, :borrowed])) : 0f0
    pricehint = _asset_price_hint(assets, mask)

    longavg = max(0f0, Float32(gs.longta.openprice))
    shortavg = max(0f0, Float32(gs.shortta.openprice))
    (longavg <= 0f0) && (longavg = pricehint)
    (shortavg <= 0f0) && (shortavg = pricehint)

    longix = max(Int(gs.longta.openix), 0)
    shortix = max(Int(gs.shortta.openix), 0)

    TradingStrategy.setreconciliation!(
        gs;
        long_open_qty=freebase,
        long_avg_entry=longavg,
        long_open_ix=longix,
        short_open_qty=borrowedbase,
        short_avg_entry=shortavg,
        short_open_ix=shortix,
    )
    return nothing
end

function _strategy_sell_limitprice(cache::TradeCache, base::AbstractString, tradelabel::Targets.TradeLabel)
    if _strategyengine(cache) != :getgainsalgo
        return nothing
    end
    if tradelabel in [longstrongclose, shortstrongclose]
        return nothing
    end
    gs = get(cache.mc[:strategy_state], String(base), nothing)
    isnothing(gs) && return nothing
    if tradelabel == longclose
        v = Float32(gs.longta.closeprice)
        return v > 0f0 ? v : nothing
    elseif tradelabel == shortclose
        v = Float32(gs.shortta.closeprice)
        return v > 0f0 ? v : nothing
    end
    return nothing
end

function _managedclosestate(cache::TradeCache)::Dict{String, Dict{Symbol, Any}}
    return get!(cache.mc, :managed_close_orders, Dict{String, Dict{Symbol, Any}}())
end

function _managedcloseside(tradelabel::Targets.TradeLabel)::String
    if tradelabel in [longclose, longstrongclose]
        return "Sell"
    elseif tradelabel in [shortclose, shortstrongclose]
        return "Buy"
    end
    throw(ArgumentError("managed close label=$(tradelabel) must be a close label"))
end

function _managedclosekey(base::AbstractString, tradelabel::Targets.TradeLabel)::String
    return string(uppercase(String(base)), "|", _managedcloseside(tradelabel))
end

function _managedcloseget(cache::TradeCache, base::AbstractString, tradelabel::Targets.TradeLabel)
    return get(_managedclosestate(cache), _managedclosekey(base, tradelabel), nothing)
end

function _managedcloseset!(cache::TradeCache, base::AbstractString, orderid, tradelabel::Targets.TradeLabel; limitprice=nothing, baseqty::Real=0f0, inherited::Bool=false, timeinforce=nothing)
    _managedclosestate(cache)[_managedclosekey(base, tradelabel)] = Dict{Symbol, Any}(
        :orderid => String(orderid),
        :tradelabel => tradelabel,
        :limitprice => isnothing(limitprice) ? nothing : Float32(limitprice),
        :baseqty => Float32(baseqty),
        :inherited => Bool(inherited),
        :timeinforce => isnothing(timeinforce) ? nothing : String(timeinforce),
        :updated => Dates.now(Dates.UTC),
    )
    return nothing
end

function _managedcloseclear!(cache::TradeCache, base::AbstractString, tradelabel::Targets.TradeLabel)
    delete!(_managedclosestate(cache), _managedclosekey(base, tradelabel))
    return nothing
end

function _positioncloselabels(assets::AbstractDataFrame, base::AbstractString; sellenabled::Bool=true)::Vector{Targets.TradeLabel}
    basekey = uppercase(String(base))
    freebase = Float32(sum(assets[uppercase.(String.(assets[!, :coin])) .== basekey, :free]))
    borrowedbase = Float32(sum(assets[uppercase.(String.(assets[!, :coin])) .== basekey, :borrowed]))
    sellablelong = max(0f0, freebase - borrowedbase)
    labels = Targets.TradeLabel[]
    if sellenabled && (sellablelong > 0f0)
        push!(labels, longclose)
    end
    if sellenabled && (borrowedbase > 0f0)
        push!(labels, shortclose)
    end
    return labels
end

function _cfgrow_for_base(cache::TradeCache, base::AbstractString)
    hasproperty(cache.cfg, :basecoin) || return nothing
    rowix = findfirst(==(uppercase(String(base))), uppercase.(String.(cache.cfg[!, :basecoin])))
    if isnothing(rowix)
        return nothing
    end
    return cache.cfg[rowix, :]
end

function _basecfg_for_close(cache::TradeCache, base::AbstractString, sellenabled::Bool)::DataFrameRow
    cfgrow = _cfgrow_for_base(cache, base)
    if !isnothing(cfgrow)
        return cfgrow
    end

    # Fallback row allows close-only maintenance for held bases that are not in runtime cfg.
    fallback = DataFrame(
        basecoin=[uppercase(String(base))],
        buyenabled=[false],
        sellenabled=[Bool(sellenabled)],
        classifieraccepted=[false],
        inportfolio=[true],
        minquotevol=[false],
        continuousminvol=[false],
        whitelisted=[false],
        robotownedlongqty=[0f0],
        robotownedshortqty=[0f0],
        datetime=[isnothing(cache.xc.currentdt) ? Dates.now() : cache.xc.currentdt],
    )
    return fallback[1, :]
end

function _close_management_bases(cache::TradeCache, assets::AbstractDataFrame)::Vector{String}
    quote_coin = uppercase(String(EnvConfig.cryptoquote))
    bases = String[]
    if hasproperty(cache.cfg, :basecoin)
        for base in String.(cache.cfg[!, :basecoin])
            push!(bases, uppercase(base))
        end
    end
    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        (base == quote_coin) && continue
        freebase = Float32(getproperty(row, :free))
        borrowedbase = Float32(getproperty(row, :borrowed))
        if (freebase > 0f0) || (borrowedbase > 0f0)
            push!(bases, base)
        end
    end
    return unique(bases)
end

function _orderbase_from_symbol(cache::TradeCache, assets::AbstractDataFrame, symbol::AbstractString)::Union{Nothing, String}
    sym = uppercase(String(symbol))
    for base in _close_management_bases(cache, assets)
        expected = uppercase(String(CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)))
        if sym == expected
            return base
        end
    end
    return nothing
end

function _managed_close_label_from_order(orow)::Union{Nothing, Targets.TradeLabel}
    side = uppercase(String(getproperty(orow, :side)))
    is_leverage = _orderisleverage(orow)
    if (side == "SELL") && !is_leverage
        return longclose
    elseif (side == "BUY") && is_leverage
        return shortclose
    end
    return nothing
end

function _managed_order_baseqty(orow)::Float32
    if hasproperty(orow, :baseqty)
        return Float32(getproperty(orow, :baseqty))
    elseif hasproperty(orow, :qty)
        return Float32(getproperty(orow, :qty))
    end
    return 0f0
end

function _managed_order_limitprice(orow)
    if hasproperty(orow, :price)
        v = Float32(getproperty(orow, :price))
        return v > 0f0 ? v : nothing
    elseif hasproperty(orow, :limitprice)
        v = Float32(getproperty(orow, :limitprice))
        return v > 0f0 ? v : nothing
    end
    return nothing
end

function _managed_order_timeinforce(orow)
    if hasproperty(orow, :timeinforce)
        return String(getproperty(orow, :timeinforce))
    elseif hasproperty(orow, :timeInForce)
        return String(getproperty(orow, :timeInForce))
    end
    return nothing
end

"Bind managed close state to an existing matching open order from the current snapshot."
function _recover_managed_close_from_snapshot!(cache::TradeCache, base::AbstractString, closelabel)::Union{Nothing, String}
    oo = get(cache.mc, :openorders_snapshot, DataFrame())
    size(oo, 1) == 0 && return nothing
    symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    target_side = closelabel == longclose ? "SELL" : "BUY"
    target_leverage = closelabel == longclose ? false : true

    best_ix = nothing
    best_qty = -1f0
    for (ix, orow) in enumerate(eachrow(oo))
        CryptoXch.openstatus(String(getproperty(orow, :status))) || continue
        (String(getproperty(orow, :symbol)) == symbol) || continue
        (uppercase(String(getproperty(orow, :side))) == target_side) || continue
        (_orderisleverage(orow) == target_leverage) || continue
        qty = _managed_order_baseqty(orow)
        if qty > best_qty
            best_qty = qty
            best_ix = ix
        end
    end

    isnothing(best_ix) && return nothing
    row = oo[best_ix, :]
    oid = String(getproperty(row, :orderid))
    _managedcloseset!(
        cache,
        base,
        oid,
        closelabel;
        limitprice=_managed_order_limitprice(row),
        baseqty=_managed_order_baseqty(row),
        inherited=true,
        timeinforce=_managed_order_timeinforce(row),
    )
    return oid
end

function _reconstruct_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, oo::AbstractDataFrame)
    state = _managedclosestate(cache)
    empty!(state)
    for orow in eachrow(oo)
        CryptoXch.openstatus(String(getproperty(orow, :status))) || continue
        closelabel = _managed_close_label_from_order(orow)
        isnothing(closelabel) && continue
        base = _orderbase_from_symbol(cache, assets, String(getproperty(orow, :symbol)))
        isnothing(base) && continue
        _managedcloseset!(
            cache,
            base,
            String(getproperty(orow, :orderid)),
            closelabel;
            limitprice=_managed_order_limitprice(orow),
            baseqty=_managed_order_baseqty(orow),
            inherited=true,
            timeinforce=_managed_order_timeinforce(orow),
        )
    end
    return nothing
end

function _cancel_unmanaged_open_orders!(cache::TradeCache, oo::AbstractDataFrame)
    managed_ids = Set{String}(String(v[:orderid]) for v in values(_managedclosestate(cache)))
    candidates = NamedTuple{(:orderid, :symbol, :base, :side, :is_leverage, :created), Tuple{String, String, String, String, Bool, DateTime}}[]
    for orow in eachrow(oo)
        CryptoXch.openstatus(String(getproperty(orow, :status))) || continue
        oid = String(getproperty(orow, :orderid))
        oid in managed_ids && continue
        symbol = String(getproperty(orow, :symbol))
        base = _normalize_basecoin_token(symbol, EnvConfig.cryptoquote)
        isnothing(base) && continue
        created = (hasproperty(orow, :created) && !ismissing(getproperty(orow, :created))) ? DateTime(getproperty(orow, :created)) : DateTime(1970, 1, 1)
        side = hasproperty(orow, :side) ? uppercase(String(getproperty(orow, :side))) : ""
        push!(candidates, (orderid=oid, symbol=symbol, base=String(base), side=side, is_leverage=_orderisleverage(orow), created=created))
    end
    unmanaged_total = length(candidates)
    cache.mc[:unmanaged_open_orders] = unmanaged_total
    cache.mc[:backlog_drain_mode] = unmanaged_total > UNMANAGED_BACKLOG_DRAIN_THRESHOLD

    nowdt = Dates.now(Dates.UTC)
    cooldown_until = get(cache.mc, :unmanaged_cancel_cooldown_until, nothing)
    if !isnothing(cooldown_until) && (nowdt < cooldown_until)
        (verbosity >= 1) && @warn "skip unmanaged-order cancellation because cooldown is still active" cooldown_until=cooldown_until remaining_seconds=max(0.0, Dates.value(cooldown_until - nowdt) / 1000) unmanaged_open_orders=unmanaged_total
        return nothing
    end

    dupcounts = Dict{Tuple{String, String, Bool}, Int}()
    for c in candidates
        key = (c.symbol, c.side, c.is_leverage)
        dupcounts[key] = get(dupcounts, key, 0) + 1
    end
    cancel_order = sortperm(candidates; by = c -> ((get(dupcounts, (c.symbol, c.side, c.is_leverage), 0) > 1) ? 0 : 1, c.created, c.symbol, c.orderid))

    cancel_attempts = 0
    throttled = false
    cooldown_skips = 0
    first_cooldown = nothing
    first_base = ""
    first_symbol = ""
    first_orderid = ""
    for ix in cancel_order
        c = candidates[ix]
        oid = c.orderid
        symbol = c.symbol
        base = c.base
        if cancel_attempts >= UNMANAGED_CANCEL_MAX_PER_STEP
            throttled = true
            break
        end
        cancel_attempts += 1
        try
            CryptoXch.cancelorder(cache.xc, base, oid)
            haskey(cache.mc, :unmanaged_cancel_cooldown_until) && delete!(cache.mc, :unmanaged_cancel_cooldown_until)
            (verbosity >= 1) && @warn "cancelled unmanaged open order" base=base symbol=symbol orderid=oid
        catch err
            if _isunknownordererror(err)
                continue
            elseif _isprivatecooldownerror(err)
                cooldown_skips += 1
                until = _extractcooldownuntil(err)
                if isnothing(until)
                    until = Dates.now(Dates.UTC) + UNMANAGED_CANCEL_COOLDOWN_FALLBACK
                end
                cache.mc[:unmanaged_cancel_cooldown_until] = until
                if isnothing(first_cooldown)
                    first_cooldown = sprint(showerror, err)
                    first_base = base
                    first_symbol = symbol
                    first_orderid = oid
                end
                break
            else
                rethrow(err)
            end
        end
    end
    if throttled
        (verbosity >= 1) && @warn "throttle unmanaged-order cancellations this tick" max_attempts=UNMANAGED_CANCEL_MAX_PER_STEP remaining_orders=max(0, unmanaged_total - cancel_attempts)
    end
    if cooldown_skips > 0
        (verbosity >= 1) && @warn "skip unmanaged-order cancellation due to transient private-read cooldown" skipped_orders=cooldown_skips cooldown_until=cache.mc[:unmanaged_cancel_cooldown_until] first_base=first_base first_symbol=first_symbol first_orderid=first_orderid error=first_cooldown
    end
    return nothing
end

function _advicebybase(tradeadvices::Vector{StrategyAdvice})::Dict{String, StrategyAdvice}
    bybase = Dict{String, StrategyAdvice}()
    for ta in tradeadvices
        base = uppercase(String(ta.base))
        haskey(bybase, base) && continue
        bybase[base] = ta
    end
    return bybase
end

function _close_side_from_label(tradelabel)
    if tradelabel in [longclose, longstrongclose]
        return "Sell"
    elseif tradelabel in [shortclose, shortstrongclose]
        return "Buy"
    end
    return nothing
end

"Return why one open trade should be blocked until opposite exposure is fully closed."
function _opentrade_block_reason(ta::StrategyAdvice, assets::AbstractDataFrame)::Union{Nothing, String}
    if !(ta.tradelabel in [longbuy, longstrongbuy, shortbuy, shortstrongbuy])
        return nothing
    end
    basekey = uppercase(String(ta.base))
    mask = uppercase.(String.(assets[!, :coin])) .== basekey
    freebase = hasproperty(assets, :free) ? Float32(sum(assets[mask, :free])) : 0f0
    borrowedbase = hasproperty(assets, :borrowed) ? Float32(sum(assets[mask, :borrowed])) : 0f0
    sellablelong = max(0f0, freebase - borrowedbase)

    if ta.tradelabel in [longbuy, longstrongbuy]
        if borrowedbase > 0f0
            return "short exposure remains (borrowedbase=$(round(Float64(borrowedbase); digits=6)))"
        end
    elseif ta.tradelabel in [shortbuy, shortstrongbuy]
        if sellablelong > 0f0
            return "long exposure remains (sellablelong=$(round(Float64(sellablelong); digits=6)))"
        end
    end
    return nothing
end

"Return cached OHLCV for a base when it exists in the exchange cache."
function _cachedohlcv(cache::TradeCache, base::AbstractString)
    haskey(cache.xc.bases, base) || return nothing
    try
        return CryptoXch.ohlcv(cache.xc, base)
    catch err
        (verbosity >= 1) && @warn "skip base because cached OHLCV lookup failed" base=base error=sprint(showerror, err)
        return nothing
    end
end

function _recentcloserejectstate!(cache::TradeCache)
    if !haskey(cache.mc, :recent_close_rejects)
        cache.mc[:recent_close_rejects] = Dict{Tuple{String, String}, NamedTuple{(:at, :reason), Tuple{DateTime, String}}}()
    end
    return cache.mc[:recent_close_rejects]
end

function _markclosereject!(cache::TradeCache, base::AbstractString, side::AbstractString, reason::AbstractString)
    nowdt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : cache.xc.currentdt
    _recentcloserejectstate!(cache)[(uppercase(String(base)), uppercase(String(side)))] = (at=nowdt, reason=String(reason))
    return nothing
end

function _recentcloserejectreason(cache::TradeCache, base::AbstractString, side::AbstractString; maxage=Minute(20))
    state = _recentcloserejectstate!(cache)
    key = (uppercase(String(base)), uppercase(String(side)))
    haskey(state, key) || return nothing
    entry = state[key]
    nowdt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : cache.xc.currentdt
    if (nowdt - entry.at) <= maxage
        return entry.reason
    end
    delete!(state, key)
    return nothing
end

function _ensure_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, tradeadvices::Vector{StrategyAdvice})
    advbybase = _advicebybase(tradeadvices)
    effectivebudgetquote = _effectivebudgetquote(cache, assets)
    allocatedbudgetquote = _allocatedbudgetquote(assets)
    overallocatedbudgetquote = max(0.0, allocatedbudgetquote - effectivebudgetquote)
    maxassetquote = cache.mc[:maxassetfraction] * effectivebudgetquote
    for base in _close_management_bases(cache, assets)
        cfgrow = _cfgrow_for_base(cache, base)
        sellenabled = isnothing(cfgrow) ? true : _cfgbool(cfgrow, :sellenabled, true)
        basecfg = _basecfg_for_close(cache, base, sellenabled)
        base_mask = uppercase.(String.(assets[!, :coin])) .== uppercase(String(base))
        freebase = Float32(sum(assets[base_mask, :free]))
        lockedbase = Float32(sum(assets[base_mask, :locked]))
        borrowedbase = Float32(sum(assets[base_mask, :borrowed]))
        price = try
            Float32(currentprice(CryptoXch.ohlcv(cache.xc, base)))
        catch
            0f0
        end
        longexposurequote = max(0f0, (freebase + lockedbase) * price)
        shortexposurequote = max(0f0, borrowedbase * price)
        for closelabel in _positioncloselabels(assets, base; sellenabled=sellenabled)

            ta = if haskey(advbybase, base)
                deepcopy(advbybase[base])
            else
                StrategyAdvice(classifier=cache.cl, base=base, datetime=isnothing(cache.xc.currentdt) ? Dates.now() : cache.xc.currentdt)
            end
            if (closelabel == longclose) && ((longexposurequote > maxassetquote) || ((overallocatedbudgetquote > 0.0) && (longexposurequote > 0f0)))
                ta.tradelabel = longstrongclose
            elseif (closelabel == shortclose) && ((shortexposurequote > maxassetquote) || ((overallocatedbudgetquote > 0.0) && (shortexposurequote > 0f0)))
                ta.tradelabel = shortstrongclose
            else
                ta.tradelabel = closelabel
            end
            ta.price = _strategy_sell_limitprice(cache, base, ta.tradelabel)
            ta.source = :managedclose
            ta.allowreversal = false

            try
                trade!(cache, basecfg, ta, assets)
            catch err
                if _ispermissionrestrictederror(err)
                    _objective4_inc!(cache, :objective4_order_rejects)
                    _objective4_inc!(cache, :objective4_permission_rejects)
                    _disablerestrictedbase!(cache, base, sprint(showerror, err))
                elseif _isinsufficientfundserror(err)
                    _objective4_inc!(cache, :objective4_order_rejects)
                    side = _close_side_from_label(closelabel)
                    !isnothing(side) && _markclosereject!(cache, base, side, sprint(showerror, err))
                    (verbosity >= 1) && @warn "skip managed close order due to insufficient funds" base=base error=sprint(showerror, err)
                elseif _isreduceonlynopositionerror(err)
                    _objective4_inc!(cache, :objective4_order_rejects)
                    (verbosity >= 1) && @warn "skip managed close order because reduce-only close found no open position" base=base error=sprint(showerror, err)
                elseif _isprivatecooldownerror(err)
                    _objective4_inc!(cache, :objective4_privatecooldown_skips)
                    (verbosity >= 1) && @warn "skip managed close order due to transient private-read cooldown" base=base error=sprint(showerror, err)
                else
                    rethrow(err)
                end
            end
        end
    end
    return nothing
end

function _getgainsalgo_lane_advices(gs::TradingStrategy.GainSegment, ta::StrategyAdvice)::Vector{StrategyAdvice}
    advices = StrategyAdvice[]
    buygain_zero = abs(Float32(gs.buygain)) <= eps(Float32)

    long_open = gs.longta.label in [longbuy, longstrongbuy]
    short_open = gs.shortta.label in [shortbuy, shortstrongbuy]
    @assert !(long_open && short_open) "objective-7 consistency violation: simultaneous long and short open labels are not allowed; longta=$(gs.longta.label), shortta=$(gs.shortta.label)"

    if long_open
        openprice = (gs.longta.label == longstrongbuy || buygain_zero) ? nothing : (gs.longta.openprice > 0f0 ? Float32(gs.longta.openprice) : nothing)
        push!(advices, _strategyadvice(ta; tradelabel=gs.longta.label, limitprice=openprice, source=:getgainsalgo, allowreversal=false))
    end
    if gs.longta.closeprice > 0f0
        closelabel = gs.longta.label == longstrongclose ? longstrongclose : longclose
        closeprice = closelabel == longstrongclose ? nothing : Float32(gs.longta.closeprice)
        push!(advices, _strategyadvice(ta; tradelabel=closelabel, limitprice=closeprice, source=:getgainsalgo, allowreversal=false))
    end

    if short_open
        openprice = (gs.shortta.label == shortstrongbuy || buygain_zero) ? nothing : (gs.shortta.openprice > 0f0 ? Float32(gs.shortta.openprice) : nothing)
        push!(advices, _strategyadvice(ta; tradelabel=gs.shortta.label, limitprice=openprice, source=:getgainsalgo, allowreversal=false))
    end
    if gs.shortta.closeprice > 0f0
        closelabel = gs.shortta.label == shortstrongclose ? shortstrongclose : shortclose
        closeprice = closelabel == shortstrongclose ? nothing : Float32(gs.shortta.closeprice)
        push!(advices, _strategyadvice(ta; tradelabel=closelabel, limitprice=closeprice, source=:getgainsalgo, allowreversal=false))
    end

    return advices
end

function _getgainsalgo_advices!(cache::TradeCache, base::AbstractString, ta::StrategyAdvice, assets::AbstractDataFrame)::Vector{StrategyAdvice}
    ohlcv = _cachedohlcv(cache, base)
    if isnothing(ohlcv)
        (verbosity >= 1) && @warn "base OHLCV unavailable in exchange cache; skipping getgainsalgo advices" base=base
        return StrategyAdvice[]
    end
    history = _strategyhistory!(cache, base)
    _upsert_getgainsalgo_sample!(history, ohlcv, ta.tradelabel, ta.probability)
    gs = _strategystate!(cache, base)
    _set_gainsalgo_reconciliation!(gs, base, assets)
    lastix = length(history.scores)
    if lastix > 0
        TradingStrategy.getgains(gs, history.predictionsdf, history.scores, history.labels, false; lastix=lastix, openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
        lane_advices = _getgainsalgo_lane_advices(gs, ta)
        if !isempty(lane_advices)
            return lane_advices
        end
    end
    return StrategyAdvice[_strategyadvice(ta; source=:getgainsalgo)]
end

function _getgainsalgo_advice!(cache::TradeCache, base::AbstractString, ta::StrategyAdvice, assets::AbstractDataFrame)::StrategyAdvice
    advices = _getgainsalgo_advices!(cache, base, ta, assets)
    return advices[1]
end

"Return one symbol tick size for limit-price nudging; falls back to 0 when unavailable."
function _symbolticksize(cache::TradeCache, base::AbstractString)::Float32
    symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    syminfo = try
        CryptoXch._exchangesymbolinfo(cache.xc, symbol)
    catch
        nothing
    end
    if isnothing(syminfo)
        return 0f0
    end
    tick = try
        Float32(syminfo.ticksize)
    catch
        0f0
    end
    return (isfinite(tick) && tick > 0f0) ? tick : 0f0
end

"Price close-side reversal one tick more aggressively than its opposite open to increase close-first probability."
function _reversalcloseprice(cache::TradeCache, ta::StrategyAdvice)::Union{Nothing, Float32}
    isnothing(ta.price) && return nothing
    openprice = Float32(ta.price)
    tick = _symbolticksize(cache, ta.base)
    if ta.tradelabel in [longbuy, longstrongbuy]
        # close short is a buy; more aggressive means higher buy price.
        return tick > 0f0 ? (openprice + tick) : openprice
    elseif ta.tradelabel in [shortbuy, shortstrongbuy]
        # close long is a sell; more aggressive means lower sell price.
        return tick > 0f0 ? max(0f0, openprice - tick) : openprice
    end
    return nothing
end

function _expand_reversal_advice(cache::TradeCache, ta::StrategyAdvice, assets::AbstractDataFrame)::Vector{StrategyAdvice}
    if !ta.allowreversal
        return StrategyAdvice[ta]
    end
    base = ta.base
    freebase = sum(assets[assets[!, :coin] .== base, :free])
    borrowedbase = sum(assets[assets[!, :coin] .== base, :borrowed])
    if (ta.tradelabel in [longbuy, longstrongbuy]) && (borrowedbase > 0)
        closeadvice = deepcopy(ta)
        closeadvice.tradelabel = shortclose
        closeadvice.price = _reversalcloseprice(cache, ta)
        return StrategyAdvice[closeadvice, ta]
    elseif (ta.tradelabel in [shortbuy, shortstrongbuy]) && (freebase > 0)
        closeadvice = deepcopy(ta)
        closeadvice.tradelabel = longclose
        closeadvice.price = _reversalcloseprice(cache, ta)
        return StrategyAdvice[closeadvice, ta]
    end
    return StrategyAdvice[ta]
end

function _strategy_reconciliation_input(cache::TradeCache, base::AbstractString, assets::AbstractDataFrame)::TradingStrategy.StrategyReconciliationInput
    _ = cache
    mask = _assets_base_mask(assets, base)
    freebase = hasproperty(assets, :free) ? Float32(sum(assets[mask, :free])) : 0f0
    borrowedbase = hasproperty(assets, :borrowed) ? Float32(sum(assets[mask, :borrowed])) : 0f0
    pricehint = _asset_price_hint(assets, mask)
    return TradingStrategy.StrategyReconciliationInput(
        has_long_open=freebase > 0f0,
        long_avg_entry=pricehint,
        long_open_ix=0,
        has_short_open=borrowedbase > 0f0,
        short_avg_entry=pricehint,
        short_open_ix=0,
    )
end

function _snapshot_to_strategy_advices(cache::TradeCache, snap::TradingStrategy.StrategySnapshot)::Vector{StrategyAdvice}
    advices = StrategyAdvice[]
    base = String(snap.base)
    dt = snap.datetime
    cls = cache.cl
    cfgid = Int(snap.configid)
    prob = Float32(snap.probability)
    shared = snap.label

    function _push(lbl::Targets.TradeLabel, price::Union{Nothing, Float32})
        push!(advices, StrategyAdvice(
            classifier=cls,
            configid=cfgid,
            tradelabel=lbl,
            relativeamount=1f0,
            base=base,
            price=price,
            datetime=dt,
            hourlygain=0f0,
            probability=prob,
            investmentid=nothing,
            source=:tradingstrategy,
            allowreversal=false,
        ))
    end

    if shared in [longbuy, longstrongbuy]
        lp = (shared == longstrongbuy || snap.long_openprice <= 0f0) ? nothing : Float32(snap.long_openprice)
        _push(shared, lp)
    elseif shared in [shortbuy, shortstrongbuy]
        lp = (shared == shortstrongbuy || snap.short_openprice <= 0f0) ? nothing : Float32(snap.short_openprice)
        _push(shared, lp)
    end

    if snap.long_closeprice > 0f0
        closelabel = (shared == longstrongclose) ? longstrongclose : longclose
        cp = (closelabel == longstrongclose) ? nothing : Float32(snap.long_closeprice)
        _push(closelabel, cp)
    end
    if snap.short_closeprice > 0f0
        closelabel = (shared == shortstrongclose) ? shortstrongclose : shortclose
        cp = (closelabel == shortstrongclose) ? nothing : Float32(snap.short_closeprice)
        _push(closelabel, cp)
    end

    return advices
end

function _collect_strategy_advices(cache::TradeCache, assets::AbstractDataFrame)
    tradeadvices = StrategyAdvice[]
    closed_dt = _next_closed_candle_dt!(cache)
    if isnothing(closed_dt)
        return tradeadvices
    end
    evaldt = DateTime(closed_dt)
    if _runtime_api_enabled(cache)
        rt = _strategyruntime(cache)
        if !isnothing(rt)
            bases = hasproperty(cache.cfg, :basecoin) ? String.(cache.cfg[!, :basecoin]) : String[]
            recon = Dict{String, TradingStrategy.StrategyReconciliationInput}()
            for base in bases
                recon[uppercase(String(base))] = _strategy_reconciliation_input(cache, base, assets)
            end
            snaps = TradingStrategy.getsnapshots!(rt, cache.xc, bases, evaldt; reconciliation_by_base=recon)
            for snap in snaps
                append!(tradeadvices, _snapshot_to_strategy_advices(cache, snap))
            end
            _mark_closed_candle_consumed!(cache, evaldt)
            return tradeadvices
        end
    end

    Classify.supplement!(cache.cl)
    for basecfg in eachrow(cache.cfg)
        rawadvice = Classify.advice(cache.cl, basecfg.basecoin, evaldt, investment=nothing)
        if isnothing(rawadvice)
            (verbosity > 3) && println("no trade advice for $(basecfg.basecoin)")
            continue
        end
        sa = _strategyadvice(rawadvice; source=:classifier)
        if _strategyengine(cache) == :getgainsalgo
            append!(tradeadvices, _getgainsalgo_advices!(cache, basecfg.basecoin, sa, assets))
        else
            append!(tradeadvices, _expand_reversal_advice(cache, sa, assets))
        end
    end
    _mark_closed_candle_consumed!(cache, evaldt)
    return tradeadvices
end

function _log_tradeadvice_summary!(cache::TradeCache, tradeadvices::Vector{StrategyAdvice})
    verbosity >= 1 || return nothing
    isempty(tradeadvices) && return nothing

    counts = Dict{Targets.TradeLabel, Int}()
    short_bases = String[]
    for ta in tradeadvices
        counts[ta.tradelabel] = get(counts, ta.tradelabel, 0) + 1
        if ta.tradelabel in [shortbuy, shortstrongbuy]
            push!(short_bases, String(ta.base))
        end
    end

    parts = String[]
    labels = [longstrongbuy, longbuy, longstrongclose, longclose, shortstrongbuy, shortbuy, shortstrongclose, shortclose, allclose]
    for lbl in labels
        c = get(counts, lbl, 0)
        c > 0 || continue
        push!(parts, "$(String(Symbol(lbl)))=$(c)")
    end
    summary = isempty(parts) ? "none" : join(parts, ",")
    shortbases = isempty(short_bases) ? "none" : join(unique(short_bases), ",")
    @info "strategy advice summary" datetime=cache.xc.currentdt labels=summary short_open_bases=shortbases
    return nothing
end

function _should_refresh_tradeselection(cache::TradeCache)::Bool
    currentdt = cache.xc.currentdt
    if isnothing(currentdt)
        return false
    end
    refresh_times = get(cache.mc, :reloadtimes, Time[])
    currentminute = floor(currentdt, Minute(1))
    if !(Time(currentminute) in refresh_times)
        return false
    end
    lastrefresh = get(cache.mc, :last_traderefresh_dt, nothing)
    return isnothing(lastrefresh) || (lastrefresh != currentminute)
end

function _mark_tradeselection_refreshed!(cache::TradeCache)
    currentdt = cache.xc.currentdt
    cache.mc[:last_traderefresh_dt] = isnothing(currentdt) ? nothing : floor(currentdt, Minute(1))
    return cache
end

"Log a compact runtime `cache.cfg` summary after one trade-selection pass."
function _log_cachecfg_summary!(cache::TradeCache, stage::AbstractString)
    verbosity >= 1 || return nothing
    selected = size(cache.cfg, 1)
    buyenabled = hasproperty(cache.cfg, :buyenabled) ? Int(sum(Bool.(coalesce.(cache.cfg[!, :buyenabled], false)))) : 0
    sellenabled = hasproperty(cache.cfg, :sellenabled) ? Int(sum(Bool.(coalesce.(cache.cfg[!, :sellenabled], false)))) : 0
    base_preview = if hasproperty(cache.cfg, :basecoin) && (selected > 0)
        join(String.(cache.cfg[1:min(selected, 12), :basecoin]), ",")
    else
        "none"
    end
    @info "cache.cfg after tradeselection" stage=String(stage) datetime=cache.xc.currentdt selected=selected buyenabled=buyenabled sellenabled=sellenabled bases_preview=base_preview
    return nothing
end

function _maybe_refresh_tradeselection!(cache::TradeCache; assets::Union{Nothing, AbstractDataFrame}=nothing)
    if !_should_refresh_tradeselection(cache)
        return false
    end
    assets_df = isnothing(assets) ? CryptoXch.portfolio!(cache.xc) : assets
    (verbosity >= 1) && println("\n$(tradetime(cache)): start reassessing trading strategy")
    tradeselection!(cache, assets_df[!, :coin]; datetime=cache.xc.currentdt, updatecache=true)
    cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
    _log_cachecfg_summary!(cache, "refresh")
    _mark_tradeselection_refreshed!(cache)
    if verbosity >= 1
        selected = size(cache.cfg, 1)
        buyenabled = hasproperty(cache.cfg, :buyenabled) ? Int(sum(Bool.(coalesce.(cache.cfg[!, :buyenabled], false)))) : 0
        sellenabled = hasproperty(cache.cfg, :sellenabled) ? Int(sum(Bool.(coalesce.(cache.cfg[!, :sellenabled], false)))) : 0
        println("$(tradetime(cache)) reassessed trading strategy summary: selected=$(selected) buyenabled=$(buyenabled) sellenabled=$(sellenabled)")
    end
    (verbosity >= 3) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.cfg)"
    return true
end

"Return position-side gaps where no matching open close order currently exists."
function _positions_without_close_orders(cache::TradeCache, assets::AbstractDataFrame, oo::AbstractDataFrame)
    quote_coin = uppercase(String(EnvConfig.cryptoquote))
    missing = NamedTuple{(:base, :side, :qty, :required, :covered, :minimum), Tuple{String, String, Float32, Float32, Float32, Float32}}[]
    openorderids = Set{String}(String(orow.orderid) for orow in eachrow(oo) if hasproperty(orow, :orderid) && CryptoXch.openstatus(String(orow.status)))

    function _remaining_open_qty(orow)::Float32
        total = Float32(getproperty(orow, :baseqty))
        executed = hasproperty(orow, :executedqty) ? Float32(getproperty(orow, :executedqty)) : 0f0
        return max(0f0, total - executed)
    end

    function _covered_qty(symbol::AbstractString, side::AbstractString; require_leverage::Union{Nothing, Bool}=nothing)::Float32
        wanted_side = uppercase(String(side))
        total = 0f0
        for orow in eachrow(oo)
            CryptoXch.openstatus(String(orow.status)) || continue
            (String(orow.symbol) == String(symbol)) || continue
            (uppercase(String(orow.side)) == wanted_side) || continue
            if !isnothing(require_leverage)
                (_orderisleverage(orow) == require_leverage) || continue
            end
            total += _remaining_open_qty(orow)
        end
        return total
    end

    function _managed_covered_qty(base::AbstractString, side::AbstractString)::Float32
        entry = if uppercase(String(side)) == "SELL"
            _managedcloseget(cache, base, longclose)
        else
            _managedcloseget(cache, base, shortclose)
        end
        isnothing(entry) && return 0f0
        oid = haskey(entry, :orderid) ? String(entry[:orderid]) : ""
        (isempty(oid) || (oid in openorderids)) && return 0f0
        return Float32(max(0f0, Float32(get(entry, :baseqty, 0f0))))
    end

    function _min_base_qty(base::AbstractString, symbol::AbstractString, row)::Float32
        price = try
            hasproperty(row, :usdtprice) ? Float32(getproperty(row, :usdtprice)) : 0f0
        catch
            0f0
        end
        if !(price > 0f0)
            price = try
                Float32(currentprice(CryptoXch.ohlcv(cache.xc, base)))
            catch
                0f0
            end
        end
        syminfo = try
            CryptoXch.minimumqty(cache.xc, symbol)
        catch
            nothing
        end
        if isnothing(syminfo)
            return 0f0
        end
        if price > 0f0
            return Float32(1.01f0 * max(Float32(syminfo.minbaseqty), Float32(syminfo.minquoteqty) / price))
        end
        return Float32(1.01f0 * Float32(syminfo.minbaseqty))
    end

    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        base == quote_coin && continue
        freebase = Float32(row.free)
        borrowedbase = Float32(row.borrowed)
        if (freebase <= 0f0) && (borrowedbase <= 0f0)
            continue
        end

        symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
        minimumbasequantity = _min_base_qty(base, symbol, row)
        sellablelong = max(0f0, freebase - borrowedbase)
        if sellablelong > 0f0
            # Long-close orders are spot sells (non-leverage). Exclude short-entry margin sells.
            sell_covered = _covered_qty(symbol, "Sell"; require_leverage=false) + _managed_covered_qty(base, "Sell")
            sell_gap = max(0f0, sellablelong - sell_covered)
            (sell_gap >= minimumbasequantity) && push!(missing, (base=base, side="Sell", qty=sell_gap, required=sellablelong, covered=sell_covered, minimum=minimumbasequantity))
        end
        if borrowedbase > 0f0
            # Short-close orders are margin buys. Exclude long-entry spot buys.
            buy_covered = _covered_qty(symbol, "Buy"; require_leverage=true) + _managed_covered_qty(base, "Buy")
            buy_gap = max(0f0, borrowedbase - buy_covered)
            (buy_gap >= minimumbasequantity) && push!(missing, (base=base, side="Buy", qty=buy_gap, required=borrowedbase, covered=buy_covered, minimum=minimumbasequantity))
        end
    end
    return missing
end

"""
Execute one trading tick: reconstruct managed close-order state from live open orders,
cancel unmanaged orders, keep one close order active per open robot-owned position,
execute open/reversal entries, and handle daily trade-selection reload.
Called by the loop runners once per iterate step.
"""
function _tradestep!(cache::TradeCache)
    (verbosity > 3) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")
    _set_ws_runtime_flags!(cache)
    _objective4_inc!(cache, :objective4_cycle_count)
    stale_openorders_snapshot = false
    oo = try
        CryptoXch.getopenorders(cache.xc)
    catch err
        if _isprivatecooldownerror(err)
            prev = _openorderssnapshot(cache)
            if size(prev, 1) > 0
                (verbosity >= 1) && @warn "using previous openorders snapshot due to transient private-read cooldown" rows=size(prev, 1) error=sprint(showerror, err)
                stale_openorders_snapshot = true
                prev
            else
                (verbosity >= 1) && @warn "skip tradestep due to transient private-read cooldown and missing openorders snapshot" error=sprint(showerror, err)
                return nothing
            end
        else
            rethrow(err)
        end
    end
    cache.mc[:openorders_snapshot_stale] = stale_openorders_snapshot
    cache.mc[:openorders_snapshot] = oo
    _refreshactiveopenbuysymbols!(cache, oo)
    _refreshactiveopensellsymbols!(cache, oo)
    assets = CryptoXch.portfolio!(cache.xc)
    _sync_exchange_balances_snapshot!(cache, assets)
    _run_ws_shadow_compare!(cache, oo, assets)
    _reconstruct_managed_close_orders!(cache, assets, oo)
    _cancel_unmanaged_open_orders!(cache, oo)
    backlog_drain = get(cache.mc, :backlog_drain_mode, false)
    equity_snapshot = _update_account_equity_snapshot!(cache)
    _maybe_writeportfoliosnapshot!(cache, assets)
    tradeadvices = StrategyAdvice[]
    if backlog_drain
        (verbosity >= 1) && @warn "backlog drain mode active; skipping managed-close maintenance and strategy order placement" unmanaged_open_orders=get(cache.mc, :unmanaged_open_orders, 0) threshold=UNMANAGED_BACKLOG_DRAIN_THRESHOLD
    else
        pre_sync_closed_dt = get(cache.mc, :strategy_last_closed_candle_dt, nothing)
        tradeadvices = _collect_strategy_advices(cache, assets)
        _log_tradeadvice_summary!(cache, tradeadvices)
        if Bool(get(cache.mc, :async_engine_enabled, false))
            sync_count = length(tradeadvices)
            async_advices = _run_async_shadow_topology!(cache, assets, oo, tradeadvices; seed_closed_dt=pre_sync_closed_dt)
            compare_ok = _run_async_shadow_compare!(cache, tradeadvices, async_advices)
            if !Bool(get(cache.mc, :async_shadow_mode, true))
                if compare_ok
                    tradeadvices = async_advices
                    (verbosity >= 1) && @info "async engine full cutover active" sync_count=sync_count async_count=length(async_advices)
                else
                    (verbosity >= 1) && @warn "async full cutover requested but compare did not pass; keeping synchronous execution authoritative" compare_ok=compare_ok
                end
            end
        end
        _ensure_managed_close_orders!(cache, assets, tradeadvices)
    end
    if cache.mc[:usenewtrade] || backlog_drain
    else # legacy trade!()
        openedlongbases = String[]
        openedshortbases = String[]
        closedlongbases = String[]
        closedshortbases = String[]
        sort!(tradeadvices, lt=tradeadvicelessthan)  # close first, then buy high-gain first
        for ta in tradeadvices
            if _strategyengine(cache) == :getgainsalgo && _isclosetrade(ta.tradelabel)
                continue
            end
            block_reason = _opentrade_block_reason(ta, assets)
            if !isnothing(block_reason)
                (verbosity >= 1) && @info "skip open trade until opposite close is fully filled" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) reason=block_reason
                continue
            end
            rowix = hasproperty(cache.cfg, :basecoin) ? findfirst(==(ta.base), cache.cfg[!, :basecoin]) : nothing
            if isnothing(rowix)
                (verbosity >= 1) && @warn "skip trade advice because base is missing in runtime config" base=ta.base
                continue
            end
            basecfg = cache.cfg[rowix, :]
            res = try
                trade!(cache, basecfg, ta, assets)
            catch err
                if _ispermissionrestrictederror(err)
                    _objective4_inc!(cache, :objective4_order_rejects)
                    _objective4_inc!(cache, :objective4_permission_rejects)
                    _disablerestrictedbase!(cache, ta.base, sprint(showerror, err))
                    nothing
                elseif _isinsufficientfundserror(err)
                    _objective4_inc!(cache, :objective4_order_rejects)
                    side = _close_side_from_label(ta.tradelabel)
                    !isnothing(side) && _markclosereject!(cache, ta.base, side, sprint(showerror, err))
                    (verbosity >= 1) && @warn "skip trade advice due to insufficient funds" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) error=sprint(showerror, err)
                    nothing
                elseif _isprivatecooldownerror(err)
                    _objective4_inc!(cache, :objective4_privatecooldown_skips)
                    (verbosity >= 1) && @warn "skip trade advice due to transient private-read cooldown" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) error=sprint(showerror, err)
                    nothing
                else
                    rethrow(err)
                end
            end
            if !isnothing(res) && (res.trade in [longbuy, longstrongbuy])
                push!(openedlongbases, basecfg.basecoin)
            elseif !isnothing(res) && (res.trade in [shortbuy, shortstrongbuy])
                push!(openedshortbases, basecfg.basecoin)
            elseif !isnothing(res) && (res.trade in [longstrongclose, longclose])
                push!(closedlongbases, basecfg.basecoin)
            elseif !isnothing(res) && (res.trade in [shortclose, shortstrongclose])
                push!(closedshortbases, basecfg.basecoin)
            elseif !isnothing(res)
                @warn "case not handled: $res"
            end
        end
        equity_delta_text = isnothing(equity_snapshot.equity_delta) ? "delta=NA" : "delta=$(round(Int, equity_snapshot.equity_delta))"
        (verbosity >= 2) && println("\r$(tradetime(cache)): equity=$(round(Int, equity_snapshot.equity_quote)), $(equity_delta_text), $(USDTmsg(cache, assets)), opened long: $(openedlongbases), opened short: $(openedshortbases), closed long: $(closedlongbases), closed short: $(closedshortbases)                                          ")
    end

    # Avoid extra live API calls: only refresh post-trade portfolio in simulation mode.
    assets_after = (cache.xc.mc[:simmode] == CryptoXch.nosimulation) ? assets : CryptoXch.portfolio!(cache.xc)

    # Live safety summary: highlight open positions that currently have no opposite-side close order.
    # In simulation mode orders can fill immediately, so there is no persistent open-order coverage to inspect.
    if (cache.xc.mc[:simmode] == CryptoXch.nosimulation) && (cache.mc[:trademode] in [buysell, closeonly, quickexit]) && !backlog_drain
        oo_after = oo
        cache.mc[:openorders_snapshot] = oo_after
        missing = _positions_without_close_orders(cache, assets_after, oo_after)
        if !isempty(missing)
            details = String[]
            rejected = 0
            for x in missing
                reason = _recentcloserejectreason(cache, x.base, x.side)
                if isnothing(reason)
                    push!(details, "$(x.base):$(x.side):cause=no_active_close_order gap=$(round(Float64(x.qty); digits=6)) req=$(round(Float64(x.required); digits=6)) cov=$(round(Float64(x.covered); digits=6)) min=$(round(Float64(x.minimum); digits=6))")
                else
                    rejected += 1
                    push!(details, "$(x.base):$(x.side):cause=recent_reject reason=$(reason) gap=$(round(Float64(x.qty); digits=6)) req=$(round(Float64(x.required); digits=6)) cov=$(round(Float64(x.covered); digits=6)) min=$(round(Float64(x.minimum); digits=6))")
                end
            end
            (verbosity >= 1) && @warn "open positions without active close order" count=length(missing) recentrejects=rejected details=details
        end
    end

    _maybe_refresh_tradeselection!(cache; assets=assets_after)
    #TODO low prio: for closed orders check fees
    #TODO low prio: aggregate orders and transactions in bookkeeping
    return nothing
end

"Load or derive the initial trade configuration if `cache.cfg` is empty."
function _ensure_tradeloop_initialized!(cache::TradeCache)
    if size(cache.cfg, 1) == 0
        assets = CryptoXch.balances(cache.xc)
        (verbosity >= 1) && print("\r$(tradetime(cache)): start calculating trading strategy on the fly")
        tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.startdt)
        cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
        _log_cachecfg_summary!(cache, "initial")
        if verbosity >= 1
            selected = size(cache.cfg, 1)
            buyenabled = hasproperty(cache.cfg, :buyenabled) ? Int(sum(Bool.(coalesce.(cache.cfg[!, :buyenabled], false)))) : 0
            sellenabled = hasproperty(cache.cfg, :sellenabled) ? Int(sum(Bool.(coalesce.(cache.cfg[!, :sellenabled], false)))) : 0
            @info "initial trading strategy summary" datetime=cache.xc.startdt selected=selected buyenabled=buyenabled sellenabled=sellenabled
        end
        (verbosity >= 3) && @info "$(tradetime(cache)) initial trading strategy: $(cache.cfg)"
    end
end

"Log end-of-loop summary statistics."
function _tradefinish!(cache::TradeCache)
    (verbosity >= 2) && println("$(tradetime(cache)): finished trading core loop")
    (verbosity >= 3) && @info (size(cache.xc.closedorders, 1) > 0) ? "$(EnvConfig.now()): closed orders log $(cache.xc.closedorders)" : "$(EnvConfig.now()): no closed orders"
    (verbosity >= 3) && @info (size(cache.xc.orders, 1) > 0) ? "$(EnvConfig.now()): open orders log $(cache.xc.orders)" : "$(EnvConfig.now()): no open orders"
    (verbosity >= 2) && @info "$(EnvConfig.now()): closed orders $(size(cache.xc.closedorders, 1)), open orders $(size(cache.xc.orders, 1))"
    assets = CryptoXch.portfolio!(cache.xc)
    (verbosity >= 3) && @info "assets = $assets"
    (verbosity >= 2) && @info "total $(EnvConfig.cryptoquote) = $(sum(assets.usdtvalue))"
end

"""
Shared iteration engine used by both backtest and live runners.
Advances through `cache.xc` one tick at a time, calling `_tradestep!` each step.
Respects `pause!`/`resume!`/`stop!` loop control requests.
"""
function _run_tradeloop!(cache::TradeCache)
    _setloopstate!(cache, loop_running)
    try
        for c in cache.xc
            st = _waitforactive_loopstate!(cache)
            (st == loop_stopping) && break
            _tradestep!(cache)
        end
    catch ex
        if isa(ex, InterruptException)
            (verbosity >= 0) && println("\nCtrl+C pressed within tradeloop")
        else
            (verbosity >= 0) && @error "exception=$ex"
            bt = catch_backtrace()
            for ptr in bt
                frame = StackTraces.lookup(ptr)
                for fr in frame
                    if occursin("CryptoTimeSeries", string(fr.file))
                        (verbosity >= 1) && println("fr.func=$(fr.func) fr.file=$(fr.file) fr.line=$(fr.line)")
                    end
                end
            end
        end
    finally
        _setloopstate!(cache, loop_stopped)
    end
    _tradefinish!(cache)
    return cache
end

"""
Run a full backtest replay over the cached OHLCV window defined by `cache.xc.startdt`…`cache.xc.enddt`.
When `skip_init=false` (default) the trade configuration is loaded or rebuilt if `cache.cfg` is empty.
Pass `skip_init=true` when the caller has already populated `cache.cfg`.
"""
function run_backtest!(cache::TradeCache; skip_init::Bool=false)
    skip_init || _ensure_tradeloop_initialized!(cache)
    _run_tradeloop!(cache)
    return cache
end

"""
Run the live trading loop, advancing one minute per tick and sleeping until the next wall-clock minute.
When `skip_init=false` (default) the trade configuration is loaded or rebuilt if `cache.cfg` is empty.
Pass `skip_init=true` when the caller has already populated `cache.cfg`.
"""
function run_live!(cache::TradeCache; skip_init::Bool=false)
    skip_init || _ensure_tradeloop_initialized!(cache)
    _run_tradeloop!(cache)
    return cache
end

end  # module

