# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, longbuy when an uptrend starts and longclose when it ends.
It generates the OHLCV data, executes the trades in a loop and selects the basecoins to trade.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, Statistics
using EnvConfig, Ohlcv, CryptoXch, Classify, Targets, TradeLog, TradingStrategy

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
const ORDER_AMEND_PRICE_REL_THRESHOLD_DEFAULT = 1f-3
const TRADE_CYCLE_BUDGET_SECONDS_DEFAULT = 60.0

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

function _envfloat(name::AbstractString, default::Real)::Float64
    raw = _envstr(name, string(default))
    parsed = tryparse(Float64, strip(raw))
    if isnothing(parsed) || !isfinite(parsed)
        return Float64(default)
    end
    return Float64(parsed)
end

function _exchangewsdefault(exchange::AbstractString)::Bool
    ex = uppercase(strip(String(exchange)))
    return (ex == uppercase(CryptoXch.EXCHANGE_KRAKENSPOT)) || (ex == uppercase(CryptoXch.EXCHANGE_KRAKENFUTURES))
end

function _init_runtime_state!(mc::AbstractDict, defaultws::Bool)
    mc[:ws_orders_enabled] = _envtrue("CTS_WS_ORDERS_ENABLED", get(mc, :ws_orders_enabled, defaultws))
    mc[:ws_balances_enabled] = _envtrue("CTS_WS_BALANCES_ENABLED", get(mc, :ws_balances_enabled, defaultws))
    mc[:ws_primary_mode] = _envtrue("CTS_WS_PRIMARY_MODE", get(mc, :ws_primary_mode, defaultws))
    mc[:tradable_ohlcv_state_by_base] = get(mc, :tradable_ohlcv_state_by_base, Dict{String, Symbol}())
    mc[:tradable_ohlcv_state_dt_by_base] = get(mc, :tradable_ohlcv_state_dt_by_base, Dict{String, DateTime}())
    mc[:strategy_last_closed_candle_dt] = get(mc, :strategy_last_closed_candle_dt, nothing)
    mc[:strategy_closed_candle_pending_reason] = get(mc, :strategy_closed_candle_pending_reason, nothing)
    return mc
end

function _set_ws_runtime_flags!(cache)
    cache.xc.mc[:ws_orders_enabled] = Bool(get(cache.mc, :ws_orders_enabled, false))
    cache.xc.mc[:ws_balances_enabled] = Bool(get(cache.mc, :ws_balances_enabled, false))
    cache.xc.mc[:ws_primary_mode] = Bool(get(cache.mc, :ws_primary_mode, false))
    return nothing
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

function _setstrategyruntimefromsegment!(mc::AbstractDict, gs::TradingStrategy.GainSegment, source::AbstractString)
    mc[:strategy_template] = deepcopy(gs)
    mc[:strategy_algorithm] = gs.algorithm
    mc[:strategy_source] = String(source)
    return mc
end

function _portfoliototal(assets::AbstractDataFrame)::Float64
    return size(assets, 1) == 0 ? 0.0 : Float64(sum(assets[!, :usdtvalue]))
end

function _asset_quote_totals(assets::AbstractDataFrame)::NamedTuple{(:totalusdt, :quotefree), Tuple{Float64, Float64}}
    cols = propertynames(assets)
    totalusdt = (:usdtvalue in cols) ? Float64(sum(assets[!, :usdtvalue])) : 0.0
    quotefree = 0.0
    if (:coin in cols) && (:free in cols)
        quotecoin = uppercase(String(EnvConfig.cryptoquote))
        for row in eachrow(assets)
            if uppercase(String(row.coin)) == quotecoin
                quotefree += max(0.0, Float64(row.free))
            end
        end
    end
    return (totalusdt=max(0.0, totalusdt), quotefree=max(0.0, quotefree))
end

function _resolve_capacity_quote(cache, assets::AbstractDataFrame; context::Symbol=:trade)
    cap = CryptoXch.accountcapacity(cache.xc)
    totals = _asset_quote_totals(assets)

    equity_cap = max(0.0, Float64(get(cap, :equity_quote, 0.0)))
    opening_cap = max(0.0, Float64(get(cap, :available_opening_quote, 0.0)))
    long_cap = max(0.0, Float64(get(cap, :available_long_quote, opening_cap)))
    short_cap = max(0.0, Float64(get(cap, :available_short_quote, opening_cap)))
    equity_quote = equity_cap
    available_opening_quote = opening_cap
    available_long_quote = long_cap
    available_short_quote = short_cap

    info_th = clamp(Float64(get(cache.mc, :capacity_divergence_info_threshold, 0.02)), 0.0, 10.0)
    warn_th = clamp(Float64(get(cache.mc, :capacity_divergence_warn_threshold, 0.05)), 0.0, 10.0)
    warn_th = max(warn_th, info_th)
    if (equity_cap > 0.0) && (totals.totalusdt > 0.0)
        rel = abs(equity_cap - totals.totalusdt) / max(equity_cap, totals.totalusdt, 1e-9)
        if rel > info_th
            nowdt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : DateTime(cache.xc.currentdt)
            lastwarn = get(cache.mc, :capacity_divergence_last_warn_dt, nothing)
            shouldwarn = isnothing(lastwarn) || ((nowdt - DateTime(lastwarn)) >= Dates.Minute(10))
            if shouldwarn
                cache.mc[:capacity_divergence_last_warn_dt] = nowdt
                cache.mc[:capacity_divergence_events] = Int(get(cache.mc, :capacity_divergence_events, 0)) + 1
                if rel >= warn_th
                    @warn "capacity/asset equity divergence requires action" context equity_cap totals_totalusdt=totals.totalusdt reldiff=rel source=get(cap, :source, "unknown") info_threshold=info_th warn_threshold=warn_th
                else
                    @info "capacity/asset equity divergence within expected range" context equity_cap totals_totalusdt=totals.totalusdt reldiff=rel source=get(cap, :source, "unknown") info_threshold=info_th warn_threshold=warn_th
                end
            end
        end
    end

    diag = (
        context=context,
        equity_cap=equity_cap,
        equity_quote=equity_quote,
        opening_cap=opening_cap,
        available_opening_quote=available_opening_quote,
        long_cap=long_cap,
        available_long_quote=available_long_quote,
        short_cap=short_cap,
        available_short_quote=available_short_quote,
        asset_total_quote=totals.totalusdt,
        asset_quote_free=totals.quotefree,
        source=get(cap, :source, "unknown"),
    )
    cache.mc[:capacity_last_diagnostic] = diag
    return diag
end

"Return the effective trading budget in quote currency, capped by `mc[:maxbudgetquote]` and reduced by safety margin when configured."
function _effectivebudgetquote(cache, assets::AbstractDataFrame)::Float64
    capdiag = _resolve_capacity_quote(cache, assets; context=:budget)
    available_opening_quote = Float64(capdiag.available_opening_quote)
    safetymargin = Float64(get(cache.mc, :budgetsafetymargin, 0.0))
    safetymargin = clamp(safetymargin, 0.0, 0.99)
    budgetwithsafety = max(0.0, available_opening_quote * (1.0 - safetymargin))
    maxbudget = get(cache.mc, :maxbudgetquote, nothing)
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

"Return TradeLog event timestamp in UTC; prefer cache.xc.currentdt when available."
function _tradelogeventtimeutc(cache)::DateTime
    if !isnothing(cache.xc.currentdt)
        return DateTime(cache.xc.currentdt)
    end
    return Dates.now(Dates.UTC)
end

function _writeportfoliosnapshot!(cache, assets::AbstractDataFrame; source_module::AbstractString="Trade")
    rowcount = size(assets, 1)
    simmode = String(Symbol(cache.xc.mc[:simmode]))
    event_time = _tradelogeventtimeutc(cache)
    created_at = Dates.now(Dates.UTC)
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
                created_at_utc=created_at,
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
                created_at_utc=created_at,
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
        cache.mc[:restrictedcoins] = String[] # coins excluded from the robot universe (e.g. account-region restrictions)
        cache.mc[:whitelistcoins] = ["ADA", "AI16Z", "APEX", "AAVE", "BNB", "BTC", "CAKE", "DOGE", "ELX", "ENA", "ETH", "HBAR", "HFT", "JUP", "LINK", "LTC", "MNT", "ONDO", "PEPE", "POPCAT", "S", "SOL", "SUI", "TON", "TRX", "VIRTUAL", "W", "WAL", "WIF", "WLD", "X", "XLM", "XRP"] 
        # not whitelisted: "ANIME", "B3", "BERA", "CMETH", "LDO", "PLUME", "SOSO", "TRUMP"
        cache.mc[:hourlygainlimit] = 0.1f0 # limit hourly gain to a realistic 10% max
        cache.mc[:maxassetfraction] = 0.1f0 # defines the maximum ratio of (a specific asset) / ( total assets) - only close trades, if this is exceeded
        cache.mc[:maxbudgetquote] = nothing # optional overall quote-currency budget cap; if set, trading uses min(totalusdt, maxbudgetquote)
        cache.mc[:budgetsafetymargin] = 0.05 # budget limit uses sum(balance) * (1 - budgetsafetymargin)
        cache.mc[:capacity_divergence_info_threshold] = _envfloat("CTS_CAPACITY_DIVERGENCE_INFO_THRESHOLD", 0.02)
        cache.mc[:capacity_divergence_warn_threshold] = _envfloat("CTS_CAPACITY_DIVERGENCE_WARN_THRESHOLD", 0.05)
        cache.mc[:capacity_divergence_events] = 0
        cache.mc[:capacity_last_diagnostic] = nothing
        cache.mc[:capacity_divergence_last_warn_dt] = nothing
        cache.mc[:reloadtimes] = [Time("04:00:00")]
        cache.mc[:last_traderefresh_dt] = nothing
        cache.mc[:trademode] = trademode  # see TradeMode definition above
        cache.mc[:strategy_engine] = :getgainsalgo  # runtime strategy source metadata
        cache.mc[:managed_close_orders] = Dict{String, Dict{Symbol, Any}}()  # per-base reconstructed/managed close orders
        cache.mc[:openorders_snapshot] = DataFrame()
        cache.mc[:cyclebudgetseconds] = TRADE_CYCLE_BUDGET_SECONDS_DEFAULT
        cache.mc[:pending_unmanaged_cancel_orderids] = String[]
        cache.mc[:pending_managed_close_bases] = String[]
        cache.mc[:pending_tradeadvice_bases] = String[]
        _setstrategyruntimefromsegment!(cache.mc, TradingStrategy.GainSegment(), "default")
        cache.mc[:strategy_runtime] = try
            TradingStrategy.GainSegmentRuntime(classifier=cl, strategy=deepcopy(cache.mc[:strategy_template]), source="default")
        catch
            nothing
        end
        cache.mc[:tradelog_portfolio_snapshot_mode] = :all  # :all, :session_start, :none
        cache.mc[:tradelog_portfolio_snapshot_written] = false
        cache.mc[:loop_state] = loop_idle
        _init_runtime_state!(cache.mc, _exchangewsdefault(CryptoXch.exchange(cache.xc)))
        (verbosity >= 4) && println("TradeCache trademode = $(cache.mc[:trademode]), maxassetfraction = $(cache.mc[:maxassetfraction]), maxbudgetquote = $(cache.mc[:maxbudgetquote]), reloadtimes = $(cache.mc[:reloadtimes]), exitcoins = $(cache.mc[:exitcoins]), whitelistcoins = $(cache.mc[:whitelistcoins])")
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

function _tradeselection_history_minutes(tc::TradeCache)::Int
    rt = _strategyruntime(tc)
    isnothing(rt) && error("objective-7 runtime API is mandatory: missing strategy runtime in TradeCache")
    classifier_minutes = max(0, Int(TradingStrategy.requiredhistoryminutes(rt)))
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

"""Return the wall-clock processing budget in seconds for one trade cycle."""
function _cyclebudgetseconds(cache::TradeCache)::Float64
    budget = Float64(get(cache.mc, :cyclebudgetseconds, TRADE_CYCLE_BUDGET_SECONDS_DEFAULT))
    return max(0.0, budget)
end

"""Start one new wall-clock budget window for the current trading cycle."""
function _startcyclebudget!(cache::TradeCache)::Nothing
    budget_seconds = _cyclebudgetseconds(cache)
    cache.mc[:cycle_budget_started_ns] = time_ns()
    cache.mc[:cycle_budget_deadline_ns] = cache.mc[:cycle_budget_started_ns] + round(Int, budget_seconds * 1_000_000_000)
    cache.mc[:cycle_budget_overrun_stage] = nothing
    return nothing
end

"""Return `true` when the current cycle has exhausted its wall-clock budget."""
function _cyclebudgetexpired(cache::TradeCache)::Bool
    deadline = get(cache.mc, :cycle_budget_deadline_ns, nothing)
    isnothing(deadline) && return false
    return time_ns() >= Int(deadline)
end

"""Record one cycle overrun stage and emit a concise warning about pending work."""
function _markcycleoverrun!(cache::TradeCache, stage::AbstractString; pending::Int=0)::Nothing
    cache.mc[:cycle_budget_overrun_stage] = String(stage)
    (verbosity >= 1) && @warn "trade cycle budget exceeded; deferring pending work to next cycle" datetime=cache.xc.currentdt stage=String(stage) pending
    return nothing
end

"""Return one case-insensitive pending priority map preserving the original pending order."""
function _pendingprioritymap(pending::AbstractVector)::Dict{String, Int}
    priority = Dict{String, Int}()
    for (ix, item) in enumerate(pending)
        key = uppercase(String(item))
        haskey(priority, key) || (priority[key] = ix)
    end
    return priority
end

"""Return `items` reordered so pending entries come first while preserving relative order."""
function _pendingfirststrings(items::AbstractVector, pending::AbstractVector)::Vector{String}
    priority = _pendingprioritymap(pending)
    normalized = String[String(item) for item in items]
    return sort(normalized; by=item -> (haskey(priority, uppercase(item)) ? 0 : 1, get(priority, uppercase(item), typemax(Int))))
end

"""Group strategy advices by base while preserving the first base occurrence order."""
function _group_tradeadvices_by_base(tradeadvices)
    grouped = Dict{String, Vector{Any}}()
    ordered_bases = String[]
    for ta in tradeadvices
        base = uppercase(String(ta.base))
        if !haskey(grouped, base)
            grouped[base] = Any[]
            push!(ordered_bases, base)
        end
        push!(grouped[base], ta)
    end
    return [(base=base, advices=grouped[base]) for base in ordered_bases]
end

"""Return strategy-advice groups reordered so previously deferred bases are handled first."""
function _pendingfirstadvicegroups(groups, pending_bases::AbstractVector)
    priority = _pendingprioritymap(pending_bases)
    return sort(groups; by=group -> (haskey(priority, uppercase(group.base)) ? 0 : 1, get(priority, uppercase(group.base), typemax(Int))))
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
    rt = _strategyruntime(cache)
    if !isnothing(rt)
        try
            TradingStrategy.dropbase!(rt, base_upper)
        catch err
            (verbosity >= 1) && @warn "failed removing restricted base from strategy runtime" base=base_upper error=sprint(showerror, err)
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
        rt = _strategyruntime(tc)
        !isnothing(rt) && TradingStrategy.dropbase!(rt, rb)
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
        rt = _strategyruntime(tc)
        !isnothing(rt) && TradingStrategy.dropbase!(rt, rb)
    end

    selectedbases = String[]
    rt = _strategyruntime(tc)
    isnothing(rt) && error("objective-7 runtime API is mandatory: missing strategy runtime in TradeCache during tradeselection")
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

    # `addassetsconfig!` can be called repeatedly by dashboards/live loops.
    # Drop previously joined asset columns to keep leftjoin idempotent.
    stale_asset_cols = String[]
    for col in names(assets)
        if (col != "coin") && (col in names(tc.cfg))
            push!(stale_asset_cols, col)
        end
    end
    if "coin" in names(tc.cfg)
        push!(stale_asset_cols, "coin")
    end
    if !isempty(stale_asset_cols)
        select!(tc.cfg, Not(unique(stale_asset_cols)))
    end

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

function _requested_limitprice(cache::TradeCache, ta::StrategyAdvice, fallback_price::Real)
    if ta.tradelabel in [longstrongbuy, shortstrongbuy, longstrongclose, shortstrongclose]
        return nothing
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
    capdiag = _resolve_capacity_quote(cache, assets; context=:trade)
    equityquote = Float64(capdiag.equity_quote)
    totalusdt = Float64(capdiag.asset_total_quote)
    if equityquote <= 0
        @warn "equityquote=$equityquote is insufficient, totalusdt=$totalusdt, assets=$assets"
        return nothing
    end
    freeusdt = Float64(capdiag.available_long_quote)
    freeshortquote = Float64(capdiag.available_short_quote)

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
            existing_tif = uppercase(String(something(get(existing, :timeinforce, ""), "")))
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
            existing_tif = uppercase(String(something(get(existing, :timeinforce, ""), "")))
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

_strategyengine(cache::TradeCache) = Symbol(get(cache.mc, :strategy_engine, :getgainsalgo))

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
    haskey(mc, :strategy_template) || error("missing strategy_template in Trade strategy runtime config")
    gs = mc[:strategy_template]
    gs isa TradingStrategy.GainSegment || error("strategy_template must be a TradingStrategy.GainSegment, got $(typeof(gs))")

    openthreshold = Float32(gs.openthreshold)
    closethreshold = Float32(gs.closethreshold)
    buygain = Float32(gs.buygain)
    sellgain = Float32(gs.sellgain)
    limitreduction = Float32(gs.limitreduction)
    minpricedelta = Float32(gs.minpricedelta)
    max_classify_staleness_minutes = Int(gs.max_classify_staleness_minutes)
    maxwindow = Int(gs.maxwindow)

    @assert 0f0 <= openthreshold <= 1f0 "strategy_openthreshold must be in [0, 1], got $(openthreshold)"
    @assert 0f0 <= closethreshold <= 1f0 "strategy_closethreshold must be in [0, 1], got $(closethreshold)"
    @assert 0f0 <= buygain <= 1f0 "strategy_buygain must be in [0, 1], got $(buygain)"
    @assert 0f0 <= sellgain <= 1f0 "strategy_sellgain must be in [0, 1], got $(sellgain)"
    @assert 0f0 <= limitreduction <= 1f0 "strategy_limitreduction must be in [0, 1], got $(limitreduction)"
    @assert 0f0 <= minpricedelta <= 1f0 "strategy_minpricedelta must be in [0, 1], got $(minpricedelta)"
    @assert max_classify_staleness_minutes >= 0 "strategy_max_classify_staleness_minutes must be >= 0, got $(max_classify_staleness_minutes)"
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

    managed_close_orders = get!(mc, :managed_close_orders, Dict{String, Dict{Symbol, Any}}())
    empty!(managed_close_orders)
    return _validatestrategyconfig!(mc)
end

function apply_tradingstrategy!(mc::AbstractDict, cfg::NamedTuple; strategy_engine::Symbol=:getgainsalgo, source::AbstractString="manual")
    hasproperty(cfg, :tradingstrategy) || error("strategy config payload must define tradingstrategy, got $(typeof(cfg))")
    return apply_tradingstrategy!(mc, getproperty(cfg, :tradingstrategy); strategy_engine=strategy_engine, source=source)
end

function apply_tradingstrategy!(cache::TradeCache, cfg::NamedTuple; strategy_engine::Symbol=:getgainsalgo, source::AbstractString="manual")
    hasproperty(cfg, :tradingstrategy) || error("strategy config payload must define tradingstrategy, got $(typeof(cfg))")
    return apply_tradingstrategy!(cache, getproperty(cfg, :tradingstrategy); strategy_engine=strategy_engine, source=source)
end

function apply_tradingstrategy!(cache::TradeCache, gs::TradingStrategy.GainSegment; strategy_engine::Symbol=:getgainsalgo, source::AbstractString="manual")
    apply_tradingstrategy!(cache.mc, gs; strategy_engine=strategy_engine, source=source)
    rt = _strategyruntime(cache)
    !isnothing(rt) && TradingStrategy.apply_strategy!(rt, gs; source=source)
    return cache
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

function _strategy_sell_limitprice(cache::TradeCache, base::AbstractString, tradelabel::Targets.TradeLabel; assets::Union{Nothing, AbstractDataFrame}=nothing)
    if tradelabel in [longstrongclose, shortstrongclose]
        return nothing
    end
    rt = _strategyruntime(cache)
    isnothing(rt) && return nothing
    evaldt = isnothing(cache.xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : DateTime(cache.xc.currentdt)
    recon = isnothing(assets) ? TradingStrategy.StrategyReconciliationInput() : _strategy_reconciliation_input(cache, base, assets)
    snap = try
        TradingStrategy.getsnapshot!(rt, cache.xc, base, evaldt; reconciliation=recon)
    catch err
        (verbosity >= 1) && @warn "failed to fetch strategy snapshot for managed close price" base=String(base) tradelabel=String(Symbol(tradelabel)) error=sprint(showerror, err)
        nothing
    end
    isnothing(snap) && return nothing
    if tradelabel == longclose
        v = Float32(snap.long_closeprice)
        return v > 0f0 ? v : nothing
    elseif tradelabel == shortclose
        v = Float32(snap.short_closeprice)
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

function _cancel_unmanaged_open_orders!(cache::TradeCache, oo::AbstractDataFrame)::Bool
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
    pending_orderids = get(cache.mc, :pending_unmanaged_cancel_orderids, String[])
    pending_priority = _pendingprioritymap(pending_orderids)
    cancel_order = sortperm(candidates; by = c -> (
        haskey(pending_priority, uppercase(c.orderid)) ? 0 : 1,
        get(pending_priority, uppercase(c.orderid), typemax(Int)),
        (get(dupcounts, (c.symbol, c.side, c.is_leverage), 0) > 1) ? 0 : 1,
        c.created,
        c.symbol,
        c.orderid,
    ))

    cancel_attempts = 0
    throttled = false
    cooldown_skips = 0
    first_cooldown = nothing
    first_base = ""
    first_symbol = ""
    first_orderid = ""
    for (pos, ix) in enumerate(cancel_order)
        if _cyclebudgetexpired(cache)
            cache.mc[:pending_unmanaged_cancel_orderids] = [candidates[cancel_order[j]].orderid for j in pos:length(cancel_order)]
            _markcycleoverrun!(cache, "unmanaged_open_orders"; pending=length(cache.mc[:pending_unmanaged_cancel_orderids]))
            return false
        end
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
    cache.mc[:pending_unmanaged_cancel_orderids] = String[]
    return true
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

function _ensure_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, tradeadvices::Vector{StrategyAdvice})::Bool
    advbybase = _advicebybase(tradeadvices)
    effectivebudgetquote = _effectivebudgetquote(cache, assets)
    allocatedbudgetquote = _allocatedbudgetquote(assets)
    overallocatedbudgetquote = max(0.0, allocatedbudgetquote - effectivebudgetquote)
    maxassetquote = cache.mc[:maxassetfraction] * effectivebudgetquote
    closebases = _pendingfirststrings(_close_management_bases(cache, assets), get(cache.mc, :pending_managed_close_bases, String[]))
    for bix in eachindex(closebases)
        if _cyclebudgetexpired(cache)
            cache.mc[:pending_managed_close_bases] = closebases[bix:end]
            _markcycleoverrun!(cache, "managed_close_orders"; pending=length(cache.mc[:pending_managed_close_bases]))
            return false
        end
        base = closebases[bix]
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
            if ta.tradelabel in [longstrongclose, shortstrongclose]
                ta.price = nothing
            elseif isnothing(ta.price)
                ta.price = _strategy_sell_limitprice(cache, base, ta.tradelabel; assets=assets)
            end
            ta.source = :managedclose
            ta.allowreversal = false

            try
                trade!(cache, basecfg, ta, assets)
            catch err
                if _ispermissionrestrictederror(err)
                    _disablerestrictedbase!(cache, base, sprint(showerror, err))
                elseif _isinsufficientfundserror(err)
                    side = _close_side_from_label(closelabel)
                    !isnothing(side) && _markclosereject!(cache, base, side, sprint(showerror, err))
                    (verbosity >= 1) && @warn "skip managed close order due to insufficient funds" base=base error=sprint(showerror, err)
                elseif _isreduceonlynopositionerror(err)
                    (verbosity >= 1) && @warn "skip managed close order because reduce-only close found no open position" base=base error=sprint(showerror, err)
                elseif _isprivatecooldownerror(err)
                    (verbosity >= 1) && @warn "skip managed close order due to transient private-read cooldown" base=base error=sprint(showerror, err)
                else
                    rethrow(err)
                end
            end
        end
    end
    cache.mc[:pending_managed_close_bases] = String[]
    return true
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
    rt = _strategyruntime(cache)
    if isnothing(rt)
        throw(ArgumentError("strategy runtime is required but missing in TradeCache (mc[:strategy_runtime])"))
    end

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

"""
Apply a shared post-selection filter to `tc.cfg`.

Supported modes:
- `:tradeloop` keeps rows where `buyenabled || sellenabled`.
- `:accepted` keeps rows where `classifieraccepted` is true.
"""
function filtertradeconfig!(tc::TradeCache; mode::Symbol=:tradeloop)
    (size(tc.cfg, 1) == 0) && return tc.cfg
    if mode == :tradeloop
        tc.cfg = tc.cfg[(tc.cfg[!, :buyenabled] .|| tc.cfg[:, :sellenabled]), :]
    elseif mode == :accepted
        tc.cfg = tc.cfg[coalesce.(tc.cfg[!, :classifieraccepted], false), :]
    else
        error("unsupported filtertradeconfig mode=$(mode); expected :tradeloop or :accepted")
    end
    return tc.cfg
end

function _maybe_refresh_tradeselection!(cache::TradeCache; assets::Union{Nothing, AbstractDataFrame}=nothing)
    if !_should_refresh_tradeselection(cache)
        return false
    end
    assets_df = isnothing(assets) ? CryptoXch.portfolio!(cache.xc) : assets
    (verbosity >= 1) && println("\n$(tradetime(cache)): start reassessing trading strategy")
    tradeselection!(cache, assets_df[!, :coin]; datetime=cache.xc.currentdt, updatecache=true)
    filtertradeconfig!(cache; mode=:tradeloop)
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
    _startcyclebudget!(cache)
    _set_ws_runtime_flags!(cache)
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
    _reconstruct_managed_close_orders!(cache, assets, oo)
    cancel_completed = _cancel_unmanaged_open_orders!(cache, oo)
    cancel_completed || return nothing
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
        managed_close_completed = _ensure_managed_close_orders!(cache, assets, tradeadvices)
        managed_close_completed || return nothing
    end
    if !backlog_drain
        openedlongbases = String[]
        openedshortbases = String[]
        closedlongbases = String[]
        closedshortbases = String[]
        sort!(tradeadvices, lt=tradeadvicelessthan)  # close first, then buy high-gain first
        advicegroups = _pendingfirstadvicegroups(_group_tradeadvices_by_base(tradeadvices), get(cache.mc, :pending_tradeadvice_bases, String[]))
        tradeadvices_completed = true
        for gix in eachindex(advicegroups)
            if _cyclebudgetexpired(cache)
                cache.mc[:pending_tradeadvice_bases] = [advicegroups[j].base for j in gix:length(advicegroups)]
                _markcycleoverrun!(cache, "trade_advices"; pending=length(cache.mc[:pending_tradeadvice_bases]))
                tradeadvices_completed = false
                break
            end
            for ta in advicegroups[gix].advices
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
                    _disablerestrictedbase!(cache, ta.base, sprint(showerror, err))
                    nothing
                elseif _isinsufficientfundserror(err)
                    side = _close_side_from_label(ta.tradelabel)
                    !isnothing(side) && _markclosereject!(cache, ta.base, side, sprint(showerror, err))
                    (verbosity >= 1) && @warn "skip trade advice due to insufficient funds" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) error=sprint(showerror, err)
                    nothing
                elseif _isprivatecooldownerror(err)
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
        end
        tradeadvices_completed || return nothing
        cache.mc[:pending_tradeadvice_bases] = String[]
        equity_delta_text = isnothing(equity_snapshot.equity_delta) ? "delta=NA" : "delta=$(round(Int, equity_snapshot.equity_delta))"
        (verbosity >= 2) && println("\r$(tradetime(cache)): equity=$(round(Int, equity_snapshot.equity_quote)), $(equity_delta_text), $(USDTmsg(cache, assets)), opened long: $(openedlongbases), opened short: $(openedshortbases), closed long: $(closedlongbases), closed short: $(closedshortbases)                                          ")
    end  # !backlog_drain

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
        filtertradeconfig!(cache; mode=:tradeloop)
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
    total_steps = 0
    total_step_seconds = 0.0
    try
        for c in cache.xc
            st = _waitforactive_loopstate!(cache)
            (st == loop_stopping) && break
            step_start_ns = time_ns()
            _tradestep!(cache)
            step_seconds = (time_ns() - step_start_ns) / 1_000_000_000
            total_steps += 1
            total_step_seconds += step_seconds
            cache.mc[:last_step_seconds] = step_seconds
            cache.mc[:avg_step_seconds] = total_step_seconds / max(total_steps, 1)

            # In live mode, sustained per-minute processing over 60s means TT lag risk.
            if isnothing(cache.xc.enddt) && (cache.xc.mc[:simmode] == CryptoXch.nosimulation) && (step_seconds > 45.0)
                @warn "slow trade step in live mode" datetime=cache.xc.currentdt step_seconds avg_step_seconds=cache.mc[:avg_step_seconds] steps=total_steps
            end
            if (verbosity >= 2) && (total_steps % 30 == 0)
                println("$(tradetime(cache)) tradeloop performance: steps=$(total_steps), avg_step_seconds=$(round(cache.mc[:avg_step_seconds], digits=3)), last_step_seconds=$(round(step_seconds, digits=3))")
            end
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

