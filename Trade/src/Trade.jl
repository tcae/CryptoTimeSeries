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
# Backward compatibility alias (deprecated): `openclose` == `buysell`.
const openclose = buysell

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

function _setstrategyruntimefromsegment!(mc::AbstractDict, gs::TradingStrategy.GainSegment, source::AbstractString)
    mc[:strategy_template] = deepcopy(gs)
    mc[:strategy_algorithm] = gs.algorithm
    mc[:strategy_source] = String(source)
    return mc
end

@inline function _classifier_advice(cl, base, dt)
    return getproperty(Classify, :advice)(cl, base, dt, investment=nothing)
end

function _portfoliototal(assets::AbstractDataFrame)::Float64
    return size(assets, 1) == 0 ? 0.0 : Float64(sum(assets[!, :usdtvalue]))
end

"Return the effective trading budget in quote currency, capped by `mc[:maxbudgetquote]` when configured."
function _effectivebudgetquote(cache, assets::AbstractDataFrame)::Float64
    totalusdt = _portfoliototal(assets)
    maxbudget = get(cache.mc, :maxbudgetquote, get(cache.mc, :maxbudgetusdt, nothing))
    if isnothing(maxbudget)
        return totalusdt
    end
    cap = Float64(maxbudget)
    if !isfinite(cap) || (cap <= 0.0)
        return totalusdt
    end
    return min(totalusdt, cap)
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

function _writeportfoliosnapshot!(cache, assets::AbstractDataFrame; source_module::AbstractString="Trade")
    rowcount = size(assets, 1)
    simmode = String(Symbol(cache.xc.mc[:simmode]))
    event_time = Dates.now(Dates.UTC)
    portfolio_total = _portfoliototal(assets)
    cash_after = _portfolioquotevalue(assets)
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
                run_mode=CryptoXch.auditrunmode(cache.xc),
                run_id=CryptoXch.auditrunid(cache.xc),
                exchange=exchange_name,
                account_alias=account_alias,
                routing_role=TradeLog.routing_trade_exchange_spot,
                market_type=TradeLog.market_unknown,
                asset_class=TradeLog.crypto,
                instrument_type=TradeLog.instrument_unknown,
                symbol="PORTFOLIO",
                cash_after=cash_after,
                portfolio_value_after=portfolio_total,
                notes="rows=0; simmode=$(simmode)"
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
                run_mode=CryptoXch.auditrunmode(cache.xc),
                run_id=CryptoXch.auditrunid(cache.xc),
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
                notes="asset=$(coin); rows=$(rowcount); simmode=$(simmode)"
            )
            TradeLog.writeeventwithhash(event)
        end
    catch audit_error
        (verbosity >= 1) && @warn "failed to persist portfolio snapshot" exception=(audit_error, catch_backtrace())
    end
    return nothing
end

"Write portfolio audit snapshots according to `cache.mc[:audit_portfolio_snapshot_mode]`."
function _maybe_writeportfoliosnapshot!(cache, assets::AbstractDataFrame)
    mode = get(cache.mc, :audit_portfolio_snapshot_mode, :all)
    if mode == :none
        return nothing
    elseif mode == :session_start
        if !get(cache.mc, :audit_portfolio_snapshot_written, false)
            _writeportfoliosnapshot!(cache, assets)
            cache.mc[:audit_portfolio_snapshot_written] = true
        end
        return nothing
    elseif mode == :all
        _writeportfoliosnapshot!(cache, assets)
        return nothing
    end
    @warn "unknown audit portfolio snapshot mode=$(mode); expected :all, :session_start or :none"
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
        cache.mc[:reloadtimes] = [Time("04:00:00")]
        cache.mc[:last_traderefresh_dt] = nothing
        cache.mc[:trademode] = trademode  # see TradeMode definition above
        cache.mc[:usenewtrade] = false # implementation switch between old and new trade! method
        cache.mc[:strategy_engine] = :classifier  # :classifier (legacy) or :getgainsalgo
        cache.mc[:strategy_state] = Dict{String, Any}()  # per-base TradingStrategy.GainSegment
        cache.mc[:strategy_history] = Dict{String, Any}()  # per-base rolling price+signal history
        cache.mc[:managed_close_orders] = Dict{String, Dict{Symbol, Any}}()  # per-base reconstructed/managed close orders
        _setstrategyruntimefromsegment!(cache.mc, TradingStrategy.GainSegment(), "default")
        cache.mc[:audit_portfolio_snapshot_mode] = :all  # :all, :session_start, :none
        cache.mc[:audit_portfolio_snapshot_written] = false
        cache.mc[:loop_state] = loop_idle
        (verbosity >= 4) && println("TradeCache trademode = $(cache.mc[:trademode]), maxassetfraction = $(cache.mc[:maxassetfraction]), maxbudgetquote = $(cache.mc[:maxbudgetquote]), reloadtimes = $(cache.mc[:reloadtimes]), exitcoins = $(cache.mc[:exitcoins]), whitelistcoins = $(cache.mc[:whitelistcoins]), longopencoins = $(cache.mc[:longopencoins]), shortopencoins = $(cache.mc[:shortopencoins])")
        return cache
    end
end

"""
Script-facing strategy-layer configuration that bundles one cross-coin strategy stack.
The stack is expected to be shared across selected coins in one run.
"""
Base.@kwdef struct StrategyLayerConfig
    configname::String = ""
    featconfig = nothing
    targetconfig = nothing
    classifiermodel = nothing
    tradingstrategy::TradingStrategy.GainSegment
end

"""
Script-facing trade runtime configuration.
Owns allocation and refresh controls while referencing one strategy layer.
"""
Base.@kwdef struct TradeRuntimeConfig
    maxassetfraction::Float32 = 0.1f0
    maxbudgetquote::Union{Nothing, Float64} = nothing
    maxbudgetusdt::Union{Nothing, Float64} = nothing
    reloadtimes::Vector{Time} = [Time("04:00:00")]
    whitelistcoins::Vector{String} = String[]
    restrictedcoins::Vector{String} = String[]
    strategy_engine::Symbol = :getgainsalgo
end

"Apply runtime controls from script config to the trade cache."
function apply_runtime_config!(cache::TradeCache, cfg::TradeRuntimeConfig)
    cache.mc[:maxassetfraction] = Float32(cfg.maxassetfraction)
    if !isnothing(cfg.maxbudgetquote) && !isnothing(cfg.maxbudgetusdt)
        @assert Float64(cfg.maxbudgetquote) == Float64(cfg.maxbudgetusdt) "maxbudgetquote=$(cfg.maxbudgetquote) must match deprecated maxbudgetusdt=$(cfg.maxbudgetusdt)"
    end
    runtimebudget = !isnothing(cfg.maxbudgetquote) ? Float64(cfg.maxbudgetquote) : (isnothing(cfg.maxbudgetusdt) ? nothing : Float64(cfg.maxbudgetusdt))
    cache.mc[:maxbudgetquote] = runtimebudget
    cache.mc[:maxbudgetusdt] = runtimebudget
    cache.mc[:reloadtimes] = collect(cfg.reloadtimes)
    cache.mc[:strategy_engine] = Symbol(cfg.strategy_engine)
    if !isempty(cfg.whitelistcoins)
        cache.mc[:whitelistcoins] = uppercase.(strip.(String.(cfg.whitelistcoins)))
    end
    if !isempty(cfg.restrictedcoins)
        cache.mc[:restrictedcoins] = unique(uppercase.(strip.(String.(cfg.restrictedcoins))))
    end
    return cache
end

"Apply a strategy layer config coming from caller scripts (TrendDetector style wiring)."
function apply_strategy_layer!(cache::TradeCache, cfg::StrategyLayerConfig)
    if !isnothing(cfg.classifiermodel)
        cache.cl = cfg.classifiermodel()
    end
    source = isempty(cfg.configname) ? "strategy-layer" : "strategy-layer:$(cfg.configname)"
    apply_tradingstrategy!(cache, cfg.tradingstrategy; strategy_engine=:getgainsalgo, source=source)
    return cache
end

function Base.show(io::IO, cache::TradeCache)
    println(io::IO, "TradeCache: trademode = $(cache.mc[:trademode]), maxassetfraction = $(cache.mc[:maxassetfraction]), maxbudgetquote = $(cache.mc[:maxbudgetquote]), reloadtimes = $(cache.mc[:reloadtimes])")
    println(io::IO, "TradeCache: exitcoins = $(cache.mc[:exitcoins]), whitelistcoins = $(cache.mc[:whitelistcoins]), longopencoins = $(cache.mc[:longopencoins]), shortopencoins = $(cache.mc[:shortopencoins])")
    print(io::IO, "TradeCache: startdt=$(cache.xc.startdt) currentdt=$(cache.xc.currentdt) enddt=$(cache.xc.enddt)")
end

ohlcvdf(cache, base) = Ohlcv.dataframe(cache.bd[base].ohlcv)
ohlcv(cache, base) = cache.bd[base].ohlcv
classifier(cache, base) = cache.bd[base].classifier
backtest(cache) = cache.backtestperiod >= Dates.Minute(1)
dummytime() = DateTime("2000-01-01T00:00:00")

function _tradeselection_history_minutes(tc::TradeCache)::Int
    classifier_minutes = try
        Int(Classify.requiredminutes(tc.cl))
    catch
        0
    end
    runtime_minutes = try
        Int(TradingStrategy.requiredhistoryminutes(_strategyruntime(tc)))
    catch
        0
    end
    liquidity_minutes = Int(Ohlcv.ld.checkperiod + Ohlcv.ld.accumulate + LIQUIDITY_LOOKBACK_MARGIN_MINUTES)
    return max(classifier_minutes + 1, runtime_minutes + 1, liquidity_minutes, 24 * 60)
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
    mask = (ot .> startdt) .&& (ot .<= enddt)
    if !any(mask)
        return 0.0
    end
    if :quotevolume in propertynames(df)
        return Float64(sum(Float64.(df[mask, :quotevolume])))
    end
    @assert (:basevolume in propertynames(df)) && (:close in propertynames(df)) "OHLCV dataframe must include quotevolume or basevolume+close; names=$(names(df))"
    basevol = Float64.(df[mask, :basevolume])
    closes = Float64.(df[mask, :close])
    return sum(basevol .* closes)
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

function _ensure_marketview_ohlcv!(tc::TradeCache, base::AbstractString, startdt::DateTime, enddt::DateTime)
    loaded = Set(String.(CryptoXch.bases(tc.xc)))
    if String(base) in loaded
        ohlcv = CryptoXch.ohlcv(tc.xc, base)
        CryptoXch.cryptoupdate!(tc.xc, ohlcv, startdt, enddt)
        return ohlcv
    end
    return CryptoXch.cryptodownload(tc.xc, base, "1m", startdt, enddt)
end

"Build a synthetic USDT market snapshot from OHLCV at `datetime` for simulation/backtest selection."
function _simulated_usdtmarketview(tc::TradeCache, datetime::DateTime, bases::Set{String}, history_startdt::DateTime)::DataFrame
    rows = NamedTuple[]
    for base in sort!(collect(bases))
        isempty(base) && continue
        CryptoXch.validbase(tc.xc, base) || continue
        ohlcv = _ensure_marketview_ohlcv!(tc, base, history_startdt, datetime)
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
        push!(rows, (basecoin=String(base), quotevolume24h=Float64(quotevolume24h), pricechangepercent=Float32(pricechangepercent), lastprice=Float32(lastprice)))
    end
    return isempty(rows) ? DataFrame(basecoin=String[], quotevolume24h=Float64[], pricechangepercent=Float32[], lastprice=Float32[]) : DataFrame(rows)
end

"""
Return the canonical audit partition root for the currently routed spot-trading venue.

The returned folder is scoped to the current environment, run mode, exchange,
account alias, asset class, and instrument type so ownership reconstruction does
not mix fills from different exchanges or market types.
"""
function _spot_audit_partition_root(cache::TradeCache)::String
    exchange_name = CryptoXch._routeexchange(cache.xc.routing, CryptoXch.trade_exchange_spot, CryptoXch.exchange(cache.xc))
    account_alias = exchange_name
    return joinpath(
        TradeLog.auditroot(),
        "environment=$(string(Symbol(EnvConfig.configmode)))",
        "run_mode=$(CryptoXch.auditrunmode(cache.xc))",
        "exchange=$(exchange_name)",
        "account=$(account_alias)",
        "asset_class=$(String(Symbol(TradeLog.crypto)))",
        "instrument_type=$(String(Symbol(TradeLog.spot_pair)))",
    )
end

"""Return one audit payload field as a normalized string."""
function _auditstring(event::AbstractDict, key::AbstractString)::String
    value = get(event, key, "")
    return (ismissing(value) || isnothing(value)) ? "" : String(value)
end

"""Return one audit payload field as `Float64`, defaulting to `0.0` when absent."""
function _auditfloat(event::AbstractDict, key::AbstractString)::Float64
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
Accumulate one filled audit event into directional robot-owned exposure.

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
Reconstruct directional robot-owned quantities per base from audit fills.

Only fills from the currently routed spot-trading audit partition are considered,
which keeps ownership separated by exchange/account scope and prevents cross-venue
mixing. Long and short quantities are tracked independently.
"""
function _robotownedqtymap(cache::TradeCache, bases)::Dict{String, NamedTuple{(:longqty, :shortqty), Tuple{Float32, Float32}}}
    wanted = Set(uppercase.(String.(bases)))
    owned = Dict{String, NamedTuple{(:longqty, :shortqty), Tuple{Float32, Float32}}}()
    isempty(wanted) && return owned

    partition_root = _spot_audit_partition_root(cache)
    isdir(partition_root) || return owned

    for (root, _, files) in walkdir(partition_root)
        "events.jsonl" in files || continue
        for event in TradeLog.readjsonlauditevents(joinpath(root, "events.jsonl"))
            event_type = uppercase(_auditstring(event, "event_type"))
            event_type in ["ORDER_PARTIAL_FILL", "ORDER_FILLED"] || continue
            _auditstring(event, "routing_role") == String(Symbol(TradeLog.routing_trade_exchange_spot)) || continue

            base = uppercase(_auditstring(event, "baseasset"))
            if isempty(base)
                pair = CryptoXch.basequote(_auditstring(event, "symbol"))
                base = isnothing(pair) ? "" : uppercase(String(pair.basecoin))
            end
            isempty(base) && continue
            base in wanted || continue

            side = _auditstring(event, "side")
            fillqty = _auditfloat(event, "fill_base_qty")
            leverage = _auditfloat(event, "leverage")
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

"Return true when Kraken private-read cooldown/rate-limit transiently blocks order flow." 
function _isprivatecooldownerror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("private read cooldown", msg) || occursin("rate limit", msg)
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
    try
        TradingStrategy.dropbase!(_strategyruntime(cache), base_upper)
    catch err
        (verbosity >= 1) && @warn "failed removing restricted base from strategy runtime" base=base_upper error=sprint(showerror, err)
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
    assetbaseset = Set(filter(!isnothing, assetbase_tokens))
    whitelistset = Set(filter(!isnothing, whitelist_tokens))
    restrictedset = _restrictedbaseset(tc, quotecoin)
    assetbaseset = setdiff(assetbaseset, restrictedset)
    whitelistset = setdiff(whitelistset, restrictedset)
    history_minutes = _tradeselection_history_minutes(tc)
    history_startdt = datetime - Minute(history_minutes)

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    # CryptoXch.removeallbases(tc.xc)  #* reuse what is in cache
    # Classify.removebase!(tc.cl, nothing)  #* reuse what is in cache

    marketbases = assetonly ? Set(String.(collect(assetbaseset))) : Set(String.(collect(union(assetbaseset, whitelistset, Set(String.(CryptoXch.bases(tc.xc)))))))
    marketbases = setdiff(marketbases, restrictedset)
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
        usdtdf = filter(row -> !(String(row.basecoin) in restrictedset), usdtdf)
    end
    (verbosity >= 3) && println("USDT market of size=$(size(usdtdf, 1)) at $datetime")
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent, :lastprice)
    if size(tc.cfg, 1) == 0
        tc.cfg[:, :datetime] = DateTime[]
        tc.cfg[:, :minquotevol] = Bool[]
        tc.cfg[:, :continuousminvol] = Bool[]
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
    tc.cfg[:, :inportfolio] = [base in assetbaseset for base in tc.cfg[!, :basecoin]]
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
        Classify.removebase!(tc.cl, rb)
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
        if updatecache
            Ohlcv.write(ohlcv) # write ohlcv even if data length is too short to calculate features
        end
        row.continuousminvol = true #TODO check disabled until debugged _continuous_liquidity_now(Ohlcv.dataframe(ohlcv), datetime, minquotevol=5000f0, accumulate=60, checkperiod=24*60, threshold=0.8)
        if row.inportfolio || (row.whitelisted && row.minquotevol)
            push!(candidatebaseset, String(row.basecoin))
        end
    end

    # Keep classifier/feature workload limited to liquidity candidates and portfolio holdings.
    for rb in setdiff(Set(CryptoXch.bases(tc.xc)), candidatebaseset)
        CryptoXch.removebase!(tc.xc, rb)
        Classify.removebase!(tc.cl, rb)
    end

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

    tc.cfg[:, :classifieraccepted] = [base in classifierbaseset for base in tc.cfg[!, :basecoin]]
    _sync_tradeflags!(tc; assetonly=assetonly)
    (verbosity >= 2) && println("$(CryptoXch.ttstr(tc.xc)) result of tradeselection! $(tc.cfg)")
    # tc.cfg = tc.cfg[(tc.cfg[!, :buyenabled] .|| tc.cfg[:, :sellenabled]), :]
    (verbosity >= 2) && println("$(EnvConfig.now()) #tc.cfg=$(size(tc.cfg, 1)) sum(classifieraccepted)=$(sum(tc.cfg[!, :classifieraccepted])) classifierbases($(length(classifierbases)))=$(classifierbases) ")

    if !assetonly
        (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) trained trade config on the fly including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    end
    return tc
end

"Adds usdtprice and usdtvalue added as well as the portfolio dataframe to trade config and returns trade config and portfolio as tuple"
function addassetsconfig!(tc::TradeCache, assets=CryptoXch.portfolio!(tc.xc))
    sort!(assets, [:coin])  # for readability only

    tc.cfg = leftjoin(tc.cfg, assets, on = :basecoin => :coin)
    tc.cfg = tc.cfg[!, Not([:borrowed, :accruedinterest, :locked, :free])]
    tc.cfg[:, :inportfolio] = .!ismissing.(tc.cfg[:, :usdtvalue])
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
    tradeadvice::Vector # vector of all used trade advices
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

function _strategyadvice(ta; tradelabel::Targets.TradeLabel=ta.tradelabel, limitprice::Union{Nothing, Real}=nothing, source::Symbol=:classifier, allowreversal::Bool=true)
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
function policyenforcement(cache::TradeCache, tavec::AbstractVector, assets::AbstractDataFrame)
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

function _withtradeauditcontext(f::Function, cache::TradeCache, ta)
    signal_score = try
        Float64(ta.probability)
    catch
        missing
    end
    strategy_engine = String(Symbol(_strategyengine(cache)))
    strategy_ref = string(get(cache.mc, :strategy_source, "default"))
    CryptoXch.setauditcontext!(
        cache.xc;
        strategy_engine=strategy_engine,
        strategy_config_ref=strategy_ref,
        signal_label=_tradetolabeltext(ta.tradelabel),
        signal_score=signal_score,
    )
    try
        return f()
    finally
        CryptoXch.clearauditcontext!(cache.xc)
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
function tradeamount(cache::TradeCache, tavec::AbstractVector, assets::AbstractDataFrame) #TODO consider negative short amounts
    tadf = policyenforcement(cache, tavec, assets) # returns a dataframe with tradeadvice per line plus corresponding asset info
    if sze(tadf, 1) > 0
        tadf.minquoteqty = [minimumquotequantity(xc, base) for base in tadf[!, :basecoin]]
        _tradeamounts!(tadf)
        sort!(tadf, [order(:tradelabel, by=_traderank), order(:hourlygain, rev=true)])  # order such that close before open and high before low hourlygain
        ta.basetradeqty .=0f0
        ta.vol1hmedian .= 0f0
        ta.baseamount .= 0f0
        for tarow in tadf
            ohlcv = CryptoXch.ohlcv(cache.xc, base)
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
            oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : CryptoXch.createbuyorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=ta.basetradeqty, maker=true, marginleverage=2)
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
function trade!(cache::TradeCache, basecfg::DataFrameRow, ta, assets::AbstractDataFrame)
    return trade!(cache, basecfg, _strategyadvice(ta; source=:classifier), assets)
end

function _requested_limitprice(cache::TradeCache, ta::StrategyAdvice, fallback_price::Real)
    return isnothing(ta.price) ? _orderlimitprice(cache, fallback_price) : Float32(ta.price)
end

function trade!(cache::TradeCache, basecfg::DataFrameRow, ta::StrategyAdvice, assets::AbstractDataFrame)
    sellbuyqtyratio = 2 # longclose qty / longbuy qty per order, if > 1 longclose quicker than buying it
    qtyacceleration = 4 # if > 1 then increase longbuy and longclose order qty by this factor
    short_margin_leverage = 2
    result = nothing
    base = ta.base
    totalusdt = sum(assets.usdtvalue)
    if totalusdt <= 0
        @warn "totalusdt=$totalusdt is insufficient, assets=$assets"
        return nothing
    end
    effectivebudgetquote = _effectivebudgetquote(cache, assets)
    if effectivebudgetquote <= 0
        (verbosity > 2) && println("$(tradetime(cache)) skip $base: effectivebudgetquote=$effectivebudgetquote is insufficient")
        return nothing
    end
    basequantity = missing
    freeusdtfractionmargin = 0.05
    totalborrowedusdt = sum(assets[!, :borrowed] .* assets[!, :usdtprice])
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free]) - totalborrowedusdt
    freebase = sum(assets[assets[!, :coin] .== base, :free]) *(1-eps(Float32))
    borrowedbase = sum(assets[assets[!, :coin] .== base, :borrowed])
    quotequantity = cache.mc[:maxassetfraction] * effectivebudgetquote / 10  # distribute over 10 trades within effective budget
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    price = currentprice(ohlcv)
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
        existing = _managedcloseget(cache, base)
        if !isnothing(existing) && (existing[:tradelabel] in [shortclose, shortstrongclose])
            try
                CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
            catch err
                (verbosity >= 1) && @warn "failed to cancel stale opposite managed close order" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
            end
            _managedcloseclear!(cache, base)
            existing = nothing
        end
        closeablelong = freebase
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = closeablelong <= 2 * minimumbasequantity ? (closeablelong >= minimumbasequantity ? closeablelong : minimumbasequantity) : minimumbasequantity
        basequantity = min(max(sellbuyqtyratio * qtyacceleration * quotequantity/price, minimumbasequantity), closeablelong)
        sufficientsellbalance = (basequantity <= closeablelong) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if sufficientsellbalance && exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = nothing
            if !isnothing(existing)
                existing_limit = get(existing, :limitprice, nothing)
                existing_qty = Float32(get(existing, :baseqty, 0f0))
                if !_material_order_change(existing_limit, requested_limitprice, existing_qty, basequantity)
                    oid = String(existing[:orderid])
                else
                    amended = try
                        (cache.mc[:trademode] == notrade) ? String(existing[:orderid]) : _withtradeauditcontext(cache, ta) do
                            CryptoXch.changeorder(cache.xc, String(existing[:orderid]); limitprice=requested_limitprice, basequantity=basequantity)
                        end
                    catch err
                        if _isunknownordererror(err)
                            (verbosity >= 1) && @warn "managed longclose amend skipped because order is no longer present" base=base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                            nothing
                        else
                            rethrow(err)
                        end
                    end
                    if !isnothing(amended)
                        oid = amended
                    else
                        try
                            CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
                        catch err
                            (verbosity >= 1) && @warn "failed to cancel managed longclose before recreate" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                        end
                        _managedcloseclear!(cache, base)
                    end
                end
            end
            if isnothing(oid)
                oid = (cache.mc[:trademode] == notrade) ? "SellSpotSim" : _withtradeauditcontext(cache, ta) do
                    CryptoXch.createsellorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=0)
                end
            end
            if !isnothing(oid)
                _managedcloseset!(cache, base, oid, longclose; limitprice=requested_limitprice, baseqty=basequantity)
                result = (trade=longclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longclose order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longclose order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no longclose $base due to sufficientsellbalance=$sufficientsellbalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    elseif (ta.tradelabel in [longbuy, longstrongbuy]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        basequantity = max(0f0, min(max(qtyacceleration * quotequantity/price, minimumbasequantity) * price, freeusdt - freeusdtfractionmargin * effectivebudgetquote) / price) #* keep 5% * effective budget as head room
        sufficientbuybalance = (basequantity * price < freeusdt) && ((basequantity + borrowedbase) > 0.0)
        # basequantity += borrowedbase # buy all short as well when switching to long
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        basefraction = (sum(sum(eachcol(assets[assets.coin .== base, [:free, :locked, :borrowed]]))) + basequantity) * price / effectivebudgetquote
        # basefraction = (sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt)

        # if base == "ADA"
        #     println("coin=$(ta.base) tradelabel=$(ta.tradelabel) price=$price basequantity=$basequantity sufficientbuybalance=$sufficientbuybalance minimumbasequantity=$minimumbasequantity quotequantity=$quotequantity freeusdt=$freeusdt totalusdt=$totalusdt")
        # end
    
        if basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 3) && println("$(tradetime(cache)) skip $base longbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif sufficientbuybalance && exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = (cache.mc[:trademode] == notrade) ? "BuySpotSim" : _withtradeauditcontext(cache, ta) do
                CryptoXch.createbuyorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=0)
            end
            if !isnothing(oid)
                result = (trade=longbuy, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longbuy order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longbuy order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base longbuy due to sufficientbuybalance=$sufficientbuybalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    elseif (ta.tradelabel in [shortstrongbuy, shortbuy]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        basequantity = max(qtyacceleration * quotequantity / price, minimumbasequantity)
        sufficientbuybalance = ((basequantity - freebase) * price < freeusdt) && (basequantity > 0.0)
        basefraction = (sum(sum(eachcol(assets[assets.coin .== base, [:free, :locked, :borrowed]]))) + basequantity) * price / (effectivebudgetquote + totalborrowedusdt)
        symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
        marginok = CryptoXch.marginpermitted(cache.xc, symbol, "Sell", short_margin_leverage; role=CryptoXch.trade_exchange_spot)
        if basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 2) && println("$(tradetime(cache)) skip $base shortbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif !marginok
            limits = CryptoXch.marginlimits(cache.xc, symbol; role=CryptoXch.trade_exchange_spot)
            (verbosity >= 1) && @warn "skip $base shortbuy due to Kraken margin metadata limits" symbol=symbol requested_leverage=short_margin_leverage maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell
        elseif sufficientbuybalance
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = (cache.mc[:trademode] == notrade) ? "SellMarginSim" : _withtradeauditcontext(cache, ta) do
                try
                    CryptoXch.createsellorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=short_margin_leverage)
                catch err
                    _log_margin_order_diagnostics(cache, basecfg, ta, base, "Sell", short_margin_leverage, requested_limitprice, basequantity, freebase, borrowedbase, freeusdt, totalborrowedusdt, effectivebudgetquote, err)
                    rethrow(err)
                end
            end
            if !isnothing(oid)
                result = (trade=shortbuy, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortbuy order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortbuy order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base shortbuy due to sufficientbuybalance=$sufficientbuybalance")
        end
    elseif (ta.tradelabel in [shortclose, shortstrongclose]) && (cache.mc[:trademode] in [buysell, closeonly, quickexit]) && basecfg.sellenabled
        existing = _managedcloseget(cache, base)
        if !isnothing(existing) && (existing[:tradelabel] in [longclose, longstrongclose])
            try
                CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
            catch err
                (verbosity >= 1) && @warn "failed to cancel stale opposite managed close order" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
            end
            _managedcloseclear!(cache, base)
            existing = nothing
        end
        closeableshort = borrowedbase
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = closeableshort <= 2 * minimumbasequantity ? (closeableshort >= minimumbasequantity ? closeableshort : minimumbasequantity) : minimumbasequantity # increase minimumbasequantity if otherwise a too small base amount remains that cannot be sold
        basequantity = max(0f0, min(max(sellbuyqtyratio * qtyacceleration * quotequantity/price, minimumbasequantity), closeableshort))
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = nothing
            if !isnothing(existing)
                existing_limit = get(existing, :limitprice, nothing)
                existing_qty = Float32(get(existing, :baseqty, 0f0))
                if !_material_order_change(existing_limit, requested_limitprice, existing_qty, basequantity)
                    oid = String(existing[:orderid])
                else
                    amended = try
                        (cache.mc[:trademode] == notrade) ? String(existing[:orderid]) : _withtradeauditcontext(cache, ta) do
                            CryptoXch.changeorder(cache.xc, String(existing[:orderid]); limitprice=requested_limitprice, basequantity=basequantity)
                        end
                    catch err
                        if _isunknownordererror(err)
                            (verbosity >= 1) && @warn "managed shortclose amend skipped because order is no longer present" base=base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                            nothing
                        else
                            rethrow(err)
                        end
                    end
                    if !isnothing(amended)
                        oid = amended
                    else
                        try
                            CryptoXch.cancelorder(cache.xc, base, String(existing[:orderid]))
                        catch err
                            (verbosity >= 1) && @warn "failed to cancel managed shortclose before recreate" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                        end
                        _managedcloseclear!(cache, base)
                    end
                end
            end
            if isnothing(oid)
                oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : _withtradeauditcontext(cache, ta) do
                    try
                        CryptoXch.createbuyorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=short_margin_leverage)
                    catch err
                        _log_margin_order_diagnostics(cache, basecfg, ta, base, "Buy", short_margin_leverage, requested_limitprice, basequantity, freebase, borrowedbase, freeusdt, totalborrowedusdt, effectivebudgetquote, err)
                        rethrow(err)
                    end
                end
            end
            if !isnothing(oid)
                _managedcloseset!(cache, base, oid, shortclose; limitprice=requested_limitprice, baseqty=basequantity)
                result = (trade=shortclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortclose order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortclose order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no shortclose $base due to exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    end
    if !isnothing(result)

    end 
    return result
end

tradetime(cache::TradeCache) = CryptoXch.ttstr(cache.xc)
# USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), locked=$(round(Int, sum(assets.locked .* assets.usdtprice))), free=$(round(Int, sum(assets.free .* assets.usdtprice)))")
function USDTmsg(assets)
    totalusdt = sum(assets.usdtvalue)
    totalborrowedusdt = sum(assets[!, :borrowed] .* assets[!, :usdtprice])
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free]) - totalborrowedusdt
    freepct = totalusdt > 0f0 ? min(100, round(Int, freeusdt / totalusdt * 100)) : 0
    return string("$(EnvConfig.cryptoquote): total=$(round(Int, totalusdt)), quotefree=$(freepct)%")
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

# ── Loop control ────────────────────────────────────────────────────────────

"Returns the current loop lifecycle state."
_loopstate_nolock(cache::TradeCache) = LoopState(Int(cache.mc[:loop_state]))
_setloopstate_nolock!(cache::TradeCache, s::LoopState) = (cache.mc[:loop_state] = s; nothing)

function loopstate(cache::TradeCache)
    lock(cache.looplock)
    try
        return _loopstate_nolock(cache)
    finally
        unlock(cache.looplock)
    end
end

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

"""
Request the loop to pause after the current tick.
Only effective when the loop state is `loop_running`.
"""
function pause!(cache::TradeCache)
    lock(cache.looplock)
    try
        (_loopstate_nolock(cache) == loop_running) && _setloopstate_nolock!(cache, loop_paused)
    finally
        unlock(cache.looplock)
    end
    return cache
end

"""
Resume a paused loop.
Only effective when the loop state is `loop_paused`.
"""
function resume!(cache::TradeCache)
    lock(cache.looplock)
    try
        if _loopstate_nolock(cache) == loop_paused
            _setloopstate_nolock!(cache, loop_running)
            notify(cache.loopcond; all=true)
        end
    finally
        unlock(cache.looplock)
    end
    return cache
end

"""
Request the loop to stop gracefully after the current tick completes.
Effective when loop state is `loop_running` or `loop_paused`.
"""
function stop!(cache::TradeCache)
    lock(cache.looplock)
    try
        st = _loopstate_nolock(cache)
        if st in (loop_running, loop_paused)
            _setloopstate_nolock!(cache, loop_stopping)
            notify(cache.loopcond; all=true)
        end
    finally
        unlock(cache.looplock)
    end
    return cache
end

# ── Strategy config ─────────────────────────────────────────────────────────

function _validatestrategyconfig!(mc::AbstractDict)
    gs = haskey(mc, :strategy_template) ? mc[:strategy_template] : TradingStrategy.GainSegment()
    openthreshold = Float32(gs.openthreshold)
    closethreshold = Float32(gs.closethreshold)
    buygain = Float32(gs.buygain)
    sellgain = Float32(gs.sellgain)
    limitreduction = Float32(gs.limitreduction)
    maxwindow = Int(gs.maxwindow)

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
    return cache
end

"Apply strategy runtime settings from a TrendDetector-style configuration reference."
function apply_trenddetector_strategy!(mc::AbstractDict, tdref)
    @assert hasproperty(tdref, :tradingstrategy) "tdref must expose field :tradingstrategy, got type=$(typeof(tdref))"
    gs = getproperty(tdref, :tradingstrategy)
    @assert gs isa TradingStrategy.GainSegment "tdref.tradingstrategy must be TradingStrategy.GainSegment, got type=$(typeof(gs))"
    source = hasproperty(tdref, :configname) ? "trenddetector:$(getproperty(tdref, :configname))" : "trenddetector"
    return apply_tradingstrategy!(mc, gs; strategy_engine=:getgainsalgo, source=source)
end

function apply_trenddetector_strategy!(cache::TradeCache, tdref)
    apply_trenddetector_strategy!(cache.mc, tdref)
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
        deepcopy(get(cache.mc, :strategy_template, TradingStrategy.GainSegment()))
    end
end
function _strategyruntime(cache::TradeCache)
    if haskey(cache.mc, :strategy_runtime)
        return cache.mc[:strategy_runtime]
    end
    template = get(cache.mc, :strategy_template, TradingStrategy.GainSegment())
    rt = TradingStrategy.GainSegmentRuntime(; classifier=cache.cl, strategy=template, source=String(get(cache.mc, :strategy_source, "default")))
    cache.mc[:strategy_runtime] = rt
    return rt
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

function _getgainsalgo_action2label(gs::TradingStrategy.GainSegment, fallback::Targets.TradeLabel=allclose)::Targets.TradeLabel
    # Mapping note:
    # - TradingStrategy keeps two actions: buyta (open intent) and sellta (close intent).
    # - In gain_limit_reversal! a filled buyta is handed off to sellta within the same minute.
    # - Therefore the mapped label can be a close even when the raw classifier label is an open label.
    if gs.buyta.orderlabel in [longbuy, longstrongbuy]
        return longbuy
    elseif gs.buyta.orderlabel in [shortbuy, shortstrongbuy]
        return shortbuy
    elseif gs.sellta.orderlabel in [longclose, longstrongclose]
        return longclose
    elseif gs.sellta.orderlabel in [shortclose, shortstrongclose]
        return shortclose
    end
    return fallback
end

function _getgainsalgo_limitprice(gs::TradingStrategy.GainSegment, mappedlabel::Targets.TradeLabel)::Union{Nothing, Float32}
    if (mappedlabel in [longbuy, longstrongbuy, shortbuy, shortstrongbuy]) && (gs.buyta.orderlabel in [longbuy, longstrongbuy, shortbuy, shortstrongbuy])
        return Float32(gs.buyta.buyprice)
    elseif (mappedlabel in [longclose, longstrongclose, shortclose, shortstrongclose]) && (gs.sellta.orderlabel in [longclose, longstrongclose, shortclose, shortstrongclose])
        return Float32(gs.sellta.orderlimit)
    end
    return nothing
end

function _managedclosestate(cache::TradeCache)::Dict{String, Dict{Symbol, Any}}
    return get!(cache.mc, :managed_close_orders, Dict{String, Dict{Symbol, Any}}())
end

function _managedclosekey(base::AbstractString, tradelabel::Targets.TradeLabel)::String
    return string(uppercase(String(base)), "|", String(Symbol(tradelabel)))
end

function _managedcloseget(cache::TradeCache, base::AbstractString)
    managed = _managedclosestate(cache)
    longk = _managedclosekey(base, longclose)
    shortk = _managedclosekey(base, shortclose)
    if haskey(managed, longk)
        return managed[longk]
    elseif haskey(managed, shortk)
        return managed[shortk]
    end
    return nothing
end

function _managedcloseset!(cache::TradeCache, base::AbstractString, orderid, tradelabel::Targets.TradeLabel; limitprice=nothing, baseqty::Real=0f0)
    _managedclosestate(cache)[_managedclosekey(base, tradelabel)] = Dict{Symbol, Any}(
        :orderid => String(orderid),
        :tradelabel => tradelabel,
        :limitprice => isnothing(limitprice) ? nothing : Float32(limitprice),
        :baseqty => Float32(baseqty),
        :updated => Dates.now(Dates.UTC),
    )
    return nothing
end

function _managedcloseclear!(cache::TradeCache, base::AbstractString)
    managed = _managedclosestate(cache)
    delete!(managed, _managedclosekey(base, longclose))
    delete!(managed, _managedclosekey(base, shortclose))
    return nothing
end

function _order_amend_price_rel_threshold(cache::TradeCache)::Float32
    return Float32(get(cache.mc, :order_amend_price_rel_threshold, 1f-3))
end

function _activeopenbuysymbols!(cache::TradeCache)
    return get!(cache.mc, :activeopenbuysymbols, Set{String}())
end

function _activeopensellsymbols!(cache::TradeCache)
    return get!(cache.mc, :activeopensellsymbols, Set{String}())
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

function _refreshactiveopenbuysymbols!(cache::TradeCache, oo::AbstractDataFrame)
    buys = _activeopenbuysymbols!(cache)
    empty!(buys)
    size(oo, 1) == 0 && return cache
    haslev = hasproperty(oo, :isLeverage)
    for row in eachrow(oo)
        CryptoXch.openstatus(String(row.status)) || continue
        haslev && Bool(row.isLeverage) && continue
        lowercase(String(row.side)) == "buy" || continue
        push!(buys, uppercase(String(row.symbol)))
    end
    return cache
end

function _refreshactiveopensellsymbols!(cache::TradeCache, oo::AbstractDataFrame)
    sells = _activeopensellsymbols!(cache)
    empty!(sells)
    size(oo, 1) == 0 && return cache
    haslev = hasproperty(oo, :isLeverage)
    for row in eachrow(oo)
        CryptoXch.openstatus(String(row.status)) || continue
        haslev && Bool(row.isLeverage) && continue
        lowercase(String(row.side)) == "sell" || continue
        push!(sells, uppercase(String(row.symbol)))
    end
    return cache
end

function _positioncloselabel(basecfg::DataFrameRow, assets::AbstractDataFrame, base::AbstractString)::Union{Nothing, Targets.TradeLabel}
    freebase = Float32(sum(assets[assets[!, :coin] .== base, :free]))
    borrowedbase = Float32(sum(assets[assets[!, :coin] .== base, :borrowed]))
    closeablelong = freebase
    closeableshort = borrowedbase
    if (closeablelong > 0f0) && _cfgbool(basecfg, :sellenabled, true)
        return longclose
    elseif (closeableshort > 0f0) && _cfgbool(basecfg, :sellenabled, true)
        return shortclose
    end
    return nothing
end

function _strategy_sell_limitprice(cache::TradeCache, base::AbstractString, closelabel::Targets.TradeLabel; assets::Union{Nothing, AbstractDataFrame}=nothing)::Union{Nothing, Float32}
    if haskey(cache.mc, :strategy_runtime)
        rt = _strategyruntime(cache)
        dt = isnothing(cache.xc.currentdt) ? Dates.now(Dates.UTC) : cache.xc.currentdt
        reconciliation = TradingStrategy.StrategyReconciliationInput()
        if !isnothing(assets)
            freebase = Float32(sum(assets[assets[!, :coin] .== base, :free]))
            borrowedbase = Float32(sum(assets[assets[!, :coin] .== base, :borrowed]))
            base_rows = assets[assets[!, :coin] .== base, :]
            entry_price = size(base_rows, 1) > 0 ? Float32(base_rows[1, :usdtprice]) : 0f0
            long_avg = freebase > 0f0 ? entry_price : 0f0
            short_avg = borrowedbase > 0f0 ? entry_price : 0f0
            reconciliation = TradingStrategy.StrategyReconciliationInput(
                has_long_open=freebase > 0f0,
                long_avg_entry=long_avg,
                long_open_ix=0,
                has_short_open=borrowedbase > 0f0,
                short_avg_entry=short_avg,
                short_open_ix=0,
            )
        end
        snap = TradingStrategy.getsnapshot!(rt, cache.xc, base, dt; reconciliation=reconciliation)
        if !isnothing(snap)
            if closelabel in [longclose, longstrongclose]
                return snap.long_closeprice > 0f0 ? Float32(snap.long_closeprice) : nothing
            elseif closelabel in [shortclose, shortstrongclose]
                return snap.short_closeprice > 0f0 ? Float32(snap.short_closeprice) : nothing
            end
        end
    end
    _strategyengine(cache) == :getgainsalgo || return nothing
    gs = get(get(cache.mc, :strategy_state, Dict{String, Any}()), String(base), nothing)
    isnothing(gs) && return nothing
    if (closelabel in [longclose, longstrongclose]) && (gs.sellta.orderlabel in [longclose, longstrongclose])
        return Float32(gs.sellta.orderlimit)
    elseif (closelabel in [shortclose, shortstrongclose]) && (gs.sellta.orderlabel in [shortclose, shortstrongclose])
        return Float32(gs.sellta.orderlimit)
    end
    return nothing
end

function _material_order_change(existing_limit, requested_limit, existing_qty::Real, requested_qty::Real; price_reltol::Real=0.002, qty_reltol::Real=0.02)::Bool
    if isnothing(existing_limit) || isnothing(requested_limit)
        return true
    end
    old_limit = Float64(existing_limit)
    new_limit = Float64(requested_limit)
    limit_ref = max(abs(old_limit), 1e-6)
    limit_delta = abs(new_limit - old_limit) / limit_ref

    old_qty = abs(Float64(existing_qty))
    new_qty = abs(Float64(requested_qty))
    qty_ref = max(old_qty, 1e-6)
    qty_delta = abs(new_qty - old_qty) / qty_ref
    return (limit_delta > Float64(price_reltol)) || (qty_delta > Float64(qty_reltol))
end

function _reconstruct_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, oo::AbstractDataFrame)
    managed = _managedclosestate(cache)
    empty!(managed)
    (size(cache.cfg, 1) == 0) && return managed
    (size(oo, 1) == 0) && return managed

    for basecfg in eachrow(cache.cfg)
        base = String(basecfg.basecoin)
        closelabel = _positioncloselabel(basecfg, assets, base)
        isnothing(closelabel) && continue
        wanted_side = closelabel in [longclose, longstrongclose] ? "Sell" : "Buy"
        symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
        for row in eachrow(oo)
            CryptoXch.openstatus(String(row.status)) || continue
            (String(row.symbol) == symbol) || continue
            (String(row.side) == wanted_side) || continue
            _managedcloseset!(cache, base, row.orderid, closelabel; limitprice=row.limitprice, baseqty=row.baseqty)
            break
        end
    end
    return managed
end

function _cancel_unmanaged_open_orders!(cache::TradeCache, oo::AbstractDataFrame)
    managedids = Set{String}()
    for state in values(_managedclosestate(cache))
        haskey(state, :orderid) && push!(managedids, String(state[:orderid]))
    end
    for row in eachrow(oo)
        CryptoXch.openstatus(String(row.status)) || continue
        oid = String(row.orderid)
        oid in managedids && continue
        base = let pair = CryptoXch.basequote(String(row.symbol))
            isnothing(pair) ? String(row.symbol) : String(pair.basecoin)
        end
        try
            CryptoXch.cancelorder(cache.xc, base, oid)
        catch err
            (verbosity >= 1) && @warn "failed to cancel unmanaged open order during tradestep; continuing" orderid=oid symbol=row.symbol status=row.status error=sprint(showerror, err)
        end
    end
    return nothing
end

function _advicebybase(tradeadvices::Vector{StrategyAdvice})::Dict{String, StrategyAdvice}
    bybase = Dict{String, StrategyAdvice}()
    for ta in tradeadvices
        bybase[String(ta.base)] = ta
    end
    return bybase
end

function _ensure_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, tradeadvices::Vector{StrategyAdvice})
    advbybase = _advicebybase(tradeadvices)
    for basecfg in eachrow(cache.cfg)
        base = String(basecfg.basecoin)
        closelabel = _positioncloselabel(basecfg, assets, base)
        isnothing(closelabel) && continue

        ta = if haskey(advbybase, base)
            deepcopy(advbybase[base])
        else
            StrategyAdvice(classifier=cache.cl, base=base, datetime=isnothing(cache.xc.currentdt) ? Dates.now() : cache.xc.currentdt)
        end
        ta.tradelabel = closelabel
        ta.price = _strategy_sell_limitprice(cache, base, closelabel)
        ta.source = :managedclose
        ta.allowreversal = false

        try
            trade!(cache, basecfg, ta, assets)
        catch err
            if _ispermissionrestrictederror(err)
                _disablerestrictedbase!(cache, base, sprint(showerror, err))
            elseif _isinsufficientfundserror(err)
                (verbosity >= 1) && @warn "skip managed close order due to insufficient funds" base=base error=sprint(showerror, err)
            elseif _isprivatecooldownerror(err)
                (verbosity >= 1) && @warn "skip managed close order due to transient private-read cooldown" base=base error=sprint(showerror, err)
            else
                rethrow(err)
            end
        end
    end
    return nothing
end

function _getgainsalgo_advice!(cache::TradeCache, base::AbstractString, ta)::StrategyAdvice
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    history = _strategyhistory!(cache, base)
    _upsert_getgainsalgo_sample!(history, ohlcv, ta.tradelabel, ta.probability)
    gs = _strategystate!(cache, base)
    lastix = length(history.scores)
    mappedlabel = ta.tradelabel
    limitprice = nothing
    if lastix > 0
        TradingStrategy.getgains(gs, history.predictionsdf, history.scores, history.labels, false; lastix=lastix, openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
        mappedlabel = _getgainsalgo_action2label(gs, ta.tradelabel)
        limitprice = _getgainsalgo_limitprice(gs, mappedlabel)
    end
    return _strategyadvice(ta; tradelabel=mappedlabel, limitprice=limitprice, source=:getgainsalgo)
end

function _expand_reversal_advice(ta::StrategyAdvice, assets::AbstractDataFrame)::Vector{StrategyAdvice}
    if !ta.allowreversal
        return StrategyAdvice[ta]
    end
    base = ta.base
    freebase = sum(assets[assets[!, :coin] .== base, :free])
    borrowedbase = sum(assets[assets[!, :coin] .== base, :borrowed])
    if (ta.tradelabel in [longbuy, longstrongbuy]) && (borrowedbase > 0)
        closeadvice = deepcopy(ta)
        closeadvice.tradelabel = shortclose
        closeadvice.price = nothing
        return StrategyAdvice[closeadvice, ta]
    elseif (ta.tradelabel in [shortbuy, shortstrongbuy]) && (freebase > 0)
        closeadvice = deepcopy(ta)
        closeadvice.tradelabel = longclose
        closeadvice.price = nothing
        return StrategyAdvice[closeadvice, ta]
    end
    return StrategyAdvice[ta]
end

function _collect_strategy_advices(cache::TradeCache, assets::AbstractDataFrame)
    if haskey(cache.mc, :strategy_runtime)
        rt = _strategyruntime(cache)
        dt = isnothing(cache.xc.currentdt) ? Dates.now(Dates.UTC) : cache.xc.currentdt
        bases = String.(cache.cfg[!, :basecoin])
        reconciliation_by_base = Dict{String, TradingStrategy.StrategyReconciliationInput}()
        for base in bases
            freebase = Float32(sum(assets[assets[!, :coin] .== base, :free]))
            borrowedbase = Float32(sum(assets[assets[!, :coin] .== base, :borrowed]))
            base_rows = assets[assets[!, :coin] .== base, :]
            entry_price = size(base_rows, 1) > 0 ? Float32(base_rows[1, :usdtprice]) : 0f0
            long_avg = freebase > 0f0 ? entry_price : 0f0
            short_avg = borrowedbase > 0f0 ? entry_price : 0f0
            reconciliation_by_base[uppercase(base)] = TradingStrategy.StrategyReconciliationInput(
                has_long_open=freebase > 0f0,
                long_avg_entry=long_avg,
                long_open_ix=0,
                has_short_open=borrowedbase > 0f0,
                short_avg_entry=short_avg,
                short_open_ix=0,
            )
        end
        snapshots = TradingStrategy.getsnapshots!(rt, cache.xc, bases, dt; reconciliation_by_base=reconciliation_by_base)
        tradeadvices = StrategyAdvice[]
        for snap in snapshots
            label = snap.label
            if label != Targets.ignore
                pricemap = label in [longbuy, longstrongbuy] ? snap.long_openprice : label in [shortbuy, shortstrongbuy] ? snap.short_openprice : label in [longclose, longstrongclose] ? snap.long_closeprice : label in [shortclose, shortstrongclose] ? snap.short_closeprice : 0f0
                push!(tradeadvices, StrategyAdvice(
                    classifier=cache.cl,
                    configid=snap.configid,
                    tradelabel=label,
                    relativeamount=1f0,
                    base=String(snap.base),
                    price=Float32(pricemap),
                    datetime=snap.datetime,
                    probability=snap.probability,
                    source=:tradingstrategy,
                    allowreversal=true,
                ))
            end
            recon = get(reconciliation_by_base, uppercase(String(snap.base)), TradingStrategy.StrategyReconciliationInput())
            if recon.has_long_open && (snap.long_closeprice > 0f0)
                push!(tradeadvices, StrategyAdvice(
                    classifier=cache.cl,
                    configid=snap.configid,
                    tradelabel=Targets.longclose,
                    relativeamount=1f0,
                    base=String(snap.base),
                    price=Float32(snap.long_closeprice),
                    datetime=snap.datetime,
                    probability=snap.probability,
                    source=:tradingstrategy,
                    allowreversal=false,
                ))
            end
            if recon.has_short_open && (snap.short_closeprice > 0f0)
                push!(tradeadvices, StrategyAdvice(
                    classifier=cache.cl,
                    configid=snap.configid,
                    tradelabel=Targets.shortclose,
                    relativeamount=1f0,
                    base=String(snap.base),
                    price=Float32(snap.short_closeprice),
                    datetime=snap.datetime,
                    probability=snap.probability,
                    source=:tradingstrategy,
                    allowreversal=false,
                ))
            end
        end
        return tradeadvices
    end
    tradeadvices = StrategyAdvice[]
    Classify.supplement!(cache.cl)
    for basecfg in eachrow(cache.cfg)
        rawadvice = _classifier_advice(cache.cl, basecfg.basecoin, cache.xc.currentdt)
        if isnothing(rawadvice)
            (verbosity > 3) && println("no trade advice for $(basecfg.basecoin)")
            continue
        end
        sa = if _strategyengine(cache) == :getgainsalgo
            _getgainsalgo_advice!(cache, basecfg.basecoin, rawadvice)
        else
            _strategyadvice(rawadvice; source=:classifier)
        end
        append!(tradeadvices, _expand_reversal_advice(sa, assets))
    end
    return tradeadvices
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

function _maybe_refresh_tradeselection!(cache::TradeCache)
    if !_should_refresh_tradeselection(cache)
        return false
    end
    assets = CryptoXch.portfolio!(cache.xc)
    (verbosity >= 2) && println("\n$(tradetime(cache)): start reassessing trading strategy")
    tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.currentdt, updatecache=true)
    cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
    _mark_tradeselection_refreshed!(cache)
    (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.cfg)"
    return true
end

"Return position-side gaps where no matching open close order currently exists."
function _positions_without_close_orders(cache::TradeCache, assets::AbstractDataFrame, oo::AbstractDataFrame)
    quote_coin = uppercase(String(EnvConfig.cryptoquote))
    missing = NamedTuple{(:base, :side, :qty), Tuple{String, String, Float32}}[]
    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        base == quote_coin && continue
        freebase = Float32(row.free)
        borrowedbase = Float32(row.borrowed)
        if (freebase <= 0f0) && (borrowedbase <= 0f0)
            continue
        end

        symbol = CryptoXch.symboltoken(cache.xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
        managed = _managedcloseget(cache, base)
        if freebase > 0f0
            has_sell = any((String(orow.symbol) == symbol) && (String(orow.side) == "Sell") && CryptoXch.openstatus(String(orow.status)) for orow in eachrow(oo))
            if !has_sell && !isnothing(managed)
                has_sell = (get(managed, :tradelabel, ignore) in [longclose, longstrongclose])
            end
            !has_sell && push!(missing, (base=base, side="Sell", qty=freebase))
        end
        if borrowedbase > 0f0
            has_buy = any((String(orow.symbol) == symbol) && (String(orow.side) == "Buy") && CryptoXch.openstatus(String(orow.status)) for orow in eachrow(oo))
            if !has_buy && !isnothing(managed)
                has_buy = (get(managed, :tradelabel, ignore) in [shortclose, shortstrongclose])
            end
            !has_buy && push!(missing, (base=base, side="Buy", qty=borrowedbase))
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
    oo = CryptoXch.getopenorders(cache.xc)
    assets = CryptoXch.portfolio!(cache.xc)
    _reconstruct_managed_close_orders!(cache, assets, oo)
    _cancel_unmanaged_open_orders!(cache, oo)
    _maybe_writeportfoliosnapshot!(cache, assets)
    tradeadvices = _collect_strategy_advices(cache, assets)
    _ensure_managed_close_orders!(cache, assets, tradeadvices)
    if cache.mc[:usenewtrade]
    else # legacy trade!()
        sellbases = []
        buybases = []
        sort!(tradeadvices, lt=tradeadvicelessthan)  # close first, then buy high-gain first
        for ta in tradeadvices
            if _strategyengine(cache) == :getgainsalgo && _isclosetrade(ta.tradelabel)
                continue
            end
            rowix = findfirst(==(ta.base), cache.cfg[!, :basecoin])
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
                    (verbosity >= 1) && @warn "skip trade advice due to insufficient funds" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) error=sprint(showerror, err)
                    nothing
                elseif _isprivatecooldownerror(err)
                    (verbosity >= 1) && @warn "skip trade advice due to transient private-read cooldown" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) error=sprint(showerror, err)
                    nothing
                else
                    rethrow(err)
                end
            end
            if !isnothing(res) && (res.trade in [longbuy, longstrongbuy, shortclose, shortstrongclose])
                push!(buybases, basecfg.basecoin)
            elseif !isnothing(res) && (res.trade in [longstrongclose, longclose, shortstrongbuy, shortbuy])
                push!(sellbases, basecfg.basecoin)
            elseif !isnothing(res)
                @warn "case not handled: $res"
            end
        end
        (verbosity >= 2) && print("\r$(tradetime(cache)): $(USDTmsg(assets)), bought: $(buybases), sold: $(sellbases)                                          ")
    end

    # Live safety summary: highlight open positions that currently have no opposite-side close order.
    if cache.mc[:trademode] in [buysell, closeonly, quickexit]
        assets_now = CryptoXch.portfolio!(cache.xc)
        oo_now = CryptoXch.getopenorders(cache.xc)
        missing = _positions_without_close_orders(cache, assets_now, oo_now)
        if !isempty(missing)
            details = ["$(x.base):$(x.side):qty=$(round(Float64(x.qty); digits=6))" for x in missing]
            (verbosity >= 1) && @warn "open positions without active close order" count=length(missing) details=details
        end
    end

    _maybe_refresh_tradeselection!(cache)
    #TODO low prio: for closed orders check fees
    #TODO low prio: aggregate orders and transactions in bookkeeping
    return nothing
end

"Load or derive the initial trade configuration if `cache.cfg` is empty."
function _ensure_tradeloop_initialized!(cache::TradeCache)
    if size(cache.cfg, 1) == 0
        assets = CryptoXch.balances(cache.xc)
        (verbosity >= 2) && print("\r$(tradetime(cache)): start calculating trading strategy on the fly")
        tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.startdt)
        cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
        (verbosity > 2) && @info "$(tradetime(cache)) initial trading strategy: $(cache.cfg)"
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

"""
Start the trading loop (blocking). Selects backtest or live mode from `cache.xc.enddt` presence.
Use `stop!` to request early termination from another task.
"""
function start!(cache::TradeCache)
    if loopstate(cache) != loop_idle
        @warn "start! called but loop is not idle (state=$(loopstate(cache)))"
    end
    _ensure_tradeloop_initialized!(cache)
    _run_tradeloop!(cache)
    return cache
end

"""
Asynchronously start the trading loop in a background task.
Returns immediately with a `Task` handle. The loop executes in the background, and the caller
can control it from another task/thread using `pause!()`, `resume!()`, and `stop!()`.

When `skip_init=false` (default) the trade configuration is loaded or rebuilt if `cache.cfg` is empty.
Pass `skip_init=true` when the caller has already populated `cache.cfg`.

Returns:
    `Task`: a background task running `_run_tradeloop!(cache)`. 
    Caller can `wait(task)` for completion or check task status.

Example:
```julia
cache = Trade.setup_backtest(...)
task = Trade.async_start!(cache)
# ... from another task:
Trade.pause!(cache)   # pause the loop
Trade.resume!(cache)  # resume it
Trade.stop!(cache)    # request exit
result = wait(task)   # block until loop finishes
```
"""
function async_start!(cache::TradeCache; skip_init::Bool=false)
    if loopstate(cache) != loop_idle
        @warn "async_start! called but loop is not idle (state=$(loopstate(cache)))"
    end
    skip_init || _ensure_tradeloop_initialized!(cache)
    return @async _run_tradeloop!(cache)
end

"""
Execute exactly one trading tick without the iteration engine.
Useful for step-by-step debugging or custom replay harnesses.
The caller is responsible for advancing `cache.xc.currentdt` before calling `step!`.
"""
function step!(cache::TradeCache)
    _tradestep!(cache)
    return cache
end

"""
**`tradeloop`** — compatibility wrapper calling `start!`.
Prefer using `run_backtest!`, `run_live!`, or `start!` directly for new code.

+ get initial TradeStrategy config (if not present at entry) and refresh daily according to `reloadtimes`
+ get new exchange data (preferably non-blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non-blocking)
+ follow up on open orders (preferably non-blocking)
"""
function tradeloop(cache::TradeCache)
    start!(cache)
end

function tradelooptest(cache::TradeCache)
    for c in cache.xc
        println("$(Dates.now(UTC)) $c  $(CryptoXch.ohlcv(c.xc, "BTC"))")
    end
end


end  # module

