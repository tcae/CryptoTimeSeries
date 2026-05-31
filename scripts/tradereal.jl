"""
tradereal.jl — Live trading script using TrendDetector config 046.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before starting. The loop runs until Ctrl+C is pressed.

Usage:
    julia --project=scripts scripts/tradereal.jl
    julia --project=scripts scripts/tradereal.jl xch=KrakenFutures
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using Dates, Logging, LoggingExtras
using EnvConfig, TradingStrategy, Trade, Classify, CryptoXch, Features, Ohlcv, Targets

include(joinpath(@__DIR__, "optimizationconfigs.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these values before running
# ─────────────────────────────────────────────────────────────────────────────

# Exchange to use for live trading: CryptoXch.EXCHANGE_BYBIT, EXCHANGE_KRAKENSPOT,
# or EXCHANGE_KRAKENFUTURES.
const EXCHANGE = CryptoXch.EXCHANGE_KRAKENSPOT

# Trade mode: Trade.buysell, Trade.closeonly, Trade.quickexit, or Trade.notrade.
const TRADE_MODE = Trade.buysell

# Whitelist of base coins to consider for trading.
# Only coins in this list and meeting liquidity requirements will be traded.
# Entries can be either base symbols (e.g. "BTC") or pair symbols (e.g. "BTC/USDT").
const QUOTE_COIN = "USD"
const WHITELIST_INPUT = ["BTC", "ETH", "ZEC", "XRP", "SOL", "HYPE", "SUI", "VVV", "NEAR", 
                         "DOGE", "ONDO", "TAO", "XLM", "LINK", "BCH", "ADA", "LTC", "XDC", 
                         "TRX", "ALGO", "TON", "INJ", "MON","DASH", "XMR", "PENGU", "CC", 
                         "AVAX", "ICP", "OG", "UNI"]

# Coins/pairs excluded from the trading robot universe (e.g. region/account restrictions).
# Entries can be base symbols ("XMR") or pair symbols ("XMR/USD").
const RESTRICTED_COINS_INPUT = ["XMR"]

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# Optional cap for overall budget considered by trade sizing.
# If set, sizing uses min(real portfolio quote value, MAX_BUDGET_QUOTE).
const MAX_BUDGET_QUOTE = 500 # nothing

# Safety margin applied to exchange-reported opening capacity before the budget cap.
# Budget limit = min(MAX_BUDGET_QUOTE, available_opening_quote * (1 - SAFETY_MARGIN)).
const SAFETY_MARGIN = 0.1

# TrendDetector config 046: GainSegment strategy parameters.
# These come from mk046config() → tradingstrategy03().
# Override individual fields here if needed.
const CONFIG046_STRATEGY = tradingstrategy04()  # GainSegment(maxwindow=240, algorithm=gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015)
const CONFIG046_NAME = "046"
const MODEL046_FOLDER = "Trend-046-production"

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER_PREFIX = "tradereal-" * CONFIG046_NAME

"Return the value for key from args entries in the form key=value, or default."
function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing, AbstractString}=nothing)
    prefix = String(key) * "="
    for arg in args
        startswith(arg, prefix) || continue
        return strip(arg[(length(prefix)+1):end])
    end
    return default
end

"Resolve exchange override from args (xch=...) or return the configured default exchange."
function _resolve_exchange(args::Vector{String}, default_exchange::AbstractString)::String
    raw = _argvalue(args, "xch", nothing)
    isnothing(raw) && return String(default_exchange)
    key = lowercase(strip(String(raw)))
    aliases = Dict(
        "bybit" => CryptoXch.EXCHANGE_BYBIT,
        "krakenspot" => CryptoXch.EXCHANGE_KRAKENSPOT,
        "krakenfutures" => CryptoXch.EXCHANGE_KRAKENFUTURES,
    )
    haskey(aliases, key) && return aliases[key]
    valid = collect(values(aliases))
    error("unsupported xch=$(raw). Expected one of $(valid) or aliases bybit|krakenspot|krakenfutures")
end

"Return a filesystem-safe token for exchange-specific log folder names."
function _exchange_logtoken(exchange::AbstractString)::String
    token = lowercase(strip(String(exchange)))
    token = replace(token, r"[^a-z0-9]+" => "-")
    token = strip(token, '-')
    return isempty(token) ? "exchange" : token
end

# Normalize whitelist entries to base coins for the configured quote coin.
function normalize_whitelist(entries, quote_coin::AbstractString)
    quote_up = uppercase(quote_coin)
    bases = String[]
    for raw in entries
        token = uppercase(strip(raw))
        if isempty(token)
            continue
        elseif occursin('/', token)
            parts = split(token, '/'; limit=2)
            length(parts) == 2 || continue
            base, q = parts
            q == quote_up || continue
            base == quote_up && continue
            push!(bases, base)
        elseif token != quote_up
            push!(bases, token)
        end
    end
    return unique(bases)
end

function max_budget_from_env(default_budget::Union{Nothing, Real})::Union{Nothing, Float64}
    raw = strip(get(ENV, "TRADEREAL_MAX_BUDGET_QUOTE", ""))
    if isempty(raw)
        # backward compatibility for previous env name
        raw = strip(get(ENV, "TRADEREAL_MAX_BUDGET_USDT", ""))
    end
    if isempty(raw)
        return isnothing(default_budget) ? nothing : Float64(default_budget)
    end
    budget = parse(Float64, raw)
    @assert budget > 0.0 "TRADEREAL_MAX_BUDGET_QUOTE must be > 0; got $(budget)"
    return budget
end

function _env_true(name::AbstractString, default::Bool)::Bool
    raw = lowercase(strip(get(ENV, String(name), default ? "true" : "false")))
    return raw in ("1", "true", "yes", "on")
end

"Pick a KrakenSpot startup probe base from whitelist that supports short margin leverage."
function _pick_krakenspot_probe_base(xc::CryptoXch.XchCache, whitelist::Vector{String}, short_leverage::Int)::Union{Nothing, String}
    for base in whitelist
        symbol = CryptoXch.symboltoken(xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
        if CryptoXch.validsymbol(xc, symbol) && CryptoXch.marginpermitted(xc, symbol, "Sell", short_leverage; role=CryptoXch.trade_exchange_spot)
            return base
        end
    end
    return nothing
end

"""
Compute probe order base quantity for a given price:
  2 × max(minimum_base_qty, minimum_quote_qty / price)
This gives 2× the exchange minimum notional in base units.
"""
function _probe_basequantity(xc::CryptoXch.XchCache, base::AbstractString, price::Float64)::Union{Nothing, Float32}
    symbol = CryptoXch.symboltoken(xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    minqty = CryptoXch.minimumqty(xc, symbol)
    isnothing(minqty) && return nothing
    minbase = Float64(minqty.minbaseqty)
    minquote = Float64(minqty.minquoteqty)
    qty = 2.0 * max(minbase, price > 0.0 ? minquote / price : minbase)
    return Float32(qty)
end

"""
Run startup KrakenSpot order capability checks for one probe pair.

Creates two post-only limit orders 2% away from current price:
- long buy (spot, leverage 0)
- short sell (margin, leverage 2)
"""
function krakenspot_startup_order_capability_probe!(xc::CryptoXch.XchCache, whitelist::Vector{String}; price_offset::Float64=0.02, short_leverage::Int=2)
    if !_env_true("TRADEREAL_KRAKENSPOT_STARTUP_ORDER_PROBE", true)
        @info "KrakenSpot startup order capability probe disabled via TRADEREAL_KRAKENSPOT_STARTUP_ORDER_PROBE"
        return nothing
    end

    base_override = uppercase(strip(get(ENV, "TRADEREAL_KRAKENSPOT_STARTUP_ORDER_PROBE_BASE", "")))
    probe_base = isempty(base_override) ? _pick_krakenspot_probe_base(xc, whitelist, short_leverage) : base_override
    isnothing(probe_base) && error("KrakenSpot startup probe failed: no whitelist base supports short margin leverage=$(short_leverage)x")

    symbol = CryptoXch.symboltoken(xc, probe_base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    ticker = CryptoXch.KrakenSpot.get24h(xc.bc, symbol)
    isnothing(ticker) && error("KrakenSpot startup probe failed: missing ticker for base=$(probe_base) symbol=$(symbol)")
    lastprice = Float64(ticker.lastprice)
    lastprice > 0.0 || error("KrakenSpot startup probe failed: invalid lastprice=$(lastprice) for base=$(probe_base) symbol=$(symbol)")

    # Long buy 2% below current price; short sell 2% above — both safely away from the market.
    buy_price = Float32(lastprice * (1.0 - price_offset))
    sell_price = Float32(lastprice * (1.0 + price_offset))
    # 2 × minimum notional in base units so the order clearly clears exchange minimums.
    probe_qty = _probe_basequantity(xc, probe_base, lastprice)
    isnothing(probe_qty) && error("KrakenSpot startup probe failed: minimum base quantity unavailable for base=$(probe_base)")

    limits = CryptoXch.marginlimits(xc, symbol; role=CryptoXch.trade_exchange_spot)
    probe_quote = probe_qty * buy_price
    @info "KrakenSpot startup order capability probe placing test orders" base=probe_base symbol=symbol probe_qty=probe_qty probe_quote_usdt=probe_quote buy_price=buy_price sell_price=sell_price short_leverage=short_leverage maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell

    oid_long = CryptoXch.createbuyorder(xc, probe_base; limitprice=buy_price, basequantity=probe_qty, maker=true, marginleverage=0)
    isnothing(oid_long) && error("KrakenSpot startup probe failed: long buy order was not accepted for base=$(probe_base) symbol=$(symbol)")
    @info "KrakenSpot startup probe long buy order placed" base=probe_base order_id=String(oid_long) buy_price=buy_price

    oid_short = try
        CryptoXch.createsellorder(xc, probe_base; limitprice=sell_price, basequantity=probe_qty, maker=true, marginleverage=short_leverage)
    catch err
        rethrow(err)
    end
    if isnothing(oid_short)
        error("KrakenSpot startup probe failed: short sell margin order was not accepted for base=$(probe_base) symbol=$(symbol) leverage=$(short_leverage)x")
    end
    @info "KrakenSpot startup probe short sell order placed" base=probe_base order_id=String(oid_short) sell_price=sell_price

    @info "KrakenSpot startup order capability probe passed" base=probe_base symbol=symbol long_order_id=String(oid_long) short_order_id=String(oid_short)
    return nothing
end

"Pick a KrakenFutures startup probe base from whitelist that supports short leverage."
function _pick_krakenfutures_probe_base(xc::CryptoXch.XchCache, whitelist::Vector{String}, short_leverage::Int)::Union{Nothing, String}
    for base in whitelist
        symbol = CryptoXch.symboltoken(xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
        if CryptoXch.validsymbol(xc, symbol) && CryptoXch.marginpermitted(xc, symbol, "Sell", short_leverage; role=CryptoXch.trade_exchange_spot)
            return base
        end
    end
    return nothing
end

"""
Run startup KrakenFutures order capability checks for one probe pair.

Creates two post-only limit orders 2% away from current price:
- long buy (leverage 0, matching runtime long flow)
- short sell (margin leverage default 2, matching runtime short flow)
"""
function krakenfutures_startup_order_capability_probe!(xc::CryptoXch.XchCache, whitelist::Vector{String}; price_offset::Float64=0.02, short_leverage::Int=2)
    if !_env_true("TRADEREAL_KRAKENFUTURES_STARTUP_ORDER_PROBE", true)
        @info "KrakenFutures startup order capability probe disabled via TRADEREAL_KRAKENFUTURES_STARTUP_ORDER_PROBE"
        return nothing
    end

    base_override = uppercase(strip(get(ENV, "TRADEREAL_KRAKENFUTURES_STARTUP_ORDER_PROBE_BASE", "")))
    probe_base = isempty(base_override) ? _pick_krakenfutures_probe_base(xc, whitelist, short_leverage) : base_override
    isnothing(probe_base) && error("KrakenFutures startup probe failed: no whitelist base supports short leverage=$(short_leverage)x")

    symbol = CryptoXch.symboltoken(xc, probe_base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)
    ticker = CryptoXch.KrakenFutures.get24h(xc.bc, symbol)
    isnothing(ticker) && error("KrakenFutures startup probe failed: missing ticker for base=$(probe_base) symbol=$(symbol)")
    lastprice = Float64(ticker.lastprice)
    lastprice > 0.0 || error("KrakenFutures startup probe failed: invalid lastprice=$(lastprice) for base=$(probe_base) symbol=$(symbol)")

    # Long buy 2% below current price; short sell 2% above — both safely away from the market.
    buy_price = Float32(lastprice * (1.0 - price_offset))
    sell_price = Float32(lastprice * (1.0 + price_offset))
    # 2 × minimum notional in base units so the order clearly clears exchange minimums.
    probe_qty = _probe_basequantity(xc, probe_base, lastprice)
    isnothing(probe_qty) && error("KrakenFutures startup probe failed: minimum base quantity unavailable for base=$(probe_base)")

    limits = CryptoXch.marginlimits(xc, symbol; role=CryptoXch.trade_exchange_spot)
    probe_quote = probe_qty * buy_price
    @info "KrakenFutures startup order capability probe placing test orders" base=probe_base symbol=symbol probe_qty=probe_qty probe_quote_usdt=probe_quote buy_price=buy_price sell_price=sell_price short_leverage=short_leverage maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell

    oid_long = CryptoXch.createbuyorder(xc, probe_base; limitprice=buy_price, basequantity=probe_qty, maker=true, marginleverage=0)
    isnothing(oid_long) && error("KrakenFutures startup probe failed: long buy order was not accepted for base=$(probe_base) symbol=$(symbol)")
    @info "KrakenFutures startup probe long buy order placed" base=probe_base order_id=String(oid_long) buy_price=buy_price

    oid_short = try
        CryptoXch.createsellorder(xc, probe_base; limitprice=sell_price, basequantity=probe_qty, maker=true, marginleverage=short_leverage)
    catch err
        rethrow(err)
    end
    if isnothing(oid_short)
        error("KrakenFutures startup probe failed: short sell order was not accepted for base=$(probe_base) symbol=$(symbol) leverage=$(short_leverage)x")
    end
    @info "KrakenFutures startup probe short sell order placed" base=probe_base order_id=String(oid_short) sell_price=sell_price

    @info "KrakenFutures startup order capability probe passed" base=probe_base symbol=symbol long_order_id=String(oid_long) short_order_id=String(oid_short)
    return nothing
end

function loadtrend046classifier(model_folder::AbstractString)::Classify.RuntimeNNClassifier
    cfg046 = mk046config()
    nnstub = cfg046.classifiermodel(Features.featurecount(cfg046.featconfig), Targets.uniquelabels(cfg046.targetconfig), "mix")
    required_minutes = max(Features.requiredminutes(cfg046.featconfig), 2)
    return Classify.loadclassifier(
        nnstub.fileprefix,
        trendf6config09,
        required_minutes;
        search_folders=[String(model_folder), "Trend-046-training"],
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Redirect Ctrl+C to Julia exception handling so the finally block runs.
ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

EnvConfig.init(production)
EnvConfig.cryptoquote = QUOTE_COIN
selected_exchange = _resolve_exchange(ARGS, EXCHANGE)
exchange_log_token = _exchange_logtoken(selected_exchange)
EnvConfig.setcoinspath!("coins_" * exchange_log_token)
log_subfolder = LOG_SUBFOLDER_PREFIX * "-" * exchange_log_token * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
orders_subfolder = joinpath(log_subfolder, "orders")
classifier = try
    loadtrend046classifier(MODEL046_FOLDER)
catch err
    println(stderr, "$(EnvConfig.now()): ERROR failed to load configured classifier: $(sprint(showerror, err))")
    println(stderr, "$(EnvConfig.now()): tradereal aborted")
    exit(1)
end
EnvConfig.setlogpath(log_subfolder)

messagelogfn = EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt")
println("$(EnvConfig.now()): starting tradereal with config=$CONFIG046_NAME")
println("$(EnvConfig.now()): messages logged to $messagelogfn")

demux_logger = TeeLogger(
    MinLevelLogger(FileLogger(messagelogfn, always_flush=true), Logging.Info),
    MinLevelLogger(ConsoleLogger(stdout), Logging.Info),
)
defaultlogger = global_logger(demux_logger)

CryptoXch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 2

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

xc = CryptoXch.XchCache(; enddt=nothing, exchange=selected_exchange)
CryptoXch.setstartdt(xc, CryptoXch.tradetime(xc))

cache = Trade.TradeCache(xc=xc, cl=classifier, trademode=TRADE_MODE)
run_max_budget_quote = max_budget_from_env(MAX_BUDGET_QUOTE)

# Apply config 046 strategy parameters.
Trade.apply_tradingstrategy!(cache, CONFIG046_STRATEGY;
    strategy_engine=:getgainsalgo,
    source="trenddetector:$CONFIG046_NAME")

# Override whitelist and risk parameters.
whitelist = normalize_whitelist(WHITELIST_INPUT, QUOTE_COIN)
restricted = normalize_whitelist(RESTRICTED_COINS_INPUT, QUOTE_COIN)
if !isempty(restricted)
    whitelist = [b for b in whitelist if !(b in Set(restricted))]
end
cache.mc[:whitelistcoins]    = whitelist
cache.mc[:restrictedcoins]   = restricted
cache.mc[:maxassetfraction]  = MAX_ASSET_FRACTION
cache.mc[:maxbudgetquote]    = run_max_budget_quote
cache.mc[:budgetsafetymargin] = SAFETY_MARGIN
cache.mc[:tradelog_portfolio_snapshot_mode] = :session_start
cache.mc[:tradelog_migration_worker_probe_enabled] = _env_true("CTS_TRADELOG_MIGRATION_WORKER_PROBE_ENABLED", true)
cache.xc.mc[:tradelog_migration_fill_balance_enabled] = _env_true("CTS_TRADELOG_MIGRATION_FILL_BALANCE_ENABLED", true)
cache.mc[:ws_orders_enabled] = _env_true("CTS_WS_ORDERS_ENABLED", false)
cache.mc[:ws_balances_enabled] = _env_true("CTS_WS_BALANCES_ENABLED", false)
cache.mc[:ws_shadow_mode] = _env_true("CTS_WS_SHADOW_MODE", true)
cache.mc[:ws_primary_mode] = _env_true("CTS_WS_PRIMARY_MODE", false)
cache.mc[:ws_primary_autofallback_on_mismatch] = _env_true("CTS_WS_PRIMARY_AUTOFALLBACK_ON_MISMATCH", true)
cache.xc.mc[:ws_orders_enabled] = cache.mc[:ws_orders_enabled]
cache.xc.mc[:ws_balances_enabled] = cache.mc[:ws_balances_enabled]
cache.xc.mc[:ws_primary_mode] = cache.mc[:ws_primary_mode]

println("$(EnvConfig.now()): exchange=$selected_exchange, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG046_NAME, engine=getgainsalgo")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN")
println("$(EnvConfig.now()): max budget cap quote=$(isnothing(run_max_budget_quote) ? "none" : run_max_budget_quote)")
println("$(EnvConfig.now()): budget safety margin=$(SAFETY_MARGIN)")
println("$(EnvConfig.now()): whitelist ($(length(whitelist)) bases): $whitelist")
println("$(EnvConfig.now()): restricted coins ($(length(restricted)) bases): $restricted")
println("$(EnvConfig.now()): starting live trade loop — press Ctrl+C to stop")

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ─────────────────────────────────────────────────────────────────────────────

# Validate exchange credentials by fetching the live balance.
# Fails fast with a clear error if API keys are missing or rejected.

let
    init_balances = CryptoXch.balances(xc, ignoresmallvolume=false)
    @info "Startup credential check OK: $(selected_exchange) returned $(size(init_balances, 1)) balance entries"
    init_assets = CryptoXch.portfolio!(xc, init_balances; ignoresmallvolume=false)
    capacity = CryptoXch.accountcapacity(xc; force_refresh=true)
    budget_limit_quote = Trade._effectivebudgetquote(cache, init_assets)
    allocated_budget_quote = Trade._allocatedbudgetquote(init_assets)
    maxassetquote = cache.mc[:maxassetfraction] * budget_limit_quote
    overallocated_quote = max(0.0, allocated_budget_quote - budget_limit_quote)
    @info "Startup budget allocation (quote)" equity_quote=capacity.equity_quote available_opening_quote=capacity.available_opening_quote available_long_quote=capacity.available_long_quote available_short_quote=capacity.available_short_quote source=capacity.source budget_limit_quote=budget_limit_quote allocated_budget_quote=allocated_budget_quote overallocated_quote=overallocated_quote maxassetquote=maxassetquote maxassetfraction=cache.mc[:maxassetfraction] safety_margin=cache.mc[:budgetsafetymargin] max_budget_cap_quote=cache.mc[:maxbudgetquote]
end

# Log any pre-existing open orders — they will be cancelled at the first trade step.
let
    preexisting_oo = try
        attempts = 6
        last_err = nothing
        preexisting_oo_local = nothing
        while attempts > 0
            try
                preexisting_oo_local = CryptoXch.getopenorders(xc)
                last_err = nothing
                break
            catch err
                last_err = err
                attempts -= 1
                if occursin("invalid nonce", lowercase(sprint(showerror, err))) && (attempts > 0)
                    retry_ix = 6 - attempts
                    wait_s = min(1.0, 0.1 * retry_ix)
                    @warn "startup open-orders check hit invalid nonce; retrying" attempts_left=attempts sleep_seconds=wait_s
                    sleep(wait_s)
                    continue
                end
                rethrow(err)
            end
        end
        isnothing(last_err) ? preexisting_oo_local : throw(last_err)
    catch err
        if occursin("invalid nonce", lowercase(sprint(showerror, err)))
            @error "startup auth validation failed: persistent Kraken invalid nonce" exchange=selected_exchange remediation="Use an API key dedicated to this bot, ensure no other process/client uses the same key, or rotate to a fresh key and restart."
            error("startup auth validation failed: open orders request still returned invalid nonce after retries; trading aborted")
        end
        @warn "startup open-orders check skipped after retries" error=sprint(showerror, err)
        nothing
    end
    if !isnothing(preexisting_oo) && (size(preexisting_oo, 1) > 0)
        for row in eachrow(preexisting_oo)
            @info "  pre-existing order: $(row.symbol) $(row.side) qty=$(row.baseqty) @ $(row.limitprice) status=$(row.status) id=$(row.orderid)"
        end
    elseif !isnothing(preexisting_oo)
        @info "No pre-existing open orders found at startup"
    end
end

# if selected_exchange == CryptoXch.EXCHANGE_KRAKENSPOT
#     krakenspot_startup_order_capability_probe!(xc, whitelist)
# elseif selected_exchange == CryptoXch.EXCHANGE_KRAKENFUTURES
#     krakenfutures_startup_order_capability_probe!(xc, whitelist)
# end

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

try
    Trade.run_live!(cache)
finally
    try
        for ex in unique([
            CryptoXch.exchange(cache.xc),
            CryptoXch._routeexchange(cache.xc.routing, CryptoXch.trade_exchange_spot, CryptoXch.exchange(cache.xc)),
            CryptoXch._routeexchange(cache.xc.routing, CryptoXch.trade_exchange_futures, CryptoXch.exchange(cache.xc)),
        ])
            if ex == CryptoXch.EXCHANGE_KRAKENSPOT
                CryptoXch.KrakenSpot.log_private_call_summary!()
            elseif ex == CryptoXch.EXCHANGE_KRAKENFUTURES
                CryptoXch.KrakenFutures.log_private_call_summary!()
            end
        end
    catch err
        @warn "failed to write private call summary" error=sprint(showerror, err)
    end
    EnvConfig.setlogpath(orders_subfolder)
    CryptoXch.writeorders(cache.xc)
    @info "$(EnvConfig.now()): saved production orders to $(EnvConfig.logfolder())"
    @info "$(EnvConfig.now()): tradereal finished"
    global_logger(defaultlogger)
end
