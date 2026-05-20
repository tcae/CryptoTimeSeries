"""
tradereal.jl — Live trading script using TrendDetector config 046.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before starting. The loop runs until Ctrl+C is pressed.

Usage:
    julia --project=scripts scripts/tradereal.jl
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
                         "TRX", "ALGO", "TON", "INJ", "MON"]

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# Optional cap for overall budget considered by trade sizing.
# If set, sizing uses min(real portfolio quote value, MAX_BUDGET_QUOTE).
const MAX_BUDGET_QUOTE = nothing

# TrendDetector config 046: GainSegment strategy parameters.
# These come from mk046config() → tradingstrategy03().
# Override individual fields here if needed.
const CONFIG046_STRATEGY = tradingstrategy03()  # GainSegment(maxwindow=240, algorithm=gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015)
const CONFIG046_NAME = "046"
const MODEL046_FOLDER = "Trend-046-production"

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradereal-" * CONFIG046_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
const ORDERS_SUBFOLDER = joinpath(LOG_SUBFOLDER, "orders")

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
classifier = try
    loadtrend046classifier(MODEL046_FOLDER)
catch err
    println(stderr, "$(EnvConfig.now()): ERROR failed to load configured classifier: $(sprint(showerror, err))")
    println(stderr, "$(EnvConfig.now()): tradereal aborted")
    exit(1)
end
EnvConfig.setlogpath(LOG_SUBFOLDER)

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

xc = CryptoXch.XchCache(; enddt=nothing, exchange=EXCHANGE)
CryptoXch.setstartdt(xc, CryptoXch.tradetime(xc))

cache = Trade.TradeCache(xc=xc, cl=classifier, trademode=TRADE_MODE)
run_max_budget_quote = max_budget_from_env(MAX_BUDGET_QUOTE)

# Apply config 046 strategy parameters.
Trade.apply_tradingstrategy!(cache, CONFIG046_STRATEGY;
    strategy_engine=:getgainsalgo,
    source="trenddetector:$CONFIG046_NAME")

# Override whitelist and risk parameters.
whitelist = normalize_whitelist(WHITELIST_INPUT, QUOTE_COIN)
cache.mc[:whitelistcoins]    = whitelist
cache.mc[:maxassetfraction]  = MAX_ASSET_FRACTION
cache.mc[:maxbudgetquote]    = run_max_budget_quote
cache.mc[:audit_portfolio_snapshot_mode] = :session_start

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG046_NAME, engine=getgainsalgo")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN")
println("$(EnvConfig.now()): max budget cap quote=$(isnothing(run_max_budget_quote) ? "none" : run_max_budget_quote)")
println("$(EnvConfig.now()): whitelist ($(length(whitelist)) bases): $whitelist")
println("$(EnvConfig.now()): starting live trade loop — press Ctrl+C to stop")

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ─────────────────────────────────────────────────────────────────────────────

# Validate exchange credentials by fetching the live balance.
# Fails fast with a clear error if API keys are missing or rejected.
let
    init_balances = CryptoXch.balances(xc, ignoresmallvolume=false)
    @info "Startup credential check OK: $(EXCHANGE) returned $(size(init_balances, 1)) balance entries"
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
            @error "startup auth validation failed: persistent Kraken invalid nonce" exchange=EXCHANGE remediation="Use an API key dedicated to this bot, ensure no other process/client uses the same key, or rotate to a fresh key and restart."
            error("startup auth validation failed: open orders request still returned invalid nonce after retries; trading aborted")
        end
        @warn "startup open-orders check skipped after retries" error=sprint(showerror, err)
        nothing
    end
    if !isnothing(preexisting_oo) && (size(preexisting_oo, 1) > 0)
        @warn "$(size(preexisting_oo, 1)) pre-existing open order(s) found — will be cancelled at first trade step"
        for row in eachrow(preexisting_oo)
            @info "  pre-existing order: $(row.symbol) $(row.side) qty=$(row.baseqty) @ $(row.limitprice) status=$(row.status) id=$(row.orderid)"
        end
    elseif !isnothing(preexisting_oo)
        @info "No pre-existing open orders found at startup"
    end
end

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
    EnvConfig.setlogpath(ORDERS_SUBFOLDER)
    CryptoXch.writeorders(cache.xc)
    @info "$(EnvConfig.now()): saved production orders to $(EnvConfig.logfolder())"
    @info "$(EnvConfig.now()): tradereal finished"
    global_logger(defaultlogger)
end
