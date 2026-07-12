"""
tradereal.jl — Live trading script using a selected TrendDetector config.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before starting. The loop runs until Ctrl+C is pressed.

Usage:
    julia --project=scripts scripts/tradereal.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using Dates, Logging, LoggingExtras
using EnvConfig, TradingStrategy, Trade, Classify, Xch, Features, Ohlcv, Targets
using Bybit, KrakenFutures, KrakenSpot

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these values before running
# ─────────────────────────────────────────────────────────────────────────────

# Exchange to use for live trading: Xch.EXCHANGE_BYBIT, EXCHANGE_KRAKENSPOT,
# or EXCHANGE_KRAKENFUTURES.
const EXCHANGE = Xch.EXCHANGE_KRAKENSPOT

# Optional auth alias (name of the credentials entry in EnvConfig).
# Set to nothing to use the default credentials for the exchange.
const AUTH_ALIAS = nothing

# Trade mode: Trade.buysell, Trade.closeonly, Trade.quickexit, or Trade.notrade.
const TRADE_MODE = Trade.buysell

const QUOTE_COIN = "USD"

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

const CONFIG_REF = get(ENV, "TRADEREAL_CONFIG_REF", "046")
const CONFIG = TradingStrategy.trenddetectorconfig(CONFIG_REF)
const CONFIG_NAME = String(CONFIG.configname)
const MODEL_FOLDER = TradingStrategy.trendconfigfolder(CONFIG, "production")

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradereal-" * CONFIG_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
const ORDERS_SUBFOLDER = joinpath(LOG_SUBFOLDER, "orders")

"Build one adapter cache matching the configured exchange id."
function build_adapter_cache(exchange::AbstractString)
    ex = String(exchange)
    if ex == Xch.EXCHANGE_BYBIT
        return Bybit.BybitCache()
    elseif ex == Xch.EXCHANGE_KRAKENSPOT
        return KrakenSpot.KrakenSpotCache()
    elseif ex == Xch.EXCHANGE_KRAKENFUTURES
        return KrakenFutures.KrakenFuturesCache()
    end
    error("unsupported tradereal exchange=$(exchange)")
end

function safe_runid()::String
    try
        return EnvConfig.runid()
    catch err
        println(stderr, "$(EnvConfig.now()): WARNING runid fallback active (no git repo in cwd): $(sprint(showerror, err))")
        return Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Redirect Ctrl+C to Julia exception handling so the finally block runs.
ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

EnvConfig.init(production)
EnvConfig.setpairquote!(QUOTE_COIN)
EnvConfig.setlogpath(LOG_SUBFOLDER)

messagelogfn = EnvConfig.logpath("messagelog_$(safe_runid()).txt")
println("$(EnvConfig.now()): starting tradereal with config=$CONFIG_NAME")
println("$(EnvConfig.now()): messages logged to $messagelogfn")

demux_logger = TeeLogger(
    MinLevelLogger(FileLogger(messagelogfn, always_flush=true), Logging.Info),
    MinLevelLogger(ConsoleLogger(stdout), Logging.Info),
)
defaultlogger = global_logger(demux_logger)

Xch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 2

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

bc = build_adapter_cache(EXCHANGE)
xc = Xch.XchCache(bc; enddt=nothing)
Xch.setstartdt(xc, Xch.tradetime(xc))

Xch.ensuretradesschema(xc, Xch.tradesdf_all_contributors())

cache = Trade.TradeCache(xc=xc, strategy=TradingStrategy.TsCache(CONFIG_REF; source="trenddetector:$CONFIG_NAME"), trademode=TRADE_MODE)

# Override risk parameters.
cache.mc[:maxassetfraction]  = MAX_ASSET_FRACTION
cache.mc[:audit_portfolio_snapshot_mode] = :session_start

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG_NAME, engine=tradingstrategy")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN")
println("$(EnvConfig.now()): blacklist ($(length(cache.mc[:blacklistbases])) bases): $(cache.mc[:blacklistbases])")
println("$(EnvConfig.now()): starting live trade loop — press Ctrl+C to stop")

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

try
    Trade.run_live!(cache)
finally
    EnvConfig.setlogpath(ORDERS_SUBFOLDER)
    @info "$(EnvConfig.now()): tradereal finished"
    global_logger(defaultlogger)
end
