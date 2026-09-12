"""
tradereal.jl — Live trading script using a selected TrendDetector config.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before starting. The loop runs until Ctrl+C is pressed.

Set `TRADEREAL_CONFIG_REF` to pick a `TREND_DETECTOR_CONFIGS` preset, or
`TRADEREAL_STRAT_REF` to pick a `TS_CONFIGS` preset that keeps the classifier of its
referenced trend detector config but replaces that config's trading strategy.

Usage:
    julia --project=scripts scripts/tradereal.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using Dates, Logging, LoggingExtras
using EnvConfig, TradingStrategy, Trade, Classify, Xch, Features, Ohlcv, Targets, TSM
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
const STRAT_REF = begin
    raw = strip(get(ENV, "TRADEREAL_STRAT_REF", ""))
    isempty(raw) ? nothing : String(raw)
end
const STRAT_CONFIG = isnothing(STRAT_REF) ? nothing : TradingStrategy.tsconfig(STRAT_REF)
const CONFIG_REF = isnothing(STRAT_CONFIG) ? get(ENV, "TRADEREAL_CONFIG_REF", "046") : String(STRAT_CONFIG.tdconfigname)
const CONFIG = TradingStrategy.trenddetectorconfig(CONFIG_REF)
const CONFIG_NAME = String(CONFIG.configname)
const MODEL_FOLDER = TradingStrategy.trendconfigfolder(CONFIG, "production")
# Run label distinguishing log output of different trading strategies applied to the same classifier.
const RUN_LABEL = isnothing(STRAT_CONFIG) ? CONFIG_NAME : "$(CONFIG_NAME)-ts$(String(STRAT_CONFIG.configname))"

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradereal-" * RUN_LABEL * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
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
println("$(EnvConfig.now()): starting tradereal with config=$RUN_LABEL")
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

TSM.ensuretradesschema!(xc.tsm, TSM.tradesdf_all_contributors())

strategy_runtime = isnothing(STRAT_CONFIG) ?
    TradingStrategy.TsCache(CONFIG_REF; source="trenddetector:$CONFIG_NAME") :
    TradingStrategy.TsCache(strategy=TradingStrategy.tsstrategyconfig(STRAT_CONFIG), source=TradingStrategy.tsconfigsource(STRAT_CONFIG))

cache = Trade.TradeCache(strategy_runtime, xc=xc, trademode=TRADE_MODE)

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$RUN_LABEL, engine=tradingstrategy")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN")
println("$(EnvConfig.now()): blacklist ($(length(cache.blacklistbases)) bases): $(cache.blacklistbases)")
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
