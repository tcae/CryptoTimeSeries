"""
tradereal.jl — Live trading script using a selected trading strategy context.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before starting. The loop runs until Ctrl+C is pressed.

Set `ts=<configname>` or `TRADEREAL_TS_REF` to pick a `TS_CONFIGS` preset (classifier + strategy, default `046`).
The adapted classifier of the production coin set and date range (`CS01`/`D01`) is loaded.

Usage:
    julia --project=scripts scripts/tradereal.jl [help] [ts=<configname>]
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

"Return the `key=value` value of `key` in `args` or `default`."
function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) && return split(arg, "="; limit=2)[2]
    end
    return default
end

const CLI_FLAGS = ("help", "--help", "-h")
const CLI_KEYS = ("ts",)

"Assert that every CLI argument is a known flag or `key=value` parameter to fail fast on typos and obsolete parameters."
function _assertknownargs(args::Vector{String})
    for arg in args
        if occursin("=", arg)
            key = first(split(arg, "="; limit=2))
            @assert key in CLI_KEYS "unknown parameter $(arg); known key=value parameters: $(join(CLI_KEYS, ", ")) - see `help`"
        else
            @assert arg in CLI_FLAGS "unknown flag $(arg); known flags: $(join(CLI_FLAGS, ", ")) - see `help`"
        end
    end
    return nothing
end

"Return CLI help text for `tradereal.jl`."
function traderealhelp()::String
    return """
Usage:
    julia --project=scripts scripts/tradereal.jl [help] [ts=<configname>]

Live trading loop on the exchange configured by `EXCHANGE` in the CONFIG block of the script
(trade mode, quote coin and auth alias are configured there as well). Stop with Ctrl+C.

Flag parameters:
  help, --help, -h
      Show this message and exit.

Key=value parameters:
  ts=<configname>
      Trading strategy context from `TS_CONFIGS` in `TradingStrategy/src/tradingstrategyconfig.jl`.
      Its classifier must be adapted for coin set `CS01` and date range `D01`, e.g. via
      `julia --project=. scripts/TrendDetector.jl train ts=<configname> stage=train`.
      Default: `TRADEREAL_TS_REF` env var, or `046`

Unknown flags or parameters abort the run.
"""
end

if any(arg -> arg in CLI_FLAGS, ARGS)
    println(traderealhelp())
    exit(0)
end
_assertknownargs(ARGS)

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

const TS_CONFIG = TradingStrategy.tsconfig(_argvalue(ARGS, "ts", get(ENV, "TRADEREAL_TS_REF", "046")))
# Run label distinguishing log output of different trading strategies applied to the same classifier.
const RUN_LABEL = "ts$(TS_CONFIG.configname)"

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

strategy_runtime = TradingStrategy.TsCache(strategy=TradingStrategy.strategyconfig(TS_CONFIG), source=TradingStrategy.tsconfigsource(TS_CONFIG))

cache = Trade.TradeCache(strategy_runtime, xc=xc, trademode=TRADE_MODE)

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy context=$(TradingStrategy.tsconfigsource(TS_CONFIG)), classifier=$(TradingStrategy.tsclassifiername(TS_CONFIG))")
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
