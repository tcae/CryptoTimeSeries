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

# Optional auth alias (name of the credentials entry in EnvConfig).
# Set to nothing to use the default credentials for the exchange.
const AUTH_ALIAS = nothing

# Trade mode: Trade.buysell, Trade.sellonly, Trade.quickexit, or Trade.notrade.
const TRADE_MODE = Trade.buysell

# Whitelist of base coins to consider for trading.
# Only coins in this list and meeting liquidity requirements will be traded.
# Entries can be either base symbols (e.g. "BTC") or pair symbols (e.g. "BTC/USDT").
const QUOTE_COIN = "USDT"
const WHITELIST_INPUT = [
    "ADA", "AI16Z", "APEX", "AAVE", "BNB", "BTC", "CAKE", "DOGE",
    "ELX", "ENA", "ETH", "HBAR", "HFT", "JUP", "LINK", "LTC",
    "MNT", "ONDO", "PEPE", "POPCAT", "S", "SOL", "SUI", "TON",
    "TRX", "VIRTUAL", "W", "WAL", "WIF", "WLD", "X", "XLM", "XRP",
]

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# TrendDetector config 046: GainSegment strategy parameters.
# These come from mk046config() → tradingstrategy03().
# Override individual fields here if needed.
const CONFIG046_STRATEGY = tradingstrategy03()  # GainSegment(maxwindow=240, algorithm=gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015)
const CONFIG046_NAME = "046"
const MODEL046_FOLDER = "Trend-046-production"

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradereal-" * CONFIG046_NAME * "-" * Dates.format(now(), Dates.DateFormat("yymmdd-HHMMSS"))
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

mutable struct Trend046RuntimeClassifier <: Classify.AbstractClassifier
    bc::Dict{AbstractString, NamedTuple}
    nn::Classify.NN
    cfgid::Int
    function Trend046RuntimeClassifier(nn::Classify.NN)
        new(Dict{AbstractString, NamedTuple}(), nn, 1)
    end
end

function Classify.addbase!(cl::Trend046RuntimeClassifier, ohlcv::Ohlcv.OhlcvData)
    f6 = trendf6config09()
    Features.setbase!(f6, ohlcv, usecache=true)
    cl.bc[ohlcv.base] = (ohlcv=ohlcv, f6=f6)
end

function Classify.supplement!(cl::Trend046RuntimeClassifier)
    for basecfg in values(cl.bc)
        Features.supplement!(basecfg.f6)
    end
end

Classify.requiredminutes(::Trend046RuntimeClassifier)::Integer = max(Features.requiredminutes(trendf6config09()), 2)

function Classify.advice(cl::Trend046RuntimeClassifier, base::AbstractString, dt::DateTime; investment::Union{Nothing, Classify.TradeAdvice}=nothing)::Union{Nothing, Classify.TradeAdvice}
    haskey(cl.bc, base) || return nothing
    basecfg = cl.bc[base]
    fdf = Features.features(basecfg.f6, dt, dt)
    (isnothing(fdf) || size(fdf, 1) == 0) && return nothing
    x = permutedims(Matrix(fdf), (2, 1))
    scores, labels = Classify.maxpredict(cl.nn, x)
    isempty(labels) && return nothing
    label = labels[1] isa Targets.TradeLabel ? labels[1] : Targets.tradelabel(string(labels[1]))
    oix = Ohlcv.rowix(basecfg.ohlcv, dt)
    price = Ohlcv.dataframe(basecfg.ohlcv)[oix, :pivot]
    return Classify.TradeAdvice(cl, cl.cfgid, label, 1f0, base, price, dt, 0f0, Float32(scores[1]), investment)
end

function loadtrend046classifier(model_folder::AbstractString)::Trend046RuntimeClassifier
    cfg046 = mk046config()
    nnstub = cfg046.classifiermodel(Features.featurecount(cfg046.featconfig), Targets.uniquelabels(cfg046.targetconfig), "mix")
    for folder in unique([String(model_folder), "Trend-046-training"])
        EnvConfig.setlogpath(folder)
        nnpath = Classify.nnfilename(nnstub.fileprefix)
        if isfile(nnpath)
            try
                nn = Classify.loadnn(nnstub.fileprefix)
                return Trend046RuntimeClassifier(nn)
            catch err
                shorterr = sprint(showerror, err)
                error("TrendDetector 046 classifier exists but could not be loaded: nnpath=$nnpath. Cause=$shorterr. Likely classifier artifact compatibility mismatch (Flux/Optimisers/BSON versions).")
            end
        end
    end
    error("TrendDetector 046 classifier file not found for fileprefix=$(nnstub.fileprefix), checked folders=[$(model_folder), Trend-046-training]")
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

xc = CryptoXch.XchCache(; enddt=nothing, exchange=EXCHANGE, authname=AUTH_ALIAS)
CryptoXch.setstartdt(xc, CryptoXch.tradetime(xc))

cache = Trade.TradeCache(xc=xc, cl=classifier, trademode=TRADE_MODE)

# Apply config 046 strategy parameters.
Trade.apply_tradingstrategy!(cache, CONFIG046_STRATEGY;
    strategy_engine=:getgainsalgo,
    source="trenddetector:$CONFIG046_NAME")

# Override whitelist and risk parameters.
whitelist = normalize_whitelist(WHITELIST_INPUT, QUOTE_COIN)
cache.mc[:whitelistcoins]    = whitelist
cache.mc[:maxassetfraction]  = MAX_ASSET_FRACTION
cache.mc[:audit_portfolio_snapshot_mode] = :session_start

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG046_NAME, engine=getgainsalgo")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN")
println("$(EnvConfig.now()): whitelist ($(length(whitelist)) bases): $whitelist")
println("$(EnvConfig.now()): starting live trade loop — press Ctrl+C to stop")

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

try
    Trade.run_live!(cache)
finally
    EnvConfig.setlogpath(ORDERS_SUBFOLDER)
    CryptoXch.writeorders(cache.xc)
    @info "$(EnvConfig.now()): saved production orders to $(EnvConfig.logfolder())"
    @info "$(EnvConfig.now()): tradereal finished"
    global_logger(defaultlogger)
end
