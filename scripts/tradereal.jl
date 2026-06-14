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

include(joinpath(@__DIR__, "optimizationconfigs.jl"))

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

# Whitelist of base coins to consider for trading.
# Only coins in this list and meeting liquidity requirements will be traded.
# Entries can be either base symbols (e.g. "BTC") or pair symbols (e.g. "BTC/USD").
const QUOTE_COIN = "USD"
const WHITELIST_INPUT = [
    "ADA", "AI16Z", "AKT", "APEX", "AAVE", "BNB", "BTC", "CAKE", "CC", "DOGE",
    "ELX", "ENA", "ESPORTS", "ETH", "FET", "HBAR", "HFT", "HYPE", "ICP", "JUP", "LINK", "LTC",
    "MNT", "NEAR", "ONDO", "PEPE", "POPCAT", "S", "SOL", "STG", "SUI", "TAO", "TON",
    "TRX", "VIRTUAL", "VVV", "W", "WAL", "WIF", "WLD", "X", "XLM", "XMR", "XRP", "ZEC"
]

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

const CONFIG_REF = get(ENV, "TRADEREAL_CONFIG_REF", "046")
const CONFIG = trenddetectorconfig(CONFIG_REF)
const CONFIG_NAME = String(CONFIG.configname)
const CONFIG_STRATEGY = CONFIG.tradingstrategy
const MODEL_FOLDER = trendconfigfolder(CONFIG, "production")

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradereal-" * CONFIG_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
const ORDERS_SUBFOLDER = joinpath(LOG_SUBFOLDER, "orders")

function safe_runid()::String
    try
        return EnvConfig.runid()
    catch err
        println(stderr, "$(EnvConfig.now()): WARNING runid fallback active (no git repo in cwd): $(sprint(showerror, err))")
        return Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
    end
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

function env_bool(name::AbstractString, default::Bool)::Bool
    raw = strip(lowercase(get(ENV, String(name), default ? "true" : "false")))
    raw in ["1", "true", "yes", "on"] && return true
    raw in ["0", "false", "no", "off"] && return false
    return default
end

mutable struct TrendConfigRuntimeClassifier <: Classify.AbstractClassifier
    cfg::NamedTuple
    bc::Dict{AbstractString, NamedTuple}
    nn::Classify.NN
    cfgid::Int
    function TrendConfigRuntimeClassifier(cfg::NamedTuple, nn::Classify.NN)
        new(cfg, Dict{AbstractString, NamedTuple}(), nn, 1)
    end
end

function Classify.addbase!(cl::TrendConfigRuntimeClassifier, ohlcv::Ohlcv.OhlcvData)
    f6 = deepcopy(cl.cfg.featconfig)
    Features.setbase!(f6, ohlcv, usecache=true)
    cl.bc[ohlcv.base] = (ohlcv=ohlcv, f6=f6)
end

function Classify.supplement!(cl::TrendConfigRuntimeClassifier)
    for basecfg in values(cl.bc)
        Features.supplement!(basecfg.f6)
    end
end

Classify.requiredminutes(cl::TrendConfigRuntimeClassifier)::Integer = max(Features.requiredminutes(cl.cfg.featconfig), 2)

function Classify.advice(cl::TrendConfigRuntimeClassifier, base::AbstractString, dt::DateTime; investment::Union{Nothing, Classify.TradeAdvice}=nothing)::Union{Nothing, Classify.TradeAdvice}
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

function loadtrendclassifier(cfg::NamedTuple; mnemonic::AbstractString="mix", model_folder::AbstractString=trendconfigfolder(cfg, "production"), training_folder::AbstractString=trendconfigfolder(cfg, "training"))::TrendConfigRuntimeClassifier
    nnstub = cfg.classifiermodel(Features.featurecount(cfg.featconfig), Targets.uniquelabels(cfg.targetconfig), mnemonic)
    for folder in unique([String(model_folder), String(training_folder)])
        EnvConfig.setlogpath(folder)
        nnpath = Classify.nnfilename(nnstub.fileprefix)
        if isfile(nnpath)
            try
                nn = Classify.loadnn(nnstub.fileprefix)
                return TrendConfigRuntimeClassifier(cfg, nn)
            catch err
                shorterr = sprint(showerror, err)
                error("TrendDetector classifier exists but could not be loaded: nnpath=$nnpath. Cause=$shorterr. Likely classifier artifact compatibility mismatch (Flux/Optimisers/BSON versions).")
            end
        end
    end
    error("TrendDetector classifier file not found for fileprefix=$(nnstub.fileprefix), checked folders=[$(model_folder), $(training_folder)]")
end

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Redirect Ctrl+C to Julia exception handling so the finally block runs.
ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

EnvConfig.init(production)
EnvConfig.setpairquote!(QUOTE_COIN)
classifier = try
    loadtrendclassifier(CONFIG; model_folder=MODEL_FOLDER)
catch err
    println(stderr, "$(EnvConfig.now()): ERROR failed to load configured classifier: $(sprint(showerror, err))")
    println(stderr, "$(EnvConfig.now()): tradereal aborted")
    exit(1)
end
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

xc = Xch.XchCache(; enddt=nothing, exchange=EXCHANGE, defaultquote=QUOTE_COIN)
Xch.setstartdt(xc, Xch.tradetime(xc))

cache = Trade.TradeCache(xc=xc, cl=classifier, trademode=TRADE_MODE)

# Apply the selected TrendDetector strategy parameters.
Trade.apply_tradingstrategy!(cache, CONFIG_STRATEGY;
    strategy_engine=:getgainsalgo,
    source="trenddetector:$CONFIG_NAME")

# Override whitelist and risk parameters.
whitelist = normalize_whitelist(WHITELIST_INPUT, QUOTE_COIN)
cache.mc[:whitelistcoins]    = whitelist
cache.mc[:maxassetfraction]  = MAX_ASSET_FRACTION
cache.mc[:usenewtrade]       = env_bool("TRADEREAL_USE_NEW_TRADE", true)
cache.mc[:audit_portfolio_snapshot_mode] = :session_start

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG_NAME, engine=getgainsalgo")
println("$(EnvConfig.now()): usenewtrade=$(cache.mc[:usenewtrade])")
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
    Xch.writeorders(cache.xc)
    @info "$(EnvConfig.now()): saved production orders to $(EnvConfig.logfolder())"
    @info "$(EnvConfig.now()): tradereal finished"
    global_logger(defaultlogger)
end
