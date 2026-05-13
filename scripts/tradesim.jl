"""
tradesim.jl — Backtest simulation script using TrendDetector config 046,
followed by a performance report.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before running.

Usage:
    julia --project=scripts scripts/tradesim.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using Dates, Statistics, Printf, Logging
using EnvConfig, TradingStrategy, Trade, Classify, CryptoXch, Bybit, Ohlcv, Features, Targets

include(joinpath(@__DIR__, "optimizationconfigs.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these values before running
# ─────────────────────────────────────────────────────────────────────────────

# Exchange used for the simulation exchange backend. BybitSim keeps the exchange
# explicit while still allowing the common trading code path to run.
const EXCHANGE = CryptoXch.EXCHANGE_BYBITSIM

# Backtest time range (UTC).
const BACKTEST_STARTDT = DateTime("2025-01-01T00:00:00")
const BACKTEST_ENDDT   = DateTime("2025-08-01T00:00:00")

# Trade mode during backtest: Trade.buysell, Trade.sellonly, Trade.notrade.
const TRADE_MODE = Trade.buysell

# Whitelist of base coins to consider for trading during the simulation.
const QUOTE_COIN = "USDT"
const WHITELIST_INPUT = [
    "BTC", "ETH", "HBAR", "PEPE", "XRP"
]

# Initial quote-asset balance used in simulation mode (cryptoxchsim).
const INITIAL_QUOTE_BALANCE = 100.0

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# TrendDetector config 046: GainSegment strategy parameters.
# These come from mk046config() → tradingstrategy03().
const CONFIG046_STRATEGY = tradingstrategy03()  # GainSegment(maxwindow=240, algorithm=gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015)
const CONFIG046_NAME = "046"
const MODEL046_FOLDER = "Trend-046-training"

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradesim-" * CONFIG046_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))

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

function seed_quote_balance!(xc::CryptoXch.XchCache, quote_coin::AbstractString, amount::Real)
    # Seed CryptoXch simulation state
    quote_ix = findfirst(==(quote_coin), xc.assets[!, :coin])
    if isnothing(quote_ix)
        push!(xc.assets, (
            coin=quote_coin,
            free=Float32(amount),
            locked=0f0,
            marginfree=0f0,
            marginlocked=0f0,
            assetborrowed=0f0,
            orderborrowed=0f0,
            accruedinterest=0f0,
        ))
    else
        xc.assets[quote_ix, :free] = Float32(amount)
    end
    
    # Seed underlying exchange cache simulation state if needed
    if CryptoXch.exchange(xc) == CryptoXch.EXCHANGE_BYBITSIM
        bc = xc.bc
        if !isnothing(bc.assets)  # Simulation state initialized
            Bybit.seedportfolio!(bc, quote_coin, amount)
        end
    elseif CryptoXch.exchange(xc) == CryptoXch.EXCHANGE_TESTXCH
        tc = xc.bc
        if !isnothing(tc.assets)  # Simulation state initialized
            CryptoXch.TestXch.seedportfolio!(tc, quote_coin, amount)
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

"""
    backtest_report(cache)

Print a performance report to stdout based on closed orders and the final
portfolio state recorded in `cache`.

Metrics reported:
- Total return (%) relative to initial USDT value
- Annualised return (%)
- Number of filled buy / sell orders
- Win rate of closed round-trips
- Sharpe ratio of daily portfolio returns (annualised, assuming 365 trading days)
- Maximum drawdown (%)
"""
function backtest_report(cache::Trade.TradeCache, startdt::DateTime, enddt::DateTime)
    co = cache.xc.closedorders
    println()
    println("=" ^ 60)
    println("  BACKTEST PERFORMANCE REPORT — config $CONFIG046_NAME")
    println("  Period : $(Dates.format(startdt, "yyyy-mm-dd")) → $(Dates.format(enddt, "yyyy-mm-dd"))")
    println("=" ^ 60)

    # ── Order statistics ───────────────────────────────────────────────────
    norders = size(co, 1)
    if norders == 0
        println("  No filled orders recorded.")
        println("=" ^ 60)
        return
    end
    nbuys  = count(r -> uppercasefirst(string(r)) == "Buy",  co[!, :side])
    nsells = count(r -> uppercasefirst(string(r)) == "Sell", co[!, :side])
    @printf("  Filled orders : %d  (buys: %d, sells: %d)\n", norders, nbuys, nsells)

    # ── Portfolio value series from debug info (if available) ──────────────
    # The cache.dbgdf rows carry :opentime and indirectly the USDT value.
    # For a robust value series we rely on closedorder fill prices.

    # Try to reconstruct a daily portfolio value series from closed orders.
    # We track cumulative PnL per filled sell order (long-close gains/losses).
    # This is an approximation; a full mark-to-market series would require
    # the PORTFOLIO_SNAPSHOT audit rows.
    daily_pnl = Dict{Date, Float64}()
    for row in eachrow(co)
        day = Date(row.created)
        if !ismissing(row.executedqty) && !ismissing(row.avgprice) && uppercasefirst(string(row.side)) == "Sell"
            pnl = Float64(row.executedqty) * Float64(row.avgprice)
            daily_pnl[day] = get(daily_pnl, day, 0.0) + pnl
        end
    end

    # ── Return computation from portfolio snapshots in dbgdf ───────────────
    # If cache.dbgdf has :freeusdt and :totalusdt columns we can compute
    # proper returns; otherwise fall back to closed-order P&L approximation.
    dbg = cache.dbgdf
    has_portfolio = (size(dbg, 1) > 0) &&
                    (:freeusdt in propertynames(dbg)) &&
                    (:opentime in propertynames(dbg))

    if has_portfolio
        sort!(dbg, :opentime)
        dbg_valid = dbg[.!ismissing.(dbg[!, :freeusdt]) .&& .!ismissing.(dbg[!, :opentime]), :]

        if size(dbg_valid, 1) >= 2
            # Daily close values by date
            dates_series = Date.(dbg_valid[!, :opentime])
            vals = Float64.(dbg_valid[!, :freeusdt])
            unique_dates = sort(unique(dates_series))
            daily_vals = [last(vals[dates_series .== d]) for d in unique_dates]

            v0 = daily_vals[1]
            v1 = daily_vals[end]
            total_return_pct = (v1 / v0 - 1.0) * 100.0
            days = max(1, (enddt - startdt).value ÷ (1000 * 60 * 60 * 24))
            ann_return_pct = ((v1 / v0)^(365.0 / days) - 1.0) * 100.0

            @printf("  Initial portfolio value : %.2f USDT\n", v0)
            @printf("  Final   portfolio value : %.2f USDT\n", v1)
            @printf("  Total return            : %+.2f %%\n", total_return_pct)
            @printf("  Annualised return       : %+.2f %%\n", ann_return_pct)

            # Daily returns for Sharpe and drawdown
            daily_returns = diff(daily_vals) ./ daily_vals[1:end-1]
            if length(daily_returns) >= 2
                mu  = mean(daily_returns)
                sig = std(daily_returns)
                sharpe = sig > 0 ? mu / sig * sqrt(365.0) : 0.0
                @printf("  Sharpe ratio (ann.)     : %.3f\n", sharpe)
            end

            # Maximum drawdown
            peak = daily_vals[1]
            max_dd = 0.0
            for v in daily_vals
                peak = max(peak, v)
                dd = (peak - v) / peak
                max_dd = max(max_dd, dd)
            end
            @printf("  Maximum drawdown        : %.2f %%\n", max_dd * 100.0)
        else
            println("  Insufficient portfolio time-series for return metrics.")
        end
    else
        println("  (No portfolio time-series in cache.dbgdf; skipping return metrics.)")
    end

    # ── Win rate from closed orders ────────────────────────────────────────
    # Pair buys and sells by base coin, compute simple win/loss count.
    if :symbol in propertynames(co) && :side in propertynames(co)
        buy_prices  = Dict{String, Vector{Float64}}()
        sell_prices = Dict{String, Vector{Float64}}()
        for row in eachrow(co)
            sym = string(row.symbol)
            if !ismissing(row.avgprice)
                if uppercasefirst(string(row.side)) == "Buy"
                    push!(get!(buy_prices,  sym, Float64[]), Float64(row.avgprice))
                elseif uppercasefirst(string(row.side)) == "Sell"
                    push!(get!(sell_prices, sym, Float64[]), Float64(row.avgprice))
                end
            end
        end
        wins = losses = 0
        for sym in keys(sell_prices)
            bvec = get(buy_prices,  sym, Float64[])
            svec = sell_prices[sym]
            pairs = min(length(bvec), length(svec))
            for i in 1:pairs
                if svec[i] > bvec[i]
                    wins += 1
                else
                    losses += 1
                end
            end
        end
        total_pairs = wins + losses
        if total_pairs > 0
            @printf("  Matched round-trips     : %d  (wins: %d, losses: %d, win rate: %.1f %%)\n",
                total_pairs, wins, losses, 100.0 * wins / total_pairs)
        end
    end

    println("=" ^ 60)
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

EnvConfig.init(test)  # test mode → cryptoxchsim, no live credentials needed
EnvConfig.cryptoquote = QUOTE_COIN
classifier = try
    loadtrend046classifier(MODEL046_FOLDER)
catch err
    println(stderr, "$(EnvConfig.now()): ERROR failed to load configured classifier: $(sprint(showerror, err))")
    println(stderr, "$(EnvConfig.now()): tradesim aborted")
    exit(1)
end
EnvConfig.setlogpath(LOG_SUBFOLDER)

CryptoXch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 2

println("$(EnvConfig.now()): starting tradesim with config=$CONFIG046_NAME")
println("$(EnvConfig.now()): backtest $BACKTEST_STARTDT → $BACKTEST_ENDDT")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

xc = CryptoXch.XchCache(;
    startdt  = BACKTEST_STARTDT,
    enddt    = BACKTEST_ENDDT,
    exchange = EXCHANGE,
)

cache = Trade.TradeCache(xc=xc, cl=classifier, trademode=TRADE_MODE)
seed_quote_balance!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)

# Apply config 046 strategy parameters.
Trade.apply_tradingstrategy!(cache, CONFIG046_STRATEGY;
    strategy_engine=:getgainsalgo,
    source="trenddetector:$CONFIG046_NAME")

# Override whitelist and risk parameters.
whitelist = normalize_whitelist(WHITELIST_INPUT, QUOTE_COIN)
cache.mc[:whitelistcoins]   = whitelist
cache.mc[:maxassetfraction] = MAX_ASSET_FRACTION

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG046_NAME, engine=getgainsalgo")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN, initial balance=$INITIAL_QUOTE_BALANCE")
println("$(EnvConfig.now()): whitelist ($(length(whitelist)) bases): $whitelist")
println("$(EnvConfig.now()): running backtest...")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

Trade.run_backtest!(cache)

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

backtest_report(cache, BACKTEST_STARTDT, BACKTEST_ENDDT)
