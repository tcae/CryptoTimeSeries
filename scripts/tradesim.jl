"""
tradesim.jl — Backtest simulation script using a selected TrendDetector config,
followed by a performance report.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before running.

Usage:
    julia --project=scripts scripts/tradesim.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using Dates, Statistics, Printf, Logging
using DataFrames
using EnvConfig, TradingStrategy, Trade, Classify, Xch, Bybit, Ohlcv, Features, Targets

include(joinpath(@__DIR__, "optimizationconfigs.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these values before running
# ─────────────────────────────────────────────────────────────────────────────

# Exchange used for the simulation exchange backend. BybitSim keeps the exchange
# explicit while still allowing the common trading code path to run.
const EXCHANGE = Xch.EXCHANGE_BYBITSIM

# Backtest time range (UTC).
const BACKTEST_STARTDT = DateTime("2025-01-01T00:00:00")
const BACKTEST_ENDDT   = DateTime("2025-08-01T00:00:00")

# Trade mode during backtest: Trade.buysell, Trade.closeonly, Trade.notrade.
const TRADE_MODE = Trade.buysell

# Whitelist of base coins to consider for trading during the simulation.
const QUOTE_COIN = "USDT"
const WHITELIST_INPUT = [
    "BTC", "ETH", "HBAR", "PEPE", "XRP"
]

function whitelist_from_env(default::Vector{String})::Vector{String}
    raw = get(ENV, "TRADESIM_WHITELIST", "")
    isempty(strip(raw)) && return default
    vals = [strip(tok) for tok in split(raw, ',') if !isempty(strip(tok))]
    return isempty(vals) ? default : vals
end

# Initial quote-asset balance used in simulation mode (cryptoxchsim).
const INITIAL_QUOTE_BALANCE = 100000.0

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# Buy signal score threshold used by GainSegment strategy.
const BUY_OPEN_THRESHOLD = 0.4f0

# GainSegment strategy parameters used by the backtest.
const CONFIG_REF = get(ENV, "TRADESIM_CONFIG_REF", "046")
const CONFIG = trenddetectorconfig(CONFIG_REF)
const CONFIG_NAME = String(CONFIG.configname)
const CONFIG_STRATEGY = TradingStrategy.GainSegment(
    maxwindow=CONFIG.tradingstrategy.maxwindow,
    openthreshold=BUY_OPEN_THRESHOLD,
    closethreshold=CONFIG.tradingstrategy.closethreshold,
    algorithm=CONFIG.tradingstrategy.algorithm,
    makerfee=CONFIG.tradingstrategy.makerfee,
    takerfee=CONFIG.tradingstrategy.takerfee,
    buygain=CONFIG.tradingstrategy.buygain,
    sellgain=CONFIG.tradingstrategy.sellgain,
    limitreduction=CONFIG.tradingstrategy.limitreduction,
    minpricedelta=CONFIG.tradingstrategy.minpricedelta,
    max_classify_staleness_minutes=CONFIG.tradingstrategy.max_classify_staleness_minutes,
)
const MODEL_FOLDER = trendconfigfolder(CONFIG, "training")

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradesim-" * CONFIG_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
const ORDERS_SUBFOLDER = joinpath(LOG_SUBFOLDER, "orders")

function backtest_bounds_from_env(default_start::DateTime, default_end::DateTime)
    sraw = strip(get(ENV, "TRADESIM_STARTDT", ""))
    eraw = strip(get(ENV, "TRADESIM_ENDDT", ""))
    sdt = isempty(sraw) ? default_start : DateTime(sraw)
    edt = isempty(eraw) ? default_end : DateTime(eraw)
    @assert sdt <= edt "TRADESIM_STARTDT must be <= TRADESIM_ENDDT; got start=$(sdt), end=$(edt)"
    return sdt, edt
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

"Seed the simulation quote-currency balance in the exchange backend cache."
function seed_quote_balance!(xc::Xch.XchCache, quote_coin::AbstractString, amount::Real)
    isnothing(xc.bc) && error("cannot seed quote balance: exchange cache is not initialized")
    if applicable(Bybit.seedportfolio!, xc.bc, quote_coin, amount)
        Bybit.seedportfolio!(xc.bc, quote_coin, amount)
        return nothing
    end
    error("cannot seed quote balance for backend cache type=$(typeof(xc.bc))")
end

"Ensure the simulation starts with at least `minimum_free` quote balance."
function ensure_quote_budget!(xc::Xch.XchCache, quote_coin::AbstractString, minimum_free::Real)
    q = uppercase(String(quote_coin))
    balancesdf = Xch.balances(xc, ignoresmallvolume=false)
    qix = size(balancesdf, 1) > 0 ? findfirst(==(q), uppercase.(String.(balancesdf[!, :coin]))) : nothing
    current_free = isnothing(qix) ? 0.0 : Float64(balancesdf[qix, :free])
    if current_free + 1e-6 < Float64(minimum_free)
        seed_quote_balance!(xc, q, minimum_free)
        balancesdf = Xch.balances(xc, ignoresmallvolume=false)
        qix = size(balancesdf, 1) > 0 ? findfirst(==(q), uppercase.(String.(balancesdf[!, :coin]))) : nothing
        reseeded_free = isnothing(qix) ? 0.0 : Float64(balancesdf[qix, :free])
        @assert reseeded_free + 1e-6 >= Float64(minimum_free) "totalusdt seed $(q) budget is insufficient after reseed; expected >= $(minimum_free), got $(reseeded_free)"
        println("$(EnvConfig.now()): reseeded $(q) free balance from $(round(current_free, digits=2)) to $(round(reseeded_free, digits=2))")
    else
        println("$(EnvConfig.now()): confirmed $(q) free seed budget $(round(current_free, digits=2))")
    end
end

"Runtime classifier wrapper for the selected TrendDetector config inside tradesim."
mutable struct TrendConfigRuntimeClassifier <: Classify.AbstractClassifier
    cfg::NamedTuple
    bc::Dict{AbstractString, NamedTuple}
    nn::Classify.NN
    cfgid::Int
    function TrendConfigRuntimeClassifier(cfg::NamedTuple, nn::Classify.NN)
        new(cfg, Dict{AbstractString, NamedTuple}(), nn, 1)
    end
end

"Register one base in the runtime classifier and initialize its feature config."
function Classify.addbase!(cl::TrendConfigRuntimeClassifier, ohlcv::Ohlcv.OhlcvData)
    f6 = deepcopy(cl.cfg.featconfig)
    Features.setbase!(f6, ohlcv, usecache=true)
    cl.bc[ohlcv.base] = (ohlcv=ohlcv, f6=f6)
end

"Update all runtime feature states before requesting advice."
function Classify.supplement!(cl::TrendConfigRuntimeClassifier)
    for basecfg in values(cl.bc)
        Features.supplement!(basecfg.f6)
    end
end

Classify.requiredminutes(cl::TrendConfigRuntimeClassifier)::Integer = max(Features.requiredminutes(cl.cfg.featconfig), 2)

"Return one trade advice at dt, or nothing when data/features are insufficient."
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

"Load the selected TrendDetector classifier artifacts and return a runtime classifier instance."
function loadtrendclassifier(cfg::NamedTuple; model_folder::AbstractString=trendconfigfolder(cfg, "training"), training_folder::AbstractString=trendconfigfolder(cfg, "training"))::TrendConfigRuntimeClassifier
    nnstub = cfg.classifiermodel(Features.featurecount(cfg.featconfig), Targets.uniquelabels(cfg.targetconfig), "mix")
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
    println("  BACKTEST PERFORMANCE REPORT — config $CONFIG_NAME")
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

    # ── Win rate and gain metrics from closed orders ───────────────────────
    # Pair buys and sells by symbol in chronological order and calculate
    # realized gain metrics per matched round-trip.
    if (:symbol in propertynames(co)) && (:side in propertynames(co)) &&
       (:avgprice in propertynames(co)) && (:executedqty in propertynames(co))

        function symbol_base(sym::AbstractString)
            token = uppercase(strip(String(sym)))
            if occursin('/', token)
                return split(token, '/'; limit=2)[1]
            end
            quote_up = uppercase(QUOTE_COIN)
            if endswith(token, quote_up) && (length(token) > length(quote_up))
                return token[1:end-length(quote_up)]
            end
            return token
        end

        ordered = (:created in propertynames(co)) ? sort(co, :created) : co
        buy_fills  = Dict{String, Vector{Tuple{Float64, Float64}}}()
        sell_fills = Dict{String, Vector{Tuple{Float64, Float64}}}()
        for row in eachrow(ordered)
            if ismissing(row.symbol) || ismissing(row.side) || ismissing(row.avgprice) || ismissing(row.executedqty)
                continue
            end
            sym = string(row.symbol)
            px = Float64(row.avgprice)
            qty = Float64(row.executedqty)
            (px <= 0.0 || qty <= 0.0) && continue

            side = uppercasefirst(string(row.side))
            if side == "Buy"
                push!(get!(buy_fills, sym, Tuple{Float64, Float64}[]), (px, qty))
            elseif side == "Sell"
                push!(get!(sell_fills, sym, Tuple{Float64, Float64}[]), (px, qty))
            end
        end

        per_coin_gains_pct = Dict{String, Vector{Float64}}()
        per_coin_gains_usdt = Dict{String, Vector{Float64}}()
        wins = 0
        losses = 0

        for sym in keys(sell_fills)
            bvec = get(buy_fills, sym, Tuple{Float64, Float64}[])
            svec = sell_fills[sym]
            pairs = min(length(bvec), length(svec))
            pairs == 0 && continue

            coin = symbol_base(sym)
            gains_pct = get!(per_coin_gains_pct, coin, Float64[])
            gains_usdt = get!(per_coin_gains_usdt, coin, Float64[])

            for i in 1:pairs
                buy_px, buy_qty = bvec[i]
                sell_px, sell_qty = svec[i]
                qty = min(buy_qty, sell_qty)
                qty <= 0.0 && continue
                gain_usdt = (sell_px - buy_px) * qty
                gain_pct = ((sell_px / buy_px) - 1.0) * 100.0
                push!(gains_usdt, gain_usdt)
                push!(gains_pct, gain_pct)
                if gain_usdt > 0
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

            println("  Gain metrics by coin     :")
            @printf("    %-8s %7s %12s %16s\n", "coin", "count", "avg gain %", "total gain USDT")

            all_gain_pcts = Float64[]
            all_gain_usdt = Float64[]
            for coin in sort(collect(keys(per_coin_gains_usdt)))
                g_usdt = per_coin_gains_usdt[coin]
                g_pct = per_coin_gains_pct[coin]
                count_coin = length(g_usdt)
                count_coin == 0 && continue
                avg_gain_pct = mean(g_pct)
                total_gain_usdt = sum(g_usdt)
                @printf("    %-8s %7d %12.3f %16.4f\n", coin, count_coin, avg_gain_pct, total_gain_usdt)
                append!(all_gain_pcts, g_pct)
                append!(all_gain_usdt, g_usdt)
            end

            total_count = length(all_gain_usdt)
            if total_count > 0
                @printf("  Gain metrics total       : count=%d, avg gain=%.3f %%, total gain=%.4f USDT\n",
                    total_count, mean(all_gain_pcts), sum(all_gain_usdt))
            end
        end
    end

    println("=" ^ 60)
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

EnvConfig.init(test)  # test mode → cryptoxchsim, no live credentials needed
EnvConfig.setpairquote!(QUOTE_COIN)
classifier = try
    loadtrendclassifier(CONFIG; model_folder=MODEL_FOLDER)
catch err
    println(stderr, "$(EnvConfig.now()): ERROR failed to load configured classifier: $(sprint(showerror, err))")
    println(stderr, "$(EnvConfig.now()): tradesim aborted")
    exit(1)
end
EnvConfig.setdebugpath(LOG_SUBFOLDER)

Xch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 3

println("$(EnvConfig.now()): starting tradesim with config=$CONFIG_NAME")
println("$(EnvConfig.now()): backtest $BACKTEST_STARTDT → $BACKTEST_ENDDT")

effective_startdt, effective_enddt = backtest_bounds_from_env(BACKTEST_STARTDT, BACKTEST_ENDDT)

whitelist = normalize_whitelist(whitelist_from_env(WHITELIST_INPUT), QUOTE_COIN)
run_startdt, run_enddt = effective_startdt, effective_enddt

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

xc = Xch.XchCache(;
    startdt  = run_startdt,
    enddt    = run_enddt,
    exchange = EXCHANGE,
)

cache = Trade.TradeCache(xc=xc, cl=classifier, trademode=TRADE_MODE)
seed_quote_balance!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)
ensure_quote_budget!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)

# Apply the selected TrendDetector strategy parameters.
Trade.apply_tradingstrategy!(cache, CONFIG_STRATEGY;
    strategy_engine=:getgainsalgo,
    source="tradesim:$CONFIG_NAME")

# Override whitelist and risk parameters.
cache.mc[:whitelistcoins]   = whitelist
cache.mc[:maxassetfraction] = MAX_ASSET_FRACTION
cache.mc[:usenewtrade]      = false
cache.mc[:audit_portfolio_snapshot_mode] = :session_start

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG_NAME, engine=getgainsalgo, openthreshold=$BUY_OPEN_THRESHOLD")
println("$(EnvConfig.now()): usenewtrade=$(cache.mc[:usenewtrade])")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN, initial balance=$INITIAL_QUOTE_BALANCE")
println("$(EnvConfig.now()): whitelist ($(length(whitelist)) bases): $whitelist")
println("$(EnvConfig.now()): running backtest over $run_startdt → $run_enddt")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

Trade.run_backtest!(cache)

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

backtest_report(cache, run_startdt, run_enddt)

# Persist order log separately from other simulation artifacts.
EnvConfig.setdebugpath(ORDERS_SUBFOLDER)
Xch.writeorders(cache.xc)
println("$(EnvConfig.now()): saved simulation orders to $(EnvConfig.logfolder())")
