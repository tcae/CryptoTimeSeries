"""
tradesim.jl — Backtest simulation script using GainSegment config 046,
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

# Trade mode during backtest: Trade.buysell, Trade.closeonly, Trade.notrade.
const TRADE_MODE = Trade.buysell

# Whitelist of base coins to consider for trading during the simulation.
const QUOTE_COIN = "USDT"
const WHITELIST_INPUT = String[]
# const WHITELIST_INPUT = ["BTC", "ETH", "HBAR", "PEPE", "XRP"]

function whitelist_from_env(default::Vector{String})::Vector{String}
    raw = get(ENV, "TRADESIM_WHITELIST", "")
    isempty(strip(raw)) && return default
    vals = [strip(tok) for tok in split(raw, ',') if !isempty(strip(tok))]
    return isempty(vals) ? default : vals
end

# Initial quote-asset balance used in simulation mode (bybitsim).
const INITIAL_QUOTE_BALANCE = 100000.0

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# Optional cap for overall budget considered by trade sizing.
# If set, sizing uses min(real portfolio quote value, MAX_BUDGET_QUOTE).
const MAX_BUDGET_QUOTE = 10000  # nothing

# Buy signal score threshold used by GainSegment strategy.
const BUY_OPEN_THRESHOLD = 0.4f0

# GainSegment strategy parameters used by the backtest.
const CONFIG046_STRATEGY = tradingstrategy03()  # GainSegment(maxwindow=240, algorithm=gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015)
CONFIG046_STRATEGY.openthreshold = BUY_OPEN_THRESHOLD
const CONFIG046_NAME = "046"
const MODEL046_FOLDER = "Trend-046-training"

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradesim-" * CONFIG046_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
const ORDERS_SUBFOLDER = joinpath(LOG_SUBFOLDER, "orders")

function backtest_bounds_from_env(default_start::DateTime, default_end::DateTime)
    sraw = strip(get(ENV, "TRADESIM_STARTDT", ""))
    eraw = strip(get(ENV, "TRADESIM_ENDDT", ""))
    sdt = isempty(sraw) ? default_start : DateTime(sraw)
    edt = isempty(eraw) ? default_end : DateTime(eraw)
    @assert sdt <= edt "TRADESIM_STARTDT must be <= TRADESIM_ENDDT; got start=$(sdt), end=$(edt)"
    return sdt, edt
end

function max_budget_from_env(default_budget::Union{Nothing, Real})::Union{Nothing, Float64}
    raw = strip(get(ENV, "TRADESIM_MAX_BUDGET_QUOTE", ""))
    if isempty(raw)
        # backward compatibility for previous env name
        raw = strip(get(ENV, "TRADESIM_MAX_BUDGET_USDT", ""))
    end
    if isempty(raw)
        return isnothing(default_budget) ? nothing : Float64(default_budget)
    end
    budget = parse(Float64, raw)
    @assert budget > 0.0 "TRADESIM_MAX_BUDGET_QUOTE must be > 0; got $(budget)"
    return budget
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
function seed_quote_balance!(xc::CryptoXch.XchCache, quote_coin::AbstractString, amount::Real)
    isnothing(xc.bc) && error("cannot seed quote balance: exchange cache is not initialized")
    routed = CryptoXch._routedbc(xc, CryptoXch.trade_exchange_spot)
    if isnothing(routed)
        error("cannot seed quote balance: routed trade backend is not initialized")
    end
    if routed !== xc.bc
        error("cannot seed quote balance: routed trade backend differs from primary cache (routed=$(typeof(routed)), primary=$(typeof(xc.bc))). Fix routing/backend wiring before running tradesim.")
    end

    if applicable(Bybit.seedportfolio!, xc.bc, quote_coin, amount)
        Bybit.seedportfolio!(xc.bc, quote_coin, amount)
        return nothing
    end
    error("cannot seed quote balance for backend cache type=$(typeof(xc.bc))")
end

"Ensure the simulation starts with at least `minimum_free` quote balance."
function ensure_quote_budget!(xc::CryptoXch.XchCache, quote_coin::AbstractString, minimum_free::Real)
    q = uppercase(String(quote_coin))
    balancesdf = CryptoXch.balances(xc, ignoresmallvolume=false)
    qix = size(balancesdf, 1) > 0 ? findfirst(==(q), uppercase.(String.(balancesdf[!, :coin]))) : nothing
    current_free = isnothing(qix) ? 0.0 : Float64(balancesdf[qix, :free])
    if current_free + 1e-6 < Float64(minimum_free)
        seed_quote_balance!(xc, q, minimum_free)
        balancesdf = CryptoXch.balances(xc, ignoresmallvolume=false)
        qix = size(balancesdf, 1) > 0 ? findfirst(==(q), uppercase.(String.(balancesdf[!, :coin]))) : nothing
        reseeded_free = isnothing(qix) ? 0.0 : Float64(balancesdf[qix, :free])
        @assert reseeded_free + 1e-6 >= Float64(minimum_free) "totalusdt seed $(q) budget is insufficient after reseed; expected >= $(minimum_free), got $(reseeded_free)"
        println("$(EnvConfig.now()): reseeded $(q) free balance from $(round(current_free, digits=2)) to $(round(reseeded_free, digits=2))")
    else
        println("$(EnvConfig.now()): confirmed $(q) free seed budget $(round(current_free, digits=2))")
    end
end

"Load Trend 046 classifier artifacts and return a runtime classifier instance."
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

"Write the textual backtest report to `filepath` and return the report text."
function write_backtest_report_file(cache::Trade.TradeCache, startdt::DateTime, enddt::DateTime, filepath::AbstractString)::String
    open(filepath, "w") do io
        redirect_stdout(io) do
            backtest_report(cache, startdt, enddt)
        end
    end
    report_text = read(filepath, String)
    return report_text
end

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

EnvConfig.init(test)  # test mode -> bybitsim simulation, no live credentials needed
EnvConfig.cryptoquote = QUOTE_COIN
classifier = try
    loadtrend046classifier(MODEL046_FOLDER)
catch err
    println(stderr, "$(EnvConfig.now()): ERROR failed to load configured classifier: $(sprint(showerror, err))")
    println(stderr, "$(EnvConfig.now()): tradesim aborted")
    exit(1)
end
EnvConfig.setdebugpath(LOG_SUBFOLDER)
run_debug_folder = EnvConfig.logfolder()
report_file = joinpath(run_debug_folder, "backtest-report.txt")

CryptoXch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 3

println("$(EnvConfig.now()): starting tradesim with config=$CONFIG046_NAME")
println("$(EnvConfig.now()): backtest $BACKTEST_STARTDT → $BACKTEST_ENDDT")

effective_startdt, effective_enddt = backtest_bounds_from_env(BACKTEST_STARTDT, BACKTEST_ENDDT)
run_max_budget_quote = max_budget_from_env(MAX_BUDGET_QUOTE)

whitelist = normalize_whitelist(whitelist_from_env(WHITELIST_INPUT), QUOTE_COIN)
has_whitelist_override = !isempty(whitelist)
run_startdt, run_enddt = effective_startdt, effective_enddt

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

xc = CryptoXch.XchCache(;
    startdt  = run_startdt,
    enddt    = run_enddt,
    exchange = EXCHANGE,
)

cache = Trade.TradeCache(xc=xc, cl=classifier, trademode=TRADE_MODE)
seed_quote_balance!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)
ensure_quote_budget!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)

# Apply config 046 strategy parameters.
Trade.apply_tradingstrategy!(cache, CONFIG046_STRATEGY;
    strategy_engine=:getgainsalgo,
    source="tradesim:$CONFIG046_NAME")

# Override whitelist and risk parameters.
if has_whitelist_override
    cache.mc[:whitelistcoins] = whitelist
end
cache.mc[:maxassetfraction] = MAX_ASSET_FRACTION
cache.mc[:maxbudgetquote] = run_max_budget_quote
cache.mc[:usenewtrade]      = false
cache.mc[:audit_portfolio_snapshot_mode] = :session_start

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG046_NAME, engine=getgainsalgo, openthreshold=$BUY_OPEN_THRESHOLD")
println("$(EnvConfig.now()): usenewtrade=$(cache.mc[:usenewtrade])")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN, initial balance=$INITIAL_QUOTE_BALANCE")
println("$(EnvConfig.now()): max budget cap quote=$(isnothing(run_max_budget_quote) ? "none" : run_max_budget_quote)")
if has_whitelist_override
    println("$(EnvConfig.now()): whitelist override ($(length(whitelist)) bases): $whitelist")
else
    println("$(EnvConfig.now()): whitelist override disabled; using TradeCache default universe ($(length(cache.mc[:whitelistcoins])) bases)")
end
println("$(EnvConfig.now()): running backtest over $run_startdt → $run_enddt")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

local report_written = false
try
    Trade.run_backtest!(cache; skip_init=true)

    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE REPORT
    # ─────────────────────────────────────────────────────────────────────────

    backtest_report(cache, run_startdt, run_enddt)
    write_backtest_report_file(cache, run_startdt, run_enddt, report_file)
    report_written = true
    println("$(EnvConfig.now()): saved backtest report to $report_file")
finally
    if !report_written
        try
            write_backtest_report_file(cache, run_startdt, run_enddt, report_file)
            println("$(EnvConfig.now()): saved partial backtest report to $report_file")
        catch err
            println(stderr, "$(EnvConfig.now()): failed to persist backtest report: $(sprint(showerror, err))")
        end
    end

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
        println(stderr, "$(EnvConfig.now()): failed to write private call summary: $(sprint(showerror, err))")
    end

    # Persist order log separately from other simulation artifacts.
    EnvConfig.setdebugpath(ORDERS_SUBFOLDER)
    CryptoXch.writeorders(cache.xc)
    println("$(EnvConfig.now()): saved simulation orders to $(EnvConfig.logfolder())")
end
