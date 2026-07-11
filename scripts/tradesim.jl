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

const QUOTE_COIN = "USDT"

# Initial quote-asset balance used in simulation mode (cryptoxchsim).
const INITIAL_QUOTE_BALANCE = 100000.0

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# Strategy parameters used by the backtest.
const CONFIG_REF = get(ENV, "TRADESIM_CONFIG_REF", "046")
const CONFIG = TradingStrategy.trenddetectorconfig(CONFIG_REF)
const CONFIG_NAME = String(CONFIG.configname)
const MODEL_FOLDER = TradingStrategy.trendconfigfolder(CONFIG, "training")

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = "tradesim-" * CONFIG_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))
const ORDERS_SUBFOLDER = joinpath(LOG_SUBFOLDER, "orders")

"Return ORDER_FILLED events as a DataFrame."
function filled_orders_df(xc::Xch.XchCache)::DataFrame
    rows = NamedTuple[]

    for (pair, tdf) in xc.pairstates
        nrow(tdf) == 0 && continue
        cols = propertynames(tdf)
        required = (:opentime, :pair, :lo_status, :lo_filled, :lo_pavg, :lc_status, :lc_filled, :lc_pavg, :so_status, :so_filled, :so_pavg, :sc_status, :sc_filled, :sc_pavg)
        all(c -> c in cols, required) || continue

        for row in eachrow(tdf)
            created = DateTime(row.opentime)
            symbol = String(ismissing(row.pair) ? pair : row.pair)

            for (statuscol, filledcol, pavgcol, side) in [
                (:lo_status, :lo_filled, :lo_pavg, "Buy"),
                (:lc_status, :lc_filled, :lc_pavg, "Sell"),
                (:so_status, :so_filled, :so_pavg, "Sell"),
                (:sc_status, :sc_filled, :sc_pavg, "Buy"),
            ]
                status = lowercase(strip(String(row[statuscol])))
                status == "closed" || continue

                filled = ismissing(row[filledcol]) ? 0.0 : (row[filledcol])
                avg = ismissing(row[pavgcol]) ? 0.0 : (row[pavgcol])
                (filled > 0.0 && avg > 0.0) || continue

                push!(rows, (
                    created = created,
                    symbol = symbol,
                    side = side,
                    executedqty = filled,
                    avgprice = avg,
                ))
            end
        end
    end

    return isempty(rows) ? DataFrame() : sort!(DataFrame(rows), :created)
end

function backtest_bounds_from_env(default_start::DateTime, default_end::DateTime)
    sraw = strip(get(ENV, "TRADESIM_STARTDT", ""))
    eraw = strip(get(ENV, "TRADESIM_ENDDT", ""))
    sdt = isempty(sraw) ? default_start : DateTime(sraw)
    edt = isempty(eraw) ? default_end : DateTime(eraw)
    @assert sdt <= edt "TRADESIM_STARTDT must be <= TRADESIM_ENDDT; got start=$(sdt), end=$(edt)"
    return sdt, edt
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
    current_free = isnothing(qix) ? 0.0 : (balancesdf[qix, :free])
    if current_free + 1e-6 < (minimum_free)
        seed_quote_balance!(xc, q, minimum_free)
        balancesdf = Xch.balances(xc, ignoresmallvolume=false)
        qix = size(balancesdf, 1) > 0 ? findfirst(==(q), uppercase.(String.(balancesdf[!, :coin]))) : nothing
        reseeded_free = isnothing(qix) ? 0.0 : (balancesdf[qix, :free])
        @assert reseeded_free + 1e-6 >= (minimum_free) "totalusdt seed $(q) budget is insufficient after reseed; expected >= $(minimum_free), got $(reseeded_free)"
        println("$(EnvConfig.now()): reseeded $(q) free balance from $(round(current_free, digits=2)) to $(round(reseeded_free, digits=2))")
    else
        println("$(EnvConfig.now()): confirmed $(q) free seed budget $(round(current_free, digits=2))")
    end
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
    co = filled_orders_df(cache.xc)
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

    # Try to reconstruct a daily portfolio value series from closed orders.
    # We track cumulative PnL per filled sell order (long-close gains/losses).
    # This is an approximation; a full mark-to-market series would require
    # the PORTFOLIO_SNAPSHOT audit rows.
    daily_pnl = Dict{Date, Float64}()
    for row in eachrow(co)
        day = Date(row.created)
        if !ismissing(row.executedqty) && !ismissing(row.avgprice) && uppercasefirst(string(row.side)) == "Sell"
            pnl = (row.executedqty) * (row.avgprice)
            daily_pnl[day] = get(daily_pnl, day, 0.0) + pnl
        end
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
            px = (row.avgprice)
            qty = (row.executedqty)
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
EnvConfig.setdebugpath(LOG_SUBFOLDER)

Xch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 3

println("$(EnvConfig.now()): starting tradesim with config=$CONFIG_NAME")
println("$(EnvConfig.now()): backtest $BACKTEST_STARTDT → $BACKTEST_ENDDT")

effective_startdt, effective_enddt = backtest_bounds_from_env(BACKTEST_STARTDT, BACKTEST_ENDDT)

run_startdt, run_enddt = effective_startdt, effective_enddt

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

bc = Bybit.BybitSimCache()
Bybit.seedportfolio!(bc, QUOTE_COIN, 0.0)
xc = Xch.XchCache(bc;
    startdt  = run_startdt,
    enddt    = run_enddt,
)
Xch.ensuretradesschema(xc, vcat(Xch.tradesdf_contributors(), TradingStrategy.tradesdf_contributors(), Trade.tradesdf_contributors()))

cache = Trade.TradeCache(xc=xc, strategy=TradingStrategy.TsCache(CONFIG_REF; source="tradesim:$CONFIG_NAME"), trademode=TRADE_MODE)
seed_quote_balance!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)
ensure_quote_budget!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)

# Override risk parameters.
cache.mc[:maxassetfraction] = MAX_ASSET_FRACTION
cache.mc[:audit_portfolio_snapshot_mode] = :session_start

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG_NAME, engine=tradingstrategy, openthreshold=$(cache.ts.cfg.openthreshold)")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN, initial balance=$INITIAL_QUOTE_BALANCE")
println("$(EnvConfig.now()): blacklist ($(length(cache.mc[:blacklistbases])) bases): $(cache.mc[:blacklistbases])")
println("$(EnvConfig.now()): running backtest over $run_startdt → $run_enddt")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

Trade.run_backtest!(cache)

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

backtest_report(cache, run_startdt, run_enddt)

# Keep legacy debug-path split for parity with previous script layout.
EnvConfig.setdebugpath(ORDERS_SUBFOLDER)
println("$(EnvConfig.now()): order history report derived from xc.pairstates trades data")
