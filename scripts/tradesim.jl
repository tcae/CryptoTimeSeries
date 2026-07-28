"""
tradesim.jl — Backtest simulation script using a selected TrendDetector config,
followed by a performance report.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before running.

Usage:
    julia --project=scripts scripts/tradesim.jl
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."), io=devnull)

using Dates, Statistics, Printf, Logging
using DataFrames
using EnvConfig, TradingStrategy, Trade, Classify, Xch, Bybit, Ohlcv, Features, Targets, TSM

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these values before running
# ─────────────────────────────────────────────────────────────────────────────

# Exchange used for the simulation exchange backend. BybitSim keeps the exchange
# explicit while still allowing the common trading code path to run.
const EXCHANGE = Xch.EXCHANGE_BYBITSIM

# Backtest time range (UTC).
const BACKTEST_STARTDT = DateTime("2025-07-01T04:01:00")
const BACKTEST_ENDDT   = DateTime("2025-07-02T03:59:00")

function env_bases(default_bases::Vector{String})::Vector{String}
    raw = strip(get(ENV, "TRADESIM_BASES", ""))
    if isempty(raw)
        return default_bases
    end
    bases = [uppercase(strip(token)) for token in split(raw, ",") if !isempty(strip(token))]
    @assert !isempty(bases) "TRADESIM_BASES must contain at least one base symbol when provided"
    return unique(bases)
end

const BACKTEST_BASES = env_bases(["SINE"])

# Trade mode during backtest: Trade.buysell, Trade.closeonly, Trade.notrade.
const TRADE_MODE = Trade.buysell

const QUOTE_COIN = "USDT"

# Initial quote-asset balance used in simulation mode (cryptoxchsim).
const INITIAL_QUOTE_BALANCE = 1000.0

# Maximum budget in quote coin allocated in total
const MAX_BUDGET_QUOTE = 500f0

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# Strategy parameters used by the backtest.
const CONFIG_REF = get(ENV, "TRADESIM_CONFIG_REF", "046")
const CLFOLDER = get(ENV, "TRADESIM_CLFOLDER", "test") # training
const CONFIG = TradingStrategy.trenddetectorconfig(CONFIG_REF)
const CONFIG_NAME = String(CONFIG.configname)
const MODEL_FOLDER = TradingStrategy.trendconfigfolder(CONFIG, CLFOLDER)

# Replay source folder containing classifier artifacts and prediction outputs.
const REPLAY_SOURCE_SUBFOLDER = begin
    raw = strip(get(ENV, "TRADESIM_REPLAY_SOURCE", "Trend-046-test"))
    isempty(raw) ? "Trend-046-test" : raw
end

# Log subfolder under EnvConfig.logfolder().
const LOG_SUBFOLDER = begin
    raw = strip(get(ENV, "TRADESIM_LOG_SUBFOLDER", ""))
    isempty(raw) ? ("tradesim-" * CONFIG_NAME ) : raw
    # isempty(raw) ? ("tradesim-" * CONFIG_NAME * "-" * Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS"))) : raw
end

"Return ORDER_FILLED events as a DataFrame."
function filled_orders_df(xc::Xch.XchCache)::DataFrame
    rows = NamedTuple[]

    for (pair, tdf) in xc.tsm.pairstates
        nrow(tdf) == 0 && continue
        cols = propertynames(tdf)
        required = (:opentime, :pair, :lo_status, :lol_filled, :lol_pavg, :lc_status, :lcl_filled, :lcl_pavg, :so_status, :sol_filled, :sol_pavg, :sc_status, :scl_filled, :scl_pavg)
        all(c -> c in cols, required) || continue

        for row in eachrow(tdf)
            created = DateTime(row.opentime)
            symbol = String(ismissing(row.pair) ? pair : row.pair)

            for (statuscol, filledcol, pavgcol, side) in [
                (:lo_status, :lol_filled, :lol_pavg, "Buy"),
                (:lc_status, :lcl_filled, :lcl_pavg, "Sell"),
                (:so_status, :sol_filled, :sol_pavg, "Sell"),
                (:sc_status, :scl_filled, :scl_pavg, "Buy"),
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

"Load one replay source dataframe from a configured log subfolder and subdirectory."
function _load_replay_df(logsubfolder::AbstractString, subdir::AbstractString, stem::AbstractString)::DataFrame
    logsroot = dirname(EnvConfig.logfolder())
    folderpath = joinpath(logsroot, String(logsubfolder), String(subdir))
    df = EnvConfig.readdf(String(stem); folderpath=folderpath)
    @assert !isnothing(df) "missing replay source $(subdir)/$(stem).arrow in $(folderpath)"
    return DataFrame(df)
end

"Build replay trades input from result and prediction artifacts."
function _build_replay_input(resultsdf::DataFrame, preddf::DataFrame, quotecoin::AbstractString)::DataFrame
    required_result = (:opentime, :coin, :set, :rangeid, :high, :low, :close)
    for c in required_result
        @assert c in propertynames(resultsdf) "results/all must contain column $(c); names=$(names(resultsdf))"
    end
    required_pred = (:label, :score)
    for c in required_pred
        @assert c in propertynames(preddf) "predictions/maxpredictions must contain column $(c); names=$(names(preddf))"
    end
    @assert nrow(resultsdf) == nrow(preddf) "results and predictions row mismatch: results=$(nrow(resultsdf)) predictions=$(nrow(preddf))"

    q = uppercase(String(quotecoin))
    paircol = [uppercase(String(resultsdf[ix, :coin])) * q for ix in 1:nrow(resultsdf)]
    setcol = [ismissing(resultsdf[ix, :set]) ? TSM.TSM_NO_SET : String(resultsdf[ix, :set]) for ix in 1:nrow(resultsdf)]

    replaydf = DataFrame(
        opentime=DateTime.(resultsdf[!, :opentime]),
        pair=paircol,
        set=setcol,
        rangeid=Int32.(resultsdf[!, :rangeid]),
        high=Float32.(resultsdf[!, :high]),
        low=Float32.(resultsdf[!, :low]),
        close=Float32.(resultsdf[!, :close]),
        label=Targets.tradelabel.(String.(preddf[!, :label])),
        score=Float32.(preddf[!, :score]),
    )
    return replaydf
end

"Validate replay sequence: same pair/set/rangeid groups must have strictly increasing opentime."
function _validate_replay_sequence!(replaydf::DataFrame)
    required = (:opentime, :pair, :set, :rangeid, :high, :low, :close, :label, :score)
    for c in required
        @assert c in propertynames(replaydf) "replaydf missing column $(c); names=$(names(replaydf))"
    end

    for g in groupby(replaydf, [:pair, :set, :rangeid])
        nrow(g) == 0 && continue
        prev = g[1, :opentime]
        for ix in 2:nrow(g)
            cur = g[ix, :opentime]
            @assert prev < cur "invalid replay sequence for pair=$(g[ix, :pair]) set=$(g[ix, :set]) rangeid=$(g[ix, :rangeid]); expected opentime[ix-1] < opentime[ix], got $(prev) and $(cur)"
            prev = cur
        end
    end

    # Enforce one row per replay key tuple.
    keydf = combine(groupby(replaydf, [:pair, :set, :rangeid, :opentime]), nrow => :count)
    dup = keydf[keydf.count .> 1, :]
    @assert nrow(dup) == 0 "duplicate replay rows detected for (pair,set,rangeid,opentime) keys; duplicates=$(nrow(dup))"
    return nothing
end

"Validate that tradeloop replay state is materialized in the resulting trades dataframe."
function _validate_tradesim_replay_result!(tradesdf::DataFrame)
    @assert nrow(tradesdf) > 0 "trades replay result is empty"

    limit_cols = [:lo_limit, :lc_limit, :so_limit, :sc_limit]
    amount_cols = [:lo_amount, :lc_amount, :so_amount, :sc_amount, :lp_amount, :sp_amount]
    status_cols = [:lo_status, :lc_status, :so_status, :sc_status]
    account_cols = [:equity, :freequote]
    required_cols = vcat(limit_cols, amount_cols, status_cols, account_cols)

    for col in required_cols
        @assert col in propertynames(tradesdf) "missing replay result column $(col); names=$(names(tradesdf))"
    end

    for col in vcat(limit_cols, amount_cols, account_cols)
        values = tradesdf[!, col]
        @assert !any(ismissing, values) "column $(col) contains missing values"
        @assert all(x -> isfinite((x)), values) "column $(col) contains non-finite values"
    end

    for col in account_cols
        @assert all(x -> (x) >= 0f0, tradesdf[!, col]) "column $(col) contains negative values"
    end

    for col in status_cols
        values = String.(tradesdf[!, col])
        @assert !any(ismissing, values) "status column $(col) contains missing values"
    end

    has_limit_activity = any(col -> any(tradesdf[!, col] .!= 0f0), limit_cols)
    has_amount_activity = any(col -> any(tradesdf[!, col] .!= 0f0), amount_cols)
    has_status_activity = any(col -> any(lowercase.(String.(tradesdf[!, col])) .!= "none"), status_cols)

    @assert has_limit_activity "replay result has no non-zero limits"
    @assert has_amount_activity "replay result has no non-zero amounts"
    @assert has_status_activity "replay result has no status activity"
    return nothing
end

"Append or reuse one replay row in the pair dataframe."
function _upsert_replay_row!(tsm::TSM.TsmCache, pairdf::DataFrame, pair::AbstractString, quotecoin::AbstractString, row)::Integer
    row_opentime = DateTime(row.opentime)
    if nrow(pairdf) > 0
        last_opentime = pairdf[nrow(pairdf), :opentime]
        if last_opentime == row_opentime
            return nrow(pairdf)
        end
        @assert last_opentime < row_opentime "non-increasing replay opentime for pair=$(pair): last=$(last_opentime), new=$(row_opentime)"
    end

    bq = Xch.basequote(String(pair))
    rowref = TSM.ensuretradesrow!(tsm, String(bq.basecoin), String(quotecoin), row_opentime)
    return Int(rowref.rowix)
end

function _strategy_with_algorithm(spec::TradingStrategy.StrategyConfig, algorithm::Function)::TradingStrategy.StrategyConfig
    return TradingStrategy.StrategyConfig(
        classifier=spec.classifier,
        algorithm=algorithm,
        maxwindow=spec.maxwindow,
        openthreshold=spec.openthreshold,
        closethreshold=spec.closethreshold,
        makerfee=spec.makerfee,
        takerfee=spec.takerfee,
        buygain=spec.buygain,
        sellgain=spec.sellgain,
        limitreduction=spec.limitreduction,
        minpricedelta=spec.minpricedelta,
        max_classify_staleness_minutes=spec.max_classify_staleness_minutes,
    )
end

"Reset mutable runtime state before running one independent replay group."
function _reset_replay_runtime!(cache::Trade.TradeCache, quote_coin::AbstractString, initial_quote_balance::Real)
    cache.xc.tsm = TSM.TsmCache()
    TSM.ensuretradesschema!(cache.xc.tsm, TSM.tradesdf_all_contributors())

    if cache.xc.bc isa Bybit.BybitCache
        bc = cache.xc.bc
        bc.assets = nothing
        bc.orders = nothing
        bc.closedorders = nothing
        Bybit._init_simulation!(bc)
        Bybit.seedportfolio!(bc, quote_coin, initial_quote_balance)
    else
        error("replay runtime reset currently supports BybitCache only, got $(typeof(cache.xc.bc))")
    end
    return nothing
end

"Build one independent replay group into Xch-owned Trades state."
function _prepare_replay_group!(cache::Trade.TradeCache, groupdf::DataFrame, quotecoin::AbstractString)
    pair = uppercase(String(groupdf[1, :pair]))
    setname = String(groupdf[1, :set])
    rangeid = Int32(groupdf[1, :rangeid])
    bq = Xch.basequote(pair)
    base = uppercase(String(bq.basecoin))

    cache.xc.startdt = DateTime(groupdf[1, :opentime])
    cache.xc.enddt = DateTime(groupdf[nrow(groupdf), :opentime])
    cache.xc.currentdt = nothing
    cache.mc[:reloadtimes] = Time[]

    cache.cfg = DataFrame(
        basecoin=[base],
        pair=[pair],
        quotevolume24h_M=Float32[0f0],
        pricechangepercent=Float32[0f0],
        lastprice=Float32[groupdf[nrow(groupdf), :close]],
        datetime=DateTime[cache.xc.startdt],
        minquotevol=Bool[true],
        continuousminvol=Bool[true],
        inportfolio=Bool[true],
        classifieraccepted=Bool[true],
        openenabled=Bool[true],
        closeenabled=Bool[true],
        blacklisted=Bool[false],
    )

    seeddf = select(groupdf, :opentime, :pair, :set, :rangeid, :high, :low, :close, :label, :score)
    TSM.settrades!(cache.xc.tsm, pair, seeddf)

    rowmap = Dict{DateTime, NamedTuple}()
    for row in eachrow(groupdf)
        rowmap[DateTime(row.opentime)] = (
            set=String(setname),
            rangeid=Int32(rangeid),
            high=Float32(row.high),
            low=Float32(row.low),
            close=Float32(row.close),
            label=row.label,
            score=Float32(row.score),
        )
    end
    return (pair=pair, base=base, setname=setname, rangeid=rangeid, rowmap=rowmap)
end

"Run one replay group through Trade tradeloop step execution."
function _run_replay_group_tradeloop!(cache::Trade.TradeCache, groupdf::DataFrame, quotecoin::AbstractString)
    _reset_replay_runtime!(cache, quotecoin, INITIAL_QUOTE_BALANCE)
    prep = _prepare_replay_group!(cache, groupdf, quotecoin)

    base_algorithm = TradingStrategy.gain_limit_reversal!
    replay_algorithm = function (cfg::TradingStrategy.StrategyConfig, tradesdf::DataFrame, ix::Integer)
        dt = DateTime(tradesdf[ix, :opentime])
        src = get(prep.rowmap, dt, nothing)
        if isnothing(src)
            return base_algorithm(cfg, tradesdf, ix)
        end
        TSM.settrades_set!(tradesdf, ix, src.set)
        TSM.settrades_rangeid!(tradesdf, ix, src.rangeid)
        TSM.settrades_high!(tradesdf, ix, src.high)
        TSM.settrades_low!(tradesdf, ix, src.low)
        TSM.settrades_close!(tradesdf, ix, src.close)
        TSM.settrades_label!(tradesdf, ix, src.label)
        TSM.settrades_score!(tradesdf, ix, src.score)
        return base_algorithm(cfg, tradesdf, ix)
    end
    cache.ts.cfg = _strategy_with_algorithm(cache.ts.cfg, replay_algorithm)

    for dt in groupdf[!, :opentime]
        cache.xc.currentdt = DateTime(dt)
        Trade._tradestep!(cache)
    end

    tdf = DataFrame(TSM.trades(cache.xc.tsm, prep.pair))
    fills = filled_orders_df(cache.xc)
    if nrow(fills) > 0
        fills[!, :pair] = fill(prep.pair, nrow(fills))
        fills[!, :set] = fill(prep.setname, nrow(fills))
        fills[!, :rangeid] = fill(prep.rangeid, nrow(fills))
    end
    return tdf, fills
end

"Run artifact-driven trades replay through Trade tradeloop and return (tradesdf, fillsdf)."
function run_replay_from_artifacts!(cache::Trade.TradeCache; logsubfolder::AbstractString=REPLAY_SOURCE_SUBFOLDER, quotecoin::AbstractString=QUOTE_COIN)
    resultsdf = _load_replay_df(logsubfolder, "results", "all")
    preddf = _load_replay_df(logsubfolder, "predictions", "maxpredictions")
    replaydf = _build_replay_input(resultsdf, preddf, quotecoin)
    sort!(replaydf, [:pair, :set, :rangeid, :opentime])
    _validate_replay_sequence!(replaydf)

    trades_parts = DataFrame[]
    fills_parts = DataFrame[]
    groups = groupby(replaydf, [:pair, :set, :rangeid])
    for g in groups
        tradesdf, fillsdf = _run_replay_group_tradeloop!(cache, DataFrame(g), quotecoin)
        push!(trades_parts, DataFrame(tradesdf))
        nrow(fillsdf) > 0 && push!(fills_parts, DataFrame(fillsdf))
    end

    alltrades = isempty(trades_parts) ? DataFrame() : reduce(vcat, trades_parts; cols=:union)
    allfills = isempty(fills_parts) ? DataFrame() : reduce(vcat, fills_parts; cols=:union)
    return alltrades, allfills
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
EnvConfig.setlogpath(LOG_SUBFOLDER)

Xch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 3

println("$(EnvConfig.now()): starting tradesim with config=$CONFIG_NAME")
println("$(EnvConfig.now()): backtest $BACKTEST_STARTDT → $BACKTEST_ENDDT")

println("$(EnvConfig.now()): replay source folder=$REPLAY_SOURCE_SUBFOLDER")

effective_startdt, effective_enddt = backtest_bounds_from_env(BACKTEST_STARTDT, BACKTEST_ENDDT)

run_startdt, run_enddt = effective_startdt, effective_enddt
strategy_runtime = TradingStrategy.TsCache(CONFIG_REF; source="tradesim:$CONFIG_NAME")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

bc = Bybit.BybitCache()
Bybit.seedportfolio!(bc, QUOTE_COIN, 0.0)
xc = Xch.XchCache(bc;
    startdt  = run_startdt,
    enddt    = run_enddt,
)
    TSM.ensuretradesschema!(xc.tsm, TSM.tradesdf_all_contributors())

cache = Trade.TradeCache(xc=xc, strategy=strategy_runtime, trademode=TRADE_MODE)
seed_quote_balance!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)
ensure_quote_budget!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)

# Override risk parameters.
cache.mc[:maxassetfraction] = MAX_ASSET_FRACTION
cache.mc[:maxbudgetquote]   = MAX_BUDGET_QUOTE

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG_NAME, engine=tradingstrategy, openthreshold=$(cache.ts.cfg.openthreshold)")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN, initial balance=$INITIAL_QUOTE_BALANCE")
println("$(EnvConfig.now()): blacklist ($(length(cache.mc[:blacklistbases])) bases): $(cache.mc[:blacklistbases])")
println("$(EnvConfig.now()): running backtest over $run_startdt → $run_enddt")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

alltrades, allfills = run_replay_from_artifacts!(cache; logsubfolder=REPLAY_SOURCE_SUBFOLDER, quotecoin=QUOTE_COIN)
_validate_tradesim_replay_result!(alltrades)
println("$(EnvConfig.now()): replay simulation finished, trades rows=$(nrow(alltrades)), fills rows=$(nrow(allfills))")

replay_out_folder = joinpath(EnvConfig.logfolder(), "tradesim-replay")
mkpath(replay_out_folder)
saved_trades = EnvConfig.savedf(alltrades, "trades-replay"; folderpath=replay_out_folder)
saved_fills = EnvConfig.savedf(allfills, "fills-replay"; folderpath=replay_out_folder)
println("$(EnvConfig.now()): saved replay trades to $saved_trades")
println("$(EnvConfig.now()): saved replay fills to $saved_fills")

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

if nrow(allfills) == 0
    println("$(EnvConfig.now()): replay produced no filled-order rows")
else
    println("$(EnvConfig.now()): replay filled-order summary: rows=$(nrow(allfills))")
end
EnvConfig.setlogpath(LOG_SUBFOLDER)
tradespath = TSM.savetradesdf(xc.tsm; stem="trades-ts", folderpath=EnvConfig.logfolder())
println("$(EnvConfig.now()): saved trades dataframe to $tradespath")

# Keep legacy log-path split for parity with previous script layout.
println("$(EnvConfig.now()): order history report derived from xc.tsm.pairstates trades data")
