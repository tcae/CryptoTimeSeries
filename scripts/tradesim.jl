"""
tradesim.jl — Backtest simulation script using a selected TrendDetector config,
followed by a performance report.

Configuration is defined in the CONFIG block below. Adjust the parameters
to your requirements before running.

Usage:
    julia --project=scripts scripts/tradesim.jl [help] [test|train] [config=<name>] [startdt=<DateTime>] [enddt=<DateTime>] [coins=<CSV>]
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."), io=devnull)

using Dates, Statistics, Printf, Logging
using DataFrames
using CategoricalArrays
using EnvConfig, TradingStrategy, Trade, Classify, Xch, Bybit, Ohlcv, Features, Targets, TSM

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these values before running
# ─────────────────────────────────────────────────────────────────────────────

# Exchange used for the simulation exchange backend. BybitSim keeps the exchange
# explicit while still allowing the common trading code path to run.
const EXCHANGE = Xch.EXCHANGE_BYBITSIM

# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENTS — help, test|train, config, startdt, enddt, coins
# ─────────────────────────────────────────────────────────────────────────────

function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)
    prefix = key * "="
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "="; limit=2)[2]
        end
    end
    return default
end

function _wants_help(args::Vector{String})::Bool
    for arg in args
        normalized = lowercase(strip(arg))
        if normalized in ("help", "--help", "-h")
            return true
        elseif startswith(normalized, "help=")
            value = split(normalized, "="; limit=2)[2]
            return value in ("1", "true", "yes", "on")
        end
    end
    return false
end

function tradesimhelp()::String
    return """
Usage:
    julia --project=scripts scripts/tradesim.jl [help] [test|train] [key=value ...]

Flag parameters:
  help, --help, -h
      Show this message and exit.
      Default: false

  test
      Use `EnvConfig.init(test)` and default replay source/classifier folder phase `test`.
      Default: true

  train
      Use `EnvConfig.init(training)` and default replay source/classifier folder phase `training`.
      Default: false

Key=value parameters:
  config=<configname>
      Trend preset from `TREND_DETECTOR_CONFIGS` in `TradingStrategy/src/tradingstrategyconfig.jl`.
      Default: `046`, or `TRADESIM_CONFIG_REF` env var when set

  startdt=<DateTime>
      Override backtest start datetime (ISO-8601 format).
      Example: `startdt=2025-07-01T07:00:00`
      Default: `TRADESIM_STARTDT` env var, or unset (use full replay source range)

  enddt=<DateTime>
      Override backtest end datetime (ISO-8601 format).
      Example: `enddt=2025-07-01T09:00:00`
      Default: `TRADESIM_ENDDT` env var, or unset (use full replay source range)

  coins=<CSV>
      Override backtest trading pair bases as a comma-separated list.
      Example: `coins=SINE,BTC,ETH`
      Default: `TRADESIM_BASES` env var, or `SINE`

  usepartitions=<bool>
      When true, replay runs one independent tradeloop per (pair, set, rangeid)
      group, resetting the portfolio between groups (legacy behaviour).
      When false (default), set/rangeid are ignored and all configured pairs are
      processed together minute by minute in a single continuous tradeloop,
      resembling the live tradereal loop.
      Default: `TRADESIM_USE_PARTITIONS` env var, or `false`
"""
end

if _wants_help(ARGS)
    println(tradesimhelp())
    exit(0)
end

const HAS_TEST = "test" in ARGS
const HAS_TRAIN = "train" in ARGS
@assert !(HAS_TEST && HAS_TRAIN) "mode flags are exclusive; use only one of test or train"
const TESTMODE = !HAS_TRAIN  # default true (test), matches previous hardcoded behavior

# Backtest time range (UTC).
const BACKTEST_STARTDT = begin
    raw = _argvalue(ARGS, "startdt", nothing)
    envraw = strip(get(ENV, "TRADESIM_STARTDT", ""))
    raw = isnothing(raw) ? (isempty(envraw) ? nothing : envraw) : raw
    isnothing(raw) ? nothing : DateTime(String(raw))
end
const BACKTEST_ENDDT = begin
    raw = _argvalue(ARGS, "enddt", nothing)
    envraw = strip(get(ENV, "TRADESIM_ENDDT", ""))
    raw = isnothing(raw) ? (isempty(envraw) ? nothing : envraw) : raw
    isnothing(raw) ? nothing : DateTime(String(raw))
end
@assert isnothing(BACKTEST_STARTDT) || isnothing(BACKTEST_ENDDT) || (BACKTEST_STARTDT <= BACKTEST_ENDDT) "startdt=$(BACKTEST_STARTDT) must be <= enddt=$(BACKTEST_ENDDT)"

function env_bases(args::Vector{String}, default_bases::Vector{String})::Vector{String}
    araw = _argvalue(args, "coins", nothing)
    raw = isnothing(araw) ? strip(get(ENV, "TRADESIM_BASES", "")) : araw
    if isempty(raw)
        return default_bases
    end
    bases = [uppercase(strip(token)) for token in split(raw, ",") if !isempty(strip(token))]
    @assert !isempty(bases) "coins/TRADESIM_BASES must contain at least one base symbol when provided"
    return unique(bases)
end

const BACKTEST_BASES = env_bases(ARGS, ["SINE"])

# When false (default), set/rangeid partitions are ignored and all pairs run together
# in one continuous tradeloop (resembles tradereal). When true, replay one independent
# tradeloop per (pair, set, rangeid) group with a portfolio reset between groups.
const USE_PARTITIONS = begin
    raw = _argvalue(ARGS, "usepartitions", get(ENV, "TRADESIM_USE_PARTITIONS", "false"))
    lowercase(strip(String(raw))) in ("1", "true", "yes", "on")
end

# Trade mode during backtest: Trade.buysell, Trade.closeonly, Trade.notrade.
const TRADE_MODE = Trade.buysell

const QUOTE_COIN = "USDT"

# Initial quote-asset balance used in simulation mode (cryptoxchsim).
const INITIAL_QUOTE_BALANCE = 1000.0

# Maximum budget in quote coin allocated in total
const MAX_BUDGET_QUOTE = 500f0

# Maximum fraction of total portfolio value allocated to a single asset.
const MAX_ASSET_FRACTION = 0.1f0

# Mandatory stop-loss distance from each open order's price, as a fraction (e.g. 0.05 = 5%).
const STOPLOSSPCT = 0.05f0

# Strategy parameters used by the backtest.
const CONFIG_REF = _argvalue(ARGS, "config", get(ENV, "TRADESIM_CONFIG_REF", "046"))
const CLFOLDER = TESTMODE ? "test" : "training"
const CONFIG = TradingStrategy.trenddetectorconfig(CONFIG_REF)
const CONFIG_NAME = String(CONFIG.configname)
const MODEL_FOLDER = TradingStrategy.trendconfigfolder(CONFIG, CLFOLDER)

# Replay source folder containing classifier artifacts and prediction outputs.
const REPLAY_SOURCE_SUBFOLDER = begin
    raw = strip(get(ENV, "TRADESIM_REPLAY_SOURCE", ""))
    isempty(raw) ? "Trend-$(CONFIG_NAME)-$(CLFOLDER)" : raw
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

function backtest_bounds_from_env(default_start::Union{Nothing, DateTime}, default_end::Union{Nothing, DateTime})
    # startdt/enddt are already resolved from CLI args and env vars into BACKTEST_STARTDT/BACKTEST_ENDDT.
    if !isnothing(default_start) && !isnothing(default_end)
        @assert default_start <= default_end "startdt=$(default_start) must be <= enddt=$(default_end)"
    end
    return default_start, default_end
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

"Convert categorical columns to plain strings to keep vcat stable across group pools."
function _stringify_categorical_columns!(df::DataFrame)::DataFrame
    for col in propertynames(df)
        values = df[!, col]
        if values isa CategoricalVector
            df[!, col] = string.(values)
        end
    end
    return df
end

"""
    tradescompare(trades_td, trades_replay)

Compare TrendDetector (`trades_td`) and tradesim replay (`trades_replay`) on
key identity `(set, pair, rangeid, opentime)` and report per-column equality.

Returns a named tuple with:
- `rowstats`: row and join cardinalities
- `equal_cols`: columns with exact value parity
- `unequal_counts`: per-column mismatch counts
- `label_transitions`: source→replay transition counts for mismatching labels
- `limit_mismatch_counts`: mismatch counts for `lo/lc/so/sc` limit columns
"""
function tradescompare(trades_td::DataFrame, trades_replay::DataFrame)
    _cellsequal(a, b) = begin
        if ismissing(a) || ismissing(b)
            return ismissing(a) && ismissing(b)
        elseif (a isa Number) && (b isa Number)
            return isequal(a, b)
        else
            return string(a) == string(b)
        end
    end

    key = [:set, :pair, :rangeid, :opentime]
    td_props = Set(propertynames(trades_td))
    replay_props = Set(propertynames(trades_replay))
    for c in key
        @assert c in td_props "trades_td missing key column $(c); names=$(names(trades_td))"
        @assert c in replay_props "trades_replay missing key column $(c); names=$(names(trades_replay))"
    end

    joined = innerjoin(trades_replay, trades_td; on=key, makeunique=true)
    rowstats = (
        replay_rows=nrow(trades_replay),
        td_rows=nrow(trades_td),
        joined_rows=nrow(joined),
    )

    common_cols = [c for c in propertynames(trades_replay) if c in td_props && !(c in key)]
    equal_cols = Symbol[]
    unequal_counts = Dict{Symbol, Int}()
    for c in common_cols
        leftcol = c
        rightcol = Symbol(string(c, "_1"))
        @assert rightcol in propertynames(joined) "joined comparison column missing for $(c)"
        leftv = joined[!, leftcol]
        rightv = joined[!, rightcol]
        ndiff = count(ix -> !_cellsequal(leftv[ix], rightv[ix]), eachindex(leftv))
        if ndiff == 0
            push!(equal_cols, c)
        else
            unequal_counts[c] = ndiff
        end
    end

    label_transitions = Dict{String, Int}()
    if (:label in propertynames(joined)) && (Symbol("label_1") in propertynames(joined))
        src = string.(joined[!, Symbol("label_1")])
        dst = string.(joined[!, :label])
        for ix in 1:nrow(joined)
            if src[ix] != dst[ix]
                k = string(src[ix], " -> ", dst[ix])
                label_transitions[k] = get(label_transitions, k, 0) + 1
            end
        end
    end

    limit_cols = [:lo_limit, :lc_limit, :so_limit, :sc_limit]
    limit_mismatch_counts = Dict{Symbol, Int}()
    for c in limit_cols
        if c in propertynames(joined) && Symbol(string(c, "_1")) in propertynames(joined)
            l = joined[!, c]
            r = joined[!, Symbol(string(c, "_1"))]
            limit_mismatch_counts[c] = count(ix -> !_cellsequal(l[ix], r[ix]), eachindex(l))
        end
    end

    return (
        rowstats=rowstats,
        common_cols=common_cols,
        equal_cols=equal_cols,
        unequal_counts=unequal_counts,
        label_transitions=label_transitions,
        limit_mismatch_counts=limit_mismatch_counts,
    )
end

"Transpose one DataFrame into a string-valued field x time table."
function replay_window_transpose(df::DataFrame)::DataFrame
    out = DataFrame(field=String.(names(df)))
    for r in 1:nrow(df)
        out[!, Symbol("t$(r)")] = [string(df[r, c]) for c in 1:ncol(df)]
    end
    return out
end

"""
    replay_focus_first_open_close_windows(trades_replay; focus_cols=..., return_transposed=true)

Find the first open signal that effectively increases position amount,
then find the corresponding close-order signal and first subsequent
position decrease in the same `(pair, set, rangeid)` group.

Returned named tuple fields:
- `meta`: group id and event timestamps
- `open_window`: rows from 1 minute before open signal until position increase
- `close_window`: rows from 1 minute before close-order signal until position decrease
- `open_transposed`: transposed `open_window` (if requested)
- `close_transposed`: transposed `close_window` (if requested)
"""
function replay_focus_first_open_close_windows(
    trades_replay::DataFrame;
    focus_cols::Vector{Symbol}=[
        :opentime, :pair, :set, :rangeid,
        :high, :low, :close,
        :label, :score,
        :lp_amount, :sp_amount,
        :lo_amount, :lc_amount, :so_amount, :sc_amount,
        :lo_limit, :lc_limit, :so_limit, :sc_limit,
        :lo_status, :lc_status, :so_status, :sc_status,
        :lo_id, :lc_id, :so_id, :sc_id,
        :lo_msg, :lc_msg, :so_msg, :sc_msg,
        :lol_id, :lol_status, :lol_filled, :lol_pavg, :lol_msg,
        :lcl_id, :lcl_status, :lcl_filled, :lcl_pavg, :lcl_msg,
        :sol_id, :sol_status, :sol_filled, :sol_pavg, :sol_msg,
        :scl_id, :scl_status, :scl_filled, :scl_pavg, :scl_msg,
        :equity, :freemargin, :freequote,
    ],
    return_transposed::Bool=true,
)
    @assert nrow(trades_replay) > 2 "trades_replay is empty"
    df = DataFrame(trades_replay)
    sort!(df, [:opentime, :pair, :set, :rangeid])

    open_labels_long = Set(["longopen", "longstrongopen"])
    open_labels_short = Set(["shortopen", "shortstrongopen"])
    labels = lowercase.(string.(df[!, :label]))

    open_ix = nothing
    open_fill_ix = nothing
    side = nothing
    poscol = nothing

    for ix in 1:nrow(df)
        current = labels[ix]
        if current in open_labels_long
            s = :long
            p = :lp_amount
        elseif current in open_labels_short
            s = :short
            p = :sp_amount
        else
            continue
        end

        pair = df[ix, :pair]
        setv = df[ix, :set]
        rid = df[ix, :rangeid]
        gix = findall(j -> (df[j, :pair] == pair) && (df[j, :set] == setv) && (df[j, :rangeid] == rid), 1:nrow(df))
        local_ix = findfirst(==(ix), gix)
        before = local_ix == 1 ? 0.0 : Float64(df[gix[local_ix - 1], p])
        fill_local = findfirst(k -> Float64(df[gix[k], p]) > before, local_ix:length(gix))
        if !isnothing(fill_local)
            open_ix = ix
            open_fill_ix = gix[fill_local]
            side = s
            poscol = p
            break
        end
    end

    @assert !isnothing(open_ix) "no open signal with subsequent position increase found"

    pair = df[open_ix, :pair]
    setv = df[open_ix, :set]
    rid = df[open_ix, :rangeid]
    gmask = (df.pair .== pair) .& (df.set .== setv) .& (df.rangeid .== rid)
    gdf = df[gmask, :]

    grouptimes = DateTime.(gdf[!, :opentime])
    open_local = findfirst(==(DateTime(df[open_ix, :opentime])), grouptimes)
    open_fill_local = findfirst(==(DateTime(df[open_fill_ix, :opentime])), grouptimes)
    @assert !isnothing(open_local) && !isnothing(open_fill_local) "failed to map open indices into group"

    close_amount_col = side == :long ? :lc_amount : :sc_amount
    close_local_scan = findfirst(i -> Float64(gdf[i, close_amount_col]) > 0.0, open_fill_local:nrow(gdf))
    @assert !isnothing(close_local_scan) "no close order signal ($(close_amount_col)>0) found after first effective open"
    close_local = (open_fill_local:nrow(gdf))[close_local_scan]

    close_before = close_local == 1 ? 0.0 : Float64(gdf[close_local - 1, poscol])
    close_fill_scan = findfirst(i -> Float64(gdf[i, poscol]) < close_before, close_local:nrow(gdf))
    @assert !isnothing(close_fill_scan) "no position decrease found after close order signal"
    close_fill_local = (close_local:nrow(gdf))[close_fill_scan]

    open_start_dt = DateTime(gdf[open_local, :opentime]) - Minute(2)
    open_end_dt = DateTime(gdf[open_fill_local, :opentime])
    close_start_dt = DateTime(gdf[close_local, :opentime]) - Minute(2)
    close_end_dt = DateTime(gdf[close_fill_local, :opentime])

    cols = [c for c in focus_cols if c in propertynames(gdf)]
    open_window = gdf[(gdf.opentime .>= open_start_dt) .& (gdf.opentime .<= open_end_dt), :]
    close_window = gdf[(gdf.opentime .>= close_start_dt) .& (gdf.opentime .<= close_end_dt), :]

    meta = (
        pair=String(pair),
        set=String(setv),
        rangeid=Int32(rid),
        side=side,
        position_col=poscol,
        close_amount_col=close_amount_col,
        open_signal_dt=DateTime(gdf[open_local, :opentime]),
        open_fill_dt=DateTime(gdf[open_fill_local, :opentime]),
        close_signal_dt=DateTime(gdf[close_local, :opentime]),
        close_fill_dt=DateTime(gdf[close_fill_local, :opentime]),
    )

    if return_transposed
        return (
            meta=meta,
            open_window=open_window,
            close_window=close_window,
            open_transposed=replay_window_transpose(open_window),
            close_transposed=replay_window_transpose(close_window),
        )
    end

    return (
        meta=meta,
        open_window=open_window,
        close_window=close_window,
    )
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

    # Strict sync contract: every synced base must already exist in xc.bases
    # and be advanced only by the Xch iterator.
    Xch.removeallbases(cache.xc)
    Xch.addbase!(cache.xc, base, cache.xc.startdt, cache.xc.enddt)

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

    return (pair=pair, base=base, setname=setname, rangeid=rangeid)
end

"Run one replay group through Trade tradeloop step execution."
function _run_replay_group_tradeloop!(cache::Trade.TradeCache, groupdf::DataFrame, quotecoin::AbstractString)
    _reset_replay_runtime!(cache, quotecoin, INITIAL_QUOTE_BALANCE)
    prep = _prepare_replay_group!(cache, groupdf, quotecoin)

    # Execute the native Trade backtest loop over the replay-configured window.
    # skip_init=true keeps the replay-provided cfg and avoids tradeselection rebuild.
    Trade.run_backtest!(cache; skip_init=true)

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
function run_replay_from_artifacts!(cache::Trade.TradeCache;
    logsubfolder::AbstractString=REPLAY_SOURCE_SUBFOLDER,
    quotecoin::AbstractString=QUOTE_COIN,
    startdt::Union{Nothing, DateTime}=nothing,
    enddt::Union{Nothing, DateTime}=nothing,
)
    resultsdf = _load_replay_df(logsubfolder, "results", "all")
    preddf = _load_replay_df(logsubfolder, "predictions", "maxpredictions")
    @assert nrow(resultsdf) > 0 "replay source results/all is empty"

    source_opentimes = DateTime.(resultsdf[!, :opentime])
    source_startdt = minimum(source_opentimes)
    source_enddt = maximum(source_opentimes)
    effective_startdt = isnothing(startdt) ? source_startdt : startdt
    effective_enddt = isnothing(enddt) ? source_enddt : enddt
    @assert effective_startdt <= effective_enddt "invalid replay bounds: start=$(effective_startdt), end=$(effective_enddt)"

    mask = (source_opentimes .>= effective_startdt) .& (source_opentimes .<= effective_enddt)
    allowedcoins = Set(uppercase.(String.(BACKTEST_BASES)))
    mask = mask .& [uppercase(String(c)) in allowedcoins for c in resultsdf[!, :coin]]
    resultsdf = resultsdf[mask, :]
    preddf = preddf[mask, :]
    @assert nrow(resultsdf) > 0 "replay input is empty after coins filter; coins=$(BACKTEST_BASES)"
    replaydf = _build_replay_input(resultsdf, preddf, quotecoin)
    @assert nrow(replaydf) > 0 "replay input is empty after timestamp filter; startdt=$(effective_startdt), enddt=$(effective_enddt)"

    sort!(replaydf, [:pair, :set, :rangeid, :opentime])
    _validate_replay_sequence!(replaydf)

    trades_parts = DataFrame[]
    fills_parts = DataFrame[]
    groups = groupby(replaydf, [:pair, :set, :rangeid])
    for g in groups
        tradesdf, fillsdf = _run_replay_group_tradeloop!(cache, DataFrame(g), quotecoin)
        push!(trades_parts, _stringify_categorical_columns!(DataFrame(tradesdf)))
        nrow(fillsdf) > 0 && push!(fills_parts, _stringify_categorical_columns!(DataFrame(fillsdf)))
    end

    alltrades = isempty(trades_parts) ? DataFrame() : reduce(vcat, trades_parts; cols=:union)
    allfills = isempty(fills_parts) ? DataFrame() : reduce(vcat, fills_parts; cols=:union)

    # Keep only rows that correspond to replay source keys. This removes
    # framework-introduced placeholder rows (for example set="none", rangeid=0)
    # and guarantees key-space parity with the source artifacts.
    replay_keys = unique(select(replaydf, [:pair, :set, :rangeid, :opentime]))
    alltrades = innerjoin(alltrades, replay_keys; on=[:pair, :set, :rangeid, :opentime])

    return alltrades, allfills
end

"Build one continuous multi-pair replay run into Xch-owned Trades state, ignoring set/rangeid boundaries (resembles tradereal's continuous per-minute loop across all pairs)."
function _prepare_replay_continuous!(cache::Trade.TradeCache, replaydf::DataFrame, quotecoin::AbstractString)
    overall_startdt = minimum(replaydf[!, :opentime])
    overall_enddt = maximum(replaydf[!, :opentime])
    cache.xc.startdt = overall_startdt
    cache.xc.enddt = overall_enddt
    cache.xc.currentdt = nothing
    cache.mc[:reloadtimes] = Time[]

    Xch.removeallbases(cache.xc)

    pairs = String[]
    basecoins = String[]
    lastprices = Float32[]
    for g in groupby(replaydf, :pair)
        gdf = DataFrame(g)
        pair = uppercase(String(gdf[1, :pair]))
        bq = Xch.basequote(pair)
        base = uppercase(String(bq.basecoin))

        Xch.addbase!(cache.xc, base, overall_startdt, overall_enddt)
        seeddf = select(gdf, :opentime, :pair, :set, :rangeid, :high, :low, :close, :label, :score)
        TSM.settrades!(cache.xc.tsm, pair, seeddf)

        push!(pairs, pair)
        push!(basecoins, base)
        push!(lastprices, Float32(gdf[nrow(gdf), :close]))
    end

    cache.cfg = DataFrame(
        basecoin=basecoins,
        pair=pairs,
        quotevolume24h_M=fill(0f0, length(pairs)),
        pricechangepercent=fill(0f0, length(pairs)),
        lastprice=lastprices,
        datetime=fill(overall_startdt, length(pairs)),
        minquotevol=fill(true, length(pairs)),
        continuousminvol=fill(true, length(pairs)),
        inportfolio=fill(true, length(pairs)),
        classifieraccepted=fill(true, length(pairs)),
        openenabled=fill(true, length(pairs)),
        closeenabled=fill(true, length(pairs)),
        blacklisted=fill(false, length(pairs)),
    )
    return (pairs=pairs, startdt=overall_startdt, enddt=overall_enddt)
end

"""
Run artifact-driven trades replay as ONE continuous multi-pair tradeloop that ignores
set/rangeid boundaries: all configured pairs are synced and traded together minute by
minute over the full timestamp range, resembling the live tradereal loop. Returns
(tradesdf, fillsdf).
"""
function run_replay_continuous!(cache::Trade.TradeCache;
    logsubfolder::AbstractString=REPLAY_SOURCE_SUBFOLDER,
    quotecoin::AbstractString=QUOTE_COIN,
    startdt::Union{Nothing, DateTime}=nothing,
    enddt::Union{Nothing, DateTime}=nothing,
)
    resultsdf = _load_replay_df(logsubfolder, "results", "all")
    preddf = _load_replay_df(logsubfolder, "predictions", "maxpredictions")
    @assert nrow(resultsdf) > 0 "replay source results/all is empty"

    source_opentimes = DateTime.(resultsdf[!, :opentime])
    source_startdt = minimum(source_opentimes)
    source_enddt = maximum(source_opentimes)
    effective_startdt = isnothing(startdt) ? source_startdt : startdt
    effective_enddt = isnothing(enddt) ? source_enddt : enddt
    @assert effective_startdt <= effective_enddt "invalid replay bounds: start=$(effective_startdt), end=$(effective_enddt)"

    mask = (source_opentimes .>= effective_startdt) .& (source_opentimes .<= effective_enddt)
    allowedcoins = Set(uppercase.(String.(BACKTEST_BASES)))
    mask = mask .& [uppercase(String(c)) in allowedcoins for c in resultsdf[!, :coin]]
    resultsdf = resultsdf[mask, :]
    preddf = preddf[mask, :]
    @assert nrow(resultsdf) > 0 "replay input is empty after coins filter; coins=$(BACKTEST_BASES)"
    replaydf = _build_replay_input(resultsdf, preddf, quotecoin)
    @assert nrow(replaydf) > 0 "replay input is empty after timestamp filter; startdt=$(effective_startdt), enddt=$(effective_enddt)"

    sort!(replaydf, [:pair, :opentime])
    # Only per-pair opentime ordering matters for a continuous run; set/rangeid boundaries are ignored.
    for g in groupby(replaydf, :pair)
        nrow(g) == 0 && continue
        prev = g[1, :opentime]
        for ix in 2:nrow(g)
            cur = g[ix, :opentime]
            @assert prev < cur "invalid replay sequence for pair=$(g[ix, :pair]); expected opentime[ix-1] < opentime[ix], got $(prev) and $(cur)"
            prev = cur
        end
    end

    _reset_replay_runtime!(cache, quotecoin, INITIAL_QUOTE_BALANCE)
    _prepare_replay_continuous!(cache, replaydf, quotecoin)

    # skip_init=true keeps the replay-provided cfg and avoids tradeselection rebuild.
    Trade.run_backtest!(cache; skip_init=true)
    finaldt = cache.xc.currentdt

    alltrades = _stringify_categorical_columns!(TSM.collecttradesdf(cache.xc.tsm))
    allfills = _stringify_categorical_columns!(filled_orders_df(cache.xc))
    (nrow(allfills) > 0) && (allfills[!, :pair] = allfills[!, :symbol])

    # Drop rows pre-seeded from replaydf but never reached by the tick loop (e.g. one
    # pair's OHLCV data ends before the shared overall_enddt): they still carry seeded
    # label/high/low/close but no trading update, so a still-open position there shows a
    # spurious flat position with no close price and breaks gains compilation.
    !isnothing(finaldt) && filter!(:opentime => <=(finaldt), alltrades)

    # Keep only rows that correspond to replay source keys (drops framework-introduced
    # placeholder rows, e.g. minutes without a seeded replay row for that pair).
    replay_keys = unique(select(replaydf, [:pair, :opentime]))
    alltrades = innerjoin(alltrades, replay_keys; on=[:pair, :opentime])

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

EnvConfig.init(TESTMODE ? test : training)  # test mode → cryptoxchsim, no live credentials needed
EnvConfig.setpairquote!(QUOTE_COIN)
EnvConfig.setlogpath(LOG_SUBFOLDER)

Xch.verbosity = 1
Classify.verbosity  = 2
Trade.verbosity     = 3

println("$(EnvConfig.now()): starting tradesim with config=$CONFIG_NAME phase=$(TESTMODE ? "test" : "training") coins=$BACKTEST_BASES usepartitions=$USE_PARTITIONS")
# println("$(EnvConfig.now()): backtest $BACKTEST_STARTDT → $BACKTEST_ENDDT")

println("$(EnvConfig.now()): replay source folder=$REPLAY_SOURCE_SUBFOLDER")

effective_startdt, effective_enddt = backtest_bounds_from_env(BACKTEST_STARTDT, BACKTEST_ENDDT)
run_startdt, run_enddt = effective_startdt, effective_enddt
resultsdf_window = _load_replay_df(REPLAY_SOURCE_SUBFOLDER, "results", "all")
@assert nrow(resultsdf_window) > 0 "replay source results/all is empty"
source_opentimes_window = DateTime.(resultsdf_window[!, :opentime])
cache_startdt = isnothing(run_startdt) ? minimum(source_opentimes_window) : run_startdt
cache_enddt = isnothing(run_enddt) ? maximum(source_opentimes_window) : run_enddt
strategy_runtime = TradingStrategy.TsCache(CONFIG_REF; source="tradesim:$CONFIG_NAME")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE CACHE
# ─────────────────────────────────────────────────────────────────────────────

bc = Bybit.BybitCache()
Bybit.seedportfolio!(bc, QUOTE_COIN, 0.0)
xc = Xch.XchCache(bc;
    startdt  = cache_startdt,
    enddt    = cache_enddt,
)
    TSM.ensuretradesschema!(xc.tsm, TSM.tradesdf_all_contributors())

cache = Trade.TradeCache(xc=xc, strategy=strategy_runtime, trademode=TRADE_MODE, stoplosspct=STOPLOSSPCT)
seed_quote_balance!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)
ensure_quote_budget!(xc, QUOTE_COIN, INITIAL_QUOTE_BALANCE)

# Override risk parameters.
cache.mc[:maxassetfraction] = MAX_ASSET_FRACTION
cache.mc[:maxbudgetquote]   = MAX_BUDGET_QUOTE

println("$(EnvConfig.now()): exchange=$EXCHANGE, trademode=$TRADE_MODE")
println("$(EnvConfig.now()): strategy config=$CONFIG_NAME, engine=tradingstrategy, openthreshold=$(cache.ts.cfg.openthreshold)")
println("$(EnvConfig.now()): quote coin=$QUOTE_COIN, initial balance=$INITIAL_QUOTE_BALANCE")
println("$(EnvConfig.now()): blacklist ($(length(cache.mc[:blacklistbases])) bases): $(cache.mc[:blacklistbases])")
# println("$(EnvConfig.now()): running backtest over $run_startdt → $run_enddt")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

alltrades, allfills = if USE_PARTITIONS
    run_replay_from_artifacts!(cache;
        logsubfolder=REPLAY_SOURCE_SUBFOLDER,
        quotecoin=QUOTE_COIN,
        startdt=run_startdt,
        enddt=run_enddt,
    )
else
    run_replay_continuous!(cache;
        logsubfolder=REPLAY_SOURCE_SUBFOLDER,
        quotecoin=QUOTE_COIN,
        startdt=run_startdt,
        enddt=run_enddt,
    )
end
_validate_tradesim_replay_result!(alltrades)
println("$(EnvConfig.now()): replay simulation finished, trades rows=$(nrow(alltrades)), fills rows=$(nrow(allfills))")

replay_out_folder = joinpath(EnvConfig.logfolder(), "tradesim-replay")
mkpath(replay_out_folder)
saved_trades = EnvConfig.savedf(alltrades, "trades-replay"; folderpath=replay_out_folder)
saved_fills = EnvConfig.savedf(allfills, "fills-replay"; folderpath=replay_out_folder)
replay_gains_stem = "xchgains-replay"
replay_report_stem = "xchgainsreport-replay"
# Continuous replay (usepartitions=false, the default) can hold one position open across
# set/rangeid boundaries; only the legacy partitioned mode has independent per-range runs.
replay_gainsdf = TSM.compilegainsdf(alltrades; stem=replay_gains_stem, folderpath=replay_out_folder, grouppartitions=USE_PARTITIONS)
replay_reportdf = TSM.gainsreport(instem=replay_gains_stem, stem=replay_report_stem, folderpath=replay_out_folder)
saved_gains = EnvConfig.tablepath(replay_gains_stem; folderpath=replay_out_folder, format=:auto)
saved_gainsreport = EnvConfig.tablepath(replay_report_stem; folderpath=replay_out_folder, format=:auto)
println("$(EnvConfig.now()): replay gains report")
println(replay_gainsdf[begin:min(begin + 10, end), :])
println(replay_gainsdf[max(begin, end - 10):end, :])
println(replay_reportdf)
println("$(EnvConfig.now()): saved replay trades to $saved_trades rows=$(nrow(alltrades)) $(nrow(alltrades) > 0 ? (string(alltrades[begin, :opentime]) * " - " * string(alltrades[end, :opentime])) : nothing)")
# println(alltrades)
println("$(EnvConfig.now()): saved replay fills to $saved_fills rows=$(nrow(allfills))")
# println(allfills)
println("$(EnvConfig.now()): saved replay gains to $saved_gains rows=$(nrow(replay_gainsdf))")
# println(replay_gainsdf)
println("$(EnvConfig.now()): saved replay gains report to $saved_gainsreport rows=$(nrow(replay_reportdf))")
# println(replay_reportdf)

# replay_compare_root = joinpath(dirname(EnvConfig.logfolder()), REPLAY_SOURCE_SUBFOLDER)
# trades_td = EnvConfig.readdf("trades-td"; folderpath=replay_compare_root)
# if !isnothing(trades_td)
#     cmp = tradescompare(DataFrame(trades_td), alltrades)
#     println("$(EnvConfig.now()): trades compare rowstats replay=$(cmp.rowstats.replay_rows) td=$(cmp.rowstats.td_rows) joined=$(cmp.rowstats.joined_rows)")
#     println("$(EnvConfig.now()): trades compare equal_cols=$(length(cmp.equal_cols)) unequal_cols=$(length(cmp.unequal_counts))")
#     if haskey(cmp.unequal_counts, :label)
#         println("$(EnvConfig.now()): trades compare label mismatches=$(cmp.unequal_counts[:label])")
#         top_label_transitions = sort(collect(cmp.label_transitions); by=last, rev=true)
#         shown = min(8, length(top_label_transitions))
#         for ix in 1:shown
#             println("$(EnvConfig.now()): label transition $(top_label_transitions[ix][1]) count=$(top_label_transitions[ix][2])")
#         end
#     end
#     for lane in [:lo_limit, :lc_limit, :so_limit, :sc_limit]
#         if haskey(cmp.limit_mismatch_counts, lane)
#             println("$(EnvConfig.now()): trades compare $(lane) mismatches=$(cmp.limit_mismatch_counts[lane])")
#         end
#     end
# else
#     println("$(EnvConfig.now()): trades compare skipped; missing trades-td.arrow in $(replay_compare_root)")
# end

# focus = replay_focus_first_open_close_windows(alltrades; return_transposed=true)
# println("$(EnvConfig.now()): replay focus group pair=$(focus.meta.pair) set=$(focus.meta.set) rangeid=$(focus.meta.rangeid) side=$(focus.meta.side)")
# println("$(EnvConfig.now()): replay focus open signal=$(focus.meta.open_signal_dt) fill=$(focus.meta.open_fill_dt)")
# println("$(EnvConfig.now()): replay focus close signal=$(focus.meta.close_signal_dt) fill=$(focus.meta.close_fill_dt)")
# println("$(EnvConfig.now()): replay focus open window (transposed)\n$(focus.open_transposed)")
# println("$(EnvConfig.now()): replay focus close window (transposed)\n$(focus.close_transposed)")

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE REPORT
# ─────────────────────────────────────────────────────────────────────────────

if nrow(allfills) == 0
    println("$(EnvConfig.now()): replay produced no filled-order rows")
else
    println("$(EnvConfig.now()): replay filled-order summary: rows=$(nrow(allfills))")
end
EnvConfig.setlogpath(LOG_SUBFOLDER)
println("$(EnvConfig.now()): skipped trades-ts persistence; using tradesim-replay/trades-replay.arrow as canonical replay output")

# Keep legacy log-path split for parity with previous script layout.
println("$(EnvConfig.now()): order history report derived from xc.tsm.pairstates trades data")
