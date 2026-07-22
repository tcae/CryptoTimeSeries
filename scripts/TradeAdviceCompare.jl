"""
TradeAdviceCompare.jl

Run TrendDetector and tradesim with aligned configuration, then compare
trade-advice columns between TrendDetector (`trades-td.arrow`) and tradesim
(`trades.arrow`).

Usage:
  julia --project=. scripts/TradeAdviceCompare.jl

Environment overrides:
    TRADE_ADVICE_CONFIG_REF   (default: 046)
    TRADE_ADVICE_STARTDT      (default: 2025-07-01T01:00:00)
    TRADE_ADVICE_ENDDT        (default: 2025-07-30T01:00:00)
    TRADE_ADVICE_COINS        (default: SINE)          # comma-separated
    TRADE_ADVICE_MODE         (default: test)          # test|train
    TRADE_ADVICE_CLASSIFIER_FOLDER (default: Trend-<config>-test)
    TRADE_ADVICE_TREND_ROOT   (default: \$HOME/crypto/logs)
    TRADE_ADVICE_SIM_ROOT     (default: \$HOME/crypto/debug)
"""

using Dates, DataFrames
using EnvConfig

function _env_or_default(key::AbstractString, default::AbstractString)::String
    raw = strip(get(ENV, String(key), ""))
    return isempty(raw) ? String(default) : raw
end

function _parse_coins(raw::AbstractString)::Vector{String}
    coins = [uppercase(strip(token)) for token in split(String(raw), ",") if !isempty(strip(token))]
    @assert !isempty(coins) "TRADE_ADVICE_COINS must contain at least one coin"
    return unique(coins)
end

function _run_or_fail(cmd::Cmd)
    println("Running: ", cmd)
    proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=false)
    try
        wait(proc)
    catch ex
        if ex isa InterruptException
            println("Interrupted: forwarding SIGINT to child process")
            try
                kill(proc, Base.SIGINT)
            catch
            end
            try
                wait(proc)
            catch
                # If child does not stop on SIGINT, force termination.
                try
                    kill(proc, Base.SIGTERM)
                catch
                end
            end
            rethrow()
        end
        rethrow()
    end
    success(proc) || error("command failed with exitcode=$(proc.exitcode): $(cmd)")
    return nothing
end

function _eq_cell(a, b; atol::Float64=1e-6)::Bool
    if ismissing(a) && ismissing(b)
        return true
    elseif ismissing(a) || ismissing(b)
        return false
    end

    if (a isa Number) && (b isa Number)
        return isapprox(Float64(a), Float64(b); atol=atol, rtol=0.0)
    end
    return string(a) == string(b)
end

function _pick_join_keys(tddf::DataFrame, simdf::DataFrame)::Vector{Symbol}
    preferred = (:pair, :opentime, :coin, :set, :rangeid)
    td_cols = Set(Symbol.(names(tddf)))
    sim_cols = Set(Symbol.(names(simdf)))
    keys = Symbol[s for s in preferred if (s in td_cols) && (s in sim_cols)]
    @assert !isempty(keys) "no common join keys found between trades-td and trades-ts"
    return keys
end

function _comparison_df(tddf::DataFrame, simdf::DataFrame)::DataFrame
    joinkeys = _pick_join_keys(tddf, simdf)

    # Fields requested by user.
    compare_fields = Symbol[:lo_limit, :lc_limit, :so_limit, :sc_limit, :label, :score, :high, :close, :low]
    td_cols = Set(Symbol.(names(tddf)))
    sim_cols = Set(Symbol.(names(simdf)))
    fields = Symbol[f for f in compare_fields if (f in td_cols) && (f in sim_cols)]
    @assert !isempty(fields) "no common compare fields found for limits/label/score/high/close/low"

    td_keep = vcat(joinkeys, fields)
    sim_keep = vcat(joinkeys, fields)

    td = select(tddf, td_keep)
    sim = select(simdf, sim_keep)

    rename!(td, Dict(f => Symbol(string(f), "_td") for f in fields))
    rename!(sim, Dict(f => Symbol(string(f), "_sim") for f in fields))

    cmp = outerjoin(td, sim; on=joinkeys, makeunique=false)

    eqcols = Symbol[]
    for f in fields
        ctd = Symbol(string(f), "_td")
        csim = Symbol(string(f), "_sim")
        ceq = Symbol(string(f), "_eq")
        cmp[!, ceq] = [_eq_cell(a, b) for (a, b) in zip(cmp[!, ctd], cmp[!, csim])]
        push!(eqcols, ceq)
    end

    cmp[!, :match_all] = [all(Bool[row[ceq] for ceq in eqcols]) for row in eachrow(cmp)]
    sort!(cmp, joinkeys)
    return cmp
end

function _load_arrow_df(folderpath::AbstractString, stem::AbstractString)::DataFrame
    loaded = EnvConfig.readdf(String(stem); folderpath=String(folderpath))
    @assert !isnothing(loaded) "expected table $(stem) in folder $(folderpath)"
    return DataFrame(loaded)
end

function _has_table(folderpath::AbstractString, stem::AbstractString)::Bool
    try
        return !isnothing(EnvConfig.readdf(String(stem); folderpath=String(folderpath)))
    catch err
        @warn "Skipping unreadable artifact during probe" folderpath stem exception=(err, catch_backtrace())
        return false
    end
end

function _resolve_artifact_folder(subfolder::AbstractString, stem::AbstractString, roots::Vector{String})::String
    for root in roots
        folder = joinpath(root, String(subfolder))
        if _has_table(folder, stem)
            return folder
        end
    end
    return joinpath(first(roots), String(subfolder))
end

function main()
    config_ref = _env_or_default("TRADE_ADVICE_CONFIG_REF", "046")
    startdt = _env_or_default("TRADE_ADVICE_STARTDT", "2025-07-01T04:01:00")
    enddt = _env_or_default("TRADE_ADVICE_ENDDT", "2025-07-20T04:01:00")
    mode = lowercase(_env_or_default("TRADE_ADVICE_MODE", "test"))
    @assert mode in ("test", "train") "TRADE_ADVICE_MODE must be test or train"
    coins = _parse_coins(_env_or_default("TRADE_ADVICE_COINS", "SINE"))
    classifier_folder = _env_or_default("TRADE_ADVICE_CLASSIFIER_FOLDER", "Trend-$(config_ref)-test")
    trend_root = _env_or_default("TRADE_ADVICE_TREND_ROOT", joinpath(homedir(), "crypto", "logs"))
    sim_root = _env_or_default("TRADE_ADVICE_SIM_ROOT", joinpath(homedir(), "crypto", "debug"))
    legacy_logs_root = joinpath(homedir(), "crypto", "logs")

    trend_folder = "tradeadvicecompare-td-$(config_ref)"
    sim_folder = "tradeadvicecompare-sim-$(config_ref)"
    trend_folderpath = _resolve_artifact_folder(trend_folder, "trades-td", [trend_root])
    sim_folderpath = _resolve_artifact_folder(sim_folder, "trades-ts", unique([sim_root, legacy_logs_root]))

    trend_args = [
        mode,
        "special",
        "config=$(config_ref)",
        "folder=$(trend_folder)",
        "startdt=$(startdt)",
        "enddt=$(enddt)",
        "coins=$(join(coins, ","))",
    ]

    if _has_table(trend_folderpath, "trades-td")
        println("Reusing existing TrendDetector artifact in ", trend_folderpath)
    else
        trend_cmd = addenv(
            Cmd(vcat(
                collect(Base.julia_cmd().exec),
                ["--project=.", "scripts/TrendDetector.jl"],
                trend_args,
            )),
            "TRENDDETECTOR_CLASSIFIER_FOLDER" => classifier_folder,
        )
        Base.invokelatest(_run_or_fail, trend_cmd)
    end

    if _has_table(sim_folderpath, "trades-ts")
        println("Reusing existing tradesim artifact in ", sim_folderpath)
    else
        sim_cmd = addenv(
            Cmd(vcat(
                collect(Base.julia_cmd().exec),
                ["--project=.", "scripts/tradesim.jl"],
            )),
            "TRADESIM_CONFIG_REF" => config_ref,
            "TRADESIM_STARTDT" => startdt,
            "TRADESIM_ENDDT" => enddt,
            "TRADESIM_BASES" => join(coins, ","),
            "TRADESIM_LOG_SUBFOLDER" => sim_folder,
        )
        Base.invokelatest(_run_or_fail, sim_cmd)
    end

    tddf = _load_arrow_df(trend_folderpath, "trades-td")
    simdf = _load_arrow_df(sim_folderpath, "trades-ts")

    cmp = _comparison_df(tddf, simdf)

    total = nrow(cmp)
    td_present = .!ismissing.(cmp[!, :label_td])
    sim_present = .!ismissing.(cmp[!, :label_sim])
    both_present = td_present .& sim_present
    only_td = td_present .& .!sim_present
    only_sim = .!td_present .& sim_present

    matched_all = count(cmp[!, :match_all])
    mismatched_all = total - matched_all

    overlap_total = count(both_present)
    overlap_matched = overlap_total == 0 ? 0 : count(cmp[both_present, :match_all])
    overlap_mismatched = overlap_total - overlap_matched

    println("\nTradeAdviceCompare summary")
    println("  config_ref: ", config_ref)
    println("  mode: ", mode)
    println("  coins: ", join(coins, ","))
    println("  range: ", startdt, " -> ", enddt)
    println("  trend root: ", trend_root)
    println("  sim root:   ", sim_root)
    println("  trend folder: ", trend_folderpath)
    println("  sim folder:   ", sim_folderpath)
    println("  rows (outer): ", total, " matched: ", matched_all, " mismatched: ", mismatched_all)
    println("  rows (overlap): ", overlap_total, " matched: ", overlap_matched, " mismatched: ", overlap_mismatched)
    println("  row origins: only td=", count(only_td), " only sim=", count(only_sim), " both=", overlap_total)

    println("\nComparison DataFrame:")
    show(cmp; allrows=true, allcols=true, truncate=0)
    println()

    return nothing
end

Base.invokelatest(main)
