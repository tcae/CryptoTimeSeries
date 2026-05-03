#!/usr/bin/env julia

"""
Ensure shared F4 Arrow caches contain regression features for all required windows.

For each coin pair folder in EnvConfig.coinspath(), the script checks whether
`f4.arrow` contains these columns for every window in Features.regressionwindows004:
- <window>_regry
- <window>_grad
- <window>_std

If any window is missing one or more of those columns, the script computes only
those missing windows from OHLCV and writes them into shared f4.arrow.

Usage:
  julia --project=. scripts/ensure_f4_regression_features.jl
  julia --project=. scripts/ensure_f4_regression_features.jl --coin ADA
  julia --project=. scripts/ensure_f4_regression_features.jl --quote USDT
  julia --project=. scripts/ensure_f4_regression_features.jl --coin ADA --quote USDT
    julia --project=. scripts/ensure_f4_regression_features.jl --cleanup
"""

using EnvConfig
using Features
using Ohlcv
using DataFrames
using Dates

"""
Parse optional CLI arguments.
Returns `(coin_filter, quote_filter, cleanup_skipped)` where each value is
`nothing` or uppercase string and `cleanup_skipped` is a Bool.
"""
function parse_args(args::Vector{String})
    coin_filter = nothing
    quote_filter = nothing
    cleanup_skipped = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--coin"
            if i == length(args)
                error("missing value for --coin")
            end
            coin_filter = uppercase(args[i + 1])
            i += 2
        elseif arg == "--quote"
            if i == length(args)
                error("missing value for --quote")
            end
            quote_filter = uppercase(args[i + 1])
            i += 2
        elseif arg == "--cleanup"
            cleanup_skipped = true
            i += 1
        else
            error("unknown argument: $(arg)")
        end
    end
    return coin_filter, quote_filter, cleanup_skipped
end

"""
Move one skipped coin pair folder to `coins/archive`.
Returns `true` if a folder was moved.
"""
function archive_pair_folder!(coinsroot::AbstractString, pairdirname::AbstractString)::Bool
    source = joinpath(coinsroot, pairdirname)
    if !isdir(source)
        return false
    end

    archivedir = joinpath(coinsroot, "archive")
    mkpath(archivedir)

    target = joinpath(archivedir, pairdirname)
    if ispath(target)
        stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        target = joinpath(archivedir, "$(pairdirname)_$(stamp)")
    end

    mv(source, target; force=false)
    return true
end

"""
Parse pair directory name like `BASE-QUOTE`.
Returns `nothing` for non-pair names.
"""
function parse_pair(dirname::AbstractString)
    parts = split(dirname, "-")
    if length(parts) != 2
        return nothing
    end
    return uppercase(parts[1]), uppercase(parts[2])
end

"""
Return the required shared F4 column symbols for all configured regression windows.
"""
function required_f4_columns()::Vector{Symbol}
    cols = Symbol[:opentime]
    for w in Features.regressionwindows004
        push!(cols, Symbol("$(w)_regry"))
        push!(cols, Symbol("$(w)_grad"))
        push!(cols, Symbol("$(w)_std"))
    end
    return cols
end

"""
Return windows that are missing one or more expected columns in the current shared F4 dataframe.
"""
function missing_windows(df::AbstractDataFrame)::Vector{Int}
    names_set = Set(Symbol.(names(df)))
    missing = Int[]
    for w in Features.regressionwindows004
        regry_col = Symbol("$(w)_regry")
        grad_col = Symbol("$(w)_grad")
        std_col = Symbol("$(w)_std")
        if !(regry_col in names_set) || !(grad_col in names_set) || !(std_col in names_set)
            push!(missing, w)
        end
    end
    return missing
end

"""
Ensure one pair has all required F4 regression columns.
Returns status symbol in (:already_ok, :updated, :skipped).
"""
function ensure_pair!(basecoin::AbstractString, quotecoin::AbstractString)::Symbol
    shared = Features._read_shared_f4_arrow(basecoin, quotecoin)
    missing = missing_windows(shared)
    if isempty(missing)
        return :already_ok
    end

    ohlcv = Ohlcv.read(basecoin)
    if isnothing(ohlcv) || (size(ohlcv.df, 1) == 0)
        return :skipped
    end

    f4 = Features.Features004(ohlcv; regrwindows=missing, usecache=false)
    if isnothing(f4)
        return :skipped
    end

    Features.write(f4)
    return :updated
end

"""
Iterate all coin pair directories and ensure required F4 columns exist.
Returns `(already_ok, updated, skipped)` counts.
"""
function ensure_all_pairs!(; coin_filter=nothing, quote_filter=nothing, cleanup_skipped::Bool=false)
    coinsroot = EnvConfig.coinspath()
    if !isdir(coinsroot)
        println("coinspath does not exist: $(coinsroot)")
        return 0, 0, 0
    end

    already_ok = 0
    updated = 0
    skipped = 0

    for entry in readdir(coinsroot, join=false, sort=true)
        pair = parse_pair(entry)
        isnothing(pair) && continue
        basecoin, quotecoin = pair

        if !isnothing(coin_filter) && (basecoin != coin_filter)
            continue
        end
        if !isnothing(quote_filter) && (quotecoin != quote_filter)
            continue
        end

        status = ensure_pair!(basecoin, quotecoin)
        if status == :already_ok
            already_ok += 1
        elseif status == :updated
            updated += 1
            println("updated $(basecoin)-$(quotecoin)")
        else
            skipped += 1
            println("skipped $(basecoin)-$(quotecoin)")
            if cleanup_skipped
                if archive_pair_folder!(coinsroot, entry)
                    println("archived $(basecoin)-$(quotecoin) into $(joinpath(coinsroot, "archive"))")
                else
                    println("archive skipped for $(basecoin)-$(quotecoin): source folder missing")
                end
            end
        end
    end

    return already_ok, updated, skipped
end

function main(args::Vector{String})
    coin_filter, quote_filter, cleanup_skipped = parse_args(args)
    already_ok, updated, skipped = ensure_all_pairs!(; coin_filter=coin_filter, quote_filter=quote_filter, cleanup_skipped=cleanup_skipped)
    println("done: already_ok=$(already_ok), updated=$(updated), skipped=$(skipped)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
