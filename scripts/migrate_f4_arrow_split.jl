#!/usr/bin/env julia

"""
Convert legacy split-column F4 Arrow caches (`coins/<pair>/f4/*.arrow`) into one
shared `coins/<pair>/f4.arrow` per coin pair.

Usage:
  julia --project=. scripts/migrate_f4_arrow_split.jl
  julia --project=. scripts/migrate_f4_arrow_split.jl --keep-legacy
"""

using EnvConfig
using Features
using Ohlcv

"""
Parse a coin pair folder name like `BTC-USDT` into `(base, quote)`.
Returns `nothing` when the name does not match the expected pattern.
"""
function parse_pair(dirname::AbstractString)
    parts = split(dirname, "-")
    if length(parts) != 2
        return nothing
    end
    return (uppercase(parts[1]), uppercase(parts[2]))
end

"""
Migrate all legacy split-column F4 Arrow caches under `EnvConfig.coinspath()`.
Returns `(migrated, skipped)` counts.
"""
function migrate_all_split_f4!(; delete_legacy::Bool=true)
    coinsroot = EnvConfig.coinspath()
    if !isdir(coinsroot)
        println("coinspath does not exist: $(coinsroot)")
        return (0, 0)
    end

    migrated = 0
    skipped = 0
    failed = 0

    for entry in readdir(coinsroot, join=false, sort=true)
        pair = parse_pair(entry)
        isnothing(pair) && continue

        basecoin, quotecoin = pair
        legacydir = joinpath(coinsroot, entry, "f4")
        !isdir(legacydir) && continue

        try
            didmigrate = Features.migrate_split_f4_arrow!(basecoin, quotecoin)
            if didmigrate
                migrated += 1
                println("migrated $(basecoin)-$(quotecoin)")
                if delete_legacy && isdir(legacydir)
                    rm(legacydir; force=true, recursive=true)
                end
            else
                # Fallback: recompute F4 from OHLCV if legacy split files are unreadable/empty.
                ohlcv = Ohlcv.read(basecoin)
                if !isnothing(ohlcv) && (size(ohlcv.df, 1) > 0)
                    f4 = Features.Features004(ohlcv; usecache=false)
                    if !isnothing(f4)
                        Features.write(f4)
                        migrated += 1
                        println("migrated $(basecoin)-$(quotecoin) (recomputed from OHLCV)")
                        if delete_legacy && isdir(legacydir)
                            rm(legacydir; force=true, recursive=true)
                        end
                    else
                        skipped += 1
                        println("skipped $(basecoin)-$(quotecoin) (no readable split Arrow files and recompute unavailable)")
                    end
                else
                    skipped += 1
                    println("skipped $(basecoin)-$(quotecoin) (no readable split Arrow files and empty OHLCV)")
                end
            end
        catch e
            failed += 1
            println("failed $(basecoin)-$(quotecoin): $(e)")
        end
    end

    return (migrated, skipped, failed)
end

function main(args::Vector{String})
    keep_legacy = any(arg -> arg == "--keep-legacy", args)
    migrated, skipped, failed = migrate_all_split_f4!(; delete_legacy=!keep_legacy)
    println("done: migrated=$(migrated), skipped=$(skipped), failed=$(failed), delete_legacy=$(!keep_legacy)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
