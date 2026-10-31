"""
Audit stored Trend05 entry/hold segments against P02 gaps without rebuilding targets.

Run with `julia --project=. scripts/investigate_trend05_horizon.jl`.
Add `--invalidate` to remove only target Arrow files with proven partition-gap
crossings, after logging their evidence. Existing classifiers are never deleted.
"""
module Trend05HorizonInvestigation

using Dates, DataFrames, Arrow
using EnvConfig, Targets, Classify, TradingStrategy

"""
    auditsegments(targets, sets, gapminutes)

Return summary counts and crossing examples from stored entry/hold labels.
The end of a contiguous hold segment is the successful candidate's target-reach row.
Inspect every source partition overlapping that segment, including holds whose
entry starts in a gap. This establishes successful-target dependencies only:
failed and unresolved entry candidates cannot be reconstructed from stored labels.
"""
function auditsegments(targets::AbstractDataFrame, sets::AbstractDataFrame, gapminutes::Integer)
    @assert gapminutes > 0 "gapminutes=$(gapminutes) must be positive"
    @assert issorted(targets.opentime) && allunique(targets.opentime) "target timestamps must be sorted and unique, rows=$(nrow(targets))"
    @assert issorted(sets.opentime) && allunique(sets.opentime) "sets timestamps must be sorted and unique, rows=$(nrow(sets))"
    starts = Int[]
    ends = Int[]
    for ix in 1:nrow(sets)
        if ix == 1 || sets.rangeid[ix] != sets.rangeid[ix - 1]
            push!(starts, ix)
            ix > 1 && push!(ends, ix - 1)
        end
    end
    isempty(starts) || push!(ends, nrow(sets))
    starttimes = sets.opentime[starts]
    endtimes = sets.opentime[ends]
    entries = overgap = crossings = mixedsetcrossings = 0
    maximumduration = Millisecond(0)
    examples = NamedTuple[]
    ix = 1
    while ix <= nrow(targets)
        code = targets.label[ix]
        if code != Targets.tradelabelcode(Targets.longopen) && code != Targets.tradelabelcode(Targets.shortopen)
            ix += 1
            continue
        end
        entries += 1
        holdcode = Targets.tradelabelcode(code == Targets.tradelabelcode(Targets.longopen) ? Targets.longhold : Targets.shorthold)
        endpoint = ix
        while endpoint < nrow(targets) && targets.label[endpoint + 1] == holdcode
            endpoint += 1
        end
        @assert endpoint > ix "stored entry at $(targets.opentime[ix]) has no target-reaching hold segment"
        duration = targets.opentime[endpoint] - targets.opentime[ix]
        maximumduration = max(maximumduration, duration)
        overgap += duration > Minute(gapminutes)
        firstpartition = searchsortedfirst(endtimes, targets.opentime[ix])
        lastpartition = searchsortedlast(starttimes, targets.opentime[endpoint])
        bridged = mixed = false
        for partition in firstpartition:(lastpartition - 1)
            nextpartition = partition + 1
            # There are labelled source rows before a gap and a partition after it
            # inside the same target dependency interval.
            source = max(starttimes[partition], targets.opentime[ix])
            if source <= endtimes[partition] && starttimes[nextpartition] <= targets.opentime[endpoint]
                bridged = true
                differentset = sets.set[starts[partition]] != sets.set[starts[nextpartition]]
                mixed |= differentset
                if length(examples) < 20
                    push!(examples, (entry=targets.opentime[ix], reached=targets.opentime[endpoint],
                        source=source, source_set=string(sets.set[starts[partition]]),
                        next_set=string(sets.set[starts[nextpartition]]),
                        next_start=starttimes[nextpartition], elapsed_minutes=duration / Minute(1),
                        different_set=differentset))
                end
            end
        end
        crossings += bridged
        mixedsetcrossings += mixed
        ix = endpoint + 1
    end
    summary = (entries=entries, overgap=overgap, gapcrossings=crossings,
        differentsetcrossings=mixedsetcrossings, maximum_minutes=maximumduration / Minute(1))
    return summary, examples
end

"""
    main(args=ARGS)

Audit existing T24/T25/T26 stores for CS01/D01 using their embedded metadata,
not the newly bounded config hashes. Missing stores are reported explicitly.
With `--invalidate`, remove only proven-crossing target files and log each removal.
"""
function main(args::Vector{String}=ARGS)
    @assert all(arg -> arg == "--invalidate", args) "unknown args=$(args); supported flag is --invalidate"
    EnvConfig.init(EnvConfig.training)
    EnvConfig.setcoinspath!("Bybit")
    EnvConfig.setdfformat!(:arrow)
    folder = mkpath(joinpath(EnvConfig.defaultdebugfilespath, "trend05-horizon"))
    logpath = joinpath(folder, "audit-$(Dates.format(Dates.now(), dateformat"yyyymmdd-HHMMSS")).log")
    open(logpath, "w") do output
        println(output, "Successful stored Trend05 segments only; failed candidate dependencies are not encoded in target stores.")
        println(output, "Current pre-change implementation searched through the available base without a fixed deadline.")
        println(output, "invalidate=$("--invalidate" in args) coins=CS01 daterange=D01")
        flush(output)
        totals = Dict(id => (stores=0, entries=0, overgap=0, gapcrossings=0, differentsetcrossings=0, removed=0, invalid=0) for id in ("T24", "T25", "T26"))
        for coin in TradingStrategy.coinsetconfig("CS01").coins
            setfolder = Classify.setsfolder(coin, EnvConfig.pairquote)
            setpath = EnvConfig.tablepath("P02_D01"; folderpath=setfolder, format=:arrow)
            if !isfile(setpath)
                println(output, "MISSING sets coin=$coin path=$setpath")
                flush(output)
                continue
            end
            sets = DataFrame(Arrow.Table(setpath))
            for id in ("T24", "T25", "T26")
                targetfolder = Targets.targetsfolder(coin, EnvConfig.pairquote)
                path = EnvConfig.tablepath(id; folderpath=targetfolder, format=:arrow)
                if !isfile(path)
                    println(output, "MISSING targets coin=$coin target=$id path=$path")
                    flush(output)
                    continue
                end
                table = Arrow.Table(path)
                metadata = Arrow.getmetadata(table)
                if !issorted(table.opentime) || !allunique(table.opentime)
                    @error "Cannot audit target store with unsorted or duplicate timestamps; file retained" coin target=id path
                    println(output, "INVALID_TARGET_STORE coin=$coin target=$id path=$path: unsorted or duplicate timestamps; retained without a gap-safety conclusion")
                    totals[id] = merge(totals[id], (invalid=totals[id].invalid + 1,))
                    flush(output)
                    continue
                end
                summary, examples = auditsegments(DataFrame(table), sets, TradingStrategy.partitionconfig("P02").gapsize)
                println(output, "coin=$coin target=$id path=$path metadata=$metadata summary=$summary")
                storedhash = isnothing(metadata) ? nothing : get(metadata, "confighash", nothing)
                if storedhash != TradingStrategy.targetshash(id)
                    println(output, "CONFIG_STALE coin=$coin target=$id storedhash=$storedhash boundedhash=$(TradingStrategy.targetshash(id)); old stores/classifiers must not be reused under bounded config")
                end
                for example in examples
                    println(output, "EVIDENCE $example")
                end
                flush(output)
                removed = 0
                if "--invalidate" in args && summary.gapcrossings > 0
                    rm(path)
                    removed = 1
                    println(output, "REMOVED target=$id coin=$coin path=$path proven_gapcrossings=$(summary.gapcrossings)")
                    flush(output)
                end
                previous = totals[id]
                totals[id] = (stores=previous.stores + 1, entries=previous.entries + summary.entries,
                    overgap=previous.overgap + summary.overgap, gapcrossings=previous.gapcrossings + summary.gapcrossings,
                    differentsetcrossings=previous.differentsetcrossings + summary.differentsetcrossings,
                    removed=previous.removed + removed, invalid=previous.invalid)
                println("$(EnvConfig.now()) coin=$coin target=$id $summary removed=$removed")
            end
        end
        for id in ("T24", "T25", "T26")
            println(output, "TOTAL $id $(totals[id])")
            println("TOTAL $id $(totals[id])")
        end
    end
    println("Audit log: $logpath")
    return logpath
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

end
