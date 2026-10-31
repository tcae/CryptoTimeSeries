"""
Update cached Bybit USDT one-minute OHLCV and discover new liquid coins.
Run from the repository root: `julia --project=. scripts/supplement.jl`.
No orders, classifier loading, or training. Stop with Ctrl+C; rerun to resume.
"""
module Supplement

using Dates, DataFrames, Arrow, SHA
import EnvConfig, Bybit, Xch, Ohlcv, Trade

"""Return sorted USDT bases with an existing shared OHLCV Arrow cache."""
function cachedbases(folder::AbstractString)::Vector{String}
    @assert isdir(folder) "OHLCV folder does not exist: $folder"
    return sort!([chop(pair; tail=5) for pair in readdir(folder)
        if endswith(pair, "-USDT") && isfile(joinpath(folder, pair, "ohlcv.arrow"))])
end

"""Assert sorted, unique, finite OHLCV data; existing historical gaps are allowed."""
function checkohlcv(df::AbstractDataFrame, base::AbstractString)::Nothing
    @assert nrow(df) > 0 "$base OHLCV is empty, rows=$(nrow(df))"
    @assert issorted(df.opentime) && allunique(df.opentime) "$base timestamps are not sorted and unique, rows=$(nrow(df))"
    @assert all(col -> all(isfinite, col), eachcol(select(df, Not(:opentime)))) "$base contains nonfinite OHLCV values, rows=$(nrow(df))"
    return nothing
end

"""
Combine old candles with fetched data, replacing only the old final candle.
Require that the fetched range includes that candle; never discard older history.
"""
function extendedohlcv(old::DataFrame, fetched::DataFrame, base::AbstractString)::DataFrame
    checkohlcv(fetched, base)
    nrow(old) == 0 && return fetched
    checkohlcv(old, base)
    @assert names(old) == names(fetched) "$base schema mismatch: $(names(old)) versus $(names(fetched))"
    endpoint = last(old.opentime)
    joinix = searchsortedfirst(fetched.opentime, endpoint)
    @assert joinix <= nrow(fetched) && fetched.opentime[joinix] == endpoint "$base download does not include cached endpoint=$endpoint; fetched=$(first(fetched.opentime)) to $(last(fetched.opentime)); cache unchanged"
    result = vcat(old[1:end-1, :], fetched[joinix:end, :]; cols=:orderequal)
    checkohlcv(result, base)
    return result
end

"""
Supplement one coin through `enddt`, or request two years for a newly discovered coin.
Use atomic Arrow replacement, verify persisted rows, and reject concurrent cache edits.
"""
function supplementcoin!(xc::Xch.XchCache, base::AbstractString, enddt::DateTime;
        newcoin::Bool=false)::Nothing
    path = EnvConfig.coinfile(base, "USDT", "ohlcv"; extension=".arrow")
    existed = isfile(path)
    digest = existed ? open(sha256, path) : nothing
    old = existed ? DataFrame(Arrow.Table(path)) : DataFrame()
    existed && checkohlcv(old, base)
    if existed && last(old.opentime) >= enddt && !newcoin
        println("$base already current through $(last(old.opentime))")
        return nothing
    end
    startdt = newcoin || !existed ? enddt - Year(2) : last(old.opentime)
    println("$(EnvConfig.now()) $base downloading $startdt to $enddt")
    flush(stdout)
    ohlcv = Ohlcv.defaultohlcv(base)
    Xch.cryptoupdate!(xc, ohlcv, startdt, enddt)
    fetched = select(Ohlcv.dataframe(ohlcv), Ohlcv.save_cols)
    if nrow(fetched) == 0
        @warn "No available Bybit history; cache unchanged" base startdt enddt
        Xch.removebase!(xc, base)
        return nothing
    end
    merged = extendedohlcv(old, fetched, base)
    if last(merged.opentime) < enddt
        @warn "Bybit history ends before requested cutoff" base requested=enddt received=last(merged.opentime)
    end
    staged = EnvConfig.writetable(merged, "ohlcv-supplement"; folderpath=dirname(path), format=:arrow)
    stored = DataFrame(Arrow.Table(staged))
    @assert isequal(stored, merged) "$base persisted OHLCV differs, rows=$(nrow(merged))"
    @assert existed ? (isfile(path) && open(sha256, path) == digest) : !isfile(path) "$base cache changed concurrently; staged=$staged"
    mv(staged, path; force=true)
    @assert isequal(DataFrame(Arrow.Table(path)), merged) "$base final OHLCV verification failed, path=$path"
    println("$base saved $(nrow(merged)) rows ($(nrow(merged)-nrow(old)) added), $(first(merged.opentime)) to $(last(merged.opentime))")
    flush(stdout)
    Xch.removebase!(xc, base)
    return nothing
end

"""Update cached coins, then discover and backfill new liquid Bybit USDT coins."""
function main(args::Vector{String}=ARGS)::Nothing
    if args in (["help"], ["--help"], ["-h"])
        println("Usage: julia --project=. scripts/supplement.jl\nUpdate all cached Bybit USDT coins, then discover liquid new coins with up to two years of history.\nPublic API only; no orders or classifiers. Ctrl+C stops; rerun to resume.")
        return nothing
    end
    @assert isempty(args) "unknown supplement arguments=$args; use --help"
    EnvConfig.init(EnvConfig.training)
    EnvConfig.setpairquote!("USDT")
    EnvConfig.setcoinspath!("Bybit")
    enddt = floor(Dates.now(UTC), Minute) - Minute(1)
    xc = Xch.XchCache(Bybit.BybitCache(false, "", ""); startdt=enddt)
    cached = cachedbases(EnvConfig.activecoinspath)
    println("Updating $(length(cached)) cached USDT coins through $enddt UTC")
    for (ix, base) in enumerate(cached)
        if !Xch.validbase(xc, base)
            @warn "Cached coin is unavailable on Bybit; unchanged" base
            continue
        end
        println("Cached coin $ix/$(length(cached)): $base")
        supplementcoin!(xc, base, enddt)
        GC.gc()
    end
    println("Discovering new liquid USDT coins")
    candidates = Trade.tradeselection!(xc; datetime=enddt, excludedbases=cached)
    newbases = sort!(collect(candidates[candidates.minquotevol .& candidates.continuousminvol, :basecoin]))
    println("New liquid coins ($(length(newbases))): $(join(newbases, ", "))")
    Xch.removeallbases(xc)
    for base in newbases
        supplementcoin!(xc, base, enddt; newcoin=true)
        GC.gc()
    end
    println("Supplementation complete. Existing feature/target/set caches are not rebuilt by this script.")
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        Supplement.main()
    catch error
        if error isa InterruptException
            println(stderr, "\nInterrupted. Completed coin caches are retained; rerun to continue.")
            exit(130)
        end
        rethrow()
    end
end
