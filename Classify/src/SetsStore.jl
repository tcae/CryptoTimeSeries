#region SetsStore

# Per-coin sets store `coins/<BASE-QUOTE>/sets/<setsid>.arrow` with columns `opentime`, `set`
# (train/test/eval) and coin-local `rangeid`. Only liquid OHLCV rows within the configured date
# range that belong to a set partition are stored; gaps between partitions are omitted.
# `rangeid` = liquidity range index * RANGEID_SUBRANGE_SPAN + subrange index within that liquidity range.

setsfolder(base::AbstractString, quotecoin::AbstractString) = EnvConfig.coinfolderpath(base, quotecoin, "sets")

"""
    readsets(base, quotecoin, setsid; confighash)

Return the per-coin sets store `setsid` as (memory mapped) DataFrame or `nothing` if not stored.
Asserts that the stored config hash equals `confighash`.
"""
function readsets(base::AbstractString, quotecoin::AbstractString, setsid::AbstractString; confighash::AbstractString)
    folder = setsfolder(base, quotecoin)
    meta = EnvConfig.tablemetadata(setsid; folderpath=folder)
    isnothing(meta) && return nothing
    @assert get(meta, "confighash", "") == confighash "sets store $(setsid) of $(base) has confighash=$(get(meta, "confighash", "")) but expected $(confighash) - the partition or date range config was changed without a new id"
    return EnvConfig.readdf(setsid; folderpath=folder, format=:arrow)
end

"""
Return the liquid row ranges of the complete OHLCV history of `ohlcv` clipped to `startix:endix`.
Liquidity parameters are scaled down for histories that are shorter than the defaults assume.
"""
function _liquidranges(ohlcv::Ohlcv.OhlcvData, startix::Integer, endix::Integer)
    history_minutes = size(Ohlcv.dataframe(ohlcv), 1)
    checkperiod = min(Ohlcv.ld.checkperiod, max(15, fld(history_minutes, 4)))
    rv = Ohlcv.liquiditycheck(
        ohlcv;
        minquotevol=Ohlcv.ld.minquotevol,
        accumulate=min(Ohlcv.ld.accumulate, checkperiod),
        checkperiod=checkperiod,
        startthreshold=Ohlcv.ld.startthreshold,
        stopthreshold=Ohlcv.ld.stopthreshold,
        minliquidminutes=min(Ohlcv.ld.minliquidminutes, max(1, history_minutes - checkperiod + 1)),
        startdistance=min(Ohlcv.ld.startdistance, max(0, history_minutes - checkperiod)),
    )
    clipped = [max(first(rng), startix):min(last(rng), endix) for rng in rv]
    return filter(rng -> length(rng) > 0, clipped)
end

"""
    updatesets!(ohlcv, setsid, partitionconfig, startdt, enddt; confighash)

Return the per-coin sets store `setsid` after (re)building it from the liquid ranges of the
complete OHLCV history of `ohlcv` within `startdt..enddt`, partitioned according to
`partitionconfig` (NamedTuple with `samplesets, partitionsize, gapsize, minpartitionsize, maxpartitionsize`).
The store is reused as long as the OHLCV coverage of the date range is unchanged.
"""
function updatesets!(ohlcv::Ohlcv.OhlcvData, setsid::AbstractString, partitionconfig::NamedTuple, startdt::DateTime, enddt::DateTime; confighash::AbstractString)
    @assert startdt < enddt "startdt=$(startdt) must be < enddt=$(enddt)"
    odf = Ohlcv.dataframe(ohlcv)
    @assert !(odf isa SubDataFrame) "updatesets! requires the complete OHLCV history of $(ohlcv.base), not a view"
    ot = odf[!, :opentime]
    startix = searchsortedfirst(ot, startdt)
    endix = searchsortedlast(ot, enddt)
    coverage = endix >= startix ? "$(ot[startix])..$(ot[endix])" : "empty"
    folder = setsfolder(ohlcv.base, ohlcv.quotecoin)
    stored = readsets(ohlcv.base, ohlcv.quotecoin, setsid; confighash=confighash)
    if !isnothing(stored) && (EnvConfig.tablemetadata(setsid; folderpath=folder)["coverage"] == coverage)
        return stored
    end
    setot = DateTime[]
    setnames = String[]
    rangeids = Int[]
    if endix > startix
        for (liqix, liqrange) in enumerate(_liquidranges(ohlcv, startix, endix))
            (length(liqrange) > 1) || continue
            psets = setpartitions(liqrange, partitionconfig.samplesets;
                partitionsize=partitionconfig.partitionsize,
                gapsize=partitionconfig.gapsize,
                minpartitionsize=partitionconfig.minpartitionsize,
                maxpartitionsize=partitionconfig.maxpartitionsize)
            @assert length(psets) < RANGEID_SUBRANGE_SPAN "liquidity range $(liqix) of $(ohlcv.base) produced $(length(psets)) subranges, which does not fit below RANGEID_SUBRANGE_SPAN=$(RANGEID_SUBRANGE_SPAN)"
            for (subix, (setname, psrange)) in enumerate(psets)
                append!(setot, view(ot, psrange))
                append!(setnames, fill(string(setname), length(psrange)))
                append!(rangeids, fill(liqix * Int(RANGEID_SUBRANGE_SPAN) + subix - 1, length(psrange)))
            end
        end
    end
    meta = Dict("confighash" => String(confighash), "coverage" => coverage)
    EnvConfig.savedf(DataFrame(opentime=setot, set=setnames, rangeid=rangeids), setsid; folderpath=folder, format=:arrow, metadata=meta)
    (verbosity >= 3) && println("$(EnvConfig.now()) updated sets store $(setsid) of $(ohlcv.base) with $(length(setot)) rows covering $(coverage)")
    return readsets(ohlcv.base, ohlcv.quotecoin, setsid; confighash=confighash)
end

#endregion SetsStore
