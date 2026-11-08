#region TargetStore

# Per-coin target store `coins/<BASE-QUOTE>/targets/<targetsid>.arrow` with columns
# `opentime` and `label` (Int8 TradeLabel code) over the complete OHLCV history.
# Rows whose look-ahead horizon exceeds the available history are stored as `ignore`.

targetsfolder(base::AbstractString, quotecoin::AbstractString) = EnvConfig.coinfolderpath(base, quotecoin, "targets")

"Return the canonical config-only description of a target config that identifies its labels independent of any OHLCV base."
configdescription(trd::AbstractTargets)::String = error("configdescription not implemented for $(typeof(trd))")
configdescription(trd::Trend04)::String = "Trend04(minwindow=$(trd.minwindow),maxwindow=$(trd.maxwindow),holdanchormode=$(trd.holdanchormode),holdbehaviormode=$(trd.holdbehaviormode),longopen=$(trd.thres.longopen),longhold=$(trd.thres.longhold),shorthold=$(trd.thres.shorthold),shortopen=$(trd.thres.shortopen))"
configdescription(trd::TrendRegression)::String = "TrendRegression(window=$(trd.window),longthreshold=$(trd.longthreshold),shortthreshold=$(trd.shortthreshold))"
configdescription(trd::Trend05)::String = "Trend05(leadregr=$(trd.leadregr),supportregr=$(trd.supportregr),trendregr=$(trd.trendregr),triggerdist=$(trd.triggerdist),targetgain=$(trd.targetgain),targetminutes=$(trd.targetminutes),distance=(pivot-regry)/pivot)"
"""Identify bounded first-passage semantics independently of the OHLCV base."""
configdescription(trd::Trend07)::String = "Trend07(bandratio=$(repr(trd.bandratio)),openwindow=$(trd.openwindow),maxwindow=$(trd.maxwindow),windows=elapsed-inclusive,anchor=inclusive-invalidation,recent=rolling-5-full,priority=open-then-first-hit,extremes=preserve-established-opposite,split-minute-gaps)"

"""Return Trend07's maximum future dependency in elapsed minutes."""
lookaheadminutes(trd::Trend07)::Int = trd.maxwindow

"Return the look-ahead horizon in minutes of a target config with a bounded look-ahead."
lookaheadminutes(trd::AbstractTargets)::Int = error("lookaheadminutes not implemented for $(typeof(trd))")
lookaheadminutes(trd::Trend04)::Int = trd.maxwindow
lookaheadminutes(trd::TrendRegression)::Int = trd.window
lookaheadminutes(trd::Trend05)::Int = trd.targetminutes

"Return the number of leading labels of a based target config with `nrows` labels that do not depend on OHLCV data beyond its base."
finallabelcount(trd::AbstractTargets, nrows::Integer)::Int = error("finallabelcount not implemented for $(typeof(trd))")
finallabelcount(trd::Union{Trend04, TrendRegression}, nrows::Integer)::Int = max(0, nrows - lookaheadminutes(trd))
function finallabelcount(trd::Union{Trend05, Trend07}, nrows::Integer)::Int
    @assert size(trd.df, 1) == nrows "target df rows=$(size(trd.df, 1)) != label rows=$(nrows); target=$(typeof(trd))"
    return trd.lastfinalix
end

"Return an unbased Features006 with the regressions required by `trd` or `nothing` if `trd` is calculated from OHLCV only."
regressionfeatures(trd::AbstractTargets) = nothing
function regressionfeatures(trd::TrendRegression)
    f6 = Features.Features006()
    Features.addgrad!(f6, window=trd.window, offset=0)
    return f6
end
function regressionfeatures(trd::Trend05)
    f6 = Features.Features006()
    for window in unique((trd.leadregr, trd.supportregr, trd.trendregr))
        Features.addgrad!(f6, window=window, offset=0)
    end

    return f6
end

"""Prepare Trend07's bounded causal extremes; the recent slope is calculated internally."""
function regressionfeatures(trd::Trend07)
    f6 = Features.Features006()
    Features.addextremes!(f6;bandratio=trd.bandratio,maxwindow=trd.maxwindow)
    return f6
end

"""
    readtargets(base, quotecoin, targetsid; confighash)

Return the per-coin target store `targetsid` as (memory mapped) DataFrame with columns
`opentime, label` (Int8 TradeLabel codes) or `nothing` if not stored. Asserts that the stored
config hash equals `confighash`.
"""
function readtargets(base::AbstractString, quotecoin::AbstractString, targetsid::AbstractString; confighash::AbstractString)
    folder = targetsfolder(base, quotecoin)
    meta = EnvConfig.tablemetadata(targetsid; folderpath=folder)
    isnothing(meta) && return nothing
    @assert get(meta, "confighash", "") == confighash "target store $(targetsid) of $(base) has confighash=$(get(meta, "confighash", "")) but expected $(confighash) - the target config was changed without a new id"
    return EnvConfig.readdf(targetsid; folderpath=folder, format=:arrow)
end

"Return the Int8 label codes of a copy of the unbased target config `trd` for all rows of `ohlcv` (complete history or view) with `ignore` for rows that are not final."
function _targetcodes(trd::AbstractTargets, ohlcv::Ohlcv.OhlcvData)::Vector{Int8}
    trd = deepcopy(trd)
    f6 = regressionfeatures(trd)
    if isnothing(f6)
        setbase!(trd, ohlcv)
    else
        windows, stdwindows = Features.regressionwindows(f6)
        for window in windows
            Features.updateregression!(ohlcv, window; std=(window in stdwindows))
        end
        featureohlcv = trd isa Trend07 ? deepcopy(ohlcv) : ohlcv
        if trd isa Trend07
            Ohlcv.setdataframe!(featureohlcv, DataFrame(Ohlcv.dataframe(ohlcv); copycols=true))
        end
        Features.setbase!(f6, featureohlcv, usecache=true)
        setbase!(trd, f6)
    end
    ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    targetlabels = labels(trd, ot[begin], ot[end])
    @assert length(targetlabels) == length(ot) "targets of $(ohlcv.base): length(labels)=$(length(targetlabels)) != OHLCV rows=$(length(ot))"
    codes = Int8[tradelabelcode(label) for label in targetlabels]
    codes[(finallabelcount(trd, length(codes)) + 1):end] .= tradelabelcode(ignore)
    return codes
end

"""
Return the label codes of `trd` for the complete history of `ohlcv` by recalculating only a tail
of the stored codes `storedcodes`, which must cover a prefix of the OHLCV history.

Labels look ahead and Trend04 segments carry state, so the tail is recalculated from an anchor
before the last complete stored label. It is spliced in once the recalculated labels equal the
stored ones over one look-ahead horizon before that label, otherwise the anchor moves further back.
"""
function _extendedtargetcodes(trd::AbstractTargets, ohlcv::Ohlcv.OhlcvData, storedcodes::AbstractVector{<:Integer})::Vector{Int8}
    lookahead = lookaheadminutes(trd)
    lastcomplete = findlast(!=(tradelabelcode(ignore)), storedcodes)
    (isnothing(lastcomplete) || (lastcomplete <= 4 * lookahead)) && return _targetcodes(trd, ohlcv)
    lastix = lastindex(Ohlcv.dataframe(ohlcv)[!, :opentime])
    overlap = (lastcomplete - lookahead + 1):lastcomplete
    span = 4 * lookahead
    while lastcomplete - span > firstindex(storedcodes)
        anchor = lastcomplete - span
        tail = _targetcodes(trd, Ohlcv.ohlcvview(ohlcv, anchor:lastix))
        if all(tail[overlap .- (anchor - 1)] .== storedcodes[overlap])
            return vcat(Int8[code for code in view(storedcodes, firstindex(storedcodes):lastcomplete)], tail[(lastcomplete - anchor + 2):end])
        end
        (verbosity >= 2) && println("$(EnvConfig.now()) target tail of $(ohlcv.base) from row $(anchor) differs from stored labels; moving anchor back")
        span *= 2
    end
    return _targetcodes(trd, ohlcv)
end

# Trend05 labels before the first unresolved entry decision are final and the scan restarts without state at that row.
function _extendedtargetcodes(trd::Trend05, ohlcv::Ohlcv.OhlcvData, storedcodes::AbstractVector{<:Integer})::Vector{Int8}
    lastcomplete = findlast(!=(tradelabelcode(ignore)), storedcodes)
    isnothing(lastcomplete) && return _targetcodes(trd, ohlcv)
    lastix = lastindex(Ohlcv.dataframe(ohlcv)[!, :opentime])
    tail = _targetcodes(trd, Ohlcv.ohlcvview(ohlcv, (lastcomplete + 1):lastix))
    return vcat(Int8[code for code in view(storedcodes, firstindex(storedcodes):lastcomplete)], tail)
end

"""Recalculate unresolved Trend07 labels with exactly the bounded backward context."""
function _extendedtargetcodes(trd::Trend07, ohlcv::Ohlcv.OhlcvData,
        storedcodes::AbstractVector{<:Integer})::Vector{Int8}
    lastcomplete = findlast(!=(tradelabelcode(ignore)),storedcodes)
    isnothing(lastcomplete) && return _targetcodes(trd,ohlcv)
    firstrecalculate = lastcomplete+1
    anchor = max(1,firstrecalculate-max(trd.maxwindow-1,4))
    tail = _targetcodes(trd,Ohlcv.ohlcvview(ohlcv,anchor:nrow(Ohlcv.dataframe(ohlcv))))
    return vcat(Int8[code for code in view(storedcodes,1:lastcomplete)],
        tail[(firstrecalculate-anchor+1):end])
end

"""
    updatetargets!(ohlcv, targetsid, trd; confighash)

Return the per-coin target store `targetsid` after bringing it up to date with the complete
OHLCV history of `ohlcv`. `trd` is an unbased target config instance used for the calculation.
A store that matches the OHLCV history is extended by recalculating only its tail. Stored
coverage is reused only when the row count and first/last timestamps match the full history.
"""
function updatetargets!(ohlcv::Ohlcv.OhlcvData, targetsid::AbstractString, trd::AbstractTargets; confighash::AbstractString)
    odf = Ohlcv.dataframe(ohlcv)
    @assert !(odf isa SubDataFrame) "updatetargets! requires the complete OHLCV history of $(ohlcv.base), not a view"
    ot = odf[!, :opentime]
    @assert !isempty(ot) "target store $(targetsid) of $(ohlcv.base) requires non-empty OHLCV history"
    stored = readtargets(ohlcv.base, ohlcv.quotecoin, targetsid; confighash=confighash)
    expectedrows = length(ot)
    if !isnothing(stored) && (size(stored, 1) == expectedrows) &&
            (stored[begin, :opentime] == ot[begin]) && (stored[end, :opentime] == ot[end])
        return stored
    end
    extendable = !isnothing(stored) && (size(stored, 1) > 0) &&
        (size(stored, 1) < expectedrows) && (stored[begin, :opentime] == ot[begin]) &&
        (stored[end, :opentime] == ot[size(stored, 1)])
    codes = extendable ? _extendedtargetcodes(trd, ohlcv, stored[!, :label]) : _targetcodes(trd, ohlcv)
    @assert length(codes) == length(ot) "targets $(targetsid) of $(ohlcv.base): $(length(codes)) labels != OHLCV rows=$(length(ot))"
    targetdf = DataFrame(opentime=copy(ot), label=codes)
    @assert (size(targetdf, 1) == expectedrows) && (targetdf[begin, :opentime] == ot[begin]) && (targetdf[end, :opentime] == ot[end]) "target store $(targetsid) of $(ohlcv.base) covers $(size(targetdf, 1)) rows from $(targetdf[begin, :opentime]) to $(targetdf[end, :opentime]); expected $(expectedrows) rows from $(ot[begin]) to $(ot[end])"
    meta = Dict("confighash" => String(confighash), "labels" => join(string.(uniquelabels(trd)), ","))
    EnvConfig.savedf(targetdf, targetsid; folderpath=targetsfolder(ohlcv.base, ohlcv.quotecoin), format=:arrow, metadata=meta)
    (verbosity >= 3) && println("$(EnvConfig.now()) $(extendable ? "extended" : "calculated") target store $(targetsid) of $(ohlcv.base) to $(length(codes)) rows")
    return readtargets(ohlcv.base, ohlcv.quotecoin, targetsid; confighash=confighash)
end

#endregion TargetStore
