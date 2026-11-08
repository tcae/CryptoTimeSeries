"""
    Trend07(; bandratio::Real=0.01, openwindow::Int=120, maxwindow::Int=240, f6=nothing)

Bounded first-passage targets gated by causal minimum/maximum relative slopes.
Future pivot return must reach +/-bandratio within maxwindow elapsed minutes,
without first touching or crossing the current sample's fixed opposite anchor.
Opens additionally require a hit within openwindow minutes and the matching sign
of the existing five-sample rolling regression. Opens take priority over holds;
otherwise the first valid directional hit wins. Confirmation is not required.
Matching Features006 extremes are required; they are never modified. Minute gaps
split sequences. Incomplete trailing decisions remain nonfinal in reusable stores.
"""
mutable struct Trend07 <: AbstractTargets
    bandratio::Real
    openwindow::Int
    maxwindow::Int
    ohlcv::Union{Ohlcv.OhlcvData,Nothing}
    f6::Union{Features.Features006,Nothing}
    df::Union{DataFrame,Nothing}
    lastfinalix::Int

    function Trend07(; bandratio::Real=0.01, openwindow::Int=120,
            maxwindow::Int=240, f6::Union{Features.Features006,Nothing}=nothing)
        @assert isfinite(bandratio) && 0 < bandratio < 1 "require 0 < bandratio < 1; bandratio=$bandratio"
        @assert 2 <= openwindow <= maxwindow <= 14400 "require 2 <= openwindow <= maxwindow <= 14400; openwindow=$openwindow, maxwindow=$maxwindow"
        isnothing(f6) || _extremerequiredfeatures(bandratio,maxwindow,f6)
        return new(bandratio,openwindow,maxwindow,nothing,f6,nothing,0)
    end
end

"""Validate the four causal extreme inputs required by Trend07."""
function _extremerequiredfeatures(bandratio::Real, maxwindow::Int, f6::Features.Features006)
    available = Features.f6all(f6)
    required = NamedTuple[]
    for code in ("minnrg", "minage", "maxnrg", "maxage")
        index = findfirst(feature -> feature.f == code && hasproperty(feature, :b) &&
            feature.b == bandratio && feature.w == maxwindow && feature.o == 0, available)
        @assert !isnothing(index) "required feature=$code, bandratio=$bandratio, maxwindow=$maxwindow; available=$available"
        push!(required, available[index])
    end
    return required
end

"""Reject an OHLCV-only base: Trend07 requires prepared matching extreme features."""
function setbase!(trd::Trend07, ohlcv::Ohlcv.OhlcvData)
    throw(ArgumentError("Trend07 requires setbase!(trd, f6::Features006) with matching extreme inputs"))
end

"""Set causal extreme features as the target base and calculate bounded labels."""
function setbase!(trd::Trend07, f6::Features.Features006)
    _extremerequiredfeatures(trd.bandratio,trd.maxwindow,f6)
    @assert !isnothing(f6.ohlcv) "Trend07 requires based Features006; base=$(f6.ohlcv)"
    trd.f6 = f6
    trd.ohlcv = f6.ohlcv
    trd.df = nothing
    trd.lastfinalix = 0
    supplement!(trd)
    return nothing
end

"""Remove Trend07's base and calculated labels."""
function removebase!(trd::Trend07)
    trd.ohlcv = nothing
    trd.f6 = nothing
    trd.df = nothing
    trd.lastfinalix = 0
    return nothing
end

"""Find the first inclusive upper/lower price crossing using existing range summaries."""
function _trend07_firstcrossing(tree::Features.ExtremeSummaryTree, firstix::Int,
        lastix::Int, threshold::Real, upper::Bool, node::Int=1,
        low::Int=1, high::Int=tree.leafcount)::Int
    (high < firstix || low > lastix || tree.moments[node].count == 0) && return 0
    extremeix = upper ? tree.maximumix[node] : tree.minimumix[node]
    price = tree.pivot[extremeix]
    (upper ? price < threshold : price > threshold) && return 0
    low == high && return low
    middle = div(low+high,2)
    before = _trend07_firstcrossing(tree,firstix,lastix,threshold,upper,2node,low,middle)
    before != 0 && return before
    return _trend07_firstcrossing(tree,firstix,lastix,threshold,upper,2node+1,middle+1,high)
end

"""Recover an integer anchor coordinate from an exact feature age without rounding."""
function _trend07_anchor(ix::Int, age::Real)::Int
    anchor = searchsortedfirst(1:ix,ix-age)
    @assert 1 <= anchor <= ix && ix-anchor == age "invalid extreme age=$age at sample=$ix; anchor=$anchor"
    return anchor
end

"""Return the first valid gain hit, or zero when the fixed anchor is touched first."""
function _trend07_hit(tree::Features.ExtremeSummaryTree, ix::Int, lastix::Int,
        anchorprice::Real, bandratio::Real, direction::Int)::Int
    threshold = tree.pivot[ix]*(1+direction*bandratio)
    hit = _trend07_firstcrossing(tree,ix+1,lastix,threshold,direction == 1)
    hit == 0 && return 0
    invalid = _trend07_firstcrossing(tree,ix+1,hit,anchorprice,direction == -1)
    return invalid == 0 ? hit : 0
end

"""Choose opens before holds, then the earliest valid directional threshold hit."""
function _trend07_label(longhit::Int, shorthit::Int, opendeadline::Int,
        recentgrad::Real)::Tuple{TradeLabel,Int}
    longopens = longhit != 0 && longhit <= opendeadline && recentgrad > 0
    shortopens = shorthit != 0 && shorthit <= opendeadline && recentgrad < 0
    longopens && return (longopen,longhit)
    shortopens && return (shortopen,shorthit)
    if longhit != 0 && (shorthit == 0 || longhit < shorthit)
        return (longhold,longhit)
    elseif shorthit != 0 && (longhit == 0 || shorthit < longhit)
        return (shorthold,shorthit)
    end
    return (allclose,0)
end

"""Calculate one contiguous segment with bounded-memory overlapping forward summaries."""
function _trend07_sequence!(trd::Trend07, pivot::AbstractVector,
        minslopes::AbstractVector, minages::AbstractVector,
        maxslopes::AbstractVector, maxages::AbstractVector, offset::Int)
    _, recentgrads = Features.rollingregression(pivot,5)
    for chunkfirst in 1:trd.maxwindow:length(pivot)
        chunklast = min(length(pivot),chunkfirst+trd.maxwindow-1)
        tree = Features.ExtremeSummaryTree(view(pivot,
            chunkfirst:min(length(pivot),chunklast+trd.maxwindow)))
        for ix in chunkfirst:chunklast
            localix = ix-chunkfirst+1
            deadline = min(length(tree.pivot),localix+trd.maxwindow)
            minimumix = _trend07_anchor(ix,minages[ix])
            maximumix = _trend07_anchor(ix,maxages[ix])
            longhit = minslopes[ix] > 0 ? _trend07_hit(tree,localix,deadline,
                pivot[minimumix],trd.bandratio,1) : 0
            shorthit = maxslopes[ix] < 0 ? _trend07_hit(tree,localix,deadline,
                pivot[maximumix],trd.bandratio,-1) : 0
            recentgrad = ix >= 5 ? recentgrads[ix] : zero(recentgrads[ix])
            label, hit = _trend07_label(longhit,shorthit,localix+trd.openwindow,recentgrad)
            trd.df[offset+ix,:label] = label
            if hit != 0
                trd.df[offset+ix,:relgain] = (tree.pivot[hit]-pivot[ix])/pivot[ix]
            end
        end
    end
    return nothing
end

"""Recalculate independent minute sequences; only the final maxwindow tail is nonfinal."""
function supplement!(trd::Trend07)
    @assert !isnothing(trd.ohlcv) && !isnothing(trd.f6) "Trend07 requires based features; base=$(trd.ohlcv), f6=$(trd.f6)"
    @assert isfinite(trd.bandratio) && 0 < trd.bandratio < 1 "invalid bandratio=$(trd.bandratio)"
    @assert 2 <= trd.openwindow <= trd.maxwindow <= 14400 "invalid windows: openwindow=$(trd.openwindow), maxwindow=$(trd.maxwindow)"
    @assert Features.issupplementedcurrent(trd.f6) "Trend07 requires current features; base=$(trd.ohlcv.base)"
    odf = Ohlcv.dataframe(trd.ohlcv)
    fdfno = trd.f6.fdfno
    @assert fdfno.opentime == odf.opentime "Trend07 feature/OHLCV timestamps differ; features=$(nrow(fdfno)), OHLCV=$(nrow(odf))"
    columns = _extremerequiredfeatures(trd.bandratio,trd.maxwindow,trd.f6)
    inputs = [fdfno[!,Features.fdfnocol(trd.f6,column)] for column in columns]
    trd.df = DataFrame(opentime=copy(odf.opentime),label=fill(allclose,nrow(odf)),
        relgain=zeros(Float32,nrow(odf)))
    sequences = Features.minutesequences(odf.opentime)
    for rows in sequences
        _trend07_sequence!(trd,view(odf.pivot,rows),
            (view(values,rows) for values in inputs)...,first(rows)-1)
    end
    trd.lastfinalix = isempty(sequences) ? 0 :
        max(first(last(sequences))-1,nrow(odf)-trd.maxwindow)
    return nothing
end

"""Return Trend07's five supported labels."""
uniquelabels(::Trend07) = [longopen,longhold,shortopen,shorthold,allclose]

"""Describe Trend07's base and bounded target parameters."""
function describe(trd::Trend07)
    base = isnothing(trd.ohlcv) ? "Base?" : trd.ohlcv.base
    return "Trend07_$(base)_bandratio=$(repr(trd.bandratio))_openwindow=$(trd.openwindow)_maxwindow=$(trd.maxwindow)"
end
