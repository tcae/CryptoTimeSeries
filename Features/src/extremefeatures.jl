"""
Return minimum/maximum anchor indices after causally processing an inclusive pivot segment.
Direction zero seeds both candidates; +/-1 denotes the active maximum/minimum respectively.
Only a strictly threshold-sized opposite move changes direction.
"""
function extremeanchors(pivot::AbstractVector, firstix::Int, lastix::Int, bandratio::Real)
    state = ExtremeState(firstix, firstix, 0)
    for ix in (firstix + 1):lastix
        advanceextreme!(state, pivot, ix, bandratio)
    end
    return state.minimumix, state.maximumix
end

"""Mutable anchor state for one causal extreme sequence."""
mutable struct ExtremeState
    minimumix::Int
    maximumix::Int
    direction::Int
end

"""
Return chronological confirmed extreme events.
An event contains its pivot `index`, `direction` (+1 maximum, -1 minimum), and
`confirmedix`. Confirmation occurs only on a qualifying opposite move, without a
deadline. The initial seed is not emitted as a confirmed extreme.
With `sequenceended=true`, append the final established opposite extreme if its
distance from the latest confirmed extreme strictly exceeds bandratio. Its
`confirmedix` is the sequence end, representing boundary acceptance, not a reversal.
"""
function confirmedextremes(pivot::AbstractVector; bandratio::Real, sequenceended::Bool=false)
    @assert !isempty(pivot) "confirmed extremes require samples; length=$(length(pivot))"
    @assert isfinite(bandratio) && 0 < bandratio < 1 "require 0 < bandratio < 1; bandratio=$bandratio"
    events = NamedTuple{(:index, :direction, :confirmedix), Tuple{Int, Int, Int}}[]
    state = ExtremeState(1, 1, 0)
    for ix in eachindex(pivot)
        @assert isfinite(pivot[ix]) && pivot[ix] > 0 "require positive finite pivot; index=$ix, pivot=$(pivot[ix])"
        previousdirection = state.direction
        previousanchor = previousdirection == -1 ? state.minimumix : state.maximumix
        advanceextreme!(state, pivot, ix, bandratio)
        if previousdirection != 0 && state.direction != previousdirection
            push!(events, (index=previousanchor, direction=previousdirection, confirmedix=ix))
        end
    end
    if sequenceended && !isempty(events)
        anchor = state.direction == -1 ? state.minimumix : state.maximumix
        previous = events[end]
        distance = state.direction * (pivot[anchor] - pivot[previous.index])
        if anchor > previous.index && state.direction == -previous.direction &&
                distance > bandratio * pivot[previous.index]
            push!(events, (index=anchor, direction=state.direction, confirmedix=lastindex(pivot)))
        end
    end
    return events
end

"""
Process one pivot, preserving the established opposite anchor during continuation.
A qualifying reversal starts the new active extreme at the crossing sample.
"""
function advanceextreme!(state::ExtremeState, pivot::AbstractVector, ix::Int, bandratio::Real)
    price = pivot[ix]
    if state.direction == 0
        price < pivot[state.minimumix] && (state.minimumix = ix)
        price > pivot[state.maximumix] && (state.maximumix = ix)
        if price - pivot[state.minimumix] > bandratio * pivot[state.minimumix]
            state.direction = 1
        elseif pivot[state.maximumix] - price > bandratio * pivot[state.maximumix]
            state.direction = -1
        end
    elseif state.direction == -1
        if price < pivot[state.minimumix]
            state.minimumix = ix
        elseif price - pivot[state.minimumix] > bandratio * pivot[state.minimumix]
            state.direction = 1
            state.maximumix = ix
        end
    else
        if price > pivot[state.maximumix]
            state.maximumix = ix
        elseif pivot[state.maximumix] - price > bandratio * pivot[state.maximumix]
            state.direction = -1
            state.minimumix = ix
        end
    end
    return state
end

"""
Range summaries for exact bounded-window replay and centered regression queries.
Band-contained and monotone stretches can be processed without visiting every pivot.
"""
struct ExtremeSummaryTree{V<:AbstractVector}
    pivot::V
    leafcount::Int
    moments::Vector{RegressionMoments}
    minimumix::Vector{Int}
    maximumix::Vector{Int}
    ascending::BitVector
    descending::BitVector
end

"""Build adjacent-segment summaries, preserving the earliest index on price ties."""
function ExtremeSummaryTree(pivot::AbstractVector)
    leafcount = nextpow(2, length(pivot))
    moments = fill(RegressionMoments(), 2 * leafcount)
    minimumix = zeros(Int, 2 * leafcount)
    maximumix = zeros(Int, 2 * leafcount)
    ascending = trues(2 * leafcount)
    descending = trues(2 * leafcount)
    for ix in eachindex(pivot)
        node = leafcount + ix - 1
        moments[node] = appendprice(RegressionMoments(), pivot[ix])
        minimumix[node] = maximumix[node] = ix
    end
    for node in (leafcount - 1):-1:1
        left, right = 2 * node, 2 * node + 1
        moments[node] = mergeregression(moments[left], moments[right])
        if moments[right].count == 0
            minimumix[node] = minimumix[left]
            maximumix[node] = maximumix[left]
            ascending[node] = ascending[left]
            descending[node] = descending[left]
        else
            minimumix[node] = pivot[minimumix[left]] <= pivot[minimumix[right]] ? minimumix[left] : minimumix[right]
            maximumix[node] = pivot[maximumix[left]] >= pivot[maximumix[right]] ? maximumix[left] : maximumix[right]
            levelstart = prevpow(2, node)
            nodewidth = div(leafcount, levelstart)
            rightfirst = (node - levelstart) * nodewidth + div(nodewidth, 2) + 1
            ascending[node] = ascending[left] && ascending[right] && pivot[rightfirst - 1] <= pivot[rightfirst]
            descending[node] = descending[left] && descending[right] && pivot[rightfirst - 1] >= pivot[rightfirst]
        end
    end
    return ExtremeSummaryTree(pivot, leafcount, moments, minimumix, maximumix, ascending, descending)
end

"""Query centered regression moments for an inclusive range without raw prefix subtraction."""
function rangeregression(tree::ExtremeSummaryTree, firstix::Int, lastix::Int)
    left, right = firstix + tree.leafcount - 1, lastix + tree.leafcount - 1
    before, after = RegressionMoments(), RegressionMoments()
    while left <= right
        if isodd(left)
            before = mergeregression(before, tree.moments[left])
            left += 1
        end
        if iseven(right)
            after = mergeregression(tree.moments[right], after)
            right -= 1
        end
        left = div(left, 2)
        right = div(right, 2)
    end
    return mergeregression(before, after)
end

"""Replay a range in time order, skipping summaries that cannot contain a state transition."""
function replayextremes!(state::ExtremeState, tree::ExtremeSummaryTree, bandratio::Real,
        firstix::Int, lastix::Int, node::Int=1, low::Int=1, high::Int=tree.leafcount)
    (high < firstix || low > lastix) && return state
    pivot = tree.pivot
    if firstix <= low && high <= lastix
        minimumix, maximumix = tree.minimumix[node], tree.maximumix[node]
        minprice, maxprice = pivot[minimumix], pivot[maximumix]
        if state.direction == 0
            combinedmin = min(minprice, pivot[state.minimumix])
            combinedmax = max(maxprice, pivot[state.maximumix])
            if combinedmax - combinedmin <= bandratio * combinedmin
                minprice < pivot[state.minimumix] && (state.minimumix = minimumix)
                maxprice > pivot[state.maximumix] && (state.maximumix = maximumix)
                return state
            end
        elseif state.direction == -1
            if minprice >= pivot[state.minimumix] && maxprice - pivot[state.minimumix] <= bandratio * pivot[state.minimumix]
                return state
            elseif tree.descending[node] && pivot[low] <= pivot[state.minimumix] && minprice < pivot[state.minimumix]
                state.minimumix = minimumix
                return state
            end
        else
            if maxprice <= pivot[state.maximumix] && pivot[state.maximumix] - minprice <= bandratio * pivot[state.maximumix]
                return state
            elseif tree.ascending[node] && pivot[low] >= pivot[state.maximumix] && maxprice > pivot[state.maximumix]
                state.maximumix = maximumix
                return state
            end
        end
    end
    low == high && return advanceextreme!(state, pivot, low, bandratio)
    middle = div(low + high, 2)
    replayextremes!(state, tree, bandratio, firstix, lastix, 2 * node, low, middle)
    replayextremes!(state, tree, bandratio, firstix, lastix, 2 * node + 1, middle + 1, high)
    return state
end

"""
Calculate bounded, causal minimum/maximum relative slopes and ages from positive pivots.
Window replay makes expired seeds and reestablished anchors independent of older history.
Float64 centered regression arithmetic is shared with `rollingregression`.
"""
function extremesfeatures(pivot::AbstractVector; bandratio::Real, maxwindow::Integer)
    @assert isfinite(bandratio) && 0 < bandratio < 1 "require 0 < bandratio < 1; bandratio=$bandratio"
    @assert maxwindow > 1 "require maxwindow > 1; maxwindow=$maxwindow"
    @assert !isempty(pivot) "extreme features require samples; length=$(length(pivot))"
    for ix in eachindex(pivot)
        @assert isfinite(pivot[ix]) && pivot[ix] > 0 "require positive finite pivot; index=$ix, pivot=$(pivot[ix])"
    end
    minnrg = zeros(Float32, length(pivot))
    maxnrg = similar(minnrg)
    minage = similar(minnrg)
    maxage = similar(minnrg)
    # Overlapping chunks bound summary memory by maxwindow, not the full coin history.
    for chunkfirst in firstindex(pivot):maxwindow:lastindex(pivot)
        chunklast = min(lastindex(pivot), chunkfirst + maxwindow - 1)
        historyfirst = max(firstindex(pivot), chunkfirst - maxwindow + 1)
        tree = ExtremeSummaryTree(view(pivot, historyfirst:chunklast))
        for ix in chunkfirst:chunklast
            localix = ix - historyfirst + 1
            firstix = max(1, localix - maxwindow + 1)
            state = ExtremeState(firstix, firstix, 0)
            replayextremes!(state, tree, bandratio, firstix + 1, localix)
            minimumix, maximumix = state.minimumix, state.maximumix
            minimumstats = rangeregression(tree, minimumix, localix)
            maximumstats = rangeregression(tree, maximumix, localix)
            minnrg[ix] = regressionresult(minimumstats).grad / minimumstats.mean
            maxnrg[ix] = regressionresult(maximumstats).grad / maximumstats.mean
            minage[ix] = localix - minimumix
            maxage[ix] = localix - maximumix
        end
    end
    @assert all(isfinite, minnrg) && all(isfinite, maxnrg) "extreme slopes must be finite; bandratio=$bandratio, maxwindow=$maxwindow, minrange=$(extrema(minnrg)), maxrange=$(extrema(maxnrg))"
    return (; minnrg, minage, maxnrg, maxage)
end

"""
Return consecutive one-minute ranges, splitting at missing samples.
Reject duplicate, reversed, or non-minute-aligned timestamps rather than hiding invalid data.
"""
function minutesequences(times::AbstractVector{DateTime})::Vector{UnitRange{Int}}
    ranges = UnitRange{Int}[]
    isempty(times) && return ranges
    firstix = 1
    for ix in 2:length(times)
        elapsed = Dates.value(times[ix] - times[ix - 1])
        @assert elapsed > 0 && elapsed % 60000 == 0 "require increasing minute-aligned timestamps; previous=$(times[ix - 1]), current=$(times[ix]), elapsed_ms=$elapsed"
        if elapsed != 60000
            push!(ranges, firstix:(ix - 1))
            firstix = ix
        end
    end
    push!(ranges, firstix:length(times))
    return ranges
end

"""Materialize all four columns, resetting anchors and regressions at missing-minute gaps."""
function _extremes!(fdfno::DataFrame, f6::Features006, feature, odf::AbstractDataFrame)
    fdfnocol(f6, feature) in names(fdfno) && return fdfno
    @assert f6.ohlcv.interval == "1m" "extreme features require minute OHLCV; interval=$(f6.ohlcv.interval)"
    codes = ("minnrg", "minage", "maxnrg", "maxage")
    columns = [zeros(Float32, size(odf, 1)) for _ in codes]
    for rows in minutesequences(odf[!, :opentime])
        values = extremesfeatures(view(odf[!, :pivot], rows); bandratio=feature.b, maxwindow=feature.w)
        for (ix, code) in enumerate(codes)
            columns[ix][rows] = getproperty(values, Symbol(code))
        end
    end
    for (ix, code) in enumerate(codes)
        column = (f=code, b=feature.b, w=feature.w, o=0)
        fdfno[:, fdfnocol(f6, column)] = columns[ix]
    end
    return fdfno
end
