"""The five mutually exclusive Trend05 labels; allclose is no established trend, not abstention."""
const SIGNAL_LABELS = ("longopen", "longhold", "shortopen", "shorthold", "allclose")

"""Normalize supported enum/string labels; strong opens/holds share their normal signal."""
function _signalname(label)::String
    name = string(label)
    mapping = Dict("longstrongopen"=>"longopen", "longstronghold"=>"longhold",
        "shortstrongopen"=>"shortopen", "shortstronghold"=>"shorthold")
    name = get(mapping, name, name)
    @assert name in SIGNAL_LABELS || name == "ignore" "unsupported signal label=$(repr(label))"
    return name
end

"""
    signaldecision(label, score, threshold)
    signaldecision(probabilities::AbstractVector, labels::AbstractVector, threshold)

Apply the consumer's mutually exclusive argmax label + score rule, including
threshold equality. Return a normalized signal string or `missing` for abstention
(never allclose). Ignore winners abstain. Ties select the first label, like argmax.
Threshold can be a shared number or a Dict of normalized label => threshold.
"""
function signaldecision(label, score::Real, threshold)
    name = _signalname(label)
    name == "ignore" && return missing
    cutoff = threshold isa AbstractDict ? threshold[name] : threshold
    @assert isfinite(score) && 0 <= score <= 1 "score=$(score) must be within [0,1]"
    @assert isfinite(cutoff) && 0 <= cutoff <= 1 "threshold=$(cutoff) must be within [0,1]"
    return name != "ignore" && score >= cutoff ? name : missing
end

"""Select the argmax label and score, then apply `signaldecision`."""
function signaldecision(probabilities::AbstractVector{<:Real}, labels::AbstractVector, threshold)
    @assert length(probabilities) == length(labels) > 0 "probabilities=$(length(probabilities)), labels=$(length(labels)) must agree and be nonempty"
    @assert all(value -> isfinite(value) && 0 <= value <= 1, probabilities) "probabilities=$(probabilities) must be finite within [0,1]"
    winner = argmax(probabilities)
    return signaldecision(labels[winner], probabilities[winner], threshold)
end

"""Return a ratio with explicit missing for an unsupported denominator."""
_signalratio(numerator::Integer, denominator::Integer) = denominator == 0 ? missing : numerator / denominator

"""
Return percentile 95% precision/recall/coverage intervals from shared-calendar-block
resampling. All coins and minutes within a sampled block move jointly. Invalid
denominators are omitted per metric; no valid replicates yields missing bounds.
"""
function _signalblockintervals(blockcounts::AbstractVector, bootstrap::Integer,
    rng::AbstractRNG, minblocks::Integer)
    emptyresult = (; precision_lower=missing, precision_upper=missing,
        recall_lower=missing, recall_upper=missing, coverage_lower=missing,
        coverage_upper=missing, precision_bootstrap_valid=0,
        recall_bootstrap_valid=0, coverage_bootstrap_valid=0)
    (bootstrap == 0 || length(blockcounts) < minblocks) && return emptyresult
    precision, recall, coverage = Float64[], Float64[], Float64[]
    for _ in 1:bootstrap
        tp = fp = fn = eligible = 0
        for _ in eachindex(blockcounts)
            block = blockcounts[rand(rng, eachindex(blockcounts))]
            tp += block.tp
            fp += block.fp
            fn += block.fn
            eligible += block.eligible
        end
        tp+fp > 0 && push!(precision, tp/(tp+fp))
        tp+fn > 0 && push!(recall, tp/(tp+fn))
        eligible > 0 && push!(coverage, (tp+fp)/eligible)
    end
    bounds = values -> isempty(values) ? (missing, missing) : Tuple(quantile(values, [0.025, 0.975]))
    pl, pu = bounds(precision)
    rl, ru = bounds(recall)
    cl, cu = bounds(coverage)
    return (; precision_lower=pl, precision_upper=pu, recall_lower=rl,
        recall_upper=ru, coverage_lower=cl, coverage_upper=cu,
        precision_bootstrap_valid=length(precision), recall_bootstrap_valid=length(recall),
        coverage_bootstrap_valid=length(coverage))
end

"""Test whether a canonical label belongs to a single or grouped signal."""
function _insignal(label, signal::String)::Bool
    ismissing(label) && return false
    return signal == "long" ? label in ("longopen", "longhold") :
        signal == "short" ? label in ("shortopen", "shorthold") : label == signal
end

"""Validate report metadata and identify eligible rows; ignored targets and unknown regimes are excluded."""
function _signaleligible(predictions::AbstractDataFrame)::BitVector
    required = ["set", "regime", "foldid", "blockid", "coin", "opentime", "label", "target", "score"]
    @assert all(name -> name in names(predictions), required) "columns=$(names(predictions)) must include $(required)"
    @assert all(column -> !any(ismissing, predictions[!, column]), required) "report metadata contains missing values in columns=$(required)"
    targets = _signalname.(predictions.target)
    _signalname.(predictions.label)
    @assert all(score -> isfinite(score) && 0 <= score <= 1, predictions.score) "scores=$(predictions.score) must be finite within [0,1]"
    return (targets .!= "ignore") .& (string.(predictions.regime) .!= "unknown")
end

"""Return deterministic overall, regime, fold and regime/fold row groups separately for each set."""
function _signalgroups(predictions::AbstractDataFrame)
    groups = NamedTuple[]
    for set in unique(predictions.set)
        setmask = predictions.set .== set
        push!(groups, (; set, grouping="overall", regime=missing, foldid=missing, mask=setmask))
        for regime in unique(predictions.regime[setmask])
            push!(groups, (; set, grouping="regime", regime, foldid=missing,
                mask=setmask .& (predictions.regime .== regime)))
        end
        for foldid in unique(predictions.foldid[setmask])
            push!(groups, (; set, grouping="fold", regime=missing, foldid,
                mask=setmask .& (predictions.foldid .== foldid)))
        end
        for (regime, foldid) in unique(collect(zip(predictions.regime[setmask], predictions.foldid[setmask])))
            push!(groups, (; set, grouping="regime_fold", regime, foldid,
                mask=setmask .& (predictions.regime .== regime) .& (predictions.foldid .== foldid)))
        end
    end
    return groups
end

"""
    signalmetrics(predictions; thresholds=0.0:0.05:1.0, minblocks=2,
                  bootstrap=0, seed=1)::DataFrame

Minute-level raw one-vs-rest TP/FP/FN/TN, precision, recall and coverage for five
labels plus grouped long/short, by set overall/regime/fold/regime_fold. Input has
`set, regime, foldid, blockid, coin, opentime, label, target, score`; label/score
must already be argmax predictions. Enum and string labels are supported.

Ignore targets and unknown regimes never enter denominators. Exclusion and
abstention counts remain explicit. Coverage is issued signal / eligible minutes.
Independent support counts shared blocks, not coins or minutes; insufficient
support is flagged. Optional `bootstrap > 0` resamples whole shared block IDs
jointly across all coins with replacement, providing percentile 95% confidence
intervals (`precision_lower/upper`, `recall_lower/upper`, `coverage_lower/upper`).
Intervals are missing below `minblocks`, when bootstrap is disabled, or when
no replicate has a valid denominator; valid replicate counts are explicit.
`seed` controls a private RNG; minute-level resampling is never performed.
Empty denominators are missing. Tables contain micro counts, not macro averages.
"""
function signalmetrics(predictions::AbstractDataFrame; thresholds=0.0:0.05:1.0,
    minblocks::Integer=2, bootstrap::Integer=0, seed::Integer=1)::DataFrame
    @assert minblocks > 0 "minblocks=$(minblocks) must be positive"
    @assert bootstrap >= 0 "bootstrap=$(bootstrap) must be nonnegative"
    eligible = _signaleligible(predictions)
    targets = _signalname.(predictions.target)
    rows = NamedTuple[]
    rng = MersenneTwister(seed)
    for threshold in thresholds
        decisions = [signaldecision(row.label, row.score, threshold) for row in eachrow(predictions)]
        for group in _signalgroups(predictions)
            selected = group.mask .& eligible
            counteligible = count(selected)
            blockids = unique(predictions.blockid[selected])
            blocks = length(blockids)
            for signal in (SIGNAL_LABELS..., "long", "short")
                actual = _insignal.(targets[selected], Ref(signal))
                issued = _insignal.(decisions[selected], Ref(signal))
                tp, fp = count(actual .& issued), count(.!actual .& issued)
                fn, tn = count(actual .& .!issued), count(.!actual .& .!issued)
                blockcounts = bootstrap == 0 ? NamedTuple[] : [
                    (; tp=count(actual .& issued .& mask), fp=count(.!actual .& issued .& mask),
                        fn=count(actual .& .!issued .& mask), eligible=count(mask))
                    for blockid in blockids
                    for mask in [predictions.blockid[selected] .== blockid]]
                intervals = _signalblockintervals(blockcounts, bootstrap, rng, minblocks)
                push!(rows, merge((; group.set, group.grouping, group.regime, group.foldid,
                    signal, threshold, tp, fp, fn, tn, eligible=counteligible,
                    excluded=count(group.mask .& .!eligible), issued=tp+fp,
                    abstained=count(ismissing, decisions[selected]), targetcount=tp+fn,
                    precision=_signalratio(tp, tp+fp), recall=_signalratio(tp, tp+fn),
                    coverage=_signalratio(tp+fp, counteligible), blocks,
                    insufficient_support=blocks < minblocks, bootstrap, bootstrap_seed=seed,
                    confidence_level=0.95), intervals))
            end
        end
    end
    return DataFrame(rows)
end

"""Build contiguous positive intervals without bridging missing minutes, excluded rows or metadata boundaries."""
function _signalintervals(predictions::AbstractDataFrame, selected::AbstractVector{Bool},
    positive::AbstractVector{Bool})
    intervals = NamedTuple[]
    order = sortperm(1:nrow(predictions); by=ix ->
        (predictions.coin[ix], predictions.set[ix], predictions.opentime[ix]))
    current = Int[]
    previous = nothing
    for ix in order
        key = (predictions.coin[ix], predictions.set[ix], predictions.blockid[ix],
            predictions.regime[ix], predictions.foldid[ix])
        continuous = !isnothing(previous) && previous.key == key &&
            predictions.opentime[ix] - previous.time == Minute(1)
        if !selected[ix] || !positive[ix] || !continuous
            if !isempty(current)
                push!(intervals, (; key=previous.key, start=predictions.opentime[first(current)],
                    stop=predictions.opentime[last(current)], count=length(current)))
                empty!(current)
            end
        end
        selected[ix] && positive[ix] && push!(current, ix)
        previous = (; key, time=predictions.opentime[ix])
    end
    if !isempty(current)
        push!(intervals, (; key=previous.key, start=predictions.opentime[first(current)],
            stop=predictions.opentime[last(current)], count=length(current)))
    end
    return intervals
end

"""
    signaleventmetrics(predictions; thresholds=0.0:0.05:1.0, minblocks=2)::DataFrame

Target opportunities and prediction episodes are contiguous same-signal minutes
within coin/set/block/regime/fold, split at missing minutes and exclusions.
Exact event TP requires identical opportunity/episode boundaries (zero tolerance);
each interval matches at most once. Event precision/recall use those exact matches.

Separately report one-to-one opportunity detection by overlapping prediction episodes,
first detection delay (minutes), duplicate overlapping episodes, repeated issued
minutes and unmatched prediction episodes. These overlap diagnostics are not
exact-match TP and do not describe executed trades. Episode frequency is per
eligible minute; no independent-minute confidence intervals are supplied.
"""
function signaleventmetrics(predictions::AbstractDataFrame; thresholds=0.0:0.05:1.0, minblocks::Integer=2)::DataFrame
    @assert minblocks > 0 "minblocks=$(minblocks) must be positive"
    eligible = _signaleligible(predictions)
    @assert length(unique(collect(zip(predictions.coin, predictions.set, predictions.opentime)))) == nrow(predictions) "duplicate coin/set/minute rows=$(nrow(predictions))"
    targets = _signalname.(predictions.target)
    rows = NamedTuple[]
    for threshold in thresholds
        decisions = [signaldecision(row.label, row.score, threshold) for row in eachrow(predictions)]
        for group in _signalgroups(predictions), signal in (SIGNAL_LABELS..., "long", "short")
            selected = group.mask .& eligible
            actual = _insignal.(targets, Ref(signal))
            issued = _insignal.(decisions, Ref(signal))
            opportunities = _signalintervals(predictions, selected, actual)
            episodes = _signalintervals(predictions, selected, issued)
            exactkeys = Set((event.key, event.start, event.stop) for event in opportunities)
            tp = count(event -> (event.key, event.start, event.stop) in exactkeys, episodes)
            detected = 0
            delays = Float64[]
            matched = Set{Int}()
            overlaps = Set{Int}()
            for opportunity in opportunities
                overlapping = findall(episode -> episode.key == opportunity.key &&
                    episode.start <= opportunity.stop && episode.stop >= opportunity.start, episodes)
                union!(overlaps, overlapping)
                available = filter(ix -> !(ix in matched), overlapping)
                # Reserve exact matches first; otherwise use the earliest unused episode.
                exact = findfirst(ix -> episodes[ix].start == opportunity.start &&
                    episodes[ix].stop == opportunity.stop, available)
                candidate = isempty(available) ? nothing :
                    isnothing(exact) ? first(available) : available[exact]
                if !isnothing(candidate)
                    detected += 1
                    push!(matched, candidate)
                    firsttime = max(opportunity.start, episodes[candidate].start)
                    push!(delays, Dates.value(firsttime-opportunity.start) / 60000)
                end
            end
            blocks = length(unique(predictions.blockid[selected]))
            fp, fn = length(episodes)-tp, length(opportunities)-tp
            push!(rows, (; group.set, group.grouping, group.regime, group.foldid,
                signal, threshold, tp, fp, fn, opportunities=length(opportunities),
                episodes=length(episodes), unmatched_episodes=fp, detected,
                duplicate_episodes=length(setdiff(overlaps, matched)),
                detection_unmatched_episodes=length(episodes)-detected,
                repeated_signal_minutes=sum(event.count-1 for event in episodes; init=0),
                mean_delay_minutes=isempty(delays) ? missing : mean(delays),
                precision=_signalratio(tp, tp+fp), recall=_signalratio(tp, tp+fn),
                detection_rate=_signalratio(detected, length(opportunities)),
                episode_frequency=_signalratio(length(episodes), count(selected)),
                eligible=count(selected), excluded=count(group.mask .& .!eligible),
                blocks, insufficient_support=blocks < minblocks))
        end
    end
    return DataFrame(rows)
end
