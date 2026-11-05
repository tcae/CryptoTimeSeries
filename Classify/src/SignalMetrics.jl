"""The five mutually exclusive Trend05 labels; allclose is no established trend, not abstention."""
const SIGNAL_LABELS = ("longopen", "longhold", "shortopen", "shorthold", "allclose")

"""Normalize supported enum/string labels; strong opens/holds share their normal signal."""
function _signalname(label)::String
    name = string(label)
    name = name == "longstrongopen" ? "longopen" :
        name == "longstronghold" ? "longhold" :
        name == "shortstrongopen" ? "shortopen" :
        name == "shortstronghold" ? "shorthold" : name
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

"""Normalize the small label vocabulary once and encode rows; code six means ignore/abstention."""
function _signalcodes(labels::AbstractVector)::Vector{UInt8}
    codes = Dict(label => only(code for (code,name) in zip(
        (0x01,0x02,0x03,0x04,0x05,0x06), (SIGNAL_LABELS...,"ignore"))
        if name == _signalname(label)) for label in unique(labels))
    return [codes[label] for label in labels]
end

"""Validate report columns and return compact label codes and eligible-minute flags."""
function _signalinputs(predictions::AbstractDataFrame)
    required = ["set", "regime", "foldid", "blockid", "coin", "opentime", "label", "target", "score"]
    @assert all(name -> name in names(predictions), required) "columns=$(names(predictions)) must include $(required)"
    @assert all(column -> !any(ismissing, predictions[!, column]), required) "report metadata contains missing values in columns=$(required)"
    targets = _signalcodes(predictions.target)
    labels = _signalcodes(predictions.label)
    @assert all(score -> isfinite(score) && 0 <= score <= 1, predictions.score) "scores=$(predictions.score) must be finite within [0,1]"
    eligible = (targets .!= 0x06) .& (predictions.regime .!= "unknown")
    return (;targets,labels,eligible)
end

"""
Build one row-to-cell vector and small report groups, not full-row masks per group.
Cells distinguish set/regime/fold/block. Eligible blocks retain first-input-occurrence
order so seeded bootstrap draws are identical to the row-scanning implementation.
"""
function _signallayout(predictions::AbstractDataFrame, eligible::AbstractVector{Bool})
    partition = groupby(DataFrame(predictions;copycols=false),[:set,:regime,:foldid,:blockid];sort=false)
    cellids = groupindices(partition)
    cells = [let indices = parentindices(table)[1], row = first(indices)
        firsteligible = findfirst(ix -> eligible[ix],indices)
        (;set=predictions.set[row],regime=predictions.regime[row],
            foldid=predictions.foldid[row],blockid=predictions.blockid[row],
            eligible=count(ix -> eligible[ix],indices),
            excluded=count(ix -> !eligible[ix],indices),
            firsteligible=isnothing(firsteligible) ? typemax(Int) : indices[firsteligible])
    end for table in partition]
    groups = NamedTuple[]
    for set in unique(cell.set for cell in cells)
        setcells = findall(cell -> cell.set==set,cells)
        push!(groups, (; set, grouping="overall", regime=missing, foldid=missing, cells=setcells))
        for regime in unique(cells[ix].regime for ix in setcells)
            push!(groups, (; set, grouping="regime", regime, foldid=missing,
                cells=filter(ix -> cells[ix].regime==regime,setcells)))
        end
        for foldid in unique(cells[ix].foldid for ix in setcells)
            push!(groups, (; set, grouping="fold", regime=missing, foldid,
                cells=filter(ix -> cells[ix].foldid==foldid,setcells)))
        end
        for (regime,foldid) in unique((cells[ix].regime,cells[ix].foldid) for ix in setcells)
            push!(groups, (; set, grouping="regime_fold", regime, foldid,
                cells=filter(ix -> cells[ix].regime==regime && cells[ix].foldid==foldid,setcells)))
        end
    end
    reports = map(groups) do group
        active = sort(filter(ix -> cells[ix].eligible>0,group.cells);
            by=ix -> cells[ix].firsteligible)
        blockids = unique(cells[ix].blockid for ix in active)
        blockcells = [filter(ix -> cells[ix].blockid==blockid,active) for blockid in blockids]
        merge(group,(;blockcells,
            eligible=sum(cells[ix].eligible for ix in group.cells;init=0),
            excluded=sum(cells[ix].excluded for ix in group.cells;init=0)))
    end
    return (;cellids,cells,groups=reports)
end

"""Apply a fixed threshold to compact label codes without per-minute strings or dictionaries."""
function _signaldecisions!(decisions::Vector{UInt8}, labels::Vector{UInt8},
        scores::AbstractVector, threshold)::Nothing
    cutoffs = [threshold isa AbstractDict ? get(threshold,name,NaN) : threshold for name in SIGNAL_LABELS]
    for code in unique(labels)
        code==0x06 && continue
        cutoff = cutoffs[code]
        @assert isfinite(cutoff) && 0 <= cutoff <= 1 "threshold=$(cutoff) for label=$(SIGNAL_LABELS[code]) must be within [0,1]"
    end
    for ix in eachindex(labels)
        code = labels[ix]
        decisions[ix] = code!=0x06 && scores[ix]>=cutoffs[code] ? code : 0x06
    end
    return nothing
end

"""Count eligible target/winner pairs once per threshold into small cell confusion tables."""
function _signalconfusions!(counts::Array{Int,3}, layout::NamedTuple,
        targets::Vector{UInt8}, decisions::Vector{UInt8}, eligible::AbstractVector{Bool})::Nothing
    fill!(counts,0)
    for ix in eachindex(targets)
        eligible[ix] || continue
        counts[targets[ix],decisions[ix],layout.cellids[ix]] += 1
    end
    return nothing
end

"""Return one-vs-rest counts, including grouped long/short, from cell confusion tables."""
function _signalcellcounts(counts::Array{Int,3}, cellids::AbstractVector{Int}, signalix::Int)
    codes = signalix<=5 ? (signalix,) : signalix==6 ? (1,2) : (3,4)
    tp = fp = fn = eligible = abstained = 0
    for cell in cellids, issued in 1:6, target in 1:5
        n = counts[target,issued,cell]
        actual = target in codes
        positive = issued in codes
        tp += actual && positive ? n : 0
        fp += !actual && positive ? n : 0
        fn += actual && !positive ? n : 0
        eligible += n
        abstained += issued==6 ? n : 0
    end
    return (;tp,fp,fn,eligible,abstained)
end

"""
    signalmetrics(predictions; thresholds=0.0:0.05:1.0, minblocks=2,
                  bootstrap=0, seed=1, onprogress=nothing)::DataFrame

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
Rows are encoded once; each threshold accumulates cell confusion counts in one
pass. Bootstrap operates on small shared-block counts, never repeated minute scans.
`onprogress`, when supplied, receives phase, threshold, grouping, groupix, groups,
signal, blockix and blocks before each group/signal/block is calculated. It does not
change metric values or random sampling; callers should throttle expensive log output.
"""
function signalmetrics(predictions::AbstractDataFrame; thresholds=0.0:0.05:1.0,
    minblocks::Integer=2, bootstrap::Integer=0, seed::Integer=1,
    onprogress::Union{Nothing,Function}=nothing)::DataFrame
    @assert minblocks > 0 "minblocks=$(minblocks) must be positive"
    @assert bootstrap >= 0 "bootstrap=$(bootstrap) must be nonnegative"
    (;targets,labels,eligible) = _signalinputs(predictions)
    layout = _signallayout(predictions,eligible)
    decisions = similar(labels)
    counts = zeros(Int,5,6,length(layout.cells))
    rows = NamedTuple[]
    rng = MersenneTwister(seed)
    for threshold in thresholds
        _signaldecisions!(decisions,labels,predictions.score,threshold)
        _signalconfusions!(counts,layout,targets,decisions,eligible)
        groups = layout.groups
        for (groupix,group) in enumerate(groups)
            isnothing(onprogress) || onprogress((phase="minute metrics",threshold,
                grouping=group.grouping,groupix,groups=length(groups),signal="",blockix=0,blocks=0))
            counteligible = group.eligible
            blocks = length(group.blockcells)
            for (signalix,signal) in enumerate((SIGNAL_LABELS..., "long", "short"))
                isnothing(onprogress) || onprogress((phase="minute metrics",threshold,
                    grouping=group.grouping,groupix,groups=length(groups),signal,blockix=0,blocks))
                totals = _signalcellcounts(counts,group.cells,signalix)
                (;tp,fp,fn) = totals
                tn = counteligible-tp-fp-fn
                blockcounts = typeof(totals)[]
                if bootstrap>0
                    for (blockix,cellids) in enumerate(group.blockcells)
                        isnothing(onprogress) || onprogress((phase="minute block counts",threshold,
                            grouping=group.grouping,groupix,groups=length(groups),signal,blockix,blocks))
                        push!(blockcounts,_signalcellcounts(counts,cellids,signalix))
                    end
                end
                intervals = _signalblockintervals(blockcounts, bootstrap, rng, minblocks)
                push!(rows, merge((; group.set, group.grouping, group.regime, group.foldid,
                    signal, threshold, tp, fp, fn, tn, eligible=counteligible,
                    excluded=group.excluded, issued=tp+fp,
                    abstained=totals.abstained, targetcount=tp+fn,
                    precision=_signalratio(tp, tp+fp), recall=_signalratio(tp, tp+fn),
                    coverage=_signalratio(tp+fp, counteligible), blocks,
                    insufficient_support=blocks < minblocks, bootstrap, bootstrap_seed=seed,
                    confidence_level=0.95), intervals))
            end
        end
    end
    return DataFrame(rows)
end

"""Test a compact label code against a single or grouped signal."""
_insignalcode(code::UInt8, signalix::Int)::Bool =
    signalix<=5 ? code==signalix : signalix==6 ? code in (0x01,0x02) : code in (0x03,0x04)

"""Build positive position ranges inside one consecutive coin/set/regime/fold/block segment."""
function _signalruns(order::AbstractVector{Int}, positions::UnitRange{Int},
        codes::Vector{UInt8}, eligible::AbstractVector{Bool}, signalix::Int)::Vector{UnitRange{Int}}
    runs = UnitRange{Int}[]
    start = 0
    for position in positions
        ix = order[position]
        positive = eligible[ix] && _insignalcode(codes[ix],signalix)
        if positive && start==0
            start = position
        elseif !positive && start!=0
            push!(runs,start:(position-1))
            start = 0
        end
    end
    start!=0 && push!(runs,start:last(positions))
    return runs
end

"""
Match sorted, internally disjoint target/prediction intervals with a forward cursor.
An episode can overlap later opportunities but can be matched only once. Exact
matches cannot compete with another overlapping episode because episodes are disjoint.
"""
function _signaleventcounts(opportunities::Vector{UnitRange{Int}}, episodes::Vector{UnitRange{Int}})
    tp = detected = overlapping = repeated = delayminutes = 0
    cursor = 1
    lastmatched = lastoverlap = 0
    for opportunity in opportunities
        while cursor<=length(episodes) && last(episodes[cursor])<first(opportunity)
            cursor += 1
        end
        candidate = 0
        ix = cursor
        while ix<=length(episodes) && first(episodes[ix])<=last(opportunity)
            if ix>lastoverlap
                overlapping += 1
                lastoverlap = ix
            end
            candidate==0 && ix>lastmatched && (candidate=ix)
            episodes[ix]==opportunity && (tp+=1)
            ix += 1
        end
        if candidate!=0
            detected += 1
            lastmatched = candidate
            delayminutes += max(first(opportunity),first(episodes[candidate]))-first(opportunity)
        end
    end
    for episode in episodes
        repeated += length(episode)-1
    end
    return (;tp,opportunities=length(opportunities),episodes=length(episodes),
        detected,overlapping,repeated,delayminutes)
end

"""Reject duplicate coin/set/minute keys in the shared sorted order without a timestamp set."""
function _checksignalorder(order::AbstractVector{Int}, coins::AbstractVector,
        sets::AbstractVector, times::AbstractVector)::Nothing
    for position in 2:length(order)
        previous, current = order[position-1], order[position]
        @assert !(coins[previous]==coins[current] && sets[previous]==sets[current] &&
            times[previous]==times[current]) "duplicate coin/set/minute at coin=$(coins[current]) set=$(sets[current]) opentime=$(times[current])"
    end
    return nothing
end

"""Aggregate bounded chronological segments once per threshold; retain no universe-sized event lists."""
function _signaleventcells!(counts::AbstractMatrix, order::AbstractVector{Int},
        layout::NamedTuple, times::AbstractVector, coins::AbstractVector,
        targets::Vector{UInt8}, decisions::Vector{UInt8}, eligible::AbstractVector{Bool},
        threshold, onprogress::Union{Nothing,Function})::Nothing
    start = 1
    nextprogress = 100000
    while start<=length(order)
        stop = start
        row = order[start]
        cell = layout.cellids[row]
        while stop<length(order)
            previous, current = order[stop], order[stop+1]
            layout.cellids[current]==cell && coins[current]==coins[row] &&
                times[current]-times[previous]==Minute(1) || break
            stop += 1
        end
        if layout.cells[cell].eligible>0
            for signalix in 1:7
                opportunities = _signalruns(order,start:stop,targets,eligible,signalix)
                episodes = _signalruns(order,start:stop,decisions,eligible,signalix)
                addition = _signaleventcounts(opportunities,episodes)
                counts[signalix,cell] = map(+,counts[signalix,cell],addition)
            end
        end
        if !isnothing(onprogress) && (stop>=nextprogress || stop==length(order))
            onprogress((phase="event intervals",threshold,grouping="overall",groupix=1,
                groups=length(layout.groups),signal="all",blockix=stop,blocks=length(order)))
            nextprogress = stop+100000
        end
        start = stop+1
    end
    return nothing
end

"""
    signaleventmetrics(predictions; thresholds=0.0:0.05:1.0, minblocks=2,
                       onprogress=nothing)::DataFrame

Target opportunities and prediction episodes are contiguous same-signal minutes
within coin/set/block/regime/fold, split at missing minutes and exclusions.
Exact event TP requires identical opportunity/episode boundaries (zero tolerance);
each interval matches at most once. Event precision/recall use those exact matches.

Separately report one-to-one opportunity detection by overlapping prediction episodes,
first detection delay (minutes), duplicate overlapping episodes, repeated issued
minutes and unmatched prediction episodes. These overlap diagnostics are not
exact-match TP and do not describe executed trades. Episode frequency is per
eligible minute; no independent-minute confidence intervals are supplied.
Input is sorted at most once. Consecutive metadata segments bound temporary
interval storage; sorted interval walks and cell aggregation replace all-pairs
matching and repeated per-group input sorts.
`onprogress` uses the same status fields as `signalmetrics`, before each group/signal.
During interval accumulation, `blockix/blocks` report processed/total input rows.
"""
function signaleventmetrics(predictions::AbstractDataFrame; thresholds=0.0:0.05:1.0,
        minblocks::Integer=2, onprogress::Union{Nothing,Function}=nothing)::DataFrame
    @assert minblocks > 0 "minblocks=$(minblocks) must be positive"
    (;targets,labels,eligible) = _signalinputs(predictions)
    layout = _signallayout(predictions,eligible)
    columns = [:coin,:set,:opentime]
    order = issorted(predictions,columns) ? (1:nrow(predictions)) : sortperm(predictions,columns)
    _checksignalorder(order,predictions.coin,predictions.set,predictions.opentime)
    decisions = similar(labels)
    emptycounts = (;tp=0,opportunities=0,episodes=0,detected=0,overlapping=0,repeated=0,delayminutes=0)
    counts = fill(emptycounts,7,length(layout.cells))
    rows = NamedTuple[]
    for threshold in thresholds
        groups = layout.groups
        isnothing(onprogress) || onprogress((phase="event metrics",threshold,
            grouping="overall",groupix=1,groups=length(groups),signal="",blockix=0,blocks=0))
        _signaldecisions!(decisions,labels,predictions.score,threshold)
        fill!(counts,emptycounts)
        _signaleventcells!(counts,order,layout,predictions.opentime,predictions.coin,
            targets,decisions,eligible,threshold,onprogress)
        for (groupix,group) in enumerate(groups)
            isnothing(onprogress) || onprogress((phase="event metrics",threshold,
                grouping=group.grouping,groupix,groups=length(groups),signal="",blockix=0,blocks=0))
            for (signalix,signal) in enumerate((SIGNAL_LABELS..., "long", "short"))
                isnothing(onprogress) || onprogress((phase="event metrics",threshold,
                    grouping=group.grouping,groupix,groups=length(groups),signal,blockix=0,blocks=0))
                totals = emptycounts
                for cell in group.cells
                    totals = map(+,totals,counts[signalix,cell])
                end
                (;tp,opportunities,episodes,detected) = totals
                blocks = length(group.blockcells)
                fp, fn = episodes-tp, opportunities-tp
                push!(rows, (; group.set, group.grouping, group.regime, group.foldid,
                    signal, threshold, tp, fp, fn, opportunities,
                    episodes, unmatched_episodes=fp, detected,
                    duplicate_episodes=totals.overlapping-detected,
                    detection_unmatched_episodes=episodes-detected,
                    repeated_signal_minutes=totals.repeated,
                    mean_delay_minutes=detected==0 ? missing : totals.delayminutes/detected,
                    precision=_signalratio(tp, tp+fp), recall=_signalratio(tp, tp+fn),
                    detection_rate=_signalratio(detected, opportunities),
                    episode_frequency=_signalratio(episodes, group.eligible),
                    eligible=group.eligible, excluded=group.excluded,
                    blocks, insufficient_support=blocks < minblocks))
            end
        end
    end
    return DataFrame(rows)
end
