"""Return the original first-occurrence report groups with independent row masks."""
function referencesignalgroups(df)
    groups = NamedTuple[]
    for set in unique(df.set)
        mask = df.set .== set
        push!(groups,(;set,grouping="overall",regime=missing,foldid=missing,mask))
        for regime in unique(df.regime[mask])
            push!(groups,(;set,grouping="regime",regime,foldid=missing,
                mask=mask .& (df.regime .== regime)))
        end
        for foldid in unique(df.foldid[mask])
            push!(groups,(;set,grouping="fold",regime=missing,foldid,
                mask=mask .& (df.foldid .== foldid)))
        end
        for (regime,foldid) in unique(collect(zip(df.regime[mask],df.foldid[mask])))
            push!(groups,(;set,grouping="regime_fold",regime,foldid,
                mask=mask .& (df.regime .== regime) .& (df.foldid .== foldid)))
        end
    end
    return groups
end

"""Compute the pre-optimization row-scanning minute oracle, including seeded bootstrap intervals."""
function referencesignalmetrics(df;thresholds,bootstrap=0,seed=1,minblocks=2)
    targets = Classify._signalname.(df.target)
    eligible = (targets .!= "ignore") .& (df.regime .!= "unknown")
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for threshold in thresholds
        decisions = [Classify.signaldecision(row.label,row.score,threshold) for row in eachrow(df)]
        for group in referencesignalgroups(df)
            selected = group.mask .& eligible
            blocks = unique(df.blockid[selected])
            for signal in (Classify.SIGNAL_LABELS...,"long","short")
                actual = Classify._insignal.(targets[selected],Ref(signal))
                issued = Classify._insignal.(decisions[selected],Ref(signal))
                tp, fp = count(actual .& issued),count(.!actual .& issued)
                fn, tn = count(actual .& .!issued),count(.!actual .& .!issued)
                blockcounts = [(;tp=count(actual .& issued .& mask),
                    fp=count(.!actual .& issued .& mask),fn=count(actual .& .!issued .& mask),
                    eligible=count(mask)) for block in blocks for mask in [df.blockid[selected].==block]]
                intervals = Classify._signalblockintervals(blockcounts,bootstrap,rng,minblocks)
                push!(rows,merge((;group.set,group.grouping,group.regime,group.foldid,
                    signal,threshold,tp,fp,fn,tn,eligible=count(selected),
                    excluded=count(group.mask .& .!eligible),issued=tp+fp,
                    abstained=count(ismissing,decisions[selected]),targetcount=tp+fn,
                    precision=Classify._signalratio(tp,tp+fp),recall=Classify._signalratio(tp,tp+fn),
                    coverage=Classify._signalratio(tp+fp,count(selected)),blocks=length(blocks),
                    insufficient_support=length(blocks)<minblocks,bootstrap,bootstrap_seed=seed,
                    confidence_level=0.95),intervals))
            end
        end
    end
    return DataFrame(rows)
end

"""Build original timestamp intervals by sorting and filtering the complete input for every group."""
function referencesignalintervals(df,selected,positive)
    result = NamedTuple[]
    current = Int[]
    previous = nothing
    for ix in sortperm(1:nrow(df);by=ix -> (df.coin[ix],df.set[ix],df.opentime[ix]))
        key = (df.coin[ix],df.set[ix],df.blockid[ix],df.regime[ix],df.foldid[ix])
        continuous = !isnothing(previous) && previous.key==key &&
            df.opentime[ix]-previous.time==Minute(1)
        if !selected[ix] || !positive[ix] || !continuous
            if !isempty(current)
                push!(result,(;key=previous.key,start=df.opentime[first(current)],
                    stop=df.opentime[last(current)],count=length(current)))
                empty!(current)
            end
        end
        selected[ix] && positive[ix] && push!(current,ix)
        previous = (;key,time=df.opentime[ix])
    end
    if !isempty(current)
        push!(result,(;key=previous.key,start=df.opentime[first(current)],
            stop=df.opentime[last(current)],count=length(current)))
    end
    return result
end

"""Compute the pre-optimization all-pairs event-matching oracle."""
function referencesignalevents(df;thresholds,minblocks=2)
    targets = Classify._signalname.(df.target)
    eligible = (targets .!= "ignore") .& (df.regime .!= "unknown")
    rows = NamedTuple[]
    for threshold in thresholds
        decisions = [Classify.signaldecision(row.label,row.score,threshold) for row in eachrow(df)]
        for group in referencesignalgroups(df), signal in (Classify.SIGNAL_LABELS...,"long","short")
            selected = group.mask .& eligible
            opportunities = referencesignalintervals(df,selected,Classify._insignal.(targets,Ref(signal)))
            episodes = referencesignalintervals(df,selected,Classify._insignal.(decisions,Ref(signal)))
            tp = count(episode -> any(opportunity -> episode.key==opportunity.key &&
                episode.start==opportunity.start && episode.stop==opportunity.stop,opportunities),episodes)
            matched, overlaps = Set{Int}(),Set{Int}()
            delays = Float64[]
            for opportunity in opportunities
                overlapping = findall(episode -> episode.key==opportunity.key &&
                    episode.start<=opportunity.stop && episode.stop>=opportunity.start,episodes)
                union!(overlaps,overlapping)
                available = filter(ix -> !(ix in matched),overlapping)
                exact = findfirst(ix -> episodes[ix].start==opportunity.start &&
                    episodes[ix].stop==opportunity.stop,available)
                candidate = isempty(available) ? nothing : isnothing(exact) ? first(available) : available[exact]
                if !isnothing(candidate)
                    push!(matched,candidate)
                    push!(delays,Dates.value(max(opportunity.start,episodes[candidate].start)-opportunity.start)/60000)
                end
            end
            detected = length(matched)
            fp, fn = length(episodes)-tp,length(opportunities)-tp
            blocks = length(unique(df.blockid[selected]))
            push!(rows,(;group.set,group.grouping,group.regime,group.foldid,signal,threshold,
                tp,fp,fn,opportunities=length(opportunities),episodes=length(episodes),
                unmatched_episodes=fp,detected,duplicate_episodes=length(setdiff(overlaps,matched)),
                detection_unmatched_episodes=length(episodes)-detected,
                repeated_signal_minutes=sum(event.count-1 for event in episodes;init=0),
                mean_delay_minutes=isempty(delays) ? missing : mean(delays),
                precision=Classify._signalratio(tp,tp+fp),recall=Classify._signalratio(tp,tp+fn),
                detection_rate=Classify._signalratio(detected,length(opportunities)),
                episode_frequency=Classify._signalratio(length(episodes),count(selected)),
                eligible=count(selected),excluded=count(group.mask .& .!eligible),
                blocks,insufficient_support=blocks<minblocks))
        end
    end
    return DataFrame(rows)
end
