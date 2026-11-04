"""Versioned market detector settings; synthetic tests use explicit shorter windows."""
Base.@kwdef struct RegimeConfig
    longminutes::Int = 14400
    shortminutes::Int = 1440
    stepminutes::Int = 60
    mincoins::Int = 10
    weightcap::Float64 = 0.1
    threshold::Float64 = 0.0025
    strength::Float64 = 0.5
    confirmation::Int = 12
    liquidity::NamedTuple = Ohlcv.ld
end

"""Shared calendar settings independent of coin listing dates."""
Base.@kwdef struct FoldConfig
    anchor::DateTime = DateTime(2017, 1, 1)
    foldminutes::Int = 90 * 1440
    blockminutes::Int = 1440
    gapminutes::Int = 480
    minminutes::Int = 720
end

"""Validate explicitly configured detector and calendar invariants."""
function checkregimeconfig(config::RegimeConfig)::Nothing
    @assert config.longminutes > config.shortminutes > 1 "invalid slope windows: $(config.longminutes), $(config.shortminutes)"
    @assert config.stepminutes > 0 && config.confirmation > 0 "invalid step/confirmation: $(config.stepminutes), $(config.confirmation)"
    @assert config.mincoins > 0 && 0 < config.weightcap <= 1 && config.mincoins * config.weightcap >= 1 "infeasible weight cap/mincoins: $(config.weightcap), $(config.mincoins)"
    @assert config.threshold > 0 && config.strength >= 0 "invalid regime thresholds: $(config.threshold), $(config.strength)"
    return nothing
end

"""Check sorted, unique timestamps without allocating a timestamp set."""
function _strictregimetimes(times::AbstractVector{DateTime})::Bool
    return all(ix -> times[ix-1] < times[ix], (firstindex(times)+1):lastindex(times))
end

"""Normalize positive turnover weights, bounding every individual coin's influence."""
function regimeweights(volumes::AbstractVector; cap::Float64)::Vector{Float64}
    @assert !isempty(volumes) && all(v -> isfinite(v) && v > 0, volumes) "invalid turnover weights=$volumes"
    @assert length(volumes) * cap >= 1 "insufficient coins=$(length(volumes)) for cap=$cap"
    weights = zeros(length(volumes))
    active = trues(length(volumes))
    while any(active)
        available = 1 - sum(weights[.!active])
        indices = findall(active)
        proposed = available .* volumes[indices] ./ sum(volumes[indices])
        excessive = proposed .> cap + 1e-12
        if !any(excessive)
            weights[indices] .= proposed
            break
        end
        fixed = indices[excessive]
        weights[fixed] .= cap
        active[fixed] .= false
    end
    return weights
end

"""
Compute complete-window price slopes and volatility at shared observation cutoffs.
Only completed, contiguous minute history with a liquid endpoint is eligible.
"""
function regimeobservations(df::AbstractDataFrame, times::AbstractVector{DateTime},
        config::RegimeConfig; cutoff::Union{Nothing, DateTime}=nothing)::DataFrame
    checkregimeconfig(config)
    @assert issorted(df.opentime) && allunique(df.opentime) "malformed OHLCV timestamps, rows=$(nrow(df))"
    prices = Ohlcv.pivot(df)
    @assert all(p -> isfinite(p) && p > 0, prices) "nonpositive/nonfinite prices, rows=$(length(prices))"
    @assert all(v -> isfinite(v) && v >= 0, df.basevolume) "invalid base volumes, rows=$(nrow(df))"
    liquid = falses(nrow(df))
    # At a chronological training cutoff, range confirmation cannot use later rows.
    lastix = isnothing(cutoff) ? nrow(df) : searchsortedlast(df.opentime, cutoff)
    if lastix > 0
        for range in Ohlcv.liquiditycheck(DataFrame(pivot=prices[1:lastix],
                basevolume=df.basevolume[1:lastix]); config.liquidity...)
            liquid[range] .= true
        end
    end
    sums = zeros(nrow(df)+1)
    moments = zeros(nrow(df)+1)
    turnover = zeros(nrow(df)+1)
    for ix in eachindex(prices)
        sums[ix+1] = sums[ix]+prices[ix]
        moments[ix+1] = moments[ix]+ix*prices[ix]
        turnover[ix+1] = turnover[ix]+prices[ix]*df.basevolume[ix]
    end
    returns = vcat(0.0, diff(log.(prices)))
    returnsum = vcat(0.0, cumsum(returns))
    returnsquare = vcat(0.0, cumsum(returns .^ 2))
    result = DataFrame(opentime=DateTime[], slope=Float64[], short_slope=Float64[],
        volume=Float64[], volatility=Float64[])
    for time in times
        ix = searchsortedlast(df.opentime, time - Minute(1))
        ix >= config.longminutes || continue
        liquid[ix] || continue
        firstix = ix - config.longminutes + 1
        df.opentime[ix] == time - Minute(1) || continue
        df.opentime[ix] - df.opentime[firstix] == Minute(config.longminutes-1) || continue
        gradients = Float64[]
        for window in (config.longminutes, config.shortminutes)
            start = ix-window+1
            total = sums[ix+1]-sums[start]
            xy = moments[ix+1]-moments[start]-(start-1)*total
            gradient = (xy-(window+1)/2*total)/(window*(window^2-1)/12)
            push!(gradients, gradient/(total/window)*1440)
        end
        count = config.longminutes-1
        totalreturn = returnsum[ix+1]-returnsum[firstix+1]
        totalsquare = returnsquare[ix+1]-returnsquare[firstix+1]
        variance = max(0.0, (totalsquare-totalreturn^2/count)/(count-1))
        push!(result, (time, gradients[1], gradients[2],
            (turnover[ix+1]-turnover[firstix])/config.longminutes, sqrt(variance*1440)))
    end
    return result
end

"""
Detect shared up/flat/down regimes; missing coverage resets to unknown.
Initial flat, like any regime, requires confirmation. Historical candidate
backdating records its future confirmation dependency explicitly.
"""
function detectregimes(market::DataFrame, config::RegimeConfig; retrospective::Bool=true)::DataFrame
    checkregimeconfig(config)
    result = copy(market)
    result[!, :regime] = fill("unknown", nrow(result))
    result[!, :confirmed] = copy(result.opentime)
    state = "unknown"
    candidate = ""
    candidateix = 0
    for ix in 1:nrow(result)
        if !result.eligible[ix] || (ix > 1 && result.opentime[ix]-result.opentime[ix-1] != Minute(config.stepminutes))
            state = "unknown"
            candidate = ""
        end
        result.eligible[ix] || continue
        slope = result.slope[ix]
        volatility = result.volatility[ix]
        strength = volatility > 0 ? abs(slope)*sqrt(config.longminutes/1440)/volatility : 0.0
        upward = slope >= config.threshold && strength >= config.strength && result.breadth[ix] >= 0.6
        downward = slope <= -config.threshold && strength >= config.strength && result.breadth[ix] <= 0.4
        desired = upward ? "up" : downward ? "down" :
            state == "up" && slope >= config.threshold/2 ? "up" :
            state == "down" && slope <= -config.threshold/2 ? "down" : "flat"
        if desired == state
            candidate = ""
        else
            if candidate != desired
                candidate = desired
                candidateix = ix
            end
            reversal = (state == "up" && result.short_slope[ix] <= -2config.threshold) ||
                (state == "down" && result.short_slope[ix] >= 2config.threshold)
            required = reversal ? min(config.confirmation, cld(180, config.stepminutes)) : config.confirmation
            if ix-candidateix+1 >= required
                state = desired
                if retrospective
                    result.regime[candidateix:ix] .= state
                    result.confirmed[candidateix:ix] .= result.opentime[ix]
                end
                candidate = ""
            end
        end
        result.regime[ix] = state
    end
    return result
end

"""Build a common market calendar from an explicit immutable coin universe."""
function marketcalendar(histories::AbstractDict, config::RegimeConfig;
        startdt::DateTime, enddt::DateTime, retrospective::Bool=true,
        cutoff::Union{Nothing, DateTime}=nothing)::DataFrame
    checkregimeconfig(config)
    @assert startdt < enddt "invalid market range=$startdt to $enddt"
    times = collect(ceil(startdt, Minute(config.stepminutes)):Minute(config.stepminutes):enddt)
    observations = [regimeobservations(histories[coin], times, config; cutoff)
        for coin in sort!(collect(keys(histories)))]
    return marketcalendar(observations,config;startdt,enddt,retrospective)
end

"""
Build the shared market calendar from sorted per-coin observations, including memory-mapped views.
Advance one cursor per coin instead of allocating timestamp lookup dictionaries.
"""
function marketcalendar(observations::AbstractVector{<:AbstractDataFrame}, config::RegimeConfig;
        startdt::DateTime, enddt::DateTime, retrospective::Bool=true)::DataFrame
    checkregimeconfig(config)
    @assert startdt < enddt "invalid market range=$startdt to $enddt"
    times = collect(ceil(startdt, Minute(config.stepminutes)):Minute(config.stepminutes):enddt)
    observationtimes = [df[!, :opentime] for df in observations]
    for (coinix, timestamps) in enumerate(observationtimes)
        @assert _strictregimetimes(timestamps) "observation timestamps must be sorted and unique: coin index=$(coinix), rows=$(length(timestamps))"
    end
    cursors = ones(Int,length(observations))
    market = DataFrame(opentime=times, slope=zeros(length(times)), short_slope=zeros(length(times)),
        volatility=zeros(length(times)), breadth=zeros(length(times)),
        coins=zeros(Int,length(times)), eligible=falses(length(times)))
    for (ix, time) in enumerate(times)
        for coinix in eachindex(observationtimes)
            timestamps = observationtimes[coinix]
            while cursors[coinix] <= length(timestamps) && timestamps[cursors[coinix]] < time
                cursors[coinix] += 1
            end
        end
        rows = [(df, cursors[coinix]) for (coinix,df) in enumerate(observations)
            if cursors[coinix] <= length(observationtimes[coinix]) &&
                observationtimes[coinix][cursors[coinix]] == time]
        market.coins[ix] = length(rows)
        length(rows) >= config.mincoins || continue
        volumes = [df.volume[row] for (df,row) in rows]
        all(>(0), volumes) || continue
        weights = regimeweights(volumes; cap=config.weightcap)
        market.slope[ix] = sum(weights .* [df.slope[row] for (df,row) in rows])
        market.short_slope[ix] = sum(weights .* [df.short_slope[row] for (df,row) in rows])
        market.volatility[ix] = sum(weights .* [df.volatility[row] for (df,row) in rows])
        market.breadth[ix] = sum(weights .* [df.slope[row] > 0 for (df,row) in rows])
        market.eligible[ix] = true
    end
    return detectregimes(market, config; retrospective)
end

"""
Create shared calendar blocks and greedily balance complete blocks 4:1:1 within
each represented regime/fold. Exclude unknown coverage and enforce horizon-safe
gaps at every block boundary, including an optional inclusive experiment enddt.
Count eligible samples using timestamp tables.
"""
function foldcalendar(market::DataFrame, sampletimes::AbstractDict, config::FoldConfig;
        horizonminutes::Int, enddt::Union{Nothing,DateTime}=nothing)::DataFrame
    @assert config.foldminutes > 0 && config.blockminutes > 0 && config.minminutes > 0 "invalid fold configuration=$config"
    @assert config.gapminutes >= horizonminutes >= 0 "gap=$(config.gapminutes) is shorter than horizon=$horizonminutes"
    @assert issorted(market.opentime) && allunique(market.opentime) "market calendar is not sorted and unique"
    @assert nrow(market) > 1 "insufficient market observations=$(nrow(market))"
    step = market.opentime[2]-market.opentime[1]
    blocks = DataFrame(blockid=Int[], foldid=Int[], regime=String[], start=DateTime[],
        stop=DateTime[], confirmed=DateTime[], samples=Int[], set=String[])
    ix = 1
    while ix <= nrow(market)
        if market.regime[ix] == "unknown"
            ix += 1
            continue
        end
        start = market.opentime[ix]
        foldid = fld(Dates.value(start-config.anchor), config.foldminutes*60000)+1
        foldstop = config.anchor+Minute(foldid*config.foldminutes)
        stop = min(start+Minute(config.blockminutes), foldstop)
        lastix = ix
        while lastix < nrow(market) && market.opentime[lastix+1] < stop &&
                market.opentime[lastix+1]-market.opentime[lastix] == step &&
                market.regime[lastix+1] == market.regime[ix]
            lastix += 1
        end
        stop = min(stop, market.opentime[lastix]+step)
        isnothing(enddt) || (stop = min(stop,enddt+Minute(1)))
        usableend = stop-Minute(config.gapminutes+1)
        if usableend-start+Minute(1) >= Minute(config.minminutes)
            count = sum((searchsortedlast(times, usableend)-searchsortedfirst(times,start)+1
                for times in values(sampletimes));init=0)
            push!(blocks, (nrow(blocks)+1, foldid, market.regime[ix], start, usableend,
                maximum(market.confirmed[ix:lastix]), count, ""))
        end
        ix = lastix+1
    end
    return balancecalendar!(blocks)
end

"""Assign complete calendar blocks 4:1:1 using their final eligible sample counts."""
function balancecalendar!(blocks::DataFrame)::DataFrame
    totals = Dict{Tuple{Int,String}, Vector{Int}}()
    ratios = [4/6,1/6,1/6]
    setnames = ["train","eval","test"]
    celltotals = Dict((first(cell.foldid),first(cell.regime)) => sum(cell.samples)
        for cell in groupby(blocks, [:foldid,:regime]))
    # Largest first improves sample balance; IDs and chronology remain unchanged.
    for ix in sortperm(blocks.samples; rev=true)
        key = (blocks.foldid[ix],blocks.regime[ix])
        actual = get!(totals,key,zeros(Int,3))
        desired = celltotals[key] .* ratios
        costs = [sum((actual[j]+(j==candidate ? blocks.samples[ix] : 0)-desired[j])^2
            for j in 1:3) for candidate in 1:3]
        selected = argmin(costs)
        blocks.set[ix] = setnames[selected]
        actual[selected] += blocks.samples[ix]
    end
    return blocks
end

"""
Build identical shared folds while loading eligible timestamps one coin at a time.
The loader must return sorted, unique timestamps; references are released between coins.
"""
function foldcalendar(market::DataFrame, sampletimesloader::Function,
        coins::AbstractVector{<:AbstractString}, config::FoldConfig;
        horizonminutes::Int, enddt::Union{Nothing,DateTime}=nothing)::DataFrame
    blocks = foldcalendar(market,Dict{String,Vector{DateTime}}(),config;horizonminutes,enddt)
    for coin in coins
        times = sampletimesloader(coin)
        @assert issorted(times) && allunique(times) "sample timestamps for $coin must be sorted and unique: rows=$(length(times))"
        for block in eachrow(blocks)
            block.samples += searchsortedlast(times,block.stop)-searchsortedfirst(times,block.start)+1
        end
        times = nothing
        GC.gc()
    end
    return balancecalendar!(blocks)
end

"""Count train/eval/test membership without constructing minute-level projected rows."""
function calendarcounts(times::AbstractVector{DateTime}, calendar::DataFrame)::NamedTuple
    @assert issorted(times) && allunique(times) "sample timestamps must be sorted and unique: rows=$(length(times))"
    counts = Dict("train"=>0,"eval"=>0,"test"=>0)
    for block in eachrow(calendar)
        @assert haskey(counts,block.set) "unsupported calendar set=$(block.set)"
        counts[block.set] += searchsortedlast(times,block.stop)-searchsortedfirst(times,block.start)+1
    end
    return (train=counts["train"],eval=counts["eval"],test=counts["test"])
end

"""Project shared calendar assignments onto eligible coin samples, excluding unknown."""
function calendarsets(times::AbstractVector{DateTime}, calendar::DataFrame)::DataFrame
    @assert issorted(times) && allunique(times) "sample timestamps are not sorted and unique"
    result = DataFrame(opentime=DateTime[], set=String[], regime=String[],
        foldid=Int[], blockid=Int[], rangeid=Int[])
    for block in eachrow(calendar)
        @assert block.regime in ("up","flat","down") "unsupported block regime=$(block.regime)"
        firstix = searchsortedfirst(times,block.start)
        lastix = searchsortedlast(times,block.stop)
        for ix in firstix:lastix
            push!(result,(times[ix],block.set,block.regime,block.foldid,block.blockid,block.blockid))
        end
    end
    return result
end

"""Chronological expanding train/eval/test schedules with non-overlapping test windows."""
function walkforwardschedule(startdt::DateTime, enddt::DateTime;
        trainminutes::Int=2*365*1440, evalminutes::Int=90*1440,
        testminutes::Int=90*1440, gapminutes::Int=480)::DataFrame
    @assert trainminutes > 0 && evalminutes > 0 && testminutes > 0 && gapminutes >= 0 "invalid chronological windows=$((trainminutes,evalminutes,testminutes,gapminutes))"
    result = DataFrame(walkforwardid=Int[], trainstart=DateTime[], trainstop=DateTime[],
        evalstart=DateTime[], evalstop=DateTime[], teststart=DateTime[], teststop=DateTime[])
    trainstop = startdt+Minute(trainminutes-1)
    while true
        evalstart = trainstop+Minute(gapminutes+1)
        evalstop = evalstart+Minute(evalminutes-1)
        teststart = evalstop+Minute(gapminutes+1)
        teststop = teststart+Minute(testminutes-1)
        teststop <= enddt || break
        push!(result,(nrow(result)+1,startdt,trainstop,evalstart,evalstop,teststart,teststop))
        trainstop += Minute(testminutes)
    end
    return result
end

"""Select chronological sets; reject retrospective confirmation after training cutoff."""
function walkforwardsets(sets::DataFrame, market::DataFrame, window)::DataFrame
    result = copy(sets)
    result.set .= "excluded"
    for row in eachrow(result)
        if window.trainstart <= row.opentime <= window.trainstop
            ix = searchsortedlast(market.opentime,row.opentime)
            if ix > 0 && market.confirmed[ix] <= window.trainstop
                row.set = "train"
            end
        elseif window.evalstart <= row.opentime <= window.evalstop
            row.set = "eval"
        elseif window.teststart <= row.opentime <= window.teststop
            row.set = "test"
        end
    end
    return filter(row -> row.set != "excluded" && row.regime != "unknown",result)
end
