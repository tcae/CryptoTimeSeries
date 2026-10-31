"""
Read-only CS01/D01 market-regime diagnostics.

Run `julia --project=. scripts/diagnose_market_regimes.jl`.
Outputs are written under `~/crypto/debug/market-regimes`; no source stores or
classifier artifacts are changed. Thresholds are diagnostic candidates, not fitted
production settings.
"""
module MarketRegimeDiagnostics

using Dates, Statistics, DataFrames, Arrow
using EnvConfig, Ohlcv, Features, Targets, TradingStrategy

"""
    coinindicator(df; longminutes=14400, shortminutes=1440, liquid=trues(nrow(df)))

Return hourly observations of full-window minute-price normalized slopes.
Use the last completed minute before each hour as the information cutoff.
Prefix moments calculate exact linear price regressions without expensive
14,400-row rolling dot products. Quote turnover is approximated by pivot times
base volume. Require contiguous minute coverage throughout both windows.
Hours whose cutoff minute is not `liquid` (see `liquidmask`) are skipped.
"""
function coinindicator(df::AbstractDataFrame; longminutes::Int=14400, shortminutes::Int=1440,
        liquid::AbstractVector{Bool}=trues(nrow(df)))
    @assert longminutes > shortminutes > 1 "windows long=$(longminutes), short=$(shortminutes) must satisfy long > short > 1"
    @assert issorted(df.opentime) && allunique(df.opentime) "OHLCV timestamps must be sorted and unique, rows=$(nrow(df))"
    @assert length(liquid) == nrow(df) "liquid mask length=$(length(liquid)) must equal rows=$(nrow(df))"
    timestamps = df.opentime
    basevolume = df.basevolume
    prices = df.open .* 0.25 .+ df.high .* 0.25 .+ df.low .* 0.25 .+ df.close .* 0.25
    @assert all(p -> isfinite(p) && p > 0, prices) "prices must be finite and positive"
    @assert all(v -> isfinite(v) && v >= 0, df.basevolume) "base volumes must be finite and nonnegative"
    sums = zeros(nrow(df) + 1)
    moments = zeros(nrow(df) + 1)
    turnover = zeros(nrow(df) + 1)
    for ix in 1:nrow(df)
        sums[ix + 1] = sums[ix] + prices[ix]
        moments[ix + 1] = moments[ix] + ix * prices[ix]
        turnover[ix + 1] = turnover[ix] + prices[ix] * basevolume[ix]
    end
    result = DataFrame(opentime=DateTime[], slope=Float64[], short_slope=Float64[], quote_turnover=Float64[], price=Float64[])
    for ix in longminutes:nrow(df)
        time = timestamps[ix] + Minute(1)
        time == floor(time, Hour) || continue
        liquid[ix] || continue
        firstix = ix - longminutes + 1
        timestamps[ix] - timestamps[firstix] == Minute(longminutes - 1) || continue
        slopes = Float64[]
        for window in (longminutes, shortminutes)
            start = ix - window + 1
            total = sums[ix + 1] - sums[start]
            xy = moments[ix + 1] - moments[start] - (start - 1) * total
            gradient = (xy - (window + 1) / 2 * total) / (window * (window^2 - 1) / 12)
            push!(slopes, gradient / (total / window) * 1440)
        end
        push!(result, (time, slopes[1], slopes[2], (turnover[ix + 1] - turnover[firstix]) / longminutes, prices[ix]))
    end
    return result
end

"""
    liquidmask(df)

Return a per-row mask that is `true` inside the `Ohlcv.liquiditycheck` ranges
computed with the `Ohlcv.ld` defaults on the complete OHLCV history `df`.
This is the same liquidity definition used for P02 set generation.
"""
function liquidmask(df::AbstractDataFrame)
    mask = falses(nrow(df))
    for range in Ohlcv.liquiditycheck(DataFrame(pivot=Ohlcv.pivot(df), basevolume=df.basevolume))
        mask[range] .= true
    end
    return mask
end

"""
    cappedweights(volumes; cap=0.1)

Return normalized positive turnover weights with no coin above `cap`.
Redistribute excess proportionally among uncapped coins.
"""
function cappedweights(volumes::AbstractVector; cap::Float64=0.1)
    @assert all(v -> isfinite(v) && v > 0, volumes) "turnover weights must be positive and finite, values=$(volumes)"
    @assert 0 < cap <= 1 && length(volumes) * cap >= 1 "cap=$(cap) infeasible for coins=$(length(volumes))"
    weights = zeros(length(volumes))
    active = trues(length(volumes))
    remaining = 1.0
    while any(active)
        proposal = remaining .* volumes[active] ./ sum(volumes[active])
        indices = findall(active)
        excessive = proposal .> cap
        if !any(excessive)
            weights[indices] .= proposal
            break
        end
        fixed = indices[excessive]
        weights[fixed] .= cap
        active[fixed] .= false
        remaining = 1 - sum(weights)
    end
    return weights
end

"""
    regimes(times, slopes, shorts, volatility, breadth; threshold, strength, confirmation)

Detect up/flat/down with 50% exit hysteresis and hourly persistence confirmation.
Trend entry also requires volatility-normalized 10-day strength and >=60%
volume-weighted directional breadth. A opposite one-day slope at twice the entry
threshold confirms a candidate sooner (three hours), but never without persistence.
Record retrospective segment starts separately from confirmation timestamps.
Missing market coverage resets the detector; no labels are inferred across gaps.
"""
function regimes(times, slopes, shorts, volatility, breadth;
    threshold::Float64, strength::Float64, confirmation::Int)
    @assert threshold > 0 && strength >= 0 && confirmation > 0 "invalid detector threshold=$(threshold), strength=$(strength), confirmation=$(confirmation)"
    labels = fill("unknown", length(times))
    segments = DataFrame(start=DateTime[], confirmed=DateTime[], stop=DateTime[], regime=String[], confirmation_hours=Int[])
    state = "flat"
    candidate = ""
    candidateix = 1
    segmentix = 0
    for ix in eachindex(times)
        newblock = ix == 1 || times[ix] - times[ix - 1] != Hour(1)
        if newblock
            state = "flat"
            candidate = ""
            push!(segments, (times[ix], times[ix], times[ix], state, 0))
            segmentix = nrow(segments)
        end
        score = volatility[ix] > 0 ? abs(slopes[ix]) * sqrt(10) / volatility[ix] : 0.0
        up = slopes[ix] >= threshold && score >= strength && breadth[ix] >= 0.6
        down = slopes[ix] <= -threshold && score >= strength && breadth[ix] <= 0.4
        desired = up ? "up" : down ? "down" :
            state == "up" && slopes[ix] >= threshold / 2 ? "up" :
            state == "down" && slopes[ix] <= -threshold / 2 ? "down" : "flat"
        if desired == state
            candidate = ""
        else
            if candidate != desired
                candidate = desired
                candidateix = ix
            end
            reversal = (state == "up" && shorts[ix] <= -2threshold) ||
                (state == "down" && shorts[ix] >= 2threshold)
            required = reversal ? min(3, confirmation) : confirmation
            if ix - candidateix + 1 >= required
                segments.stop[segmentix] = times[candidateix] - Hour(1)
                push!(segments, (times[candidateix], times[ix], times[ix], desired, ix - candidateix + 1))
                segmentix = nrow(segments)
                state = desired
                candidate = ""
            end
        end
        labels[ix] = state
        segments.stop[segmentix] = times[ix]
    end
    retrospective = copy(labels)
    for segment in eachrow(segments)
        firstix = searchsortedfirst(times, segment.start)
        lastix = searchsortedlast(times, segment.stop)
        retrospective[firstix:lastix] .= segment.regime
    end
    return labels, retrospective, segments
end

"""
Accumulate eligible samples using monotone timestamp joins and aggregate consecutive
identical reporting buckets before updating counts. Concrete Arrow column arguments
keep the per-minute loop specialized independently of heterogeneous store schemas.
"""
function counteligible!(counts::AbstractDict, featuretimes::AbstractVector, targettimes::AbstractVector,
    targetlabels::AbstractVector, settimes::AbstractVector, setlabels::AbstractVector,
    markettimes::AbstractVector, regimes::AbstractVector, months::AbstractVector,
    featureid::String, targetid::String, coin::String)
    fi = ti = hi = 1
    previouskey = nothing
    accumulated = 0
    ignorecode = Targets.tradelabelcode(Targets.ignore)
    for ix in eachindex(settimes)
        time = settimes[ix]
        while fi <= length(featuretimes) && featuretimes[fi] < time
            fi += 1
        end
        while ti <= length(targettimes) && targettimes[ti] < time
            ti += 1
        end
        fi <= length(featuretimes) && ti <= length(targettimes) || break
        featuretimes[fi] == time && targettimes[ti] == time || continue
        targetlabels[ti] != ignorecode || continue
        hour = floor(time, Hour)
        while hi <= length(markettimes) && markettimes[hi] < hour
            hi += 1
        end
        available = hi <= length(markettimes) && markettimes[hi] == hour
        regime = available ? regimes[hi] : "unknown"
        month = available ? months[hi] : Dates.format(time, dateformat"yyyy-mm")
        key = (featureid, targetid, coin, string(setlabels[ix]), regime, month)
        if key != previouskey
            if !isnothing(previouskey)
                counts[previouskey] = get(counts, previouskey, 0) + accumulated
            end
            previouskey = key
            accumulated = 0
        end
        accumulated += 1
    end
    if !isnothing(previouskey)
        counts[previouskey] = get(counts, previouskey, 0) + accumulated
    end
    return nothing
end

"""
    samplecoverage(folder; coins=TradingStrategy.coinsetconfig("CS01").coins)

Count currently stored eligible P02_D01 samples by coin, set, retrospective regime
and calendar month, for F19/T24, F20/T25 and F21/T26. Use the middle diagnostic
candidate (0.25% daily slope, strength 0.5, twelve-hour confirmation) only as an
illustration, not as a chosen setting. Read old target stores directly and report
missing/malformed files and config-hash status; never recreate or migrate them.
Month is a reporting bucket, not a walk-forward fold.
"""
function samplecoverage(folder::AbstractString; coins::AbstractVector{<:AbstractString}=TradingStrategy.coinsetconfig("CS01").coins)
    labels = DataFrame(Arrow.Table(joinpath(folder, "slope0.0025-strength0.5-confirm12-hourly.arrow")))
    months = Dates.format.(labels.opentime, dateformat"yyyy-mm")
    counts = Dict{Tuple{String,String,String,String,String,String},Int}()
    status = DataFrame(coin=String[], features=String[], targets=String[], status=String[])
    for coin in coins
        coinpath = EnvConfig.coinfolderpath(coin, EnvConfig.pairquote)
        setpath = joinpath(coinpath, "sets", "P02_D01.arrow")
        if !isfile(setpath)
            @warn "Missing P02 sets for sample coverage" coin setpath
            push!(status, (coin, "*", "*", "missing sets"))
            continue
        end
        sets = DataFrame(Arrow.Table(setpath))
        for (featureid, targetid) in (("F19", "T24"), ("F20", "T25"), ("F21", "T26"))
            featurepath = joinpath(coinpath, "features", "$featureid.arrow")
            targetpath = joinpath(coinpath, "targets", "$targetid.arrow")
            if !isfile(featurepath) || !isfile(targetpath)
                push!(status, (coin, featureid, targetid, "missing features or targets"))
                continue
            end
            features = Arrow.Table(featurepath)
            targets = Arrow.Table(targetpath)
            if !issorted(features.opentime) || !allunique(features.opentime) ||
                !issorted(targets.opentime) || !allunique(targets.opentime)
                @error "Malformed feature/target timestamps excluded from sample coverage" coin featureid targetid
                push!(status, (coin, featureid, targetid, "invalid timestamps"))
                continue
            end
            metadata = Arrow.getmetadata(targets)
            current = !isnothing(metadata) && get(metadata, "confighash", "") == TradingStrategy.targetshash(targetid)
            push!(status, (coin, featureid, targetid, current ? "current target hash" : "historical target hash"))
            counteligible!(counts, features.opentime, targets.opentime, targets.label,
                sets.opentime, sets.set, labels.opentime, labels.retrospective, months,
                featureid, targetid, string(coin))
        end
        println("$(EnvConfig.now()) sample coverage $coin")
        flush(stdout)
    end
    table = DataFrame(features=String[], targets=String[], coin=String[], set=String[],
        regime=String[], month=String[], samples=Int[])
    for (key, count) in sort!(collect(counts); by=first)
        push!(table, (key..., count))
    end
    Arrow.write(joinpath(folder, "sample-coverage.arrow"), table)
    Arrow.write(joinpath(folder, "sample-store-status.arrow"), status)
    open(joinpath(folder, "report.txt"), "a") do io
        println(io, "\nStored eligible sample coverage: illustrative candidate only, not a selected regime definition.")
        println(io, "Historical targets are counted as stored, NOT declared compatible with bounded Trend05.")
        println(io, "Missing T26 files include the previous gap-crossing invalidations; no stores rebuilt.")
        println(io, "Calendar months are reporting buckets, NOT fold assignments.")
        if nrow(table) > 0
            totals = combine(groupby(table, [:features, :targets, :set, :regime]), :samples => sum => :samples)
            show(io, MIME"text/plain"(), totals; allrows=true, allcols=true)
        end
        println(io, "\nStore statuses")
        show(io, MIME"text/plain"(), combine(groupby(status, [:features, :targets, :status]), nrow => :stores); allrows=true)
    end
    return table
end

"""
    main()

Build the CS01 market series from stored Arrow OHLCV, compare candidate detectors,
and persist hourly indicators, boundaries, distributions and diagnostic summaries.
Use at least ten liquid coins per hour and freeze no thresholds automatically.
"""
function main()
    EnvConfig.init(EnvConfig.training)
    EnvConfig.setcoinspath!("Bybit")
    folder = mkpath(joinpath(EnvConfig.defaultdebugfilespath, "market-regimes",
        Dates.format(Dates.now(), dateformat"yyyymmdd-HHMMSS")))
    dr = TradingStrategy.daterangeconfig("D01")
    times = collect(ceil(dr.startdt, Hour):Hour(1):floor(dr.enddt, Hour))
    coins = TradingStrategy.coinsetconfig("CS01").coins
    slope = fill(NaN, length(times), length(coins))
    short = similar(slope)
    short .= NaN
    volumes = zeros(size(slope))
    prices = fill(NaN, size(slope))
    coverage = DataFrame(coin=String[], hourly_rows=Int[], status=String[])
    for (coinix, coin) in enumerate(coins)
        path = EnvConfig.coinfile(coin, EnvConfig.pairquote, "ohlcv"; extension=".arrow")
        if !isfile(path)
            @warn "Missing OHLCV; excluded from diagnostic" coin path
            push!(coverage, (coin, 0, "missing"))
            continue
        end
        df = DataFrame(Arrow.Table(path))
        if !issorted(df.opentime) || !allunique(df.opentime)
            @error "Malformed OHLCV timestamps; excluded from diagnostic" coin path
            push!(coverage, (coin, 0, "invalid timestamps"))
            continue
        end
        firstix = searchsortedfirst(df.opentime, first(times) - Day(10))
        lastix = searchsortedlast(df.opentime, last(times) - Minute(1))
        if lastix - firstix + 1 < 14400
            @warn "Insufficient OHLCV for ten-day indicator" coin
            push!(coverage, (coin, 0, "insufficient history"))
            continue
        end
        liquid = liquidmask(df)
        observations = coinindicator(view(df, firstix:lastix, :); liquid=view(liquid, firstix:lastix))
        for row in eachrow(observations)
            ix = searchsortedfirst(times, row.opentime)
            if ix <= length(times) && times[ix] == row.opentime
                slope[ix, coinix] = row.slope
                short[ix, coinix] = row.short_slope
                volumes[ix, coinix] = row.quote_turnover
                prices[ix, coinix] = row.price
            end
        end
        push!(coverage, (coin, nrow(observations), "ok"))
        println("$(EnvConfig.now()) indicator $(coinix)/$(length(coins)) $coin rows=$(nrow(observations))")
        flush(stdout)
    end
    market = DataFrame(opentime=DateTime[], slope=Float64[], short_slope=Float64[],
        breadth=Float64[], daily_volatility=Float64[], coins=Int[], maximum_weight=Float64[])
    for ix in 241:length(times)
        eligible = Int[]
        volatilities = Float64[]
        for coinix in eachindex(coins)
            history = view(prices, ix-240:ix, coinix)
            if isfinite(slope[ix, coinix]) && volumes[ix, coinix] > 0 && all(isfinite, history)
                returns = diff(log.(history))
                push!(eligible, coinix)
                push!(volatilities, std(returns) * sqrt(24))
            end
        end
        length(eligible) >= 10 || continue
        weights = cappedweights(volumes[ix, eligible])
        push!(market, (times[ix], sum(weights .* slope[ix, eligible]),
            sum(weights .* short[ix, eligible]), sum(weights .* (slope[ix, eligible] .> 0)),
            sum(weights .* volatilities), length(eligible), maximum(weights)))
    end
    @assert nrow(market) > 0 "no market hours meet ten-coin coverage; inspect coverage=$(coverage)"
    Arrow.write(joinpath(folder, "market.arrow"), market)
    Arrow.write(joinpath(folder, "coverage.arrow"), coverage)
    summary = DataFrame(threshold=Float64[], strength=Float64[], confirmation=Int[],
        up_hours=Int[], flat_hours=Int[], down_hours=Int[], segments=Int[],
        median_duration_hours=Float64[], short_segments=Int[], median_confirmation_hours=Float64[])
    for threshold in (0.001, 0.0025, 0.005), strength in (0.5, 1.0), confirmation in (6, 12, 24)
        online, historical, segments = regimes(market.opentime, market.slope, market.short_slope,
            market.daily_volatility, market.breadth; threshold, strength, confirmation)
        stem = "slope$(threshold)-strength$(strength)-confirm$(confirmation)"
        Arrow.write(joinpath(folder, "$stem-hourly.arrow"),
            DataFrame(opentime=market.opentime, online=online, retrospective=historical))
        Arrow.write(joinpath(folder, "$stem-segments.arrow"), segments)
        durations = (segments.stop .- segments.start) ./ Hour(1) .+ 1
        push!(summary, (threshold, strength, confirmation, count(==("up"), historical),
            count(==("flat"), historical), count(==("down"), historical), nrow(segments),
            median(durations), count(<(24), durations), median(segments.confirmation_hours)))
    end
    Arrow.write(joinpath(folder, "candidates.arrow"), summary)
    report = joinpath(folder, "report.txt")
    open(report, "w") do io
        println(io, "CS01/D01 ten-day normalized minute-price slope diagnostics")
        println(io, "Hourly cutoffs use completed minutes only. Turnover approximates pivot*basevolume.")
        println(io, "Coin hours count only if the cutoff minute is inside an Ohlcv.liquiditycheck range (Ohlcv.ld=$(Ohlcv.ld)).")
        println(io, "Weights use trailing ten-day turnover, cap 10%; minimum 10 coins; contiguous windows only.")
        println(io, "Volatility is the weighted mean per-coin ten-day hourly-return volatility, daily-scaled.")
        println(io, "Strength is abs(daily slope)*sqrt(10)/daily volatility: a heuristic, NOT a significance test.")
        println(io, "Breadth is volume weighted; enter at 60% up or 60% down. Exit slope threshold is half entry.")
        println(io, "Short-slope reversal confirms after three candidate hours; normal confirmation 6/12/24 hours.")
        println(io, "Historical labels backdate to candidate start; online labels never backdate.")
        println(io, "Gap starts initialize flat without claiming a confirmed flat regime.")
        println(io, "No threshold is selected or used in classifier training; no partitions changed.")
        println(io, "market hours=$(nrow(market)) first=$(first(market.opentime)) last=$(last(market.opentime))")
        for (name, rows) in (("development_before_2023", market[market.opentime .< DateTime(2023), :]),
            ("later_2023_onward", market[market.opentime .>= DateTime(2023), :]))
            println(io, "\n$name hours=$(nrow(rows))")
            if nrow(rows) > 0
                for col in (:slope, :short_slope, :daily_volatility, :breadth, :coins)
                    println(io, "$col quantiles [1%,10%,25%,50%,75%,90%,99%]=$(quantile(rows[!, col], [0.01,0.1,0.25,0.5,0.75,0.9,0.99]))")
                end
            end
        end
        println(io, "\nCoin coverage")
        show(io, MIME"text/plain"(), coverage; allrows=true)
        println(io, "\nCandidate comparison (durations include censored edge segments)")
        show(io, MIME"text/plain"(), summary; allrows=true, allcols=true)
    end
    samplecoverage(folder)
    println("Report: $report")
    return folder
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

end
