"""
    collecttradesdf(tsm)

Collect per-pair Trades tables from one `TsmCache` into one DataFrame suitable
for cross-script comparison and aggregation.
"""
function collecttradesdf(tsm::TsmCache)::DataFrame
    pairkeys = tradingpairs(tsm)
    if isempty(pairkeys)
        return DataFrame()
    end

    parts = DataFrame[]
    for pair in pairkeys
        tdf = trades(tsm, pair)
        size(tdf, 1) > 0 || continue
        push!(parts, tdf)
    end
    if isempty(parts)
        return DataFrame()
    end

    tradesdf = reduce(vcat, parts; cols=:union)
    sortcols = Symbol[]
    for col in (:coin, :set, :rangeid, :pair, :opentime)
        (col in names(tradesdf)) && push!(sortcols, col)
    end
    !isempty(sortcols) && sort!(tradesdf, sortcols)
    return tradesdf
end

"""
    savetradesdf(tradesdf; stem="trades", folderpath=EnvConfig.logfolder())

Persist one Trades DataFrame in the current log folder as `<stem>.arrow`.
"""
function savetradesdf(tradesdf::AbstractDataFrame; stem::AbstractString="trades", folderpath::AbstractString=EnvConfig.logfolder())::String
    return EnvConfig.savedf(DataFrame(tradesdf; copycols=false), String(stem); folderpath=String(folderpath))
end

"""
    savetradesdf(tsm; stem="trades", folderpath=EnvConfig.logfolder())

Collect and persist the combined Trades DataFrame from one `TsmCache`.
"""
function savetradesdf(tsm::TsmCache; stem::AbstractString="trades", folderpath::AbstractString=EnvConfig.logfolder())::String
    return savetradesdf(collecttradesdf(tsm); stem=stem, folderpath=folderpath)
end

"""
    readtradesdf(; stem="trades", folderpath=EnvConfig.logfolder())

Load one Trades DataFrame from the current log folder, returning an empty
DataFrame when the file is missing.
"""
function readtradesdf(; stem::AbstractString="trades", folderpath::AbstractString=EnvConfig.logfolder())::DataFrame
    loaded = EnvConfig.readdf(String(stem); folderpath=String(folderpath))
    return isnothing(loaded) ? DataFrame() : loaded
end

"""Return the grouping columns used to compile gains from concatenated Trades rows."""
function _compilegains_groupcols(tradesdf::AbstractDataFrame)::Vector{Symbol}
    @assert :pair in propertynames(tradesdf) "tradesdf must contain :pair to compile gains; names=$(names(tradesdf))"
    cols = Symbol[:pair]
    (:set in propertynames(tradesdf)) && push!(cols, :set)
    (:rangeid in propertynames(tradesdf)) && push!(cols, :rangeid)
    return cols
end

"""Return an empty gains dataframe with optional grouping columns mirrored from `tradesdf`."""
function _emptygainsdf(tradesdf::AbstractDataFrame)::DataFrame
    gainsdf = DataFrame(
        pair=String[],
        opentime=DateTime[],
        closetime=DateTime[],
        openprice=Float32[],
        closeprice=Float32[],
        volume=Float32[],
        side=String[],
        gain=Float32[],
        gainquote=Float32[],
    )
    if :set in propertynames(tradesdf)
        insertcols!(gainsdf, 2, :set => copy(tradesdf[1:0, :set]))
    end
    if :rangeid in propertynames(tradesdf)
        insert_at = :set in propertynames(tradesdf) ? 3 : 2
        insertcols!(gainsdf, insert_at, :rangeid => copy(tradesdf[1:0, :rangeid]))
    end
    return gainsdf
end

"""Return one numeric Trades column value, defaulting to `0f0` for missing rows."""
function _compilegainsvalue(tradesdf::AbstractDataFrame, ix::Integer, col::Symbol)
    if !(col in propertynames(tradesdf))
        return 0f0
    end
    value = tradesdf[ix, col]
    return (ismissing(value) || isnothing(value)) ? 0f0 : value
end

"""Return the execution timestamp for one reflected position change row.

The position snapshot change is observed on row `ix`, but the execution
itself happened on the previous minute row whose bar triggered the fill.
"""
function _compilegainstime(tradesdf::AbstractDataFrame, ix::Integer)::DateTime
    @assert :opentime in propertynames(tradesdf) "tradesdf must contain :opentime to compile gains; names=$(names(tradesdf))"
    @assert ix > 1 "compile gain timestamps require a previous row; got ix=$(ix)"
    return tradesdf[ix - 1, :opentime]
end

"""Return the execution price stored on the position-change row for one order lane."""
function _compilegainsprice(tradesdf::AbstractDataFrame, ix::Integer, pricecol::Symbol)::Float32
    @assert pricecol in propertynames(tradesdf) "tradesdf must contain $(pricecol) to compile gains; names=$(names(tradesdf))"
    price = tradesdf[ix, pricecol]
    @assert !ismissing(price) && !isnothing(price) && (price > 0f0) "Expected positive $(pricecol) on position change at ix=$(ix), opentime=$(_compilegainstime(tradesdf, ix)), pair=$(tradesdf[ix, :pair]); got $(price)"
    return price
end

"""Append one compiled gain row, mirroring optional `set` and `rangeid` columns from `tradesdf`."""
function _pushcompiledgain!(gainsdf::DataFrame, tradesdf::AbstractDataFrame, ix::Integer, opentime::DateTime, openprice::Float32, closetime::DateTime, closeprice::Float32, volume::Float32, side::Symbol)::Nothing
    gain = side == :long ? (closeprice - openprice) / openprice : (openprice - closeprice) / openprice
    gainquote = side == :long ? volume * (closeprice - openprice) : volume * (openprice - closeprice)

    if (:set in propertynames(gainsdf)) && (:rangeid in propertynames(gainsdf))
        push!(gainsdf, (
            pair=String(tradesdf[ix, :pair]),
            set=tradesdf[ix, :set],
            rangeid=tradesdf[ix, :rangeid],
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
        ))
    elseif :set in propertynames(gainsdf)
        push!(gainsdf, (
            pair=String(tradesdf[ix, :pair]),
            set=tradesdf[ix, :set],
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
        ))
    elseif :rangeid in propertynames(gainsdf)
        push!(gainsdf, (
            pair=String(tradesdf[ix, :pair]),
            rangeid=tradesdf[ix, :rangeid],
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
        ))
    else
        push!(gainsdf, (
            pair=String(tradesdf[ix, :pair]),
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
        ))
    end
    return nothing
end

"""Queue one open execution for later FIFO close matching."""
function _enqueuecompiledopen!(openqueue::Vector{NamedTuple{(:opentime, :openprice, :remaining), Tuple{DateTime, Float32, Float32}}}, opentime::DateTime, openprice::Float32, volume::Float32)::Nothing
    volume > 0f0 || return nothing
    push!(openqueue, (opentime=opentime, openprice=openprice, remaining=volume))
    return nothing
end

"""Consume one close execution against queued opens in FIFO order and emit gain rows."""
function _matchcompiledclose!(gainsdf::DataFrame, openqueue::Vector{NamedTuple{(:opentime, :openprice, :remaining), Tuple{DateTime, Float32, Float32}}}, tradesdf::AbstractDataFrame, ix::Integer, closeprice::Float32, closevolume::Float32, side::Symbol)::Nothing
    remaining = closevolume
    closetime = _compilegainstime(tradesdf, ix)
    while remaining > 0f0
        @assert !isempty(openqueue) "Encountered unmatched $(side) close volume=$(remaining) at ix=$(ix), opentime=$(closetime), pair=$(tradesdf[ix, :pair])"
        opentrade = first(openqueue)
        matched = min(opentrade.remaining, remaining)
        _pushcompiledgain!(gainsdf, tradesdf, ix, opentrade.opentime, opentrade.openprice, closetime, closeprice, matched, side)
        remaining -= matched
        if matched == opentrade.remaining
            popfirst!(openqueue)
        else
            openqueue[1] = (opentime=opentrade.opentime, openprice=opentrade.openprice, remaining=opentrade.remaining - matched)
        end
    end
    return nothing
end

"""Compile gain rows for one pair-scoped Trades partition."""
function _compilegainspartition!(gainsdf::DataFrame, tradesview::AbstractDataFrame)::Nothing
    nrow(tradesview) == 0 && return nothing
    ordered = sort(DataFrame(tradesview; copycols=false), :opentime)

    longopens = NamedTuple{(:opentime, :openprice, :remaining), Tuple{DateTime, Float32, Float32}}[]
    shortopens = NamedTuple{(:opentime, :openprice, :remaining), Tuple{DateTime, Float32, Float32}}[]

    for ix in 1:nrow(ordered)
        longamount = _compilegainsvalue(ordered, ix, :lp_amount)
        shortamount = _compilegainsvalue(ordered, ix, :sp_amount)
        @assert !((longamount > 0f0) && (shortamount > 0f0)) "Expected at most one open position side per row while compiling gains at ix=$(ix), opentime=$(_compilegainstime(ordered, ix)), pair=$(ordered[ix, :pair]); got lp_amount=$(longamount), sp_amount=$(shortamount)"

        ix == 1 && continue

        prevlong = _compilegainsvalue(ordered, ix - 1, :lp_amount)
        prevshort = _compilegainsvalue(ordered, ix - 1, :sp_amount)
        longdelta = longamount - prevlong
        shortdelta = shortamount - prevshort

        # Position changes on row `ix` indicate executions that happened in the prior minute.
        if longdelta > 0f0
            _enqueuecompiledopen!(longopens, _compilegainstime(ordered, ix), _compilegainsprice(ordered, ix, :lo_pavg), longdelta)
        elseif longdelta < 0f0
            _matchcompiledclose!(gainsdf, longopens, ordered, ix, _compilegainsprice(ordered, ix, :lc_pavg), -longdelta, :long)
        end

        if shortdelta > 0f0
            _enqueuecompiledopen!(shortopens, _compilegainstime(ordered, ix), _compilegainsprice(ordered, ix, :so_pavg), shortdelta)
        elseif shortdelta < 0f0
            _matchcompiledclose!(gainsdf, shortopens, ordered, ix, _compilegainsprice(ordered, ix, :sc_pavg), -shortdelta, :short)
        end
    end
    return nothing
end

"""
    compilegainsdf(tradesdf; stem="xchgains", folderpath=EnvConfig.logfolder())

Compile open/close gain pairs from one Trades DataFrame, scoping matching by
`pair` plus optional `set` and `rangeid`, then persist the result in the current
log folder as `<stem>.arrow`.
"""
function compilegainsdf(tradesdf::AbstractDataFrame; stem::AbstractString="xchgains", folderpath::AbstractString=EnvConfig.logfolder())::DataFrame
    gainsdf = _emptygainsdf(tradesdf)
    if nrow(tradesdf) == 0
        EnvConfig.savedf(gainsdf, String(stem); folderpath=String(folderpath))
        return gainsdf
    end

    groupcols = _compilegains_groupcols(tradesdf)
    for tradesview in groupby(DataFrame(tradesdf; copycols=false), groupcols; sort=false)
        _compilegainspartition!(gainsdf, tradesview)
    end

    sortcols = Symbol[]
    (:set in propertynames(gainsdf)) && push!(sortcols, :set)
    (:rangeid in propertynames(gainsdf)) && push!(sortcols, :rangeid)
    append!(sortcols, [:pair, :opentime, :closetime])
    !isempty(sortcols) && sort!(gainsdf, sortcols)
    EnvConfig.savedf(gainsdf, String(stem); folderpath=String(folderpath))
    return gainsdf
end

"""
    compilegainsdf(tsm; stem="xchgains", folderpath=EnvConfig.logfolder())

Collect the combined Trades DataFrame from one `TsmCache`, compile gain pairs,
and persist the result in the current log folder as `<stem>.arrow`.
"""
function compilegainsdf(tsm::TsmCache; stem::AbstractString="xchgains", folderpath::AbstractString=EnvConfig.logfolder())::DataFrame
    return compilegainsdf(collecttradesdf(tsm); stem=stem, folderpath=folderpath)
end

"""Return one gain-segment duration in minutes, inclusive of open and close rows."""
function _gainsegmentminutes(opentime::DateTime, closetime::DateTime)::Int
    @assert closetime >= opentime "closetime=$(closetime) must be >= opentime=$(opentime)"
    return Int(div(Dates.value(closetime - opentime), 60000)) + 1
end

"""Return the arithmetic mean for a non-empty vector."""
function _mean_nonempty(values::AbstractVector{<:Real})
    @assert !isempty(values) "values must be non-empty"
    return sum(values) / length(values)
end

"""Return the empirical 75th percentile via nearest-rank on a non-empty vector."""
function _q75_nonempty(values::AbstractVector{<:Real})
    @assert !isempty(values) "values must be non-empty"
    ordered = sort!(collect(values))
    rankix = Int(ceil(0.75 * length(ordered)))
    rankix = max(firstindex(ordered), min(lastindex(ordered), rankix))
    return ordered[rankix]
end

"""Return an empty gains report dataframe."""
function _emptygainsreportdf()::DataFrame
    return DataFrame(
        set=String[],
        avggain=Float32[],
        avgminutes=Float32[],
        q75minutes=Float32[],
        maxminutes=Int[],
        segments=Int[],
    )
end

"""
    gainsreport(gainsdf)

Aggregate gains across all pairs and ranges per set.
"""
function gainsreport(gainsdf::AbstractDataFrame)::DataFrame
    if nrow(gainsdf) == 0
        return _emptygainsreportdf()
    end

    @assert :opentime in propertynames(gainsdf) "gainsdf must contain :opentime; names=$(names(gainsdf))"
    @assert :closetime in propertynames(gainsdf) "gainsdf must contain :closetime; names=$(names(gainsdf))"
    @assert :gain in propertynames(gainsdf) "gainsdf must contain :gain; names=$(names(gainsdf))"

    reportinput = DataFrame(gainsdf; copycols=false)
    if :set ∉ propertynames(reportinput)
        reportinput[!, :set] = fill("all", nrow(reportinput))
    end

    reportinput[!, :minutes] = [_gainsegmentminutes(reportinput[ix, :opentime], reportinput[ix, :closetime]) for ix in 1:nrow(reportinput)]

    grouped = groupby(reportinput, :set; sort=true)
    report = combine(
        grouped,
        :gain => _mean_nonempty => :avggain,
        :minutes => _mean_nonempty => :avgminutes,
        :minutes => _q75_nonempty => :q75minutes,
        :minutes => maximum => :maxminutes,
        nrow => :segments,
    )
    sort!(report, :set)
    return report
end

"""
    gainsreport(; instem="xchgains", stem="xchgainsreport", folderpath=EnvConfig.logfolder())

Load `<instem>.arrow` from the current log folder, aggregate gains across all
pairs and ranges per set, persist `<stem>.arrow`, and return the report table.
"""
function gainsreport(; instem::AbstractString="xchgains", stem::AbstractString="xchgainsreport", folderpath::AbstractString=EnvConfig.logfolder())::DataFrame
    loaded = EnvConfig.readdf(String(instem); folderpath=String(folderpath))
    report = if isnothing(loaded)
        _emptygainsreportdf()
    else
        gainsreport(loaded)
    end
    EnvConfig.savedf(report, String(stem); folderpath=String(folderpath))
    return report
end
