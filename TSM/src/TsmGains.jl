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
        (col in propertynames(tradesdf)) && push!(sortcols, col)
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

"""Return the grouping columns used to compile gains from concatenated Trades rows.

Continuous replay isolates each pair and range, while partitioned replay additionally
isolates each set. A position open at a group boundary is therefore intentionally dropped
instead of being carried into the next range or set."""
function _compilegains_groupcols(tradesdf::AbstractDataFrame; setpartitions::Bool=false)::Vector{Symbol}
    @assert :pair in propertynames(tradesdf) "tradesdf must contain :pair to compile gains; names=$(names(tradesdf))"
    cols = Symbol[:pair]
    (:rangeid in propertynames(tradesdf)) && push!(cols, :rangeid)
    if setpartitions
        (:set in propertynames(tradesdf)) && push!(cols, :set)
    end
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
        closereason=String[],
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

"""Per-row Trades columns read while compiling gains; all are mandatory."""
const _COMPILEGAINS_HOTCOLUMNS = Symbol[:opentime, :lp_amount, :sp_amount, :lol_pavg, :lcl_pavg, :sol_pavg, :scl_pavg]

"""One opentime-ordered, pair-scoped partition with typed handles to its hot columns.

Gain compilation touches every row of every partition, so the hot columns are resolved
once here; `df[ix, :col]` would instead cost a column lookup plus a dynamic dispatch per
cell because `DataFrame` erases its column types."""
struct GainPartition
    rows::DataFrame
    opentime::Vector{DateTime}
    lp_amount::Vector{Float32}
    sp_amount::Vector{Float32}
    lol_pavg::Vector{Float32}
    lcl_pavg::Vector{Float32}
    sol_pavg::Vector{Float32}
    scl_pavg::Vector{Float32}
end

"""Project `tradesview` onto the columns gain compilation reads and order it by `opentime`."""
function GainPartition(tradesview::AbstractDataFrame)
    available = propertynames(tradesview)
    absent = Symbol[col for col in _COMPILEGAINS_HOTCOLUMNS if !(col in available)]
    @assert isempty(absent) "tradesdf must contain $(absent) to compile gains; names=$(names(tradesview))"
    @assert :pair in available "tradesdf must contain :pair to compile gains; names=$(names(tradesview))"

    keep = vcat(_COMPILEGAINS_HOTCOLUMNS, Symbol[:pair])
    for col in (:close, :closereason, :set, :rangeid)
        (col in available) && push!(keep, col)
    end
    rows = select(tradesview, keep)
    issorted(rows[!, :opentime]) || sort!(rows, :opentime)

    return GainPartition(
        rows,
        rows[!, :opentime]::Vector{DateTime},
        rows[!, :lp_amount]::Vector{Float32},
        rows[!, :sp_amount]::Vector{Float32},
        rows[!, :lol_pavg]::Vector{Float32},
        rows[!, :lcl_pavg]::Vector{Float32},
        rows[!, :sol_pavg]::Vector{Float32},
        rows[!, :scl_pavg]::Vector{Float32},
    )
end

"""Return the execution timestamp for one reflected position change row.

The position snapshot change is observed on row `ix`, but the execution
itself happened on the previous minute row whose bar triggered the fill.
"""
function _compilegainstime(part::GainPartition, ix::Integer)::DateTime
    @assert ix > 1 "compile gain timestamps require a previous row; got ix=$(ix)"
    return part.opentime[ix - 1]
end

"""Return the execution price stored on the position-change row for one order lane.

Falls back to this row's `close` price when the lane price is zero. A genuine data gap
can close a position without ever recording its own fill/liquidation price, and gains
compilation still needs a usable (if approximate) result rather than aborting the run."""
function _compilegainsprice(part::GainPartition, ix::Integer, prices::Vector{Float32}, pricecol::Symbol)::Float32
    price = prices[ix]
    (price > 0f0) && return price

    @assert :close in propertynames(part.rows) "tradesdf must contain :close to fall back for $(pricecol); names=$(names(part.rows))"
    fallback = part.rows[ix, :close]::Float32
    @assert fallback > 0f0 "Expected positive $(pricecol) or fallback :close on position change at ix=$(ix), opentime=$(_compilegainstime(part, ix)), pair=$(part.rows[ix, :pair]); got $(pricecol)=$(price), close=$(fallback)"
    return fallback
end

"""Append one compiled gain row, mirroring optional metadata columns from the partition."""
function _pushcompiledgain!(gainsdf::DataFrame, part::GainPartition, ix::Integer, opentime::DateTime, openprice::Float32, closetime::DateTime, closeprice::Float32, volume::Float32, side::Symbol, makerfee::Float32, takerfee::Float32)::Nothing
    rows = part.rows
    closereason = (:closereason in propertynames(rows)) ? String(rows[ix, :closereason]) : "none"
    closefee = closereason == "liquidation" ? takerfee : makerfee
    open_cost = volume * openprice * (1f0 + makerfee)
    open_proceeds = volume * openprice * (1f0 - makerfee)
    close_proceeds = volume * closeprice * (1f0 - closefee)
    close_cost = volume * closeprice * (1f0 + closefee)
    gainquote = side == :long ? close_proceeds - open_cost : open_proceeds - close_cost
    gain = gainquote / (volume * openprice)

    if (:set in propertynames(gainsdf)) && (:rangeid in propertynames(gainsdf))
        push!(gainsdf, (
            pair=String(rows[ix, :pair]),
            set=rows[ix, :set],
            rangeid=rows[ix, :rangeid],
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
            closereason=closereason,
        ))
    elseif :set in propertynames(gainsdf)
        push!(gainsdf, (
            pair=String(rows[ix, :pair]),
            set=rows[ix, :set],
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
            closereason=closereason,
        ))
    elseif :rangeid in propertynames(gainsdf)
        push!(gainsdf, (
            pair=String(rows[ix, :pair]),
            rangeid=rows[ix, :rangeid],
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
            closereason=closereason,
        ))
    else
        push!(gainsdf, (
            pair=String(rows[ix, :pair]),
            opentime=opentime,
            closetime=closetime,
            openprice=openprice,
            closeprice=closeprice,
            volume=volume,
            side=String(side),
            gain=gain,
            gainquote=gainquote,
            closereason=closereason,
        ))
    end
    return nothing
end

const _OpenTrade = NamedTuple{(:opentime, :openprice, :remaining), Tuple{DateTime, Float32, Float32}}

"""Queue one open execution for later FIFO close matching."""
function _enqueuecompiledopen!(openqueue::Vector{_OpenTrade}, opentime::DateTime, openprice::Float32, volume::Float32)::Nothing
    volume > 0f0 || return nothing
    push!(openqueue, (opentime=opentime, openprice=openprice, remaining=volume))
    return nothing
end

"""Consume one close execution against queued opens in FIFO order and emit gain rows."""
function _matchcompiledclose!(gainsdf::DataFrame, openqueue::Vector{_OpenTrade}, part::GainPartition, ix::Integer, closeprice::Float32, closevolume::Float32, side::Symbol, makerfee::Float32, takerfee::Float32)::Nothing
    remaining = closevolume
    closetime = _compilegainstime(part, ix)
    # Opens are queued from per-row Float32 deltas, so their sum drifts from the stored
    # position by a few ULPs per open. A residue at that scale is rounding, not a missing
    # open, and must not be matched or asserted on.
    tolerance = 64f0 * eps(Float32) * max(closevolume, 1f0)
    while remaining > tolerance
        @assert !isempty(openqueue) "Encountered unmatched $(side) close volume=$(remaining) at ix=$(ix), opentime=$(closetime), pair=$(part.rows[ix, :pair])"
        opentrade = first(openqueue)
        matched = min(opentrade.remaining, remaining)
        _pushcompiledgain!(gainsdf, part, ix, opentrade.opentime, opentrade.openprice, closetime, closeprice, matched, side, makerfee, takerfee)
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
function _compilegainspartition!(gainsdf::DataFrame, tradesview::AbstractDataFrame, makerfee::Float32, takerfee::Float32)::Nothing
    nrow(tradesview) == 0 && return nothing
    part = GainPartition(tradesview)
    longamounts = part.lp_amount
    shortamounts = part.sp_amount

    longopens = _OpenTrade[]
    shortopens = _OpenTrade[]

    for ix in eachindex(longamounts)
        longamount = longamounts[ix]
        shortamount = shortamounts[ix]
        @assert !((longamount > 0f0) && (shortamount > 0f0)) "Expected at most one open position side per row while compiling gains at ix=$(ix), opentime=$(part.opentime[ix]), pair=$(part.rows[ix, :pair]); got lp_amount=$(longamount), sp_amount=$(shortamount)"

        ix == 1 && continue

        longdelta = longamount - longamounts[ix - 1]
        shortdelta = shortamount - shortamounts[ix - 1]

        # Position changes on row `ix` indicate executions that happened in the prior minute.
        if longdelta > 0f0
            _enqueuecompiledopen!(longopens, _compilegainstime(part, ix), _compilegainsprice(part, ix, part.lol_pavg, :lol_pavg), longdelta)
        elseif longdelta < 0f0
            _matchcompiledclose!(gainsdf, longopens, part, ix, _compilegainsprice(part, ix, part.lcl_pavg, :lcl_pavg), -longdelta, :long, makerfee, takerfee)
        end

        if shortdelta > 0f0
            _enqueuecompiledopen!(shortopens, _compilegainstime(part, ix), _compilegainsprice(part, ix, part.sol_pavg, :sol_pavg), shortdelta)
        elseif shortdelta < 0f0
            _matchcompiledclose!(gainsdf, shortopens, part, ix, _compilegainsprice(part, ix, part.scl_pavg, :scl_pavg), -shortdelta, :short, makerfee, takerfee)
        end
    end
    return nothing
end

"""
    compilegains(tradesdf; setpartitions=false, makerfee=0f0, takerfee=0f0)

Compile open/close gain pairs from one Trades DataFrame without persisting them, scoping
matching by `pair` plus optional `set` and `rangeid`. `setpartitions=false` (default) is
for continuous replay data where one position can span set/rangeid subrange boundaries;
matching then scopes to `(pair, liquidityrangeid)` instead, so a position still cannot
span two distinct liquidity ranges. `gainsreport` still aggregates across all liquidity
ranges per set, i.e. reports out for the whole coin rather than per liquidity range. Set
`setpartitions=true` to instead scope matching exactly to `(pair, set, rangeid)`.

`gain` and `gainquote` include the opening maker fee and the closing maker fee, except
that liquidation closes use `takerfee`. Use this when compiling per pair and concatenating
afterwards; `compilegainsdf` wraps it with persistence.
"""
function compilegains(tradesdf::AbstractDataFrame; setpartitions::Bool=false, makerfee::Real=0f0, takerfee::Real=0f0)::DataFrame
    @assert makerfee >= 0 "makerfee=$(makerfee) must be nonnegative"
    @assert takerfee >= 0 "takerfee=$(takerfee) must be nonnegative"
    gainsdf = _emptygainsdf(tradesdf)
    nrow(tradesdf) == 0 && return gainsdf

    working = DataFrame(tradesdf; copycols=false)
    groupcols = _compilegains_groupcols(working; setpartitions=setpartitions)
    for tradesview in groupby(working, groupcols; sort=false)
        _compilegainspartition!(gainsdf, tradesview, Float32(makerfee), Float32(takerfee))
    end

    sortcols = Symbol[]
    (:set in propertynames(gainsdf)) && push!(sortcols, :set)
    (:rangeid in propertynames(gainsdf)) && push!(sortcols, :rangeid)
    append!(sortcols, [:pair, :opentime, :closetime])
    sort!(gainsdf, sortcols)
    return gainsdf
end

"""
    sortgainsdf!(gainsdf)

Order compiled gain rows by the canonical `set`/`rangeid`/`pair`/time key. Needed when
gain rows from several per-pair `compilegains` calls are concatenated.
"""
function sortgainsdf!(gainsdf::DataFrame)::DataFrame
    sortcols = Symbol[]
    (:set in propertynames(gainsdf)) && push!(sortcols, :set)
    (:rangeid in propertynames(gainsdf)) && push!(sortcols, :rangeid)
    append!(sortcols, [:pair, :opentime, :closetime])
    return sort!(gainsdf, sortcols)
end

"""
    compilegainsdf(tradesdf; stem="tsmgains", folderpath=EnvConfig.logfolder(), setpartitions=false, makerfee=0f0, takerfee=0f0)

Compile gain pairs via `compilegains`, including configured fees, and persist them as
`<stem>.arrow` in `folderpath`.
"""
function compilegainsdf(tradesdf::AbstractDataFrame; stem::AbstractString="tsmgains", folderpath::AbstractString=EnvConfig.logfolder(), setpartitions::Bool=false, makerfee::Real=0f0, takerfee::Real=0f0)::DataFrame
    gainsdf = compilegains(tradesdf; setpartitions=setpartitions, makerfee=makerfee, takerfee=takerfee)
    EnvConfig.savedf(gainsdf, String(stem); folderpath=String(folderpath))
    return gainsdf
end

"""
    compilegainsdf(tsm; stem="tsmgains", folderpath=EnvConfig.logfolder(), setpartitions=false, makerfee=0f0, takerfee=0f0)

Collect the combined Trades DataFrame from one `TsmCache`, compile gain pairs,
and persist the result in the current log folder as `<stem>.arrow`.
"""
function compilegainsdf(tsm::TsmCache; stem::AbstractString="tsmgains", folderpath::AbstractString=EnvConfig.logfolder(), setpartitions::Bool=false)::DataFrame
    return compilegainsdf(collecttradesdf(tsm); stem=stem, folderpath=folderpath, setpartitions=setpartitions)
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
function _emptygainsreportdf(groupcols::AbstractVector{Symbol}=[:set])::DataFrame
    report = DataFrame([col => String[] for col in groupcols])
    report[!, :avggain] = Float32[]
    report[!, :avgminutes] = Float32[]
    report[!, :q75minutes] = Int[]
    report[!, :maxminutes] = Int[]
    report[!, :segments] = Int[]
    report[!, :stoplosses] = Int[]
    report[!, :trendchanges] = Int[]
    report[!, :liquidations] = Int[]
    report[!, :takeprofits] = Int[]
    return report
end

"""
    gainsreport(gainsdf)

    Aggregate gains by `groupcols`, optionally adding totals grouped by `totalcols`.

The report includes `stoplosses` and `trendchanges`, counted from the canonical
`closereason` segment metadata. Positions still open at a partition end are not
compiled into gain segments and therefore cannot be classified as `endrange`.
"""
function gainsreport(gainsdf::AbstractDataFrame; groupcols::AbstractVector{Symbol}=[:set], totalcols::Union{Nothing, AbstractVector{Symbol}}=nothing)::DataFrame
    if nrow(gainsdf) == 0
        return _emptygainsreportdf(groupcols)
    end

    @assert :opentime in propertynames(gainsdf) "gainsdf must contain :opentime; names=$(names(gainsdf))"
    @assert :closetime in propertynames(gainsdf) "gainsdf must contain :closetime; names=$(names(gainsdf))"
    @assert :gain in propertynames(gainsdf) "gainsdf must contain :gain; names=$(names(gainsdf))"

    reportinput = DataFrame(gainsdf; copycols=false)
    if (:set in groupcols) && (:set ∉ propertynames(reportinput))
        reportinput[!, :set] = fill("all", nrow(reportinput))
    end
    for col in groupcols
        @assert col in propertynames(reportinput) "gainsdf must contain report grouping column :$(col); names=$(names(gainsdf))"
    end
    if isempty(groupcols)
        reportinput[!, :set] = fill("all", nrow(reportinput))
        groupcols = [:set]
    end

    reportinput[!, :minutes] = [_gainsegmentminutes(reportinput[ix, :opentime], reportinput[ix, :closetime]) for ix in 1:nrow(reportinput)]

    report = _aggregategainsreport(reportinput, groupcols)
    if !isnothing(totalcols) && !isempty(totalcols) && length(unique(reportinput[!, :pair])) > 1
        for col in totalcols
            @assert col in propertynames(reportinput) "gainsdf must contain total grouping column :$(col); names=$(names(gainsdf))"
        end
        totals = _aggregategainsreport(reportinput, collect(totalcols))
        insertcols!(totals, 1, :pair => fill("total", nrow(totals)))
        report = vcat(report, totals; cols=:union)
    elseif !isnothing(totalcols) && isempty(totalcols) && length(unique(reportinput[!, :pair])) > 1
        total = _aggregategainsreport(reportinput, Symbol[])
        insertcols!(total, 1, :pair => ["total"])
        report = vcat(report, total; cols=:union)
    end
    if :set in propertynames(report)
        set_strings = [ismissing(v) ? "all" : String(v) for v in report[!, :set]]
        report[!, :set] = set_strings
        report[!, :set] = categorical(report[!, :set]; levels=unique(report[!, :set]), compress=true)
    end
    return report
end

function _aggregategainsreport(reportinput::DataFrame, groupcols::AbstractVector{Symbol})::DataFrame
    if isempty(groupcols)
        reportinput[!, :_all] = fill("all", nrow(reportinput))
        groupcols = [:_all]
    end
    grouped = groupby(reportinput, groupcols; sort=true)
    report = combine(
        grouped,
        :gain => _mean_nonempty => :avggain,
        :minutes => _mean_nonempty => :avgminutes,
        :minutes => _q75_nonempty => :q75minutes,
        :minutes => maximum => :maxminutes,
        nrow => :segments,
        :closereason => (reasons -> count(==("stoploss"), String.(reasons))) => :stoplosses,
        :closereason => (reasons -> count(==("trendchange"), String.(reasons))) => :trendchanges,
        :closereason => (reasons -> count(==("liquidation"), String.(reasons))) => :liquidations,
        :closereason => (reasons -> count(==("takeprofit"), String.(reasons))) => :takeprofits,
    )
    :_all in propertynames(report) && select!(report, Not(:_all))
    return report
end

"""
    gainsreport(; instem="tsmgains", stem="xchgainsreport", folderpath=EnvConfig.logfolder(), groupcols=[:set], totalcols=nothing)

Load `<instem>.arrow`, aggregate by `groupcols`, optionally add multi-pair totals
grouped by `totalcols`, persist `<stem>.arrow`, and return the report table.
"""
function gainsreport(; instem::AbstractString="tsmgains", stem::AbstractString="xchgainsreport", folderpath::AbstractString=EnvConfig.logfolder(), groupcols::AbstractVector{Symbol}=[:set], totalcols::Union{Nothing, AbstractVector{Symbol}}=nothing)::DataFrame
    loaded = EnvConfig.readdf(String(instem); folderpath=String(folderpath))
    report = if isnothing(loaded)
        _emptygainsreportdf(groupcols)
    else
        gainsreport(loaded; groupcols=groupcols, totalcols=totalcols)
    end
    EnvConfig.savedf(report, String(stem); folderpath=String(folderpath))
    return report
end
