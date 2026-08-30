module TSM
# akronym for Trade State Machine

using DataFrames, CategoricalArrays, Dates
using EnvConfig
using Targets

# Intentionally no exports: call public API via TSM.<name> to avoid namespace clashes.

"""Pair-state owner for Trades DataFrames and the cached one-row row template."""
mutable struct TsmCache
    pairstates::Dict{String, DataFrame}
    tradesrowtemplate::DataFrame
    schema_contributors::Vector{Function}
    # Per-pair next-row-to-consume cursor (1-based). Both bulk-seeded replay rows and
    # live ticks are strictly time-ordered, so a single forward cursor finds/creates the
    # right row in O(1) without any per-timestamp index.
    nextrowix::Dict{String, Int}

    function TsmCache(; schema_contributors::Vector{Function}=Function[])
        return new(Dict{String, DataFrame}(), DataFrame(), Function[schema_contributors...], Dict{String, Int}())
    end
end

const TsmCacche = TsmCache

const TSM_NO_ORDER_ID = "none"
const TSM_NO_ORDER_MSG = "none"
const TSM_NO_CONFIG = "none"
const TSM_NO_STATE = "none"
const TSM_NO_SET = "none"
"Value every categorical Trades column carries while unset; shared by all of them."
const TSM_CATEGORICAL_DEFAULT = "none"
const TSM_STATUS_LEVELS = ["none", "submitted", "closed", "cancelled", "rejected"]
const TSM_CATEGORICAL_COLUMNS = Set([:pair, :set, :lo_id, :lo_status, :lo_msg, :lol_id, :lol_status, :lol_msg, :lc_id, :lc_status, :lc_msg, :lcl_id, :lcl_status, :lcl_msg, :lcsl_id, :lcsl_status, :lcsl_msg, :so_id, :so_status, :so_msg, :sol_id, :sol_status, :sol_msg, :sc_id, :sc_status, :sc_msg, :scl_id, :scl_status, :scl_msg, :scsl_id, :scsl_status, :scsl_msg, :config, :tsmstate])
const TSM_FLOAT_COLUMNS = Set([:lol_filled, :lol_pavg, :lcl_filled, :lcl_pavg, :sol_filled, :sol_pavg, :scl_filled, :scl_pavg, :lp_amount, :sp_amount, :close, :high, :low, :equity, :freemargin, :freequote, :score, :lo_limit, :lc_limit, :so_limit, :sc_limit, :lcsl_limit, :scsl_limit, :lo_amount, :lc_amount, :so_amount, :sc_amount])
const TSM_INT_COLUMNS = Set([:rangeid])
const TSM_TRADE_LANES = Set([:lo, :lc, :so, :sc])
"Order id columns; unbounded cardinality requires uncompressed categoricals (see `_uncompressedcategorical`)."
const TSM_ID_COLUMNS = Set([:lo_id, :lol_id, :lc_id, :lcl_id, :lcsl_id, :so_id, :sol_id, :sc_id, :scl_id, :scsl_id])

"Canonical categorical Trades column with a compressed UInt8 level pool (status/msg/config/set/tsmstate/pair)."
const TradesCat8Column = CategoricalVector{String, UInt8, String, CategoricalValue{String, UInt8}, Union{}}
"Canonical categorical Trades column with an uncompressed UInt32 level pool (order id columns)."
const TradesCat32Column = CategoricalVector{String, UInt32, String, CategoricalValue{String, UInt32}, Union{}}

const RANGEID_SUBRANGE_SPAN = EnvConfig.RANGEID_SUBRANGE_SPAN

"Return the liquidity range id owning subrange `rangeid` (see `EnvConfig.RANGEID_SUBRANGE_SPAN`)."
liquidityrangeid(rangeid::Integer) = fld(rangeid, RANGEID_SUBRANGE_SPAN) * RANGEID_SUBRANGE_SPAN

"""Map one trade label (or lane symbol) to its canonical lane symbol (`:lo`, `:lc`, `:so`, `:sc`)."""
function tradelane(label)::Symbol
    if label isa Symbol
        lane = Symbol(lowercase(String(label)))
        @assert lane in TSM_TRADE_LANES "unsupported trade lane=$(lane); supported lanes are $(collect(TSM_TRADE_LANES))"
        return lane
    end

    tl = label isa TradeLabel ? label : Targets.tradelabel(String(label))
    if tl === longopen || tl === longstrongopen
        return :lo
    elseif tl === longclose || tl === longstrongclose
        return :lc
    elseif tl === shortopen || tl === shortstrongopen
        return :so
    elseif tl === shortclose || tl === shortstrongclose
        return :sc
    end

    @assert false "trade label $(tl) does not map to a lane; expected open/close labels"
end

"""Precomputed lane column names, keyed by `(laneprefix, suffix)`.

Building these with `Symbol(lane, "_", suffix)` per call interns a new symbol on every cell
access, which dominates the replay row loop."""
const _LANE_COLUMN = Dict{Tuple{Symbol, Symbol}, Symbol}(
    (Symbol(lane, part), suffix) => Symbol(lane, part, "_", suffix)
    for lane in (:lo, :lc, :so, :sc)
    for part in ("", "l", "sl")
    for suffix in (:id, :status, :msg, :limit, :amount, :filled, :pavg)
)
"""Return the Trades column name for one lane prefix and suffix."""
@inline function _lanecolumn(laneprefix::Symbol, suffix::Symbol)::Symbol
    field = get(_LANE_COLUMN, (laneprefix, suffix), nothing)
    @assert !isnothing(field) "no Trades column for lane prefix=$(laneprefix) and suffix=$(suffix)"
    return field
end

"""Last-lane (fill state) column prefix per order lane."""
const _LASTLANE_PREFIX = Dict{Symbol, Symbol}(:lo => :lol, :lc => :lcl, :so => :sol, :sc => :scl)

"""Stop-loss bracket leg column prefix per close lane."""
const _STOPLANE_PREFIX = Dict{Symbol, Symbol}(:lc => :lcsl, :sc => :scsl)

"""Return the last-lane column prefix for one trade label."""
@inline function _lastlaneprefix(label)::Symbol
    lane = tradelane(label)
    prefix = get(_LASTLANE_PREFIX, lane, nothing)
    @assert !isnothing(prefix) "no last-lane prefix for lane=$(lane)"
    return prefix
end

"""Map one close label to the stop-loss bracket leg column for `suffix` (`:id`, `:status`, `:msg`, `:limit`, `:amount`). The stop leg shares the close lane (`lc`/`sc`) as the second leg of its bracket."""
function _stoplanefield(label, suffix::Symbol)::Symbol
    lane = tradelane(label)
    prefix = get(_STOPLANE_PREFIX, lane, nothing)
    @assert !isnothing(prefix) "stop-loss bracket leg requires a close lane, got lane=$(lane)"
    return _lanecolumn(prefix, suffix)
end

_nrows(df::AbstractDataFrame) = nrow(df)

function _assert_row_bounds(tradesdf::AbstractDataFrame, ix::Integer, field::Symbol)
    @assert 1 <= ix <= nrow(tradesdf) "$(field): ix=$(ix) is out of bounds for trades rows=$(nrow(tradesdf))"
    return nothing
end

function _assert_hasfield(tradesdf::AbstractDataFrame, field::Symbol)
    # hasproperty hits the column index directly; `field in propertynames(df)` would
    # allocate a fresh 65-element name vector on every cell access.
    @assert hasproperty(tradesdf, field) "tradesdf must contain $(field); names=$(names(tradesdf))"
    return nothing
end

function _compressedcategorical(values; levels=nothing)
    if isnothing(levels)
        return categorical(values; compress=true)
    end
    return categorical(values; levels=levels, compress=true)
end

"Order id columns carry unbounded cardinality (one level per exchange order id), so they must stay uncompressed to avoid the compressed pool reftype overflowing during long runs."
function _uncompressedcategorical(values; levels=nothing)
    if isnothing(levels)
        return categorical(values; compress=false)
    end
    return categorical(values; levels=levels, compress=false)
end

function _ensurecategoricallevel!(col, value::AbstractString)
    level = String(value)
    if !(level in levels(col))
        levels!(col, vcat(levels(col), [level]))
    end
    return col
end

function _setcategoricalcell!(tradesdf::DataFrame, field::Symbol, ix::Integer, value)
    col = tradesdf[!, field]
    sval = String(value)
    _ensurecategoricallevel!(col, sval)
    col[ix] = sval
    return tradesdf
end

function _schema_contributors(tsm::TsmCache)::Vector{Function}
    return isempty(tsm.schema_contributors) ? tradesdf_all_contributors() : tsm.schema_contributors
end

tradingpairkey(base::AbstractString, quotecoin::AbstractString)::String = uppercase(String(base)) * uppercase(String(quotecoin))

function _buildtradesrowtemplate(tsm::TsmCache)::DataFrame
    template = DataFrame(opentime=[DateTime(1970, 1, 1)])
    _applytradescontributors!(tsm, template)
    return template
end

function _tradesrowtemplate!(tsm::TsmCache)::DataFrame
    if nrow(tsm.tradesrowtemplate) != 1
        tsm.tradesrowtemplate = _buildtradesrowtemplate(tsm)
    end
    return tsm.tradesrowtemplate
end

function _appendtradesrow!(tsm::TsmCache, tdf::DataFrame, pairkey::AbstractString, opentime::DateTime)::DataFrame
    rowdf = DataFrame(_tradesrowtemplate!(tsm); copycols=true)
    rowdf[1, :opentime] = opentime
    if :pair in propertynames(rowdf)
        rowdf[1, :pair] = uppercase(String(pairkey))
    end
    push!(tdf, rowdf[1, :]; cols=:subset)
    return tdf
end

"Insert one fresh row for `opentime` at position `at`, shifting later rows down, to keep tdf time-ordered when a live-loop gap falls before an already-seeded future row."
function _inserttradesrow!(tsm::TsmCache, tdf::DataFrame, pairkey::AbstractString, opentime::DateTime, at::Integer)::DataFrame
    rowdf = DataFrame(_tradesrowtemplate!(tsm); copycols=true)
    rowdf[1, :opentime] = opentime
    if :pair in propertynames(rowdf)
        rowdf[1, :pair] = uppercase(String(pairkey))
    end
    newdf = vcat(tdf[1:(at - 1), :], rowdf, tdf[at:end, :]; cols=:setequal)
    tsm.pairstates[pairkey] = newdf
    return newdf
end

"""Extend one pair's Trades frame so every minute up to `enddt` already has a row.

A scheduled `tradeselection!` defines an epoch, so the rows of that epoch can be allocated
once instead of being grown per tick: growing mid-loop either copies the whole frame
(`_inserttradesrow!`) or invalidates held `TradesColumns`. Rows added here keep their
schema defaults, so `score == 0f0` and `tsmstate == TSM_NO_STATE` mark a minute the loop
never processed - a data gap, exchange downtime, or simply a not-yet-reached minute.

`enddt === nothing` is the live case (`XchCache.enddt` is open ended): the epoch then spans
`epochminutes` beyond the last stored row, and the next scheduled `tradeselection!` extends
it again. Minutes missing *inside* the stored range are filled too, because a seeded replay
source only carries liquid minutes. Existing row values are preserved. Returns the frame so
callers can build their column handles from it."""
function preparetradesepoch!(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString, enddt::Union{Nothing, DateTime}; startdt::Union{Nothing, DateTime}=nothing, epochminutes::Integer=0)::DataFrame
    pairkey = tradingpairkey(base, quotecoin)
    tdf = trades(tsm, pairkey)
    lastdt = nrow(tdf) > 0 ? tdf[nrow(tdf), :opentime] : nothing
    epochend = if isnothing(enddt)
        @assert epochminutes > 0 "preparetradesepoch! needs epochminutes>0 for live (enddt===nothing) pair=$(pairkey)"
        anchor = isnothing(lastdt) ? startdt : lastdt
        @assert !isnothing(anchor) "preparetradesepoch! needs startdt for the empty pair=$(pairkey)"
        floor(anchor, Minute(1)) + Minute(epochminutes)
    else
        floor(enddt, Minute(1))
    end
    # An epoch only ever extends; a shorter enddt must not drop already stored rows.
    isnothing(lastdt) || (epochend = max(epochend, lastdt))
    gridstart = if nrow(tdf) > 0
        tdf[1, :opentime]
    else
        @assert !isnothing(startdt) "preparetradesepoch! needs startdt for the empty pair=$(pairkey)"
        floor(startdt, Minute(1))
    end
    grid = collect(gridstart:Minute(1):epochend)
    (length(grid) == nrow(tdf)) && (nrow(tdf) == 0 || tdf[!, :opentime] == grid) && return tdf

    if nrow(tdf) == 0
        for opentime in grid
            _appendtradesrow!(tsm, tdf, pairkey, opentime)
        end
        return tdf
    end

    # Rows are missing inside the stored range, so the frame is rebuilt once on the full
    # grid; per-tick insertion would copy the whole frame for every missing minute.
    # Columns are built from `_defaultcolumn` and the stored values scattered in, because a
    # join would collapse the eltype of columns whose default is `missing`.
    gridpos = Dict{DateTime, Int}(dt => i for (i, dt) in enumerate(grid))
    target = [gridpos[dt] for dt in tdf[!, :opentime]]
    rebuilt = DataFrame(opentime=grid)
    for field in propertynames(tdf)
        field === :opentime && continue
        newcol = _defaultcolumn(field, length(grid))
        oldcol = tdf[!, field]
        for (source, dest) in enumerate(target)
            newcol[dest] = oldcol[source]
        end
        rebuilt[!, field] = newcol
    end
    settrades!(tsm, pairkey, rebuilt)
    return trades(tsm, pairkey)
end

"""Extend every pair of `pairs` to cover the epoch ending at `enddt`."""
function preparetradesepoch!(tsm::TsmCache, pairs, quotecoin::AbstractString, enddt::Union{Nothing, DateTime}; startdt::Union{Nothing, DateTime}=nothing, epochminutes::Integer=0)::Nothing
    for base in pairs
        preparetradesepoch!(tsm, String(base), quotecoin, enddt; startdt=startdt, epochminutes=epochminutes)
    end
    return nothing
end

"""Apply one contributor set to a Trades dataframe using the TSM cache schema."""
function _applytradescontributors!(tsm::TsmCache, df::DataFrame=DataFrame())::DataFrame
    return _applytradescontributors!(tsm, df, _schema_contributors(tsm))
end

function _applytradescontributors!(tsm::TsmCache, df::DataFrame, contributors::Vector{Function})::DataFrame
    for contributor in contributors
        contributor(df)
    end
    return df
end

"""Ensure per-row Trades identity metadata (`pair`) is populated."""
function _ensuretradesidentity!(df::DataFrame, pairkey::AbstractString)::DataFrame
    pkey = uppercase(String(pairkey))

    values = if :pair ∉ propertynames(df)
        fill(pkey, nrow(df))
    else
        [
            (ismissing(v) || isempty(strip(String(v))) || (uppercase(strip(String(v))) == "NONE")) ? pkey : String(v)
            for v in df[!, :pair]
        ]
    end
    df[!, :pair] = _compressedcategorical(values)

    return df
end

"""Register Trades schema contributor functions used to materialize pair state rows."""
function ensuretradesschema!(tsm::TsmCache, contributors)::TsmCache
    tsm.schema_contributors = Function[contributors...]
    for pair in keys(tsm.pairstates)
        _applytradescontributors!(tsm, tsm.pairstates[pair])
    end
    tsm.tradesrowtemplate = _buildtradesrowtemplate(tsm)
    return tsm
end

"""Return an empty Trades dataframe with the exchange-owned minimum schema."""
function _emptytradesv1df()::DataFrame
    df = DataFrame(opentime=DateTime[])
    for contributor in xch_tradesdf_contributors()
        contributor(df)
    end
    return df
end

"""Return the stored Trades dataframe for one pair, creating an empty one when missing."""
function trades(tsm::TsmCache, pair::AbstractString)::DataFrame
    key = uppercase(String(pair))
    return get!(tsm.pairstates, key) do
        _applytradescontributors!(tsm, _emptytradesv1df())
    end
end

"""Return the stored Trades dataframe for one `(base, quotecoin)` pair."""
function trades(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString)::DataFrame
    return trades(tsm, tradingpairkey(base, quotecoin))
end

"""Assert a stored Trades frame owns its columns.

A view-backed column aliases the caller's frame, so every write in a row loop would mutate
the source data instead of the Trades state."""
function _assert_owned_columns(df::DataFrame, pairkey::AbstractString)
    for col in propertynames(df)
        column = df[!, col]
        @assert !(column isa SubArray) "Trades column $(col) for pair=$(pairkey) is a $(typeof(column)) view; store an owning DataFrame (copycols=true) so writes cannot alias the source"
    end
    return nothing
end

"""Store one Trades dataframe for a pair and return the cache."""
function settrades!(tsm::TsmCache, pair::AbstractString, df::AbstractDataFrame)
    normalized = DataFrame(df; copycols=false)
    _applytradescontributors!(tsm, normalized)
    pairkey = uppercase(String(pair))
    _ensuretradesidentity!(normalized, pairkey)
    _assert_owned_columns(normalized, pairkey)
    tsm.pairstates[pairkey] = normalized
    tsm.nextrowix[pairkey] = 1
    return tsm
end

"""Store one Trades dataframe for a pair and return the cache."""
function settrades!(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString, df::AbstractDataFrame)
    pairkey = tradingpairkey(base, quotecoin)
    normalized = DataFrame(df; copycols=false)
    _applytradescontributors!(tsm, normalized)
    _ensuretradesidentity!(normalized, pairkey)
    _assert_owned_columns(normalized, pairkey)
    tsm.pairstates[pairkey] = normalized
    tsm.nextrowix[pairkey] = 1
    return tsm
end

"""Return the stored pair keys in deterministic order."""
function tradingpairs(tsm::TsmCache)::Vector{String}
    return sort!(collect(keys(tsm.pairstates)))
end

"""
Prime the per-pair row cursor to the row matching `opentime` in an already-seeded pair
dataframe, so the next `ensuretradesrow!` call resumes there instead of restarting from
row 1. Used to resume an interrupted tradesim run mid-dataframe.
"""
function primenextrowix!(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString, opentime::DateTime)::Int
    pairkey = tradingpairkey(uppercase(String(base)), quotecoin)
    tdf = trades(tsm, pairkey)
    ix = findfirst(==(opentime), tdf[!, :opentime])
    @assert !isnothing(ix) "cannot prime cursor: opentime=$(opentime) not found in seeded trades for pair=$(pairkey)"
    tsm.nextrowix[pairkey] = ix - 1
    return ix
end

"""
Row index (1-based) of the last row whose `tsmstate` differs from `TSM_NO_STATE`,
or `0` when no row was ever visited. Used to resume an interrupted run: everything
strictly before this row is fully processed; this row itself is reprocessed because
an interruption most likely happened mid-processing of it.
"""
function lastcheckpointedrowindex(checkpoint::AbstractDataFrame)::Int
    (:tsmstate in propertynames(checkpoint)) || return 0
    ix = findlast(!=(TSM_NO_STATE), String.(checkpoint[!, :tsmstate]))
    return isnothing(ix) ? 0 : ix
end

"""
Overwrite rows `1:prefixn` of a freshly schema-normalized `tradesdf` with the matching
rows of a previously persisted `checkpoint` dataframe (same row order/opentimes
assumed), restoring exchange/account/strategy state for rows already fully processed
in an earlier, interrupted tradesim run.
"""
function restorecheckpointrows!(tradesdf::DataFrame, checkpoint::AbstractDataFrame, prefixn::Integer)::DataFrame
    prefixn <= 0 && return tradesdf
    @assert prefixn <= nrow(tradesdf) "prefixn=$(prefixn) exceeds nrow(tradesdf)=$(nrow(tradesdf))"
    @assert prefixn <= nrow(checkpoint) "prefixn=$(prefixn) exceeds nrow(checkpoint)=$(nrow(checkpoint))"
    for col in propertynames(checkpoint)
        (col in propertynames(tradesdf)) || continue
        srccol = checkpoint[!, col]
        if col === :label
            restored = [v isa TradeLabel ? v : Targets.tradelabel(String(v)) for v in srccol[1:prefixn]]
            tradesdf[1:prefixn, col] = restored
            continue
        end
        if tradesdf[!, col] isa CategoricalArray
            for ix in 1:prefixn
                _setcategoricalcell!(tradesdf, col, ix, srccol[ix])
            end
        else
            tradesdf[1:prefixn, col] = srccol[1:prefixn]
        end
    end
    return tradesdf
end

"""Return true when the TSM cache already tracks one pair state entry."""
function haspairstate(tsm::TsmCache, pair::AbstractString)::Bool
    return haskey(tsm.pairstates, uppercase(String(pair)))
end

"""Drop one pair state entry from the TSM cache."""
function droppair!(tsm::TsmCache, pair::AbstractString)::Nothing
    pairkey = uppercase(String(pair))
    delete!(tsm.pairstates, pairkey)
    delete!(tsm.nextrowix, pairkey)
    return nothing
end

"""Return the pair-state entry for one pair, creating it when missing."""
function getpairstate!(tsm::TsmCache, pair::AbstractString)::DataFrame
    return trades(tsm, pair)
end

"""Return the pair-state entry for one `(base, quotecoin)` pair."""
function getpairstate!(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString)::DataFrame
    return trades(tsm, base, quotecoin)
end

"""Return the row index for one sample timestamp if it already exists in the pair state."""
function tradesrowindex(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString, opentime::DateTime)::Union{Nothing, Int}
    pairkey = tradingpairkey(uppercase(String(base)), quotecoin)
    tdf = trades(tsm, pairkey)
    ix = findfirst(==(opentime), tdf[!, :opentime])
    return isnothing(ix) ? nothing : Int(ix)
end

"""Return the writable pair row for one sample timestamp, creating a row when needed."""
function ensuretradesrow!(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString, opentime::DateTime)
    basekey = uppercase(String(base))
    pairkey = tradingpairkey(basekey, quotecoin)
    tdf = trades(tsm, pairkey)
    n = nrow(tdf)

    # Both bulk-seeded replay rows and live ticks are strictly time-ordered, so a single
    # last-row cursor per pair suffices: repeat calls for the same opentime (e.g. a
    # caller re-fetching the current row) reuse the cursor row itself; otherwise look at
    # the next row to reuse an already-seeded match, insert-ahead for a live-loop gap, or
    # append once the cursor runs past seeded data.
    cursor = clamp(get(tsm.nextrowix, pairkey, 0), 0, n)
    if (cursor >= 1) && (tdf[cursor, :opentime] == opentime)
        rowix = cursor
    else
        nxt = cursor + 1
        if (nxt <= n) && (tdf[nxt, :opentime] == opentime)
            rowix = nxt
        elseif (nxt <= n) && (tdf[nxt, :opentime] > opentime)
            tdf = _inserttradesrow!(tsm, tdf, pairkey, opentime, nxt)
            rowix = nxt
        else
            tdf = _appendtradesrow!(tsm, tdf, pairkey, opentime)
            rowix = nrow(tdf)
        end
    end
    tsm.nextrowix[pairkey] = rowix

    tdf[rowix, :opentime] = opentime
    tdf[rowix, :pair] = pairkey
    (:tsmstate in propertynames(tdf)) && settrades_tsmstate!(tdf, rowix, "sync")
    return (tradesdf=tdf, rowix=Int(rowix))
end

function _defaultcolumn(field::Symbol, n::Integer)
    if field === :opentime
        return DateTime[]
    elseif field === :lastopentrade
        return Vector{Union{Missing, DateTime}}(missing, n)
    elseif field === :label
        return fill(ignore, n)
    elseif field === :lo_status || field === :lol_status || field === :lc_status || field === :lcl_status || field === :lcsl_status || field === :so_status || field === :sol_status || field === :sc_status || field === :scl_status || field === :scsl_status
        return _compressedcategorical(fill("none", n); levels=TSM_STATUS_LEVELS)
    elseif field === :lo_id || field === :lol_id || field === :lc_id || field === :lcl_id || field === :lcsl_id || field === :so_id || field === :sol_id || field === :sc_id || field === :scl_id || field === :scsl_id
        return _uncompressedcategorical(fill(TSM_NO_ORDER_ID, n); levels=[TSM_NO_ORDER_ID])
    elseif field === :lo_msg || field === :lol_msg || field === :lc_msg || field === :lcl_msg || field === :lcsl_msg || field === :so_msg || field === :sol_msg || field === :sc_msg || field === :scl_msg || field === :scsl_msg
        return _compressedcategorical(fill(TSM_NO_ORDER_MSG, n); levels=[TSM_NO_ORDER_MSG])
    elseif field === :pair || field === :set || field === :config || field === :tsmstate
        default = field === :pair ? "none" : field === :set ? TSM_NO_SET : field === :config ? TSM_NO_CONFIG : TSM_NO_STATE
        return _compressedcategorical(fill(default, n); levels=[default])
    elseif field in TSM_INT_COLUMNS
        return fill(Int32(0), n)
    elseif field in TSM_FLOAT_COLUMNS
        return fill(0f0, n)
    end
    throw(ArgumentError("unsupported trades column $(field)"))
end

"""Return `values` as a canonical Trades categorical column with reference type `R`.

Producers hand over pools that differ from the schema in eltype (`allowmissing!` leaves
`Union{Missing,String}`) or in reference width (Arrow dictionary encoding yields `UInt32`),
so adopted columns are rebuilt rather than accepted as-is. An unset cell - `push!` with
`cols=:subset` leaves `missing` - materializes to the column default."""
function _canonicalcategorical(field::Symbol, values, ::Type{R}) where {R <: Unsigned}
    strings = String[ismissing(v) ? TSM_CATEGORICAL_DEFAULT : String(v) for v in values]
    nlevels = length(Set(strings))
    @assert nlevels <= typemax(R) "tradesdf[$(field)] has $(nlevels) distinct values but its pool reference type $(R) holds at most $(typemax(R)) levels"
    return CategoricalArray{String, 1, R}(strings)
end

function _ensurecolumn!(tradesdf::DataFrame, field::Symbol)
    if field ∉ propertynames(tradesdf)
        tradesdf[!, field] = _defaultcolumn(field, nrow(tradesdf))
    elseif field === :label
        col = tradesdf[!, :label]
        if !(eltype(col) <: TradeLabel)
            @assert all(!ismissing(v) for v in col) "tradesdf[:label] contains missing values and cannot be normalized to TradeLabel"
            tradesdf[!, :label] = [v isa TradeLabel ? v : Targets.tradelabel(String(v)) for v in col]
        end
    elseif field in TSM_ID_COLUMNS
        col = tradesdf[!, field]
        (col isa TradesCat32Column) || (tradesdf[!, field] = _canonicalcategorical(field, col, UInt32))
    elseif field in TSM_CATEGORICAL_COLUMNS
        # a caller-supplied plain string column, a missing-allowing pool or a foreign reftype
        # would otherwise silently violate the schema
        col = tradesdf[!, field]
        (col isa TradesCat8Column) || (tradesdf[!, field] = _canonicalcategorical(field, col, UInt8))
    elseif field in TSM_INT_COLUMNS
        # EnvConfig compacts integer columns to their narrowest type on Arrow write, so a
        # reloaded frame can carry any width; the Trades schema is Int32.
        col = tradesdf[!, field]
        if !(col isa Vector{Int32})
            @assert !any(ismissing, col) "tradesdf[$(field)] contains missing values and cannot be normalized to Int32"
            @assert all(v -> typemin(Int32) <= Int(v) <= typemax(Int32), col) "tradesdf[$(field)] holds values outside Int32; extrema=$(extrema(col))"
            tradesdf[!, field] = Int32.(col)
        end
    elseif field in TSM_FLOAT_COLUMNS
        col = tradesdf[!, field]
        if !(col isa Vector{Float32})
            @assert !any(ismissing, col) "tradesdf[$(field)] contains missing values and cannot be normalized to Float32"
            tradesdf[!, field] = Float32.(col)
        end
    end
    return tradesdf
end

"""Return one Trades column with its concrete element type, using a single index lookup.

`tradesdf[ix, field]` instead costs two index lookups, a bounds check that re-derives
`nrow` through a dynamic dispatch (`DataFrame` stores no row count), and - on writes -
non-note metadata invalidation. Resolving the column once avoids all of that and makes
the subsequent element access statically dispatched."""
@inline function _tradescolumn(tradesdf::DataFrame, field::Symbol, ::Type{T})::Vector{T} where {T}
    return tradesdf[!, field]::Vector{T}
end

@inline function _assert_col_bounds(col::AbstractVector, ix::Integer, field::Symbol)
    @assert 1 <= ix <= length(col) "$(field): ix=$(ix) is out of bounds for trades rows=$(length(col))"
    return nothing
end

function _categorical_setter!(tradesdf::DataFrame, field::Symbol, ix::Integer, value)
    _assert_hasfield(tradesdf, field)
    _assert_col_bounds(tradesdf[!, field], ix, field)
    return _setcategoricalcell!(tradesdf, field, ix, value)
end

function _float_getter(tradesdf::AbstractDataFrame, ix::Integer, field::Symbol)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    return tradesdf[ix, field]
end

function _float_getter(tradesdf::DataFrame, ix::Integer, field::Symbol)
    col = _tradescolumn(tradesdf, field, Float32)
    _assert_col_bounds(col, ix, field)
    return @inbounds col[ix]
end

function _float_setter!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    col = _tradescolumn(tradesdf, field, Float32)
    _assert_col_bounds(col, ix, field)
    @inbounds col[ix] = value
    return tradesdf
end

function _int_getter(tradesdf::AbstractDataFrame, ix::Integer, field::Symbol)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    return tradesdf[ix, field]
end

function _int_getter(tradesdf::DataFrame, ix::Integer, field::Symbol)
    col = _tradescolumn(tradesdf, field, Int32)
    _assert_col_bounds(col, ix, field)
    return @inbounds col[ix]
end

function _int_setter!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    col = _tradescolumn(tradesdf, field, Int32)
    _assert_col_bounds(col, ix, field)
    @inbounds col[ix] = value
    return tradesdf
end

function _datetime_setter!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    # :opentime and :lastopentrade differ in whether they admit missing, so stay untyped here.
    col = tradesdf[!, field]
    _assert_col_bounds(col, ix, field)
    @inbounds col[ix] = value
    return tradesdf
end

function _label_setter!(tradesdf::DataFrame, ix::Integer, value)
    col = _tradescolumn(tradesdf, :label, TradeLabel)
    _assert_col_bounds(col, ix, :label)
    @inbounds col[ix] = value isa TradeLabel ? value : Targets.tradelabel(String(value))
    return tradesdf
end

"""Get one lane-scoped trades field cell using a trade label and suffix (for example `:limit`, `:amount`, `:id`)."""
function gettrades_lanefield(tradesdf::AbstractDataFrame, ix::Integer, label, suffix::Symbol)
    return gettradesfield(tradesdf, ix, _lanecolumn(tradelane(label), suffix))
end

"""Set one lane-scoped trades field cell using a trade label and suffix (for example `:limit`, `:amount`, `:id`)."""
function settrades_lanefield!(tradesdf::DataFrame, ix::Integer, label, suffix::Symbol, value)
    return settradesfield!(tradesdf, ix, _lanecolumn(tradelane(label), suffix), value)
end

"""Get one last-lane trades field cell using a trade label and suffix (for example `:id`, `:status`, `:msg`)."""
function gettrades_lastlanefield(tradesdf::AbstractDataFrame, ix::Integer, label, suffix::Symbol)
    return gettradesfield(tradesdf, ix, _lanecolumn(_lastlaneprefix(label), suffix))
end

"""Get one stop-loss bracket leg field cell of a close lane using a close label and suffix."""
function gettrades_stoplanefield(tradesdf::AbstractDataFrame, ix::Integer, label, suffix::Symbol)
    return gettradesfield(tradesdf, ix, _stoplanefield(label, suffix))
end

"""Set one stop-loss bracket leg field cell of a close lane using a close label and suffix."""
function settrades_stoplanefield!(tradesdf::DataFrame, ix::Integer, label, suffix::Symbol, value)
    return settradesfield!(tradesdf, ix, _stoplanefield(label, suffix), value)
end

"""Set one last-lane trades field cell using a trade label and suffix (for example `:id`, `:status`, `:msg`)."""
function settrades_lastlanefield!(tradesdf::DataFrame, ix::Integer, label, suffix::Symbol, value)
    return settradesfield!(tradesdf, ix, _lanecolumn(_lastlaneprefix(label), suffix), value)
end

"""Get lane order id (`lo/lc/so/sc`) addressed via a trade label."""
gettrades_id(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lanefield(tradesdf, ix, label, :id)
"""Set lane order id (`lo/lc/so/sc`) addressed via a trade label."""
settrades_id!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lanefield!(tradesdf, ix, label, :id, value)

"""Get lane order status (`lo/lc/so/sc`) addressed via a trade label."""
gettrades_status(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lanefield(tradesdf, ix, label, :status)
"""Set lane order status (`lo/lc/so/sc`) addressed via a trade label."""
settrades_status!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lanefield!(tradesdf, ix, label, :status, value)

"""Get lane order message (`lo/lc/so/sc`) addressed via a trade label."""
gettrades_msg(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lanefield(tradesdf, ix, label, :msg)
"""Set lane order message (`lo/lc/so/sc`) addressed via a trade label."""
settrades_msg!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lanefield!(tradesdf, ix, label, :msg, value)

"""Get lane request limit (`lo/lc/so/sc`) addressed via a trade label."""
gettrades_limit(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lanefield(tradesdf, ix, label, :limit)
"""Set lane request limit (`lo/lc/so/sc`) addressed via a trade label."""
settrades_limit!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lanefield!(tradesdf, ix, label, :limit, value)

"""Get lane request amount (`lo/lc/so/sc`) addressed via a trade label."""
gettrades_amount(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lanefield(tradesdf, ix, label, :amount)
"""Set lane request amount (`lo/lc/so/sc`) addressed via a trade label."""
settrades_amount!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lanefield!(tradesdf, ix, label, :amount, value)

"""Get last-lane order id (`lol/lcl/sol/scl`) addressed via a trade label."""
gettrades_last_id(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lastlanefield(tradesdf, ix, label, :id)
"""Set last-lane order id (`lol/lcl/sol/scl`) addressed via a trade label."""
settrades_last_id!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lastlanefield!(tradesdf, ix, label, :id, value)

"""Get last-lane order status (`lol/lcl/sol/scl`) addressed via a trade label."""
gettrades_last_status(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lastlanefield(tradesdf, ix, label, :status)
"""Set last-lane order status (`lol/lcl/sol/scl`) addressed via a trade label."""
settrades_last_status!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lastlanefield!(tradesdf, ix, label, :status, value)

"""Get last-lane order message (`lol/lcl/sol/scl`) addressed via a trade label."""
gettrades_last_msg(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lastlanefield(tradesdf, ix, label, :msg)
"""Set last-lane order message (`lol/lcl/sol/scl`) addressed via a trade label."""
settrades_last_msg!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lastlanefield!(tradesdf, ix, label, :msg, value)

"""Get last-lane filled amount (`lol/lcl/sol/scl`) addressed via a trade label."""
gettrades_last_filled(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lastlanefield(tradesdf, ix, label, :filled)
"""Set last-lane filled amount (`lol/lcl/sol/scl`) addressed via a trade label."""
settrades_last_filled!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lastlanefield!(tradesdf, ix, label, :filled, value)

"""Get last-lane average fill price (`lol/lcl/sol/scl`) addressed via a trade label."""
gettrades_last_pavg(tradesdf::AbstractDataFrame, ix::Integer, label) = gettrades_lastlanefield(tradesdf, ix, label, :pavg)
"""Set last-lane average fill price (`lol/lcl/sol/scl`) addressed via a trade label."""
settrades_last_pavg!(tradesdf::DataFrame, ix::Integer, label, value) = settrades_lastlanefield!(tradesdf, ix, label, :pavg, value)

"""Every Trades column, in canonical order. Drives both the generated per-field accessors and `TradesColumns`."""
const TSM_TRADES_COLUMNS = (:opentime, :lastopentrade, :pair, :set, :rangeid, :lo_id, :lo_status, :lol_id, :lol_status, :lol_filled, :lol_pavg, :lo_msg, :lol_msg, :lc_id, :lc_status, :lcl_id, :lcl_status, :lcl_filled, :lcl_pavg, :lc_msg, :lcl_msg, :lcsl_id, :lcsl_status, :lcsl_msg, :lcsl_limit, :so_id, :so_status, :sol_id, :sol_status, :sol_filled, :sol_pavg, :so_msg, :sol_msg, :sc_id, :sc_status, :scl_id, :scl_status, :scl_filled, :scl_pavg, :sc_msg, :scl_msg, :scsl_id, :scsl_status, :scsl_msg, :scsl_limit, :lp_amount, :sp_amount, :close, :high, :low, :equity, :freemargin, :freequote, :label, :score, :lo_limit, :lc_limit, :so_limit, :sc_limit, :lo_amount, :lc_amount, :so_amount, :sc_amount, :config, :tsmstate)

for field in TSM_TRADES_COLUMNS
    ensurefn = Symbol("ensuretrades_", field, "!")
    getfn = Symbol("gettrades_", field)
    setfn = Symbol("settrades_", field, "!")
    if field === :label
        @eval begin
            function $ensurefn(tradesdf::DataFrame)::DataFrame
                return _ensurecolumn!(tradesdf, $(QuoteNode(field)))
            end
            function $getfn(tradesdf::AbstractDataFrame, ix::Integer)
                _assert_row_bounds(tradesdf, ix, $(QuoteNode(field)))
                _assert_hasfield(tradesdf, $(QuoteNode(field)))
                return tradesdf[ix, $(QuoteNode(field))]
            end
            function $setfn(tradesdf::DataFrame, ix::Integer, value)
                return _label_setter!(tradesdf, ix, value)
            end
        end
    elseif field === :opentime || field === :lastopentrade
        @eval begin
            function $ensurefn(tradesdf::DataFrame)::DataFrame
                return _ensurecolumn!(tradesdf, $(QuoteNode(field)))
            end
            function $getfn(tradesdf::AbstractDataFrame, ix::Integer)
                _assert_row_bounds(tradesdf, ix, $(QuoteNode(field)))
                _assert_hasfield(tradesdf, $(QuoteNode(field)))
                return tradesdf[ix, $(QuoteNode(field))]
            end
            function $setfn(tradesdf::DataFrame, ix::Integer, value)
                return _datetime_setter!(tradesdf, ix, $(QuoteNode(field)), value)
            end
        end
    elseif field in TSM_CATEGORICAL_COLUMNS
        @eval begin
            function $ensurefn(tradesdf::DataFrame)::DataFrame
                return _ensurecolumn!(tradesdf, $(QuoteNode(field)))
            end
            function $getfn(tradesdf::AbstractDataFrame, ix::Integer)
                _assert_row_bounds(tradesdf, ix, $(QuoteNode(field)))
                _assert_hasfield(tradesdf, $(QuoteNode(field)))
                return tradesdf[ix, $(QuoteNode(field))]
            end
            function $setfn(tradesdf::DataFrame, ix::Integer, value)
                return _categorical_setter!(tradesdf, $(QuoteNode(field)), ix, value)
            end
        end
    elseif field in TSM_FLOAT_COLUMNS
        @eval begin
            function $ensurefn(tradesdf::DataFrame)::DataFrame
                return _ensurecolumn!(tradesdf, $(QuoteNode(field)))
            end
            function $getfn(tradesdf::AbstractDataFrame, ix::Integer)
                return _float_getter(tradesdf, ix, $(QuoteNode(field)))
            end
            function $setfn(tradesdf::DataFrame, ix::Integer, value)
                return _float_setter!(tradesdf, ix, $(QuoteNode(field)), value)
            end
        end
    elseif field in TSM_INT_COLUMNS
        @eval begin
            function $ensurefn(tradesdf::DataFrame)::DataFrame
                return _ensurecolumn!(tradesdf, $(QuoteNode(field)))
            end
            function $getfn(tradesdf::AbstractDataFrame, ix::Integer)
                return _int_getter(tradesdf, ix, $(QuoteNode(field)))
            end
            function $setfn(tradesdf::DataFrame, ix::Integer, value)
                return _int_setter!(tradesdf, ix, $(QuoteNode(field)), value)
            end
        end
    else
        error("unhandled trades field $(field)")
    end
end

"""Return Trades column contributor functions across Xch, TradingStrategy, Trade, and TSM-owned state."""
function tradesdf_all_contributors()::Vector{Function}
    return vcat(xch_tradesdf_contributors(), tradingstrategy_tradesdf_contributors(), trade_tradesdf_contributors(), tsm_tradesdf_contributors())
end

"""Apply the full Trades schema to one dataframe in place."""
function ensuretradeschema!(tradesdf::DataFrame)::DataFrame
    for contributor in tradesdf_all_contributors()
        contributor(tradesdf)
    end
    return tradesdf
end

"""Return Xch-owned Trades column contributor functions."""
function xch_tradesdf_contributors()::Vector{Function}
    return Function[
        xch_tradesdf_opentime,
        xch_tradesdf_pair,
        xch_tradesdf_lastopentrade,
        df -> xch_tradesdf_id(df, longopen),
        df -> xch_tradesdf_status(df, longopen),
        df -> xch_tradesdf_last_id(df, longopen),
        df -> xch_tradesdf_last_status(df, longopen),
        df -> xch_tradesdf_last_filled(df, longopen),
        df -> xch_tradesdf_last_pavg(df, longopen),
        df -> xch_tradesdf_msg(df, longopen),
        df -> xch_tradesdf_last_msg(df, longopen),
        df -> xch_tradesdf_id(df, longclose),
        df -> xch_tradesdf_status(df, longclose),
        df -> xch_tradesdf_last_id(df, longclose),
        df -> xch_tradesdf_last_status(df, longclose),
        df -> xch_tradesdf_last_filled(df, longclose),
        df -> xch_tradesdf_last_pavg(df, longclose),
        df -> xch_tradesdf_msg(df, longclose),
        df -> xch_tradesdf_last_msg(df, longclose),
        df -> xch_tradesdf_id(df, shortopen),
        df -> xch_tradesdf_status(df, shortopen),
        df -> xch_tradesdf_last_id(df, shortopen),
        df -> xch_tradesdf_last_status(df, shortopen),
        df -> xch_tradesdf_last_filled(df, shortopen),
        df -> xch_tradesdf_last_pavg(df, shortopen),
        df -> xch_tradesdf_msg(df, shortopen),
        df -> xch_tradesdf_last_msg(df, shortopen),
        df -> xch_tradesdf_id(df, shortclose),
        df -> xch_tradesdf_status(df, shortclose),
        df -> xch_tradesdf_last_id(df, shortclose),
        df -> xch_tradesdf_last_status(df, shortclose),
        df -> xch_tradesdf_last_filled(df, shortclose),
        df -> xch_tradesdf_last_pavg(df, shortclose),
        df -> xch_tradesdf_msg(df, shortclose),
        df -> xch_tradesdf_last_msg(df, shortclose),
        df -> xch_tradesdf_stop_id(df, longclose),
        df -> xch_tradesdf_stop_status(df, longclose),
        df -> xch_tradesdf_stop_msg(df, longclose),
        df -> xch_tradesdf_stop_id(df, shortclose),
        df -> xch_tradesdf_stop_status(df, shortclose),
        df -> xch_tradesdf_stop_msg(df, shortclose),
        xch_tradesdf_lp_amount,
        xch_tradesdf_sp_amount,
        xch_tradesdf_close,
        xch_tradesdf_high,
        xch_tradesdf_low,
        xch_tradesdf_equity,
        xch_tradesdf_freemargin,
        xch_tradesdf_freequote,
    ]
end

"""Return TradingStrategy-contributed Trades schema initializer functions."""
function tradingstrategy_tradesdf_contributors()::Vector{Function}
    return Function[
        tradingstrategy_tradesdf_label,
        tradingstrategy_tradesdf_score,
        df -> tradingstrategy_tradesdf_limit(df, longopen),
        df -> tradingstrategy_tradesdf_limit(df, longclose),
        df -> tradingstrategy_tradesdf_limit(df, shortopen),
        df -> tradingstrategy_tradesdf_limit(df, shortclose),
        df -> tradingstrategy_tradesdf_stop_limit(df, longclose),
        df -> tradingstrategy_tradesdf_stop_limit(df, shortclose),
    ]
end

"""Return Trade-contributed Trades schema initializer functions."""
function trade_tradesdf_contributors()::Vector{Function}
    return Function[
        df -> trade_tradesdf_amount(df, longopen),
        df -> trade_tradesdf_amount(df, longclose),
        df -> trade_tradesdf_amount(df, shortopen),
        df -> trade_tradesdf_amount(df, shortclose),
    ]
end

"""Return TSM-owned Trades schema initializer functions."""
function tsm_tradesdf_contributors()::Vector{Function}
    return Function[
        tsm_tradesdf_set,
        tsm_tradesdf_rangeid,
        tsm_tradesdf_config,
        tsm_tradesdf_tsmstate,
    ]
end

"""Ensure Trades column `set` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Denotes the logical run set (for example train/test/eval/production)."""
function tsm_tradesdf_set(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :set)
end

"""Ensure Trades column `rangeid` exists. Owner: TSM. Eltype: `Int32`. Denotes one consecutive liquidity range identifier within one pair data set."""
function tsm_tradesdf_rangeid(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :rangeid)
end

"""Ensure Trades column `opentime` exists. Owner: Xch. Eltype: `DateTime`. Note: Required unique and sorted timestamp derived from sample data. Represents the time stamp of the most recent fully closed minute as UTC."""
function xch_tradesdf_opentime(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :opentime)
end

"""Ensure Trades column `lastopentrade` exists. Owner: Xch. Eltype: `Union{Missing,DateTime}`. Note: Timestamp of the last open position trade, i.e. lp_amount or sp_amount increased; otherwise `missing`."""
function xch_tradesdf_lastopentrade(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lastopentrade)
end

"""Ensure Trades column `pair` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Identifier of the trading pair."""
function xch_tradesdf_pair(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :pair)
end

"""Ensure Trades lane column `<lane>_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: exchange provided id of currently active order; otherwise TSM_NO_ORDER_ID."""
function xch_tradesdf_id(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "_id"))
end

"""Ensure Trades lane column `<lane>_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: order status of currently active order as one of the following: TSM_NO_STATE, `submitted`, `closed`, `cancelled`, `rejected`."""
function xch_tradesdf_status(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "_status"))
end

"""Ensure Trades lane column `<lane>_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: rejection/error message text for the currently active order."""
function xch_tradesdf_msg(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "_msg"))
end

"""Ensure Trades last-lane column `<lane>l_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: exchange provided id of last minute active order; otherwise TSM_NO_ORDER_ID."""
function xch_tradesdf_last_id(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "l_id"))
end

"""Ensure Trades last-lane column `<lane>l_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: order status of the last minute active order as one of the following: TSM_NO_STATE, `submitted`, `closed`, `cancelled`, `rejected`."""
function xch_tradesdf_last_status(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "l_status"))
end

"""Ensure Trades last-lane column `<lane>l_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: rejection/error message text for the last minute active order."""
function xch_tradesdf_last_msg(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "l_msg"))
end

"""Ensure Trades last-lane column `<lane>l_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as default. Note: Filled/executed base quantity of the last minute active order."""
function xch_tradesdf_last_filled(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "l_filled"))
end

"""Ensure Trades last-lane column `<lane>l_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as default. Note: Average fill price in quote units of the last minute active order."""
function xch_tradesdf_last_pavg(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, Symbol(tradelane(label), "l_pavg"))
end

"""Ensure Trades column `lp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Long position amount of trading pair holdings."""
function xch_tradesdf_lp_amount(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lp_amount)
end

"""Ensure Trades close-bracket stop column `<lcsl|scsl>_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: exchange provided id of the resting stop-loss leg of the close bracket; otherwise TSM_NO_ORDER_ID."""
function xch_tradesdf_stop_id(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, _stoplanefield(label, :id))
end

"""Ensure Trades close-bracket stop column `<lcsl|scsl>_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: order status of the resting stop-loss leg of the close bracket."""
function xch_tradesdf_stop_status(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, _stoplanefield(label, :status))
end

"""Ensure Trades close-bracket stop column `<lcsl|scsl>_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: rejection/error message text for the stop-loss leg of the close bracket."""
function xch_tradesdf_stop_msg(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, _stoplanefield(label, :msg))
end

"""Ensure Trades close-bracket stop column `<lcsl|scsl>_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: stop-loss limit price of the close bracket; `0f0` means no stop-loss leg is requested. Both bracket legs cover the same quantity, tracked by `<lc|sc>_amount`."""
function tradingstrategy_tradesdf_stop_limit(df::DataFrame, label)::DataFrame
    return _ensurecolumn!(df, _stoplanefield(label, :limit))
end

"""Ensure Trades column `sp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Short position amount of trading pair holdings."""
function xch_tradesdf_sp_amount(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :sp_amount)
end

"""Ensure Trades column `close` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Last completed minute close price of the trading pair."""
function xch_tradesdf_close(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :close)
end

"""Ensure Trades column `high` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Last completed minute high price of trading pair."""
function xch_tradesdf_high(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :high)
end

"""Ensure Trades column `low` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Last completed minute low price of trading pair."""
function xch_tradesdf_low(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :low)
end

"""Ensure Trades column `equity` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Most recent equity in quote units as constraint for maximum relative allocation  of a trading pair."""
function xch_tradesdf_equity(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :equity)
end

"""Ensure Trades column `freemargin` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free account margin amount in quote units. Currently equal to freequote."""
function xch_tradesdf_freemargin(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :freemargin)
end

"""Ensure Trades column `freequote` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free account amount for orders in quote units."""
function xch_tradesdf_freequote(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :freequote)
end

"""Ensure Trades column `label` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: label represents the TradingStrategy trading advice."""
function tradingstrategy_tradesdf_label(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :label)
end

"""Ensure Trades column `score` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: likelihood of the label to be correct from TradingStrategy."""
function tradingstrategy_tradesdf_score(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :score)
end

"""Ensure Trades lane column `<lane>_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: order limit in case of a currently active order for that trade lane."""
function tradingstrategy_tradesdf_limit(tradesdf::DataFrame, label)::DataFrame
    return _ensurecolumn!(tradesdf, Symbol(tradelane(label), "_limit"))
end

"""Ensure Trades lane column `<lane>_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: if order amount > 0 then order shall be placed, otherwise not. If a close order amount > 0 then a close order shall be placed """
function trade_tradesdf_amount(tradesdf::DataFrame, label)::DataFrame
    return _ensurecolumn!(tradesdf, Symbol(tradelane(label), "_amount"))
end

"""Ensure Trades column `config` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Identifies the Trade configuration id. Any change in config, e.g. different openthresholds, shall result in a different config marker"""
function tsm_tradesdf_config(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :config)
end

"""Ensure Trades column `tsmstate` exists. Owner: TSM. Eltype: `CategoricalVector{String}`.
Per-row progression (each row is visited once per minute, `TSM_NO_STATE` default until visited):
- *sync*: the row becomes the active row for its minute; price/execution fields are synced; next is *request*
- *request*: TradingStrategy and `Trade.trade!` evaluate the row before handing it to Xch; next is *xch*
- *xch*: the row is handed to Xch for order request processing; terminal state, the row does not revisit *sync*
"""
function tsm_tradesdf_tsmstate(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :tsmstate)
end

function gettradesfield(tradesdf::AbstractDataFrame, ix::Integer, field::Symbol)
    if field === :opentime
        return gettrades_opentime(tradesdf, ix)
    elseif field === :lastopentrade
        return gettrades_lastopentrade(tradesdf, ix)
    elseif field === :pair
        return gettrades_pair(tradesdf, ix)
    elseif field === :set
        return gettrades_set(tradesdf, ix)
    elseif field === :rangeid
        return gettrades_rangeid(tradesdf, ix)
    elseif field === :lo_id
        return gettrades_lo_id(tradesdf, ix)
    elseif field === :lo_status
        return gettrades_lo_status(tradesdf, ix)
    elseif field === :lol_id
        return gettrades_lol_id(tradesdf, ix)
    elseif field === :lol_status
        return gettrades_lol_status(tradesdf, ix)
    elseif field === :lol_filled
        return gettrades_lol_filled(tradesdf, ix)
    elseif field === :lol_pavg
        return gettrades_lol_pavg(tradesdf, ix)
    elseif field === :lo_msg
        return gettrades_lo_msg(tradesdf, ix)
    elseif field === :lol_msg
        return gettrades_lol_msg(tradesdf, ix)
    elseif field === :lc_id
        return gettrades_lc_id(tradesdf, ix)
    elseif field === :lc_status
        return gettrades_lc_status(tradesdf, ix)
    elseif field === :lcl_id
        return gettrades_lcl_id(tradesdf, ix)
    elseif field === :lcl_status
        return gettrades_lcl_status(tradesdf, ix)
    elseif field === :lcl_filled
        return gettrades_lcl_filled(tradesdf, ix)
    elseif field === :lcl_pavg
        return gettrades_lcl_pavg(tradesdf, ix)
    elseif field === :lc_msg
        return gettrades_lc_msg(tradesdf, ix)
    elseif field === :lcl_msg
        return gettrades_lcl_msg(tradesdf, ix)
    elseif field === :lcsl_id
        return gettrades_lcsl_id(tradesdf, ix)
    elseif field === :lcsl_status
        return gettrades_lcsl_status(tradesdf, ix)
    elseif field === :lcsl_msg
        return gettrades_lcsl_msg(tradesdf, ix)
    elseif field === :lcsl_limit
        return gettrades_lcsl_limit(tradesdf, ix)
    elseif field === :so_id
        return gettrades_so_id(tradesdf, ix)
    elseif field === :so_status
        return gettrades_so_status(tradesdf, ix)
    elseif field === :sol_id
        return gettrades_sol_id(tradesdf, ix)
    elseif field === :sol_status
        return gettrades_sol_status(tradesdf, ix)
    elseif field === :sol_filled
        return gettrades_sol_filled(tradesdf, ix)
    elseif field === :sol_pavg
        return gettrades_sol_pavg(tradesdf, ix)
    elseif field === :so_msg
        return gettrades_so_msg(tradesdf, ix)
    elseif field === :sol_msg
        return gettrades_sol_msg(tradesdf, ix)
    elseif field === :sc_id
        return gettrades_sc_id(tradesdf, ix)
    elseif field === :sc_status
        return gettrades_sc_status(tradesdf, ix)
    elseif field === :scl_id
        return gettrades_scl_id(tradesdf, ix)
    elseif field === :scl_status
        return gettrades_scl_status(tradesdf, ix)
    elseif field === :scl_filled
        return gettrades_scl_filled(tradesdf, ix)
    elseif field === :scl_pavg
        return gettrades_scl_pavg(tradesdf, ix)
    elseif field === :sc_msg
        return gettrades_sc_msg(tradesdf, ix)
    elseif field === :scl_msg
        return gettrades_scl_msg(tradesdf, ix)
    elseif field === :scsl_id
        return gettrades_scsl_id(tradesdf, ix)
    elseif field === :scsl_status
        return gettrades_scsl_status(tradesdf, ix)
    elseif field === :scsl_msg
        return gettrades_scsl_msg(tradesdf, ix)
    elseif field === :scsl_limit
        return gettrades_scsl_limit(tradesdf, ix)
    elseif field === :lp_amount
        return gettrades_lp_amount(tradesdf, ix)
    elseif field === :sp_amount
        return gettrades_sp_amount(tradesdf, ix)
    elseif field === :close
        return gettrades_close(tradesdf, ix)
    elseif field === :high
        return gettrades_high(tradesdf, ix)
    elseif field === :low
        return gettrades_low(tradesdf, ix)
    elseif field === :equity
        return gettrades_equity(tradesdf, ix)
    elseif field === :freemargin
        return gettrades_freemargin(tradesdf, ix)
    elseif field === :freequote
        return gettrades_freequote(tradesdf, ix)
    elseif field === :label
        return gettrades_label(tradesdf, ix)
    elseif field === :score
        return gettrades_score(tradesdf, ix)
    elseif field === :lo_limit
        return gettrades_lo_limit(tradesdf, ix)
    elseif field === :lc_limit
        return gettrades_lc_limit(tradesdf, ix)
    elseif field === :so_limit
        return gettrades_so_limit(tradesdf, ix)
    elseif field === :sc_limit
        return gettrades_sc_limit(tradesdf, ix)
    elseif field === :lo_amount
        return gettrades_lo_amount(tradesdf, ix)
    elseif field === :lc_amount
        return gettrades_lc_amount(tradesdf, ix)
    elseif field === :so_amount
        return gettrades_so_amount(tradesdf, ix)
    elseif field === :sc_amount
        return gettrades_sc_amount(tradesdf, ix)
    elseif field === :config
        return gettrades_config(tradesdf, ix)
    elseif field === :tsmstate
        return gettrades_tsmstate(tradesdf, ix)
    end
    throw(ArgumentError("unsupported trades field $(field)"))
end

function settradesfield!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    if field === :opentime
        return settrades_opentime!(tradesdf, ix, value)
    elseif field === :lastopentrade
        return settrades_lastopentrade!(tradesdf, ix, value)
    elseif field === :pair
        return settrades_pair!(tradesdf, ix, value)
    elseif field === :set
        return settrades_set!(tradesdf, ix, value)
    elseif field === :rangeid
        return settrades_rangeid!(tradesdf, ix, value)
    elseif field === :lo_id
        return settrades_lo_id!(tradesdf, ix, value)
    elseif field === :lo_status
        return settrades_lo_status!(tradesdf, ix, value)
    elseif field === :lol_id
        return settrades_lol_id!(tradesdf, ix, value)
    elseif field === :lol_status
        return settrades_lol_status!(tradesdf, ix, value)
    elseif field === :lol_filled
        return settrades_lol_filled!(tradesdf, ix, value)
    elseif field === :lol_pavg
        return settrades_lol_pavg!(tradesdf, ix, value)
    elseif field === :lo_msg
        return settrades_lo_msg!(tradesdf, ix, value)
    elseif field === :lol_msg
        return settrades_lol_msg!(tradesdf, ix, value)
    elseif field === :lc_id
        return settrades_lc_id!(tradesdf, ix, value)
    elseif field === :lc_status
        return settrades_lc_status!(tradesdf, ix, value)
    elseif field === :lcl_id
        return settrades_lcl_id!(tradesdf, ix, value)
    elseif field === :lcl_status
        return settrades_lcl_status!(tradesdf, ix, value)
    elseif field === :lcl_filled
        return settrades_lcl_filled!(tradesdf, ix, value)
    elseif field === :lcl_pavg
        return settrades_lcl_pavg!(tradesdf, ix, value)
    elseif field === :lc_msg
        return settrades_lc_msg!(tradesdf, ix, value)
    elseif field === :lcl_msg
        return settrades_lcl_msg!(tradesdf, ix, value)
    elseif field === :lcsl_id
        return settrades_lcsl_id!(tradesdf, ix, value)
    elseif field === :lcsl_status
        return settrades_lcsl_status!(tradesdf, ix, value)
    elseif field === :lcsl_msg
        return settrades_lcsl_msg!(tradesdf, ix, value)
    elseif field === :lcsl_limit
        return settrades_lcsl_limit!(tradesdf, ix, value)
    elseif field === :so_id
        return settrades_so_id!(tradesdf, ix, value)
    elseif field === :so_status
        return settrades_so_status!(tradesdf, ix, value)
    elseif field === :sol_id
        return settrades_sol_id!(tradesdf, ix, value)
    elseif field === :sol_status
        return settrades_sol_status!(tradesdf, ix, value)
    elseif field === :sol_filled
        return settrades_sol_filled!(tradesdf, ix, value)
    elseif field === :sol_pavg
        return settrades_sol_pavg!(tradesdf, ix, value)
    elseif field === :so_msg
        return settrades_so_msg!(tradesdf, ix, value)
    elseif field === :sol_msg
        return settrades_sol_msg!(tradesdf, ix, value)
    elseif field === :sc_id
        return settrades_sc_id!(tradesdf, ix, value)
    elseif field === :sc_status
        return settrades_sc_status!(tradesdf, ix, value)
    elseif field === :scl_id
        return settrades_scl_id!(tradesdf, ix, value)
    elseif field === :scl_status
        return settrades_scl_status!(tradesdf, ix, value)
    elseif field === :scl_filled
        return settrades_scl_filled!(tradesdf, ix, value)
    elseif field === :scl_pavg
        return settrades_scl_pavg!(tradesdf, ix, value)
    elseif field === :sc_msg
        return settrades_sc_msg!(tradesdf, ix, value)
    elseif field === :scl_msg
        return settrades_scl_msg!(tradesdf, ix, value)
    elseif field === :scsl_id
        return settrades_scsl_id!(tradesdf, ix, value)
    elseif field === :scsl_status
        return settrades_scsl_status!(tradesdf, ix, value)
    elseif field === :scsl_msg
        return settrades_scsl_msg!(tradesdf, ix, value)
    elseif field === :scsl_limit
        return settrades_scsl_limit!(tradesdf, ix, value)
    elseif field === :lp_amount
        return settrades_lp_amount!(tradesdf, ix, value)
    elseif field === :sp_amount
        return settrades_sp_amount!(tradesdf, ix, value)
    elseif field === :close
        return settrades_close!(tradesdf, ix, value)
    elseif field === :high
        return settrades_high!(tradesdf, ix, value)
    elseif field === :low
        return settrades_low!(tradesdf, ix, value)
    elseif field === :equity
        return settrades_equity!(tradesdf, ix, value)
    elseif field === :freemargin
        return settrades_freemargin!(tradesdf, ix, value)
    elseif field === :freequote
        return settrades_freequote!(tradesdf, ix, value)
    elseif field === :label
        return settrades_label!(tradesdf, ix, value)
    elseif field === :score
        return settrades_score!(tradesdf, ix, value)
    elseif field === :lo_limit
        return settrades_lo_limit!(tradesdf, ix, value)
    elseif field === :lc_limit
        return settrades_lc_limit!(tradesdf, ix, value)
    elseif field === :so_limit
        return settrades_so_limit!(tradesdf, ix, value)
    elseif field === :sc_limit
        return settrades_sc_limit!(tradesdf, ix, value)
    elseif field === :lo_amount
        return settrades_lo_amount!(tradesdf, ix, value)
    elseif field === :lc_amount
        return settrades_lc_amount!(tradesdf, ix, value)
    elseif field === :so_amount
        return settrades_so_amount!(tradesdf, ix, value)
    elseif field === :sc_amount
        return settrades_sc_amount!(tradesdf, ix, value)
    elseif field === :config
        return settrades_config!(tradesdf, ix, value)
    elseif field === :tsmstate
        return settrades_tsmstate!(tradesdf, ix, value)
    end
    throw(ArgumentError("unsupported trades field $(field)"))
end

"""Return the concrete vector type of one Trades column."""
function tradescolumntype(field::Symbol)
    field === :opentime && return Vector{DateTime}
    field === :lastopentrade && return Vector{Union{Missing, DateTime}}
    field === :label && return Vector{TradeLabel}
    field in TSM_ID_COLUMNS && return TradesCat32Column
    field in TSM_CATEGORICAL_COLUMNS && return TradesCat8Column
    field in TSM_FLOAT_COLUMNS && return Vector{Float32}
    field in TSM_INT_COLUMNS && return Vector{Int32}
    error("no column type known for trades field $(field)")
end

@eval begin
    """Typed handles to every column of one Trades DataFrame.

    Built once per row loop so per-cell access becomes a direct, statically dispatched
    vector store. `DataFrame` erases its column types and stores no row count, so
    `df[ix, :col]` otherwise costs an index lookup plus a dynamic dispatch on every
    read and write.

    Handles stay valid while rows are appended (`push!` keeps the column object identity)
    but are invalidated by whole-column replacement such as `df[!, :col] = newvector`."""
    struct TradesColumns
        $((:($field::$(tradescolumntype(field))) for field in TSM_TRADES_COLUMNS)...)
    end

    """Resolve typed handles for every Trades column of `tradesdf`."""
    function TradesColumns(tradesdf::DataFrame)
        return TradesColumns($((:(tradesdf[!, $(QuoteNode(field))]::$(tradescolumntype(field))) for field in TSM_TRADES_COLUMNS)...))
    end
end

"""Return the number of Trades rows the handles span."""
@inline tradesrows(cols::TradesColumns)::Int = length(cols.opentime)

"""Set one categorical Trades cell through a column handle, registering an unseen level.

The handle-based counterpart of the `settrades_*!` accessors, for row loops that already
resolved their columns via `TradesColumns`."""
@inline function setcategorical!(col::CategoricalVector, ix::Integer, value)
    sval = String(value)
    _ensurecategoricallevel!(col, sval)
    @inbounds col[ix] = sval
    return col
end

include("TsmGains.jl")

end # module