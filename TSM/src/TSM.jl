module TSM

using DataFrames, CategoricalArrays, Dates
using EnvConfig
using Targets

# Intentionally no exports: call public API via TSM.<name> to avoid namespace clashes.

"""Pair-state owner for Trades DataFrames and the cached one-row row template."""
mutable struct TsmCache
    pairstates::Dict{String, DataFrame}
    tradesrowtemplate::DataFrame
    schema_contributors::Vector{Function}

    function TsmCache(; schema_contributors::Vector{Function}=Function[])
        return new(Dict{String, DataFrame}(), DataFrame(), Function[schema_contributors...])
    end
end

const TsmCacche = TsmCache

const TSM_NO_ORDER_ID = "none"
const TSM_NO_ORDER_MSG = "none"
const TSM_NO_CONFIG = "none"
const TSM_NO_STATE = "none"
const TSM_NO_SET = "none"
const TSM_STATUS_LEVELS = ["none", "submitted", "closed", "cancelled", "rejected"]
const TSM_CATEGORICAL_COLUMNS = Set([:pair, :set, :lo_id, :lo_status, :lo_msg, :lol_id, :lol_status, :lol_msg, :lc_id, :lc_status, :lc_msg, :lcl_id, :lcl_status, :lcl_msg, :so_id, :so_status, :so_msg, :sol_id, :sol_status, :sol_msg, :sc_id, :sc_status, :sc_msg, :scl_id, :scl_status, :scl_msg, :config, :tsmstate])
const TSM_FLOAT_COLUMNS = Set([:lol_filled, :lol_pavg, :lcl_filled, :lcl_pavg, :sol_filled, :sol_pavg, :scl_filled, :scl_pavg, :lp_amount, :sp_amount, :close, :high, :low, :equity, :freemargin, :freequote, :score, :lo_limit, :lc_limit, :so_limit, :sc_limit, :lo_amount, :lc_amount, :so_amount, :sc_amount])
const TSM_INT_COLUMNS = Set([:rangeid])
const TSM_TRADE_LANES = Set([:lo, :lc, :so, :sc])

"""Map one trade label (or lane symbol) to its canonical lane symbol (`:lo`, `:lc`, `:so`, `:sc`)."""
function tradelane(label)::Symbol
    if label isa Symbol
        lane = Symbol(lowercase(String(label)))
        @assert lane in TSM_TRADE_LANES "unsupported trade lane=$(lane); supported lanes are $(collect(TSM_TRADE_LANES))"
        return lane
    end

    tl = label isa TradeLabel ? label : tradelabel(String(label))
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

_nrows(df::AbstractDataFrame) = nrow(df)

function _assert_row_bounds(tradesdf::AbstractDataFrame, ix::Integer, field::Symbol)
    @assert 1 <= ix <= nrow(tradesdf) "$(field): ix=$(ix) is out of bounds for trades rows=$(nrow(tradesdf))"
    return nothing
end

function _assert_hasfield(tradesdf::AbstractDataFrame, field::Symbol)
    @assert field in propertynames(tradesdf) "tradesdf must contain $(field); names=$(names(tradesdf))"
    return nothing
end

function _compressedcategorical(values; levels=nothing)
    if isnothing(levels)
        return categorical(values; compress=true)
    end
    return categorical(values; levels=levels, compress=true)
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

function _appendtradesrow!(tsm::TsmCache, tdf::DataFrame, pairkey::AbstractString, opentime::DateTime)::Int
    rowdf = DataFrame(_tradesrowtemplate!(tsm); copycols=true)
    rowdf[1, :opentime] = opentime
    if :pair in propertynames(rowdf)
        rowdf[1, :pair] = uppercase(String(pairkey))
    end
    push!(tdf, rowdf[1, :]; cols=:subset)
    return nrow(tdf)
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

    if :pair ∉ propertynames(df)
        df[!, :pair] = fill(pkey, nrow(df))
    else
        df[!, :pair] = [
            (ismissing(v) || isempty(strip(String(v))) || (uppercase(strip(String(v))) == "NONE")) ? pkey : String(v)
            for v in df[!, :pair]
        ]
    end

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

"""Store one Trades dataframe for a pair and return the cache."""
function settrades!(tsm::TsmCache, pair::AbstractString, df::AbstractDataFrame)
    normalized = DataFrame(df; copycols=false)
    _applytradescontributors!(tsm, normalized)
    pairkey = uppercase(String(pair))
    _ensuretradesidentity!(normalized, pairkey)
    tsm.pairstates[pairkey] = normalized
    return tsm
end

"""Store one Trades dataframe for a pair and return the cache."""
function settrades!(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString, df::AbstractDataFrame)
    pairkey = tradingpairkey(base, quotecoin)
    normalized = DataFrame(df; copycols=false)
    _applytradescontributors!(tsm, normalized)
    _ensuretradesidentity!(normalized, pairkey)
    tsm.pairstates[pairkey] = normalized
    return tsm
end

"""Return the stored pair keys in deterministic order."""
function tradingpairs(tsm::TsmCache)::Vector{String}
    return sort!(collect(keys(tsm.pairstates)))
end

"""Return true when the TSM cache already tracks one pair state entry."""
function haspairstate(tsm::TsmCache, pair::AbstractString)::Bool
    return haskey(tsm.pairstates, uppercase(String(pair)))
end

"""Drop one pair state entry from the TSM cache."""
function droppair!(tsm::TsmCache, pair::AbstractString)::Nothing
    delete!(tsm.pairstates, uppercase(String(pair)))
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

"""Return the writable pair row for one sample timestamp, creating a row when needed."""
function ensuretradesrow!(tsm::TsmCache, base::AbstractString, quotecoin::AbstractString, opentime::DateTime)
    basekey = uppercase(String(base))
    pairkey = tradingpairkey(basekey, quotecoin)
    tdf = trades(tsm, pairkey)

    rowix = nothing
    n = nrow(tdf)
    if n > 0
        last_open = tdf[n, :opentime]
        if last_open == opentime
            rowix = n
        elseif last_open < opentime
            rowix = _appendtradesrow!(tsm, tdf, pairkey, opentime)
        end
    end

    if isnothing(rowix)
        rowix = findlast(==(opentime), tdf[!, :opentime])
    end
    if isnothing(rowix)
        rowix = _appendtradesrow!(tsm, tdf, pairkey, opentime)
    end

    tdf[rowix, :opentime] = opentime
    tdf[rowix, :pair] = pairkey
    return (tradesdf=tdf, rowix=Int(rowix))
end

function _defaultcolumn(field::Symbol, n::Integer)
    if field === :opentime
        return DateTime[]
    elseif field === :lastopentrade
        return Vector{Union{Missing, DateTime}}(missing, n)
    elseif field === :label
        return fill(ignore, n)
    elseif field === :lo_status || field === :lol_status || field === :lc_status || field === :lcl_status || field === :so_status || field === :sol_status || field === :sc_status || field === :scl_status
        return _compressedcategorical(fill("none", n); levels=TSM_STATUS_LEVELS)
    elseif field === :lo_id || field === :lol_id || field === :lc_id || field === :lcl_id || field === :so_id || field === :sol_id || field === :sc_id || field === :scl_id
        return _compressedcategorical(fill(TSM_NO_ORDER_ID, n); levels=[TSM_NO_ORDER_ID])
    elseif field === :lo_msg || field === :lol_msg || field === :lc_msg || field === :lcl_msg || field === :so_msg || field === :sol_msg || field === :sc_msg || field === :scl_msg
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

function _ensurecolumn!(tradesdf::DataFrame, field::Symbol)
    if field ∉ propertynames(tradesdf)
        tradesdf[!, field] = _defaultcolumn(field, nrow(tradesdf))
    end
    return tradesdf
end

function _categorical_setter!(tradesdf::DataFrame, field::Symbol, ix::Integer, value)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    return _setcategoricalcell!(tradesdf, field, ix, value)
end

function _float_getter(tradesdf::AbstractDataFrame, ix::Integer, field::Symbol)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    return tradesdf[ix, field]
end

function _float_setter!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    tradesdf[ix, field] = value
    return tradesdf
end

function _int_getter(tradesdf::AbstractDataFrame, ix::Integer, field::Symbol)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    return tradesdf[ix, field]
end

function _int_setter!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    tradesdf[ix, field] = value
    return tradesdf
end

function _datetime_setter!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    tradesdf[ix, field] = value
    return tradesdf
end

function _label_setter!(tradesdf::DataFrame, ix::Integer, value)
    _assert_row_bounds(tradesdf, ix, :label)
    _assert_hasfield(tradesdf, :label)
    tradesdf[ix, :label] = value isa TradeLabel ? value : tradelabel(String(value))
    return tradesdf
end

"""Get one lane-scoped trades field cell using a trade label and suffix (for example `:limit`, `:amount`, `:id`)."""
function gettrades_lanefield(tradesdf::AbstractDataFrame, ix::Integer, label, suffix::Symbol)
    field = Symbol(tradelane(label), "_", suffix)
    return gettradesfield(tradesdf, ix, field)
end

"""Set one lane-scoped trades field cell using a trade label and suffix (for example `:limit`, `:amount`, `:id`)."""
function settrades_lanefield!(tradesdf::DataFrame, ix::Integer, label, suffix::Symbol, value)
    field = Symbol(tradelane(label), "_", suffix)
    return settradesfield!(tradesdf, ix, field, value)
end

"""Get one last-lane trades field cell using a trade label and suffix (for example `:id`, `:status`, `:msg`)."""
function gettrades_lastlanefield(tradesdf::AbstractDataFrame, ix::Integer, label, suffix::Symbol)
    field = Symbol(tradelane(label), "l_", suffix)
    return gettradesfield(tradesdf, ix, field)
end

"""Set one last-lane trades field cell using a trade label and suffix (for example `:id`, `:status`, `:msg`)."""
function settrades_lastlanefield!(tradesdf::DataFrame, ix::Integer, label, suffix::Symbol, value)
    field = Symbol(tradelane(label), "l_", suffix)
    return settradesfield!(tradesdf, ix, field, value)
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

for field in (:opentime, :lastopentrade, :pair, :set, :rangeid, :lo_id, :lo_status, :lol_id, :lol_status, :lol_filled, :lol_pavg, :lo_msg, :lol_msg, :lc_id, :lc_status, :lcl_id, :lcl_status, :lcl_filled, :lcl_pavg, :lc_msg, :lcl_msg, :so_id, :so_status, :sol_id, :sol_status, :sol_filled, :sol_pavg, :so_msg, :sol_msg, :sc_id, :sc_status, :scl_id, :scl_status, :scl_filled, :scl_pavg, :sc_msg, :scl_msg, :lp_amount, :sp_amount, :close, :high, :low, :equity, :freemargin, :freequote, :label, :score, :lo_limit, :lc_limit, :so_limit, :sc_limit, :lo_amount, :lc_amount, :so_amount, :sc_amount, :config, :tsmstate)
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

"""Ensure Trades lane column `<lane>_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: if order amount > 0 then order shall be placed, otherwise not."""
function trade_tradesdf_amount(tradesdf::DataFrame, label)::DataFrame
    return _ensurecolumn!(tradesdf, Symbol(tradelane(label), "_amount"))
end

"""Ensure Trades column `config` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Identifies the Trade configuration id. Any change in config, e.g. different openthresholds, shall result in a different config marker"""
function tsm_tradesdf_config(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :config)
end

"""Ensure Trades column `tsmstate` exists. Owner: TSM. Eltype: `CategoricalVector{String}`.  
- *sync*: execution and price changes of the most recent minute are updated in the current row fields; next is *request*
- *request*: based on data of the previous minute order requests are defined; next is *xch*
- *xch*: order requests are submitted to the exchange; next is *sync*
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

include("TsmGains.jl")

end # module