module TSM

using DataFrames, CategoricalArrays, Dates
using EnvConfig
using Targets

export tradesdf_all_contributors,
    TsmCache,
    TsmCacche,
    xch_tradesdf_contributors,
    tradingstrategy_tradesdf_contributors,
    trade_tradesdf_contributors,
    tsm_tradesdf_contributors,
    ensuretradeschema!,
    xch_tradesdf_opentime,
    xch_tradesdf_lastopentrade,
    xch_tradesdf_pair,
    xch_tradesdf_lo_id,
    xch_tradesdf_lo_status,
    xch_tradesdf_lo_filled,
    xch_tradesdf_lo_pavg,
    xch_tradesdf_lo_msg,
    xch_tradesdf_lc_id,
    xch_tradesdf_lc_status,
    xch_tradesdf_lc_filled,
    xch_tradesdf_lc_pavg,
    xch_tradesdf_lc_msg,
    xch_tradesdf_so_id,
    xch_tradesdf_so_status,
    xch_tradesdf_so_filled,
    xch_tradesdf_so_pavg,
    xch_tradesdf_so_msg,
    xch_tradesdf_sc_id,
    xch_tradesdf_sc_status,
    xch_tradesdf_sc_filled,
    xch_tradesdf_sc_pavg,
    xch_tradesdf_sc_msg,
    xch_tradesdf_lp_amount,
    xch_tradesdf_sp_amount,
    xch_tradesdf_close,
    xch_tradesdf_high,
    xch_tradesdf_low,
    xch_tradesdf_maintmargin,
    xch_tradesdf_equity,
    xch_tradesdf_balance,
    xch_tradesdf_freemargin,
    xch_tradesdf_freequote,
    tradingstrategy_tradesdf_label,
    tradingstrategy_tradesdf_score,
    tradingstrategy_tradesdf_lo_limit,
    tradingstrategy_tradesdf_lc_limit,
    tradingstrategy_tradesdf_so_limit,
    tradingstrategy_tradesdf_sc_limit,
    trade_tradesdf_lo_amount,
    trade_tradesdf_lc_amount,
    trade_tradesdf_so_amount,
    trade_tradesdf_sc_amount,
    tsm_tradesdf_config,
    tsm_tradesdf_tsmstate,
    gettrades_opentime,
    settrades_opentime!,
    gettrades_lastopentrade,
    settrades_lastopentrade!,
    gettrades_pair,
    settrades_pair!,
    gettrades_lo_id,
    settrades_lo_id!,
    gettrades_lo_status,
    settrades_lo_status!,
    gettrades_lo_filled,
    settrades_lo_filled!,
    gettrades_lo_pavg,
    settrades_lo_pavg!,
    gettrades_lo_msg,
    settrades_lo_msg!,
    gettrades_lc_id,
    settrades_lc_id!,
    gettrades_lc_status,
    settrades_lc_status!,
    gettrades_lc_filled,
    settrades_lc_filled!,
    gettrades_lc_pavg,
    settrades_lc_pavg!,
    gettrades_lc_msg,
    settrades_lc_msg!,
    gettrades_so_id,
    settrades_so_id!,
    gettrades_so_status,
    settrades_so_status!,
    gettrades_so_filled,
    settrades_so_filled!,
    gettrades_so_pavg,
    settrades_so_pavg!,
    gettrades_so_msg,
    settrades_so_msg!,
    gettrades_sc_id,
    settrades_sc_id!,
    gettrades_sc_status,
    settrades_sc_status!,
    gettrades_sc_filled,
    settrades_sc_filled!,
    gettrades_sc_pavg,
    settrades_sc_pavg!,
    gettrades_sc_msg,
    settrades_sc_msg!,
    gettrades_lp_amount,
    settrades_lp_amount!,
    gettrades_sp_amount,
    settrades_sp_amount!,
    gettrades_close,
    settrades_close!,
    gettrades_high,
    settrades_high!,
    gettrades_low,
    settrades_low!,
    gettrades_maintmargin,
    settrades_maintmargin!,
    gettrades_equity,
    settrades_equity!,
    gettrades_balance,
    settrades_balance!,
    gettrades_freemargin,
    settrades_freemargin!,
    gettrades_freequote,
    settrades_freequote!,
    gettrades_label,
    settrades_label!,
    gettrades_score,
    settrades_score!,
    gettrades_lo_limit,
    settrades_lo_limit!,
    gettrades_lc_limit,
    settrades_lc_limit!,
    gettrades_so_limit,
    settrades_so_limit!,
    gettrades_sc_limit,
    settrades_sc_limit!,
    gettrades_lo_amount,
    settrades_lo_amount!,
    gettrades_lc_amount,
    settrades_lc_amount!,
    gettrades_so_amount,
    settrades_so_amount!,
    gettrades_sc_amount,
    settrades_sc_amount!,
    gettrades_config,
    settrades_config!,
    gettrades_tsmstate,
    settrades_tsmstate!,
    gettradesfield,
    settradesfield!,
    collecttradesdf,
    savetradesdf,
    readtradesdf,
    compilegainsdf,
    gainsreport

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
const TSM_STATUS_LEVELS = ["none", "submitted", "closed", "canceled", "rejected"]
const TSM_CATEGORICAL_COLUMNS = Set([:pair, :lo_id, :lo_status, :lo_msg, :lc_id, :lc_status, :lc_msg, :so_id, :so_status, :so_msg, :sc_id, :sc_status, :sc_msg, :config, :tsmstate])
const TSM_FLOAT_COLUMNS = Set([:lo_filled, :lo_pavg, :lc_filled, :lc_pavg, :so_filled, :so_pavg, :sc_filled, :sc_pavg, :lp_amount, :sp_amount, :close, :high, :low, :maintmargin, :equity, :balance, :freemargin, :freequote, :score, :lo_limit, :lc_limit, :so_limit, :sc_limit, :lo_amount, :lc_amount, :so_amount, :sc_amount])

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
    _applytradescontributors!(tsm, tdf)
    rowdf = DataFrame(_tradesrowtemplate!(tsm); copycols=true)
    rowdf[1, :opentime] = opentime
    if :pair in propertynames(rowdf)
        rowdf[1, :pair] = uppercase(String(pairkey))
    end
    append!(tdf, rowdf; cols=:subset)
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
        return fill(Targets.ignore, n)
    elseif field === :lo_status || field === :lc_status || field === :so_status || field === :sc_status
        return _compressedcategorical(fill("none", n); levels=TSM_STATUS_LEVELS)
    elseif field === :lo_id || field === :lc_id || field === :so_id || field === :sc_id
        return _compressedcategorical(fill(TSM_NO_ORDER_ID, n); levels=[TSM_NO_ORDER_ID])
    elseif field === :lo_msg || field === :lc_msg || field === :so_msg || field === :sc_msg
        return _compressedcategorical(fill(TSM_NO_ORDER_MSG, n); levels=[TSM_NO_ORDER_MSG])
    elseif field === :pair || field === :config || field === :tsmstate
        default = field === :pair ? "none" : field === :config ? TSM_NO_CONFIG : TSM_NO_STATE
        return _compressedcategorical(fill(default, n); levels=[default])
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

function _datetime_setter!(tradesdf::DataFrame, ix::Integer, field::Symbol, value)
    _assert_row_bounds(tradesdf, ix, field)
    _assert_hasfield(tradesdf, field)
    tradesdf[ix, field] = value
    return tradesdf
end

function _label_setter!(tradesdf::DataFrame, ix::Integer, value)
    _assert_row_bounds(tradesdf, ix, :label)
    _assert_hasfield(tradesdf, :label)
    tradesdf[ix, :label] = value isa Targets.TradeLabel ? value : Targets.tradelabel(String(value))
    return tradesdf
end

for field in (:opentime, :lastopentrade, :pair, :lo_id, :lo_status, :lo_filled, :lo_pavg, :lo_msg, :lc_id, :lc_status, :lc_filled, :lc_pavg, :lc_msg, :so_id, :so_status, :so_filled, :so_pavg, :so_msg, :sc_id, :sc_status, :sc_filled, :sc_pavg, :sc_msg, :lp_amount, :sp_amount, :close, :high, :low, :maintmargin, :equity, :balance, :freemargin, :freequote, :label, :score, :lo_limit, :lc_limit, :so_limit, :sc_limit, :lo_amount, :lc_amount, :so_amount, :sc_amount, :config, :tsmstate)
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
        xch_tradesdf_lo_id,
        xch_tradesdf_lo_status,
        xch_tradesdf_lo_filled,
        xch_tradesdf_lo_pavg,
        xch_tradesdf_lo_msg,
        xch_tradesdf_lc_id,
        xch_tradesdf_lc_status,
        xch_tradesdf_lc_filled,
        xch_tradesdf_lc_pavg,
        xch_tradesdf_lc_msg,
        xch_tradesdf_so_id,
        xch_tradesdf_so_status,
        xch_tradesdf_so_filled,
        xch_tradesdf_so_pavg,
        xch_tradesdf_so_msg,
        xch_tradesdf_sc_id,
        xch_tradesdf_sc_status,
        xch_tradesdf_sc_filled,
        xch_tradesdf_sc_pavg,
        xch_tradesdf_sc_msg,
        xch_tradesdf_lp_amount,
        xch_tradesdf_sp_amount,
        xch_tradesdf_close,
        xch_tradesdf_high,
        xch_tradesdf_low,
        xch_tradesdf_maintmargin,
        xch_tradesdf_equity,
        xch_tradesdf_balance,
        xch_tradesdf_freemargin,
        xch_tradesdf_freequote,
    ]
end

"""Return TradingStrategy-contributed Trades schema initializer functions."""
function tradingstrategy_tradesdf_contributors()::Vector{Function}
    return Function[
        tradingstrategy_tradesdf_label,
        tradingstrategy_tradesdf_score,
        tradingstrategy_tradesdf_lo_limit,
        tradingstrategy_tradesdf_lc_limit,
        tradingstrategy_tradesdf_so_limit,
        tradingstrategy_tradesdf_sc_limit,
    ]
end

"""Return Trade-contributed Trades schema initializer functions."""
function trade_tradesdf_contributors()::Vector{Function}
    return Function[
        trade_tradesdf_lo_amount,
        trade_tradesdf_lc_amount,
        trade_tradesdf_so_amount,
        trade_tradesdf_sc_amount,
    ]
end

"""Return TSM-owned Trades schema initializer functions."""
function tsm_tradesdf_contributors()::Vector{Function}
    return Function[
        tsm_tradesdf_config,
        tsm_tradesdf_tsmstate,
    ]
end

"""Ensure Trades column `opentime` exists. Owner: Xch. Eltype: `DateTime`. Note: Required unique and sorted timestamp derived from sample data."""
function xch_tradesdf_opentime(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :opentime)
end

"""Ensure Trades column `lastopentrade` exists. Owner: Xch. Eltype: `Union{Missing,DateTime}`. Note: Timestamp of the last open-trade event for the pair while `lp_amount > 0f0` or `sp_amount > 0f0`; otherwise `missing`."""
function xch_tradesdf_lastopentrade(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lastopentrade)
end

"""Ensure Trades column `pair` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Required identity/routing column of the trading pair used by Xch."""
function xch_tradesdf_pair(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :pair)
end

"""Ensure Trades column `lo_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_lo_id(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lo_id)
end

"""Ensure Trades column `lo_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_lo_status(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lo_status)
end

"""Ensure Trades column `lo_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_lo_filled(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lo_filled)
end

"""Ensure Trades column `lo_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_lo_pavg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lo_pavg)
end

"""Ensure Trades column `lo_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_lo_msg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lo_msg)
end

"""Ensure Trades column `lc_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_lc_id(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lc_id)
end

"""Ensure Trades column `lc_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_lc_status(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lc_status)
end

"""Ensure Trades column `lc_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_lc_filled(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lc_filled)
end

"""Ensure Trades column `lc_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_lc_pavg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lc_pavg)
end

"""Ensure Trades column `lc_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_lc_msg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lc_msg)
end

"""Ensure Trades column `so_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_so_id(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :so_id)
end

"""Ensure Trades column `so_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_so_status(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :so_status)
end

"""Ensure Trades column `so_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_so_filled(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :so_filled)
end

"""Ensure Trades column `so_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_so_pavg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :so_pavg)
end

"""Ensure Trades column `so_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_so_msg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :so_msg)
end

"""Ensure Trades column `sc_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_sc_id(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :sc_id)
end

"""Ensure Trades column `sc_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_sc_status(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :sc_status)
end

"""Ensure Trades column `sc_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_sc_filled(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :sc_filled)
end

"""Ensure Trades column `sc_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_sc_pavg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :sc_pavg)
end

"""Ensure Trades column `sc_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_sc_msg(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :sc_msg)
end

"""Ensure Trades column `lp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Long position amount snapshot for the trading pair."""
function xch_tradesdf_lp_amount(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :lp_amount)
end

"""Ensure Trades column `sp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Short position amount snapshot for the trading pair."""
function xch_tradesdf_sp_amount(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :sp_amount)
end

"""Ensure Trades column `close` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Close price of OHLCV sample for the trading pair."""
function xch_tradesdf_close(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :close)
end

"""Ensure Trades column `high` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: High price of OHLCV sample for the trading pair."""
function xch_tradesdf_high(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :high)
end

"""Ensure Trades column `low` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Low price of OHLCV sample for the trading pair."""
function xch_tradesdf_low(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :low)
end

"""Ensure Trades column `maintmargin` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Maintenance margin of position."""
function xch_tradesdf_maintmargin(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :maintmargin)
end

"""Ensure Trades column `equity` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Account equity amount of trading pair base."""
function xch_tradesdf_equity(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :equity)
end

"""Ensure Trades column `balance` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Account balance amount of trading pair base."""
function xch_tradesdf_balance(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :balance)
end

"""Ensure Trades column `freemargin` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free margin amount of trading pair base."""
function xch_tradesdf_freemargin(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :freemargin)
end

"""Ensure Trades column `freequote` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free quote amount of trading pair base."""
function xch_tradesdf_freequote(df::DataFrame)::DataFrame
    return _ensurecolumn!(df, :freequote)
end

"""Ensure Trades column `label` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: TradingStrategy writes enum labels; Xch consumes them to map open/close actions."""
function tradingstrategy_tradesdf_label(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :label)
end

"""Ensure Trades column `score` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: Strategy confidence/score of trade label."""
function tradingstrategy_tradesdf_score(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :score)
end

"""Ensure Trades column `lo_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (long-open limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_lo_limit(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :lo_limit)
end

"""Ensure Trades column `lc_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (long-close limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_lc_limit(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :lc_limit)
end

"""Ensure Trades column `so_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (short-open limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_so_limit(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :so_limit)
end

"""Ensure Trades column `sc_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (short-close limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_sc_limit(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :sc_limit)
end

"""Ensure Trades column `lo_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for long-open consumed by Xch order processing."""
function trade_tradesdf_lo_amount(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :lo_amount)
end

"""Ensure Trades column `lc_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for long-close consumed by Xch order processing."""
function trade_tradesdf_lc_amount(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :lc_amount)
end

"""Ensure Trades column `so_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for short-open consumed by Xch order processing."""
function trade_tradesdf_so_amount(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :so_amount)
end

"""Ensure Trades column `sc_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for short-close consumed by Xch order processing."""
function trade_tradesdf_sc_amount(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :sc_amount)
end

"""Ensure Trades column `config` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Note: TSM configuration tag used to track the active state-machine configuration."""
function tsm_tradesdf_config(tradesdf::DataFrame)::DataFrame
    return _ensurecolumn!(tradesdf, :config)
end

"""Ensure Trades column `tsmstate` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Note: TSM state tag used to track the active state-machine state."""
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
    elseif field === :lo_id
        return gettrades_lo_id(tradesdf, ix)
    elseif field === :lo_status
        return gettrades_lo_status(tradesdf, ix)
    elseif field === :lo_filled
        return gettrades_lo_filled(tradesdf, ix)
    elseif field === :lo_pavg
        return gettrades_lo_pavg(tradesdf, ix)
    elseif field === :lo_msg
        return gettrades_lo_msg(tradesdf, ix)
    elseif field === :lc_id
        return gettrades_lc_id(tradesdf, ix)
    elseif field === :lc_status
        return gettrades_lc_status(tradesdf, ix)
    elseif field === :lc_filled
        return gettrades_lc_filled(tradesdf, ix)
    elseif field === :lc_pavg
        return gettrades_lc_pavg(tradesdf, ix)
    elseif field === :lc_msg
        return gettrades_lc_msg(tradesdf, ix)
    elseif field === :so_id
        return gettrades_so_id(tradesdf, ix)
    elseif field === :so_status
        return gettrades_so_status(tradesdf, ix)
    elseif field === :so_filled
        return gettrades_so_filled(tradesdf, ix)
    elseif field === :so_pavg
        return gettrades_so_pavg(tradesdf, ix)
    elseif field === :so_msg
        return gettrades_so_msg(tradesdf, ix)
    elseif field === :sc_id
        return gettrades_sc_id(tradesdf, ix)
    elseif field === :sc_status
        return gettrades_sc_status(tradesdf, ix)
    elseif field === :sc_filled
        return gettrades_sc_filled(tradesdf, ix)
    elseif field === :sc_pavg
        return gettrades_sc_pavg(tradesdf, ix)
    elseif field === :sc_msg
        return gettrades_sc_msg(tradesdf, ix)
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
    elseif field === :maintmargin
        return gettrades_maintmargin(tradesdf, ix)
    elseif field === :equity
        return gettrades_equity(tradesdf, ix)
    elseif field === :balance
        return gettrades_balance(tradesdf, ix)
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
    elseif field === :lo_id
        return settrades_lo_id!(tradesdf, ix, value)
    elseif field === :lo_status
        return settrades_lo_status!(tradesdf, ix, value)
    elseif field === :lo_filled
        return settrades_lo_filled!(tradesdf, ix, value)
    elseif field === :lo_pavg
        return settrades_lo_pavg!(tradesdf, ix, value)
    elseif field === :lo_msg
        return settrades_lo_msg!(tradesdf, ix, value)
    elseif field === :lc_id
        return settrades_lc_id!(tradesdf, ix, value)
    elseif field === :lc_status
        return settrades_lc_status!(tradesdf, ix, value)
    elseif field === :lc_filled
        return settrades_lc_filled!(tradesdf, ix, value)
    elseif field === :lc_pavg
        return settrades_lc_pavg!(tradesdf, ix, value)
    elseif field === :lc_msg
        return settrades_lc_msg!(tradesdf, ix, value)
    elseif field === :so_id
        return settrades_so_id!(tradesdf, ix, value)
    elseif field === :so_status
        return settrades_so_status!(tradesdf, ix, value)
    elseif field === :so_filled
        return settrades_so_filled!(tradesdf, ix, value)
    elseif field === :so_pavg
        return settrades_so_pavg!(tradesdf, ix, value)
    elseif field === :so_msg
        return settrades_so_msg!(tradesdf, ix, value)
    elseif field === :sc_id
        return settrades_sc_id!(tradesdf, ix, value)
    elseif field === :sc_status
        return settrades_sc_status!(tradesdf, ix, value)
    elseif field === :sc_filled
        return settrades_sc_filled!(tradesdf, ix, value)
    elseif field === :sc_pavg
        return settrades_sc_pavg!(tradesdf, ix, value)
    elseif field === :sc_msg
        return settrades_sc_msg!(tradesdf, ix, value)
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
    elseif field === :maintmargin
        return settrades_maintmargin!(tradesdf, ix, value)
    elseif field === :equity
        return settrades_equity!(tradesdf, ix, value)
    elseif field === :balance
        return settrades_balance!(tradesdf, ix, value)
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