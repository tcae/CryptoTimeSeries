"Build a one-row Trades template initialized via registered schema contributors."
function _buildtradesrowtemplate(xc::XchCache)::DataFrame
    template = DataFrame(opentime=[DateTime(1970, 1, 1)])
    _applytradescontributors!(xc, template)
    return template
end

"Refresh and return the cached one-row Trades template for default row appends."
function _tradesrowtemplate!(xc::XchCache)::DataFrame
    if nrow(xc.tradesrowtemplate) != 1
        xc.tradesrowtemplate = _buildtradesrowtemplate(xc)
    end
    return xc.tradesrowtemplate
end

"Append one default-initialized Trades row and return its index."
function _appendtradesrow!(xc::XchCache, tdf::DataFrame, pairkey::AbstractString, opentime::DateTime)::Int
    _applytradescontributors!(xc, tdf)
    rowdf = DataFrame(_tradesrowtemplate!(xc); copycols=true)
    rowdf[1, :opentime] = opentime
    if :pair in propertynames(rowdf)
        rowdf[1, :pair] = uppercase(String(pairkey))
    end
    append!(tdf, rowdf; cols=:subset)
    return nrow(tdf)
end

"""
    ensuretradesschema(xc, contributors)

Register contributor column-initializer functions used to materialize Trades
columns at dataframe creation/ingestion time.
"""
function ensuretradesschema(xc::XchCache, contributors)::XchCache
    xc.mc[:trades_schema_contributors] = Function[contributors...]
    for pair in keys(xc.pairstates)
        _applytradescontributors!(xc, xc.pairstates[pair])
    end
    xc.tradesrowtemplate = _buildtradesrowtemplate(xc)
    return xc
end

"""
    _emptytradesv1df()

Return an empty Trades DataFrame initialized with Xch-owned columns.
Contributor columns from higher-level modules are materialized via
`ensuretradesschema` when attached to an `XchCache`.
"""
function _emptytradesv1df()::DataFrame
    df = DataFrame(opentime=DateTime[])
    for contributor in xch_tradesdf_contributors()
        contributor(df)
    end
    return df
end

function _applytradescontributors!(xc::XchCache, df::DataFrame=DataFrame(), contributors::Vector{Function}=tradesdf_all_contributors())::DataFrame
    for contributor in contributors
        contributor(df)
    end
    return df
end

"""Return all Trades schema contributor functions across Xch, TradingStrategy, and Trade."""
function tradesdf_all_contributors()::Vector{Function}
    return vcat(xch_tradesdf_contributors(), tradingstrategy_tradesdf_contributors(), trade_tradesdf_contributors())
end



#region Xch ownership

"""Ensure Trades column `opentime` exists. Owner: Xch. Eltype: `DateTime`. Note: Required unique and sorted timestamp derived from sample data."""
function xch_tradesdf_opentime(df::DataFrame)::DataFrame
    if :opentime ∉ propertynames(df)
        df[!, :opentime] = DateTime[]
    end
    return df
end

"""Ensure Trades column `lastopentrade` exists. Owner: Xch. Eltype: `Union{Missing,DateTime}`. Note: Timestamp of the last open-trade event for the pair while `lp_amount > 0f0` or `sp_amount > 0f0`; otherwise `missing`."""
function xch_tradesdf_lastopentrade(df::DataFrame)::DataFrame
    if :lastopentrade ∉ propertynames(df)
        df[!, :lastopentrade] = Vector{Union{Missing, DateTime}}(missing, nrow(df))
    end
    return df
end

"""Ensure Trades column `pair` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Required identity/routing column of the trading pair used by Xch."""
function xch_tradesdf_pair(df::DataFrame)::DataFrame
    if :pair ∉ propertynames(df)
        df[!, :pair] = CategoricalVector(fill("none", nrow(df)))
    end
    return df
end

# Long-Open order lane (lo_)
"""Ensure Trades column `lo_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_lo_id(df::DataFrame)::DataFrame
    if :lo_id ∉ propertynames(df)
        df[!, :lo_id] = CategoricalVector(fill(NO_ORDER_ID, nrow(df)))
    end
    return df
end

"""Ensure Trades column `lo_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_lo_status(df::DataFrame)::DataFrame
    status_levels = ["none", "submitted", "closed", "canceled", "rejected"]
    if :lo_status ∉ propertynames(df)
        df[!, :lo_status] = CategoricalVector(fill("none", nrow(df)); levels=status_levels)
    end
    return df
end

"""Ensure Trades column `lo_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_lo_filled(df::DataFrame)::DataFrame
    if :lo_filled ∉ propertynames(df)
        df[!, :lo_filled] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `lo_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_lo_pavg(df::DataFrame)::DataFrame
    if :lo_pavg ∉ propertynames(df)
        df[!, :lo_pavg] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `lo_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_lo_msg(df::DataFrame)::DataFrame
    if :lo_msg ∉ propertynames(df)
        df[!, :lo_msg] = CategoricalVector(fill(NO_ORDER_MSG, nrow(df)))
    end
    return df
end

# Long-Close order lane (lc_)
"""Ensure Trades column `lc_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_lc_id(df::DataFrame)::DataFrame
    if :lc_id ∉ propertynames(df)
        df[!, :lc_id] = CategoricalVector(fill(NO_ORDER_ID, nrow(df)))
    end
    return df
end

"""Ensure Trades column `lc_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_lc_status(df::DataFrame)::DataFrame
    status_levels = ["none", "submitted", "closed", "canceled", "rejected"]
    if :lc_status ∉ propertynames(df)
        df[!, :lc_status] = CategoricalVector(fill("none", nrow(df)); levels=status_levels)
    end
    return df
end

"""Ensure Trades column `lc_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_lc_filled(df::DataFrame)::DataFrame
    if :lc_filled ∉ propertynames(df)
        df[!, :lc_filled] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `lc_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_lc_pavg(df::DataFrame)::DataFrame
    if :lc_pavg ∉ propertynames(df)
        df[!, :lc_pavg] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `lc_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_lc_msg(df::DataFrame)::DataFrame
    if :lc_msg ∉ propertynames(df)
        df[!, :lc_msg] = CategoricalVector(fill(NO_ORDER_MSG, nrow(df)))
    end
    return df
end

# Short-Open order lane (so_)
"""Ensure Trades column `so_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_so_id(df::DataFrame)::DataFrame
    if :so_id ∉ propertynames(df)
        df[!, :so_id] = CategoricalVector(fill(NO_ORDER_ID, nrow(df)))
    end
    return df
end

"""Ensure Trades column `so_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_so_status(df::DataFrame)::DataFrame
    status_levels = ["none", "submitted", "closed", "canceled", "rejected"]
    if :so_status ∉ propertynames(df)
        df[!, :so_status] = CategoricalVector(fill("none", nrow(df)); levels=status_levels)
    end
    return df
end

"""Ensure Trades column `so_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_so_filled(df::DataFrame)::DataFrame
    if :so_filled ∉ propertynames(df)
        df[!, :so_filled] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `so_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_so_pavg(df::DataFrame)::DataFrame
    if :so_pavg ∉ propertynames(df)
        df[!, :so_pavg] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `so_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_so_msg(df::DataFrame)::DataFrame
    if :so_msg ∉ propertynames(df)
        df[!, :so_msg] = CategoricalVector(fill(NO_ORDER_MSG, nrow(df)))
    end
    return df
end

# Short-Close order lane (sc_)
"""Ensure Trades column `sc_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request."""
function xch_tradesdf_sc_id(df::DataFrame)::DataFrame
    if :sc_id ∉ propertynames(df)
        df[!, :sc_id] = CategoricalVector(fill(NO_ORDER_ID, nrow(df)))
    end
    return df
end

"""Ensure Trades column `sc_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected."""
function xch_tradesdf_sc_status(df::DataFrame)::DataFrame
    status_levels = ["none", "submitted", "closed", "canceled", "rejected"]
    if :sc_status ∉ propertynames(df)
        df[!, :sc_status] = CategoricalVector(fill("none", nrow(df)); levels=status_levels)
    end
    return df
end

"""Ensure Trades column `sc_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Filled/executed base quantity from order status reconciliation."""
function xch_tradesdf_sc_filled(df::DataFrame)::DataFrame
    if :sc_filled ∉ propertynames(df)
        df[!, :sc_filled] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `sc_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis."""
function xch_tradesdf_sc_pavg(df::DataFrame)::DataFrame
    if :sc_pavg ∉ propertynames(df)
        df[!, :sc_pavg] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `sc_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical)."""
function xch_tradesdf_sc_msg(df::DataFrame)::DataFrame
    if :sc_msg ∉ propertynames(df)
        df[!, :sc_msg] = CategoricalVector(fill(NO_ORDER_MSG, nrow(df)))
    end
    return df
end

"""Ensure Trades column `lp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Long position amount snapshot for the trading pair."""
function xch_tradesdf_lp_amount(df::DataFrame)::DataFrame
    if :lp_amount ∉ propertynames(df)
        df[!, :lp_amount] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `sp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Short position amount snapshot for the trading pair."""
function xch_tradesdf_sp_amount(df::DataFrame)::DataFrame
    if :sp_amount ∉ propertynames(df)
        df[!, :sp_amount] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `close` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Close price of OHLCV sample for the trading pair."""
function xch_tradesdf_close(df::DataFrame)::DataFrame
    if :close ∉ propertynames(df)
        df[!, :close] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `high` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: High price of OHLCV sample for the trading pair."""
function xch_tradesdf_high(df::DataFrame)::DataFrame
    if :high ∉ propertynames(df)
        df[!, :high] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `low` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Low price of OHLCV sample for the trading pair."""
function xch_tradesdf_low(df::DataFrame)::DataFrame
    if :low ∉ propertynames(df)
        df[!, :low] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `maintmargin` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Maintenance margin of position."""
function xch_tradesdf_maintmargin(df::DataFrame)::DataFrame
    if :maintmargin ∉ propertynames(df)
        df[!, :maintmargin] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `equity` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Account equity amount of trading pair base."""
function xch_tradesdf_equity(df::DataFrame)::DataFrame
    if :equity ∉ propertynames(df)
        df[!, :equity] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `balance` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Account balance amount of trading pair base."""
function xch_tradesdf_balance(df::DataFrame)::DataFrame
    if :balance ∉ propertynames(df)
        df[!, :balance] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `freemargin` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free margin amount of trading pair base."""
function xch_tradesdf_freemargin(df::DataFrame)::DataFrame
    if :freemargin ∉ propertynames(df)
        df[!, :freemargin] = fill(0f0, nrow(df))
    end
    return df
end

"""Ensure Trades column `freequote` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free quote amount of trading pair base."""
function xch_tradesdf_freequote(df::DataFrame)::DataFrame
    if :freequote ∉ propertynames(df)
        df[!, :freequote] = fill(0f0, nrow(df))
    end
    return df
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

#endregion Xch ownership

#region TradingStrategy ownership

"""Ensure Trades column `label` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: TradingStrategy writes enum labels; Xch consumes them to map open/close actions."""
function tradingstrategy_tradesdf_label(tradesdf::DataFrame)::DataFrame
    if :label ∉ propertynames(tradesdf)
        tradesdf[!, :label] = fill(Targets.ignore, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `score` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: Strategy confidence/score of trade label."""
function tradingstrategy_tradesdf_score(tradesdf::DataFrame)::DataFrame
    if :score ∉ propertynames(tradesdf)
        tradesdf[!, :score] = zeros(Float32, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `lo_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (long-open limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_lo_limit(tradesdf::DataFrame)::DataFrame
    if :lo_limit ∉ propertynames(tradesdf)
        tradesdf[!, :lo_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `lc_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (long-close limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_lc_limit(tradesdf::DataFrame)::DataFrame
    if :lc_limit ∉ propertynames(tradesdf)
        tradesdf[!, :lc_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `so_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (short-open limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_so_limit(tradesdf::DataFrame)::DataFrame
    if :so_limit ∉ propertynames(tradesdf)
        tradesdf[!, :so_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `sc_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance (short-close limit) consumed by Xch as requested limit per action."""
function tradingstrategy_tradesdf_sc_limit(tradesdf::DataFrame)::DataFrame
    if :sc_limit ∉ propertynames(tradesdf)
        tradesdf[!, :sc_limit] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
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

#endregion TradingStrategy ownership

#region Trade ownership

"""Ensure Trades column `lo_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for long-open consumed by Xch order processing."""
function trade_tradesdf_lo_amount(tradesdf::DataFrame)::DataFrame
    if :lo_amount ∉ propertynames(tradesdf)
        tradesdf[!, :lo_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `lc_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for long-close consumed by Xch order processing."""
function trade_tradesdf_lc_amount(tradesdf::DataFrame)::DataFrame
    if :lc_amount ∉ propertynames(tradesdf)
        tradesdf[!, :lc_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `so_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for short-open consumed by Xch order processing."""
function trade_tradesdf_so_amount(tradesdf::DataFrame)::DataFrame
    if :so_amount ∉ propertynames(tradesdf)
        tradesdf[!, :so_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `sc_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for short-close consumed by Xch order processing."""
function trade_tradesdf_sc_amount(tradesdf::DataFrame)::DataFrame
    if :sc_amount ∉ propertynames(tradesdf)
        tradesdf[!, :sc_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
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

#endregion Trade ownership
