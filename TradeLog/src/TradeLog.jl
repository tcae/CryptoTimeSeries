"""
TradeLog defines append-only trading logs for orders, executions, and portfolio
snapshots under `\$HOME/crypto/tradelog`.

The emphasis is practical private-trader diagnostics and replay context.
Tamper-evidence/hash-chain features are intentionally relaxed.
"""
module TradeLog

using Arrow, Dates, EnvConfig, JSON3, UUIDs

export AuditEventType, AuditAssetClass, AuditInstrumentType, AuditMarketType, AuditRoutingRole
export AuditEventRow, auditroot, auditfolder, auditfile, eventpayload, writeevent, auditenabled
export computeevenhash, manifestfile, readmanifest, validatehashchain, writeeventwithhash
export readjsonlauditevents, writearrowauditexport, arrowexportfile

"Canonical audit event types."
@enum AuditEventType begin
    ORDER_SUBMITTED
    ORDER_ACK
    ORDER_PARTIAL_FILL
    ORDER_FILLED
    ORDER_CANCELED
    ORDER_REJECTED
    POSITION_SNAPSHOT
    PORTFOLIO_SNAPSHOT
end

"Canonical asset-class categories for audit rows."
@enum AuditAssetClass begin
    asset_unknown
    crypto
    equity
end

"Canonical instrument-type categories for audit rows."
@enum AuditInstrumentType begin
    instrument_unknown
    spot_pair
    perpetual_future
    share_fiat
end

"Canonical market-type categories for audit rows."
@enum AuditMarketType begin
    market_unknown
    market_spot
    market_futures
    market_equity
end

"Canonical routing-role categories for audit rows."
@enum AuditRoutingRole begin
    routing_unknown
    routing_data_exchange
    routing_trade_exchange_spot
    routing_trade_exchange_futures
end

"""
Flat persisted audit event row used as the canonical schema for replay, export, and
dashboard queries.
"""
Base.@kwdef struct AuditEventRow
    event_id::String = string(uuid4())
    event_type::AuditEventType = ORDER_SUBMITTED
    event_time_utc::DateTime = Dates.now(Dates.UTC)
    created_at_utc::DateTime = Dates.now(Dates.UTC)
    source_module::String = ""
    environment::String = ""
    run_mode::String = "unknown"
    run_id::Union{Missing, String} = missing
    loop_id::Union{Missing, String} = missing
    correlation_id::Union{Missing, String} = missing
    parent_event_id::Union{Missing, String} = missing
    exchange::String = ""
    account_alias::String = ""
    routing_role::AuditRoutingRole = routing_unknown
    market_type::AuditMarketType = market_unknown
    asset_class::AuditAssetClass = asset_unknown
    instrument_type::AuditInstrumentType = instrument_unknown
    venue_instrument_type::Union{Missing, String} = missing
    symbol::String = ""
    baseasset::Union{Missing, String} = missing
    quoteasset::Union{Missing, String} = missing
    underlying::Union{Missing, String} = missing
    settlement_asset::Union{Missing, String} = missing
    contract_class::Union{Missing, String} = missing
    client_order_id::Union{Missing, String} = missing
    exchange_order_id::Union{Missing, String} = missing
    exchange_trade_id::Union{Missing, String} = missing
    side::Union{Missing, String} = missing
    order_type::Union{Missing, String} = missing
    time_in_force::Union{Missing, String} = missing
    status::Union{Missing, String} = missing
    status_reason::Union{Missing, String} = missing
    requested_base_qty::Union{Missing, Float64} = missing
    requested_quote_qty::Union{Missing, Float64} = missing
    requested_limit_price::Union{Missing, Float64} = missing
    requested_stop_price::Union{Missing, Float64} = missing
    requested_notional::Union{Missing, Float64} = missing
    leverage::Union{Missing, Float64} = missing
    fill_base_qty::Union{Missing, Float64} = missing
    fill_quote_qty::Union{Missing, Float64} = missing
    fill_price::Union{Missing, Float64} = missing
    fill_notional::Union{Missing, Float64} = missing
    fee_amount::Union{Missing, Float64} = missing
    fee_currency::Union{Missing, String} = missing
    slippage_bps::Union{Missing, Float64} = missing
    position_qty_before::Union{Missing, Float64} = missing
    position_qty_after::Union{Missing, Float64} = missing
    cash_before::Union{Missing, Float64} = missing
    cash_after::Union{Missing, Float64} = missing
    portfolio_value_before::Union{Missing, Float64} = missing
    portfolio_value_after::Union{Missing, Float64} = missing
    strategy_engine::Union{Missing, String} = missing
    strategy_config_ref::Union{Missing, String} = missing
    signal_label::Union{Missing, String} = missing
    signal_score::Union{Missing, Float64} = missing
    algorithm_version::Union{Missing, String} = missing
    notes::Union{Missing, String} = missing
end

"Return the canonical tradelog root folder under the shared crypto root."
function auditroot(root::Union{Nothing, AbstractString}=nothing)::String
    basepath = if isnothing(root)
        normpath(get(ENV, "CTS_TRADELOG_ROOT", get(ENV, "CTS_AUDIT_ROOT", joinpath(EnvConfig.cryptopath, "tradelog"))))
    else
        normpath(String(root))
    end
    isdir(basepath) || mkpath(basepath)
    return basepath
end

"Return the partition folder for one audit event row, creating it on first use."
function auditfolder(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String
    folder = joinpath(
        auditroot(root),
        "environment=$( _pathvalue(event.environment, "unknown") )",
        "run_mode=$( _pathvalue(event.run_mode, "unknown") )",
        "exchange=$( _pathvalue(event.exchange, "unknown") )",
        "account=$( _pathvalue(event.account_alias, "unknown") )",
        "asset_class=$( _pathvalue(event.asset_class) )",
        "instrument_type=$( _pathvalue(event.instrument_type) )",
        "date=$(Dates.format(Date(event.event_time_utc), dateformat"yyyy-mm-dd"))",
    )
    isdir(folder) || mkpath(folder)
    return folder
end

"Return the append-only JSONL file path for one audit event row."
function auditfile(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String
    return normpath(joinpath(auditfolder(event; root=root), "events.jsonl"))
end

"Return a JSON-compatible flat payload dictionary for one audit event row."
function eventpayload(event::AuditEventRow)::Dict{String, Any}
    payload = Dict{String, Any}()
    for field in fieldnames(AuditEventRow)
        payload[String(field)] = _jsonvalue(getfield(event, field))
    end
    return payload
end

"Return whether audit persistence is enabled for the given event."
function auditenabled(event::AuditEventRow)::Bool
    _envbool("CTS_TRADELOG_ENABLED", _envbool("CTS_AUDIT_ENABLED", true)) || return false
    if lowercase(event.run_mode) == "simulation"
        _envbool("CTS_TRADELOG_SIMULATION_ENABLED", _envbool("CTS_AUDIT_SIMULATION_ENABLED", true)) || return false
    end
    return true
end

"Append one audit event row to the canonical JSONL file and return the written path."
function writeevent(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String
    auditenabled(event) || return ""
    path = auditfile(event; root=root)
    open(path, "a") do io
        write(io, JSON3.write(eventpayload(event)))
        write(io, "\n")
    end
    return path
end

"Return a safe path token for strings or enum values used in partition folders."
function _pathvalue(value, fallback::AbstractString="unknown")::String
    if ismissing(value)
        return String(fallback)
    elseif value isa Enum
        return replace(lowercase(String(Symbol(value))), r"[^a-z0-9_-]+" => "_")
    end
    text = strip(String(value))
    isempty(text) && return String(fallback)
    return replace(text, r"[^A-Za-z0-9._-]+" => "_")
end

"Convert one field value into a JSON-compatible scalar."
function _jsonvalue(value)
    if ismissing(value)
        return missing
    elseif value isa DateTime
        return _timestamp(value)
    elseif value isa Date
        return Dates.format(value, dateformat"yyyy-mm-dd")
    elseif value isa Enum
        return String(Symbol(value))
    end
    return value
end

"Render one UTC timestamp in ISO-8601 format with a trailing `Z`."
function _timestamp(value::DateTime)::String
    return Dates.format(value, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"
end

"Read boolean options from ENV using true/false, 1/0, yes/no, on/off values."
function _envbool(name::AbstractString, default::Bool)::Bool
    raw = get(ENV, String(name), nothing)
    isnothing(raw) && return default
    value = lowercase(strip(String(raw)))
    if value in ["1", "true", "yes", "on"]
        return true
    elseif value in ["0", "false", "no", "off"]
        return false
    end
    return default
end

"""
    computeevenhash(event::AuditEventRow)::String

Compatibility helper retained for callers expecting this API.
TradeLog does not provide tamper-evidence guarantees.
"""
function computeevenhash(event::AuditEventRow)::String
    return string(hash(JSON3.write(eventpayload(event))))
end

"""
    manifestfile(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String

Return the manifest file path for hash-chain integrity tracking for a given event's date and folder.
"""
function manifestfile(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String
    folder = auditfolder(event; root=root)
    return normpath(joinpath(folder, "manifest.json"))
end

"""
    readmanifest(manifestpath::String)::Dict{String, Any}

Read manifest file tracking event hashes for a given audit folder.
Returns empty dict if manifest does not exist or is corrupted.
"""
function readmanifest(manifestpath::String)::Dict{String, Any}
    if !isfile(manifestpath)
        return Dict(
            "date" => Dates.format(Dates.today(), dateformat"yyyy-mm-dd"),
            "event_hashes" => [],
        )
    end
    try
        obj = JSON3.read(read(manifestpath, String))
        # Convert JSON3 object to Dict with String keys
        result = Dict{String, Any}()
        for (k, v) in pairs(obj)
            key = String(k)
            if v isa JSON3.Object
                # Convert nested JSON3 objects to Dict with String keys
                nested_dict = Dict{String, Any}()
                for (nk, nv) in pairs(v)
                    nested_dict[String(nk)] = nv
                end
                result[key] = nested_dict
            elseif v isa JSON3.Array
                # Convert array of JSON3 objects to array of Dicts with String keys
                converted_array = []
                for item in v
                    if item isa JSON3.Object
                        item_dict = Dict{String, Any}()
                        for (ik, iv) in pairs(item)
                            item_dict[String(ik)] = iv
                        end
                        push!(converted_array, item_dict)
                    else
                        push!(converted_array, item)
                    end
                end
                result[key] = converted_array
            else
                result[key] = v
            end
        end
        return result
    catch
        # Manifest is corrupted or empty; return fresh structure and let the caller overwrite it
        return Dict(
            "date" => Dates.format(Dates.today(), dateformat"yyyy-mm-dd"),
            "event_hashes" => [],
        )
    end
end

"""
    writemanifest(manifestpath::String, manifest::Dict{String, Any})

Write manifest file with event hash chain.
"""
function writemanifest(manifestpath::String, manifest::Dict{String, Any})
    open(manifestpath, "w") do io
        write(io, JSON3.write(manifest))
    end
end

"""
    priorevenhash(manifestpath::String)::Union{String, Nothing}

Return the hash of the most recent event in the manifest, or nothing if empty.
"""
function priorevenhash(manifestpath::String)::Union{String, Nothing}
    manifest = readmanifest(manifestpath)
    hashes = get(manifest, "event_hashes", [])
    isempty(hashes) && return nothing
    last_entry = last(hashes)
    if last_entry isa Dict
        return get(last_entry, "hash", nothing)
    else
        return nothing
    end
end

"""
    writeeventwithhash(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String

Compatibility alias for prior callers.
In TradeLog this behaves exactly like `writeevent` (no hash-chain updates).
"""
function writeeventwithhash(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String
    return writeevent(event; root=root)
end

"""
    validatehashchain(folder::String)::Tuple{Bool, Vector{String}}

Compatibility helper retained for callers expecting this API.
TradeLog hash-chain validation is disabled by design.
"""
function validatehashchain(folder::String)::Tuple{Bool, Vector{String}}
    return (true, String["hash-chain validation disabled in TradeLog"])
end

"""
    arrowexportfile(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String

Return the Arrow export file path for a given audit event's date and folder.
"""
function arrowexportfile(event::AuditEventRow; root::Union{Nothing, AbstractString}=nothing)::String
    folder = auditfolder(event; root=root)
    return normpath(joinpath(folder, "events.arrow"))
end

"""
    readjsonlauditevents(jsonl_path::String)::Vector{Dict{String, Any}}

Read all events from an audit JSONL file and return as vector of Dicts.
"""
function readjsonlauditevents(jsonl_path::String)::Vector{Dict{String, Any}}
    events = Dict{String, Any}[]
    if !isfile(jsonl_path)
        return events
    end
    
    for line in eachline(jsonl_path)
        isempty(strip(line)) && continue
        try
            json_obj = JSON3.read(line)
            # Convert JSON3 object to Dict with String keys
            event_dict = Dict{String, Any}()
            for (k, v) in pairs(json_obj)
                event_dict[String(k)] = v
            end
            push!(events, event_dict)
        catch err
            # Skip malformed lines
        end
    end
    return events
end

"""
    writearrowauditexport(jsonl_path::String; arrow_path::Union{Nothing, String}=nothing)::String

Convert audit JSONL file to Arrow format for efficient columnar storage.
Returns the path to the written Arrow file.
"""
function writearrowauditexport(jsonl_path::String; arrow_path::Union{Nothing, String}=nothing)::String
    events = readjsonlauditevents(jsonl_path)
    isempty(events) && return ""
    
    # Determine arrow output path if not provided
    if isnothing(arrow_path)
        dir = dirname(jsonl_path)
        arrow_path = joinpath(dir, "events.arrow")
    end
    
    # Extract all column names
    all_keys = Set{String}()
    for event in events
        for key in keys(event)
            push!(all_keys, key)
        end
    end
    
    # Build column dict with proper types
    columns = Dict{String, Vector}()
    for key in sort(collect(all_keys))
        col_data = Vector{Union{Missing, Any}}()
        for event in events
            push!(col_data, get(event, key, missing))
        end
        columns[key] = col_data
    end
    
    # Write Arrow table from columns
    try
        # Create a table using a named tuple-like approach
        nt = NamedTuple(Symbol(k) => v for (k, v) in columns)
        table = Arrow.Table(nt)
        Arrow.write(arrow_path, table)
    catch
        # Fallback: write with Arrow.write directly on dict
        Arrow.write(arrow_path, columns)
    end
    
    return arrow_path
end

"""
    readarrowauditexport(arrow_path::String)::Arrow.Table

Read audit events from Arrow export file.
"""
function readarrowauditexport(arrow_path::String)::Arrow.Table
    return Arrow.Table(arrow_path)
end

end