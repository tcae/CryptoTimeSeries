const WATCHLIST_EXPORT_DIRECTORY = joinpath(homedir(), "crypto", "IbkrSpot")

"""Infer the three-letter quote currency from a TWS watchlist name."""
function _watchlist_currency(name::AbstractString)::String
    parts = split(strip(name))
    length(parts) == 2 && parts[1] == "Watchlist" ||
        error("watchlist name=$(name) must have the form `Watchlist XXX`")
    currency = uppercase(parts[2])
    length(currency) == 3 || error("watchlist name=$(name) must end with a three-letter currency")
    return currency
end

"""Read the stock contract records from one headerless TWS watchlist CSV."""
function _read_watchlist(path::AbstractString, watchlist_name::AbstractString)
    currency = _watchlist_currency(watchlist_name)
    isfile(path) || error("missing TWS watchlist CSV: $(path)")
    instruments = MarketScanInstrument[]
    skipped_nonstocks = 0
    for (row_number, row) in enumerate(CSV.File(path; header=false))
        length(row) >= 4 || error("TWS watchlist=$(watchlist_name) row=$(row_number) has fewer than four fields")
        row[1] == "DES" || error("TWS watchlist=$(watchlist_name) row=$(row_number) has record type=$(row[1]); expected DES")
        all(!ismissing(row[index]) for index in 2:4) ||
            error("TWS watchlist=$(watchlist_name) row=$(row_number) is missing a symbol, security type, or exchange")
        security_type = uppercase(row[3])
        if security_type != "STK"
            skipped_nonstocks += 1
            continue
        end

        base_symbol = uppercase(row[2])
        route_fields = split(uppercase(row[4]), '/'; limit=2)
        exchange = first(route_fields)
        isempty(base_symbol) && error("TWS watchlist=$(watchlist_name) row=$(row_number) has an empty stock symbol")
        isempty(exchange) && error("TWS watchlist=$(watchlist_name) row=$(row_number) has an empty exchange")
        push!(instruments, MarketScanInstrument(base_symbol * currency, base_symbol, currency, exchange, ""))
    end
    return (instruments=instruments, skipped_nonstocks=skipped_nonstocks)
end

"""Merge symbols from TWS-exported watchlist CSVs into the market-scan whitelist.

CSV files are named `<watchlist name>.csv` in `watchlist_directory`. Only `STK`
records are imported; other security types are reported as skipped. Existing
base/quote pairs are preserved, so an already-configured contract is not replaced
by a second exchange route from a watchlist.
"""
function addwatchlists!(
    watchlist_names::AbstractVector{<:AbstractString};
    watchlist_directory::AbstractString=WATCHLIST_EXPORT_DIRECTORY,
    config_path::AbstractString=MARKET_SCAN_CONFIG_PATH,
)
    isempty(watchlist_names) && error("at least one TWS watchlist name is required")
    imports = MarketScanInstrument[]
    skipped_nonstocks = 0
    for name in watchlist_names
        path = joinpath(watchlist_directory, string(name, ".csv"))
        result = _read_watchlist(path, name)
        append!(imports, result.instruments)
        skipped_nonstocks += result.skipped_nonstocks
    end

    isfile(config_path) || error("missing IbkrSpot market scan config: $(config_path)")
    current = JSON3.read(read(config_path, String))
    haskey(current, "universe") || error("missing IbkrSpot market scan universe section")
    universe = current["universe"]
    get(universe, "mode", "") == "whitelist" || error("IbkrSpot market scan universe mode must be whitelist")
    existing_symbols = Dict{String, Any}[]
    existing_pairs = Set{Tuple{String, String}}()
    existing_quotes_by_base = Dict{String, Set{String}}()
    for entry in universe["symbols"]
        base_symbol = uppercase(entry["base_symbol"])
        quote_currency = uppercase(entry["quote_currency"])
        push!(existing_pairs, (base_symbol, quote_currency))
        push!(get!(existing_quotes_by_base, base_symbol, Set{String}()), quote_currency)
        push!(existing_symbols, Dict{String, Any}(
            "base_symbol" => base_symbol,
            "quote_currency" => quote_currency,
            "exchange" => uppercase(entry["exchange"]),
            "primary_exchange" => uppercase(entry["primary_exchange"]),
        ))
    end

    added = 0
    duplicates = 0
    quote_conflicts = 0
    for instrument in imports
        key = (instrument.base_symbol, instrument.quote_currency)
        if key in existing_pairs
            duplicates += 1
            continue
        end
        if haskey(existing_quotes_by_base, instrument.base_symbol)
            quote_conflicts += 1
            @warn "Skipping watchlist symbol with conflicting inferred quote currency" base_symbol=instrument.base_symbol watchlist_quote=instrument.quote_currency configured_quotes=collect(existing_quotes_by_base[instrument.base_symbol])
            continue
        end
        push!(existing_pairs, key)
        push!(get!(existing_quotes_by_base, instrument.base_symbol, Set{String}()), instrument.quote_currency)
        push!(existing_symbols, Dict{String, Any}(
            "base_symbol" => instrument.base_symbol,
            "quote_currency" => instrument.quote_currency,
            "exchange" => instrument.exchange,
            "primary_exchange" => instrument.primary_exchange,
        ))
        added += 1
    end

    if added > 0
        _write_market_scan_symbols!(config_path, current, existing_symbols)
    end

    return (added=added, duplicates=duplicates, quote_conflicts=quote_conflicts, skipped_nonstocks=skipped_nonstocks)
end

"""Atomically replace the whitelist while preserving the other scan settings."""
function _write_market_scan_symbols!(config_path::AbstractString, current, symbols::Vector{Dict{String, Any}})
    updated = Dict{String, Any}(
        "version" => current["version"],
        "universe" => Dict{String, Any}("mode" => "whitelist", "symbols" => symbols),
        "minimum_daily_usd_volume" => current["minimum_daily_usd_volume"],
    )
    temporary_path = string(config_path, ".tmp")
    try
        open(temporary_path, "w") do io
            JSON3.pretty(io, updated)
            write(io, '\n')
        end
        mv(temporary_path, config_path; force=true)
    catch
        isfile(temporary_path) && rm(temporary_path; force=true)
        rethrow()
    end
    return nothing
end

"""Refresh whitelist primary exchanges from authoritative TWS contract details.

TWS watchlist CSV exchange suffixes are treated as route hints, not as the
contract's primary listing exchange. This helper asks TWS for each configured
contract's resolved `primaryExchange` and writes those values back to the scan config.
"""
function refresh_primary_exchanges!(; port::Union{Nothing, Int}=nothing, clientid::Union{Nothing, Int}=nothing)
    resolved = Dict{Tuple{String, String}, String}()
    unresolved = 0
    cache = IbkrSpotCache(port=port, clientid=clientid)
    try
        for instrument in cache.market_scan_symbols
            details = _contractdetails(cache, instrument.symbol)
            if isnothing(details) || isempty(details.contract.primaryExchange)
                unresolved += 1
                resolved[(instrument.base_symbol, instrument.quote_currency)] = ""
                @warn "TWS did not resolve a primary exchange for watchlist stock" symbol=instrument.symbol
                continue
            end
            resolved[(instrument.base_symbol, instrument.quote_currency)] = uppercase(details.contract.primaryExchange)
        end
    finally
        disconnect!(cache)
    end

    current = JSON3.read(read(MARKET_SCAN_CONFIG_PATH, String))
    symbols = Dict{String, Any}[]
    updated_count = 0
    for entry in current["universe"]["symbols"]
        base_symbol = uppercase(entry["base_symbol"])
        quote_currency = uppercase(entry["quote_currency"])
        primary_exchange = get(resolved, (base_symbol, quote_currency), uppercase(entry["primary_exchange"]))
        primary_exchange == uppercase(entry["primary_exchange"]) || (updated_count += 1)
        push!(symbols, Dict{String, Any}(
            "base_symbol" => base_symbol,
            "quote_currency" => quote_currency,
            "exchange" => uppercase(entry["exchange"]),
            "primary_exchange" => primary_exchange,
        ))
    end
    if updated_count > 0
        _write_market_scan_symbols!(MARKET_SCAN_CONFIG_PATH, current, symbols)
    end
    return (updated=updated_count, unresolved=unresolved)
end