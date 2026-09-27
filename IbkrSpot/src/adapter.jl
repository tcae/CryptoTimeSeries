"""One configured stock in the whitelist market scan."""
struct MarketScanInstrument
    symbol::String
    base_symbol::String
    quote_currency::String
    exchange::String
    primary_exchange::String
end

"""Mutable connection and contract cache owned by the IbkrSpot adapter."""
mutable struct IbkrSpotCache <: XchAdapter.XchAdapterCache
    config::NamedTuple
    market_scan_symbols::Vector{MarketScanInstrument}
    minimum_daily_usd_volume::Real
    daily_usd_volume_cache::Dict{String, Tuple{Date, Float64}}
    fx_daily_rate_cache::Dict{Tuple{String, Date}, Float64}
    connection::Union{Nothing, Jib.Connection}
    reader::Union{Nothing, Task}
    syminfodf::DataFrame
    contractdetails::Dict{String, Jib.ContractDetails}
    account::String
    connection_lock::ReentrantLock
end

const MARKET_SCAN_CONFIG_PATH = joinpath(@__DIR__, "..", "data", "market_scan_config.json")

"""Load and validate the configured whitelist and daily USD-volume threshold."""
function _market_scan_config()
    isfile(MARKET_SCAN_CONFIG_PATH) || error("missing IbkrSpot market scan config: $(MARKET_SCAN_CONFIG_PATH)")
    config = JSON3.read(read(MARKET_SCAN_CONFIG_PATH, String))
    haskey(config, "universe") || error("missing IbkrSpot market scan universe section")
    universe = config["universe"]
    get(universe, "mode", "") == "whitelist" || error("IbkrSpot market scan universe mode must be whitelist")
    haskey(universe, "symbols") || error("missing IbkrSpot market scan universe symbols")
    raw_symbols = universe["symbols"]
    raw_symbols isa AbstractVector || error("IbkrSpot market scan symbols must be an array")
    isempty(raw_symbols) && error("IbkrSpot market scan whitelist must contain at least one symbol")

    instruments = MarketScanInstrument[]
    for (index, raw) in enumerate(raw_symbols)
        for key in ("base_symbol", "quote_currency", "exchange", "primary_exchange")
            haskey(raw, key) || error("IbkrSpot market scan symbol index=$(index) is missing $(key)")
            raw[key] isa String || error("IbkrSpot market scan symbol index=$(index) $(key) must be a string")
        end
        base = uppercase(raw["base_symbol"])
        quote_currency = uppercase(raw["quote_currency"])
        route = uppercase(raw["exchange"])
        primary = uppercase(raw["primary_exchange"])
        isempty(base) && error("IbkrSpot market scan symbol index=$(index) base_symbol must not be empty")
        isempty(quote_currency) && error("IbkrSpot market scan symbol index=$(index) quote_currency must not be empty")
        length(quote_currency) == 3 || error("IbkrSpot market scan symbol index=$(index) quote_currency=$(quote_currency) must have three letters")
        isempty(route) && error("IbkrSpot market scan symbol index=$(index) exchange must not be empty")
        symbol = base * quote_currency
        any(instrument -> instrument.symbol == symbol, instruments) &&
            error("duplicate IbkrSpot market scan symbol=$(symbol)")
        push!(instruments, MarketScanInstrument(symbol, base, quote_currency, route, primary))
    end

    haskey(config, "minimum_daily_usd_volume") || error("missing minimum_daily_usd_volume in IbkrSpot market scan config")
    threshold = config["minimum_daily_usd_volume"]
    threshold isa Real || error("minimum_daily_usd_volume must be numeric")
    isfinite(threshold) && threshold > 0 || error("minimum_daily_usd_volume=$(threshold) must be finite and positive")
    return (symbols=instruments, minimum_daily_usd_volume=threshold)
end

"""Return the empty symbol metadata table used by the cache."""
function _empty_symbolinfo()::DataFrame
    return DataFrame(
        symbol=String[],
        basecoin=String[],
        quotecoin=String[],
        status=String[],
        ticksize=Float64[],
        baseprecision=Int[],
        quoteprecision=Int[],
        minbaseqty=Float64[],
        minquoteqty=Float64[],
        conid=Int[],
        longname=String[],
    )
end

"""Create a disconnected IBKR cache; TWS is connected lazily on first network operation."""
function IbkrSpotCache(; port::Union{Nothing, Int}=nothing, clientid::Union{Nothing, Int}=nothing)
    config = _connectionconfig()
    !isnothing(port) && (config = merge(config, (port=port,)))
    !isnothing(clientid) && (config = merge(config, (clientid=clientid,)))
    market_scan = _market_scan_config()
    cache = IbkrSpotCache(
        config,
        market_scan.symbols,
        market_scan.minimum_daily_usd_volume,
        Dict{String, Tuple{Date, Float64}}(),
        Dict{Tuple{String, Date}, Float64}(),
        nothing,
        nothing,
        _empty_symbolinfo(),
        Dict{String, Jib.ContractDetails}(),
        config.account,
        ReentrantLock(),
    )
    _initialize_adapter_config(cache)
    return cache
end

"""Initialize Xch's exchange-specific coin path and quote currency."""
function _initialize_adapter_config(cache::IbkrSpotCache)
    EnvConfig.setcoinspath!(exchangeid(cache))
    EnvConfig.setpairquote!(cache.config.currency)
    return nothing
end

"""Build the callback wrapper that feeds Jib responses into adapter state."""
function _adapterwrapper(cache::IbkrSpotCache)
    return Jib.Wrapper(
        nextValidId=orderid -> lock(_state_lock) do
            _nextvalidid[] = max(_nextvalidid[], orderid)
        end,
        managedAccounts=accounts -> begin
            available = filter(!isempty, strip.(split(accounts, ',')))
            isempty(cache.account) || cache.account in available || error("configured IBKR account=$(cache.account) is not managed by TWS accounts=$(accounts)")
            lock(_state_lock) do
                _managedaccounts[] = accounts
                isempty(cache.account) && !isempty(available) && (cache.account = first(available))
            end
        end,
        contractDetails=(reqid, details) -> _push!(reqid, details),
        contractDetailsEnd=reqid -> _complete!(reqid),
        currentTime=seconds -> begin
            reqid = lock(_state_lock) do
                _currenttime_reqid[]
            end
            if reqid > 0
                _push!(reqid, Dates.unix2datetime(seconds))
                _complete!(reqid)
            end
        end,
        tickPrice=(reqid, field, price, size, attributes) -> begin
            _ = attributes
            _ = size
            _push!(reqid, (kind=:price, field=field, price=price))
        end,
        tickSize=(reqid, field, size) -> _push!(reqid, (kind=:size, field=field, size=size)),
        tickString=(reqid, field, value) -> nothing,
        tickSnapshotEnd=reqid -> _complete!(reqid),
        tickGeneric=(reqid, field, value) -> nothing,
        marketDataType=(reqid, datatype) -> nothing,
        tickReqParams=(reqid, mintick, exchange, permissions) -> nothing,
        historicalData=(reqid, bars) -> foreach(bar -> _push!(reqid, bar), bars),
        historicalDataEnd=(reqid, startdate, enddate) -> begin
            _ = startdate
            _ = enddate
            _complete!(reqid)
        end,
        accountSummary=(reqid, account, tag, value, currency) ->
            _push!(reqid, (account=account, tag=tag, value=value, currency=currency)),
        accountSummaryEnd=reqid -> _complete!(reqid),
        position=(account, contract, quantity, averagecost) -> begin
            _ = averagecost
            reqid = lock(_state_lock) do
                _position_reqid[]
            end
            reqid > 0 && _push!(reqid, (account=account, contract=contract, quantity=quantity))
        end,
        positionEnd=() -> begin
            reqid = lock(_state_lock) do
                _position_reqid[]
            end
            reqid > 0 && _complete!(reqid)
        end,
        openOrder=(orderid, contract, orderdata, orderstate) -> begin
            reqid = lock(_state_lock) do
                state = get!(_orderstate, orderid, Dict{Symbol, Any}())
                state[:contract] = contract
                state[:order] = orderdata
                state[:state] = orderstate
                state[:status] = orderstate.status
                haskey(state, :created) || (state[:created] = Dates.now(Dates.UTC))
                state[:updated] = Dates.now(Dates.UTC)
                _openorders_reqid[]
            end
            reqid > 0 && _push!(reqid, (orderid=orderid, contract=contract, order=orderdata, state=orderstate))
        end,
        openOrderEnd=() -> begin
            reqid = lock(_state_lock) do
                _openorders_reqid[]
            end
            reqid > 0 && _complete!(reqid)
        end,
        orderStatus=(orderid, status, filled, remaining, averageprice, permid, parentid, lastfillprice, clientid, whyheld, marketcapprice) -> lock(_state_lock) do
            state = get!(_orderstate, orderid, Dict{Symbol, Any}())
            state[:status] = status
            state[:filled] = filled
            state[:remaining] = remaining
            state[:averageprice] = averageprice
            state[:updated] = Dates.now(Dates.UTC)
            state[:permid] = permid
            state[:parentid] = parentid
            state[:lastfillprice] = lastfillprice
            state[:clientid] = clientid
            state[:whyheld] = whyheld
            state[:marketcapprice] = marketcapprice
        end,
        error=(reqid, error_time, code, message, reject_json) -> begin
            _ = error_time
            _ = reject_json
            if !isnothing(code) && !(code in _INFO_ERROR_CODES)
                if !isnothing(reqid)
                    _fail!(reqid, "code=$(code) $(message)")
                    lock(_state_lock) do
                        if haskey(_orderstate, reqid)
                            _orderstate[reqid][:status] = "Rejected"
                            _orderstate[reqid][:whyheld] = message
                        end
                    end
                else
                    @warn "IBKR API error" code=code message=message
                end
            end
        end,
    )
end

"""Wait until TWS has supplied an order ID, its API-ready signal."""
function _wait_api_ready(; timeout::Dates.Period=REQUEST_TIMEOUT)
    deadline = Dates.now(Dates.UTC) + timeout
    while Dates.now(Dates.UTC) < deadline
        ready = lock(_state_lock) do
            _nextvalidid[] > 0
        end
        ready && return nothing
        sleep(REQUEST_POLL)
    end
    error("IbkrSpot TWS API did not provide nextValidId within $(timeout)")
end

"""Open a TWS connection and start its callback reader once per cache."""
function _connection!(cache::IbkrSpotCache)::Jib.Connection
    return lock(cache.connection_lock) do
        if !isnothing(cache.connection) && isopen(cache.connection.socket)
            return cache.connection
        end

        lock(_state_lock) do
            _nextvalidid[] = 0
            _managedaccounts[] = ""
        end
        connection = Jib.connect(cache.config.host, cache.config.port, cache.config.clientid)
        cache.connection = connection
        cache.reader = Jib.start_reader(connection, _adapterwrapper(cache))
        _wait_api_ready()
        return connection
    end
end

"""Close the TWS socket owned by `cache`."""
function disconnect!(cache::IbkrSpotCache)
    lock(cache.connection_lock) do
        if !isnothing(cache.connection)
            Jib.disconnect(cache.connection)
            cache.connection = nothing
            cache.reader = nothing
        end
    end
    return nothing
end

"""Send a request with a correlated callback collector and return its response values."""
function _requestvalues(cache::IbkrSpotCache, what::AbstractString, send::Function; timeout::Dates.Period=REQUEST_TIMEOUT)
    connection = _connection!(cache)
    reqid = _nextreqid()
    _opencollector!(reqid)
    try
        send(connection, reqid)
        return _await(reqid, what; timeout=timeout)
    catch
        _closecollector!(reqid)
        rethrow()
    end
end

"""Return the canonical symbol with separators removed and case normalized."""
function _canonical_symbol(symbol::AbstractString)::String
    return uppercase(replace(symbol, "/" => ""))
end

"""Build the stock contract requested by an adapter ticker symbol."""
function _stock_contract(cache::IbkrSpotCache, symbol::AbstractString)::Jib.Contract
    canonical = _canonical_symbol(symbol)
    configured = findfirst(instrument -> instrument.symbol == canonical, cache.market_scan_symbols)
    if !isnothing(configured)
        instrument = cache.market_scan_symbols[configured]
        contract = Jib.Contract(
            symbol=instrument.base_symbol,
            secType="STK",
            exchange=instrument.exchange,
            currency=instrument.quote_currency,
        )
        isempty(instrument.primary_exchange) || setfield!(contract, :primaryExchange, instrument.primary_exchange)
        return contract
    end

    currency = uppercase(cache.config.currency)
    @assert endswith(canonical, currency) && length(canonical) > length(currency) "IBKR symbol=$(symbol) must be a base ticker followed by quote currency=$(currency)"
    basecoin = canonical[begin:end-length(currency)]
    contract = Jib.Contract(
        symbol=basecoin,
        secType="STK",
        exchange=cache.config.exchange_route,
        currency=currency,
    )
    isempty(cache.config.primary_exchange) || setfield!(contract, :primaryExchange, cache.config.primary_exchange)
    return contract
end

"""Return cached IBKR contract details, querying TWS on first use."""
function _contractdetails(cache::IbkrSpotCache, symbol::AbstractString)::Union{Nothing, Jib.ContractDetails}
    canonical = _canonical_symbol(symbol)
    haskey(cache.contractdetails, canonical) && return cache.contractdetails[canonical]
    requested_contract = _stock_contract(cache, canonical)
    details = try
        _requestvalues(cache, "contract details $(canonical)",
            (connection, reqid) -> Jib.reqContractDetails(connection, reqid, requested_contract))
    catch exception
        occursin("code=200", sprint(showerror, exception)) && return nothing
        rethrow()
    end
    isempty(details) && return nothing
    selected = first(details)
    cache.contractdetails[canonical] = selected
    _cache_symbolinfo!(cache, canonical, selected)
    return selected
end

"""Cache the shared Xch symbol metadata row for one IBKR stock contract."""
function _cache_symbolinfo!(cache::IbkrSpotCache, symbol::String, details::Jib.ContractDetails)
    findfirst(==(symbol), cache.syminfodf.symbol) === nothing || return nothing
    contract = details.contract
    increment = something(details.sizeIncrement, 1.0)
    @assert increment > 0 "IBKR contract $(symbol) has invalid sizeIncrement=$(increment)"
    @assert details.minTick > 0 "IBKR contract $(symbol) has invalid minTick=$(details.minTick)"
    minbaseqty = something(details.minSize, increment)
    precision = increment >= 1.0 ? 0 : max(0, ceil(Int, -log10(increment)))
    push!(cache.syminfodf, (
        symbol=symbol,
        basecoin=contract.symbol,
        quotecoin=contract.currency,
        status=contract.conId > 0 ? "TRADING" : "UNTRADABLE",
        ticksize=details.minTick,
        baseprecision=precision,
        quoteprecision=max(0, ceil(Int, -log10(details.minTick))),
        minbaseqty=minbaseqty,
        minquoteqty=0.0,
        conid=contract.conId,
        longname=details.longName,
    ))
    return nothing
end

"""Return one cached symbol row, resolving contract details through TWS if needed."""
function symbolinfo(cache::IbkrSpotCache, symbol::AbstractString)::Union{Nothing, DataFrameRow}
    canonical = _canonical_symbol(symbol)
    ix = findfirst(==(canonical), cache.syminfodf.symbol)
    if isnothing(ix)
        isnothing(_contractdetails(cache, canonical)) && return nothing
        ix = findfirst(==(canonical), cache.syminfodf.symbol)
    end
    return isnothing(ix) ? nothing : cache.syminfodf[ix, :]
end

"""Resolve symbol metadata from base and quote currencies."""
symbolinfo(cache::IbkrSpotCache, basecoin::AbstractString, quotecoin::AbstractString) = symbolinfo(cache, symboltoken(cache, basecoin, quotecoin))

"""Return whether an IBKR stock symbol resolves to an eligible USD contract."""
function validsymbol(cache::IbkrSpotCache, symbol::AbstractString)::Bool
    info = symbolinfo(cache, symbol)
    return !isnothing(info) && validsymbol(cache, info)
end

"""Validate an IBKR symbol metadata row."""
function validsymbol(cache::IbkrSpotCache, info::Union{Nothing, DataFrameRow})::Bool
    return !isnothing(info) && uppercase(String(info.quotecoin)) == uppercase(cache.config.currency) && info.status == "TRADING"
end

"""Validate an IBKR base/quote pair."""
function validsymbol(cache::IbkrSpotCache, basecoin::AbstractString, quotecoin::AbstractString)::Bool
    uppercase(quotecoin) == uppercase(cache.config.currency) || return false
    return validsymbol(cache, symboltoken(cache, basecoin, quotecoin))
end

"""Return IBKR's reported minimum increments for a stock symbol."""
function marginlimits(cache::IbkrSpotCache, symbol::AbstractString)
    isnothing(symbolinfo(cache, symbol)) && return (maxleveragebuy=0, maxleveragesell=0)
    shortspec = _executionorderspec(:short)
    return (maxleveragebuy=1, maxleveragesell=shortspec.leverage)
end

"""Return whether the configured stock leverage permits the requested side."""
function marginpermitted(cache::IbkrSpotCache, symbol::AbstractString, orderside::AbstractString, leverage::Signed)::Bool
    side = lowercase(orderside)
    @assert side in ("buy", "sell") "marginpermitted symbol=$(symbol) orderside=$(orderside) must be buy or sell"
    leverage <= 1 && return true
    limits = marginlimits(cache, symbol)
    return side == "buy" ? limits.maxleveragebuy >= leverage : limits.maxleveragesell >= leverage
end

"""Return current time reported by the TWS server."""
function servertime(cache::IbkrSpotCache)::DateTime
    connection = _connection!(cache)
    reqid = _nextreqid()
    _opencollector!(reqid)
    lock(_state_lock) do
        _currenttime_reqid[] = reqid
    end
    try
        Jib.reqCurrentTime(connection)
        values = _await(reqid, "server time")
        return only(values)
    catch
        _closecollector!(reqid)
        rethrow()
    finally
        lock(_state_lock) do
            _currenttime_reqid[] = 0
        end
    end
end

const _BAR_SIZES = Dict(
    "1m" => (setting="1 min", seconds=60),
    "5m" => (setting="5 mins", seconds=300),
    "15m" => (setting="15 mins", seconds=900),
    "30m" => (setting="30 mins", seconds=1800),
    "1h" => (setting="1 hour", seconds=3600),
    "4h" => (setting="4 hours", seconds=14400),
    "1d" => (setting="1 day", seconds=86400),
)

"""Return a normalized ticker key for an IBKR tick type label or numeric code."""
function _tickkey(field::AbstractString)::String
    normalized = uppercase(replace(field, "_" => ""))
    if normalized in ("BID", "BIDPRICE", "1")
        return "BID"
    elseif normalized in ("ASK", "ASKPRICE", "2")
        return "ASK"
    elseif normalized in ("LAST", "LASTPRICE", "4")
        return "LAST"
    elseif normalized in ("CLOSE", "CLOSEPRICE", "9")
        return "CLOSE"
    elseif normalized in ("VOLUME", "8")
        return "VOLUME"
    end
    return normalized
end

"""Convert a Jib historical bar timestamp to UTC DateTime."""
function _bar_datetime(value)::DateTime
    seconds = value isa Integer ? value : parse(Int, value)
    return Dates.unix2datetime(seconds)
end

"""Parse IBKR's `yyyyMMdd` label for a daily historical bar."""
function _dailybar_date(value::AbstractString)::Date
    return Date(value, dateformat"yyyymmdd")
end

"""Convert TWS UTC server time to one contract's exchange-local calendar date."""
function _exchange_market_date(scan_time::DateTime, timezone::AbstractString)::Date
    utc_time = ZonedDateTime(scan_time, TimeZone("UTC"))
    timezone_mask = TimeZones.Class(:FIXED) | TimeZones.Class(:STANDARD) | TimeZones.Class(:LEGACY)
    local_time = astimezone(utc_time, TimeZone(timezone, timezone_mask))
    return Date(DateTime(local_time))
end

"""Select the latest daily bar strictly before the current TWS calendar date."""
function _latest_completed_daily_bar(bars, current_date::Date)
    selected = nothing
    selected_date = nothing
    for bar in bars
        bar_date = _dailybar_date(bar.time)
        if bar_date < current_date && (isnothing(selected_date) || bar_date > selected_date)
            selected = bar
            selected_date = bar_date
        end
    end
    return isnothing(selected) ? nothing : (date=selected_date, bar=selected)
end

"""Return the latest completed daily exchange rate from `currency` to USD."""
function _daily_usd_rate(cache::IbkrSpotCache, currency::String, session_date::Date, scan_time::DateTime)::Float64
    currency == "USD" && return 1.0
    key = (currency, session_date)
    cached = get(cache.fx_daily_rate_cache, key, nothing)
    !isnothing(cached) && return cached

    contract = Jib.Contract(symbol=currency, secType="CASH", exchange="IDEALPRO", currency="USD")
    endstring = Dates.format(scan_time, "yyyymmdd HH:MM:SS") * " UTC"
    bars = _requestvalues(cache, "daily FX rate $(currency)/USD",
        (connection, reqid) -> Jib.reqHistoricalData(connection, reqid, contract, endstring, "10 D", "1 day", "MIDPOINT", false, 2, false))
    selected = _latest_completed_daily_bar(bars, session_date + Dates.Day(1))
    isnothing(selected) && error("IbkrSpot has no completed $(currency)/USD daily midpoint for stock session=$(session_date)")
    rate = selected.bar.close
    isfinite(rate) && rate > 0 || error("IbkrSpot $(currency)/USD daily midpoint=$(rate) is invalid for session=$(selected.date)")
    cache.fx_daily_rate_cache[key] = rate
    return rate
end

"""Return the most recent completed RTH daily USD turnover for one configured stock."""
function _daily_usd_volume(cache::IbkrSpotCache, instrument::MarketScanInstrument, scan_time::DateTime)::Union{Nothing, Float64}
    details = _contractdetails(cache, instrument.symbol)
    isnothing(details) && return nothing
    scan_date = _exchange_market_date(scan_time, details.timeZoneId)
    cached = get(cache.daily_usd_volume_cache, instrument.symbol, nothing)
    !isnothing(cached) && cached[1] == scan_date && return cached[2]

    endstring = Dates.format(scan_time, "yyyymmdd HH:MM:SS") * " UTC"
    bars = _requestvalues(cache, "daily market scan $(instrument.symbol)",
        (connection, reqid) -> Jib.reqHistoricalData(connection, reqid, details.contract, endstring, "10 D", "1 day", "TRADES", true, 2, false))
    selected = _latest_completed_daily_bar(bars, scan_date)
    isnothing(selected) && return nothing
    selected.bar.volume >= 0 && selected.bar.wap > 0 ||
        error("IbkrSpot daily bar for $(instrument.symbol) on $(selected.date) has invalid volume=$(selected.bar.volume) WAP=$(selected.bar.wap)")
    quote_volume = selected.bar.volume * selected.bar.wap
    usd_volume = quote_volume * _daily_usd_rate(cache, instrument.quote_currency, selected.date, scan_time)
    cache.daily_usd_volume_cache[instrument.symbol] = (scan_date, usd_volume)
    return usd_volume
end

"""Return one IBKR quote snapshot in the common Xch ticker shape."""
function _tickerrow(cache::IbkrSpotCache, symbol::AbstractString)::Union{Nothing, NamedTuple}
    details = _contractdetails(cache, symbol)
    isnothing(details) && return nothing
    contract = details.contract
    ticks = _requestvalues(cache, "market data $(symbol)",
        (connection, reqid) -> Jib.reqMktData(connection, reqid, contract, "", true, false))
    values = Dict{String, Float64}()
    for tick in ticks
        key = _tickkey(tick.field)
        if tick.kind == :price && !isnothing(tick.price) && tick.price > 0
            values[key] = tick.price
        elseif tick.kind == :size && key == "VOLUME" && tick.size >= 0
            values[key] = tick.size
        end
    end
    lastprice = get(values, "LAST", get(values, "CLOSE", 0.0))
    lastprice > 0 || return nothing
    previousclose = get(values, "CLOSE", lastprice)
    volume = get(values, "VOLUME", 0.0)
    return (
        symbol=_canonical_symbol(symbol),
        askprice=get(values, "ASK", lastprice),
        bidprice=get(values, "BID", lastprice),
        lastprice=lastprice,
        quotevolume24h=volume * lastprice,
        pricechangepercent=previousclose > 0 ? (lastprice - previousclose) / previousclose : 0.0,
    )
end

"""Return one ticker row or the configured market scan.

The no-symbol form screens the configured whitelist by the most recent completed
RTH daily USD turnover, then requests quotes only for symbols above the threshold.
IBKR scanner results are not used because they are capped and do not report USD turnover.
"""
function get24h(cache::IbkrSpotCache, symbol=nothing; symbols::Union{Nothing, AbstractVector{<:AbstractString}}=nothing)
    if !isnothing(symbol)
        isnothing(symbols) || error("get24h accepts either one symbol or a scan subset, not both")
        row = _tickerrow(cache, String(symbol))
        return isnothing(row) ? nothing : DataFrame([row])[1, :]
    end
    output = DataFrame(
        askprice=Float64[],
        bidprice=Float64[],
        lastprice=Float64[],
        quotevolume24h=Float64[],
        pricechangepercent=Float64[],
        symbol=String[],
    )
    selected_instruments = if isnothing(symbols)
        cache.market_scan_symbols
    else
        configured = Dict(instrument.symbol => instrument for instrument in cache.market_scan_symbols)
        requested_symbols = unique(_canonical_symbol.(symbols))
        all(haskey(configured, requested) for requested in requested_symbols) ||
            error("get24h scan subset contains symbols not in the configured whitelist: $(requested_symbols)")
        [configured[requested] for requested in requested_symbols]
    end
    scan_time = servertime(cache)
    for instrument in selected_instruments
        daily_usd_volume = _daily_usd_volume(cache, instrument, scan_time)
        if isnothing(daily_usd_volume)
            @warn "Skipping IbkrSpot market-scan symbol without a completed daily bar" symbol=instrument.symbol
            continue
        end
        daily_usd_volume >= cache.minimum_daily_usd_volume || continue
        row = _tickerrow(cache, instrument.symbol)
        isnothing(row) || push!(output, merge(row, (quotevolume24h=daily_usd_volume,)))
    end
    return output
end

"""Return historical stock bars using the shared Ohlcv DataFrame schema."""
function getklines(cache::IbkrSpotCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m")
    haskey(_BAR_SIZES, String(interval)) || error("unsupported IbkrSpot interval=$(interval)")
    details = _contractdetails(cache, String(symbol))
    isnothing(details) && return DataFrame(opentime=DateTime[], open=Float64[], high=Float64[], low=Float64[], close=Float64[], basevolume=Float64[])
    endtime = isnothing(endDateTime) ? servertime(cache) : endDateTime
    starttime = isnothing(startDateTime) ? endtime - Dates.Day(1) : startDateTime
    durationdays = max(1, ceil(Int, Dates.value(endtime - starttime) / 86_400_000))
    duration = durationdays <= 365 ? "$(durationdays) D" : "$(ceil(Int, durationdays / 365)) Y"
    endstring = Dates.format(endtime, "yyyymmdd HH:MM:SS") * " UTC"
    barsize = _BAR_SIZES[String(interval)].setting
    bars = _requestvalues(cache, "historical data $(symbol)",
        (connection, reqid) -> Jib.reqHistoricalData(connection, reqid, details.contract, endstring, duration, barsize, "TRADES", true, 2, false))
    output = DataFrame(opentime=DateTime[], open=Float64[], high=Float64[], low=Float64[], close=Float64[], basevolume=Float64[])
    for bar in bars
        opentime = _bar_datetime(bar.time)
        starttime <= opentime <= endtime || continue
        push!(output, (opentime=opentime, open=bar.open, high=bar.high, low=bar.low, close=bar.close, basevolume=bar.volume))
    end
    return sort!(output, :opentime)
end

"""Request the IBKR account summary values required by balances and capacity."""
function _accountsummary(cache::IbkrSpotCache)
    _wait_account(cache)
    tags = "NetLiquidation,TotalCashValue,AvailableFunds,InitMarginReq,MaintMarginReq"
    return _requestvalues(cache, "account summary",
        (connection, reqid) -> Jib.reqAccountSummary(connection, reqid, "All", tags))
end

"""Return rows for the cache's selected TWS account only."""
function _selected_account_rows(cache::IbkrSpotCache, rows)
    _wait_account(cache)
    return filter(row -> row.account == cache.account, rows)
end

"""Wait for and return the configured or first TWS-managed account identifier."""
function _wait_account(cache::IbkrSpotCache)::String
    deadline = Dates.now(Dates.UTC) + REQUEST_TIMEOUT
    while Dates.now(Dates.UTC) < deadline
        accounts = lock(_state_lock) do
            _managedaccounts[]
        end
        available = filter(!isempty, strip.(split(accounts, ',')))
        if !isempty(available)
            if isempty(cache.account)
                cache.account = first(available)
            end
            cache.account in available || error("configured IBKR account=$(cache.account) is not managed by TWS accounts=$(accounts)")
            return cache.account
        end
        sleep(REQUEST_POLL)
    end
    error("IbkrSpot TWS API did not report a managed account within $(REQUEST_TIMEOUT)")
end

"""Return quote-currency account cash and buying power in the shared balance schema."""
function balances(cache::IbkrSpotCache)::DataFrame
    summary = _accountsummary(cache)
    metrics = Dict{String, Float64}()
    for row in _selected_account_rows(cache, summary)
        uppercase(row.currency) == uppercase(cache.config.currency) || continue
        metrics[row.tag] = parse(Float64, row.value)
    end
    cash = get(metrics, "TotalCashValue", 0.0)
    free = max(0.0, cash)
    return DataFrame(
        coin=[uppercase(cache.config.currency)],
        locked=[0.0],
        free=[free],
        borrowed=[max(0.0, -cash)],
        accruedinterest=[0.0],
    )
end

"""Return IBKR stock positions as separate positive long and short quantities."""
function positionsnapshot(cache::IbkrSpotCache)::DataFrame
    connection = _connection!(cache)
    _wait_account(cache)
    reqid = _nextreqid()
    _opencollector!(reqid)
    lock(_state_lock) do
        _position_reqid[] = reqid
    end
    try
        Jib.reqPositions(connection)
        rows = _await(reqid, "positions")
        output = DataFrame(coin=String[], long_qty=Float64[], short_qty=Float64[])
        for row in rows
            row.account == cache.account || continue
            uppercase(row.contract.currency) == uppercase(cache.config.currency) || continue
            quantity = row.quantity
            push!(output, (
                coin=uppercase(row.contract.symbol),
                long_qty=max(0.0, quantity),
                short_qty=max(0.0, -quantity),
            ))
        end
        return output
    catch
        _closecollector!(reqid)
        rethrow()
    finally
        Jib.cancelPositions(connection)
        lock(_state_lock) do
            _position_reqid[] = 0
        end
    end
end

"""Return IBKR net liquidation, available funds, and margin requirements."""
function accountcapacity(cache::IbkrSpotCache)
    summary = _accountsummary(cache)
    metrics = Dict{String, Float64}()
    for row in _selected_account_rows(cache, summary)
        uppercase(row.currency) == uppercase(cache.config.currency) || continue
        metrics[row.tag] = parse(Float64, row.value)
    end
    equity = get(metrics, "NetLiquidation", 0.0)
    available = get(metrics, "AvailableFunds", 0.0)
    return (
        equity_quote=max(0.0, equity),
        available_opening_quote=max(0.0, available),
        available_long_quote=max(0.0, available),
        available_short_quote=max(0.0, available),
        initial_margin_quote=max(0.0, get(metrics, "InitMarginReq", 0.0)),
        maintenance_margin_quote=max(0.0, get(metrics, "MaintMarginReq", 0.0)),
        source="IbkrSpot:AccountSummary",
    )
end

"""Normalize one Jib open-order callback to the shared Xch row schema."""
function _orderrow(cache::IbkrSpotCache, row)::NamedTuple
    orderid = row.orderid
    state = lock(_state_lock) do
        copy(get(_orderstate, orderid, Dict{Symbol, Any}()))
    end
    contract = row.contract
    orderdata = row.order
    rawstatus = String(get(state, :status, row.state.status))
    filled = Float64(get(state, :filled, 0.0))
    status = normalize_order_status(cache, rawstatus)
    displaystatus = if status == "submitted"
        filled > 0 ? "PartiallyFilled" : "New"
    elseif status == "closed"
        "Filled"
    elseif status == "cancelled"
        "Cancelled"
    elseif status == "rejected"
        "Rejected"
    else
        titlecase(status)
    end
    created = get(state, :created, Dates.now(Dates.UTC))
    updated = get(state, :updated, created)
    return (
        orderid=string(orderid),
        orderLinkId=orderdata.orderRef,
        symbol=symboltoken(cache, contract.symbol, contract.currency),
        side=uppercase(orderdata.action) == "BUY" ? "Buy" : "Sell",
        baseqty=orderdata.totalQuantity,
        ordertype=orderdata.orderType,
        isLeverage=occursin("|short", lowercase(orderdata.orderRef)),
        timeinforce=orderdata.postOnly ? "PostOnly" : orderdata.tif,
        limitprice=something(orderdata.lmtPrice, 0.0),
        avgprice=Float64(get(state, :averageprice, 0.0)),
        executedqty=filled,
        status=displaystatus,
        created=created,
        updated=updated,
        rejectreason=String(get(state, :whyheld, "")),
        reduceonly=Bool(get(state, :reduceonly, occursin("|reduceonly", lowercase(orderdata.orderRef)))),
        lastcheck=Dates.now(Dates.UTC),
    )
end

"""Return current client open orders, optionally filtered by ticker or order id."""
function openorders(cache::IbkrSpotCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)::DataFrame
    connection = _connection!(cache)
    reqid = _nextreqid()
    _opencollector!(reqid)
    lock(_state_lock) do
        _openorders_reqid[] = reqid
    end
    try
        Jib.reqOpenOrders(connection)
        rows = _await(reqid, "open orders")
        output = emptyorders(cache)
        canonical = isnothing(symbol) ? nothing : _canonical_symbol(String(symbol))
        wantedid = isnothing(orderid) ? nothing : string(orderid)
        wantedlink = isnothing(orderLinkId) ? nothing : String(orderLinkId)
        for row in rows
            normalized = _orderrow(cache, row)
            !isnothing(canonical) && normalized.symbol != canonical && continue
            !isnothing(wantedid) && normalized.orderid != wantedid && continue
            !isnothing(wantedlink) && normalized.orderLinkId != wantedlink && continue
            push!(output, normalized)
        end
        return output
    catch
        _closecollector!(reqid)
        rethrow()
    finally
        lock(_state_lock) do
            _openorders_reqid[] = 0
        end
    end
end

"""Return one current or previously observed order by IBKR order id."""
function order(cache::IbkrSpotCache, orderid)
    isnothing(orderid) && return nothing
    matches = openorders(cache; orderid=orderid)
    if nrow(matches) > 0
        return matches[1, :]
    end
    numericid = tryparse(Int, string(orderid))
    isnothing(numericid) && return nothing
    row = lock(_state_lock) do
        state = get(_orderstate, numericid, nothing)
        isnothing(state) || !haskey(state, :contract) || !haskey(state, :order) ? nothing :
            (orderid=numericid, contract=state[:contract], order=state[:order], state=state[:state])
    end
    return isnothing(row) ? nothing : _orderrow(cache, row)
end

"""Build an IBKR stock order value without transmitting it."""
function _neworder(cache::IbkrSpotCache, orderid::Int, orderside::String, quantity::Real, price, configside::Symbol; maker::Bool=true, reduceonly::Bool=false)::Jib.Order
    orderdata = Jib.Order()
    orderdata.orderId = orderid
    orderdata.clientId = cache.config.clientid
    orderdata.action = uppercase(orderside)
    orderdata.totalQuantity = quantity
    orderdata.orderType = isnothing(price) ? "MKT" : "LMT"
    orderdata.lmtPrice = isnothing(price) ? nothing : price
    orderdata.tif = "GTC"
    orderdata.transmit = true
    orderdata.postOnly = maker && !isnothing(price)
    orderdata.orderRef = "IbkrSpot|$(configside)|$(orderid)|$(reduceonly ? "reduceonly" : "open")"
    orderdata.account = cache.account
    return orderdata
end

"""Return the position quantity available to close for one symbol and side."""
function _closequantity(cache::IbkrSpotCache, symbol::AbstractString, orderside::AbstractString)::Float64
    base = _stock_contract(cache, symbol).symbol
    positions = positionsnapshot(cache)
    ix = findfirst(==(uppercase(base)), positions.coin)
    isnothing(ix) && return 0.0
    return uppercase(orderside) == "SELL" ? positions.long_qty[ix] : positions.short_qty[ix]
end

"""Submit one stock order and return its initial normalized order row.

Reduce-only intent is checked against a fresh local position snapshot because
IBKR stock orders have no atomic reduce-only flag. Notional above `max_quote`
fails fast; oversized orders are not split into sequential iceberg children.
"""
function createorder(
    cache::IbkrSpotCache,
    symbol::String,
    orderside::String,
    basequantity::Real,
    price::Union{Real, Nothing},
    maker::Bool=true;
    configside::Union{Nothing, Symbol}=nothing,
    execution_spec=nothing,
    reduceonly::Bool=false,
    validate::Bool=false,
    venuepair::Union{Nothing, AbstractString}=nothing,
    adaptivepost::Union{Nothing, Bool}=nothing,
)
    _ = venuepair
    _ = adaptivepost
    @assert basequantity > 0 "createorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
    @assert isnothing(price) || price > 0 "createorder symbol=$(symbol) price=$(price) must be > 0"
    action = uppercase(orderside)
    @assert action in ("BUY", "SELL") "createorder symbol=$(symbol) orderside=$(orderside) must be Buy or Sell"
    side = isnothing(configside) ? (action == "BUY" ? :long : :short) : configside
    spec = isnothing(execution_spec) ? executionorderspec(cache, side) : execution_spec
    contract_details = _contractdetails(cache, symbol)
    isnothing(contract_details) && return nothing
    info = symbolinfo(cache, symbol)
    isnothing(info) && return nothing
    validsymbol(cache, info) || return nothing

    increment = something(contract_details.sizeIncrement, 1.0)
    @assert increment > 0 "createorder symbol=$(symbol) has invalid sizeIncrement=$(increment)"
    quantity = floor(basequantity / increment) * increment
    @assert quantity >= info.minbaseqty "createorder symbol=$(symbol) normalized quantity=$(quantity) is below minbaseqty=$(info.minbaseqty)"

    resolvedprice = price
    if isnothing(resolvedprice) && maker
        ticker = _tickerrow(cache, symbol)
        isnothing(ticker) && return nothing
        resolvedprice = action == "BUY" ? ticker.bidprice : ticker.askprice
    end
    if !isnothing(resolvedprice)
        ticks = resolvedprice / info.ticksize
        resolvedprice = (action == "BUY" ? floor(ticks) : ceil(ticks)) * info.ticksize
    end
    referenceprice = if isnothing(resolvedprice)
        ticker = _tickerrow(cache, symbol)
        isnothing(ticker) ? nothing : ticker.lastprice
    else
        resolvedprice
    end
    isnothing(referenceprice) && error("createorder symbol=$(symbol) requires a price to enforce max_quote=$(spec.max_quote)")
    notional = quantity * referenceprice
    @assert isnothing(spec.max_quote) || notional <= spec.max_quote "createorder symbol=$(symbol) notional=$(notional) exceeds configured max_quote=$(spec.max_quote)"
    if !reduceonly
        capacity = accountcapacity(cache)
        available = action == "BUY" ? capacity.available_long_quote : capacity.available_short_quote
        @assert notional <= available "createorder symbol=$(symbol) notional=$(notional) exceeds available $(action) quote capacity=$(available)"
    end
    if reduceonly
        closable = _closequantity(cache, symbol, action)
        @assert quantity <= closable "createorder reduceonly symbol=$(symbol) side=$(action) quantity=$(quantity) exceeds current closable quantity=$(closable)"
    end

    connection = _connection!(cache)
    _wait_api_ready()
    _wait_account(cache)
    orderid = _takeorderid()
    orderdata = _neworder(cache, orderid, action, quantity, resolvedprice, side; maker=maker, reduceonly=reduceonly)
    state = lock(_state_lock) do
        current = get!(_orderstate, orderid, Dict{Symbol, Any}())
        current[:contract] = contract_details.contract
        current[:order] = orderdata
        current[:status] = "PendingSubmit"
        current[:filled] = 0.0
        current[:created] = Dates.now(Dates.UTC)
        current[:updated] = current[:created]
        current[:reduceonly] = reduceonly
        current[:state] = (status="PendingSubmit",)
        copy(current)
    end
    validate && (orderdata.whatIf = true)
    Jib.placeOrder(connection, orderid, contract_details.contract, orderdata)
    return _orderrow(cache, (orderid=orderid, contract=state[:contract], order=state[:order], state=state[:state]))
end

"""Cancel one order and return its id after IBKR confirms cancellation."""
function cancelorder(cache::IbkrSpotCache, symbol, orderid)
    _ = symbol
    isnothing(orderid) && return nothing
    current = order(cache, orderid)
    isnothing(current) && return nothing
    lowercase(String(current.status)) in ("cancelled", "filled", "rejected") && return nothing
    numericid = parse(Int, String(current.orderid))
    Jib.cancelOrder(_connection!(cache), numericid, Jib.OrderCancel())
    deadline = Dates.now(Dates.UTC) + REQUEST_TIMEOUT
    while Dates.now(Dates.UTC) < deadline
        state = lock(_state_lock) do
            get(get(_orderstate, numericid, Dict{Symbol, Any}()), :status, "")
        end
        lowercase(String(state)) in ("cancelled", "apicancelled") && return String(current.orderid)
        lowercase(String(state)) == "filled" && return nothing
        sleep(REQUEST_POLL)
    end
    error("IbkrSpot cancellation timed out orderid=$(numericid)")
end

"""Amend the quantity or limit price of an existing IBKR order."""
function amendorder(cache::IbkrSpotCache, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
    current = order(cache, orderid)
    isnothing(current) && return nothing
    return amendorder(cache, String(current.symbol), orderid; basequantity=basequantity, limitprice=limitprice)
end

"""Amend one order using IBKR's same-order-id replace operation."""
function amendorder(cache::IbkrSpotCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
    @assert isnothing(basequantity) || basequantity > 0 "amendorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
    @assert isnothing(limitprice) || limitprice > 0 "amendorder symbol=$(symbol) limitprice=$(limitprice) must be > 0"
    current = order(cache, orderid)
    isnothing(current) && return nothing
    numericid = parse(Int, orderid)
    contract, orderdata = lock(_state_lock) do
        state = _orderstate[numericid]
        (state[:contract], deepcopy(state[:order]))
    end
    _canonical_symbol(symbol) == symboltoken(cache, contract.symbol, contract.currency) ||
        error("amendorder symbol=$(symbol) does not match IBKR order contract=$(contract.symbol)$(contract.currency)")
    details = _contractdetails(cache, symbol)
    isnothing(details) && return nothing
    info = symbolinfo(cache, symbol)
    isnothing(info) && return nothing
    increment = something(details.sizeIncrement, 1.0)
    @assert increment > 0 "amendorder symbol=$(symbol) has invalid sizeIncrement=$(increment)"
    quantity = isnothing(basequantity) ? orderdata.totalQuantity : floor(basequantity / increment) * increment
    @assert quantity >= info.minbaseqty "amendorder symbol=$(symbol) normalized quantity=$(quantity) is below minbaseqty=$(info.minbaseqty)"
    requestedprice = isnothing(limitprice) ? orderdata.lmtPrice : limitprice
    if !isnothing(requestedprice)
        ticks = requestedprice / info.ticksize
        requestedprice = (uppercase(orderdata.action) == "BUY" ? floor(ticks) : ceil(ticks)) * info.ticksize
    end
    referenceprice = if isnothing(requestedprice)
        ticker = _tickerrow(cache, symbol)
        isnothing(ticker) ? nothing : ticker.lastprice
    else
        requestedprice
    end
    isnothing(referenceprice) && error("amendorder symbol=$(symbol) requires a price to enforce max_quote")
    configside = occursin("|short|", lowercase(orderdata.orderRef)) ? :short : :long
    spec = _executionorderspec(configside)
    notional = quantity * referenceprice
    @assert isnothing(spec.max_quote) || notional <= spec.max_quote "amendorder symbol=$(symbol) notional=$(notional) exceeds configured max_quote=$(spec.max_quote)"
    orderdata.totalQuantity = quantity
    orderdata.lmtPrice = requestedprice
    Jib.placeOrder(_connection!(cache), numericid, contract, orderdata)
    return order(cache, orderid)
end

"""Create or update a position-closing stock order."""
function closeorder(cache::IbkrSpotCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}, maker::Bool=true; execution_spec=nothing, reduceonly::Bool=true, validate::Bool=false, venuepair::Union{Nothing, AbstractString}=nothing, adaptivepost::Union{Nothing, Bool}=nothing)
    @assert positionside in (:long, :short) "closeorder symbol=$(symbol) positionside=$(positionside) must be :long or :short"
    orderside = positionside == :long ? "Sell" : "Buy"
    return createorder(cache, symbol, orderside, basequantity, limitprice, maker;
        configside=positionside, execution_spec=execution_spec, reduceonly=reduceonly,
        validate=validate, venuepair=venuepair, adaptivepost=adaptivepost)
end

"""Create or amend one open-position order for the requested side."""
function upsertopenorder!(cache::IbkrSpotCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=false, lane::Union{Nothing, AbstractString}=nothing, pairref::Union{Nothing, XchAdapter.TradingPairRef}=nothing, adaptivepost::Union{Nothing, Bool}=nothing)
    _ = lane
    _ = pairref
    side = positionside == :long ? "Buy" : positionside == :short ? "Sell" : error("upsertopenorder! positionside=$(positionside) must be :long or :short")
    if !isnothing(existing_orderid)
        current = order(cache, String(existing_orderid))
        if !isnothing(current) && !(current.status in ("Filled", "Cancelled", "Rejected"))
            @assert current.symbol == _canonical_symbol(symbol) && current.side == side "upsertopenorder! existing order=$(existing_orderid) symbol=$(current.symbol) side=$(current.side) does not match requested symbol=$(symbol) side=$(side)"
            return amendorder(cache, symbol, String(existing_orderid); basequantity=basequantity, limitprice=limitprice)
        end
    end
    return createorder(cache, symbol, side, basequantity, limitprice, maker; configside=positionside, reduceonly=reduceonly, adaptivepost=adaptivepost)
end

"""Create or amend one position-closing order for the requested side."""
function upsertcloseorder!(cache::IbkrSpotCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=true, lane::Union{Nothing, AbstractString}=nothing, pairref::Union{Nothing, XchAdapter.TradingPairRef}=nothing, adaptivepost::Union{Nothing, Bool}=nothing)
    _ = lane
    _ = pairref
    if !isnothing(existing_orderid)
        current = order(cache, String(existing_orderid))
        if !isnothing(current) && !(current.status in ("Filled", "Cancelled", "Rejected"))
            orderside = positionside == :long ? "Sell" : positionside == :short ? "Buy" : error("upsertcloseorder! positionside=$(positionside) must be :long or :short")
            @assert current.symbol == _canonical_symbol(symbol) && current.side == orderside && current.reduceonly "upsertcloseorder! existing order=$(existing_orderid) does not match reduce-only close symbol=$(symbol) side=$(orderside)"
            return amendorder(cache, symbol, String(existing_orderid); basequantity=basequantity, limitprice=limitprice)
        end
    end
    return closeorder(cache, symbol, positionside, basequantity, limitprice, maker; reduceonly=reduceonly, adaptivepost=adaptivepost)
end

"""Verify that the predecessor and successor belong to the same IBKR stock."""
function directsequence!(cache::IbkrSpotCache, predecessor_orderid::AbstractString, successor_orderid::AbstractString)
    predecessor = order(cache, predecessor_orderid)
    successor = order(cache, successor_orderid)
    @assert !isnothing(predecessor) "directsequence! missing predecessor_orderid=$(predecessor_orderid)"
    @assert !isnothing(successor) "directsequence! missing successor_orderid=$(successor_orderid)"
    @assert predecessor.symbol == successor.symbol "directsequence! symbol mismatch predecessor=$(predecessor.symbol) successor=$(successor.symbol)"
    return (predecessor_orderid=String(predecessor_orderid), successor_orderid=String(successor_orderid), symbol=String(predecessor.symbol), acknowledged=true)
end

"""Return the exchange identifier used by Xch."""
exchangeid(::IbkrSpotCache)::String = "IbkrSpot"

"""Return the IBKR execution specification for the requested trading side."""
executionorderspec(::IbkrSpotCache, side::Symbol) = _executionorderspec(side)

"""Return the canonical concatenated base/quote ticker symbol."""
function symboltoken(::IbkrSpotCache, basecoin::AbstractString, quotecoin::AbstractString="USD")::String
    return uppercase(basecoin) * uppercase(quotecoin)
end

"""Return an empty order table matching the shared Xch adapter schema."""
function emptyorders(::IbkrSpotCache)::DataFrame
    return DataFrame(
        orderid=String[],
        orderLinkId=String[],
        symbol=String[],
        side=String[],
        baseqty=Float32[],
        ordertype=String[],
        isLeverage=Bool[],
        timeinforce=String[],
        limitprice=Float32[],
        avgprice=Float32[],
        executedqty=Float32[],
        status=String[],
        created=DateTime[],
        updated=DateTime[],
        rejectreason=String[],
        reduceonly=Bool[],
        lastcheck=DateTime[],
    )
end

"""Normalize IBKR order statuses into the shared Xch status vocabulary."""
function normalize_order_status(::IbkrSpotCache, rawstatus::AbstractString)::String
    status = lowercase(strip(rawstatus))
    if status in ("presubmitted", "pendingsubmit", "submitted", "pendingcancel")
        return "submitted"
    elseif status == "filled"
        return "closed"
    elseif status in ("cancelled", "apicancelled")
        return "cancelled"
    elseif status in ("inactive", "rejected")
        return "rejected"
    end
    return status
end