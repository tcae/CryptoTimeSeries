module KrakenSpot

using Base64, DataFrames, Dates, Downloads, EnvConfig, HTTP, JSON3, Logging, SHA
using XchAdapter
import XchAdapter: rawcache, exchangeid, symbolinfo, validsymbol, getklines, get24h, balances, emptyorders, openorders, order, cancelorder, createorder, amendorder, servertime, symboltoken, executionorderspec, marginlimits, marginpermitted, marketdataheartbeats, marketdataheartbeat, wsorderssnapshot, wsordersheartbeat, wsbalancessnapshot, wsbalancesheartbeat, ws_orders, ws_balances, accountcapacity, closeorder, upsertcloseorder!, upsertopenorder!, directsequence!, wsclosedkline
import XchAdapter: normalize_order_status

# Use HTTP.jl 1.x built-in WebSockets (compatible with Julia 1.11+ Memory-backed buffers).
# The standalone WebSockets.jl 1.x cannot convert SubArray{UInt8,1,Memory{UInt8},...}
# to SubArray{UInt8,1,Vector{UInt8},...} which was confirmed as root cause by Kraken support.
const WebSockets = HTTP.WebSockets

const _private_call_counter_lock = ReentrantLock()
const _private_call_counter = Dict{String, Int}()

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

const EXECUTION_CONFIG_PATH = joinpath(@__DIR__, "..", "data", "execution_config.json")

"Load the Kraken Spot execution configuration for side-specific order limits and instruments."
function executionconfig()
	isfile(EXECUTION_CONFIG_PATH) || error("missing KrakenSpot execution config: $(EXECUTION_CONFIG_PATH)")
	return JSON3.read(read(EXECUTION_CONFIG_PATH, String))
end

function _executionconfigside(configside::Union{Nothing, Symbol}, orderside::AbstractString)::Symbol
	if isnothing(configside)
		return lowercase(String(orderside)) == "buy" ? :long : :short
	end
	side = Symbol(lowercase(String(configside)))
	@assert side in (:long, :short) "invalid KrakenSpot configside=$(configside)"
	return side
end

function _executionorderspec(configside::Union{Nothing, Symbol}, orderside::AbstractString)
	side = _executionconfigside(configside, orderside)
	cfg = executionconfig()
	orders = cfg["orders"]
	sidecfg = orders[String(side)]
	instrument = lowercase(String(sidecfg["instrument"]))
	leverage = haskey(sidecfg, "leverage") ? Int(sidecfg["leverage"]) : 0
	max_quote = haskey(sidecfg, "max_quote") ? (sidecfg["max_quote"]) : nothing
	return (side=side, instrument=instrument, leverage=leverage, max_quote=max_quote)
end

"Return side-specific execution config owned by the KrakenSpot adapter."
function _executionorderspec(side::Symbol)
	side in (:long, :short) || error("KrakenSpot executionorderspec side=$(side) must be :long or :short")
	cfg = executionconfig()
	haskey(cfg, "orders") || error("missing KrakenSpot execution config orders section")
	orders = cfg["orders"]
	haskey(orders, String(side)) || error("missing KrakenSpot execution config orders.$(side) section")
	sidecfg = orders[String(side)]
	instrument = haskey(sidecfg, "instrument") ? lowercase(String(sidecfg["instrument"])) : ""
	leverage = haskey(sidecfg, "leverage") ? Int(sidecfg["leverage"]) : 0
	max_quote = haskey(sidecfg, "max_quote") ? (sidecfg["max_quote"]) : nothing
	return (side=side, instrument=instrument, leverage=leverage, max_quote=max_quote)
end

const KRAKEN_APIREST = "https://api.kraken.com"
const KRAKEN_WS_PUBLIC = "wss://ws.kraken.com/v2"
const KRAKEN_WS_PRIVATE = "wss://ws-auth.kraken.com/v2"
const _marketdata_ws_last_update_dt = Ref{Union{Nothing, DateTime}}(nothing)
const _marketdata_ws_symbol_lock = ReentrantLock()
const _marketdata_ws_last_update_by_symbol = Dict{String, DateTime}()
const _ws_kline_state_lock = ReentrantLock()
const _ws_last_kline_by_key = Dict{Tuple{String, String}, NamedTuple{(:opentime, :open, :high, :low, :close, :basevolume), Tuple{DateTime, Float32, Float32, Float32, Float32, Float32}}}()
const _ws_closed_kline_by_key = Dict{Tuple{String, String}, NamedTuple{(:opentime, :open, :high, :low, :close, :basevolume), Tuple{DateTime, Float32, Float32, Float32, Float32, Float32}}}()
const _ws_orders_state_lock = ReentrantLock()
const _ws_orders_snapshot = Ref{DataFrame}(DataFrame())
const _ws_orders_last_update_dt = Ref{Union{Nothing, DateTime}}(nothing)
const _ws_balances_state_lock = ReentrantLock()
const _ws_balances_snapshot = Ref{DataFrame}(DataFrame())
const _ws_balances_last_update_dt = Ref{Union{Nothing, DateTime}}(nothing)
const _ws_private_stream_lock = ReentrantLock()
const _ws_private_worker_running = Ref(false)
const _ws_private_orders_channel = Ref{Union{Nothing, Channel{Dict}}}(nothing)
const _ws_private_balances_channel = Ref{Union{Nothing, Channel{Dict}}}(nothing)
const _ws_private_subscribe_ack_timeout = Dates.Second(20)
const _ws_private_min_reconnect_interval = Dates.Second(2)
const _ws_private_backoff_base_seconds = 2.0
const _ws_private_backoff_cap_seconds = 60.0
const _ws_private_backoff_jitter_seconds = 0.75

const _interval2minutes = Dict(
	"1m" => 1,
	"5m" => 5,
	"15m" => 15,
	"30m" => 30,
	"1h" => 60,
	"4h" => 240,
	"1d" => 1440,
	"1w" => 10080,
	"15d" => 21600,
)

const _known_quotes = ["USDT", "USD", "USDC", "EUR", "BTC", "ETH"]

const _nonce_lock = ReentrantLock()
const _last_nonce = Ref{Int}(0)
const _last_nonce_ms = Ref{Int}(0)
const _nonce_ms_counter = Ref{Int}(0)
const _openpositions_state_lock = ReentrantLock()
const _openpositions_nonce_failures = Ref{Int}(0)
const _openpositions_disabled_until = Ref{Union{Nothing, DateTime}}(nothing)
const _openpositions_cache = Ref{Union{Nothing, Dict{String, Float32}}}(nothing)
const _openpositions_cache_time = Ref{Union{Nothing, DateTime}}(nothing)
const _openorders_cache_lock = ReentrantLock()
const _openorders_cache = Ref{Union{Nothing, DataFrame}}(nothing)
const _openorders_cache_time = Ref{Union{Nothing, DateTime}}(nothing)
const _private_rl_lock = ReentrantLock()
const _private_rl_cooldown_until = Ref{Union{Nothing, DateTime}}(nothing)

const ROBOT_ORDER_PREFIX = "ROBO-"
const _order_counter_lock = ReentrantLock()
const _order_counter = Ref{Int}(0)
# Lazily initialized on first order placement (runtime, not precompile time) so that
# cl_ord_id values are unique across process restarts even when loaded from .ji cache.
# Uses a process/time-derived short token so IDs stay within Kraken Spot's
# free-text cl_ord_id length limit (<= 18 chars including prefix).
const _order_session_token = Ref{String}("")

# Balance caching to avoid Kraken API rate limits (15 req/sec for tier 2)
# Cache TTL: 5 seconds. At 1 balance call/minute, this is well under the rate limit.
const _balance_cache_lock = ReentrantLock()
const _balance_cache = Ref{Union{Nothing, DataFrame}}(nothing)
const _balance_cache_time = Ref{Union{Nothing, DateTime}}(nothing)
const BALANCE_CACHE_TTL = Dates.Second(5)
const OPENPOSITIONS_CACHE_TTL = Dates.Minute(5)
const OPENORDERS_CACHE_TTL = Dates.Second(30)
const BALANCE_CACHE_MAX_STALE = Dates.Hour(2)
const OPENORDERS_CACHE_MAX_STALE = Dates.Hour(2)
const PRIVATE_READ_COOLDOWN = Dates.Second(45)
const SERVERTIME_RETRY_SECONDS = 60

"Store one websocket marketdata heartbeat timestamp for KrakenSpot adapter state."
function setmarketdataheartbeat!(dt::DateTime)
	_marketdata_ws_last_update_dt[] = dt
	return dt
end

"Store one websocket marketdata heartbeat timestamp using a cache handle for convenience."
function setmarketdataheartbeat!(bc, dt::DateTime)
	_ = bc
	return setmarketdataheartbeat!(dt)
end

"Store websocket marketdata heartbeat timestamp for one normalized symbol."
function setmarketdataheartbeat!(symbol::AbstractString, dt::DateTime)
	key = uppercase(String(symbol))
	lock(_marketdata_ws_symbol_lock) do
		_marketdata_ws_last_update_by_symbol[key] = dt
	end
	setmarketdataheartbeat!(dt)
	return dt
end

"Store websocket marketdata heartbeat timestamp for one symbol using a cache handle."
function setmarketdataheartbeat!(bc, symbol::AbstractString, dt::DateTime)
	_ = bc
	return setmarketdataheartbeat!(symbol, dt)
end

_klinekey(symbol::AbstractString, interval::AbstractString) = (uppercase(String(symbol)), String(interval))

function _recordwskline!(symbol::AbstractString, interval::AbstractString, candle)
	key = _klinekey(symbol, interval)
	lock(_ws_kline_state_lock) do
		prev = get(_ws_last_kline_by_key, key, nothing)
		if isnothing(prev)
			_ws_last_kline_by_key[key] = candle
			return candle
		end
		if candle.opentime > prev.opentime
			_ws_closed_kline_by_key[key] = prev
			_ws_last_kline_by_key[key] = candle
		elseif candle.opentime == prev.opentime
			_ws_last_kline_by_key[key] = candle
		end
		return candle
	end
end

"Return latest websocket order snapshot maintained by KrakenSpot adapter."
function _wsorderssnapshot()
	lock(_ws_orders_state_lock) do
		return copy(_ws_orders_snapshot[])
	end
end

"Return latest websocket order heartbeat timestamp from KrakenSpot adapter state."
_wsordersheartbeat() = _ws_orders_last_update_dt[]

"Return latest websocket balances snapshot maintained by KrakenSpot adapter."
function _wsbalancessnapshot()
	lock(_ws_balances_state_lock) do
		return copy(_ws_balances_snapshot[])
	end
end

"Return latest websocket balances heartbeat timestamp from KrakenSpot adapter state."
_wsbalancesheartbeat() = _ws_balances_last_update_dt[]

function _update_ws_orders_snapshot!(df::DataFrame)
	lock(_ws_orders_state_lock) do
		_ws_orders_snapshot[] = copy(df)
		_ws_orders_last_update_dt[] = Dates.now(Dates.UTC)
	end
	return nothing
end

function _touch_ws_orders_heartbeat!()
	lock(_ws_orders_state_lock) do
		_ws_orders_last_update_dt[] = Dates.now(Dates.UTC)
	end
	return nothing
end

function _update_ws_balances_snapshot!(df::DataFrame)
	lock(_ws_balances_state_lock) do
		_ws_balances_snapshot[] = copy(df)
		_ws_balances_last_update_dt[] = Dates.now(Dates.UTC)
	end
	return nothing
end

function _touch_ws_balances_heartbeat!()
	lock(_ws_balances_state_lock) do
		_ws_balances_last_update_dt[] = Dates.now(Dates.UTC)
	end
	return nothing
end

_ws_tryget(dict::AbstractDict, keys::AbstractVector{<:AbstractString}, default=nothing) = begin
	for key in keys
		haskey(dict, key) && return dict[key]
	end
	return default
end

function _ws_todatetime(value)
	if value isa DateTime
		return value
	elseif value isa Integer
		return Dates.unix2datetime(value > 10^11 ? value ÷ 1000 : value)
	elseif value isa Real
		iv = round(Int, value)
		return Dates.unix2datetime(iv > 10^11 ? iv ÷ 1000 : iv)
	elseif value isa AbstractString
		s = strip(value)
		s = replace(s, "Z" => "")
		if s == ""
			return Dates.now(Dates.UTC)
		end
		try
			return DateTime(s)
		catch
		end
		try
			return DateTime(s, dateformat"yyyy-mm-ddTHH:MM:SS.s")
		catch
		end
		try
			return _ws_todatetime(parse(Float64, s))
		catch
		end
	end
	return Dates.now(Dates.UTC)
end

function _ws_orderrow_from_dict(bc, entry::AbstractDict)
	oid = String(_ws_tryget(entry, ["order_id", "orderid", "id", "txid"], ""))
	oid == "" && return nothing
	rawsymbol = String(_ws_tryget(entry, ["symbol", "pair", "instrument", "product_id"], ""))
	symbol = rawsymbol == "" ? "" : _resultkey2symbol(bc, rawsymbol)
	side = lowercase(String(_ws_tryget(entry, ["side", "type", "direction"], "buy"))) == "buy" ? "Buy" : "Sell"
	ordertype = titlecase(String(_ws_tryget(entry, ["ordertype", "order_type", "orderType"], "Limit")))
	baseqty = _numstrict(_firstpresent(entry, ["vol", "qty", "size", "order_qty", "orderQty"]), "ws order $(oid) baseqty")
	executedqty = _numstrict(_firstpresent(entry, ["vol_exec", "filled", "filled_qty", "cum_qty", "cumQty"]), "ws order $(oid) executedqty")
	limitprice = _numstrict(_firstpresent(entry, ["price", "limit_price", "limitPrice"]), "ws order $(oid) limitprice")
	avgprice = _numstrict(_firstpresent(entry, ["avgPrice", "fill_price", "fillPrice", "price"]), "ws order $(oid) avgprice")
	status = lowercase(String(_ws_tryget(entry, ["status", "orderStatus"], "open")))
	created = _ws_todatetime(_ws_tryget(entry, ["created", "createdTime", "timestamp", "time", "opentm"], Dates.now(Dates.UTC)))
	updated = _ws_todatetime(_ws_tryget(entry, ["updated", "updatedTime", "lastUpdateTime", "timestamp"], created))
	orderLinkId = String(_ws_tryget(entry, ["cl_ord_id", "client_order_id", "clientOrderId", "cliOrdId"], ""))
	rejectreason = String(_ws_tryget(entry, ["rejectreason", "reason", "error"], ""))
	return (
		orderid=oid,
		orderLinkId=orderLinkId,
		symbol=symbol,
		side=side,
		baseqty=baseqty,
		ordertype=ordertype,
		isLeverage=false,
		timeinforce=String(_ws_tryget(entry, ["timeinforce", "timeInForce"], "GTC")),
		limitprice=limitprice,
		avgprice=avgprice,
		executedqty=executedqty,
		status=status,
		created=created,
		updated=updated,
		rejectreason=rejectreason,
		lastcheck=Dates.now(Dates.UTC),
	)
end

function _ws_orders_df_from_payload(bc, payload)
	if payload isa DataFrame
		return copy(payload)
	end
	rows = emptyordersschema(bc)
	payload isa AbstractVector || return rows
	for item in payload
		entry = item isa AbstractDict ? Dict(item) : Dict{String, Any}()
		isempty(entry) && continue
		row = _ws_orderrow_from_dict(bc, entry)
		isnothing(row) && continue
		push!(rows, row)
	end
	return rows
end

function _ws_balances_df_from_payload(payload)
	if payload isa DataFrame
		return copy(payload)
	end
	df = DataFrame(coin=AbstractString[], locked=Float32[], free=Float32[], borrowed=Float32[], accruedinterest=Float32[])
	payload isa AbstractVector || return df
	for item in payload
		entry = item isa AbstractDict ? Dict(item) : Dict{String, Any}()
		isempty(entry) && continue
		coinraw = _ws_tryget(entry, ["asset", "coin", "currency", "ccy"], "")
		coin = _normalizeasset(String(coinraw))
		coin == "" && continue
		free = _numstrict(_firstpresent(entry, ["available", "free", "walletBalance"]), "ws balance $(coin) free")
		locked = _numstrict(_firstpresent(entry, ["hold", "locked", "hold_trade", "initialMargin"]), "ws balance $(coin) locked")
		borrowed = _numstrict(_firstpresent(entry, ["borrowed", "borrow", "liability"]), "ws balance $(coin) borrowed")
		accruedinterest = _numstrict(_firstpresent(entry, ["accruedinterest", "interest"]), "ws balance $(coin) accruedinterest")
		push!(df, (coin=coin, locked=locked, free=free, borrowed=borrowed, accruedinterest=accruedinterest))
	end
	return df
end

"""
Log and reset a private endpoint call-rate summary for the current aggregation window.
Used by shutdown handlers to emit one final diagnostics snapshot.
"""
function log_private_call_summary!()
	lock(_private_call_counter_lock) do
		counts = collect(values(_private_call_counter))
		total_calls = sum(counts)
		endpoint_count = length(counts)
		max_calls = endpoint_count == 0 ? 0 : maximum(counts)
		avg_calls = endpoint_count == 0 ? 0.0 : (total_calls / endpoint_count)
		@info "[KrakenSpot private call rate summary]" total_calls_per_min=total_calls max_calls_per_endpoint_per_min=max_calls avg_calls_per_endpoint_per_min=round(avg_calls; digits=2) endpoints_tracked=endpoint_count
		empty!(_private_call_counter)
	end
	return nothing
end

"""
Cached KrakenSpot state used by higher-level trading modules.
"""
struct KrakenSpotCache <: XchAdapter.XchAdapterCache
	syminfodf::Union{Nothing, DataFrame}
	apirest::String
	publickey::String
	secretkey::String
end

executionorderspec(bc::KrakenSpotCache, side::Symbol) = _executionorderspec(side)
exchangeid(bc::KrakenSpotCache)::String = "KrakenSpot"

"Return latest websocket order snapshot using a cache handle."
function wsordersnapshot(bc::KrakenSpotCache)
	_ = bc
	return _wsorderssnapshot()
end

"Return latest websocket order heartbeat timestamp using a cache handle."
function wsordersheartbeat(bc::KrakenSpotCache)
	_ = bc
	return _wsordersheartbeat()
end

"Return latest websocket balances snapshot using a cache handle."
function wsbalancessnapshot(bc::KrakenSpotCache)
	_ = bc
	return _wsbalancessnapshot()
end

"Return latest websocket balances heartbeat timestamp using a cache handle."
function wsbalancesheartbeat(bc::KrakenSpotCache)
	_ = bc
	return _wsbalancesheartbeat()
end

"""Normalize Kraken Spot raw order status into Xch status vocabulary."""
function normalize_order_status(bc::KrakenSpotCache, rawstatus::AbstractString)::String
	_ = bc
	st = lowercase(String(rawstatus))
	if st in ["open", "new", "untouched"]
		return "submitted"
	elseif st in ["filled", "closed"]
		return "closed"
	elseif st in ["cancelled", "canceled"]
		return "canceled"
	elseif st in ["expired", "rejected", "cancel_reject", "error"]
		return "rejected"
	end
	return st
end

"""
Build a new `KrakenSpotCache` and optionally preload symbol metadata.

- `autoloadexchangeinfo=true` loads and caches `AssetPairs`
- if API keys are omitted, they are resolved from `EnvConfig.authorization` and then `ENV`
"""
function KrakenSpotCache(; autoloadexchangeinfo::Bool=true, apirest::String=KRAKEN_APIREST, publickey::Union{Nothing, AbstractString}=nothing, secretkey::Union{Nothing, AbstractString}=nothing)
	keys = _resolve_credentials(publickey, secretkey)
	syminfo = autoloadexchangeinfo ? _exchangeinfo(apirest) : _emptyexchangeinfo()
	if autoloadexchangeinfo && (size(syminfo, 1) > 0)
		targetquote = uppercase(EnvConfig.pairquote)
		filtered = syminfo[uppercase.(syminfo.quotecoin) .== targetquote, :]
		syminfo = size(filtered, 1) > 0 ? filtered : syminfo
		sort!(syminfo, :basecoin)
	end
	bc = KrakenSpotCache(syminfo, apirest, keys.publickey, keys.secretkey)
    EnvConfig.setcoinspath!(exchangeid(bc))
	EnvConfig.setpairquote!("USD")
    return bc
end

function marketdataheartbeats(bc::KrakenSpotCache)
	_ = bc
	lock(_marketdata_ws_symbol_lock) do
		return copy(_marketdata_ws_last_update_by_symbol)
	end
end

function marketdataheartbeat(bc::KrakenSpotCache; symbol::Union{Nothing, AbstractString}=nothing)
	_ = bc
	if isnothing(symbol)
		return _marketdata_ws_last_update_dt[]
	end
	key = uppercase(String(symbol))
	lock(_marketdata_ws_symbol_lock) do
		return get(_marketdata_ws_last_update_by_symbol, key, nothing)
	end
end

function wsclosedkline(bc::KrakenSpotCache, symbol::AbstractString, interval::AbstractString)
	_ = bc
	key = _klinekey(symbol, String(interval))
	lock(_ws_kline_state_lock) do
		return get(_ws_closed_kline_by_key, key, nothing)
	end
end

"""
Resolve Kraken API credentials from explicit args, `EnvConfig`, or environment variables.
"""
function _resolve_credentials(publickey::Union{Nothing, AbstractString}, secretkey::Union{Nothing, AbstractString})
	if !isnothing(publickey) && !isnothing(secretkey)
		return (publickey=String(publickey), secretkey=String(secretkey))
	end

	cfgpublic = ""
	cfgsecret = ""
	try
		if !isnothing(EnvConfig.authorization)
			cfgpublic = String(EnvConfig.authorization.key)
			cfgsecret = String(EnvConfig.authorization.secret)
		end
	catch
		cfgpublic = ""
		cfgsecret = ""
	end

	envpublic = get(ENV, "KRAKEN_APIKEY", "")
	envsecret = get(ENV, "KRAKEN_SECRET", "")

	resolvedpublic = isnothing(publickey) || (publickey == "") ? (cfgpublic != "" ? cfgpublic : envpublic) : String(publickey)
	resolvedsecret = isnothing(secretkey) || (secretkey == "") ? (cfgsecret != "" ? cfgsecret : envsecret) : String(secretkey)
	return (publickey=resolvedpublic, secretkey=resolvedsecret)
end

"""
Return `true` when API credentials are present in the cache.
"""
_hascredentials(bc::KrakenSpotCache)::Bool = (bc.publickey != "") && (bc.secretkey != "")

"""
Convert request parameter dictionary to URL encoded query string.
"""
function _dict2paramsget(dict::Union{Dict, Nothing})::String
	if isnothing(dict) || isempty(dict)
		return ""
	end
	parts = String[]
	for key in sort!(collect(keys(dict)); by=string)
		push!(parts, string(HTTP.escapeuri(string(key)), "=", HTTP.escapeuri(string(dict[key]))))
	end
	return join(parts, "&")
end

"""
Convert request parameter dictionary to URL encoded body string.
"""
_dict2paramspost(dict::Union{Dict, Nothing})::String = _dict2paramsget(dict)

"""
Low-level HMAC implementation used for Kraken request signing.
"""
function _hmac(key::Vector{UInt8}, msg::Vector{UInt8}, hashf, blocksize::Int=64)
	if length(key) > blocksize
		key = hashf(key)
	end
	pad = blocksize - length(key)
	if pad > 0
		resize!(key, blocksize)
		key[end - pad + 1:end] .= 0
	end
	o_key_pad = key .⊻ 0x5c
	i_key_pad = key .⊻ 0x36
	return hashf(vcat(o_key_pad, hashf(vcat(i_key_pad, msg))))
end

"""
Compute Kraken private API signature for one request.
"""
function _krakensignature(urlpath::String, nonce::String, postdata::String, secret::String)::String
	decoded = try
		Base64.base64decode(secret)
	catch
		Vector{UInt8}(secret)
	end
	sha256data = SHA.sha256(vcat(Vector{UInt8}(nonce), Vector{UInt8}(postdata)))
	payload = vcat(Vector{UInt8}(urlpath), sha256data)
	signature = _hmac(decoded, payload, SHA.sha512, 128)
	return Base64.base64encode(signature)
end

"""
Raise an error if a Kraken response reports one or more API errors.
"""
function _checkresponse(response::Dict, info::AbstractString)
	errors = get(response, "error", Any[])
	if (errors isa AbstractVector) && !isempty(errors)
		throw(ErrorException("Kraken API error in $(info): $(join(string.(errors), "; "))"))
	end
end

function _httpmemorycompaterror(err)::Bool
	msg = sprint(showerror, err)
	return (err isa MethodError) && occursin("SubArray{UInt8,1,Memory{UInt8}", msg) && occursin("SubArray{UInt8,1,Vector{UInt8}", msg)
end

function _isinvalidnonceerror(err)::Bool
	return occursin("invalid nonce", lowercase(sprint(showerror, err)))
end

function _isratelimiterror(err)::Bool
	return occursin("rate limit exceeded", lowercase(sprint(showerror, err)))
end

function _istransientnetworkerror(err)::Bool
	msg = lowercase(sprint(showerror, err))
	return occursin("econnreset", msg) || occursin("connection reset", msg) || occursin("timed out", msg) || occursin("timeout", msg) || occursin("recv failure", msg)
end

function _isreadonlyprivateendpoint(endPoint::AbstractString)::Bool
	return endPoint in [
		"/0/private/OpenOrders",
		"/0/private/BalanceEx",
		"/0/private/Balance",
		"/0/private/OpenPositions",
		"/0/private/TradeBalance",
		"/0/private/QueryOrders",
	]
end

function _downloadsrequest(method::AbstractString, url::AbstractString; headers=Pair{String, String}[], body::AbstractString="")::Dict
	attempts = 3
	method_upper = uppercase(String(method))
	while attempts > 0
		responseio = IOBuffer()
		try
			if method_upper == "GET"
				Downloads.request(String(url); method="GET", headers=headers, timeout=90.0, output=responseio)
			else
				Downloads.request(String(url); method=method_upper, headers=headers, timeout=90.0, input=IOBuffer(String(body)), output=responseio)
			end
			seekstart(responseio)
			return JSON3.read(String(take!(responseio)), Dict)
		catch err
			attempts -= 1
			msg = lowercase(sprint(showerror, err))
			istimeout = occursin("timed out", msg) || occursin("timeout", msg) || occursin("recv failure", msg)
			if istimeout && (attempts > 0)
				wait_s = 0.5 * (4 - attempts)
				(verbosity >= 2) && @warn "Downloads.request timeout; retrying fallback request" method=method_upper url attempts_left=attempts sleep_seconds=wait_s
				sleep(wait_s)
				continue
			end
			rethrow(err)
		end
	end
	error("unreachable")
end

"Return a strictly increasing Kraken nonce (UTC milliseconds with in-ms counter)."
function _nextnonce()::String
	lock(_nonce_lock)
	try
		current_ms = Int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1_000))
		if current_ms == _last_nonce_ms[]
			_nonce_ms_counter[] += 1
		else
			_last_nonce_ms[] = current_ms
			_nonce_ms_counter[] = 0
		end

		candidate = _last_nonce_ms[] * 1_000 + _nonce_ms_counter[]
		if candidate <= _last_nonce[]
			candidate = _last_nonce[] + 1
		end
		_last_nonce[] = candidate
		return string(candidate)
	finally
		unlock(_nonce_lock)
	end
end

"""
Convert a value to a numeric scalar, returning `default` when parsing fails.
"""
function _num(value, default::Real=0.0)
	if isnothing(value)
		return default
	elseif value isa Real
		return (value)
	elseif value isa AbstractString
		return value == "" ? default : try
			parse(typeof(default), value)
		catch
			default
		end
	elseif value isa AbstractVector
		return isempty(value) ? default : _num(first(value), default)
	end
	return default
end

"""
Read one numeric value from a vector-like field using 1-based `index`.
Falls back to `default` when the index is unavailable or parsing fails.
"""
function _numat(value, index::Integer, default::Real=0.0)
	if value isa AbstractVector
		ix = Int(index)
		return (1 <= ix <= length(value)) ? _num(value[ix], default) : default
	end
	return _num(value, default)
end

"Return the first present value for any of `keys` in `dict`, otherwise `nothing`."
function _firstpresent(dict::AbstractDict, keys::AbstractVector{<:AbstractString})
	for key in keys
		haskey(dict, key) && return dict[key]
	end
	return nothing
end

"Parse one numeric value strictly at an API boundary; throws on missing or malformed values."
function _numstrict(value, context::AbstractString)
	isnothing(value) && error("$(context) is missing")
	if value isa Real
		return value
	elseif value isa AbstractString
		s = strip(String(value))
		isempty(s) && error("$(context) must not be empty")
		try
			return parse(Float64, s)
		catch err
			error("$(context) must be numeric, got value=$(repr(value)) type=$(typeof(value)) parse_error=$(sprint(showerror, err))")
		end
	end
	error("$(context) must be Real or numeric string, got type=$(typeof(value)) value=$(repr(value))")
end

"Read one numeric vector element strictly using 1-based `index`."
function _numatstrict(value, index::Integer, context::AbstractString)
	value isa AbstractVector || error("$(context) must be a vector, got type=$(typeof(value))")
	ix = Int(index)
	(1 <= ix <= length(value)) || error("$(context) missing index $(ix), length=$(length(value))")
	return _numstrict(value[ix], context)
end

"Parse Kraken leverage values (e.g. \"2\", \"2:1\", \"none\") into a positive ratio or 0."
function _leveragevalue(value)
	if isnothing(value)
		return 0.0
	end
	if value isa Real
		v = (value)
		return v > 0.0 ? v : 0.0
	end
	if value isa AbstractString
		s = lowercase(strip(String(value)))
		(s == "") && return 0.0
		(s == "none") && return 0.0
		if occursin(":", s)
			parts = split(s, ":")
			if !isempty(parts)
				lhs = try
					parse(Float64, strip(parts[1]))
				catch
					0.0
				end
				return lhs > 0.0 ? lhs : 0.0
			end
		end
		parsed = try
			parse(Float64, s)
		catch
			0.0
		end
		return parsed > 0.0 ? parsed : 0.0
	end
	if value isa AbstractVector
		return isempty(value) ? 0.0 : _leveragevalue(first(value))
	end
	return 0.0
end

"""
Convert a value to `Int`, returning `default` when parsing fails.
"""
function _int(value, default::Int=0)::Int
	if isnothing(value)
		return default
	elseif value isa Integer
		return Int(value)
	elseif value isa Real
		return round(Int, value)
	elseif value isa AbstractString
		return value == "" ? default : try
			parse(Int, value)
		catch
			default
		end
	end
	return default
end

"Return `true` when instrument status allows trading requests."
function _istradablestatus(status)::Bool
	st = lowercase(String(status))
	return st in ["online", "post_only", "limit_only", "reduce_only", "trading"]
end

"Return decimal digits implied by a precision step size (e.g. 0.01 => 2)."
function _precisiondigits(step::Real, defaultdigits::Int=8)::Int
	step <= 0 && return defaultdigits
	d = round(Int, log10(1 / (step)))
	return max(d, 0)
end

"Return true when an exception text indicates a post-only rejection."
function _ispostonlyrejection(err)::Bool
	msg = lowercase(sprint(showerror, err))
	return occursin("post", msg) && occursin("only", msg)
end

"Return true when Kraken rejects an order because cl_ord_id is already used by an active order."
function _isclordidnotunique(err)::Bool
	msg = lowercase(sprint(showerror, err))
	return occursin("cl_ord_id not unique", msg)
end

function _invalidate_openorders_cache!()
	lock(_openorders_cache_lock) do
		_openorders_cache[] = nothing
		_openorders_cache_time[] = nothing
	end
	return nothing
end

function _stale_openorders_cache_copy(maxage::Dates.Period)::Union{Nothing, DataFrame}
	return lock(_openorders_cache_lock) do
		now = Dates.now(Dates.UTC)
		if !isnothing(_openorders_cache[]) && !isnothing(_openorders_cache_time[])
			if (now - _openorders_cache_time[]) < maxage
				return copy(_openorders_cache[])
			end
		end
		return nothing
	end
end

function _upsert_openorders_cache_row!(row)::Nothing
	lock(_openorders_cache_lock) do
		if isnothing(_openorders_cache[])
			return nothing
		end
		df = copy(_openorders_cache[])
		ix = findfirst(==(String(row.orderid)), String.(df[!, :orderid]))
		if isnothing(ix)
			push!(df, row)
		else
			for col in names(df)
				df[ix, col] = getproperty(row, Symbol(col))
			end
		end
		_openorders_cache[] = df
		_openorders_cache_time[] = Dates.now(Dates.UTC)
	end
	return nothing
end

function _base36_upper(n::Integer)::String
	n < 0 && throw(ArgumentError("base36 input must be >= 0, got $(n)"))
	digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	n == 0 && return "0"
	chars = Char[]
	x = Int(n)
	while x > 0
		r = (x % 36) + 1
		push!(chars, digits[r])
		x ÷= 36
	end
	reverse!(chars)
	return String(chars)
end

"Normalize limit order price/qty to exchange precision and min constraints."
function _normalizelimitorderparams(syminfo::DataFrameRow, basequantity::Real, limitprice::Real)
	pricedigits = _precisiondigits((syminfo.ticksize), 5)
	qtydigits = _precisiondigits((syminfo.baseprecision), 8)

	normprice = (round((limitprice), digits=pricedigits))
	normprice > 0f0 || throw(ArgumentError("normalized limitprice must be > 0, got $(normprice)"))

	normqty = (basequantity)
	minquote = (syminfo.minquoteqty)
	minbase = (syminfo.minbaseqty)
	if (minquote > 0.0) && ((normqty * (normprice)) < minquote)
		normqty = minquote / (normprice)
	end
	normqty = max(normqty, minbase)
	normqty = floor(normqty, digits=qtydigits)
	if normqty < minbase
		normqty = minbase
	end
	if (minquote > 0.0) && ((normqty * (normprice)) < minquote)
		normqty = ceil(minquote / (normprice), digits=qtydigits)
		normqty = max(normqty, minbase)
	end
	return (basequantity=(normqty), limitprice=normprice, qtydigits=qtydigits, pricedigits=pricedigits)
end

"Validate requested spot margin leverage range before submitting Kraken orders."
function _validatemarginleverage(marginleverage::Signed)
	if marginleverage == 0
		return nothing
	end
	if !(2 <= marginleverage <= 5)
		throw(ArgumentError("Kraken spot margin leverage must be 0 or in 2:5, got $(marginleverage)"))
	end
	return nothing
end

"Parse Kraken leverage levels into a sorted integer vector."
function _leveragelevels(raw)::Vector{Int}
	if !(raw isa AbstractVector)
		return Int[]
	end
	levels = Int[]
	for entry in raw
		lvl = _int(entry, 0)
		lvl > 0 && push!(levels, lvl)
	end
	return sort(unique(levels))
end

"Return highest permitted leverage from a Kraken leverage list payload."
_maxleverage(raw)::Int = isempty(_leveragelevels(raw)) ? 0 : maximum(_leveragelevels(raw))

"""
Build the DataFrame schema used by `exchangeinfo`.
"""
function _emptyexchangeinfo()::DataFrame
	return DataFrame(
		symbol=String[],
		status=String[],
		basecoin=String[],
		quotecoin=String[],
		maxleveragebuy=Int[],
		maxleveragesell=Int[],
		ticksize=Float32[],
		baseprecision=Float32[],
		quoteprecision=Float32[],
		minbaseqty=Float32[],
		minquoteqty=Float32[],
		krakenpairname=String[],
		wsname=String[],
	)
end

"""
Generate a unique client order id with the robot prefix.
"""
function _next_client_order_id()::String
	(sessiontoken, n) = lock(_order_counter_lock) do
		if isempty(_order_session_token[])
			seedmix = (time_ns(), Base.Libc.getpid(), objectid(_order_counter_lock))
			tok = uppercase(string(abs(hash(seedmix)); base=16))
			_order_session_token[] = lpad(tok[1:min(end, 5)], 5, '0')
		end
		_order_counter[] += 1
		(_order_session_token[], _order_counter[])
	end
	# Kraken Spot free-text cl_ord_id max length is 18 chars.
	# Prefix is 5 chars ("ROBO-") and we keep 13 chars for token+counter.
	counter = lpad(_base36_upper(n % 36^8), 8, '0')
	return string(ROBOT_ORDER_PREFIX, sessiontoken, counter)
end

"""
Build the DataFrame schema used by `openorders` and `order`.
"""
function emptyordersschema(::KrakenSpotCache)::DataFrame
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

emptyorders(bc::KrakenSpotCache)::DataFrame = emptyordersschema(bc)

"""
Normalize Kraken asset names to common base/quote symbols.
"""
function _normalizeasset(asset::AbstractString)::String
	amap = Dict(
		"XBT" => "BTC",
		"XXBT" => "BTC",
		"XDG" => "DOGE",
		"XXDG" => "DOGE",
	)
	key = uppercase(asset)
	if key in keys(amap)
		return amap[key]
	end
	if startswith(key, "X") || startswith(key, "Z")
		return length(key) > 3 ? key[2:end] : key
	end
	return key
end

"""
Convert websocket symbol format (`BTC/USDT`) to normalized symbol (`BTCUSDT`).
"""
_ws2symbol(wsname::AbstractString)::String = uppercase(replace(wsname, "/" => ""))

"""
Convert normalized symbol (`BTCUSDT`) to websocket symbol format (`BTC/USDT`).
"""
function _symbol2ws(symbol::AbstractString)::String
	clean = uppercase(replace(symbol, "/" => ""))
	sortedquotes = sort(_known_quotes, by=length, rev=true)
	for q in sortedquotes
		if endswith(clean, q) && (length(clean) > length(q))
			base = clean[1:end-length(q)]
			return string(base, "/", q)
		end
	end
	return clean
end

"""
Best-effort conversion from Kraken pair aliases into normalized symbols.
"""
function _normalizepairsymbol(pair::AbstractString)::String
	if occursin("/", pair)
		return _ws2symbol(pair)
	end
	up = uppercase(pair)
	sortedquotes = sort(_known_quotes, by=length, rev=true)
	for q in sortedquotes
		if endswith(up, q) && (length(up) > length(q))
			return up
		end
	end
	return up
end

"""
Resolve the normalized internal symbol for a `(basecoin, quotecoin)` pair.
"""
function symboltoken(bc::KrakenSpotCache, basecoin::AbstractString, quotecoin::AbstractString=EnvConfig.pairquote)::String
	base = _normalizeasset(basecoin)
	qtoken = uppercase(quotecoin)
	if !isnothing(bc.syminfodf) && (size(bc.syminfodf, 1) > 0)
		matchix = findfirst(row -> (uppercase(String(row.basecoin)) == base) && (uppercase(String(row.quotecoin)) == qtoken), eachrow(bc.syminfodf))
		if !isnothing(matchix)
			return uppercase(String(bc.syminfodf[matchix, :symbol]))
		end
	end
	return uppercase(base * qtoken)
end

"""
Resolve an internal symbol (`BTCUSDT`) to Kraken pair name used by REST calls.
"""
function _symbol2pairname(bc::KrakenSpotCache, symbol::AbstractString)::String
	sym = _normalizepairsymbol(symbol)
	if !isnothing(bc.syminfodf) && (size(bc.syminfodf, 1) > 0)
		ix = findfirst(==(sym), bc.syminfodf[!, :symbol])
		if !isnothing(ix)
			return bc.syminfodf[ix, :krakenpairname]
		end
	end
	return sym
end

"""
Resolve a Kraken REST result key to normalized symbol.
"""
function _resultkey2symbol(bc::KrakenSpotCache, key::AbstractString)::String
	if !isnothing(bc.syminfodf) && (size(bc.syminfodf, 1) > 0)
		matchix = findfirst(==(key), bc.syminfodf[!, :krakenpairname])
		if !isnothing(matchix)
			return bc.syminfodf[matchix, :symbol]
		end
		matchix = findfirst(==(key), bc.syminfodf[!, :wsname])
		if !isnothing(matchix)
			return bc.syminfodf[matchix, :symbol]
		end
	end
	return _normalizepairsymbol(key)
end

"""
REST helper for public Kraken API requests.
"""
function HttpPublicRequest(bc::KrakenSpotCache, method::AbstractString, endPoint::AbstractString, params::Union{Dict, Nothing}, info::AbstractString)
	url = bc.apirest * endPoint
	query = _dict2paramsget(params)
	if !isempty(query)
		url = string(url, "?", query)
	end
	body = try
		response = HTTP.request(method, url; retries=5, retry=true, readtimeout=60)
		JSON3.read(String(response.body), Dict)
	catch err
		if _httpmemorycompaterror(err)
			(verbosity >= 3) && @warn "HTTP.request compatibility fallback to Downloads.request" method url info
			_downloadsrequest(method, url)
		else
			rethrow(err)
		end
	end
	_checkresponse(body, info)
	return body
end

"""
REST helper for private Kraken API requests.
"""
function HttpPrivateRequest(bc::KrakenSpotCache, method::AbstractString, endPoint::AbstractString, params::Union{Dict, Nothing}, info::AbstractString)
	if !_hascredentials(bc)
		throw(ArgumentError("Kraken credentials are required for $(info)"))
	end
	method != "POST" && throw(ArgumentError("Kraken private endpoints require POST, got $(method)"))
	if _isreadonlyprivateendpoint(endPoint)
		lock(_private_rl_lock) do
			now = Dates.now(Dates.UTC)
			if !isnothing(_private_rl_cooldown_until[]) && (now < _private_rl_cooldown_until[])
				throw(ErrorException("Kraken private read cooldown active until $(_private_rl_cooldown_until[])"))
			end
		end
	end

	# Diagnostics: count private endpoint calls for end-of-run summary only.
	lock(_private_call_counter_lock) do
		_private_call_counter[endPoint] = get(_private_call_counter, endPoint, 0) + 1
	end

	baseparams = isnothing(params) ? Dict{String, Any}() : Dict{String, Any}(string(k) => v for (k, v) in params)
	attempts = 8
	last_error = nothing
	while attempts > 0
		reqparams = copy(baseparams)
		nonce = _nextnonce()
		reqparams["nonce"] = nonce
		postdata = _dict2paramspost(reqparams)
		signature = _krakensignature(endPoint, nonce, postdata, bc.secretkey)

		headers = Dict(
			"API-Key" => bc.publickey,
			"API-Sign" => signature,
			"Content-Type" => "application/x-www-form-urlencoded",
		)
		body = try
			response = HTTP.request("POST", bc.apirest * endPoint, headers, postdata; retries=0, retry_non_idempotent=false, readtimeout=60)
			JSON3.read(String(response.body), Dict)
		catch err
			if _httpmemorycompaterror(err)
				(verbosity >= 3) && @warn "HTTP.request compatibility fallback to Downloads.request" method="POST" endpoint=endPoint info
				hvec = Pair{String, String}[String(k) => String(v) for (k, v) in headers]
				_downloadsrequest("POST", bc.apirest * endPoint; headers=hvec, body=postdata)
			elseif _istransientnetworkerror(err) && (attempts > 1)
				attempts -= 1
				retry_ix = 8 - attempts
				wait_s = min(5.0, 0.25 * retry_ix)
				last_error = err
				(verbosity >= 1) && @warn "Kraken private transport error; retrying request" endpoint=endPoint attempts_left=attempts sleep_seconds=wait_s exception=sprint(showerror, err)
				sleep(wait_s)
				continue
			else
				rethrow(err)
			end
		end

		try
			_checkresponse(body, info)
			return body
		catch err
			last_error = err
			attempts -= 1
			if _isratelimiterror(err) && (attempts > 0)
				lock(_private_rl_lock) do
					now = Dates.now(Dates.UTC)
					until = now + PRIVATE_READ_COOLDOWN
					if isnothing(_private_rl_cooldown_until[]) || (until > _private_rl_cooldown_until[])
						_private_rl_cooldown_until[] = until
					end
				end
				if _isreadonlyprivateendpoint(endPoint)
					# Read-only paths should fall back to cache while cooldown is active.
					rethrow(err)
				end
				retry_ix = 8 - attempts
				wait_s = min(30.0, 2.0 ^ retry_ix)
				(verbosity >= 1) && @warn "Kraken private rate limit hit; retrying request" endpoint=endPoint attempts_left=attempts sleep_seconds=wait_s
				sleep(wait_s)
				continue
			end
			if _isinvalidnonceerror(err) && (attempts > 0)
				retry_ix = 8 - attempts
				wait_s = min(0.5, 0.05 * retry_ix)
				nonce_floor = let
					lock(_nonce_lock)
					try
						now_ms = Int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1_000))
						if now_ms <= _last_nonce_ms[]
							_last_nonce_ms[] += 1
						else
							_last_nonce_ms[] = now_ms
						end
						_nonce_ms_counter[] = 0
						candidate = _last_nonce_ms[] * 1_000
						if candidate <= _last_nonce[]
							candidate = _last_nonce[] + 1
						end
						_last_nonce[] = candidate
						_last_nonce[]
					finally
						unlock(_nonce_lock)
					end
				end
				(verbosity >= 2) && @warn "Kraken invalid nonce; retrying private request" endpoint=endPoint attempts_left=attempts sleep_seconds=wait_s nonce_floor=nonce_floor
				sleep(wait_s)
				continue
			end
			rethrow(err)
		end
	end
	throw(last_error)
end

"""
Read Kraken asset pair metadata and map it into a Bybit-compatible schema.
"""
function _exchangeinfo(apirest::String, symbol=nothing)::DataFrame
	tmp = KrakenSpotCache(_emptyexchangeinfo(), apirest, "", "")
	response = HttpPublicRequest(tmp, "GET", "/0/public/AssetPairs", nothing, "asset pairs")
	result = get(response, "result", Dict{String, Any}())
	df = _emptyexchangeinfo()

	for (pairname, rawinfo) in result
		pairname == "last" && continue
		info = Dict(rawinfo)
		wsname = String(get(info, "wsname", pairname))
		symbolname = _ws2symbol(wsname)

		basecoin = ""
		quotecoin = ""
		if occursin("/", wsname)
			splitpair = split(wsname, "/")
			if length(splitpair) == 2
				basecoin = _normalizeasset(String(splitpair[1]))
				quotecoin = _normalizeasset(String(splitpair[2]))
			end
		end
		if (basecoin == "") || (quotecoin == "")
			basecoin = _normalizeasset(String(get(info, "base", "")))
			quotecoin = _normalizeasset(String(get(info, "quote", "")))
		end

		pairdecimals = _int(get(info, "pair_decimals", 5), 5)
		lotdecimals = _int(get(info, "lot_decimals", 8), 8)
		ticksize = (10.0^-pairdecimals)
		baseprecision = (10.0^-lotdecimals)
		minbaseqty = _num(get(info, "ordermin", "0"), 0f0)
		minquoteqty = _num(get(info, "costmin", "0"), 0f0)
		status = String(get(info, "status", "online"))
		maxleveragebuy = _maxleverage(get(info, "leverage_buy", Any[]))
		maxleveragesell = _maxleverage(get(info, "leverage_sell", Any[]))

		push!(df, (
			symbol=symbolname,
			status=status,
			basecoin=basecoin,
			quotecoin=quotecoin,
			maxleveragebuy=maxleveragebuy,
			maxleveragesell=maxleveragesell,
			ticksize=ticksize,
			baseprecision=baseprecision,
			quoteprecision=ticksize,
			minbaseqty=minbaseqty,
			minquoteqty=minquoteqty,
			krakenpairname=String(pairname),
			wsname=wsname,
		))
	end

	if !isnothing(symbol)
		symbol = _normalizepairsymbol(String(symbol))
		return df[df.symbol .== symbol, :]
	end
	return sort!(df, :symbol)
end

"""
Return cached symbol metadata. If `symbol` is given, returns a filtered DataFrame.
"""
exchangeinfo(bc::KrakenSpotCache, symbol=nothing) = isnothing(symbol) ? bc.syminfodf : bc.syminfodf[bc.syminfodf.symbol .== _normalizepairsymbol(String(symbol)), :]

"""
Return one symbol information row or `nothing` when symbol is unknown.
"""
function symbolinfo(bc::KrakenSpotCache, symbol::AbstractString)::Union{Nothing, DataFrameRow}
	sym = _normalizepairsymbol(symbol)
	if isnothing(bc.syminfodf) || (size(bc.syminfodf, 1) == 0)
		return nothing
	end
	ix = findfirst(==(sym), bc.syminfodf[!, :symbol])
	return isnothing(ix) ? nothing : bc.syminfodf[ix, :]
end

"Return side-specific Kraken spot margin leverage caps for a symbol."
function marginlimits(bc::KrakenSpotCache, symbol::AbstractString)
	syminfo = symbolinfo(bc, symbol)
	if isnothing(syminfo)
		return (maxleveragebuy=0, maxleveragesell=0)
	end
	return (
		maxleveragebuy=hasproperty(syminfo, :maxleveragebuy) ? Int(syminfo.maxleveragebuy) : 0,
		maxleveragesell=hasproperty(syminfo, :maxleveragesell) ? Int(syminfo.maxleveragesell) : 0,
	)
end

"Return true when Kraken spot metadata permits the requested side/leverage for this symbol."
function marginpermitted(bc::KrakenSpotCache, symbol::AbstractString, orderside::AbstractString, marginleverage::Signed)::Bool
	marginleverage <= 0 && return true
	limits = marginlimits(bc, symbol)
	side = lowercase(String(orderside))
	maxlev = side == "buy" ? limits.maxleveragebuy : limits.maxleveragesell
	return maxlev >= Int(marginleverage)
end

symbolinfo(bc::KrakenSpotCache, basecoin::AbstractString, quotecoin::AbstractString) = symbolinfo(bc, symboltoken(bc, basecoin, quotecoin))
function validsymbol(bc::KrakenSpotCache, basecoin::AbstractString, quotecoin::AbstractString)::Bool
	sym = symbolinfo(bc, basecoin, quotecoin)
	return !isnothing(sym) && (uppercase(String(sym.quotecoin)) == uppercase(quotecoin)) && _istradablestatus(sym.status)
end

"""
Validate one symbol info row according to current quote and trading status constraints.
"""
function validsymbol(bc::KrakenSpotCache, sym::Union{Nothing, DataFrameRow})::Bool
	if isnothing(sym)
		return false
	end
	return uppercase(sym.quotecoin) == uppercase(EnvConfig.pairquote) && _istradablestatus(sym.status)
end

"""
Validate one symbol string according to current quote and trading status constraints.
"""
validsymbol(bc::KrakenSpotCache, symbol::AbstractString)::Bool = validsymbol(bc, symbolinfo(bc, symbol))

"""
Return Kraken server time in UTC.
"""
function servertime(bc::KrakenSpotCache)::DateTime
	while true
		try
			response = HttpPublicRequest(bc, "GET", "/0/public/Time", nothing, "server time")
			unixtime = _int(response["result"]["unixtime"])
			return Dates.unix2datetime(unixtime)
		catch err
			(verbosity >= 1) && @warn "KrakenSpot server time unavailable; retrying" retry_seconds=SERVERTIME_RETRY_SECONDS exception=sprint(showerror, err)
			sleep(SERVERTIME_RETRY_SECONDS)
		end
	end
end

"""
Parse one Kraken ticker object into Bybit-compatible ticker fields.
"""
function _tickerrow(bc::KrakenSpotCache, key::AbstractString, ticker::Dict)
	ask = _numatstrict(get(ticker, "a", nothing), 1, "ticker $(key) ask")
	bid = _numatstrict(get(ticker, "b", nothing), 1, "ticker $(key) bid")
	lastprice = _numatstrict(get(ticker, "c", nothing), 1, "ticker $(key) lastprice")
	openprice = _numstrict(get(ticker, "o", nothing), "ticker $(key) openprice")
	# Kraken ticker volume array `v` is [today, last_24h]; selection uses rolling 24h.
	basevolume = _numatstrict(get(ticker, "v", nothing), 2, "ticker $(key) basevolume24h")
	quotevolume = basevolume * lastprice
	pricechangepercent = openprice == 0f0 ? 0f0 : ((lastprice - openprice) / openprice)
	symbol = _resultkey2symbol(bc, key)
	return (askprice=ask, bidprice=bid, lastprice=lastprice, quotevolume24h=quotevolume, pricechangepercent=pricechangepercent, symbol=symbol)
end

"""
Return spot ticker information in a Bybit-compatible shape.

- without `symbol`: one row per tradable pair
- with `symbol`: returns one `DataFrameRow` when available
"""
function get24h(bc::KrakenSpotCache, symbol=nothing)
	pairsdf = isnothing(symbol) ? exchangeinfo(bc) : exchangeinfo(bc, symbol)
	out = DataFrame(askprice=Float32[], bidprice=Float32[], lastprice=Float32[], quotevolume24h=Float32[], pricechangepercent=Float32[], symbol=String[])
	if isnothing(pairsdf) || (size(pairsdf, 1) == 0)
		return isnothing(symbol) ? out : nothing
	end

	pairs = pairsdf[!, :krakenpairname]
	batchesize = 20
	for ix in 1:batchesize:length(pairs)
		lastix = min(ix + batchesize - 1, length(pairs))
		pairbatch = pairs[ix:lastix]
		params = Dict("pair" => join(pairbatch, ","))
		response = HttpPublicRequest(bc, "GET", "/0/public/Ticker", params, "ticker 24h")
		result = get(response, "result", Dict{String, Any}())
		for (key, rawticker) in result
			ticker = Dict(rawticker)
			push!(out, _tickerrow(bc, key, ticker))
		end
	end

	if isnothing(symbol)
		return out
	end
	return size(out, 1) > 0 ? out[1, :] : nothing
end

"""
Convert Kraken OHLC response rows into an Ohlcv-compatible DataFrame.
"""
function _convertklines(klines)::DataFrame
	df = DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])
	for row in klines
		length(row) < 7 && continue
		opentime = Dates.unix2datetime(_int(row[1]))
		push!(df, (
			opentime=opentime,
			open=_num(row[2]),
			high=_num(row[3]),
			low=_num(row[4]),
			close=_num(row[5]),
			basevolume=_num(row[7]),
		))
	end
	return sort!(df, :opentime)
end

"""
Get OHLC candles in an Ohlcv-compatible format (oldest first rows).
"""
function getklines(bc::KrakenSpotCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m")
	@assert interval in keys(_interval2minutes) "unknown Kraken interval=$(interval)"
	pairname = _symbol2pairname(bc, String(symbol))
	params = Dict("pair" => pairname, "interval" => _interval2minutes[interval])
	if !isnothing(startDateTime)
		params["since"] = _int(Dates.datetime2unix(startDateTime))
	end

	response = HttpPublicRequest(bc, "GET", "/0/public/OHLC", params, "kline")
	result = get(response, "result", Dict{String, Any}())
	key = nothing
	for k in keys(result)
		if k != "last"
			key = k
			break
		end
	end
	if isnothing(key)
		return DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])
	end
	df = _convertklines(result[key])
	if !isnothing(endDateTime)
		df = df[df.opentime .<= endDateTime, :]
	end
	return df
end

"""
Return account overview from Kraken private API.
"""
function account(bc::KrakenSpotCache)
	if !_hascredentials(bc)
		return Dict{String, Any}()
	end
	response = HttpPrivateRequest(bc, "POST", "/0/private/TradeBalance", Dict("asset" => "ZUSD"), "trade balance")
	return get(response, "result", Dict{String, Any}())
end

"Case-insensitive dictionary lookup returning the first matching key value."
function _dictgetci(d::AbstractDict, keys::AbstractVector{<:AbstractString}, default=nothing)
	for wanted in keys
		wl = lowercase(String(wanted))
		for (k, v) in d
			if lowercase(String(k)) == wl
				return v
			end
		end
	end
	return default
end

"Parse one account metric from a TradeBalance-like dictionary."
function _accountfloat(d::AbstractDict, keys::AbstractVector{<:AbstractString}, default::Float64=0.0)::Float64
	return (_num(_dictgetci(d, keys, default), (default)))
end

"Return exchange-concept account capacity metrics for Kraken Spot mixed spot+margin accounts."
function accountcapacity(bc::KrakenSpotCache)
	if !_hascredentials(bc)
		return (
			equity_quote=0.0,
			available_opening_quote=0.0,
			available_long_quote=0.0,
			available_short_quote=0.0,
			initial_margin_quote=0.0,
			maintenance_margin_quote=0.0,
			source="KrakenSpot:no_credentials",
		)
	end

	tradebalance_raw = account(bc)
	tradebalance = tradebalance_raw isa AbstractDict ? Dict(tradebalance_raw) : Dict{Any, Any}()

	# Kraken TradeBalance fields are exchange-native account metrics.
	equity_quote = _accountfloat(tradebalance, ["eb", "equity", "e", "tb"], 0.0)
	available_short_quote = _accountfloat(tradebalance, ["mf", "free_margin", "freemargin", "availablemargin"], 0.0)
	initial_margin_quote = _accountfloat(tradebalance, ["m", "initialmargin"], 0.0)
	maintenance_margin_quote = _accountfloat(tradebalance, ["maintenance_margin", "maintenancemargin"], 0.0)

	# Long lane capacity is wallet quote available for spot buys.
	available_long_quote = 0.0
	quotecoin = uppercase(String(EnvConfig.pairquote))
	try
		bdf = balances(bc)
		if (:coin in names(bdf)) && (:free in names(bdf))
			for row in eachrow(bdf)
				if uppercase(String(row.coin)) == quotecoin
					available_long_quote += max(0.0, (row.free))
				end
			end
		end
	catch err
		(verbosity >= 1) && @warn "KrakenSpot accountcapacity: quote free balance lookup failed" error=sprint(showerror, err)
	end

	if available_long_quote <= 0.0
		available_long_quote = _accountfloat(tradebalance, ["tb", "trade_balance", "cash"], 0.0)
	end

	# For mixed spot+margin accounts, opening capacity must respect both lanes.
	available_opening_quote = if (available_long_quote > 0.0) && (available_short_quote > 0.0)
		min(available_long_quote, available_short_quote)
	else
		max(available_long_quote, available_short_quote)
	end

	if equity_quote <= 0.0
		equity_quote = max(available_long_quote, _accountfloat(tradebalance, ["tb"], 0.0))
	end

	return (
		equity_quote=max(0.0, equity_quote),
		available_opening_quote=max(0.0, available_opening_quote),
		available_long_quote=max(0.0, available_long_quote),
		available_short_quote=max(0.0, available_short_quote),
		initial_margin_quote=max(0.0, initial_margin_quote),
		maintenance_margin_quote=max(0.0, maintenance_margin_quote),
		source="KrakenSpot:TradeBalance+BalanceEx",
	)
end

"""
Return explicit per-base position quantities.

`short_qty` is sourced from Kraken OpenPositions (margin shorts).
`long_qty` is currently left as `0` so callers can continue deriving spot holdings
from balance/asset snapshots.
"""
function positionsnapshot(bc::KrakenSpotCache)::DataFrame
	df = DataFrame(coin=String[], long_qty=Float32[], short_qty=Float32[])
	if !_hascredentials(bc)
		return df
	end
	borrowed = _borrowedfromopenpositions(bc)
	for (coin, qty) in borrowed
		sqty = max(0f0, (qty))
		sqty <= 0f0 && continue
		push!(df, (coin=uppercase(String(coin)), long_qty=0f0, short_qty=sqty))
	end
	return df
end

"""
Convert one raw Kraken order entry into the standardized order row shape.
"""
function _orderrow(bc::KrakenSpotCache, orderid::AbstractString, entry::Dict)
	descr = haskey(entry, "descr") ? Dict(entry["descr"]) : Dict{String, Any}()
	rawpair = String(get(descr, "pair", ""))
	symbol = _resultkey2symbol(bc, rawpair)
	side = lowercase(String(get(descr, "type", "buy"))) == "buy" ? "Buy" : "Sell"
	ordertype = titlecase(String(get(descr, "ordertype", "limit")))
	baseqty = _numstrict(get(entry, "vol", nothing), "order $(orderid) vol")
	executedqty = _numstrict(get(entry, "vol_exec", nothing), "order $(orderid) vol_exec")
	limitprice = _numstrict(_firstpresent(descr, ["price"]), "order $(orderid) limitprice")
	avgprice = _numstrict(_firstpresent(entry, ["price"]), "order $(orderid) avgprice")
	status = titlecase(String(get(entry, "status", "open")))
	created = Dates.unix2datetime(_int(round(_numstrict(get(entry, "opentm", nothing), "order $(orderid) opentm"))))
	updated = created
	leverage = _leveragevalue(get(descr, "leverage", "0"))
	orderLinkId = String(get(entry, "cl_ord_id", ""))
	rawreduceonly = get(entry, "reduce_only", get(descr, "reduce_only", get(entry, "reduceOnly", get(descr, "reduceOnly", false))))
	reduceonly = rawreduceonly isa Bool ? rawreduceonly : lowercase(String(rawreduceonly)) in ["true", "1", "yes"]
	return (
		orderid=String(orderid),
		orderLinkId=orderLinkId,
		symbol=symbol,
		side=side,
		baseqty=baseqty,
		ordertype=ordertype,
		isLeverage=leverage > 0f0,
		timeinforce="GTC",
		limitprice=limitprice,
		avgprice=avgprice,
		executedqty=executedqty,
		status=status,
		created=created,
		updated=updated,
		rejectreason="",
		reduceonly=reduceonly,
		lastcheck=Dates.now(Dates.UTC),
	)
end

"""
Return open spot orders in a DataFrame compatible with Bybit order columns.
"""
function openorders(bc::KrakenSpotCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)
	if !_hascredentials(bc)
		return emptyordersschema(bc)
	end

	# Read-through cache to reduce pressure on private OpenOrders endpoint.
	# The cache holds ALL robot open orders; filter by orderLinkId here if needed.
	if isnothing(orderid) && isnothing(symbol)
		cached = _stale_openorders_cache_copy(OPENORDERS_CACHE_TTL)
		if !isnothing(cached)
			if !isnothing(orderLinkId)
				lnkspec = String(orderLinkId)
				return cached[cached.orderLinkId .== lnkspec, :]
			end
			return cached
		end
	end

	params = Dict{String, Any}()
	!isnothing(orderid) && (params["txid"] = String(orderid))
	response = nothing
	try
		response = HttpPrivateRequest(bc, "POST", "/0/private/OpenOrders", params, "open orders")
	catch err
		if isnothing(orderid) && isnothing(symbol)
			fallback = _stale_openorders_cache_copy(OPENORDERS_CACHE_MAX_STALE)
			if !isnothing(fallback)
				age = lock(_openorders_cache_lock) do
					now = Dates.now(Dates.UTC)
					isnothing(_openorders_cache_time[]) ? Dates.Millisecond(0) : (now - _openorders_cache_time[])
				end
				(verbosity >= 1) && @warn "OpenOrders request failed; returning stale cached open orders" age exception=sprint(showerror, err)
				return fallback
			end
		end
		rethrow(err)
	end
	result = get(response, "result", Dict{String, Any}())
	open = haskey(result, "open") ? Dict(result["open"]) : Dict{String, Any}()

	out = emptyordersschema(bc)
	symbolspec = isnothing(symbol) ? nothing : _normalizepairsymbol(String(symbol))
	orderlinkspec = isnothing(orderLinkId) ? nothing : String(orderLinkId)
	for (oid, rawentry) in open
		entry = Dict(rawentry)
		row = _orderrow(bc, oid, entry)
		if !isnothing(symbolspec) && (row.symbol != symbolspec)
			continue
		end
		if !isnothing(orderlinkspec)
			row.orderLinkId != orderlinkspec && continue
		else
			!startswith(row.orderLinkId, ROBOT_ORDER_PREFIX) && continue
		end
		push!(out, row)
	end

	if isnothing(orderid) && isnothing(symbol)
		lock(_openorders_cache_lock) do
			_openorders_cache[] = copy(out)
			_openorders_cache_time[] = Dates.now(Dates.UTC)
		end
	end
	return out
end

"""
Query one order by id from open orders and then from historical orders if required.
"""
function order(bc::KrakenSpotCache, orderid)
	if isnothing(orderid)
		return nothing
	end
	oo = openorders(bc, orderid=orderid)
	if size(oo, 1) > 0
		return oo[1, :]
	end
	if !_hascredentials(bc)
		return nothing
	end

	response = HttpPrivateRequest(bc, "POST", "/0/private/QueryOrders", Dict("txid" => String(orderid), "trades" => true), "query order")
	result = get(response, "result", Dict{String, Any}())
	if !haskey(result, String(orderid))
		return nothing
	end
	row = _orderrow(bc, String(orderid), Dict(result[String(orderid)]))
	df = emptyordersschema(bc)
	push!(df, row)
	return df[1, :]
end

"""
Cancel one open order and return the cancelled order id on success.
"""
function cancelorder(bc::KrakenSpotCache, symbol, orderid)
	_ = symbol
	if isnothing(orderid) || !_hascredentials(bc)
		return nothing
	end
	response = HttpPrivateRequest(bc, "POST", "/0/private/CancelOrder", Dict("txid" => String(orderid)), "cancel order")
	count = _int(get(get(response, "result", Dict{String, Any}()), "count", 0), 0)
	if count > 0
		_invalidate_openorders_cache!()
		return String(orderid)
	end
	return nothing
end

"Return a post-only limit price one tick inside the spread for maker orders with omitted price."
function _makerlimitprice(syminfo::DataFrameRow, snapshot, orderside::AbstractString)
	ticksize = (syminfo.ticksize)
	side = lowercase(String(orderside))
	price = side == "buy" ? (snapshot.askprice) - ticksize : (snapshot.bidprice) + ticksize
	return price > 0f0 ? price : nothing
end

function _icebergdisplayqty(syminfo::DataFrameRow, totalqty::Real, limitprice::Real, max_quote)::Float32
	qtydigits = _precisiondigits((syminfo.baseprecision), 0)
	targetqty = isnothing(max_quote) ? (totalqty) : (max_quote) / (limitprice)
	minimum_display = max((syminfo.minbaseqty), (totalqty) / 15.0f0)
	displayqty = max(targetqty, minimum_display)
	displayqty = floor(displayqty, digits=qtydigits)
	if displayqty < (syminfo.minbaseqty)
		displayqty = (syminfo.minbaseqty)
	end
	return (min(displayqty, (totalqty)))
end

function _usenativeiceberg(ordertype::AbstractString, totalqty::Real, limitprice, max_quote)::Bool
	if lowercase(String(ordertype)) != "limit"
		return false
	end
	if isnothing(limitprice) || isnothing(max_quote)
		return false
	end
	return ((totalqty) * (limitprice)) > (max_quote) + 1e-9
end

"Build Kraken Spot AddOrder parameters, optionally in validate-only mode."
function _addorderparams(pairname::AbstractString, orderside::AbstractString, ordertype::AbstractString, chosenqty::Real, clientOrderId::AbstractString; effectiveprice::Union{Nothing, Real}=nothing, iceberg_displayqty=nothing, maker::Bool=false, effective_marginleverage::Signed=0, reduceonly::Bool=false, validate::Bool=false)
	params = Dict{String, Any}(
		"pair" => String(pairname),
		"type" => lowercase(String(orderside)),
		"ordertype" => isnothing(iceberg_displayqty) ? String(ordertype) : "iceberg",
		"volume" => string(chosenqty),
		"cl_ord_id" => String(clientOrderId),
	)
	if lowercase(String(ordertype)) == "limit"
		!isnothing(effectiveprice) && (params["price"] = string(effectiveprice))
	end
	!isnothing(iceberg_displayqty) && (params["displayvol"] = string(iceberg_displayqty))
	(maker && (lowercase(String(ordertype)) == "limit")) && (params["oflags"] = "post")
	(effective_marginleverage > 0) && (params["leverage"] = string(effective_marginleverage))
	(reduceonly && (effective_marginleverage > 0)) && (params["reduce_only"] = true)
	validate && (params["validate"] = true)
	return params
end

"""
Create one spot order and return an order row compatible named tuple.

If `price` is omitted and `maker=true`, the adapter will choose a limit price
as close as possible to the current spread while remaining post-only so the
order can qualify for maker fees.

Set `validate=true` to ask Kraken Spot to validate order parameters without
executing the order.
"""
function createorder(bc::KrakenSpotCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; configside::Union{Nothing, Symbol}=nothing, execution_spec=nothing, reduceonly::Bool=false, validate::Bool=false)
	@assert basequantity > 0.0 "createorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
	@assert isnothing(price) || (price > 0.0) "createorder symbol=$(symbol) price=$(price) must be > 0"
	@assert lowercase(orderside) in ["buy", "sell"] "createorder symbol=$(symbol) orderside=$(orderside) must be Buy or Sell"
	if !_hascredentials(bc)
		return nothing
	end
	spec = isnothing(execution_spec) ? _executionorderspec(configside, orderside) : execution_spec
	effective_marginleverage = spec.instrument == "margin" ? spec.leverage : 0
	if spec.instrument == "margin"
		_validatemarginleverage(effective_marginleverage)
	elseif spec.instrument != "spot"
		error("unsupported KrakenSpot execution instrument $(spec.instrument) for symbol=$(symbol) configside=$(spec.side)")
	end

	syminfo = symbolinfo(bc, symbol)
	if isnothing(syminfo)
		(verbosity >= 1) && @warn "no instrument info for $(symbol)"
		return nothing
	end
	pairname = _symbol2pairname(bc, symbol)
	if !_istradablestatus(syminfo.status)
		(verbosity >= 1) && @warn "symbol $(symbol) is not tradable due to status=$(syminfo.status)"
		return nothing
	end
	if effective_marginleverage > 0
		limits = marginlimits(bc, symbol)
		if !marginpermitted(bc, symbol, orderside, effective_marginleverage)
			throw(ErrorException("Kraken spot margin not permitted for symbol=$(symbol) pair=$(pairname) side=$(orderside) requested_leverage=$(effective_marginleverage)x max_buy=$(limits.maxleveragebuy)x max_sell=$(limits.maxleveragesell)x status=$(syminfo.status)"))
		end
	end
	adaptivepost = maker && isnothing(price)
	attempts = adaptivepost ? 5 : 1
	ordertype = (maker || !isnothing(price)) ? "limit" : "market"
	chosenqty = (basequantity)
	effectiveprice = isnothing(price) ? nothing : (price)
	iceberg_displayqty = nothing
	clientOrderId = _next_client_order_id()
	txids = Any[]
	while attempts > 0
		if ordertype == "limit"
			if adaptivepost
				snapshot = get24h(bc, symbol)
				effectiveprice = isnothing(snapshot) ? nothing : _makerlimitprice(syminfo, snapshot, orderside)
			end
			if isnothing(effectiveprice)
				(verbosity >= 1) && @warn "failed to resolve limit price for $(symbol)"
				return nothing
			end
			norm = _normalizelimitorderparams(syminfo, chosenqty, effectiveprice)
			chosenqty = norm.basequantity
			effectiveprice = norm.limitprice
			if _usenativeiceberg(ordertype, chosenqty, effectiveprice, spec.max_quote)
				iceberg_displayqty = _icebergdisplayqty(syminfo, chosenqty, effectiveprice, spec.max_quote)
			else
				iceberg_displayqty = nothing
			end
		end

		params = _addorderparams(pairname, orderside, ordertype, chosenqty, clientOrderId;
			effectiveprice=effectiveprice,
			iceberg_displayqty=iceberg_displayqty,
			maker=maker,
			effective_marginleverage=effective_marginleverage,
			reduceonly=reduceonly,
			validate=validate,
		)

		try
			response = HttpPrivateRequest(bc, "POST", "/0/private/AddOrder", params, "create order")
			result = get(response, "result", Dict{String, Any}())
			txids = get(result, "txid", Any[])
			if (txids isa AbstractVector) && !isempty(txids)
				break
			end
			return nothing
		catch err
			# If Kraken already processed a previous HTTP attempt that timed out before
			# the response arrived, the internal retry sends the same cl_ord_id and
			# Kraken rejects it as "not unique". Recover by looking up the placed order.
			if _isclordidnotunique(err)
				(verbosity >= 1) && @warn "cl_ord_id not unique for $(symbol); recovering existing open order" cl_ord_id=clientOrderId
				existing = openorders(bc; orderLinkId=clientOrderId)
				if size(existing, 1) > 0
					row = existing[1, :]
					_upsert_openorders_cache_row!(row)
					return (
						orderid=row.orderid,
						orderLinkId=row.orderLinkId,
						symbol=row.symbol,
						side=row.side,
						baseqty=row.baseqty,
						ordertype=row.ordertype,
						timeinforce=row.timeinforce,
						limitprice=row.limitprice,
						avgprice=row.avgprice,
						executedqty=row.executedqty,
						status=row.status,
						created=row.created,
						updated=row.updated,
						rejectreason=row.rejectreason,
					)
				end
				return nothing
			end
			attempts -= 1
			if !adaptivepost || !_ispostonlyrejection(err) || (attempts <= 0)
				rethrow(err)
			end
			(verbosity >= 2) && @info "retrying post-only order for $(symbol) after rejection" attempts_left=attempts
		end
	end

	if !(txids isa AbstractVector) || isempty(txids)
		return nothing
	end

	orderid = String(first(txids))
	_invalidate_openorders_cache!()
	created = Dates.now(Dates.UTC)
	return (
		orderid=orderid,
		orderLinkId=clientOrderId,
		symbol=_normalizepairsymbol(symbol),
		side=lowercase(orderside) == "buy" ? "Buy" : "Sell",
		baseqty=chosenqty,
		ordertype=isnothing(iceberg_displayqty) ? titlecase(ordertype) : "Iceberg",
			timeinforce=(maker && !isnothing(effectiveprice)) ? "PostOnly" : "GTC",
			limitprice=isnothing(effectiveprice) ? 0f0 : (effectiveprice),
		avgprice=0f0,
		executedqty=0f0,
		status="New",
		created=created,
		updated=created,
		rejectreason="",
	)
end

"""
Create one close order for an existing position side.

- `positionside=:long` maps to a Sell close.
- `positionside=:short` maps to a Buy close.
"""
function closeorder(bc::KrakenSpotCache, symbol::String, positionside::Symbol, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; execution_spec=nothing, reduceonly::Bool=true, validate::Bool=false)
	side = Symbol(lowercase(String(positionside)))
	@assert side in [:long, :short] "closeorder positionside=$(positionside) must be :long or :short"
	orderside = side == :long ? "Sell" : "Buy"
	return createorder(bc, symbol, orderside, basequantity, price, maker; configside=side, execution_spec=execution_spec, reduceonly=reduceonly, validate=validate)
end

_isopenstatus(status::AbstractString)::Bool = lowercase(strip(String(status))) in ("new", "partiallyfilled", "untriggered", "open")

"Upsert one close leg independent from any open leg handling."
function upsertcloseorder!(bc::KrakenSpotCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=true)
	existing = nothing
	if !isnothing(existing_orderid)
		probe = order(bc, String(existing_orderid))
		if !isnothing(probe) && hasproperty(probe, :status) && _isopenstatus(String(probe.status))
			existing = probe
		end
	end
	if isnothing(existing)
		return closeorder(bc, symbol, positionside, basequantity, limitprice, maker; reduceonly=reduceonly, validate=false)
	end

	remaining = max(0.0, (existing.baseqty) - (existing.executedqty))
	currentlimit = hasproperty(existing, :limitprice) ? existing.limitprice : nothing
	qtychanged = remaining != basequantity
	limitchanged = (isnothing(currentlimit) && !isnothing(limitprice)) || (!isnothing(currentlimit) && isnothing(limitprice)) || (!isnothing(currentlimit) && !isnothing(limitprice) && (currentlimit != limitprice))
	if qtychanged || limitchanged
		return amendorder(bc, String(existing.symbol), String(existing.orderid); basequantity=basequantity, limitprice=limitprice)
	end
	return String(existing.orderid)
end

"Upsert one open leg independent from any close leg handling."
function upsertopenorder!(bc::KrakenSpotCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=false)
	side = Symbol(lowercase(String(positionside)))
	@assert side in [:long, :short] "upsertopenorder! positionside=$(positionside) must be :long or :short"
	orderside = side == :long ? "Buy" : "Sell"
	existing = nothing
	if !isnothing(existing_orderid)
		probe = order(bc, String(existing_orderid))
		if !isnothing(probe) && hasproperty(probe, :status) && _isopenstatus(String(probe.status))
			existing = probe
		end
	end
	if isnothing(existing)
		return createorder(bc, symbol, orderside, basequantity, limitprice, maker; configside=side, reduceonly=reduceonly, validate=false)
	end

	remaining = max(0.0, (existing.baseqty) - (existing.executedqty))
	currentlimit = hasproperty(existing, :limitprice) ? existing.limitprice : nothing
	qtychanged = remaining != basequantity
	limitchanged = (isnothing(currentlimit) && !isnothing(limitprice)) || (!isnothing(currentlimit) && isnothing(limitprice)) || (!isnothing(currentlimit) && !isnothing(limitprice) && (currentlimit != limitprice))
	if qtychanged || limitchanged
		return amendorder(bc, String(existing.symbol), String(existing.orderid); basequantity=basequantity, limitprice=limitprice)
	end
	return String(existing.orderid)
end

"Register direct predecessor/successor sequencing at adapter layer."
function directsequence!(bc::KrakenSpotCache, predecessor_orderid::AbstractString, successor_orderid::AbstractString)
	predecessor = order(bc, String(predecessor_orderid))
	successor = order(bc, String(successor_orderid))
	@assert !isnothing(predecessor) "directsequence! predecessor order missing predecessor_orderid=$(predecessor_orderid)"
	@assert !isnothing(successor) "directsequence! successor order missing successor_orderid=$(successor_orderid)"
	@assert String(predecessor.symbol) == String(successor.symbol) "directsequence! symbol mismatch predecessor_symbol=$(String(predecessor.symbol)) successor_symbol=$(String(successor.symbol)) predecessor_orderid=$(predecessor_orderid) successor_orderid=$(successor_orderid)"
	return (predecessor_orderid=String(predecessor_orderid), successor_orderid=String(successor_orderid), symbol=String(predecessor.symbol), acknowledged=true)
end

"""
Amend one open order.

For post-only maker orders, this path prefers native Kraken `EditOrder` and
intentionally avoids cancel/recreate fallback to reduce close-order churn.
"""
function amendorder(bc::KrakenSpotCache, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
	current = order(bc, orderid)
	isnothing(current) && return nothing
	return amendorder(bc, String(current.symbol), orderid; basequantity=basequantity, limitprice=limitprice)
end

function amendorder(bc::KrakenSpotCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
	@assert isnothing(basequantity) || (basequantity > 0.0) "amendorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
	@assert isnothing(limitprice) || (limitprice > 0.0) "amendorder symbol=$(symbol) limitprice=$(limitprice) must be > 0"

	current = order(bc, orderid)
	if isnothing(current)
		return nothing
	end

	qty = isnothing(basequantity) ? current.baseqty : (basequantity)
	syminfo = symbolinfo(bc, symbol)
	isnothing(syminfo) && return nothing
	maker = current.timeinforce == "PostOnly"
	price = current.limitprice
	pricechanged = false
	if maker
		snapshot = get24h(bc, symbol)
		if !isnothing(snapshot)
			price = _makerlimitprice(syminfo, snapshot, current.side)
			pricechanged = !isapprox((price), (current.limitprice); atol=0.0, rtol=0.0)
		end
	elseif !isnothing(limitprice)
		price = (limitprice)
		pricechanged = !isapprox((price), (current.limitprice); atol=0.0, rtol=0.0)
	end
	if qty == current.baseqty && !pricechanged
		return current
	end

	# Prefer native edit for both maker and non-maker orders to keep order identity.
	try
		params = Dict{String, Any}("txid" => String(orderid))
		params["volume"] = string(qty)
		if !isnothing(price)
			params["price"] = string(price)
		end
		response = HttpPrivateRequest(bc, "POST", "/0/private/EditOrder", params, "edit order")
		result = get(response, "result", Dict{String, Any}())
		txids = get(result, "txid", Any[])
		_invalidate_openorders_cache!()
		amended = if (txids isa AbstractVector) && !isempty(txids)
			order(bc, String(first(txids)))
		else
			order(bc, orderid)
		end
		if !isnothing(amended)
			return amended
		end
	catch err
		if maker
			(verbosity >= 1) && @warn "maker amend skipped without cancel/recreate" symbol=symbol orderid=orderid error=sprint(showerror, err)
			return current
		end
	end

	# Non-maker fallback: cancel and recreate when native edit is unavailable.

	cancelled = cancelorder(bc, symbol, orderid)
	if isnothing(cancelled)
		return nothing
	end

	recreated = createorder(bc, symbol, current.side, qty, price, maker)
	if isnothing(recreated)
		return nothing
	end
	return (recreated..., status="Replaced", rejectreason=string("Replaced order ", orderid))
end

"""
Return balances in a normalized DataFrame with coin, locked, free, borrowed and accrued interest.
"""
function _borrowedfromopenpositions(bc::KrakenSpotCache)::Dict{String, Float32}
	if !_hascredentials(bc)
		return Dict{String, Float32}()
	end

	# Cache borrowed estimation from OpenPositions because this is a private endpoint
	# and borrowed exposure changes relatively slowly compared to loop cadence.
	lock(_openpositions_state_lock)
	try
		now = Dates.now(Dates.UTC)
		if !isnothing(_openpositions_cache[]) && !isnothing(_openpositions_cache_time[])
			if (now - _openpositions_cache_time[]) < OPENPOSITIONS_CACHE_TTL
				return copy(_openpositions_cache[])
			end
		end
	finally
		unlock(_openpositions_state_lock)
	end

	lock(_openpositions_state_lock)
	try
		if !isnothing(_openpositions_disabled_until[]) && (Dates.now(Dates.UTC) < _openpositions_disabled_until[])
			return Dict{String, Float32}()
		end
	finally
		unlock(_openpositions_state_lock)
	end

	try
		response = HttpPrivateRequest(bc, "POST", "/0/private/OpenPositions", Dict("docalcs" => true), "open positions")
		result = get(response, "result", Dict{String, Any}())
		borrowed = _borrowedfromopenpositionsresult(bc, result)
		lock(_openpositions_state_lock)
		try
			_openpositions_nonce_failures[] = 0
			_openpositions_disabled_until[] = nothing
			_openpositions_cache[] = copy(borrowed)
			_openpositions_cache_time[] = Dates.now(Dates.UTC)
		finally
			unlock(_openpositions_state_lock)
		end
		return borrowed
	catch err
		if _isinvalidnonceerror(err)
			lock(_openpositions_state_lock)
			try
				_openpositions_nonce_failures[] += 1
				if _openpositions_nonce_failures[] >= 2
					_openpositions_disabled_until[] = Dates.now(Dates.UTC) + Minute(15)
					(verbosity >= 1) && @warn "disabling OpenPositions borrowed estimation after repeated invalid nonce" disabled_until=_openpositions_disabled_until[]
				else
					(verbosity >= 1) && @warn "invalid nonce for OpenPositions borrowed estimation; will retry on next cycle"
				end
			finally
				unlock(_openpositions_state_lock)
			end
			return Dict{String, Float32}()
		end
		rethrow(err)
	end
end

function _borrowedfromopenpositionsresult(bc::KrakenSpotCache, result)::Dict{String, Float32}
	borrowed = Dict{String, Float32}()
	for (_, rawpos) in result
		pos = rawpos isa AbstractDict ? Dict(rawpos) : Dict{String, Any}()
		positiontype = lowercase(String(get(pos, "type", "")))
		positiontype == "sell" || continue

		vol = abs(_num(get(pos, "vol", "0"), 0f0))
		vol > 0f0 || continue

		pairkey = String(get(pos, "pair", ""))
		isempty(pairkey) && continue
		symbol = _resultkey2symbol(bc, pairkey)
		syminfo = symbolinfo(bc, symbol)
		isnothing(syminfo) && continue
		basecoin = String(syminfo.basecoin)
		borrowed[basecoin] = get(borrowed, basecoin, 0f0) + vol
	end
	return borrowed
end

function _mergeborrowedbalances!(df::DataFrame, borrowed::Dict{String, Float32})::DataFrame
	for (coin, borrowedqty) in borrowed
		ix = findfirst(==(coin), String.(df[!, :coin]))
		if isnothing(ix)
			push!(df, (coin=coin, locked=0f0, free=0f0, borrowed=borrowedqty, accruedinterest=0f0))
		else
			df[ix, :borrowed] = (df[ix, :borrowed]) + borrowedqty
		end
	end
	return df
end

"Return Kraken free quantity from BalanceEx-style fields, accounting for all hold fields."
function _balancefreefromfields(value::AbstractDict{<:Any, <:Any})::Float32
	balance = _num(get(value, "balance", "0"), 0f0)
	available = _num(get(value, "available", string(balance)), balance)
	if haskey(value, "available")
		return max(0f0, min(balance, available))
	end
	hold_total = 0f0
	for (k, v) in value
		kstr = lowercase(String(k))
		if (kstr == "hold") || startswith(kstr, "hold_")
			hold_total += _num(v, 0f0)
		end
	end
	return max(0f0, balance - hold_total)
end

function balances(bc::KrakenSpotCache)
	df = DataFrame(coin=AbstractString[], locked=Float32[], free=Float32[], borrowed=Float32[], accruedinterest=Float32[])
	if !_hascredentials(bc)
		return df
	end

	# WS balances are intentionally disabled for KrakenSpot for now.
	# Reason: live private WS balances payloads use nested wallet structures and
	# do not consistently expose the normalized free/locked keys expected by the
	# adapter parser. Until WS parsing is aligned with the exchange payload shape,
	# REST BalanceEx is the authoritative source for free/locked balances.

	# Check balance cache (5s TTL to avoid Kraken API rate limits)
	lock(_balance_cache_lock) do
		now = Dates.now(UTC)
		if !isnothing(_balance_cache[]) && !isnothing(_balance_cache_time[])
			if (now - _balance_cache_time[]) < BALANCE_CACHE_TTL
				(verbosity >= 3) && println("balances: returning cached result (age=$(now - _balance_cache_time[]))")
				return copy(_balance_cache[])
			end
		end
	end

	response = nothing
	try
		response = HttpPrivateRequest(bc, "POST", "/0/private/BalanceEx", nothing, "balanceex")
	catch err_balanceex
		fallback = lock(_balance_cache_lock) do
			now = Dates.now(Dates.UTC)
			if !isnothing(_balance_cache[]) && !isnothing(_balance_cache_time[])
				if (now - _balance_cache_time[]) < BALANCE_CACHE_MAX_STALE
					(verbosity >= 1) && @warn "BalanceEx request failed; returning stale cached balances" age=(now - _balance_cache_time[]) balanceex_exception=sprint(showerror, err_balanceex)
					return copy(_balance_cache[])
				end
			end
			return nothing
		end
		if !isnothing(fallback)
			return fallback
		end
		throw(ErrorException("Kraken BalanceEx unavailable and no cached balances exist; refusing unsafe Balance fallback: $(sprint(showerror, err_balanceex))"))
	end
	result = get(response, "result", Dict{String, Any}())

	for (asset, value) in result
		coin = _normalizeasset(String(asset))
		if value isa AbstractDict
			vdict = Dict(value)
			balance = _num(get(vdict, "balance", "0"), 0f0)
			locked = _num(get(vdict, "hold_trade", "0"), 0f0)
			free = _balancefreefromfields(vdict)
			push!(df, (coin=coin, locked=locked, free=free, borrowed=0f0, accruedinterest=0f0))
		else
			free = _num(value, 0f0)
			push!(df, (coin=coin, locked=0f0, free=free, borrowed=0f0, accruedinterest=0f0))
		end
	end

	# Mitigation: derive borrowed base quantities from open margin short positions.
	# This is best-effort and complements Balance/BalanceEx data when borrowed is not exposed there.
	try
		borrowed = _borrowedfromopenpositions(bc)
		_mergeborrowedbalances!(df, borrowed)
	catch err
		(verbosity >= 1) && @warn "OpenPositions borrowed estimation failed" exception=(err, catch_backtrace())
	end

	# Cache the result for 5 seconds
	lock(_balance_cache_lock) do
		_balance_cache[] = copy(df)
		_balance_cache_time[] = Dates.now(UTC)
	end
	return df
end

"""
Extract the first data object from Kraken websocket `data` payloads.
"""
function _firstwsdata(data)
	if data isa AbstractVector
		return isempty(data) ? nothing : first(data)
	end
	return data
end

"""
Best-effort polling fallback for ticker subscriptions.
"""
function _polltickerfallback!(channel::Channel{Dict}, bc::KrakenSpotCache, symbol::String)
	while isopen(channel)
		snapshot = get24h(bc, symbol)
		if !isnothing(snapshot)
			put!(channel, Dict(
				"symbol" => String(snapshot.symbol),
				"askprice" => snapshot.askprice,
				"bidprice" => snapshot.bidprice,
				"lastprice" => snapshot.lastprice,
				"quotevolume24h" => snapshot.quotevolume24h,
				"pricechangepercent" => snapshot.pricechangepercent,
				"source" => "rest",
			))
		end
		sleep(2)
	end
end

"""
Subscribe to real-time ticker data via websocket and fall back to REST polling on failure.
"""
function ws_ticker(bc::KrakenSpotCache, symbol::String)
	channel = Channel{Dict}(32)
	@async begin
		wsok = false
		try
			wssymbol = _symbol2ws(symbol)
			WebSockets.open(KRAKEN_WS_PUBLIC) do ws
				wsok = true
				subscribe = Dict("method" => "subscribe", "params" => Dict("channel" => "ticker", "symbol" => [wssymbol]))
				WebSockets.send(ws, JSON3.write(subscribe))

				while isopen(ws) && isopen(channel)
					msgraw = WebSockets.receive(ws)
					!(msgraw isa String) && continue
					msg = JSON3.read(msgraw, Dict)
					if get(msg, "channel", "") == "ticker"
						payload = _firstwsdata(get(msg, "data", Any[]))
						if payload isa AbstractDict
							pdata = Dict(payload)
							pws = String(get(pdata, "symbol", wssymbol))
							symbol = _ws2symbol(pws)
							setmarketdataheartbeat!(bc, symbol, Dates.now(Dates.UTC))
							put!(channel, Dict(
								"symbol" => symbol,
								"askprice" => _num(get(pdata, "ask", "0"), 0f0),
								"bidprice" => _num(get(pdata, "bid", "0"), 0f0),
								"lastprice" => _num(get(pdata, "last", "0"), 0f0),
								"source" => "ws",
							))
						end
					end
				end
			end
		catch err
			(verbosity >= 1) && @warn "Kraken ws_ticker failed for $(symbol): $(err)"
		end

		if isopen(channel)
			!wsok && (verbosity >= 2) && @info "ws_ticker fallback to REST polling for $(symbol)"
			try
				_polltickerfallback!(channel, bc, symbol)
			catch err
				(verbosity >= 1) && @warn "ticker REST fallback failed for $(symbol): $(err)"
			end
		end
		isopen(channel) && close(channel)
	end
	return channel
end

"""
Best-effort polling fallback for kline subscriptions.
"""
function _pollklinefallback!(channel::Channel{Dict}, bc::KrakenSpotCache, symbol::String, interval::String)
	while isopen(channel)
		nowdt = Dates.now(Dates.UTC)
		startdt = nowdt - Dates.Minute(5 * _interval2minutes[interval])
		klines = getklines(bc, symbol; startDateTime=startdt, endDateTime=nowdt, interval=interval)
		if size(klines, 1) > 0
			lastrow = klines[end, :]
			put!(channel, Dict(
				"symbol" => _normalizepairsymbol(symbol),
				"opentime" => lastrow.opentime,
				"open" => lastrow.open,
				"high" => lastrow.high,
				"low" => lastrow.low,
				"close" => lastrow.close,
				"basevolume" => lastrow.basevolume,
				"source" => "rest",
			))
		end
		sleep(2)
	end
end

"""
Subscribe to real-time kline data via websocket and fall back to REST polling on failure.
"""
function ws_kline(bc::KrakenSpotCache, symbol::String, interval::String="1m")
	@assert interval in keys(_interval2minutes) "unknown interval=$(interval)"
	channel = Channel{Dict}(32)
	@async begin
		wsok = false
		try
			wssymbol = _symbol2ws(symbol)
			subscribe = Dict(
				"method" => "subscribe",
				"params" => Dict(
					"channel" => "ohlc",
					"symbol" => [wssymbol],
					"interval" => _interval2minutes[interval],
				),
			)
			WebSockets.open(KRAKEN_WS_PUBLIC) do ws
				wsok = true
				WebSockets.send(ws, JSON3.write(subscribe))
				while isopen(ws) && isopen(channel)
					msgraw = WebSockets.receive(ws)
					!(msgraw isa String) && continue
					msg = JSON3.read(msgraw, Dict)
					if get(msg, "channel", "") == "ohlc"
						payload = _firstwsdata(get(msg, "data", Any[]))
						if payload isa AbstractDict
							pdata = Dict(payload)
							symbol = _ws2symbol(String(get(pdata, "symbol", wssymbol)))
							candle = (
								opentime=Dates.unix2datetime(_int(get(pdata, "interval_begin", 0), 0)),
								open=_num(get(pdata, "open", "0"), 0f0),
								high=_num(get(pdata, "high", "0"), 0f0),
								low=_num(get(pdata, "low", "0"), 0f0),
								close=_num(get(pdata, "close", "0"), 0f0),
								basevolume=_num(get(pdata, "volume", "0"), 0f0),
							)
							_recordwskline!(symbol, interval, candle)
							setmarketdataheartbeat!(bc, symbol, Dates.now(Dates.UTC))
							put!(channel, Dict(
								"symbol" => symbol,
								"opentime" => candle.opentime,
								"open" => candle.open,
								"high" => candle.high,
								"low" => candle.low,
								"close" => candle.close,
								"basevolume" => candle.basevolume,
								"source" => "ws",
							))
						end
					end
				end
			end
		catch err
			(verbosity >= 1) && @warn "Kraken ws_kline failed for $(symbol): $(err)"
		end

		if isopen(channel)
			!wsok && (verbosity >= 2) && @info "ws_kline fallback to REST polling for $(symbol)"
			try
				_pollklinefallback!(channel, bc, symbol, interval)
			catch err
				(verbosity >= 1) && @warn "kline REST fallback failed for $(symbol): $(err)"
			end
		end
		isopen(channel) && close(channel)
	end
	return channel
end

"""
Request websocket auth token used by private Kraken websocket subscriptions.
"""
function _wsauthtoken(bc::KrakenSpotCache)
	response = HttpPrivateRequest(bc, "POST", "/0/private/GetWebSocketsToken", nothing, "websocket token")
	result = get(response, "result", Dict{String, Any}())
	return haskey(result, "token") ? String(result["token"]) : nothing
end

"""
Convert websocket payload to text when possible.
"""
function _wsraw2text(msgraw)
	if msgraw isa String
		return msgraw
	elseif msgraw isa AbstractVector{UInt8}
		return String(Vector{UInt8}(msgraw))
	end
	return nothing
end

function _newprivatechannel()
	return Channel{Dict}(128)
end

function _privateorderschannel!()
	ch = _ws_private_orders_channel[]
	if isnothing(ch) || !isopen(ch)
		ch = _newprivatechannel()
		_ws_private_orders_channel[] = ch
	end
	return ch
end

function _privatebalanceschannel!()
	ch = _ws_private_balances_channel[]
	if isnothing(ch) || !isopen(ch)
		ch = _newprivatechannel()
		_ws_private_balances_channel[] = ch
	end
	return ch
end

function _privatereadcooldownremainingseconds()::Float64
	lock(_private_rl_lock) do
		nowdt = Dates.now(Dates.UTC)
		until = _private_rl_cooldown_until[]
		if isnothing(until) || (nowdt >= until)
			return 0.0
		end
		return max(0.0, Dates.value(until - nowdt) / 1000)
	end
end

function _wsreconnectbackoffseconds(failure_streak::Int)::Float64
	streak = max(1, failure_streak)
	expwait = _ws_private_backoff_base_seconds * (2.0 ^ (streak - 1))
	basewait = max(Dates.value(_ws_private_min_reconnect_interval) / 1000, min(_ws_private_backoff_cap_seconds, expwait))
	jitter = ((mod(time_ns(), 1_000_000_000)) / 1_000_000_000) * _ws_private_backoff_jitter_seconds
	return basewait + jitter
end

function _run_private_ws_worker!(bc::KrakenSpotCache, order_channel::Channel{Dict}, balance_channel::Channel{Dict})
	lastordersnapshot = ""
	lastbalancesnapshot = ""
	failure_streak = 0
	next_reconnect_at = Dates.now(Dates.UTC)
	while isopen(order_channel) || isopen(balance_channel)
		nowdt = Dates.now(Dates.UTC)
		if nowdt < next_reconnect_at
			sleep(max(0.1, Dates.value(next_reconnect_at - nowdt) / 1000))
			continue
		end
		if !_hascredentials(bc)
			sleep(1)
			continue
		end
		cooldown_s = _privatereadcooldownremainingseconds()
		if cooldown_s > 0
			sleep(min(cooldown_s, 5.0))
			continue
		end
		try
			token = _wsauthtoken(bc)
			if isnothing(token)
				next_reconnect_at = Dates.now(Dates.UTC) + Dates.Millisecond(round(Int, 1000 * _wsreconnectbackoffseconds(max(1, failure_streak))))
				continue
			end
			failure_streak = 0
			next_reconnect_at = Dates.now(Dates.UTC) + _ws_private_min_reconnect_interval
			WebSockets.open(KRAKEN_WS_PRIVATE) do ws
				connection_started = Dates.now(Dates.UTC)
				orders_active = false
				WebSockets.send(ws, JSON3.write(Dict("method" => "subscribe", "params" => Dict("channel" => "executions", "token" => token))))
				while isopen(ws) && (isopen(order_channel) || isopen(balance_channel))
					msgraw = WebSockets.receive(ws)
					msgtxt = _wsraw2text(msgraw)
					isnothing(msgtxt) && continue
					msg = try
						JSON3.read(msgtxt, Dict)
					catch
						continue
					end
					ch = String(get(msg, "channel", ""))
					if ch == "" && haskey(msg, "result") && (msg["result"] isa AbstractDict)
						ch = String(get(msg["result"], "channel", ""))
					end
					if ch in ["heartbeat", "ping", "pong"]
						_touch_ws_orders_heartbeat!()
						_touch_ws_balances_heartbeat!()
					end
					if ch in ["executions", "orders", "openOrders"]
						rawdata = get(msg, "data", Any[])
						df = _ws_orders_df_from_payload(bc, rawdata)
						if size(df, 1) > 0
							_update_ws_orders_snapshot!(df)
						else
							_touch_ws_orders_heartbeat!()
						end
						orders_active = true
						isopen(order_channel) && put!(order_channel, Dict("topic" => ch, "source" => "ws", "data" => rawdata))
					end
					nowdt = Dates.now(Dates.UTC)
					if (nowdt - connection_started) > _ws_private_subscribe_ack_timeout
						if !orders_active
							throw(ErrorException("private websocket channel activation timeout orders_active=$(orders_active)"))
						end
					end
				end
			end
		catch err
			(verbosity >= 1) && @warn "Kraken private websocket worker failed: $(err)"
			failure_streak += 1
			next_reconnect_at = Dates.now(Dates.UTC) + Dates.Millisecond(round(Int, 1000 * _wsreconnectbackoffseconds(failure_streak)))
			cooldown_s = _privatereadcooldownremainingseconds()
			if isopen(order_channel)
				try
					if cooldown_s <= 0
						oo = openorders(bc)
						_update_ws_orders_snapshot!(oo)
						snapshot = string(hash(oo))
						if snapshot != lastordersnapshot
							put!(order_channel, Dict("topic" => "order", "source" => "rest", "data" => oo))
							lastordersnapshot = snapshot
						end
					end
				catch fallback_err
					(verbosity >= 1) && @warn "Kraken private worker order REST fallback failed: $(fallback_err)"
				end
			end
			if isopen(balance_channel)
				try
					if cooldown_s <= 0
						b = balances(bc)
						_update_ws_balances_snapshot!(b)
						snapshot = string(hash(b))
						if snapshot != lastbalancesnapshot
							put!(balance_channel, Dict("topic" => "balances", "source" => "rest", "data" => b))
							lastbalancesnapshot = snapshot
						end
					end
				catch fallback_err
					(verbosity >= 1) && @warn "Kraken private worker balance REST fallback failed: $(fallback_err)"
				end
			end
			sleep(min(_wsreconnectbackoffseconds(failure_streak), 5.0))
		end
	end
	return nothing
end

function _ensure_private_ws_worker!(bc::KrakenSpotCache)
	lock(_ws_private_stream_lock) do
		order_channel = _privateorderschannel!()
		balance_channel = _privatebalanceschannel!()
		if !_ws_private_worker_running[]
			_ws_private_worker_running[] = true
			@async begin
				try
					_run_private_ws_worker!(bc, order_channel, balance_channel)
				finally
					_ws_private_worker_running[] = false
				end
			end
		end
		return order_channel, balance_channel
	end
end

"""
Fallback polling for order updates when websocket order stream is unavailable.
"""
function _pollordersfallback!(channel::Channel{Dict}, bc::KrakenSpotCache)
	lastsnapshot = ""
	while isopen(channel)
		oo = openorders(bc)
		_update_ws_orders_snapshot!(oo)
		snapshot = string(hash(oo))
		if snapshot != lastsnapshot
			put!(channel, Dict("topic" => "order", "source" => "rest", "data" => oo))
			lastsnapshot = snapshot
		end
		sleep(2)
	end
end

"""
Subscribe to private order updates via websocket and fall back to polling open orders.
"""
function ws_orders(bc::KrakenSpotCache)
	if _hascredentials(bc)
		order_channel, _ = _ensure_private_ws_worker!(bc)
		return order_channel
	end
	channel = Channel{Dict}(32)
	@async begin
		(verbosity >= 2) && @info "ws_orders fallback to REST polling"
		try
			_pollordersfallback!(channel, bc)
		catch err
			(verbosity >= 1) && @warn "order REST fallback failed: $(err)"
		end
		isopen(channel) && close(channel)
	end
	return channel
end

function _pollbalancesfallback!(channel::Channel{Dict}, bc::KrakenSpotCache)
	lastsnapshot = ""
	while isopen(channel)
		b = balances(bc)
		_update_ws_balances_snapshot!(b)
		snapshot = string(hash(b))
		if snapshot != lastsnapshot
			put!(channel, Dict("topic" => "balances", "source" => "rest", "data" => b))
			lastsnapshot = snapshot
		end
		sleep(2)
	end
end

"""
Return balance updates via REST polling.

Websocket balances are intentionally disabled for KrakenSpot until private WS
balances payload parsing is aligned with live exchange payloads.
"""
function ws_balances(bc::KrakenSpotCache)
	channel = Channel{Dict}(32)
	@async begin
		(verbosity >= 2) && @info "ws_balances uses REST polling; KrakenSpot WS balances are disabled"
		try
			_pollbalancesfallback!(channel, bc)
		catch err
			(verbosity >= 1) && @warn "balance REST fallback failed: $(err)"
		end
		isopen(channel) && close(channel)
	end
	return channel
end

"""
Filter dictionary-like rows by regex over the provided key.
"""
function filterOnRegex(matcher, withDictArr; withKey="symbol")
	regex = Regex(matcher)
	return filter(x -> (haskey(x, withKey) && !isnothing(match(regex, string(x[withKey])))), withDictArr)
end

end
