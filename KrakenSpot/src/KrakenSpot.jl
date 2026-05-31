module KrakenSpot

using Base64, DataFrames, Dates, Downloads, EnvConfig, HTTP, JSON3, Logging, SHA, WebSockets

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

const KRAKEN_APIREST = "https://api.kraken.com"
const KRAKEN_WS_PUBLIC = "wss://ws.kraken.com/v2"
const KRAKEN_WS_PRIVATE = "wss://ws-auth.kraken.com/v2"

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
struct KrakenSpotCache
	syminfodf::Union{Nothing, DataFrame}
	apirest::String
	publickey::String
	secretkey::String
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
		targetquote = uppercase(EnvConfig.cryptoquote)
		filtered = syminfo[uppercase.(syminfo.quotecoin) .== targetquote, :]
		syminfo = size(filtered, 1) > 0 ? filtered : syminfo
		sort!(syminfo, :basecoin)
	end
	return KrakenSpotCache(syminfo, apirest, keys.publickey, keys.secretkey)
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

"Return a strictly increasing Kraken nonce (nanoseconds since epoch)."
function _nextnonce()::String
	lock(_nonce_lock)
	try
		# Use unix-nanoseconds scale to stay above prior ms/us-based nonces used by other clients.
		candidate = Int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1_000_000_000))
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
Convert a value to `Float32`, returning `default` when parsing fails.
"""
function _float32(value, default::Float32=0f0)::Float32
	if isnothing(value)
		return default
	elseif value isa Float32
		return value
	elseif value isa Real
		return Float32(value)
	elseif value isa AbstractString
		return value == "" ? default : try
			parse(Float32, value)
		catch
			default
		end
	elseif value isa AbstractVector
		return isempty(value) ? default : _float32(first(value), default)
	end
	return default
end

"Parse Kraken leverage values (e.g. \"2\", \"2:1\", \"none\") into a positive ratio or 0."
function _leveragevalue(value)::Float32
	if isnothing(value)
		return 0f0
	end
	if value isa Real
		v = Float32(value)
		return v > 0f0 ? v : 0f0
	end
	if value isa AbstractString
		s = lowercase(strip(String(value)))
		(s == "") && return 0f0
		(s == "none") && return 0f0
		if occursin(":", s)
			parts = split(s, ":")
			if !isempty(parts)
				lhs = try
					parse(Float32, strip(parts[1]))
				catch
					0f0
				end
				return lhs > 0f0 ? lhs : 0f0
			end
		end
		parsed = try
			parse(Float32, s)
		catch
			0f0
		end
		return parsed > 0f0 ? parsed : 0f0
	end
	if value isa AbstractVector
		return isempty(value) ? 0f0 : _leveragevalue(first(value))
	end
	return 0f0
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
	d = round(Int, log10(1 / Float64(step)))
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
	pricedigits = _precisiondigits(Float64(syminfo.ticksize), 5)
	qtydigits = _precisiondigits(Float64(syminfo.baseprecision), 8)

	normprice = Float32(round(Float64(limitprice), digits=pricedigits))
	normprice > 0f0 || throw(ArgumentError("normalized limitprice must be > 0, got $(normprice)"))

	normqty = Float64(basequantity)
	minquote = Float64(syminfo.minquoteqty)
	minbase = Float64(syminfo.minbaseqty)
	if (minquote > 0.0) && ((normqty * Float64(normprice)) < minquote)
		normqty = minquote / Float64(normprice)
	end
	normqty = max(normqty, minbase)
	normqty = floor(normqty, digits=qtydigits)
	if normqty < minbase
		normqty = minbase
	end
	if (minquote > 0.0) && ((normqty * Float64(normprice)) < minquote)
		normqty = ceil(minquote / Float64(normprice), digits=qtydigits)
		normqty = max(normqty, minbase)
	end
	return (basequantity=Float32(normqty), limitprice=normprice, qtydigits=qtydigits, pricedigits=pricedigits)
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
function emptyorders()::DataFrame
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
		lastcheck=DateTime[],
	)
end

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
function symboltoken(bc::KrakenSpotCache, basecoin::AbstractString, quotecoin::AbstractString=EnvConfig.cryptoquote)::String
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
					jump_ns = retry_ix >= 3 ? 2_000_000_000_000_000_000 : 5_000_000_000
					lock(_nonce_lock)
					try
						now_ns = Int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1_000_000_000))
						base = max(now_ns, _last_nonce[])
						limit = typemax(Int) - 1_000_000
						candidate = base >= (limit - jump_ns) ? limit : (base + jump_ns)
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
				basecoin = uppercase(splitpair[1])
				quotecoin = uppercase(splitpair[2])
			end
		end
		if (basecoin == "") || (quotecoin == "")
			basecoin = _normalizeasset(String(get(info, "base", "")))
			quotecoin = _normalizeasset(String(get(info, "quote", "")))
		end

		pairdecimals = _int(get(info, "pair_decimals", 5), 5)
		lotdecimals = _int(get(info, "lot_decimals", 8), 8)
		ticksize = Float32(10.0^-pairdecimals)
		baseprecision = Float32(10.0^-lotdecimals)
		minbaseqty = _float32(get(info, "ordermin", "0"), 0f0)
		minquoteqty = _float32(get(info, "costmin", "0"), 0f0)
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
	return uppercase(sym.quotecoin) == uppercase(EnvConfig.cryptoquote) && _istradablestatus(sym.status)
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
	ask = _float32(get(ticker, "a", Any[]), 0f0)
	bid = _float32(get(ticker, "b", Any[]), 0f0)
	lastprice = _float32(get(ticker, "c", Any[]), 0f0)
	openprice = _float32(get(ticker, "o", "0"), 0f0)
	basevolume = _float32(get(ticker, "v", Any[]), 0f0)
	quotevolume = basevolume * lastprice
	pricechangepercent = openprice == 0f0 ? 0f0 : Float32((lastprice - openprice) / openprice)
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
			open=_float32(row[2]),
			high=_float32(row[3]),
			low=_float32(row[4]),
			close=_float32(row[5]),
			basevolume=_float32(row[7]),
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
	return Float64(_float32(_dictgetci(d, keys, default), Float32(default)))
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
	quotecoin = uppercase(String(EnvConfig.cryptoquote))
	try
		bdf = balances(bc)
		if (:coin in names(bdf)) && (:free in names(bdf))
			for row in eachrow(bdf)
				if uppercase(String(row.coin)) == quotecoin
					available_long_quote += max(0.0, Float64(row.free))
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
Convert one raw Kraken order entry into the standardized order row shape.
"""
function _orderrow(bc::KrakenSpotCache, orderid::AbstractString, entry::Dict)
	descr = haskey(entry, "descr") ? Dict(entry["descr"]) : Dict{String, Any}()
	rawpair = String(get(descr, "pair", ""))
	symbol = _resultkey2symbol(bc, rawpair)
	side = lowercase(String(get(descr, "type", "buy"))) == "buy" ? "Buy" : "Sell"
	ordertype = titlecase(String(get(descr, "ordertype", "limit")))
	baseqty = _float32(get(entry, "vol", "0"), 0f0)
	executedqty = _float32(get(entry, "vol_exec", "0"), 0f0)
	limitprice = _float32(get(descr, "price", get(entry, "price", "0")), 0f0)
	avgprice = _float32(get(entry, "price", limitprice), limitprice)
	status = titlecase(String(get(entry, "status", "open")))
	created = Dates.unix2datetime(_int(round(_float32(get(entry, "opentm", 0), 0f0))))
	updated = created
	leverage = _leveragevalue(get(descr, "leverage", "0"))
	orderLinkId = String(get(entry, "cl_ord_id", ""))
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
		lastcheck=Dates.now(Dates.UTC),
	)
end

"""
Return open spot orders in a DataFrame compatible with Bybit order columns.
"""
function openorders(bc::KrakenSpotCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)
	if !_hascredentials(bc)
		return emptyorders()
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

	out = emptyorders()
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
	df = emptyorders()
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
	ticksize = Float32(syminfo.ticksize)
	side = lowercase(String(orderside))
	price = side == "buy" ? Float32(snapshot.askprice) - ticksize : Float32(snapshot.bidprice) + ticksize
	return price > 0f0 ? price : nothing
end

"""
Create one spot order and return an order row compatible named tuple.

If `price` is omitted and `maker=true`, the adapter will choose a limit price
as close as possible to the current spread while remaining post-only so the
order can qualify for maker fees.
"""
function createorder(bc::KrakenSpotCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0, reduceonly::Bool=false)
	@assert basequantity > 0.0 "createorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
	@assert isnothing(price) || (price > 0.0) "createorder symbol=$(symbol) price=$(price) must be > 0"
	@assert lowercase(orderside) in ["buy", "sell"] "createorder symbol=$(symbol) orderside=$(orderside) must be Buy or Sell"
	if !_hascredentials(bc)
		return nothing
	end
	_validatemarginleverage(marginleverage)

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
	if marginleverage > 0
		limits = marginlimits(bc, symbol)
		if !marginpermitted(bc, symbol, orderside, marginleverage)
			throw(ErrorException("Kraken spot margin not permitted for symbol=$(symbol) pair=$(pairname) side=$(orderside) requested_leverage=$(marginleverage)x max_buy=$(limits.maxleveragebuy)x max_sell=$(limits.maxleveragesell)x status=$(syminfo.status)"))
		end
	end
	adaptivepost = maker && isnothing(price)
	attempts = adaptivepost ? 5 : 1
	ordertype = (maker || !isnothing(price)) ? "limit" : "market"
	chosenqty = Float32(basequantity)
	effectiveprice = isnothing(price) ? nothing : Float32(price)
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
		end

		params = Dict{String, Any}(
			"pair" => pairname,
			"type" => lowercase(orderside),
			"ordertype" => ordertype,
			"volume" => string(chosenqty),
			"cl_ord_id" => clientOrderId,
		)
		if ordertype == "limit"
			params["price"] = string(effectiveprice)
		end
		if maker && (ordertype == "limit")
			params["oflags"] = "post"
		end
		if marginleverage > 0
			params["leverage"] = string(marginleverage)
		end
		(reduceonly && (marginleverage > 0)) && (params["reduce_only"] = true)

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
		ordertype=titlecase(ordertype),
			timeinforce=(maker && !isnothing(effectiveprice)) ? "PostOnly" : "GTC",
			limitprice=isnothing(effectiveprice) ? 0f0 : Float32(effectiveprice),
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
function closeorder(bc::KrakenSpotCache, symbol::String, positionside::Symbol, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0, reduceonly::Bool=true)
	side = Symbol(lowercase(String(positionside)))
	@assert side in [:long, :short] "closeorder positionside=$(positionside) must be :long or :short"
	orderside = side == :long ? "Sell" : "Buy"
	return createorder(bc, symbol, orderside, basequantity, price, maker; marginleverage=marginleverage, reduceonly=reduceonly)
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

	qty = isnothing(basequantity) ? current.baseqty : Float32(basequantity)
	syminfo = symbolinfo(bc, symbol)
	isnothing(syminfo) && return nothing
	maker = current.timeinforce == "PostOnly"
	price = current.limitprice
	pricechanged = false
	if maker
		snapshot = get24h(bc, symbol)
		if !isnothing(snapshot)
			price = _makerlimitprice(syminfo, snapshot, current.side)
			pricechanged = !isapprox(Float64(price), Float64(current.limitprice); atol=0.0, rtol=0.0)
		end
	elseif !isnothing(limitprice)
		price = Float32(limitprice)
		pricechanged = !isapprox(Float64(price), Float64(current.limitprice); atol=0.0, rtol=0.0)
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

		vol = abs(_float32(get(pos, "vol", "0"), 0f0))
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
			df[ix, :borrowed] = Float32(df[ix, :borrowed]) + borrowedqty
		end
	end
	return df
end

"Return Kraken free quantity from BalanceEx-style fields, accounting for all hold fields."
function _balancefreefromfields(value::AbstractDict{<:Any, <:Any})::Float32
	balance = _float32(get(value, "balance", "0"), 0f0)
	available = _float32(get(value, "available", string(balance)), balance)
	if haskey(value, "available")
		return max(0f0, min(balance, available))
	end
	hold_total = 0f0
	for (k, v) in value
		kstr = lowercase(String(k))
		if (kstr == "hold") || startswith(kstr, "hold_")
			hold_total += _float32(v, 0f0)
		end
	end
	return max(0f0, balance - hold_total)
end

function balances(bc::KrakenSpotCache)
	df = DataFrame(coin=AbstractString[], locked=Float32[], free=Float32[], borrowed=Float32[], accruedinterest=Float32[])
	if !_hascredentials(bc)
		return df
	end

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
			balance = _float32(get(vdict, "balance", "0"), 0f0)
			locked = _float32(get(vdict, "hold_trade", "0"), 0f0)
			free = _balancefreefromfields(vdict)
			push!(df, (coin=coin, locked=locked, free=free, borrowed=0f0, accruedinterest=0f0))
		else
			free = _float32(value, 0f0)
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
							put!(channel, Dict(
								"symbol" => _ws2symbol(pws),
								"askprice" => _float32(get(pdata, "ask", "0"), 0f0),
								"bidprice" => _float32(get(pdata, "bid", "0"), 0f0),
								"lastprice" => _float32(get(pdata, "last", "0"), 0f0),
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
							put!(channel, Dict(
								"symbol" => _ws2symbol(String(get(pdata, "symbol", wssymbol))),
								"opentime" => Dates.unix2datetime(_int(get(pdata, "interval_begin", 0), 0)),
								"open" => _float32(get(pdata, "open", "0"), 0f0),
								"high" => _float32(get(pdata, "high", "0"), 0f0),
								"low" => _float32(get(pdata, "low", "0"), 0f0),
								"close" => _float32(get(pdata, "close", "0"), 0f0),
								"basevolume" => _float32(get(pdata, "volume", "0"), 0f0),
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
Fallback polling for order updates when websocket order stream is unavailable.
"""
function _pollordersfallback!(channel::Channel{Dict}, bc::KrakenSpotCache)
	lastsnapshot = ""
	while isopen(channel)
		oo = openorders(bc)
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
	channel = Channel{Dict}(32)
	@async begin
		wsok = false
		if _hascredentials(bc)
			try
				token = _wsauthtoken(bc)
				if !isnothing(token)
					subscribe = Dict("method" => "subscribe", "params" => Dict("channel" => "executions", "token" => token))
					WebSockets.open(KRAKEN_WS_PRIVATE) do ws
						wsok = true
						WebSockets.send(ws, JSON3.write(subscribe))
						while isopen(ws) && isopen(channel)
							msgraw = WebSockets.receive(ws)
							!(msgraw isa String) && continue
							msg = JSON3.read(msgraw, Dict)
							ch = String(get(msg, "channel", ""))
							if ch in ["executions", "orders", "openOrders"]
								put!(channel, Dict("topic" => ch, "source" => "ws", "data" => get(msg, "data", Any[])))
							end
						end
					end
				end
			catch err
				(verbosity >= 1) && @warn "Kraken ws_orders failed: $(err)"
			end
		end

		if isopen(channel)
			!wsok && (verbosity >= 2) && @info "ws_orders fallback to REST polling"
			try
				_pollordersfallback!(channel, bc)
			catch err
				(verbosity >= 1) && @warn "order REST fallback failed: $(err)"
			end
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
