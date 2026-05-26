module KrakenFutures

using Base64, DataFrames, Dates, Downloads, EnvConfig, HTTP, JSON3, Logging, SHA, WebSockets

# Rate-limit and diagnostics state (mirroring KrakenSpot)
const _private_rl_lock = ReentrantLock()
const _private_rl_cooldown_until = Ref{Union{Nothing, DateTime}}(nothing)
const _private_call_counter_lock = ReentrantLock()
const _private_call_counter = Dict{String, Int}()
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
		@info "[KrakenFutures private call rate summary]" total_calls_per_min=total_calls max_calls_per_endpoint_per_min=max_calls avg_calls_per_endpoint_per_min=round(avg_calls; digits=2) endpoints_tracked=endpoint_count
		empty!(_private_call_counter)
	end
	return nothing
end

function _isreadonlyprivateendpoint_futures(endPoint::AbstractString)::Bool
	# Only GET endpoints for account/balance/openorders are considered read-only for cooldown
	return endPoint in [
		"/accounts",
		"/openorders",
		"/fills",
		"/openpositions",
		"/orders/status",
		"/historicalorders",
		"/historicaltriggers",
		"/historicalexecutions",
		"/accountlogcsv",
		"/accountlog",
	]
end

function _omitnonceforreadonlyenabled()::Bool
	raw = lowercase(strip(get(ENV, "KRAKEN_FUTURES_OMIT_NONCE_READS", "false")))
	return raw in ("1", "true", "yes", "on")
end

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

const KRAKEN_FUTURES_APIREST = "https://futures.kraken.com/derivatives/api/v3"
const KRAKEN_CHARTS_APIREST = "https://futures.kraken.com/api/charts/v1"
const KRAKEN_FUTURES_WS_PUBLIC = "wss://futures.kraken.com/ws/v1"
const KRAKEN_FUTURES_WS_PRIVATE = "wss://futures.kraken.com/ws/v1"

const KRAKEN_SPOT_APIREST = "https://api.kraken.com"

const _interval2minutes = Dict(
	"1m" => 1,
	"5m" => 5,
	"15m" => 15,
	"30m" => 30,
	"1h" => 60,
	"4h" => 240,
	"1d" => 1440,
)

const _known_quotes = ["USDT", "USD", "USDC", "EUR", "BTC", "ETH"]
const _nonce_lock = ReentrantLock()
const _last_nonce = Ref{Int}(0)
const _nonce_floor_poison_factor = 20
const _nonce_ms_min_increment = 1
const _nonce_ns_min_increment = 1_000_000_000  # 1 s in ns mode
const _nonce_ns_switch_threshold = 1_000_000_000_000_000

"Return current unix time in milliseconds as Int."
_nonce_ms_base()::Int = Int(floor(time() * 1000))

"Return base*2^(retry_ix-1) with overflow-safe scaling and upper cap."
function _capped_retry_jump(base::Int, retry_ix::Int, cap::Int)::Int
	retry_ix <= 1 && return min(base, cap)
	jump = min(base, cap)
	for _ in 2:retry_ix
		if jump >= (cap ÷ 2)
			return cap
		end
		jump *= 2
	end
	return min(jump, cap)
end

function _nonce_state_dir()::String
	dir = normpath(joinpath(EnvConfig.cryptopath, "debug", "krakenfutures"))
	isdir(dir) || mkpath(dir)
	return dir
end

function _nonce_state_file(publickey::AbstractString)::String
	keyhash = bytes2hex(SHA.sha1(Vector{UInt8}(String(publickey))))
	return joinpath(_nonce_state_dir(), "nonce_floor_$(keyhash).json")
end

function _restore_nonce_floor!(bc)
	!_hascredentials(bc) && return nothing
	path = _nonce_state_file(bc.publickey)
	isfile(path) || return nothing
	try
		payload = JSON3.read(read(path, String), Dict{String, Any})
		stored = Int(get(payload, "last_nonce", 0))
		now_ms = _nonce_ms_base()
		now_ns = Int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1_000_000_000))
		nowbase = stored >= _nonce_ns_switch_threshold ? now_ns : now_ms
		rebased = false
		# Keep persistent nonce state in its current scale (ms or ns).
		if (stored > 0) && (stored > nowbase) && ((stored ÷ max(nowbase, 1)) > _nonce_floor_poison_factor)
			(verbosity >= 1) && @warn "discarding poisoned KrakenFutures nonce floor and rebasing" stored_nonce=stored rebased_nonce=nowbase path
			stored = nowbase
			rebased = true
		end
		lock(_nonce_lock)
		try
			_last_nonce[] = max(_last_nonce[], stored)
		finally
			unlock(_nonce_lock)
		end
		rebased && _persist_nonce_floor!(bc, _last_nonce[])
		(verbosity >= 3) && println("restored KrakenFutures nonce floor from $(path), last_nonce=$(_last_nonce[])")
	catch err
		(verbosity >= 1) && @warn "failed to restore KrakenFutures nonce floor" path exception=(err, catch_backtrace())
	end
	return nothing
end

function _persist_nonce_floor!(bc, nonce::Int)
	!_hascredentials(bc) && return nothing
	path = _nonce_state_file(bc.publickey)
	tmppath = string(path, ".tmp")
	try
		open(tmppath, "w") do io
			write(io, JSON3.write(Dict("last_nonce" => nonce, "updated_at_utc" => string(Dates.now(Dates.UTC)))))
		end
		mv(tmppath, path; force=true)
	catch err
		isfile(tmppath) && rm(tmppath; force=true)
		(verbosity >= 1) && @warn "failed to persist KrakenFutures nonce floor" path exception=(err, catch_backtrace())
	end
	return nothing
end

# Balance caching to avoid Kraken API rate limits (10-25 req/sec for futures)
# Cache TTL: 5 seconds. At 1 balance call/minute, this is well under the rate limit.
const _balance_cache_lock = ReentrantLock()
const _balance_cache = Ref{Union{Nothing, DataFrame}}(nothing)
const _balance_cache_time = Ref{Union{Nothing, DateTime}}(nothing)
const BALANCE_CACHE_TTL = Dates.Second(5)

"""
Cached KrakenFutures state used by higher-level trading modules.
"""
struct KrakenFuturesCache
	syminfodf::Union{Nothing, DataFrame}
	apirest::String
	publickey::String
	secretkey::String
end

"""
Build a new `KrakenFuturesCache` and optionally preload symbol metadata.
"""
function KrakenFuturesCache(; autoloadexchangeinfo::Bool=true, apirest::String=KRAKEN_FUTURES_APIREST, publickey::Union{Nothing, AbstractString}=nothing, secretkey::Union{Nothing, AbstractString}=nothing)
	keys = _resolve_credentials(publickey, secretkey)
	bc = KrakenFuturesCache(_emptyexchangeinfo(), apirest, keys.publickey, keys.secretkey)
	_restore_nonce_floor!(bc)
	syminfo = _emptyexchangeinfo()
	if autoloadexchangeinfo
		try
			syminfo = _exchangeinfo(apirest)
		catch err
			(verbosity >= 1) && @warn "failed to load KrakenFutures exchange info: $(err)"
			syminfo = _emptyexchangeinfo()
		end
		if size(syminfo, 1) > 0
			targetquote = uppercase(EnvConfig.cryptoquote)
			filtered = syminfo[uppercase.(syminfo.quotecoin) .== targetquote, :]
			syminfo = size(filtered, 1) > 0 ? filtered : syminfo
			sort!(syminfo, :basecoin)
		end
	end
	return KrakenFuturesCache(syminfo, apirest, keys.publickey, keys.secretkey)
end

"""
Resolve Kraken credentials from explicit args, `EnvConfig`, or environment variables.
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

	envpublic = get(ENV, "KRAKEN_FUTURES_APIKEY", get(ENV, "KRAKEN_APIKEY", ""))
	envsecret = get(ENV, "KRAKEN_FUTURES_SECRET", get(ENV, "KRAKEN_SECRET", ""))

	resolvedpublic = isnothing(publickey) || (publickey == "") ? (cfgpublic != "" ? cfgpublic : envpublic) : String(publickey)
	resolvedsecret = isnothing(secretkey) || (secretkey == "") ? (cfgsecret != "" ? cfgsecret : envsecret) : String(secretkey)
	return (publickey=resolvedpublic, secretkey=resolvedsecret)
end

"""
Return `true` when API credentials are present in the cache.
"""
_hascredentials(bc::KrakenFuturesCache)::Bool = (bc.publickey != "") && (bc.secretkey != "")

"""
Get the first key present in `dict` from `keys`, otherwise return `default`.
"""
function _tryget(dict::Dict, keys::Vector{String}, default=nothing)
	for key in keys
		if haskey(dict, key)
			return dict[key]
		end
	end
	return default
end

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
Low-level HMAC implementation used for request signing.
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
Compute futures private API signature using the KrakenEx-compatible algorithm.

Matches the signing logic from https://github.com/btschwertfeger/KrakenEx.jl:
  1. Concatenate: data (URL-encoded params) + nonce + endpoint (without `/derivatives` prefix)
  2. SHA256 the concatenated UTF-8 bytes → 32-byte digest
  3. HMAC-SHA512 the digest using base64-decoded secret → 64-byte MAC
  4. Base64-encode the MAC

This is the correct Kraken Futures REST API signature.
"""
function _futuressignature(sigendpoint::String, nonce::String, data::String, secret::String)::String
	decoded = try
		Base64.base64decode(secret)
	catch
		Vector{UInt8}(secret)
	end
	msg = SHA.sha256(Vector{UInt8}(data * nonce * sigendpoint))
	return Base64.base64encode(_hmac(decoded, msg, SHA.sha512, 128))
end

"""
Raise an error if API response indicates a failure.
"""
function _checkresponse(response::Dict, info::AbstractString)
	if haskey(response, "error")
		errors = response["error"]
		if (errors isa AbstractVector) && !isempty(errors)
			throw(ErrorException("API error in $(info): $(join(string.(errors), "; "))"))
		elseif (errors isa AbstractString) && (errors != "")
			throw(ErrorException("API error in $(info): $(errors)"))
		end
	end
	if haskey(response, "success") && (response["success"] isa Bool) && !response["success"]
		throw(ErrorException("API reported unsuccessful result in $(info): $(response)"))
	end
	if haskey(response, "result") && (response["result"] isa AbstractString)
		rs = lowercase(String(response["result"]))
		if rs in ["error", "failed", "failure"]
			resultvalue = String(response["result"])
			throw(ErrorException("API result=$(resultvalue) in $(info): $(response)"))
		end
	end
end

function _httpmemorycompaterror(err)::Bool
	msg = sprint(showerror, err)
	return (err isa MethodError) && occursin("SubArray{UInt8,1,Memory{UInt8}", msg) && occursin("SubArray{UInt8,1,Vector{UInt8}", msg)
end

function _isinvalidnonceerror(err)::Bool
	msg = lowercase(sprint(showerror, err))
	return occursin("invalid nonce", msg) || occursin("noncebelowthreshold", msg) || occursin("too_small", msg) || occursin("nonce too small", msg) || occursin("nonceduplicate", msg) || occursin("duplicate", msg)
end

function _isnoncetoosmallerror(err)::Bool
	msg = lowercase(sprint(showerror, err))
	return occursin("noncebelowthreshold", msg) || occursin("too_small", msg) || occursin("nonce too small", msg)
end

function _isnonceduplicateerror(err)::Bool
	msg = lowercase(sprint(showerror, err))
	return occursin("nonceduplicate", msg) || occursin("duplicate", msg)
end

function _isratelimiterror(err)::Bool
	return occursin("rate limit exceeded", lowercase(sprint(showerror, err)))
end

"Return a strictly increasing Kraken nonce in ms scale."
function _nextnonce(bc::KrakenFuturesCache)::String
	lock(_nonce_lock)
	try
		use_ns = _last_nonce[] >= _nonce_ns_switch_threshold
		candidate = use_ns ? Int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1_000_000_000)) : _nonce_ms_base()
		if candidate <= _last_nonce[]
			candidate = _last_nonce[] + (use_ns ? _nonce_ns_min_increment : _nonce_ms_min_increment)
		end
		_last_nonce[] = candidate
		_persist_nonce_floor!(bc, candidate)
		return string(candidate)
	finally
		unlock(_nonce_lock)
	end
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

"""
Convert value to `Float32`, returning `default` when parsing fails.
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

"""
Convert value to `Int`, returning `default` when parsing fails.
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

"Normalize limit order price/qty to exchange precision and min constraints."
function _normalizelimitorderparams(syminfo::DataFrameRow, basequantity::Real, limitprice::Real)
	pricedigits = _precisiondigits(Float64(syminfo.ticksize), 5)
	qtydigits = _precisiondigits(Float64(syminfo.baseprecision), 0)

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

"Validate requested futures leverage before submitting Kraken futures orders."
function _validatemarginleverage(marginleverage::Signed)
	if marginleverage == 0
		return nothing
	end
	if !(1 <= marginleverage <= 50)
		throw(ArgumentError("Kraken futures leverage must be 0 or in 1:50, got $(marginleverage)"))
	end
	return nothing
end

"Parse one value into a positive leverage int, returning 0 when unavailable."
function _leverageint(value)::Int
	n = _int(value, 0)
	if n > 0
		return n
	end
	if value isa AbstractString
		s = replace(strip(String(value)), "x" => "")
		f = try
			parse(Float64, s)
		catch
			0.0
		end
		return f > 0 ? Int(round(f)) : 0
	end
	return 0
end

"Pick the best available max leverage from alternative metadata keys."
function _futuresmaxleverage(info::Dict, keys::Vector{String})::Int
	for key in keys
		if haskey(info, key)
			raw = info[key]
			if raw isa AbstractVector
				vals = Int[]
				for entry in raw
					lev = _leverageint(entry)
					lev > 0 && push!(vals, lev)
				end
				!isempty(vals) && return maximum(vals)
			else
				lev = _leverageint(raw)
				lev > 0 && return lev
			end
		end
	end
	return 0
end

"""
Convert API time values to UTC `DateTime`.
"""
function _todatetime(value)
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
			return _todatetime(parse(Float64, s))
		catch
		end
	end
	return Dates.now(Dates.UTC)
end

"""
Build exchange info schema.
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
Build open orders schema.
"""
function emptyorders()::DataFrame
	return DataFrame(
		orderid=String[],
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
Normalize Kraken asset names.
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
Map normalized assets to Kraken futures naming.
"""
_asset2kraken(asset::AbstractString)::String = uppercase(asset) == "BTC" ? "XBT" : uppercase(asset)

"""
Convert futures feed symbol to normalized symbol.
"""
function _ws2symbol(wsname::AbstractString)::String
	s = uppercase(strip(wsname))
	s = replace(s, "PF_" => "", "PI_" => "", "FI_" => "", "IN_" => "")
	s = replace(s, "/" => "", "_" => "")
	for q in sort(_known_quotes, by=length, rev=true)
		if endswith(s, q) && (length(s) > length(q))
			base = _normalizeasset(s[1:end-length(q)])
			return string(base, q)
		end
	end
	return s
end

"""
Convert normalized symbol into a futures product id.
"""
function _symbol2ws(symbol::AbstractString)::String
	clean = uppercase(replace(symbol, "/" => "", "_" => ""))
	for q in sort(_known_quotes, by=length, rev=true)
		if endswith(clean, q) && (length(clean) > length(q))
			base = _asset2kraken(clean[1:end-length(q)])
			return string("PI_", base, q)
		end
	end
	return clean
end

"""
Normalize any Kraken symbol representation.
"""
_normalizepairsymbol(pair::AbstractString)::String = _ws2symbol(pair)

"""
Resolve the normalized internal symbol for a `(basecoin, quotecoin)` pair.
"""
function symboltoken(bc::KrakenFuturesCache, basecoin::AbstractString, quotecoin::AbstractString=EnvConfig.cryptoquote)::String
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
Resolve internal symbol to exchange symbol.
"""
function _symbol2pairname(bc::KrakenFuturesCache, symbol::AbstractString)::String
	sym = _normalizepairsymbol(symbol)
	if !isnothing(bc.syminfodf) && (size(bc.syminfodf, 1) > 0)
		ix = findfirst(==(sym), bc.syminfodf[!, :symbol])
		if !isnothing(ix)
			return bc.syminfodf[ix, :krakenpairname]
		end
	end
	return _symbol2ws(sym)
end

"""
Resolve exchange symbol to internal symbol.
"""
function _resultkey2symbol(bc::KrakenFuturesCache, key::AbstractString)::String
	if !isnothing(bc.syminfodf) && (size(bc.syminfodf, 1) > 0)
		ix = findfirst(==(key), bc.syminfodf[!, :krakenpairname])
		if !isnothing(ix)
			return bc.syminfodf[ix, :symbol]
		end
		ix = findfirst(==(key), bc.syminfodf[!, :wsname])
		if !isnothing(ix)
			return bc.syminfodf[ix, :symbol]
		end
	end
	return _normalizepairsymbol(key)
end

"""
Public REST helper.
"""
function HttpPublicRequest(bc::KrakenFuturesCache, method::AbstractString, endPoint::AbstractString, params::Union{Dict, Nothing}, info::AbstractString; baseurl::String=bc.apirest)
	url = baseurl * endPoint
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
Private REST helper for Kraken Futures authenticated endpoints.

Supports GET (query params in URL) and POST (params in body).
The signature endpoint is built as `/api/v3` + endPoint, matching
the algorithm used by KrakenEx (which strips the leading `/derivatives`).
"""
function HttpPrivateRequest(bc::KrakenFuturesCache, method::AbstractString, endPoint::AbstractString, params::Union{Dict, Nothing}, info::AbstractString; baseurl::String=bc.apirest, max_attempts::Int=20)
	if !_hascredentials(bc)
		throw(ArgumentError("credentials are required for $(info)"))
	end

	method = uppercase(method)

	# Cooldown gate for read-only endpoints
	if _isreadonlyprivateendpoint_futures(endPoint)
		lock(_private_rl_lock) do
			now = Dates.now(Dates.UTC)
			if !isnothing(_private_rl_cooldown_until[]) && (now < _private_rl_cooldown_until[])
				throw(ErrorException("KrakenFutures private read cooldown active until $(_private_rl_cooldown_until[])"))
			end
		end
	end

	# Diagnostics: count private endpoint calls for end-of-run summary only.
	lock(_private_call_counter_lock) do
		_private_call_counter[endPoint] = get(_private_call_counter, endPoint, 0) + 1
	end
	baseparams = isnothing(params) ? Dict{String, Any}() : Dict{String, Any}(string(k) => v for (k, v) in params)
	initial_attempts = max(1, max_attempts)
	attempts = initial_attempts
	last_error = nothing
	while attempts > 0
		reqparams = copy(baseparams)
		omit_nonce = _omitnonceforreadonlyenabled() && (method == "GET") && _isreadonlyprivateendpoint_futures(endPoint)
		nonce = omit_nonce ? "" : _nextnonce(bc)

		# Encode params as URL query string (used for both signature and request body/URL)
		paramstr = isempty(reqparams) ? "" : _dict2paramsget(reqparams)

		# Signature endpoint: strip leading /derivatives (Kraken docs use /api/v3/... for signing)
		sigendpoint = "/api/v3" * endPoint

		headers = [
			"Content-Type" => "application/x-www-form-urlencoded; charset=utf-8",
			"APIKey" => bc.publickey,
		]
		if !omit_nonce
			push!(headers, "Nonce" => nonce)
		end
		push!(headers, "Authent" => _futuressignature(sigendpoint, nonce, paramstr, bc.secretkey))

		body = try
			if method == "GET"
				url = isempty(paramstr) ? baseurl * endPoint : baseurl * endPoint * "?" * paramstr
				# Keep retries at this layer disabled so nonce changes between attempts.
				# The surrounding attempts loop already handles retries with fresh nonce values.
				response = HTTP.request("GET", url, headers; retries=0, readtimeout=60)
				JSON3.read(String(response.body), Dict)
			else
				response = HTTP.request("POST", baseurl * endPoint, headers, paramstr; retries=0, retry_non_idempotent=false, readtimeout=60)
				JSON3.read(String(response.body), Dict)
			end
		catch err
			if _httpmemorycompaterror(err)
				(verbosity >= 3) && @warn "HTTP.request compatibility fallback to Downloads.request" method endpoint=endPoint info
				hvec = Pair{String, String}[String(k) => String(v) for (k, v) in headers]
				if method == "GET"
					url = isempty(paramstr) ? baseurl * endPoint : baseurl * endPoint * "?" * paramstr
					_downloadsrequest("GET", url; headers=hvec)
				else
					_downloadsrequest("POST", baseurl * endPoint; headers=hvec, body=paramstr)
				end
			elseif occursin("econnreset", lowercase(sprint(showerror, err))) || occursin("connection reset", lowercase(sprint(showerror, err))) || occursin("timed out", lowercase(sprint(showerror, err))) || occursin("timeout", lowercase(sprint(showerror, err))) || occursin("recv failure", lowercase(sprint(showerror, err)))
				attempts -= 1
				retry_ix = initial_attempts - attempts
				wait_s = min(5.0, 0.25 * retry_ix)
				last_error = err
				(verbosity >= 1) && @warn "KrakenFutures private transport error; retrying request" endpoint=endPoint attempts_left=attempts sleep_seconds=wait_s exception=sprint(showerror, err)
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
				if _isreadonlyprivateendpoint_futures(endPoint)
					rethrow(err)
				end
				retry_ix = initial_attempts - attempts
				wait_s = min(30.0, 2.0 ^ retry_ix)
				(verbosity >= 1) && @warn "Kraken futures private rate limit hit; retrying request" endpoint=endPoint attempts_left=attempts sleep_seconds=wait_s
				sleep(wait_s)
				continue
			end
			if _isinvalidnonceerror(err) && (attempts > 0)
				retry_ix = initial_attempts - attempts
				isdup = _isnonceduplicateerror(err)
				wait_s = isdup ? min(60.0, 2.0 ^ max(0, retry_ix - 1)) : min(1.0, 0.1 * retry_ix)
				errmsg = sprint(showerror, err)
				nonce_floor = let
					# Keep nonce recovery bounded while allowing one-way ms -> ns upshift.
					lock(_nonce_lock)
					try
						now_ms = _nonce_ms_base()
						now_ns = Int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1_000_000_000))
						if _isnoncetoosmallerror(err)
							if (_last_nonce[] < _nonce_ns_switch_threshold) && (retry_ix >= 3)
								# One-way upshift: if server floor is ns-scale, stop trying ms permanently.
								_last_nonce[] = max(now_ns, _last_nonce[] * 1_000_000)
							end
							if _last_nonce[] < now_ms
								_last_nonce[] = now_ms
							end
						end
						use_ns = _last_nonce[] >= _nonce_ns_switch_threshold
						jump = if use_ns
							isdup ?
								_capped_retry_jump(60_000_000_000_000, retry_ix, 5_000_000_000_000_000) :
								_capped_retry_jump(50_000_000_000, retry_ix, 5_000_000_000_000_000)
						else
							isdup ?
								_capped_retry_jump(600_000, retry_ix, 50_000_000_000) :
								_capped_retry_jump(50_000, retry_ix, 50_000_000_000)
						end
						base = max(use_ns ? now_ns : now_ms, _last_nonce[])
						candidate = base > (typemax(Int) - jump) ? typemax(Int) - 1 : (base + jump)
						_last_nonce[] = candidate
						_persist_nonce_floor!(bc, candidate)
						_last_nonce[]
					finally
						unlock(_nonce_lock)
					end
				end
				(verbosity >= 1) && @warn "Kraken futures invalid nonce; retrying private request" endpoint=endPoint attempts_left=attempts sleep_seconds=wait_s duplicate=isdup nonce_floor=nonce_floor error_message=errmsg
				sleep(wait_s)
				continue
			end
			rethrow(err)
		end
	end
	throw(last_error)
end

"""
Load futures instruments and map into exchange info schema.
"""
function _exchangeinfo(apirest::String, symbol=nothing)::DataFrame
	tmp = KrakenFuturesCache(_emptyexchangeinfo(), apirest, "", "")
	df = _emptyexchangeinfo()

	response = HttpPublicRequest(tmp, "GET", "/instruments", nothing, "futures instruments")
	instruments = _tryget(response, ["instruments"], Any[])
	if !(instruments isa AbstractVector)
		instruments = Any[]
	end

	for raw in instruments
		info = Dict(raw)
		pairname = String(_tryget(info, ["symbol", "instrument", "product_id"], ""))
		pairname == "" && continue

		wsname = String(_tryget(info, ["underlying", "symbol", "product_id"], pairname))
		symbolname = _ws2symbol(wsname)

		quotecoin = ""
		basecoin = ""
		for q in sort(_known_quotes, by=length, rev=true)
			if endswith(symbolname, q) && (length(symbolname) > length(q))
				quotecoin = q
				basecoin = symbolname[1:end-length(q)]
				break
			end
		end
		basecoin = _normalizeasset(basecoin)

		if (basecoin == "") || (quotecoin == "")
			continue
		end

		tradable = _tryget(info, ["tradeable", "enabled", "is_active"], true)
		status = (tradable isa Bool) ? (tradable ? "online" : "offline") : String(tradable)
		ticksize = _float32(_tryget(info, ["tickSize", "tick_size", "priceIncrement"], 0.01), 0.01f0)
		baseprecision = _float32(_tryget(info, ["contractSize", "qtyIncrement", "qty_increment"], 1.0), 1.0f0)
		minbaseqty = _float32(_tryget(info, ["minimumOrderSize", "minOrderSize", "qtyIncrement"], 0.0), 0.0f0)
		minquoteqty = _float32(_tryget(info, ["minimumOrderValue", "minOrderValue", "notionalMinimum", "notional_min"], 0.0), 0.0f0)
		maxleveragebuy = _futuresmaxleverage(info, ["maxLeverageBuy", "maxLeverageLong", "maxLeverage", "maxLeverageValue", "max_leverage", "leverageCap"])
		maxleveragesell = _futuresmaxleverage(info, ["maxLeverageSell", "maxLeverageShort", "maxLeverage", "maxLeverageValue", "max_leverage", "leverageCap"])

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
			krakenpairname=pairname,
			wsname=wsname,
		))
	end

	if size(df, 1) == 0
		# Fallback to spot metadata so interface remains usable.
		spot = HttpPublicRequest(tmp, "GET", "/0/public/AssetPairs", nothing, "spot asset pairs fallback"; baseurl=KRAKEN_SPOT_APIREST)
		result = _tryget(spot, ["result"], Dict{String, Any}())
		if result isa AbstractDict
			for (pairname, rawinfo) in result
				pairname == "last" && continue
				info = Dict(rawinfo)
				wsname = String(_tryget(info, ["wsname"], pairname))
				symbolname = _ws2symbol(wsname)
				quotecoin = ""
				basecoin = ""
				for q in sort(_known_quotes, by=length, rev=true)
					if endswith(symbolname, q) && (length(symbolname) > length(q))
						quotecoin = q
						basecoin = symbolname[1:end-length(q)]
						break
					end
				end
				basecoin = _normalizeasset(basecoin)
				if (basecoin == "") || (quotecoin == "")
					continue
				end
				ticksize = Float32(10.0^-_int(_tryget(info, ["pair_decimals"], 5), 5))
				baseprecision = Float32(10.0^-_int(_tryget(info, ["lot_decimals"], 8), 8))
				minbaseqty = _float32(_tryget(info, ["ordermin"], "0"), 0f0)
				minquoteqty = _float32(_tryget(info, ["costmin"], "0"), 0f0)
				status = String(_tryget(info, ["status"], "online"))
				push!(df, (
					symbol=symbolname,
					status=status,
					basecoin=basecoin,
					quotecoin=quotecoin,
					maxleveragebuy=0,
					maxleveragesell=0,
					ticksize=ticksize,
					baseprecision=baseprecision,
					quoteprecision=ticksize,
					minbaseqty=minbaseqty,
					minquoteqty=minquoteqty,
					krakenpairname=String(pairname),
					wsname=wsname,
				))
			end
		end
	end

	if !isnothing(symbol)
		sym = _normalizepairsymbol(String(symbol))
		return df[df.symbol .== sym, :]
	end
	return sort!(df, :symbol)
end

"""
Return cached exchange info.
"""
exchangeinfo(bc::KrakenFuturesCache, symbol=nothing) = isnothing(symbol) ? bc.syminfodf : bc.syminfodf[bc.syminfodf.symbol .== _normalizepairsymbol(String(symbol)), :]

"""
Return one symbol info row or `nothing`.
"""
function symbolinfo(bc::KrakenFuturesCache, symbol::AbstractString)::Union{Nothing, DataFrameRow}
	sym = _normalizepairsymbol(symbol)
	if isnothing(bc.syminfodf) || (size(bc.syminfodf, 1) == 0)
		return nothing
	end
	ix = findfirst(==(sym), bc.syminfodf[!, :symbol])
	return isnothing(ix) ? nothing : bc.syminfodf[ix, :]
end

"Return side-specific Kraken futures leverage caps for a symbol."
function marginlimits(bc::KrakenFuturesCache, symbol::AbstractString)
	syminfo = symbolinfo(bc, symbol)
	if isnothing(syminfo)
		return (maxleveragebuy=0, maxleveragesell=0)
	end
	return (
		maxleveragebuy=hasproperty(syminfo, :maxleveragebuy) ? Int(syminfo.maxleveragebuy) : 0,
		maxleveragesell=hasproperty(syminfo, :maxleveragesell) ? Int(syminfo.maxleveragesell) : 0,
	)
end

"Return true when futures metadata permits requested side/leverage for this symbol."
function marginpermitted(bc::KrakenFuturesCache, symbol::AbstractString, orderside::AbstractString, marginleverage::Signed)::Bool
	marginleverage <= 0 && return true
	limits = marginlimits(bc, symbol)
	side = lowercase(String(orderside))
	maxlev = side == "buy" ? limits.maxleveragebuy : limits.maxleveragesell
	# If metadata lacks leverage caps (0), keep behavior permissive and defer to exchange.
	maxlev == 0 && return true
	return maxlev >= Int(marginleverage)
end

symbolinfo(bc::KrakenFuturesCache, basecoin::AbstractString, quotecoin::AbstractString) = symbolinfo(bc, symboltoken(bc, basecoin, quotecoin))

"""
Validate one symbol row.
"""
function validsymbol(bc::KrakenFuturesCache, sym::Union{Nothing, DataFrameRow})::Bool
	if isnothing(sym)
		return false
	end
	return uppercase(sym.quotecoin) == uppercase(EnvConfig.cryptoquote) && _istradablestatus(sym.status)
end

"""
Validate symbol by name.
"""
validsymbol(bc::KrakenFuturesCache, symbol::AbstractString)::Bool = validsymbol(bc, symbolinfo(bc, symbol))
function validsymbol(bc::KrakenFuturesCache, basecoin::AbstractString, quotecoin::AbstractString)::Bool
	sym = symbolinfo(bc, basecoin, quotecoin)
	return !isnothing(sym) && (uppercase(String(sym.quotecoin)) == uppercase(quotecoin)) && _istradablestatus(sym.status)
end

"""
Return exchange server time.
"""
function servertime(bc::KrakenFuturesCache)::DateTime
	while true
		try
			response = HttpPublicRequest(bc, "GET", "/tickers", nothing, "futures server time")
			st = _tryget(response, ["serverTime", "server_time"], nothing)
			return isnothing(st) ? Dates.now(Dates.UTC) : _todatetime(st)
		catch err
			(verbosity >= 1) && @warn "KrakenFutures server time unavailable; retrying" retry_seconds=SERVERTIME_RETRY_SECONDS exception=sprint(showerror, err)
			sleep(SERVERTIME_RETRY_SECONDS)
		end
	end
end

"""
Parse one ticker object into Bybit-compatible fields.
"""
function _tickerrow(bc::KrakenFuturesCache, key::AbstractString, ticker::Dict)
	ask = _float32(_tryget(ticker, ["ask", "askPrice", "bestAsk", "ask_price"], 0), 0f0)
	bid = _float32(_tryget(ticker, ["bid", "bidPrice", "bestBid", "bid_price"], 0), 0f0)
	lastprice = _float32(_tryget(ticker, ["last", "lastPrice", "markPrice", "last_price"], 0), 0f0)
	openprice = _float32(_tryget(ticker, ["open24h", "open", "openPrice"], 0), 0f0)
	basevolume = _float32(_tryget(ticker, ["volume24h", "vol24h", "volume"], 0), 0f0)
	quotevolume = _float32(_tryget(ticker, ["turnover24h", "quoteVolume", "volumeQuote24h"], 0), basevolume * lastprice)
	pricechangepercent = openprice == 0f0 ? _float32(_tryget(ticker, ["price24hPcnt", "change"], 0), 0f0) : Float32((lastprice - openprice) / openprice)
	symbol = _resultkey2symbol(bc, key)
	return (askprice=ask, bidprice=bid, lastprice=lastprice, quotevolume24h=quotevolume, pricechangepercent=pricechangepercent, symbol=symbol)
end

"""
Return ticker info in Bybit-compatible shape.
"""
function get24h(bc::KrakenFuturesCache, symbol=nothing)
	out = DataFrame(askprice=Float32[], bidprice=Float32[], lastprice=Float32[], quotevolume24h=Float32[], pricechangepercent=Float32[], symbol=String[])

	try
		response = HttpPublicRequest(bc, "GET", "/tickers", nothing, "futures ticker")
		tickers = _tryget(response, ["tickers"], Any[])
		if tickers isa AbstractVector
			for raw in tickers
				ticker = Dict(raw)
				key = String(_tryget(ticker, ["symbol", "product_id", "instrument"], ""))
				key == "" && continue
				row = _tickerrow(bc, key, ticker)
				if isnothing(symbol) || (row.symbol == _normalizepairsymbol(String(symbol)))
					push!(out, row)
				end
			end
		end
	catch err
		(verbosity >= 1) && @warn "futures ticker request failed, trying spot-style fallback: $(err)"
	end

	if size(out, 1) == 0
		pairsdf = isnothing(symbol) ? exchangeinfo(bc) : exchangeinfo(bc, symbol)
		if !isnothing(pairsdf) && (size(pairsdf, 1) > 0)
			pairs = pairsdf[!, :krakenpairname]
			batchesize = 20
			for ix in 1:batchesize:length(pairs)
				lastix = min(ix + batchesize - 1, length(pairs))
				pairbatch = pairs[ix:lastix]
				params = Dict("pair" => join(pairbatch, ","))
				response = HttpPublicRequest(bc, "GET", "/0/public/Ticker", params, "spot ticker fallback"; baseurl=KRAKEN_SPOT_APIREST)
				result = _tryget(response, ["result"], Dict{String, Any}())
				if result isa AbstractDict
					for (key, rawticker) in result
						ticker = Dict(rawticker)
						row = _tickerrow(bc, key, ticker)
						if isnothing(symbol) || (row.symbol == _normalizepairsymbol(String(symbol)))
							push!(out, row)
						end
					end
				end
			end
		end
	end

	return isnothing(symbol) ? out : (size(out, 1) > 0 ? out[1, :] : nothing)
end

"""
Convert OHLC payload into Ohlcv-compatible DataFrame.
"""
function _convertklines(klines)::DataFrame
	df = DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])
	for row in klines
		if row isa AbstractDict
			r = Dict(row)
			push!(df, (
				opentime=_todatetime(_tryget(r, ["time", "t", "interval_begin"], Dates.now(Dates.UTC))),
				open=_float32(_tryget(r, ["open", "o"], 0), 0f0),
				high=_float32(_tryget(r, ["high", "h"], 0), 0f0),
				low=_float32(_tryget(r, ["low", "l"], 0), 0f0),
				close=_float32(_tryget(r, ["close", "c"], 0), 0f0),
				basevolume=_float32(_tryget(r, ["volume", "v"], 0), 0f0),
			))
		elseif row isa AbstractVector
			length(row) < 6 && continue
			push!(df, (
				opentime=_todatetime(row[1]),
				open=_float32(row[2]),
				high=_float32(row[3]),
				low=_float32(row[4]),
				close=_float32(row[5]),
				basevolume=_float32(row[6]),
			))
		end
	end
	return sort!(df, :opentime)
end

"""
Return OHLC candles in Ohlcv-compatible format.
"""
function getklines(bc::KrakenFuturesCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m")
	@assert interval in keys(_interval2minutes) "unknown interval=$(interval)"
	pairname = _symbol2pairname(bc, String(symbol))

	# Kraken Futures charts API: https://futures.kraken.com/api/charts/v1/trade/{symbol}/{interval}
	# Returns last 2000 candles; date filtering is done locally after fetching.
	klines = Any[]
	try
		endpoint = "/trade/$(pairname)/$(interval)"
		response = HttpPublicRequest(bc, "GET", endpoint, nothing, "futures candles"; baseurl=KRAKEN_CHARTS_APIREST)
		candles = _tryget(response, ["candles"], Any[])
		if candles isa AbstractVector
			klines = candles
		end
	catch err
		(verbosity >= 1) && @warn "futures candles request failed: $(err)"
	end

	df = _convertklines(klines)
	if !isnothing(startDateTime)
		df = df[df.opentime .>= startDateTime, :]
	end
	if !isnothing(endDateTime)
		df = df[df.opentime .<= endDateTime, :]
	end
	return df
end

"""
Return account overview.
"""
function account(bc::KrakenFuturesCache)
	if !_hascredentials(bc)
		return Dict{String, Any}()
	end
	return HttpPrivateRequest(bc, "GET", "/accounts", nothing, "futures account")
end

"""
Probe Kraken Futures private-read auth behavior with and without `Nonce`.

Returns a named tuple with booleans and error text for both modes plus a
human-readable verdict string.
"""
function startup_private_read_nonce_probe(bc::KrakenFuturesCache)
	if !_hascredentials(bc)
		return (
			with_nonce_ok=false,
			without_nonce_ok=false,
			with_nonce_error="missing credentials",
			without_nonce_error="missing credentials",
			verdict="missing_credentials",
		)
	end

	function _probe_once(omit_nonce::Bool)
		try
			withenv("KRAKEN_FUTURES_OMIT_NONCE_READS" => (omit_nonce ? "true" : "false")) do
				_ = HttpPrivateRequest(bc, "GET", "/accounts", nothing, "futures account startup probe"; max_attempts=2)
			end
			return (ok=true, err="")
		catch err
			return (ok=false, err=sprint(showerror, err))
		end
	end

	with_nonce = _probe_once(false)
	without_nonce = _probe_once(true)
	verdict = if with_nonce.ok && without_nonce.ok
		"both_modes_ok"
	elseif with_nonce.ok && !without_nonce.ok
		"nonce_required_for_private_read"
	elseif !with_nonce.ok && without_nonce.ok
		"nonce_problem_detected_read_without_nonce_ok"
	else
		"both_modes_failed"
	end

	return (
		with_nonce_ok=with_nonce.ok,
		without_nonce_ok=without_nonce.ok,
		with_nonce_error=with_nonce.err,
		without_nonce_error=without_nonce.err,
		verdict=verdict,
	)
end

"""
Convert one raw order entry into standardized row shape.
"""
function _orderrow(bc::KrakenFuturesCache, orderid::AbstractString, entry::Dict)
	descr = haskey(entry, "descr") ? Dict(entry["descr"]) : entry
	rawpair = String(_tryget(descr, ["pair", "symbol", "instrument", "product_id"], ""))
	symbol = _resultkey2symbol(bc, rawpair)
	side = lowercase(String(_tryget(descr, ["type", "side", "direction"], "buy"))) == "buy" ? "Buy" : "Sell"
	ordertype = titlecase(String(_tryget(descr, ["ordertype", "orderType", "order_type"], "limit")))
	baseqty = _float32(_tryget(entry, ["vol", "qty", "size", "orderQty"], 0), 0f0)
	executedqty = _float32(_tryget(entry, ["vol_exec", "filled", "filledSize"], 0), 0f0)
	limitprice = _float32(_tryget(descr, ["price", "limitPrice"], _tryget(entry, ["price", "limitPrice"], 0)), 0f0)
	avgprice = _float32(_tryget(entry, ["avgPrice", "fillPrice", "price"], limitprice), limitprice)
	status = titlecase(String(_tryget(entry, ["status", "orderStatus"], "open")))
	created = _todatetime(_tryget(entry, ["opentm", "created", "timestamp", "createdTime"], Dates.now(Dates.UTC)))
	updated = _todatetime(_tryget(entry, ["updated", "updatedTime", "lastUpdateTime"], created))
	leverage = _float32(_tryget(descr, ["leverage"], 1), 1f0)
	tif = String(_tryget(descr, ["timeinforce", "timeInForce"], "GTC"))
	return (
		orderid=String(orderid),
		symbol=symbol,
		side=side,
		baseqty=baseqty,
		ordertype=ordertype,
		isLeverage=leverage > 1f0,
		timeinforce=tif,
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
Return open orders in Bybit-compatible shape.
"""
function openorders(bc::KrakenFuturesCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)
	_ = orderLinkId
	out = emptyorders()
	!_hascredentials(bc) && return out
	symbolspec = isnothing(symbol) ? nothing : _normalizepairsymbol(String(symbol))

	params = Dict{String, Any}()
	!isnothing(orderid) && (params["order_id"] = String(orderid))
	response = HttpPrivateRequest(bc, "GET", "/openorders", isempty(params) ? nothing : params, "futures open orders")
	orders = _tryget(response, ["openOrders", "orders"], Any[])
	if orders isa AbstractVector
		for raw in orders
			entry = Dict(raw)
			oid = String(_tryget(entry, ["order_id", "id", "orderId", "txid"], ""))
			oid == "" && continue
			row = _orderrow(bc, oid, entry)
			if isnothing(symbolspec) || (row.symbol == symbolspec)
				push!(out, row)
			end
		end
	end
	return out
end

"""
Query one order by id.
"""
function order(bc::KrakenFuturesCache, orderid)
	if isnothing(orderid)
		return nothing
	end
	oo = openorders(bc, orderid=orderid)
	if size(oo, 1) > 0
		return oo[1, :]
	end
	!_hascredentials(bc) && return nothing

	try
		response = HttpPrivateRequest(bc, "POST", "/orders/status", Dict("orderIds" => [String(orderid)]), "futures order status")
		orders = _tryget(response, ["orders", "elements"], Any[])
		if (orders isa AbstractVector) && !isempty(orders)
			df = emptyorders()
			push!(df, _orderrow(bc, String(orderid), Dict(first(orders))))
			return df[1, :]
		end
	catch err
		(verbosity >= 1) && @warn "futures order lookup failed: $(err)"
	end
	return nothing
end

"""
Cancel one open order.
"""
function cancelorder(bc::KrakenFuturesCache, symbol, orderid)
	_ = symbol
	if isnothing(orderid) || !_hascredentials(bc)
		return nothing
	end
	try
		response = HttpPrivateRequest(bc, "POST", "/cancelorder", Dict("order_id" => String(orderid)), "futures cancel order")
		if haskey(response, "result")
			return String(orderid)
		end
	catch err
		(verbosity >= 1) && @warn "futures cancel order failed: $(err)"
	end
	return nothing
end

"""
Extract order id from mixed response payloads.
"""
function _extractorderid(response::Dict)
	if haskey(response, "order_id")
		return String(response["order_id"])
	end
	if haskey(response, "sendStatus") && (response["sendStatus"] isa AbstractDict)
		ss = Dict(response["sendStatus"])
		if haskey(ss, "order_id")
			return String(ss["order_id"])
		end
	end
	if haskey(response, "result") && (response["result"] isa AbstractDict)
		result = Dict(response["result"])
		if haskey(result, "order_id")
			return String(result["order_id"])
		end
		if haskey(result, "txid")
			txids = result["txid"]
			if (txids isa AbstractVector) && !isempty(txids)
				return String(first(txids))
			end
		end
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
Create one order and return a standardized named tuple.

If `price` is omitted and `maker=true`, the adapter will choose a limit price
as close as possible to the current spread while remaining post-only so the
order can qualify for maker fees.
"""
function createorder(bc::KrakenFuturesCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0)
	@assert basequantity > 0.0 "createorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
	@assert isnothing(price) || (price > 0.0) "createorder symbol=$(symbol) price=$(price) must be > 0"
	@assert lowercase(orderside) in ["buy", "sell"] "createorder symbol=$(symbol) orderside=$(orderside) must be Buy or Sell"
	!_hascredentials(bc) && return nothing
	_validatemarginleverage(marginleverage)

	syminfo = symbolinfo(bc, symbol)
	if isnothing(syminfo)
		(verbosity >= 1) && @warn "no instrument info for $(symbol)"
		return nothing
	end
	if !_istradablestatus(syminfo.status)
		(verbosity >= 1) && @warn "symbol $(symbol) is not tradable due to status=$(syminfo.status)"
		return nothing
	end

	pairname = _symbol2pairname(bc, symbol)
	if marginleverage > 0
		limits = marginlimits(bc, symbol)
		if !marginpermitted(bc, symbol, orderside, marginleverage)
			throw(ErrorException("Kraken futures leverage not permitted for symbol=$(symbol) pair=$(pairname) side=$(orderside) requested_leverage=$(marginleverage)x max_buy=$(limits.maxleveragebuy)x max_sell=$(limits.maxleveragesell)x status=$(syminfo.status)"))
		end
	end
	adaptivepost = maker && isnothing(price)
	attempts = adaptivepost ? 5 : 1
	ordertype = (maker || !isnothing(price)) ? "lmt" : "mkt"
	chosenqty = Float32(basequantity)
	effectiveprice = isnothing(price) ? nothing : Float32(price)
	orderid = nothing
	while attempts > 0
		if ordertype == "lmt"
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
			"symbol" => pairname,
			"side" => lowercase(orderside),
			"size" => string(chosenqty),
			"orderType" => ordertype,
		)
		(ordertype == "lmt") && (params["limitPrice"] = string(effectiveprice))
		(maker && (ordertype == "lmt")) && (params["postOnly"] = "true")
		marginleverage > 0 && (params["leverage"] = string(marginleverage))

		try
			response = HttpPrivateRequest(bc, "POST", "/sendorder", params, "futures create order")
			orderid = _extractorderid(response)
			!isnothing(orderid) && break
			return nothing
		catch err
			attempts -= 1
			if !adaptivepost || !_ispostonlyrejection(err) || (attempts <= 0)
				rethrow(err)
			end
			(verbosity >= 2) && @info "retrying futures post-only order for $(symbol) after rejection" attempts_left=attempts
		end
	end

	isnothing(orderid) && return nothing
	created = Dates.now(Dates.UTC)
	return (
		orderid=String(orderid),
		symbol=_normalizepairsymbol(symbol),
		side=lowercase(orderside) == "buy" ? "Buy" : "Sell",
		baseqty=chosenqty,
		ordertype=isnothing(effectiveprice) ? "Market" : "Limit",
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
Amend one open order.
"""
function amendorder(bc::KrakenFuturesCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
	@assert isnothing(basequantity) || (basequantity > 0.0) "amendorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
	@assert isnothing(limitprice) || (limitprice > 0.0) "amendorder symbol=$(symbol) limitprice=$(limitprice) must be > 0"

	current = order(bc, orderid)
	isnothing(current) && return nothing

	qty = isnothing(basequantity) ? current.baseqty : Float32(basequantity)
	prc = current.limitprice
	pricechanged = false
	if current.timeinforce == "PostOnly"
		snapshot = get24h(bc, symbol)
		if !isnothing(snapshot)
			prc = _makerlimitprice(symbolinfo(bc, symbol), snapshot, current.side)
			pricechanged = !isapprox(Float64(prc), Float64(current.limitprice); atol=0.0, rtol=0.0)
		end
	elseif !isnothing(limitprice)
		prc = Float32(limitprice)
		pricechanged = !isapprox(Float64(prc), Float64(current.limitprice); atol=0.0, rtol=0.0)
	end
	if qty == current.baseqty && !pricechanged
		return current
	end

	try
		params = Dict{String, Any}("order_id" => orderid)
		!isnothing(basequantity) && (params["size"] = string(basequantity))
		!isnothing(limitprice) && (params["limitPrice"] = string(limitprice))
		_ = HttpPrivateRequest(bc, "POST", "/editorder", params, "futures amend order")
		amended = order(bc, orderid)
		return isnothing(amended) ? nothing : amended
	catch
	end

	cancelled = cancelorder(bc, symbol, orderid)
	isnothing(cancelled) && return nothing
	recreated = createorder(bc, symbol, current.side, qty, prc, current.timeinforce == "PostOnly")
	isnothing(recreated) && return nothing
	return (recreated..., status="Replaced", rejectreason=string("Replaced order ", orderid))
end

"""
Return balances in normalized shape.

Parses the Kraken Futures `/accounts` response which has a nested Dict structure:
- `"cash"` account: Dict of currency => balance (free spot-style balances)
- `"flex"` account: collateral account with `availableMargin`, `initialMargin`, `currencies` fields
- `"fi_*"` accounts: per-perpetual-market margin accounts (balances tracked inside `balances` Dict)

The function aggregates cash account balances and adds flex margin info.
"""
function balances(bc::KrakenFuturesCache)
	df = DataFrame(coin=AbstractString[], locked=Float32[], free=Float32[], borrowed=Float32[], accruedinterest=Float32[])
	!_hascredentials(bc) && return df

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

	acct = account(bc)
	!(acct isa AbstractDict) && return df
	ad = Dict(acct)
	accounts = _tryget(ad, ["accounts"], Dict())
	!(accounts isa AbstractDict) && return df

	# Cash account: direct coin => balance mapping
	if haskey(accounts, "cash")
		cash = Dict(accounts["cash"])
		balmap = _tryget(cash, ["balances"], Dict())
		if balmap isa AbstractDict
			for (coin, val) in balmap
				free = _float32(val, 0f0)
				free == 0f0 && continue
				push!(df, (coin=_normalizeasset(uppercase(String(coin))), locked=0f0, free=free, borrowed=0f0, accruedinterest=0f0))
			end
		end
	end

	# Flex collateral account: report aggregate USD collateral value
	if haskey(accounts, "flex")
		flex = Dict(accounts["flex"])
		available = _float32(_tryget(flex, ["availableMargin", "marginEquity"], 0), 0f0)
		locked = _float32(_tryget(flex, ["initialMargin", "initialMarginWithOrders"], 0), 0f0)
		total = available + locked
		if total > 0f0
			push!(df, (coin="USD", locked=locked, free=available, borrowed=0f0, accruedinterest=0f0))
		end
	end

	# Cache the result for 5 seconds
	lock(_balance_cache_lock) do
		_balance_cache[] = copy(df)
		_balance_cache_time[] = Dates.now(UTC)
	end

	return df
end

"""
Extract first websocket data object.
"""
_firstwsdata(data) = (data isa AbstractVector ? (isempty(data) ? nothing : first(data)) : data)

"""
REST polling fallback for ticker stream.
"""
function _polltickerfallback!(channel::Channel{Dict}, bc::KrakenFuturesCache, symbol::String)
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
Subscribe to futures ticker stream, then fall back to REST polling.
"""
function ws_ticker(bc::KrakenFuturesCache, symbol::String)
	channel = Channel{Dict}(32)
	@async begin
		wsok = false
		try
			wssymbol = _symbol2pairname(bc, symbol)
			subscribe = Dict("event" => "subscribe", "feed" => "ticker", "product_ids" => [wssymbol])
			WebSockets.open(KRAKEN_FUTURES_WS_PUBLIC) do ws
				wsok = true
				WebSockets.send(ws, JSON3.write(subscribe))
				while isopen(ws) && isopen(channel)
					msgraw = WebSockets.receive(ws)
					!(msgraw isa String) && continue
					msg = JSON3.read(msgraw, Dict)
					if get(msg, "feed", "") == "ticker"
						p = Dict(msg)
						wskey = String(_tryget(p, ["product_id", "symbol"], wssymbol))
						put!(channel, Dict(
							"symbol" => _resultkey2symbol(bc, wskey),
							"askprice" => _float32(_tryget(p, ["ask", "askPrice"], 0), 0f0),
							"bidprice" => _float32(_tryget(p, ["bid", "bidPrice"], 0), 0f0),
							"lastprice" => _float32(_tryget(p, ["last", "lastPrice", "markPrice"], 0), 0f0),
							"source" => "ws",
						))
					end
				end
			end
		catch err
			(verbosity >= 1) && @warn "KrakenFutures ws_ticker failed for $(symbol): $(err)"
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
REST polling fallback for kline stream.
"""
function _pollklinefallback!(channel::Channel{Dict}, bc::KrakenFuturesCache, symbol::String, interval::String)
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
Subscribe to futures candle stream, then fall back to REST polling.
"""
function ws_kline(bc::KrakenFuturesCache, symbol::String, interval::String="1m")
	@assert interval in keys(_interval2minutes) "unknown interval=$(interval)"
	channel = Channel{Dict}(32)
	@async begin
		wsok = false
		try
			wssymbol = _symbol2pairname(bc, symbol)
			subscribe = Dict("event" => "subscribe", "feed" => "candles", "product_ids" => [wssymbol], "interval" => _interval2minutes[interval])
			WebSockets.open(KRAKEN_FUTURES_WS_PUBLIC) do ws
				wsok = true
				WebSockets.send(ws, JSON3.write(subscribe))
				while isopen(ws) && isopen(channel)
					msgraw = WebSockets.receive(ws)
					!(msgraw isa String) && continue
					msg = JSON3.read(msgraw, Dict)
					if get(msg, "feed", "") == "candles"
						p = Dict(msg)
						wskey = String(_tryget(p, ["product_id", "symbol"], wssymbol))
						put!(channel, Dict(
							"symbol" => _resultkey2symbol(bc, wskey),
							"opentime" => _todatetime(_tryget(p, ["time", "timestamp"], Dates.now(Dates.UTC))),
							"open" => _float32(_tryget(p, ["open"], 0), 0f0),
							"high" => _float32(_tryget(p, ["high"], 0), 0f0),
							"low" => _float32(_tryget(p, ["low"], 0), 0f0),
							"close" => _float32(_tryget(p, ["close"], 0), 0f0),
							"basevolume" => _float32(_tryget(p, ["volume"], 0), 0f0),
							"source" => "ws",
						))
					end
				end
			end
		catch err
			(verbosity >= 1) && @warn "KrakenFutures ws_kline failed for $(symbol): $(err)"
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
Sign websocket challenge for private futures subscriptions.
"""
function _wssignedchallenge(secret::String, challenge::String)::String
	decoded = try
		Base64.base64decode(secret)
	catch
		Vector{UInt8}(secret)
	end
	return Base64.base64encode(_hmac(decoded, Vector{UInt8}(challenge), SHA.sha512, 128))
end

"""
REST polling fallback for order updates.
"""
function _pollordersfallback!(channel::Channel{Dict}, bc::KrakenFuturesCache)
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
Subscribe to private order updates via websocket, then fall back to REST polling.
"""
function ws_orders(bc::KrakenFuturesCache)
	channel = Channel{Dict}(32)
	@async begin
		wsok = false
		if _hascredentials(bc)
			try
				WebSockets.open(KRAKEN_FUTURES_WS_PRIVATE) do ws
					# Request challenge and subscribe with signed challenge.
					WebSockets.send(ws, JSON3.write(Dict("event" => "challenge", "api_key" => bc.publickey)))
					challraw = WebSockets.receive(ws)
					challenge = nothing
					if challraw isa String
						chall = JSON3.read(challraw, Dict)
						challenge = haskey(chall, "message") ? String(chall["message"]) : nothing
					end
					if isnothing(challenge)
						throw(ErrorException("missing websocket challenge"))
					end
					subscribe = Dict(
						"event" => "subscribe",
						"feed" => "open_orders",
						"api_key" => bc.publickey,
						"original_challenge" => challenge,
						"signed_challenge" => _wssignedchallenge(bc.secretkey, challenge),
					)
					wsok = true
					WebSockets.send(ws, JSON3.write(subscribe))
					while isopen(ws) && isopen(channel)
						msgraw = WebSockets.receive(ws)
						!(msgraw isa String) && continue
						msg = JSON3.read(msgraw, Dict)
						if get(msg, "feed", "") == "open_orders"
							put!(channel, Dict("topic" => "open_orders", "source" => "ws", "data" => get(msg, "orders", Any[])))
						end
					end
				end
			catch err
				(verbosity >= 1) && @warn "KrakenFutures ws_orders failed: $(err)"
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
