module KrakenSpot

using Base64, DataFrames, Dates, EnvConfig, HTTP, JSON3, Logging, SHA, WebSockets

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

"""
Build the DataFrame schema used by `exchangeinfo`.
"""
function _emptyexchangeinfo()::DataFrame
	return DataFrame(
		symbol=String[],
		status=String[],
		basecoin=String[],
		quotecoin=String[],
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
Build the DataFrame schema used by `openorders` and `order`.
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
	response = HTTP.request(method, url; retries=5, retry=true, readtimeout=60)
	body = JSON3.read(String(response.body), Dict)
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

	reqparams = isnothing(params) ? Dict{String, Any}() : copy(params)
	nonce = string(_int(round(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1000)))
	reqparams["nonce"] = nonce
	postdata = _dict2paramspost(reqparams)
	signature = _krakensignature(endPoint, nonce, postdata, bc.secretkey)

	headers = Dict(
		"API-Key" => bc.publickey,
		"API-Sign" => signature,
		"Content-Type" => "application/x-www-form-urlencoded",
	)
	response = HTTP.request("POST", bc.apirest * endPoint, headers, postdata; retries=3, retry_non_idempotent=true, readtimeout=60)
	body = JSON3.read(String(response.body), Dict)
	_checkresponse(body, info)
	return body
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
		status = String(get(info, "status", "online"))

		push!(df, (
			symbol=symbolname,
			status=status,
			basecoin=basecoin,
			quotecoin=quotecoin,
			ticksize=ticksize,
			baseprecision=baseprecision,
			quoteprecision=ticksize,
			minbaseqty=minbaseqty,
			minquoteqty=0f0,
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

"""
Validate one symbol info row according to current quote and trading status constraints.
"""
function validsymbol(bc::KrakenSpotCache, sym::Union{Nothing, DataFrameRow})::Bool
	if isnothing(sym)
		return false
	end
	statuses = ["online", "post_only", "limit_only", "reduce_only", "trading"]
	return uppercase(sym.quotecoin) == uppercase(EnvConfig.cryptoquote) && (lowercase(sym.status) in statuses)
end

"""
Validate one symbol string according to current quote and trading status constraints.
"""
validsymbol(bc::KrakenSpotCache, symbol::AbstractString)::Bool = validsymbol(bc, symbolinfo(bc, symbol))

"""
Return Kraken server time in UTC.
"""
function servertime(bc::KrakenSpotCache)::DateTime
	response = HttpPublicRequest(bc, "GET", "/0/public/Time", nothing, "server time")
	unixtime = _int(response["result"]["unixtime"])
	return Dates.unix2datetime(unixtime)
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
	leverage = _float32(get(descr, "leverage", "0"), 0f0)
	return (
		orderid=String(orderid),
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
	_ = orderLinkId
	if !_hascredentials(bc)
		return emptyorders()
	end
	params = Dict{String, Any}()
	!isnothing(orderid) && (params["txid"] = String(orderid))
	response = HttpPrivateRequest(bc, "POST", "/0/private/OpenOrders", params, "open orders")
	result = get(response, "result", Dict{String, Any}())
	open = haskey(result, "open") ? Dict(result["open"]) : Dict{String, Any}()

	out = emptyorders()
	symbolspec = isnothing(symbol) ? nothing : _normalizepairsymbol(String(symbol))
	for (oid, rawentry) in open
		entry = Dict(rawentry)
		row = _orderrow(bc, oid, entry)
		if !isnothing(symbolspec) && (row.symbol != symbolspec)
			continue
		end
		push!(out, row)
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
	return count > 0 ? String(orderid) : nothing
end

"""
Create one spot order and return an order row compatible named tuple.
"""
function createorder(bc::KrakenSpotCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0)
	@assert basequantity > 0.0 "createorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
	@assert isnothing(price) || (price > 0.0) "createorder symbol=$(symbol) price=$(price) must be > 0"
	@assert lowercase(orderside) in ["buy", "sell"] "createorder symbol=$(symbol) orderside=$(orderside) must be Buy or Sell"
	if !_hascredentials(bc)
		return nothing
	end

	pairname = _symbol2pairname(bc, symbol)
	ordertype = isnothing(price) ? "market" : "limit"
	params = Dict{String, Any}(
		"pair" => pairname,
		"type" => lowercase(orderside),
		"ordertype" => ordertype,
		"volume" => string(basequantity),
	)
	if !isnothing(price)
		params["price"] = string(price)
	end
	if maker && !isnothing(price)
		params["oflags"] = "post"
	end
	if marginleverage > 0
		params["leverage"] = string(marginleverage)
	end

	response = HttpPrivateRequest(bc, "POST", "/0/private/AddOrder", params, "create order")
	result = get(response, "result", Dict{String, Any}())
	txids = get(result, "txid", Any[])
	if !(txids isa AbstractVector) || isempty(txids)
		return nothing
	end

	orderid = String(first(txids))
	created = Dates.now(Dates.UTC)
	return (
		orderid=orderid,
		symbol=_normalizepairsymbol(symbol),
		side=lowercase(orderside) == "buy" ? "Buy" : "Sell",
		baseqty=Float32(basequantity),
		ordertype=titlecase(ordertype),
		timeinforce=(maker && !isnothing(price)) ? "PostOnly" : "GTC",
		limitprice=isnothing(price) ? 0f0 : Float32(price),
		avgprice=0f0,
		executedqty=0f0,
		status="New",
		created=created,
		updated=created,
		rejectreason="",
	)
end

"""
Amend one open order by canceling and recreating it with new values.
"""
function amendorder(bc::KrakenSpotCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
	@assert isnothing(basequantity) || (basequantity > 0.0) "amendorder symbol=$(symbol) basequantity=$(basequantity) must be > 0"
	@assert isnothing(limitprice) || (limitprice > 0.0) "amendorder symbol=$(symbol) limitprice=$(limitprice) must be > 0"

	current = order(bc, orderid)
	if isnothing(current)
		return nothing
	end

	qty = isnothing(basequantity) ? current.baseqty : Float32(basequantity)
	price = isnothing(limitprice) ? current.limitprice : Float32(limitprice)
	maker = current.timeinforce == "PostOnly"

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
function balances(bc::KrakenSpotCache)
	df = DataFrame(coin=AbstractString[], locked=Float32[], free=Float32[], borrowed=Float32[], accruedinterest=Float32[])
	if !_hascredentials(bc)
		return df
	end

	response = nothing
	try
		response = HttpPrivateRequest(bc, "POST", "/0/private/BalanceEx", nothing, "balanceex")
	catch
		response = HttpPrivateRequest(bc, "POST", "/0/private/Balance", nothing, "balance")
	end
	result = get(response, "result", Dict{String, Any}())

	for (asset, value) in result
		coin = _normalizeasset(String(asset))
		if value isa AbstractDict
			balance = _float32(get(value, "balance", "0"), 0f0)
			locked = _float32(get(value, "hold_trade", "0"), 0f0)
			free = balance - locked
			push!(df, (coin=coin, locked=locked, free=free, borrowed=0f0, accruedinterest=0f0))
		else
			free = _float32(value, 0f0)
			push!(df, (coin=coin, locked=0f0, free=free, borrowed=0f0, accruedinterest=0f0))
		end
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
