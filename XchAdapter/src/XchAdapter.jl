module XchAdapter

using Dates

"Defines the shared exchange-cache interface root type used across adapters and Xch."
abstract type XchAdapterCache end

"Reference to one Trade configuration row during a prepared trading-pair epoch."
struct TradingPairRef
	pair::String
	cfgindex::UInt
	epoch::UInt
end

function _required_method_error(ac::XchAdapterCache, methodname::Symbol)
	throw(ArgumentError("adapter type $(typeof(ac)) must implement $(methodname)"))
end

rawcache(ac::XchAdapterCache) = ac
exchangeid(ac::XchAdapterCache) = _required_method_error(ac, :exchangeid)

symbolinfo(ac::XchAdapterCache, symbol) = _required_method_error(ac, :symbolinfo)
validsymbol(ac::XchAdapterCache, sym) = _required_method_error(ac, :validsymbol)
getklines(ac::XchAdapterCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m") = _required_method_error(ac, :getklines)
get24h(ac::XchAdapterCache) = _required_method_error(ac, :get24h)
get24h(ac::XchAdapterCache, symbol) = _required_method_error(ac, :get24h)
balances(ac::XchAdapterCache) = _required_method_error(ac, :balances)
positionsnapshot(ac::XchAdapterCache) = _required_method_error(ac, :positionsnapshot)
"Return an atomic `(balances, positions)` adapter snapshot when supported, otherwise `nothing`."
accountsnapshot(ac::XchAdapterCache) = nothing
emptyorders(ac::XchAdapterCache) = _required_method_error(ac, :emptyorders)

openorders(ac::XchAdapterCache; symbol=nothing, orderid=nothing, orderLinkId=nothing) = _required_method_error(ac, :openorders)

order(ac::XchAdapterCache, orderid) = _required_method_error(ac, :order)

cancelorder(ac::XchAdapterCache, symbol, orderid) = _required_method_error(ac, :cancelorder)

createorder(ac::XchAdapterCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; reduceonly::Bool=false, adaptivepost::Union{Nothing, Bool}=nothing) = _required_method_error(ac, :createorder)

amendorder(ac::XchAdapterCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing) = _required_method_error(ac, :amendorder)

amendorder(ac::XchAdapterCache, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing) = _required_method_error(ac, :amendorder)

servertime(ac::XchAdapterCache) = _required_method_error(ac, :servertime)
symboltoken(ac::XchAdapterCache, basecoin::AbstractString, quotecoin::AbstractString) = _required_method_error(ac, :symboltoken)
executionorderspec(ac::XchAdapterCache, side::Symbol) = _required_method_error(ac, :executionorderspec)
preparetradingpairs!(ac::XchAdapterCache, pairrefs::Vector{TradingPairRef}) = nothing

"""
Normalize a raw adapter order status into Xch status vocabulary.

Known normalized values are typically `none`, `submitted`, `closed`, `cancelled`,
and `rejected`. Unknown values fall back to lowercase passthrough.
"""
normalize_order_status(ac::XchAdapterCache, rawstatus::AbstractString)::String = lowercase(String(rawstatus))

marginlimits(ac::XchAdapterCache, symbol::AbstractString) = (maxleveragebuy=0, maxleveragesell=0)
marginpermitted(ac::XchAdapterCache, symbol::AbstractString, orderside::AbstractString, marginleverage::Signed) = true
marketdataheartbeats(ac::XchAdapterCache) = Dict{String, DateTime}()
marketdataheartbeat(ac::XchAdapterCache; symbol::Union{Nothing, AbstractString}=nothing) = nothing
wsorderssnapshot(ac::XchAdapterCache) = nothing
wsordersheartbeat(ac::XchAdapterCache) = nothing
wsbalancessnapshot(ac::XchAdapterCache) = nothing
wsbalancesheartbeat(ac::XchAdapterCache) = nothing
ws_orders(ac::XchAdapterCache) = nothing
ws_balances(ac::XchAdapterCache) = nothing
accountcapacity(ac::XchAdapterCache) = nothing
closeorder(ac::XchAdapterCache, symbol::String, side::Symbol, basequantity, limitprice, maker::Bool; reduceonly::Bool=true, adaptivepost::Union{Nothing, Bool}=nothing) = nothing

"upsert = update existing or insert new close order. `lane` names the trades lane (`lc`/`lcsl`/`sc`/`scsl`) so adapters can pair the two legs of a close bracket."
upsertcloseorder!(ac::XchAdapterCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=true, lane::Union{Nothing, AbstractString}=nothing, pairref::Union{Nothing, TradingPairRef}=nothing, adaptivepost::Union{Nothing, Bool}=nothing) = _required_method_error(ac, :upsertcloseorder!)

"upsert = update existing or insert new open order"
upsertopenorder!(ac::XchAdapterCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=false, lane::Union{Nothing, AbstractString}=nothing, pairref::Union{Nothing, TradingPairRef}=nothing, adaptivepost::Union{Nothing, Bool}=nothing) = _required_method_error(ac, :upsertopenorder!)

"ensure order sequence: predecessor order must be submitted before successor order is submitted"
directsequence!(ac::XchAdapterCache, predecessor_orderid::AbstractString, successor_orderid::AbstractString) = _required_method_error(ac, :directsequence!)

" returns the last closed kline as NamedTuple (opentime::DateTime, open::Float32, high::Float32, low::Float32, close::Float32, basevolume::Float32) for a given symbol and interval, or nothing if not available "
wsclosedkline(ac::XchAdapterCache, symbol::AbstractString, interval::AbstractString) = nothing

"Return and clear queued forced-close events (margin-call liquidation or exchange-side stop) since the last drain. Default: no adapter-detected forced closes."
drainliquidations!(ac::XchAdapterCache) = NamedTuple[]


end
