module XchAdapter

using Dates

"Defines the shared exchange-cache interface root type used across adapters and Xch."
abstract type XchAdapterCache end

function _required_method_error(ac::XchAdapterCache, methodname::Symbol)
	throw(ArgumentError("adapter type $(typeof(ac)) must implement $(methodname)"))
end

rawcache(ac::XchAdapterCache) = ac

symbolinfo(ac::XchAdapterCache, symbol) = _required_method_error(ac, :symbolinfo)
validsymbol(ac::XchAdapterCache, sym) = _required_method_error(ac, :validsymbol)
getklines(ac::XchAdapterCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m") = _required_method_error(ac, :getklines)
get24h(ac::XchAdapterCache) = _required_method_error(ac, :get24h)
get24h(ac::XchAdapterCache, symbol) = _required_method_error(ac, :get24h)
balances(ac::XchAdapterCache) = _required_method_error(ac, :balances)
openorders(ac::XchAdapterCache; symbol=nothing, orderid=nothing, orderLinkId=nothing) = _required_method_error(ac, :openorders)
order(ac::XchAdapterCache, orderid) = _required_method_error(ac, :order)
cancelorder(ac::XchAdapterCache, symbol, orderid) = _required_method_error(ac, :cancelorder)
createorder(ac::XchAdapterCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0, reduceonly::Bool=false) = _required_method_error(ac, :createorder)
amendorder(ac::XchAdapterCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing) = _required_method_error(ac, :amendorder)
amendorder(ac::XchAdapterCache, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing) = _required_method_error(ac, :amendorder)
servertime(ac::XchAdapterCache) = _required_method_error(ac, :servertime)
symboltoken(ac::XchAdapterCache, basecoin::AbstractString, quotecoin::AbstractString) = _required_method_error(ac, :symboltoken)

marginlimits(ac::XchAdapterCache, symbol::AbstractString) = (maxleveragebuy=0, maxleveragesell=0)
marginpermitted(ac::XchAdapterCache, symbol::AbstractString, orderside::AbstractString, marginleverage::Signed) = true
marketdataheartbeats(ac::XchAdapterCache) = Dict{String, DateTime}()
marketdataheartbeat(ac::XchAdapterCache; symbol::Union{Nothing, AbstractString}=nothing) = nothing
accountcapacity(ac::XchAdapterCache) = nothing
closeorder(ac::XchAdapterCache, symbol::String, side::Symbol, basequantity, limitprice, maker::Bool; marginleverage::Signed=0, reduceonly::Bool=true) = nothing
wsclosedkline(ac::XchAdapterCache, symbol::AbstractString, interval::AbstractString) = nothing

end
