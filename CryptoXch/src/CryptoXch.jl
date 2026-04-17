# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])


module CryptoXch

using Dates, DataFrames, DataAPI, JDF, CSV, Logging, InlineStrings
using Bybit, EnvConfig, KrakenFutures, KrakenSpot, Ohlcv, TestOhlcv
import Ohlcv: intervalperiod

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

@enum Sidefactor buy=1 sell=-1 invaid = 0
@enum SimMode nosimulation bybitsim cryptoxchsim

const EXCHANGE_BYBIT::String = "Bybit"
const EXCHANGE_KRAKENFUTURES::String = "KrakenFutures"
const EXCHANGE_KRAKENSPOT::String = "KrakenSpot"

function _normalizeexchange(exchange::AbstractString)::String
    ex = lowercase(strip(exchange))
    if ex == lowercase(EXCHANGE_BYBIT)
        return EXCHANGE_BYBIT
    elseif ex == lowercase(EXCHANGE_KRAKENFUTURES)
        return EXCHANGE_KRAKENFUTURES
    elseif ex == lowercase(EXCHANGE_KRAKENSPOT)
        return EXCHANGE_KRAKENSPOT
    end
    throw(ArgumentError("unsupported exchange=$(exchange), supported=[$(EXCHANGE_BYBIT), $(EXCHANGE_KRAKENFUTURES), $(EXCHANGE_KRAKENSPOT)]"))
end

function _exchangeModule(exchange::AbstractString)
    ex = _normalizeexchange(exchange)
    if ex == EXCHANGE_BYBIT
        return Bybit
    elseif ex == EXCHANGE_KRAKENFUTURES
        return KrakenFutures
    end
    return KrakenSpot
end

_exchangeemptyorders(exchange::AbstractString)::DataFrame = _exchangeModule(exchange).emptyorders()

function _exchangefromauthname(authname::AbstractString)::Union{Nothing, String}
    n = lowercase(strip(authname))
    if occursin("krakenfutures", n) || occursin("kraken-futures", n)
        return EXCHANGE_KRAKENFUTURES
    elseif occursin("krakenspot", n) || occursin("kraken-spot", n)
        return EXCHANGE_KRAKENSPOT
    elseif occursin("bybit", n)
        return EXCHANGE_BYBIT
    end
    return nothing
end

function _authfromname(authname::Union{Nothing, AbstractString})
    if isnothing(authname)
        return EnvConfig.authorization
    end
    return EnvConfig.Authentication(String(authname))
end

function _exchangecache(exchange::AbstractString, simmode::SimMode, authname::Union{Nothing, AbstractString}=nothing)
    ex = _normalizeexchange(exchange)
    auth = _authfromname(authname)
    publickey = isnothing(auth) ? "" : String(auth.key)
    secretkey = isnothing(auth) ? "" : String(auth.secret)
    if ex == EXCHANGE_BYBIT
        return Bybit.BybitCache(simmode == bybitsim, publickey, secretkey)
    elseif ex == EXCHANGE_KRAKENFUTURES
        (simmode == bybitsim) && @warn "simmode=$(simmode) is Bybit-specific and ignored for exchange=$(ex)"
        return KrakenFutures.KrakenFuturesCache(publickey=publickey, secretkey=secretkey)
    end
    (simmode == bybitsim) && @warn "simmode=$(simmode) is Bybit-specific and ignored for exchange=$(ex)"
    return KrakenSpot.KrakenSpotCache(publickey=publickey, secretkey=secretkey)
end

mutable struct XchCache
    orders  # ::DataFrame
    closedorders  # ::DataFrame
    assets  # :: DataFrame
    bases  # ::Dict{String, Ohlcv.OhlcvData}
    bc  # exchange specific cache, e.g. Bybit.BybitCache or KrakenSpot.KrakenSpotCache
    exchange::String
    authname::Union{Nothing, String}
    feerate  # 0.001 = 0.1% maker/taker fee by default  #TODO store exchange info and account fee rate and use it in offline backtest simulation
    startdt::Dates.DateTime
    currentdt::Union{Nothing, Dates.DateTime}  # current back testing time
    enddt::Union{Nothing, Dates.DateTime}  # end time back testing; nothing == request life data without defined termination
    mnemonic  # String or nothing
    mc::Dict # MC = module constants
    function XchCache(;startdt::DateTime=Dates.now(UTC), enddt=nothing, mnemonic=nothing, exchange::String=EXCHANGE_BYBIT, authname::Union{Nothing, AbstractString}=nothing)
        startdt = floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        # simmode = cryptoxchsim # does not call Bybit for account specific functions but uses CryptoXch simulation
        # simmode = bybitsim # does not use CryptoXch simulation but Bybit Testnet simulation
        # simmode = nosimulation # uses production mode of Bybit without any exchange simulation
        if !isnothing(authname)
            inferred = _exchangefromauthname(String(authname))
            if !isnothing(inferred) && (exchange != inferred)
                (verbosity >= 2) && @info "XchCache exchange $(exchange) overridden by authname=$(authname) -> $(inferred)"
                exchange = inferred
            end
        end
        exchange = _normalizeexchange(exchange)
        authname = isnothing(authname) ? nothing : String(authname)
        simmode = EnvConfig.configmode == production ? simmode = nosimulation : simmode = cryptoxchsim
        xc = new(_emptyorders(exchange), _emptyorders(exchange), _emptyassets(), Dict(), _exchangecache(exchange, simmode, authname), exchange, authname, 0.001, startdt, nothing, enddt, mnemonic, Dict())
        xc.mc[:simmode] = simmode
        return xc
    end
end

exchange(xc::XchCache)::String = xc.exchange
authname(xc::XchCache) = xc.authname
_exchangeModule(xc::XchCache) = _exchangeModule(xc.exchange)

_exchangeservertime(xc::XchCache) = _exchangeModule(xc).servertime(xc.bc)
_exchangesymbolinfo(xc::XchCache, symbol) = _exchangeModule(xc).symbolinfo(xc.bc, symbol)
_exchangevalidsymbol(xc::XchCache, sym) = _exchangeModule(xc).validsymbol(xc.bc, sym)
_exchangegetklines(xc::XchCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m") = _exchangeModule(xc).getklines(xc.bc, symbol; startDateTime=startDateTime, endDateTime=endDateTime, interval=interval)
_exchangeget24h(xc::XchCache) = _exchangeModule(xc).get24h(xc.bc)
_exchangebalances(xc::XchCache) = _exchangeModule(xc).balances(xc.bc)
_exchangeopenorders(xc::XchCache; symbol=nothing, orderid=nothing, orderLinkId=nothing) = _exchangeModule(xc).openorders(xc.bc; symbol=symbol, orderid=orderid, orderLinkId=orderLinkId)
_exchangeorder(xc::XchCache, orderid) = _exchangeModule(xc).order(xc.bc, orderid)
_exchangecancelorder(xc::XchCache, symbol, orderid) = _exchangeModule(xc).cancelorder(xc.bc, symbol, orderid)
_exchangecreateorder(xc::XchCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0) = _exchangeModule(xc).createorder(xc.bc, symbol, orderside, basequantity, price, maker, marginleverage=marginleverage)
_exchangeamendorder(xc::XchCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing) = _exchangeModule(xc).amendorder(xc.bc, symbol, orderid; basequantity=basequantity, limitprice=limitprice)

setstartdt(xc::XchCache, dt::DateTime) = (xc.startdt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
setenddt(xc::XchCache, dt::DateTime) = (xc.enddt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
bases(xc::XchCache) = keys(xc.bases)
ohlcv(xc::XchCache) = values(xc.bases)
ohlcv(xc::XchCache, base::AbstractString) = xc.bases[base]
baseohlcvdict(xc::XchCache) = xc.bases

basenottradable = ["MATIC", "FTM"]
basestablecoin = ["USD", "USD1", "USDT", "TUSD", "BUSD", "USDC", "USDE", "EUR", "DAI"]
quotecoins = ["USDT"]  # , "USDC"]
baseignore = uppercase.(union(basestablecoin, basenottradable))
minimumquotevolume = 10  # USDT

MAXLIMITDELTA = 0.1

_isleveraged(token) = !isnothing(token) && (length(token) > 2) && (token[end] in ['S', 'L']) && isdigit(token[end-1])

#region support

validbase(xc::XchCache, base::AbstractString) = validsymbol(xc, symboltoken(base))

removebase!(xc::XchCache, base) = delete!(xc.bases, base)
removeallbases(xc::XchCache) = xc.bases = Dict()

function addbase!(xc::XchCache, ohlcv::Ohlcv.OhlcvData)
    xc.bases[ohlcv.base] = ohlcv
    setcurrenttime!(xc, ohlcv.base, isnothing(xc.currentdt) ? xc.startdt : xc.currentdt)
end

function addbase!(xc::XchCache, base, startdt, enddt)
    base = String(base)
    enddt = isnothing(enddt) ? floor(Dates.now(UTC), Minute(1)) : floor(enddt, Minute(1))
    startdt = isnothing(startdt) ? enddt : floor(startdt, Minute(1))
    ohlcv = cryptodownload(xc, base, "1m", startdt, enddt)
    ohlcv.ix = firstindex(ohlcv.df, 1)
    xc.bases[base] = ohlcv
    setcurrenttime!(xc, base, startdt)
end

function addbases!(xc::XchCache, bases, startdt, enddt)
    for base in bases
        addbase!(xc, base, startdt, enddt)
    end
end

assetbases(xc::XchCache) = uppercase.(CryptoXch.balances(xc)[!, :coin])

symboltoken(basecoin, quotecoin=EnvConfig.cryptoquote) = isnothing(basecoin) ? nothing : uppercase(basecoin * quotecoin)

ceilbase(base, qty) = base == "usdt" ? ceil(qty, digits=3) : ceil(qty, digits=5)
floorbase(base, qty) = base == "usdt" ? floor(qty, digits=3) : floor(qty, digits=5)
roundbase(base, qty) = base == "usdt" ? round(qty, digits=3) : round(qty, digits=5)
# TODO read base specific digits from binance and use them base specific

onlyconfiguredsymbols(symbol) =
    endswith(symbol, uppercase(EnvConfig.cryptoquote)) &&
    !(uppercase(symbol[1:end-length(EnvConfig.cryptoquote)]) in baseignore)

"Returns pair of basecoin and quotecoin if quotecoin in `quotecoins` else `nothing` is returned"
function basequote(symbol)
    symbol = uppercase(symbol)
    range = nothing
    for qc in quotecoins
        range = findfirst(qc, symbol)
        if !isnothing(range)
            break
        end
    end
    return isnothing(range) ? nothing : (basecoin = symbol[begin:range[1]-1], quotecoin = symbol[range])
end

#endregion support

#region time

"""
Removes ohlcv data rows that are outside the date boundaries (nothing= no boundary) and adjusts ohlcv.ix to stay within the new data range.
"""
function timerangecut!(xc::XchCache, startdt, enddt)
    for ohlcv in CryptoXch.ohlcv(xc)
        (verbosity >= 3) && println("before Ohlcv.timerangecut!($ohlcv, $startdt, $enddt)")
        Ohlcv.timerangecut!(ohlcv, startdt, enddt)
        (verbosity >= 3) && println("after Ohlcv.timerangecut!($ohlcv, $startdt, $enddt)")
    end
end

function Base.iterate(xc::XchCache, currentdt=nothing)
    currentdt = isnothing(currentdt) ? xc.startdt : currentdt + Minute(1)
    _sleepuntil(xc, currentdt)

    (verbosity >= 3) && println("iterate: startdt=$(xc.startdt), currentdt=$(xc.currentdt), enddt=$(xc.enddt) local currentdt=$currentdt")
    # println("\rcurrentdt=$(string(currentdt)) xc.enddt=$(string(xc.enddt)) ")
    if !isnothing(xc.enddt) && (currentdt > xc.enddt)
        xc.currentdt = nothing
        return nothing
    else
        CryptoXch.setcurrenttime!(xc, currentdt)  # also updates bases if current time is > last time of xc
    end
    (verbosity >= 3) && println("iterate: utcnow=$(Dates.now(UTC)) startdt=$(xc.startdt), currentdt=$(xc.currentdt), enddt=$(xc.enddt)")
    return xc, currentdt
end

timesimulation(xc::XchCache)::Bool = !isnothing(xc.currentdt) && !isnothing(xc.enddt)
tradetime(xc::XchCache) = isnothing(xc.currentdt) ? floor(_exchangeservertime(xc), Minute(1)) : xc.currentdt
# tradetime(xc::XchCache) = (xc.mc[:simmode] != cryptoxchsim) ? _exchangeservertime(xc) : Dates.now(UTC)
ttstr(dt::DateTime) = "LT" * EnvConfig.now() * "/TT" * Dates.format(dt, EnvConfig.datetimeformat)
ttstr(xc::XchCache) = ttstr(tradetime(xc))

function _sleepuntil(xc::XchCache, dt::DateTime)
    if !isnothing(xc.enddt)  # then backtest
        return
    end
    sleepperiod = (dt + Second(2)) - _exchangeservertime(xc)
    if sleepperiod <= Dates.Second(0)
        return
    end
    if sleepperiod > Minute(1)
        (verbosity >= 2) && println("TT=$(tradetime(xc)) waiting until $dt resulting in long sleep $(floor(sleepperiod, Minute))")
    end
    # println("sleeping $(floor(sleepperiod, Second))")
    sleep(sleepperiod)
end

"Sleeps until `datetime` if reached if `datetime` is in the future, set the *current* time and updates ohlcv if required"
function setcurrenttime!(xc::XchCache, base, datetime::DateTime)
    dt = floor(datetime, Minute(1))
    ot = []
    if base in keys(xc.bases)
        ohlcv = xc.bases[base]
        ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
        if (length(ot) == 0) || (dt > ot[end])
            xc.bases[base] = cryptoupdate!(xc, ohlcv, (length(ot) == 0 ? dt : ot[begin]), dt)
        end
    else
        xc.bases[base] = ohlcv = cryptodownload(xc, base, "1m", dt, dt)
        ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    end
    Ohlcv.setix!(ohlcv, Ohlcv.rowix(ohlcv, dt))
    if (length(ot) > 0) && (ot[begin] <= dt <= ot[end]) && (ot[Ohlcv.ix(ohlcv)] != dt)
        if (verbosity >= 1) && (EnvConfig.configmode == production)
            @warn "setcurrenttime!($base, $dt) failed, opentime[ix]=$(Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)])"
        end
    end
    return ohlcv
end

"Set xc.currentdt and all cached base ohlcv.ix to the provided datetime. If isnothing(datetime) the only xc.currentdt is set to nothing"
function setcurrenttime!(xc::XchCache, datetime::Union{DateTime, Nothing})
    xc.currentdt = datetime
    if !isnothing(datetime)
        for base in keys(xc.bases)
            setcurrenttime!(xc, base, datetime)
        end
    end
end

#endregion time

#region klines

"""
Requests base/USDT from start until end (both including) in interval frequency but will return a maximum of 1000 entries.
Subsequent calls are required to get > 1000 entries.
Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""
function _ohlcfromexchange(xc::XchCache, base::AbstractString, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    df = nothing
    if base in TestOhlcv.testbasecoin()
        df = TestOhlcv.testdataframe(base, startdt, enddt, interval, cryptoquote)
        # pivot column is already added by TestOhlcv.testdataframe()
    else
        symbol = uppercase(base*cryptoquote)
        df = _exchangegetklines(xc, symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
        Ohlcv.addpivot!(df)
    end
    return df
end

"""
Requests base/USDT from start until end (both including) in interval frequency. If required Bybit is internally called several times to fill the request.

Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

time gaps will not be filled
"""
function _gethistoryohlcv(xc::XchCache, base::AbstractString, startdt::DateTime, enddt::DateTime=Dates.now(Dates.UTC), interval="1m")
    # startdt = DateTime("2020-08-11T22:45:00")
    # enddt = DateTime("2020-08-12T22:49:00")
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    fetches = 0
    # println("requesting from $startdt until $enddt $(ceil(enddt - startdt, intervalperiod(interval)) + intervalperiod(interval)) $base OHLCV from binance")

    notreachedstartdt = true
    df = Ohlcv.defaultohlcvdataframe()
    lastdt = enddt + Dates.Minute(1)  # make sure lastdt break condition is not true
    (verbosity >= 3) && @info "request from $startdt until $enddt at entry"
    while notreachedstartdt
        # fills from newest to oldest using Bybit
        fetches =+ 1
        if startdt > enddt
            (verbosity >= 3) && @warn "fetch $fetches: startdt $startdt > enddt $enddt at entry - exchanging"
            dt = startdt
            startdt = enddt
            enddt = dt
        end
        res = _ohlcfromexchange(xc, base, startdt, enddt, interval)
        if size(res, 1) == 0
            # will be the case for the timerange before the first coin data is available
            # Logging.@warn "no $base $interval data returned by last ohlcv read from $startdt until $enddt"
            break
        end
        notreachedstartdt = (res[begin, :opentime] > startdt) # Bybit loads newest first
        if res[begin, :opentime] >= lastdt
            # no progress since last ohlcv read - will be the case for all coins that have no cached data because startdt is likely before the first coin data
            (verbosity >= 3) && @warn "fetch $fetches: no progress since last ohlcv read: requested from $startdt until $enddt - received from $(res[begin, :opentime]) until $(res[end, :opentime]), lastdt=$lastdt - returning df from $(df[begin, :opentime]) until $(df[end, :opentime])"
            break
        end
        lastdt = res[begin, :opentime]
        # println("$(Dates.now()) read $(nrow(res)) $base from $enddt backwards until $lastdt")
        enddt = floor(lastdt, intervalperiod(interval))
        while (size(df,1) > 0) && (size(res,1) > 0) && (res[end, :opentime] >= df[begin, :opentime])  # replace last row with updated data
            deleteat!(res, size(res, 1))
        end
        @assert all(names(df) .== names(res)) "names(df)=$(names(df)) .== names(res)=$(names(res))"
        if size(res, 1) > 0
            if size(df, 1) > 0
                df = vcat(res, df)
            else
                df = res
            end
        end
    end
    return df
end

"""
Returns the OHLCV data of the requested time range by first checking the given (`ohlcv` parameter) cache data and if unsuccessful requesting it from the exchange.

- ohlcv containes the requested base identifier and interval - the result will be stored in the data frame of this structure
- startdt and enddt are DateTime stamps that specify the requested time range

"""
function cryptoupdate!(xc::XchCache, ohlcv, startdt, enddt)
    base = ohlcv.base
    interval = ohlcv.interval
    # println("Requesting $base $interval intervals from $startdt until $enddt")
    if enddt < startdt
        Logging.@warn "Invalid datetime range: end datetime $enddt <= start datetime $startdt"
        return ohlcv
    end
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    olddf = Ohlcv.dataframe(ohlcv)
    if (size(olddf, 1) > 0) && (startdt < olddf[end, :opentime]) && (enddt > olddf[begin, :opentime]) # there is already data available and overlapping
        if (startdt < olddf[begin, :opentime])
            # correct enddt in each case (gap between new and old range or range overlap) to avoid time range gaps
            tmpdt = olddf[begin, :opentime] - intervalperiod(interval)
            # get data of a timerange before the already available data
            newdf = _gethistoryohlcv(xc, base, startdt, tmpdt, interval)
            if size(newdf, 1) > 0
                if names(olddf) == names(newdf)
                    olddf = vcat(newdf, olddf)
                else
                    (verbosity >= 1) && @error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end
        if (enddt > olddf[end, :opentime])
            tmpdt = olddf[end, :opentime]  # update last data row
            newdf = _gethistoryohlcv(xc, base, tmpdt, enddt, interval)
            if size(newdf, 1) > 0
                while (size(olddf, 1) > 0) && (newdf[begin, :opentime] <= olddf[end, :opentime])  # replace last row with updated data
                    deleteat!(olddf, size(olddf, 1))
                end
                if names(olddf) == names(newdf)
                    olddf = vcat(olddf, newdf)
                else
                    (verbosity >= 1) && @error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end

    else # size(olddf, 1) == 0
        newdf = _gethistoryohlcv(xc, base, startdt, enddt, interval)
        Ohlcv.setdataframe!(ohlcv, newdf)
    end
    xc.bases[ohlcv.base] = ohlcv
    return ohlcv
end

"""
Returns the OHLCV data of the requested time range by first checking the stored cache data and if unsuccessful requesting it from the Exchange.

    - *base* identifier and interval specify what data is requested - the result will be returned as OhlcvData structure
    - startdt and enddt are DateTime stamps that specify the requested time range
    - any gap to chached data will be closed when asking for missing data from Bybit
"""
function cryptodownload(xc::XchCache, base, interval, startdt, enddt)::OhlcvData
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.setinterval!(ohlcv, interval)
    if validbase(xc, base)
        if Ohlcv.file(ohlcv).existing
            Ohlcv.read!(ohlcv)
        end
        cryptoupdate!(xc, ohlcv, startdt, enddt)
        ohlcv.ix = firstindex(ohlcv.df, 1)
    else
        (verbosity >= 1) && @warn "base=$base is unknown or invalid"
    end
    return ohlcv
end

"downloads missing data and merges with canned data then saves it as supplemented canned data"
function downloadupdate!(xc::XchCache, bases, enddt, period=Dates.Year(10))
    count = length(bases)
    enddt = floor(enddt, Dates.Minute)
    startdt = floor(enddt - period, Dates.Minute)
    for (ix, base) in enumerate(bases)
        # break
        (verbosity >= 2) && println("\n$(EnvConfig.now()) start updating $base ($ix of $count) request from $startdt until $enddt")
        ohlcv = CryptoXch.cryptodownload(xc, base, "1m", startdt, enddt)
        Ohlcv.write(ohlcv)
    end
end

"Downloads all basecoins with USDT quote that shows a minimumdayquotevolume and saves it as canned data"
function downloadallUSDT(xc::XchCache, enddt, period=Dates.Year(10), minimumdayquotevolume = 10000000)
    df = getUSDTmarket(xc)
    df = df[df.quotevolume24h .> minimumdayquotevolume , :]
    bases = sort!(setdiff(df[!, :basecoin], baseignore))
    (verbosity >= 2) && println("$(EnvConfig.now())downloading the following bases bases with $(EnvConfig.cryptoquote) quote: $bases")
    downloadupdate!(xc, bases, enddt, period)
    return df
end

#endregion klines

#region public

function validsymbol(xc::XchCache, symbol)
    sym = _exchangesymbolinfo(xc, symbol)
    r = _exchangevalidsymbol(xc, sym) &&
        !(sym.basecoin in baseignore) &&
        !_isleveraged(sym.basecoin)
    return r || (basequote(symbol).basecoin in TestOhlcv.testbasecoin())
end

"Returns a tuple of (minimum base quantity, minimum quote quantity)"
function minimumqty(xc::XchCache, sym::AbstractString)
    syminfo = _exchangesymbolinfo(xc, sym)
    if isnothing(syminfo)
        if validsymbol(xc, sym)
            (verbosity >= 1) && @error "cannot find symbol $sym in $(exchange(xc)) exchange info"
        end
        return nothing
    end
    return (minbaseqty=syminfo.minbaseqty, minquoteqty=syminfo.minquoteqty)
end

function minimumbasequantity(xc::XchCache, base::AbstractString, price=(base in bases(xc) ? Ohlcv.dataframe(ohlcv(xc, base))[Ohlcv.ix(ohlcv(xc, base)), :close] : nothing))
    if isnothing(price)
        return nothing
    end
    sym = CryptoXch.symboltoken(base)
    syminfo = CryptoXch.minimumqty(xc, sym)
    return isnothing(syminfo) ? nothing : 1.01 * max(syminfo.minbaseqty, syminfo.minquoteqty/price) # 1% more to avoid issues by rounding errors
end

function minimumquotequantity(xc::XchCache, base::AbstractString, price=(base in bases(xc) ? Ohlcv.dataframe(ohlcv(xc, base))[Ohlcv.ix(ohlcv(xc, base)), :close] : nothing))
    if isnothing(price)
        return nothing
    end
    sym = CryptoXch.symboltoken(base)
    syminfo = CryptoXch.minimumqty(xc, sym)
    return isnothing(syminfo) ? nothing : 1.01 * max(syminfo.minbaseqty * price, syminfo.minquoteqty) # 1% more to avoid issues by rounding errors
end

function precision(xc::XchCache, sym::AbstractString)
    syminfo = _exchangesymbolinfo(xc, sym)
    if isnothing(syminfo)
        (verbosity >= 1) && @error "cannot find symbol $sym in $(exchange(xc)) exchange info"
        return nothing
    end
    return (baseprecision=syminfo.baseprecision, quoteprecision=syminfo.quoteprecision)
end

_emptymarkets()::DataFrame = DataFrame(basecoin=String[], quotevolume24h=Float32[], pricechangepercent=Float32[], lastprice=Float32[], askprice=Float32[], bidprice=Float32[])

USDTMARKETFILE = "USDTmarket"

function _usdtmarketstem(fileprefix, timestamp::Union{Nothing, DateTime})
    return isnothing(timestamp) ? fileprefix : join([fileprefix, Dates.format(timestamp, "yy-mm-dd")], "_")
end

function _usdtmarketfilename(fileprefix, timestamp::Union{Nothing, DateTime}; format::Symbol=:arrow)
    folderpath = EnvConfig.datafolderpath(fileprefix)
    return EnvConfig.tablepath(_usdtmarketstem(fileprefix, timestamp); folderpath=folderpath, format=format)
end

"""
Returns a dataframe with 24h values of all USDT quotecoin bases that are not in baseignore list with the following columns:

- basecoin
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

getUSDTmarket: 512×6 DataFrame
 Row │ askprice       bidprice       lastprice      quotevolume24h  pricechangepercent  basecoin
     │ Float32        Float32        Float32        Float32         Float32             String
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │    0.65           0.6499         0.6499           6.51727e6             -0.0536  OP


   In case of timesimulation(xc) == true a canned USDTmarket file will be used - if one is present.
"""
function getUSDTmarket(xc::XchCache; dt::DateTime=tradetime(xc))
    if EnvConfig.configmode == test
        usdtdf = _emptymarkets()
        for basecoin in TestOhlcv.testbasecoin()
            ohlcv = TestOhlcv.testohlcv(basecoin, dt - Day(1) + Minute(1), dt)
            odf = Ohlcv.dataframe(ohlcv)
            vol24h = sum(odf[!, :basevolume] .* odf[!, :pivot])
            push!(usdtdf, (basecoin=basecoin, quotevolume24h=vol24h, pricechangepercent=((odf[end, :close] - odf[begin, :open]) / odf[begin, :open]), lastprice=odf[end, :close], askprice=odf[end, :close]*1.00001, bidprice=odf[end, :close]*0.99999))
            # println("base=$basecoin, rows=$(size(odf, 1)), start=$(odf[begin, :opentime]), stop=$(odf[end, :opentime])")
        end
        return usdtdf
    end

    if timesimulation(xc)
        usdtdf = _emptymarkets()
        cfgstem = _usdtmarketstem(CryptoXch.USDTMARKETFILE, dt)
        folderpath = EnvConfig.datafolderpath(CryptoXch.USDTMARKETFILE)
        cfgfilename = _usdtmarketfilename(CryptoXch.USDTMARKETFILE, dt; format=:arrow)
        loaded = EnvConfig.readdf(cfgstem; folderpath=folderpath, format=:arrow, copycols=true)
        if !isnothing(loaded)
            (verbosity >= 3) && println("Start loading USDT market data from $cfgfilename for $dt ")
            usdtdf = DataFrame(loaded; copycols=true)
            (verbosity >=2) && println("Loaded USDT market data for $(size(usdtdf, 1)) coins from $cfgfilename for $dt")
            legacyfile = _usdtmarketfilename(CryptoXch.USDTMARKETFILE, dt; format=:jdf)
            if isdir(legacyfile) || isfile(legacyfile)
                rm(legacyfile; force=true, recursive=true)
            end
        else
            (verbosity >=2) && println("No USDT market data file $cfgfilename for $dt found")
        end
    else  # production
        usdtdf = _exchangeget24h(xc)
        bq = [basequote(s) for s in usdtdf.symbol]  # create vector of pairs (basecoin, quotecoin)
        @assert length(bq) == size(usdtdf, 1)
        usdtdf[!, :basecoin] = [isnothing(bqe) ? missing : bqe.basecoin for bqe in bq]
        nbq = [!isnothing(bqe) && validbase(xc, bqe.basecoin) && (bqe.quotecoin == EnvConfig.cryptoquote) for bqe in bq]  # create binary vector as DataFrame filter
        usdtdf = usdtdf[nbq, Not(:symbol)]
        # usdtdf = usdtdf[nbq, :]
        # usdtdf = usdtdf[!, Not(:symbol)]
        # usdtdf = usdtdf[(usdtdf.quoteCoin .== "USDT") && (usdtdf.status .== "Trading"), :] - covered above by validbase
        # usdtdf = filter(row -> validbase(xc, row.basecoin), usdtdf) - covered above by validbase
        (verbosity >= 3) && println("writing USDTmarket file of size=$(size(usdtdf)) at enddt=$dt")
        folderpath = EnvConfig.datafolderpath(CryptoXch.USDTMARKETFILE)
        EnvConfig.savedf(usdtdf, _usdtmarketstem(CryptoXch.USDTMARKETFILE, dt); folderpath=folderpath, format=:arrow)
        legacyfile = _usdtmarketfilename(CryptoXch.USDTMARKETFILE, dt; format=:jdf)
        if isdir(legacyfile) || isfile(legacyfile)
            rm(legacyfile; force=true, recursive=true)
        end
    end
    return usdtdf
end

#endregion public

#region account

"Returns a DataFrame[:coin, :locked, :free, :borrowed, :accruedinterest] of wallet/portfolio balances"
function balances(xc::XchCache; ignoresmallvolume=true)
    bdf = nothing
    if (xc.mc[:simmode] != cryptoxchsim)
        bdf = _exchangebalances(xc)
        select = [!(coin in baseignore) || (coin == EnvConfig.cryptoquote) for coin in bdf[!, :coin]]
        bdf = bdf[select, :]
    else  # simulation
        if isnothing(xc)
            (verbosity >= 1) && @error "cannot simulate balances() with uninitialized CryptoXch cache"
            return DataFrame()
        end
        bdf = _balances(xc)
    end
    if !isnothing(bdf) && (size(bdf, 1) > 0) && ignoresmallvolume
        delrows = []
        for ix in eachindex(bdf[!, :coin])
            if bdf[ix, :coin] != EnvConfig.cryptoquote
                sym = symboltoken(bdf[ix, :coin])
                syminfo = minimumqty(xc, sym)
                if !validsymbol(xc, sym) || ((abs(bdf[ix, :free]) + abs(bdf[ix, :locked]) + abs(bdf[ix, :borrowed])) < 1.01 * syminfo.minbaseqty) # 1% more to avoid issues by rounding errors
                    push!(delrows, ix)
                end
            end
        end
        deleteat!(bdf, delrows)
    end
    return bdf
end

"""
Appends a balances DataFrame with the USDT value of the coin asset using usdtdf[:lastprice] and returns it as DataFrame[:coin, :locked, :free, :usdtprice, :usdtvalue].
"""
function portfolio!(xc::XchCache, balancesdf=balances(xc, ignoresmallvolume=false), usdtdf=getUSDTmarket(xc); ignoresmallvolume=true)
    if isnothing(xc.currentdt)
        if isnothing(usdtdf)
            usdtdf = getUSDTmarket(xc)
        end
        portfoliodf = leftjoin(balancesdf, usdtdf[!, [:basecoin, :lastprice]], on = :coin => :basecoin)
        portfoliodf.lastprice = coalesce.(portfoliodf.lastprice, 1.0f0)
        rename!(portfoliodf, :lastprice => "usdtprice")
    else
        usdtprice = Float32[]
        portfoliodf = balancesdf[:, :]
        for bix in eachindex(portfoliodf[!, :coin])
            if portfoliodf[bix, :coin] == EnvConfig.cryptoquote
                push!(usdtprice, 1f0)
            else
                ohlcv = setcurrenttime!(xc, portfoliodf[bix, :coin], xc.currentdt)
                if size(ohlcv.df, 1) > 0
                    push!(usdtprice, ohlcv.df[ohlcv.ix, :close])
                else
                    (verbosity >= 1) && @warn "found no data at $(xc.currentdt) for asset $ohlcv"  # (verbosity >= 3) &&
                    push!(usdtprice, 0f0)
                end
            end
        end
        portfoliodf.usdtprice = usdtprice
    end
    portfoliodf.usdtvalue = (abs.(portfoliodf.free .+ portfoliodf.locked) .- portfoliodf.borrowed) .* portfoliodf.usdtprice .* sign.(portfoliodf.free .+ portfoliodf.locked)
    if ignoresmallvolume
        delrows = []
        for ix in eachindex(portfoliodf[!, :coin])
            minbasequant = minimumbasequantity(xc, portfoliodf[ix, :coin], portfoliodf[ix, :usdtprice])
            if !(portfoliodf[ix, :coin] in quotecoins) && (isnothing(minbasequant) || (portfoliodf[ix, :coin] != EnvConfig.cryptoquote) && ((abs(portfoliodf[ix, :free]) + abs(portfoliodf[ix, :locked]) + abs(portfoliodf[ix, :borrowed])) < minbasequant))
                push!(delrows, ix)
            end
        end
        deleteat!(portfoliodf, delrows)
    end
    return portfoliodf
end

openstatus(st::AbstractString)::Bool = st in ["New", "PartiallyFilled", "Untriggered"]
openstatus(stvec::AbstractVector{String})::Vector{Bool} = [openstatus(st) for st in stvec]

"""
Returns an AbstractDataFrame of open **spot** orders with columns:

- orderid ::String
- symbol ::String
- side ::String (`Buy` or `Sell`)
- baseqty ::Float32
- ordertype ::String  `Market`, `Limit`
- timeinforce ::String      `GTC` GoodTillCancel, `IOC` ImmediateOrCancel, `FOK` FillOrKill, `PostOnly`
- limitprice ::Float32
- executedqty ::Float32  (to be executed qty = baseqty - executedqty)
- status ::String      `New`, `PartiallyFilled`, `Untriggered`, `Rejected`, `PartiallyFilledCanceled`, `Filled`, `Cancelled`, `Triggered`, `Deactivated`
- created ::DateTime
- updated ::DateTime
- rejectreason ::String
"""
function getopenorders(xc::XchCache, base=nothing)::AbstractDataFrame
    if (xc.mc[:simmode] != cryptoxchsim)
        oo = _exchangeopenorders(xc, symbol=symboltoken(base))
        return size(oo) == (0,0) ? _emptyorders(exchange(xc)) : oo
    else  # simulation
        if isnothing(xc)
            (verbosity >= 1) && @error "cannot simulate getopenorders() with uninitialized CryptoXch cache"
            return DataFrame()
        end
        for oix in reverse(eachindex(xc.orders[!, :orderid]))
             _updateorder!(xc, oix)
        end
        # orders = subset(xc.orders, :status => st -> openstatus(st), view=true)  # not necessary since closed orders are moved to xc.closedorders
        # return isnothing(base) ? orders[!, Not(:lastcheck)] : orders[symboltoken(base) .== orders.symbol, Not(:lastcheck)]
        return isnothing(base) ? xc.orders[!, :] : xc.orders[symboltoken(base) .== xc.orders.symbol, :]
    end
end

"Returns a named tuple with elements equal to columns of getopenorders() dataframe of the identified order or `nothing` if order is not found"
function getorder(xc::XchCache, orderid)
    if (xc.mc[:simmode] != cryptoxchsim)
        return _exchangeorder(xc, orderid)
    else  # simulation
        orderindex = _orderrefresh(xc, orderid)
        coix = isnothing(orderindex) ? findlast(x -> x == orderid, xc.closedorders[!, :orderid]) : nothing
        return isnothing(orderindex) ? (isnothing(coix) ? nothing : NamedTuple(xc.closedorders[coix, :])) : NamedTuple(xc.orders[orderindex, :])
    end
end

"Returns orderid in case of a successful cancellation"
function cancelorder(xc::XchCache, base, orderid)
    if (xc.mc[:simmode] != cryptoxchsim)
        return _exchangecancelorder(xc, symboltoken(base), orderid)
    else  # simulation
        return _cancelordersimulation(xc, orderid)
    end
end

"""
Places an order: spot order by default or margin order if 2 <= marginleverage <= 10
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` > current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createbuyorder(xc::XchCache, base::AbstractString; limitprice, basequantity, maker::Bool=false, marginleverage::Signed=0)
    base = uppercase(base)
    if (xc.mc[:simmode] != cryptoxchsim)
        oocreate = _exchangecreateorder(xc, symboltoken(base), "Buy", basequantity, limitprice, maker, marginleverage=marginleverage)
        oid = isnothing(oocreate) ? nothing : oocreate.orderid
        (verbosity >= 3) && @info "$(tradetime(xc)) $base: $(isnothing(oocreate) ? "no order info" : oocreate)"
        return oid
    else  # simulation
        return _createordersimulation(xc, base, buy, basequantity, limitprice, marginleverage)
    end
end

"""
Places an order: spot order by default or margin order if 2 <= marginleverage <= 10
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` < current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createsellorder(xc::XchCache, base::AbstractString; limitprice, basequantity, maker::Bool=true, marginleverage::Signed=0)
    base = uppercase(base)
    if (xc.mc[:simmode] != cryptoxchsim)
        oocreate = _exchangecreateorder(xc, symboltoken(base), "Sell", basequantity, limitprice, maker, marginleverage=marginleverage)
        oid = isnothing(oocreate) ? nothing : oocreate.orderid
        (verbosity >= 3) && @info "$(tradetime(cache)) $base: $(isnothing(oocreate) ? "no order info" : oocreate)"
        return oid
    else  # simulation
        return _createordersimulation(xc, base, sell, basequantity, limitprice, marginleverage)
    end
end

function changeorder(xc::XchCache, orderid; limitprice=nothing, basequantity=nothing)
    if (xc.mc[:simmode] != cryptoxchsim)
        oo = _exchangeorder(xc, orderid) #TODO in order to avoid this unnecessary order request the interface of exchange.amendorder need to remove symbol param
        if isnothing(oo)
            return nothing
        end
        ooamend = _exchangeamendorder(xc, oo.symbol, orderid; basequantity=basequantity, limitprice=limitprice)
        return isnothing(ooamend) ? nothing : ooamend.orderid
    else  # simulation
        return _changeordersimulation(xc::XchCache, orderid; limitprice=limitprice, basequantity=basequantity)
    end
end

#endregion account

#region simulation

#TODO add to longbuy orders flags that indicate why a longbuy advice is not followed:
#TODO exceedsmaxassetfraction = fraction of an asset exceed configured maximum
#TODO insufficientbuybalance = not enough USDT availble
#TODO withintradegap = trade is not executed because trade gap has notpassed yet
#
#TODO add to longbuy and longclose orders flags that indicate why an advice is not followed:
#TODO belowminimumbasequantity = longbuy/longclose amount too low
#TODO add comment with additional info
#
#TODO 3 order lists: openbuy (will only be moved to closed when longclose order is issued), opensell, closed
#TODO longbuy/longclose order records receive the longclose/longbuy counterpartprder id that links longbuy with longclose and a classifier id
#TODO a classifier remains active as long as there are open investments with its classifier id
#TODO 1 file with open and closed orders. longbuy orders without a longclose order id are open investments, orders withan executed qty are open orders


ASSETSFILENAME = "XchCacheAssets"
ORDERSFILENAME = "XchCacheOrders"
CLOSEDORDERSFILENAME = "XchCacheClosedorders"

"Finds or creates an asset order row in an asset dataframe and returns it. "
function _assetrow!(adf::DataFrame, coin)
    aorow = nothing
    adfix = size(adf, 1) > 0 ? findfirst(x -> x == coin, adf[!, :coin]) : nothing
    if isnothing(adfix)
        push!(adf, (coin = coin, free = 0f0, locked = 0f0, marginfree = 0f0, marginlocked = 0f0, assetborrowed = 0f0, orderborrowed = 0f0, accruedinterest = 0f0))
        aorow = last(adf)
    else
        aorow = adf[adfix, :]
    end
    return aorow
end

"Updates assets according to order execution."
function _updateassetsofcancelledorder!(xc::XchCache, orderrow)
    basecoin, quotecoin = basequote(orderrow.symbol)
    quoterow = _assetrow!(xc.assets, quotecoin)
    baserow = _assetrow!(xc.assets, basecoin)
    revertamount = orderrow.baseqty - orderrow.executedqty
    if orderrow.marginleverage == 0  # spot trade
        if orderrow.side == "Buy"
            quoterow.free += revertamount * orderrow.limitprice
            quoterow.locked -= revertamount * orderrow.limitprice
        else # side == "Sell"
            baserow.free += revertamount
            baserow.locked -= revertamount
        end
    else # orderrow.marginleverage in [2:10]
        # get required capital independent of free basecoin
        quoterow.free += revertamount * orderrow.limitprice / orderrow.marginleverage
        quoterow.locked -= revertamount * orderrow.limitprice / orderrow.marginleverage
        baserow.orderborrowed -= revertamount * (orderrow.marginleverage - 1) / orderrow.marginleverage  # abs amount due to pay back of borrowed only at order closure
        #TODO calc accrued interest of unused borrowed capital in slices of hours
    end
end

function _cancelordersimulation(xc::XchCache, orderid)
    if isnothing(xc)
        (verbosity >= 1) && @error "cannot simulate cancelorder() with uninitialized CryptoXch cache"
        return nothing
    end
    oix = _orderrefresh(xc, orderid)
    if !isnothing(oix)
        xco = xc.orders[oix, :]
        _updateassetsofcancelledorder!(xc, xco)
        base = basequote(xco.symbol).basecoin
        ohlcv = xc.bases[base]
        ohlcvdf = Ohlcv.dataframe(ohlcv)
        xco.updated = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
        if xco.executedqty > 0 
            xco.status = "PartiallyFilledCanceled"
        else
            xco.status = "Cancelled"
        end
        push!(xc.closedorders,xco)
        deleteat!(xc.orders, oix)
        return orderid
    else
        return nothing
    end
end

"Changes either limitprice or base quantity of an order. The simulation cancels the order and recreates a new using the same id. That already done executed quantities are not taken over."
function _changeordersimulation(xc::XchCache, orderid; limitprice=nothing, basequantity=nothing)
    oix = _orderrefresh(xc, orderid)
    # if isnothing(oix) || !openstatus(xc.orders[oix, :status]) || (xc.orders[oix, :baseqty] <= xc.orders[oix, :executedqty])
    if isnothing(oix) || (xc.orders[oix, :baseqty] <= xc.orders[oix, :executedqty])
        return nothing
    end
    xco = xc.orders[oix, :]
    baseqty = (isnothing(basequantity) ? xco.baseqty : basequantity) - xco.executedqty
    #* new new order will have zero executedqty but baseqty is reduce by the already executed amount
    marginleverage = xco.marginleverage
    base = basequote(xco.symbol).basecoin
    side = xco.side == "Buy" ? buy : sell
    limitprice = isnothing(limitprice) ? xco.limitprice : limitprice
    @assert orderid == _cancelordersimulation(xc, xco.orderid)
    orderid = _createordersimulation(xc, base, side, baseqty, limitprice, marginleverage; oid=orderid)
    return orderid
end

"Updates assets according to order execution."
function _updateassetsofexecutedorder!(xc::XchCache, orderrow, transactionqty)
    basecoin, quotecoin = basequote(orderrow.symbol)
    @assert quotecoin == EnvConfig.cryptoquote
    @assert transactionqty <= orderrow.baseqty - orderrow.executedqty "transactionqty=$transactionqty > baseqty=$(orderrow.baseqty) - executedqty=$(orderrow.executedqty)"
    @assert transactionqty > 0f0
    quoterow = _assetrow!(xc.assets, quotecoin)
    baserow = _assetrow!(xc.assets, basecoin)
    side = (orderrow.side == "Buy" ? 1f0 : -1f0)
    if orderrow.marginleverage == 0  # spot trade
        if side > 0 # side == Buy
            quoterow.locked -= transactionqty * orderrow.limitprice
            baserow.free += transactionqty
        else # side == "Sell"
            baserow.locked -= transactionqty
            quoterow.free += transactionqty * orderrow.limitprice
        end
    else # orderrow.marginleverage in [2:10]
        if side > 0 # side == Buy
            reduceqty = baserow.marginlocked < 0 ? min(abs(baserow.marginlocked), transactionqty) : 0f0
            extendqty = transactionqty - reduceqty
            baserow.marginlocked += reduceqty  # unlock negative reduce amount
            baserow.assetborrowed -= reduceqty * (orderrow.marginleverage - 1) / orderrow.marginleverage
            quoterow.free += reduceqty * orderrow.limitprice / orderrow.marginleverage
            if extendqty > 0f0
                baserow.marginfree += extendqty
            end
        else # side == "Sell"
            reduceqty = baserow.marginlocked > 0 ? min(abs(baserow.marginlocked), transactionqty) : 0f0
            extendqty = transactionqty - reduceqty
            if reduceqty > 0
                baserow.marginlocked += -reduceqty  # unlock positive reduce amount
                baserow.assetborrowed -= reduceqty * (orderrow.marginleverage - 1) / orderrow.marginleverage
                quoterow.free += reduceqty * orderrow.limitprice / orderrow.marginleverage
                reducespotqty = baserow.locked > 0 ? min(abs(baserow.locked), extendqty) : 0f0
                if reducespotqty > 0 # use case: first reduce also spot base before selling coins via margin
                    extendqty = extendqty - reducespotqty
                    baserow.locked -= reducespotqty  # unlock positive reduce amount
                    baserow.assetborrowed = max(0f0, baserow.assetborrowed - reducespotqty)  # pay back positive reduce amount
                    quoterow.free += reducespotqty * orderrow.limitprice
                end
            end
            if extendqty > 0f0
                baserow.marginfree -= extendqty
            end
        end
        if extendqty > 0f0
            baserow.orderborrowed -= extendqty * (orderrow.marginleverage - 1) / orderrow.marginleverage  # pay back of abs amount during open order
            baserow.assetborrowed += (extendqty) * (orderrow.marginleverage - 1) / orderrow.marginleverage  # pay back of abs amount during open order
            quoterow.locked -= extendqty * orderrow.limitprice / orderrow.marginleverage
        end
    end
    quoterow.free -= transactionqty * orderrow.limitprice * xc.feerate
end


"Checks ohlcv since last check and marks order as executed if limitprice is exceeded"
function _updateorder!(xc::XchCache, orderix)
    xco = xc.orders[orderix, :]
    transactionqty = xco.baseqty  #TODO too simple to assume just one transaction for the whole order
    # if !openstatus(xco.status)
    #     return
    # end
    base, _ = basequote(xco.symbol)
    ohlcv = xc.bases[base]
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    ohlcvix = oix = Ohlcv.ix(ohlcv)
    if ohlcvdf.opentime[oix] != xco.lastcheck
        while (ohlcvdf.opentime[oix] > xco.lastcheck) && (oix > firstindex(ohlcvdf.opentime))
            oix -= 1
        end
        xco.lastcheck = ohlcvdf.opentime[ohlcvix]
        while oix <= ohlcvix
            if ((xco.side == "Buy") && (ohlcvdf.low[oix] <= xco.limitprice)) || ((xco.side == "Sell") && (ohlcvdf.high[oix] >= xco.limitprice))
                _updateassetsofexecutedorder!(xc, xco, transactionqty)
                xco.updated = ohlcvdf.opentime[ohlcvix]
                xco.avgprice = (xco.avgprice * xco.executedqty + xco.limitprice * transactionqty) / (xco.executedqty + transactionqty)
                xco.executedqty += transactionqty
                minbaseqty, _ = minimumqty(xc, xco.symbol)
                if (xco.executedqty - xco.baseqty) < minbaseqty
                    xco.status = "Filled"
                else
                    xco.status = "PartiallyFilled"
                end
                push!(xc.closedorders,xco)
                xco = last(xc.closedorders)
                deleteat!(xc.orders, orderix)
                break
            end
            oix += 1
        end
    end
    return xco
end

"Checks in simulation close conditions and returns order index or nothing if not found"
function _orderrefresh(xc::XchCache, orderid)
    openstatus = ["New", "PartiallyFilled", "Untriggered"]
    if isnothing(xc)
        (verbosity >= 1) && @error "cannot simulate getorder() with uninitialized CryptoXch cache"
        return nothing
    end
    xcoix = nothing
    for oix in reverse(eachindex(xc.orders[!, :orderid]))
        uo = _updateorder!(xc, oix)
        xcoix = (uo.orderid == orderid) && (uo.status in openstatus) ? oix : xcoix
    end
    return xcoix
end


function _assetaddorder!(baserow, quoterow, orderrow)
    if orderrow.marginleverage == 0  # spot trade
        if orderrow.side == "Buy"
            quoterow.free -= orderrow.baseqty * orderrow.limitprice
            quoterow.locked += orderrow.baseqty * orderrow.limitprice
        else # side == "Sell"
            baserow.free -= orderrow.baseqty
            baserow.locked += orderrow.baseqty
        end
    else # orderrow.marginleverage in [2:10]
        # get required capital independent of free basecoin
        sidefactor = (orderrow.side == "Buy" ? 1f0 : -1f0)
        if sidefactor > 0 # buy side
            if baserow.marginfree < 0 # -> reduce
                reduceqty = min(orderrow.baseqty, abs(baserow.marginfree))
                extendqty = orderrow.baseqty - reduceqty
                baserow.marginlocked += -reduceqty  # lock negative reduce amount
                baserow.marginfree += reduceqty
            else # extend only
                extendqty = orderrow.baseqty
            end
        else # side < 0 -> sell side
            if baserow.marginfree > 0 # -> reduce
                reduceqty = min(orderrow.baseqty, baserow.marginfree)
                extendqty = orderrow.baseqty - reduceqty
                if extendqty > 0
                    reducespotqty = min(extendqty, baserow.free) # use spot free also for reduce
                    extendqty = extendqty - reducespotqty
                    if reducespotqty > 0
                        baserow.free -= reducespotqty
                        baserow.locked += reducespotqty
                    end
                end
                baserow.marginlocked += reduceqty  # lock positive reduce amount
                baserow.marginfree += -reduceqty
            else # extend only
                extendqty = orderrow.baseqty
            end
        end
        if extendqty > 0
            quoterow.free -= extendqty * orderrow.limitprice / orderrow.marginleverage
            quoterow.locked += extendqty * orderrow.limitprice / orderrow.marginleverage
            baserow.orderborrowed += extendqty * (orderrow.marginleverage - 1) / orderrow.marginleverage  # without sidefactor because borrow is always >= 0
        end
    end
end

function _createordersimulation(xc::XchCache, base, side::Sidefactor, baseqty, limitprice, marginleverage; oid=nothing)
    #* borrowed is updated in xc.assets at order close
    #* locked is not updated in xc.assets but only derived by _assetstate
    @assert baseqty >= 0 "baseqty=$baseqty < 0"
    @assert isnothing(limitprice) || (limitprice >= 0) "limitprice=$limitprice < 0"
    @assert marginleverage in [0] || marginleverage in 2:10 "marginleverage not in [0, 2:10]"
    
    if isnothing(xc)
        (verbosity >= 1) && @error "cannot simulate create longbuy/longclose order() with uninitialized CryptoXch cache"
        return nothing
    end
    ohlcv = xc.bases[base]
    dtnow = ohlcv.df.opentime[ohlcv.ix]
    sym = symboltoken(base)
    if isnothing(limitprice)  # no limitprice indicates maker=true
        syminfo = _exchangesymbolinfo(xc, sym)
        limitprice = side == buy ? ohlcv.df[ohlcv.ix, :close] - syminfo.ticksize : ohlcv.df[ohlcv.ix, :close] + syminfo.ticksize
        timeinforce = "PostOnly"
    else
        timeinforce = "GTC"
    end
    orderid = isnothing(oid) ? "$side$baseqty*$(round(limitprice, sigdigits=5))$base$(Dates.format(dtnow, "yymmddTHH:MM"))" : oid
    if (side == buy)
        if (limitprice > (1+MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])
            (verbosity >= 2) && @info "limitprice $limitprice > max delta $((1+MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])"
            return nothing
        else
            limitprice = min(limitprice, ohlcv.df.close[ohlcv.ix])
        end
    else # if (side == sell)
        if (limitprice < (1-MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])
            (verbosity >= 1) && @warn "limitprice $limitprice < max delta $((1-MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])"
            return nothing
        else
            limitprice = max(limitprice, ohlcv.df.close[ohlcv.ix])
        end
    end
    xco = _placeorder(xc, (orderid=orderid, symbol=sym, side=side, baseqty=baseqty, ordertype="Limit", marginleverage=marginleverage, timeinforce=timeinforce, limitprice=limitprice, avgprice=0.0f0, executedqty=0.0f0, status="New", created=dtnow, updated=dtnow, rejectreason="NO ERROR", lastcheck=dtnow-Minute(1)))
    quoterow = _assetrow!(xc.assets, EnvConfig.cryptoquote)
    baserow = _assetrow!(xc.assets, basequote(xco.symbol).basecoin)
    _assetaddorder!(baserow, quoterow, xco)
    return orderid
end

function _closeprice(xc::XchCache, coin::AbstractString)
    if coin == EnvConfig.cryptoquote
        return 1f0
    else
        ohlcv = xc.bases[coin]
        ohlcvdf = Ohlcv.dataframe(ohlcv)
        ohlcvix = Ohlcv.ix(ohlcv)
        return ohlcvdf[ohlcvix, :close]
    end
end

"Set a fixed asset amount for coin into simulation bookkeeping and returns the assetrow."
function _updateasset!(xc::XchCache, coin, amount)
    xca = _assetrow!(xc.assets, coin)
    xca.free = amount
end

"""
- :marginfree, :marginlocked can be negative
- :free, :marginfree, :locked, :marginlocked can be added with sign to reflect balances for the same coin
- :free / :locked sum over all coins shall take the absolute value minus sum of :borrowed
"""
function _balances(xc::XchCache)
    df = select(xc.assets, :coin, [:locked, :marginlocked] => ((sl, ml) -> sl .+ ml) => :locked, [:free, :marginfree] => ((sf, mf) -> sf .+ mf) => :free, [:assetborrowed, :orderborrowed] => ((a, b) -> abs.(a .+ b)) => :borrowed, :accruedinterest)
    @assert all(abs.(df[!, :free] .+ df[!, :locked]) .> df[!, :borrowed] .>= 0f0) "working capital > borrowed capital >= 0f0 -> balances: $df"
    return df
end


_emptyassets()::DataFrame = DataFrame(coin=String31[], free=Float32[], locked=Float32[], marginfree=Float32[], marginlocked=Float32[], assetborrowed=Float32[], orderborrowed=Float32[], accruedinterest=Float32[])

"provides an empty dataframe for simulation (with lastcheck as extra column)"
function _emptyorders(exchange::AbstractString=EXCHANGE_BYBIT)::DataFrame
    df = _exchangeemptyorders(exchange)
    insertcols!(df, :marginleverage => Vector{Int32}(undef, 0))
end

"Places an order and returns the DataFrameRow of the created order row"
function _placeorder(xc::XchCache, order::Union{DataFrameRow, NamedTuple})
    @assert haskey(order, :orderid) && (length(order.orderid) < 64) "$(haskey(order, :orderid) ? "length(order.orderid)=$(length(order.orderid)) >= 64" : "orderid is missing in order")" 
    @assert haskey(order, :symbol) && (length(order.symbol) < 32) "$(haskey(order, :symbol) ? "length(order.symbol)=$(length(order.symbol)) >= 32" : "symbol is missing in order")" 
    @assert haskey(order, :side) "side is missing in order"
    @assert haskey(order, :baseqty) "baseqty is missing in order"
    @assert haskey(order, :ordertype) && (length(order.ordertype) < 8) "$(haskey(order, :ordertype) ? "length(order.ordertype)=$(length(order.ordertype)) >= 8" : "ordertype is missing in order")" 
    @assert haskey(order, :marginleverage) "marginleverage is missing in order"
    @assert haskey(order, :timeinforce) && (length(order.timeinforce) < 16) "$(haskey(order, :timeinforce) ? "length(order.timeinforce)=$(length(order.timeinforce)) >= 16" : "timeinforce is missing in order")" 
    @assert haskey(order, :limitprice) "limitprice is missing in order"
    @assert haskey(order, :avgprice) "avgprice is missing in order"
    @assert haskey(order, :executedqty) "executedqty is missing in order"
    @assert haskey(order, :status) && (length(order.status) < 16) "$(haskey(order, :status) ? "length(order.status)=$(length(order.status)) >= 16" : "status is missing in order")" 
    @assert haskey(order, :created) "created is missing in order"
    @assert haskey(order, :updated) "updated is missing in order"
    @assert haskey(order, :rejectreason) "rejectreason is missing in order"
    @assert haskey(order, :lastcheck) "lastcheck is missing in order"
    order = (order..., side=uppercasefirst(string(order.side)))
    # println("xc.orde rs=$(xc.orders)\nORDER: $order")
    push!(xc.orders, (order..., side=uppercasefirst(string(order.side)), isLeverage=(order.marginleverage > 0)))
    return last(xc.orders)
end

function _ordersfilestem(xc::XchCache)
    ORDERPREFIX = "Orders"
    fnvec = [ORDERPREFIX]
    if !isnothing(xc.mnemonic)
        push!(fnvec, xc.mnemonic)
    end
    push!(fnvec, string(EnvConfig.configmode))
    bases = sort(collect(keys(xc.bases)))
    fnvec = vcat(fnvec, bases)
    push!(fnvec, Dates.format(xc.startdt, "yy-mm-dd"))
    enddt = isnothing(xc.enddt) ? (size(xc.orders, 1) > 0 ? xc.orders[end, :created] : (size(xc.closedorders, 1) > 0 ? xc.closedorders[end, :created] : xc.startdt)) : xc.enddt
    push!(fnvec, Dates.format(enddt, "yy-mm-dd"))
    return join(fnvec, "_")
end

_ordersfilename(xc::XchCache; format::Symbol=:arrow) = EnvConfig.tablepath(_ordersfilestem(xc); folderpath=EnvConfig.logfolder(), format=format)

function writeorders(xc::XchCache)
    fn = _ordersfilename(xc; format=:arrow)
    (verbosity >=0) && println("saving order log in filename=$fn")
    df = nothing
    if size(xc.closedorders, 1) > 0
        df = xc.closedorders
        if size(xc.orders, 1) > 0
            df = vcat(df, xc.orders)
        end
    elseif size(xc.orders, 1) > 0
        df = xc.orders
    else
        @warn "no orders to save in $fn"
        return
    end
    EnvConfig.savedf(df, _ordersfilestem(xc); folderpath=EnvConfig.logfolder(), format=:arrow)
    legacyfile = _ordersfilename(xc; format=:jdf)
    if isdir(legacyfile) || isfile(legacyfile)
        rm(legacyfile; force=true, recursive=true)
    end
end

function _assetsfilestem(xc::XchCache, dt)
    ASSETPREFIX = "Assets"
    fnvec = [ASSETPREFIX]
    if !isnothing(xc.mnemonic)
        push!(fnvec, xc.mnemonic)
    end
    push!(fnvec, string(EnvConfig.configmode))
    push!(fnvec, Dates.format(dt, "yy-mm-dd"))
    return join(fnvec, "_")
end

_assetsfilename(xc::XchCache, dt; format::Symbol=:arrow) = EnvConfig.tablepath(_assetsfilestem(xc, dt); folderpath=EnvConfig.logfolder(), format=format)

function writeassets(xc::XchCache, dt::DateTime)
    fn = _assetsfilename(xc, dt; format=:arrow)
    (verbosity >=3) && println("saving asset snapshot in filename=$fn")
    EnvConfig.savedf(xc.assets, _assetsfilestem(xc, dt); folderpath=EnvConfig.logfolder(), format=:arrow)
    legacyfile = _assetsfilename(xc, dt; format=:jdf)
    if isdir(legacyfile) || isfile(legacyfile)
        rm(legacyfile; force=true, recursive=true)
    end
end

#endregion simulation

end  # of module
