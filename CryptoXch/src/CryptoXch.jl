# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])


module CryptoXch

using Dates, DataFrames, DataAPI, JDF, CSV, Logging
using Bybit, EnvConfig, Ohlcv, TestOhlcv
import Ohlcv: intervalperiod

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

mutable struct XchCache
    orders  # ::DataFrame
    closedorders  # ::DataFrame
    assets  # :: DataFrame
    bases  # ::Dict{String, Ohlcv.OhlcvData}
    bc  # ::Union{Nothing, Bybit.BybitCache}
    feerate  # 0.001 = 0.1% at Bybit maker = taker fee  #TODO store exchange info and account fee rate and use it in offline backtest simulation
    startdt::Dates.DateTime
    currentdt::Union{Nothing, Dates.DateTime}  # current back testing time
    enddt::Union{Nothing, Dates.DateTime}  # end time back testing; nothing == request life data without defined termination
    function XchCache(bybitinit::Bool=true; startdt::DateTime=Dates.now(UTC), enddt=nothing)
        startdt = floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        return new(emptyorders(), emptyorders(), emptyassets(), Dict(), bybitinit ? Bybit.BybitCache() : nothing, 0.001, startdt, nothing, enddt)
    end
end

setstartdt(xc::XchCache, dt::DateTime) = (xc.startdt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
setenddt(xc::XchCache, dt::DateTime) = (xc.enddt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
bases(xc::XchCache) = keys(xc.bases)
ohlcv(xc::XchCache) = values(xc.bases)
ohlcv(xc::XchCache, base::String) = xc.bases[base]
baseohlcvdict(xc::XchCache) = xc.bases

basenottradable = ["SUI"]
basestablecoin = ["USD", "USDT", "TUSD", "BUSD", "USDC", "EUR", "DAI"]
quotecoins = ["USDT"]  # , "USDC"]
baseignore = [""]
baseignore = uppercase.(append!(baseignore, basestablecoin, basenottradable))
minimumquotevolume = 10  # USDT

MAXLIMITDELTA = 0.1

_isleveraged(token) = (token[end] in ['S', 'L']) && isdigit(token[end-1])

validbase(xc::XchCache, base::String) = validsymbol(xc, symboltoken(base))

function validsymbol(xc::XchCache, symbol::String)
    sym = Bybit.symbolinfo(xc.bc, symbol)
    r = Bybit.validsymbol(xc.bc, sym) &&
        !(sym.basecoin in baseignore) &&
        !_isleveraged(sym.basecoin)
    return r || (basequote(symbol).basecoin in testbasecoin())
end


function minimumqty(xc::XchCache, sym::String)
    syminfo = Bybit.symbolinfo(xc.bc, sym)
    if isnothing(syminfo)
        @error "cannot find symbol $sym in Bybit exchange info"
        return nothing
    end
    return (minbaseqty=syminfo.minbaseqty, minquoteqty=syminfo.minquoteqty)
end

function precision(xc::XchCache, sym::String)
    syminfo = Bybit.symbolinfo(xc.bc, sym)
    if isnothing(syminfo)
        @error "cannot find symbol $sym in Bybit exchange info"
        return nothing
    end
    return (baseprecision=syminfo.baseprecision, quoteprecision=syminfo.quoteprecision)
end

"Returns a `(free, locked)` named tuple with the amount of `free` and `locked` amounts of coin in portfolio assets"
function _assetfreelocked(xc::XchCache, coin::String)
    coinix = findfirst(x -> x == uppercase(coin), xc.assets[!, :coin])
    if coin == EnvConfig.cryptoquote
        #* containemnt due to negative locked values (not understood yet)
        locked = sum(xc.orders[xc.orders[!, :side] .== "Buy", :baseqty] .* (xc.orders[xc.orders[!, :side] .== "Buy", :limitprice]))
    else
        locked = sum(xc.orders[(xc.orders[!, :side] .== "Sell") .&& (xc.orders[!, :symbol] .== symboltoken(coin)), :baseqty])
    end
    # return isnothing(coinix) ? (free=0.0f0, locked=0.0f0) : (free=xc.assets[coinix, :free], locked=xc.assets[coinix, :locked])
    return isnothing(coinix) ? (free=0.0f0, locked=0.0f0) : (free=xc.assets[coinix, :free], locked=locked)
end

SIMEPS = -2*eps(Float32)
function updateasset!(xc::XchCache, coin::String, lockedqty, freeqty)
    coin = uppercase(coin)
    coinix = findfirst(x -> x == coin, xc.assets[!, :coin])
    if isnothing(coinix)
        push!(xc.assets, (coin=coin, locked=lockedqty, free=freeqty))
        (lockedqty < 0) && println("initial lockedqty < 0 for $coin with lockedqty=$lockedqty, freeqty=$freeqty")
        (freeqty < 0) && println("initial freeqty < 0 for $coin with lockedqty=$lockedqty, freeqty=$freeqty")
    else
        xc.assets[coinix, :free] += freeqty
        if xc.assets[coinix, :free] < 0
            if xc.assets[coinix, :free] < SIMEPS
                error("xc.assets[coinix, :free]=$(xc.assets[coinix, :free]) < $SIMEPS for $coin with delata lockedqty=$lockedqty, delta freeqty=$freeqty")
            else
                xc.assets[coinix, :free] = 0f0
            end
        end
        xc.assets[coinix, :locked] = _assetfreelocked(xc, coin).locked
        # xc.assets[coinix, :locked] += lockedqty
        # if xc.assets[coinix, :locked] < 0
        #     if xc.assets[coinix, :locked] < SIMEPS
        #         # put behind verbose as all amounts < 10^-5 USDT #TODO still don't understand why we have these deltas
        #         (verbosity >= 3) && @warn  "xc.assets[coinix, :locked]=$(xc.assets[coinix, :locked]) < $SIMEPS for $coin with delta lockedqty=$lockedqty, delta freeqty=$freeqty"
        #     end
        #     xc.assets[coinix, :locked] = 0f0
        # end
    end
end

# function updatecache(xc::XchCache; ohlcv=nothing, orders=nothing, assets=nothing)  # not used and DEPRECATED
#     xc = isnothing(xc) ? XchCache(false) : xc
#     if !isnothing(ohlcv)
#         xc.bases[ohlcv.base] = ohlcv
#     end
#     if !isnothing(orders)
#         xc.orders = orders
#     end
#     if !isnothing(assets)
#         xc.assets = assets
#     end
# end

emptyassets()::DataFrame = DataFrame(coin=String[], locked=Float32[], free=Float32[])

"provides an empty dataframe for simulation (with lastcheck as extra column)"
emptyorders()::DataFrame = DataFrame(orderid=String[], symbol=String[], side=String[], baseqty=Float32[], ordertype=String[], timeinforce=String[], limitprice=Float32[], avgprice=Float32[], executedqty=Float32[], status=String[], created=DateTime[], updated=DateTime[], rejectreason=String[], lastcheck=DateTime[])

removebase!(xc::XchCache, base) = delete!(xc.bases, base)
removeallbases(xc::XchCache) = xc.bases = Dict()

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

# """
# Initializes the undrelying exchange.
# """
# function XchCache(bases::Vector, startdt=nothing, enddt=nothing, usdtbudget=10000)::XchCache
#     xc = XchCache(true)
#     bases = uppercase.(bases)
#     sellbases = union(bases, setdiff(assetbases(xc), basestablecoin))
#     oo = CryptoXch.getopenorders(xc)
#     if size(oo, 1) > 0
#         oo = DataFrame(CryptoXch.basequote.(oo.symbol))
#         sellbases = union(sellbases, oo[!, :basecoin])
#     end
#     sellbases = setdiff(sellbases, [EnvConfig.cryptoquote])
#     for base in sellbases  # sellbases is superset of bases
#         addbase!(xc, base, startdt, enddt)
#         # println("startdt=$startdt enddt=$enddt xc.bases[base]=$(xc.bases[base])")
#     end
#     if EnvConfig.configmode != production
#         # push startbudget onto balance wallet for backtesting/simulation
#         push!(xc.assets, (coin=uppercase(EnvConfig.cryptoquote), locked = 0.0f0, free=usdtbudget))
#     end
#     return xc
# end

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
tradetime(xc::XchCache) = isnothing(xc.currentdt) ? floor(Bybit.servertime(xc.bc), Minute(1)) : xc.currentdt
# tradetime(xc::XchCache) = EnvConfig.configmode == production ? Bybit.servertime(xc.bc) : Dates.now(UTC)
ttstr(xc::XchCache) = "TT" * Dates.format(tradetime(xc), EnvConfig.datetimeformat)

function _sleepuntil(xc::XchCache, dt::DateTime)
    if !isnothing(xc.enddt)  # then backtest
        return
    end
    sleepperiod = dt - Bybit.servertime(xc.bc)
    if sleepperiod <= Dates.Second(0)
        return
    end
    if sleepperiod > Minute(1)
        println("TT=$(tradetime(xc)) waiting until $dt resulting in long sleep $(floor(sleepperiod, Minute))")
    end
    # println("sleeping $(floor(sleepperiod, Second))")
    sleep(sleepperiod)
end

"Sleeps until `datetime` if reached if `datetime` is in the future, set the *current* time and updates ohlcv if required"
function setcurrenttime!(xc::XchCache, base::String, datetime::DateTime)
    dt = floor(datetime, Minute(1))
    if base in keys(xc.bases)
        ohlcv = xc.bases[base]
        ohlcvdf = Ohlcv.dataframe(ohlcv)
        if (size(ohlcvdf, 1) == 0) || (dt > ohlcvdf.opentime[end])
            xc.bases[base] = cryptoupdate!(xc, ohlcv, (size(ohlcvdf, 1) == 0 ? dt : ohlcvdf.opentime[begin]), dt)
        end
    else
        xc.bases[base] = ohlcv = cryptodownload(xc, base, "1m", dt, dt)
    end
    Ohlcv.setix!(ohlcv, Ohlcv.rowix(ohlcv, dt))
    if (size(Ohlcv.dataframe(ohlcv), 1) > 0) && (Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)] != dt)
        @warn "setcurrenttime!($base, $dt) failed, opentime[ix]=$(Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)])"
        println("setcurrenttime!($base, $dt) failed, opentime[ix]=$(Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)])")
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

symboltoken(basecoin, quotecoin=EnvConfig.cryptoquote) = isnothing(basecoin) ? nothing : uppercase(basecoin * quotecoin)

"Returns a vector of basecoin strings that are supported generated test basecoins of periodic patterns"
testbasecoin() = TestOhlcv.testbasecoin()

"""
Requests base/USDT from start until end (both including) in interval frequency but will return a maximum of 1000 entries.
Subsequent calls are required to get > 1000 entries.
Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""
function _ohlcfromexchange(xc::XchCache, base::String, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    df = nothing
    if base in testbasecoin()
        df = TestOhlcv.testdataframe(base, startdt, enddt, interval, cryptoquote)
        # pivot column is already added by TestOhlcv.testdataframe()
    else
        symbol = uppercase(base*cryptoquote)
        df = Bybit.getklines(xc.bc, symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
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
function _gethistoryohlcv(xc::XchCache, base::String, startdt::DateTime, enddt::DateTime=Dates.now(Dates.UTC), interval="1m")
    # startdt = DateTime("2020-08-11T22:45:00")
    # enddt = DateTime("2020-08-12T22:49:00")
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    # println("requesting from $startdt until $enddt $(ceil(enddt - startdt, intervalperiod(interval)) + intervalperiod(interval)) $base OHLCV from binance")

    notreachedstartdt = true
    df = Ohlcv.defaultohlcvdataframe()
    lastdt = enddt + Dates.Minute(1)  # make sure lastdt break condition is not true
    while notreachedstartdt
        # fills from newest to oldest using Bybit
        res = _ohlcfromexchange(xc, base, startdt, enddt, interval)
        if size(res, 1) == 0
            # Logging.@warn "no $base $interval data returned by last ohlcv read from $startdt until $enddt"
            break
        end
        notreachedstartdt = (res[begin, :opentime] > startdt) # Bybit loads newest first
        if res[begin, :opentime] >= lastdt
            # no progress since last ohlcv  read
            Logging.@warn "no progress since last ohlcv read: requested from $startdt until $enddt - received from $(res[begin, :opentime]) until $(res[end, :opentime])"
            break
        end
        lastdt = res[begin, :opentime]
        # println("$(Dates.now()) read $(nrow(res)) $base from $enddt backwards until $lastdt")
        enddt = floor(lastdt, intervalperiod(interval))
        while (size(df,1) > 0) && (size(res,1) > 0) && (res[end, :opentime] >= df[begin, :opentime])  # replace last row with updated data
            deleteat!(res, size(res, 1))
        end
        if (size(res, 1) > 0) && (names(df) == names(res))
            df = vcat(res, df)
        else
            Logging.@error "vcat data frames names not matching df: $(names(df)) - res: $(names(res))"
            break
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
                    Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
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
                    Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
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
Removes ohlcv data rows that are outside the date boundaries (nothing= no boundary) and adjusts ohlcv.ix to stay within the new data range.
"""
function timerangecut!(xc::XchCache, startdt, enddt)
    for ohlcv in CryptoXch.ohlcv(xc)
        (verbosity >= 3) && println("before Ohlcv.timerangecut!($ohlcv, $startdt, $enddt)")
        Ohlcv.timerangecut!(ohlcv, startdt, enddt)
        (verbosity >= 3) && println("after Ohlcv.timerangecut!($ohlcv, $startdt, $enddt)")
    end
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
        @warn "base=$base is unknown or invalid"
    end
    return ohlcv
end

"downloads missing data and merges with canned data then saves it as supplemented canned data"
function downloadupdate!(xc::XchCache, bases, enddt, period=Dates.Year(10))
    count = length(bases)
    for (ix, base) in enumerate(bases)
        # break
        println()
        println("$(EnvConfig.now()) start updating $base ($ix of $count)")
        startdt = enddt - period
        ohlcv = CryptoXch.cryptodownload(xc, base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        Ohlcv.write(ohlcv)
    end
end

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

_emptymarkets()::DataFrame = DataFrame(basecoin=String[], quotevolume24h=Float32[], pricechangepercent=Float32[], lastprice=Float32[], askprice=Float32[], bidprice=Float32[])

USDTMARKETFILE = "USDTmarket"

function _usdtmarketfilename(fileprefix, timestamp::Union{Nothing, DateTime}, ext="jdf")
    if isnothing(timestamp)
        cfgfilename = fileprefix
    else
        cfgfilename = join([fileprefix, Dates.format(timestamp, "yy-mm-dd")], "_")
    end
    return EnvConfig.datafile(cfgfilename, "TradeConfig", ".jdf")
end

"""
Returns a dataframe with 24h values of all USDT quotecoin bases that are not in baseignore list with the following columns:

- basecoin
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

In case of timesimulation(xc) == true a canned USDTmarket file will be used - if one is present.
"""
function getUSDTmarket(xc::XchCache; dt::DateTime=tradetime(xc))
    if timesimulation(xc)
        usdtdf = _emptymarkets()
        cfgfilename = _usdtmarketfilename(CryptoXch.USDTMARKETFILE, dt)
        if isdir(cfgfilename)
            (verbosity >= 1) && println("Start loading USDT market data from $cfgfilename for $dt ")
            usdtdf = DataFrame(JDF.loadjdf(cfgfilename))
            if isnothing(usdtdf)
                (verbosity >=2) && println("Loading USDT market data from $cfgfilename for $dt failed")
            else
                (verbosity >=2) && println("Loaded USDT market data for $(size(usdtdf, 1)) coins from $cfgfilename for $dt")
                # for row in eachrow(usdtdf)
                #     ohlcv = Ohlcv.defaultohlcv(row.basecoin)
                #     Ohlcv.read!(ohlcv)
                #     if size(ohlcv.df, 1) > 0
                #         dtix = Ohlcv.rowix(ohlcv, dt)
                #         Ohlcv.setix!(ohlcv, dtix)
                #         orow = Ohlcv.current(ohlcv)
                #         row.lastprice = orow.close
                #         row.askprice = row.lastprice * (1 + 0.0001)
                #         row.bidprice = row.lastprice * (1 - 0.0001)
                #     end
                # end
            end
            (verbosity >=2) && println("No USDT market data file $cfgfilename for $dt found")
        end
    else  # production
        usdtdf = Bybit.get24h(xc.bc)
        bq = [basequote(s) for s in usdtdf.symbol]  # create vector of pairs (basecoin, quotecoin)
        @assert length(bq) == size(usdtdf, 1)
        usdtdf[!, :basecoin] = [isnothing(bqe) ? missing : bqe.basecoin for bqe in bq]
        nbq = [!isnothing(bqe) && validbase(xc, bqe.basecoin) && (bqe.quotecoin == EnvConfig.cryptoquote) for bqe in bq]  # create binary vector as DataFrame filter
        usdtdf = usdtdf[nbq, :]
        bq = [bqe.basecoin for bqe in bq if !isnothing(bqe)]
        usdtdf = usdtdf[!, Not(:symbol)]
        # usdtdf = usdtdf[(usdtdf.quoteCoin .== "USDT") && (usdtdf.status .== "Trading"), :] - covered above by validbase
        # usdtdf = filter(row -> validbase(xc, row.basecoin), usdtdf) - covered above by validbase
        (verbosity >= 3) && println("writing USDTmarket file of size=$(size(usdtdf)) at enddt=$dt")
        JDF.savejdf(CryptoXch._usdtmarketfilename(CryptoXch.USDTMARKETFILE, dt), usdtdf)
    end
    return usdtdf
end

"Returns a DataFrame[:coin, :locked, :free] of wallet/portfolio balances"
function balances(xc::XchCache)
    if EnvConfig.configmode == production
        bdf = Bybit.balances(xc.bc)
        select = [!(coin in baseignore) || (coin == EnvConfig.cryptoquote) for coin in bdf[!, :coin]]
        bdf = bdf[select, :]
        return bdf
    else  # simulation
        if isnothing(xc)
            @error "cannot simulate balances() with uninitialized CryptoXch cache"
            return DataFrame()
        end
        return xc.assets
    end
end

"""
Appends a balances DataFrame with the USDT value of the coin asset using usdtdf[:lastprice] and returns it as DataFrame[:coin, :locked, :free, :usdtprice, :usdtvalue].
"""
function portfolio!(xc::XchCache, balancesdf=balances(xc), usdtdf=getUSDTmarket(xc))
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
                    @warn "found no data at $(xc.currentdt) for asset $ohlcv"  # (verbosity >= 3) &&
                    push!(usdtprice, 0f0)
                end
            end
        end
        portfoliodf.usdtprice = usdtprice
    end
    portfoliodf.usdtvalue = (portfoliodf.locked + portfoliodf.free) .* portfoliodf.usdtprice
    return portfoliodf
end

"Downloads all basecoins with USDT quote that shows a minimumdayquotevolume and saves it as canned data"
function downloadallUSDT(xc::XchCache, enddt, period=Dates.Year(10), minimumdayquotevolume = 10000000)
    df = getUSDTmarket(xc)
    df = df[df.quotevolume24h .> minimumdayquotevolume , :]
    bases = sort!(setdiff(df[!, :basecoin], baseignore))
    println("$(EnvConfig.now())downloading the following bases bases with $(EnvConfig.cryptoquote) quote: $bases")
    downloadupdate!(xc, bases, enddt, period)
    return df
end

openstatus(st::String)::Bool = st in ["New", "PartiallyFilled", "Untriggered"]
openstatus(stvec::AbstractVector{String})::Vector{Bool} = [openstatus(st) for st in stvec]

function _orderbase(xc::XchCache, orderid)
    ooix = findlast(oid -> oid == orderid, xc.orders[!, :orderid])
    coix = isnothing(ooix) ? findlast(oid -> oid == orderid, xc.closedorders[!, :orderid]) : nothing
    return isnothing(ooix) ? (isnothing(coix) ? nothing : basequote(xc.closedorders[coix, :symbol])[1]) : basequote(xc.orders[ooix, :symbol])[1]
end

_orderohlcv(xc::XchCache, orderid) = (base = _orderbase(xc,orderid); isnothing(base) ? nothing : xc.bases[base])
_ordercurrenttime(xc::XchCache, orderid) = (ohlcv = _orderohlcv(xc, orderid); isnothing(ohlcv) ? nothing : (ot = Ohlcv.dataframe(ohlcv).opentime; length(ot) > 0 ? ot[Ohlcv.ix(ohlcv)] : nothing))
_ordercurrentprice(xc::XchCache, orderid) = (ohlcv = _orderohlcv(xc, orderid); isnothing(ohlcv) ? nothing : (cl = Ohlcv.dataframe(ohlcv).close; length(cl) > 0 ? cl[Ohlcv.ix(ohlcv)] : nothing))

"Checks ohlcv since last check and marks order as executed if limitprice is exceeded"
function _updateorder!(xc::XchCache, orderix)
    xco = xc.orders[orderix, :]
    # if !openstatus(xco.status)
    #     return
    # end
    base, _ = basequote(xco.symbol)
    ohlcv = xc.bases[base]
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    ohlcvix = oix = Ohlcv.ix(ohlcv)
    if ohlcvdf.opentime[oix] == xco.lastcheck
        return
    end
    while (ohlcvdf.opentime[oix] > xco.lastcheck) && (oix > firstindex(ohlcvdf.opentime))
        oix -= 1
    end
    xco.lastcheck = ohlcvdf.opentime[ohlcvix]
    while oix <= ohlcvix
        if xco.side == "Buy"
            if ohlcvdf.low[oix] <= xco.limitprice
                xco.updated = ohlcvdf.opentime[ohlcvix]
                xco.avgprice = xco.limitprice
                xco.executedqty = xco.baseqty
                xco.status = "Filled"
                executedqty = xco.executedqty
                limitprice = xco.limitprice
                push!(xc.closedorders,xco)
                deleteat!(xc.orders, orderix)
                updateasset!(xc, EnvConfig.cryptoquote, -(executedqty * limitprice), 0)
                # update quote locked part with sum of open orders
                updateasset!(xc, base, 0, executedqty * (1 - xc.feerate))
                break
            end
        else # sell side
            if ohlcvdf.high[oix] >= xco.limitprice
                xco.updated = ohlcvdf.opentime[ohlcvix]
                xco.avgprice = xco.limitprice
                xco.executedqty = xco.baseqty
                xco.status = "Filled"
                executedqty = xco.executedqty
                limitprice = xco.limitprice
                push!(xc.closedorders,xco)
                deleteat!(xc.orders, orderix)
                updateasset!(xc, base, -executedqty, 0)
                # update base locked part with sum of open orders
                updateasset!(xc, EnvConfig.cryptoquote, 0, (executedqty * limitprice * (1 - xc.feerate)))
                break
            end
        end
        oix += 1
    end
end

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
    if EnvConfig.configmode == production
        oo = Bybit.openorders(xc.bc, symbol=symboltoken(base))
        return size(oo) == (0,0) ? emptyorders() : oo
    else  # simulation
        if isnothing(xc)
            @error "cannot simulate getopenorders() with uninitialized CryptoXch cache"
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

"Checks in sumulation buy or sell conditions and returns order index or nothign if not found"
function _orderrefresh(xc::XchCache, orderid)
    if isnothing(xc)
        @error "cannot simulate getorder() with uninitialized CryptoXch cache"
        return nothing
    end
    for oix in reverse(eachindex(xc.orders[!, :orderid]))
        _updateorder!(xc, oix)
    end
    return findlast(x -> x == orderid, xc.orders[!, :orderid])
end

"Returns a named tuple with elements equal to columns of getopenorders() dataframe of the identified order or `nothing` if order is not found"
function getorder(xc::XchCache, orderid)
    if EnvConfig.configmode == production
        return Bybit.order(xc.bc, orderid)
    else  # simulation
        orderindex = _orderrefresh(xc, orderid)
        coix = isnothing(orderindex) ? findlast(x -> x == orderid, xc.closedorders[!, :orderid]) : nothing
        return isnothing(orderindex) ? (isnothing(coix) ? nothing : NamedTuple(xc.closedorders[coix, :])) : NamedTuple(xc.orders[orderindex, :])
    end
end

"Returns orderid in case of a successful cancellation"
function cancelorder(xc::XchCache, base, orderid)
    if EnvConfig.configmode == production
        return Bybit.cancelorder(xc.bc, symboltoken(base), orderid)
    else  # simulation
        if isnothing(xc)
            @error "cannot simulate cancelorder() with uninitialized CryptoXch cache"
            return nothing
        end
        oix = findlast(x -> x == orderid, xc.orders[!, :orderid])
        if !isnothing(oix)
            xco = xc.orders[oix, :]
            base, _ = basequote(xco.symbol)
            ohlcv = xc.bases[base]
            ohlcvdf = Ohlcv.dataframe(ohlcv)
            xco.updated = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
            xco.status = "Cancelled"
            baseqty = xco.baseqty
            executedqty = xco.executedqty
            limitprice = xco.limitprice
            side = xco.side
            push!(xc.closedorders,xco)
            deleteat!(xc.orders, oix)
            if side == "Buy"
                qteqty = (baseqty - executedqty) * limitprice
                updateasset!(xc, EnvConfig.cryptoquote, -qteqty, qteqty)
            else # sell side
                baseqty = baseqty - executedqty
                updateasset!(xc, base, -baseqty, baseqty)
            end
            return orderid
        else
            return nothing
        end
    end
end

function _createordersimulation(xc::XchCache, base, side, baseqty, limitprice, freeasset)
    if isnothing(xc)
        @error "cannot simulate create buy/sell order() with uninitialized CryptoXch cache"
        return nothing
    end
    ohlcv = xc.bases[base]
    dtnow = ohlcv.df.opentime[ohlcv.ix]
    if isnothing(limitprice)
        limitprice = ohlcv.df[ohlcv.ix, :close]
        timeinforce = "PostOnly"
    else
        timeinforce = "GTC"
    end
    orderid = "$side$baseqty*$(round(limitprice, sigdigits=5))$base$(Dates.format(dtnow, "yymmddTHH:MM"))"
    if (side == "Buy")
        if (limitprice > (1+MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])
            @warn "limitprice $limitprice > max delta $((1+MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])"
            return nothing
        else
            limitprice = min(limitprice, ohlcv.df.close[ohlcv.ix])
            quoteqty = baseqty * limitprice
            println("buy order: quoteqty=$quoteqty = baseqty=$baseqty * limitprice=$limitprice")
            if _assetfreelocked(xc, EnvConfig.cryptoquote).free >= quoteqty
                push!(xc.orders, (orderid=orderid, symbol=symboltoken(base), side=side, baseqty=baseqty, ordertype="Limit", timeinforce=timeinforce, limitprice=limitprice, avgprice=0.0f0, executedqty=0.0f0, status="New", created=dtnow, updated=dtnow, rejectreason="NO ERROR", lastcheck=dtnow-Minute(1)))
                updateasset!(xc, EnvConfig.cryptoquote, quoteqty, -quoteqty)
            else
                @warn "$(Dates.format(dtnow, "yymmddTHH:MM")) $side of $base insufficient free assets: requested $quoteqty $(EnvConfig.cryptoquote) > available $(_assetfreelocked(xc, EnvConfig.cryptoquote).free) $freeasset"
                return nothing
            end
                end
    elseif (side == "Sell")
        if (limitprice < (1-MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])
            @warn "limitprice $limitprice < max delta $((1-MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])"
            return nothing
        else
            limitprice = max(limitprice, ohlcv.df.close[ohlcv.ix])
            if _assetfreelocked(xc, base).free >= baseqty
                push!(xc.orders, (orderid=orderid, symbol=symboltoken(base), side=side, baseqty=baseqty, ordertype="Limit", timeinforce=timeinforce, limitprice=limitprice, avgprice=0.0f0, executedqty=0.0f0, status="New", created=dtnow, updated=dtnow, rejectreason="NO ERROR", lastcheck=dtnow-Minute(1)))
                updateasset!(xc, freeasset, baseqty, -baseqty)
            else
                @warn "$(Dates.format(dtnow, "yymmddTHH:MM")) $side of $base insufficient free assets: requested $baseqty $base > available $(_assetfreelocked(xc, base).free) $base"
                return nothing
            end
        end
    end
    return orderid
end

"""
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` > current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createbuyorder(xc::XchCache, base::String; limitprice, basequantity, maker::Bool=false)
    base = uppercase(base)
    if EnvConfig.configmode == production
        return Bybit.createorder(xc.bc, symboltoken(base), "Buy", basequantity, limitprice, maker)
    else  # simulation
        return _createordersimulation(xc, base, "Buy", basequantity, limitprice, EnvConfig.cryptoquote)
    end
end

"""
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` < current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createsellorder(xc::XchCache, base::String; limitprice, basequantity, maker::Bool=true)
    base = uppercase(base)
    if EnvConfig.configmode == production
        return Bybit.createorder(xc.bc, symboltoken(base), "Sell", basequantity, limitprice, maker)
    else  # simulation
        return _createordersimulation(xc, base, "Sell", basequantity, limitprice, base)
    end
end

function changeorder(xc::XchCache, orderid; limitprice=nothing, basequantity=nothing)
    if EnvConfig.configmode == production
        oo = Bybit.order(xc.bc, orderid) #TODO in order to avoid this unnecessary order request the interface of Bybit.amendorder need to remove symbol param
        if isnothing(oo)
            return nothing
        end
        return Bybit.amendorder(xc.bc, oo.symbol, orderid; basequantity=basequantity, limitprice=limitprice)
    else  # simulation
        oix = _orderrefresh(xc, orderid)
        # if isnothing(oix) || !openstatus(xc.orders[oix, :status]) || (xc.orders[oix, :baseqty] <= xc.orders[oix, :executedqty])
        if isnothing(oix) || (xc.orders[oix, :baseqty] <= xc.orders[oix, :executedqty])
            return nothing
        end
        xco = xc.orders[oix, :]
        if isnothing(limitprice)
            limitdelta = 0.0f0
            limit = xco.limitprice
        else
            limitdelta = limitprice - xco.limitprice
            limit = limitprice
        end
        if isnothing(basequantity)
            baseqtydelta = 0.0f0
            baseqty = xco.baseqty
        else
            basequantity = max(basequantity, xco.executedqty)
            baseqtydelta = basequantity - xco.baseqty
            baseqty = basequantity
        end
        ohlcv = xc.bases[basequote(xco.symbol)[1]]
        dtnow = ohlcv.df.opentime[ohlcv.ix]
        xco.updated = dtnow

        freeasset, freeassetqty = xco.side == "Buy" ? (EnvConfig.cryptoquote, baseqty * limitdelta + limit * baseqtydelta) : (basequote(xco.symbol)[1], baseqtydelta)
        xco.baseqty = baseqty
        xco.limitprice = limit
        if baseqty <= xco.executedqty # close order
            xco.status = "Filled"
            push!(xc.closedorders,xco)
            deleteat!(xc.orders, oix)
        end
        updateasset!(xc, freeasset, freeassetqty, -freeassetqty)
        return orderid
    end
end

ASSETSFILENAME = "XchCacheAssets"
ORDERSFILENAME = "XchCacheOrders"
CLOSEDORDERSFILENAME = "XchCacheClosedorders"

function write(xc::XchCache)

end

end  # of module
