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
    feerate  # 0.001 = 0.1% at Bybit maker = taker fee
    function XchCache(bybitinit::Bool)
        return new(emptyorders(), emptyorders(), emptyassets(), Dict(), bybitinit ? Bybit.BybitCache() : nothing, 0.001)
    end
end


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

validsymbol(xc::XchCache, sym::String) = !isnothing(Bybit.symbolinfo(xc.bc, sym)) || (sym in TestOhlcv.testbasecoin())

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
    return isnothing(coinix) ? (free=0.0f0, locked=0.0f0) : (free=xc.assets[coinix, :free], locked=xc.assets[coinix, :locked])
end

function _updateasset!(xc::XchCache, coin::String, lockedqty, freeqty)
    coin = uppercase(coin)
    coinix = findfirst(x -> x == coin, xc.assets[!, :coin])
    if isnothing(coinix)
        push!(xc.assets, (coin=coin, locked = lockedqty, free=freeqty))
    else
        xc.assets[coinix, :free] += freeqty
        xc.assets[coinix, :locked] += lockedqty
        #TODO set to zero if amount less than precision
    end
end

function updatecache(xc::XchCache; ohlcv=nothing, orders=nothing, assets=nothing)
    xc = isnothing(xc) ? XchCache(false) : xc
    if !isnothing(ohlcv)
        xc.bases[ohlcv.base] = ohlcv
    end
    if !isnothing(orders)
        xc.orders = orders
    end
    if !isnothing(assets)
        xc.assets = assets
    end
end

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
    timerangecut!(ohlcv, startdt, enddt)
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

"""
Initializes the undrelying exchange.
"""
function XchCache(bases::Vector, startdt=nothing, enddt=nothing, usdtbudget=10000)::XchCache
    xc = XchCache(true)
    bases = uppercase.(bases)
    sellbases = union(bases, setdiff(assetbases(xc), basestablecoin))
    oo = CryptoXch.getopenorders(xc)
    if size(oo, 1) > 0
        oo = DataFrame(CryptoXch.basequote.(oo.symbol))
        sellbases = union(sellbases, oo[!, :basecoin])
    end
    sellbases = setdiff(sellbases, [EnvConfig.cryptoquote])
    for base in sellbases  # sellbases is superset of bases
        addbase!(xc, base, startdt, enddt)
        # println("startdt=$startdt enddt=$enddt xc.bases[base]=$(xc.bases[base])")
    end
    if EnvConfig.configmode != production
        # push startbudget onto balance wallet for backtesting/simulation
        push!(xc.assets, (coin=uppercase(EnvConfig.cryptoquote), locked = 0.0f0, free=usdtbudget))
    end
    return xc
end

tradetime(xc::XchCache) = EnvConfig.configmode == production ? Bybit.servertime(xc.bc) : Dates.now(UTC)

function _sleepuntil(xc::XchCache, dt::DateTime)
    sleepperiod = dt - tradetime(xc)
    if sleepperiod <= Dates.Second(0)
        return
    end
    if sleepperiod > Minute(1)
        println("long sleep $(floor(sleepperiod, Minute))")
    end
    # println("sleeping $(floor(sleepperiod, Second))")
    sleep(sleepperiod)
end

"Sleeps until `datetime` if reached if `datetime` is in the future, set the *current* time and updates ohlcv if required"
function setcurrenttime!(xc::XchCache, base::String, datetime::DateTime)
    ohlcv = xc.bases[base]
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    dt = floor(datetime, intervalperiod(ohlcv.interval))
    if (size(ohlcvdf, 1)) == 0 || (dt > ohlcvdf.opentime[ohlcv.ix])
        _sleepuntil(xc, dt + Minute(1))  #  + Minute(1) to get data of the full minute and not the partial last minute
    end
    if (size(ohlcvdf, 1) == 0) || (dt > ohlcvdf.opentime[end])
        cryptoupdate!(xc, ohlcv, (size(ohlcvdf, 1) == 0 ? dt : ohlcvdf.opentime[begin]), dt)
    end
    Ohlcv.setix!(ohlcv, Ohlcv.rowix(ohlcv, dt))
    if (size(Ohlcv.dataframe(ohlcv), 1) > 0) && (Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)] != dt)
        @warn "setcurrenttime!($base, $dt) failed, opentime[ix]=$(Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)])"
        println("setcurrenttime!($base, $dt) failed, opentime[ix]=$(Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)])")
    end
end

function setcurrenttime!(xc::XchCache, datetime::DateTime)
    for base in keys(xc.bases)
        setcurrenttime!(xc, base, datetime)
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
    if base in TestOhlcv.testbasecoin()
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
    updatecache(xc, ohlcv=ohlcv)
    return ohlcv
end

"""
Removes ohlcv data rows that are outside the date boundaries (nothing= no boundary) and adjusts ohlcv.ix to stay within the new data range.
"""
function timerangecut!(ohlcv, startdt, enddt)
    if isnothing(ohlcv) || isnothing(ohlcv.df) || (size(ohlcv.df, 1) == 0)
        return
    end
    ixdt = ohlcv.df.opentime[ohlcv.ix]
    if !isnothing(startdt) && !isnothing(enddt)
        subset!(ohlcv.df, :opentime => t -> floor(startdt, intervalperiod(ohlcv.interval)) .<= t .<= floor(enddt, intervalperiod(ohlcv.interval)))
    elseif !isnothing(startdt)
        subset!(ohlcv.df, :opentime => t -> floor(startdt, intervalperiod(ohlcv.interval)) .<= t)
    elseif !isnothing(enddt)
        subset!(ohlcv.df, :opentime => t -> t .<= floor(enddt, intervalperiod(ohlcv.interval)))
    end
    ohlcv.ix = Ohlcv.rowix(ohlcv, ixdt)
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
    if isnothing(Bybit.symbolinfo(xc.bc, symboltoken(base))) && !(base in TestOhlcv.testbasecoin())
        @warn "symbol $(symboltoken(base)) of base=$base is unknown"
    else
        if Ohlcv.file(ohlcv).existing
            Ohlcv.read!(ohlcv)
        end
        cryptoupdate!(xc, ohlcv, startdt, enddt)
        ohlcv.ix = firstindex(ohlcv.df, 1)
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
_isleveraged(token) = (token[end] in ['S', 'L']) && isdigit(token[end-1])

"""
Returns a dataframe with 24h values of all USDT quotecoin bases that are not in baseignore list with the following columns:

- basecoin
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

"""
function getUSDTmarket(xc::XchCache)
    if EnvConfig.configmode == production
        usdtdf = Bybit.get24h(xc.bc)
        bq = [basequote(s) for s in usdtdf.symbol]
        @assert length(bq) == size(usdtdf, 1)
        nbq = [!isnothing(bqe) for bqe in bq]
        usdtdf = usdtdf[nbq, :]
        bq = [bqe.basecoin for bqe in bq if !isnothing(bqe)]
        @assert length(bq) == size(usdtdf, 1)
        usdtdf[!, :basecoin] = bq
        usdtdf = usdtdf[!, Not(:symbol)]
        # usdtdf = usdtdf[(usdtdf.quoteCoin .== "USDT") && (usdtdf.status .== "Trading"), :]
        usdtdf = filter(row -> !(row.basecoin in baseignore), usdtdf)
        usdtdf = filter(row -> !_isleveraged(row.basecoin), usdtdf)
        #TODO remove all entries with a base that end <unsigned digit>L/S because those are leveraged tokens
    else  # simulation
        #TODO get all canned data with common latest update and use those. For that purpose the OHLCV.OhlcvFiles iterator was created.
        usdtdf = _emptymarkets()
        if isnothing(xc)
            @error "cannot simulate getUSDTmarket() with uninitialized CryptoXch cache"
            return usdtdf
        end
        for base in keys(xc.bases)
            ohlcv = xc.bases[base]
            if size(ohlcv.df, 1) > 0
                ohlcvdf = subset(ohlcv.df, :opentime => t -> (ohlcv.df.opentime[ohlcv.ix] - Dates.Day(1)) .<= t .<= ohlcv.df.opentime[ohlcv.ix], view=true)
                # println("usdtdf:$(typeof(usdtdf)), base:$(typeof(base)), $(sum(ohlcvdf.basevolume) * ohlcvdf.close[end]), $((ohlcvdf.close[end] - ohlcvdf.open[begin]) / ohlcvdf.open[begin] * 100), $(ohlcvdf.close[end]), $(ohlcvdf.high[end]), $(ohlcvdf.low[end])")
                push!(usdtdf, (base, sum(ohlcvdf.basevolume) * ohlcvdf.close[end], (ohlcvdf.close[end] - ohlcvdf.open[begin]) / ohlcvdf.open[begin] * 100, ohlcvdf.close[end], ohlcvdf.high[end], ohlcvdf.low[end]))
            end
        end
    end
    return usdtdf
end

"Returns a DataFrame[:accounttype, :coin, :locked, :free] of wallet/portfolio balances"
function balances(xc::XchCache)
    if EnvConfig.configmode == production
        bdf = Bybit.balances(xc.bc)
        bdf.coin = uppercase.(bdf.coin)
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
Appends a balances DataFrame with the USDT value of the coin asset using usdtdf[:lastprice] and returns it as DataFrame[:coin, :locked, :free, :usdtprice].
"""
function portfolio!(xc::XchCache, balancesdf=balances(xc), usdtdf=getUSDTmarket(xc))
    balancesdf = leftjoin(balancesdf, usdtdf[!, [:basecoin, :lastprice]], on = :coin => :basecoin)
    balancesdf.lastprice = coalesce.(balancesdf.lastprice, 1.0f0)
    balancesdf.usdtvalue = (balancesdf.locked + balancesdf.free) .* balancesdf.lastprice
    rename!(balancesdf, :lastprice => "usdtprice")
    return balancesdf
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
    # if !openstatus(xc.orders[orderix, :status])
    #     return
    # end
    base, _ = basequote(xc.orders[orderix, :symbol])
    ohlcv = xc.bases[base]
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    oix = Ohlcv.ix(ohlcv)
    if ohlcvdf.opentime[oix] == xc.orders[orderix, :lastcheck]
        return
    end
    while (ohlcvdf.opentime[oix] > xc.orders[orderix, :lastcheck]) && (oix > firstindex(ohlcvdf.opentime))
        oix -= 1
    end
    xc.orders[orderix, :lastcheck] = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
    while oix <= Ohlcv.ix(ohlcv)
        if xc.orders[orderix, :side] == "Buy"
            if ohlcvdf.low[oix] <= xc.orders[orderix, :limitprice]
                xc.orders[orderix, :updated] = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
                xc.orders[orderix, :avgprice] = xc.orders[orderix, :limitprice]
                xc.orders[orderix, :executedqty] = xc.orders[orderix, :baseqty]
                xc.orders[orderix, :status] = "Filled"
                _updateasset!(xc, EnvConfig.cryptoquote, -(xc.orders[orderix, :executedqty] * xc.orders[orderix, :limitprice]), 0)
                _updateasset!(xc, base, 0, xc.orders[orderix, :executedqty] * (1 - xc.feerate))
                push!(xc.closedorders,xc.orders[orderix, :])
                deleteat!(xc.orders, orderix)
                break
            end
        else # sell side
            if ohlcvdf.high[oix] >= xc.orders[orderix, :limitprice]
                xc.orders[orderix, :updated] = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
                xc.orders[orderix, :avgprice] = xc.orders[orderix, :limitprice]
                xc.orders[orderix, :executedqty] = xc.orders[orderix, :baseqty]
                xc.orders[orderix, :status] = "Filled"
                _updateasset!(xc, base, -xc.orders[orderix, :executedqty], 0)
                _updateasset!(xc, EnvConfig.cryptoquote, 0, (xc.orders[orderix, :executedqty] * xc.orders[orderix, :limitprice] * (1 - xc.feerate)))
                push!(xc.closedorders,xc.orders[orderix, :])
                deleteat!(xc.orders, orderix)
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
            base, _ = basequote(xc.orders[oix, :symbol])
            ohlcv = xc.bases[base]
            ohlcvdf = Ohlcv.dataframe(ohlcv)
            xc.orders[oix, :updated] = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
            xc.orders[oix, :status] = "Cancelled"
            if xc.orders[oix, :side] == "Buy"
                qteqty = (xc.orders[oix, :baseqty] - xc.orders[oix, :executedqty]) * xc.orders[oix, :limitprice]
                _updateasset!(xc, EnvConfig.cryptoquote, -qteqty, qteqty)
            else # sell side
                baseqty = xc.orders[oix, :baseqty] - xc.orders[oix, :executedqty]
                _updateasset!(xc, EnvConfig.cryptoquote, -baseqty, baseqty)
            end
            push!(xc.closedorders,xc.orders[oix, :])
            deleteat!(xc.orders, oix)
            return orderid
        else
            return nothing
        end
    end
end

function _createordersimulation(xc::XchCache, base, side, baseqty, limitprice, freeasset, freeassetqty)
    if isnothing(xc)
        @error "cannot simulate create buy/sell order() with uninitialized CryptoXch cache"
        return nothing
    end
    ohlcv = xc.bases[base]
    if (side == "Buy") && (limitprice > (1+MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])
        @warn "limitprice $limitprice > max delta $((1+MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])"
        return nothing
    end
    if (side == "Sell") && (limitprice < (1-MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])
        @warn "limitprice $limitprice < max delta $((1-MAXLIMITDELTA) * ohlcv.df.close[ohlcv.ix])"
        return nothing
    end
    baseqty = round(baseqty, sigdigits=5)
    dtnow = ohlcv.df.opentime[ohlcv.ix]
    orderid = "$side$baseqty*$(round(limitprice, sigdigits=5))$base$(Dates.format(dtnow, "yymmddTHH:MM"))"
    # println("_assetfreelocked($freeasset)=$(_assetfreelocked(freeasset)) >= freeassetqty=$freeassetqty")
    timeinforce = "GTC"  # not yet "PostOnly" because maker fee == taker fee without VIP status
    if _assetfreelocked(xc, freeasset).free >= freeassetqty
        push!(xc.orders, (orderid=orderid, symbol=symboltoken(base), side=side, baseqty=baseqty, ordertype="Limit", timeinforce=timeinforce, limitprice=limitprice, avgprice=0.0f0, executedqty=0.0f0, status="New", created=dtnow, updated=dtnow, rejectreason="NO ERROR", lastcheck=dtnow-Minute(1)))
        _updateasset!(xc, freeasset, freeassetqty, -freeassetqty)
        # if ((side == "Buy") && (limitprice >= ohlcv.df.close[ohlcv.ix])) || ((side == "Sell") && (limitprice <= ohlcv.df.close[ohlcv.ix]))
        #     push!(xc.orders, (orderid=orderid, symbol=symboltoken(base), side=side, baseqty=baseqty, ordertype="Limit", timeinforce=timeinforce, limitprice=limitprice, executedqty=0.0f0, status="Rejected", created=dtnow, updated=dtnow, rejectreason="Rejected because PostOnly prevents taker orders", lastcheck=dtnow))
        # else
        #     push!(xc.orders, (orderid=orderid, symbol=symboltoken(base), side=side, baseqty=baseqty, ordertype="Limit", timeinforce=timeinforce, limitprice=limitprice, executedqty=0.0f0, status="New", created=dtnow, updated=dtnow, rejectreason="NO ERROR", lastcheck=dtnow))
        #     _updateasset!(xc, freeasset, freeassetqty, -freeassetqty)
        # end
        # println(xc.orders[end, :])
    else
        @warn "$(Dates.format(dtnow, "yymmddTHH:MM")) insufficient free assets: requested $freeassetqty $freeasset > available $(_assetfreelocked(xc, freeasset).free) $freeasset"
        return nothing
    end
    # println("_createordersimulation OK orderid=$orderid")
    return orderid
end

"""
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` > current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createbuyorder(xc::XchCache, base::String; limitprice, basequantity, maker::Bool=true)
    base = uppercase(base)
    if EnvConfig.configmode == production
        return Bybit.createorder(xc.bc, symboltoken(base), "Buy", basequantity, limitprice, maker)
    else  # simulation
        return _createordersimulation(xc, base, "Buy", basequantity, limitprice, EnvConfig.cryptoquote, basequantity * limitprice)
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
        return _createordersimulation(xc, base, "Sell", basequantity, limitprice, base, basequantity)
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
        if isnothing(limitprice)
            limitdelta = 0.0f0
            limit = xc.orders[oix, :limitprice]
        else
            limitdelta = limitprice - xc.orders[oix, :limitprice]
            limit = limitprice
        end
        if isnothing(basequantity)
            baseqtydelta = 0.0f0
            baseqty = xc.orders[oix, :baseqty]
        else
            basequantity = max(basequantity, xc.orders[oix, :executedqty])
            baseqtydelta = basequantity - xc.orders[oix, :baseqty]
            baseqty = basequantity
        end
        ohlcv = xc.bases[basequote(xc.orders[oix, :symbol])[1]]
        dtnow = ohlcv.df.opentime[ohlcv.ix]
        xc.orders[oix, :updated] = dtnow

        freeasset, freeassetqty = xc.orders[oix, :side] == "Buy" ? (EnvConfig.cryptoquote, baseqty * limitdelta + limit * baseqtydelta) : (basequote(xc.orders[oix, :symbol])[1], baseqtydelta)
        _updateasset!(xc, freeasset, freeassetqty, -freeassetqty)
        xc.orders[oix, :baseqty] = baseqty
        xc.orders[oix, :limitprice] = limit
        if baseqty <= xc.orders[oix, :executedqty] # close order
            xc.orders[oix, :status] = "Filled"
            push!(xc.closedorders,xc.orders[oix, :])
            deleteat!(xc.orders, oix)
        end
        return xc.orders[oix, :orderid]
    end
end

ASSETSFILENAME = "XchCacheAssets"
ORDERSFILENAME = "XchCacheOrders"
CLOSEDORDERSFILENAME = "XchCacheClosedorders"

function write(xc::XchCache)

end

end  # of module
