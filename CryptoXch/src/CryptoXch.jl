# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])


module CryptoXch

using Dates, DataFrames, DataAPI, JDF, CSV, Logging
using Bybit, EnvConfig, Ohlcv, TestOhlcv
import Ohlcv: intervalperiod

mutable struct XchCache
    orders::DataFrame
    assets:: DataFrame
    bases::Dict{String, Ohlcv.OhlcvData}
end

xc = nothing

basenottradable = ["boba",
    "btt", "bcc", "ven", "pax", "bchabc", "bchsv", "usds", "nano", "usdsb", "erd", "npxs", "storm", "hc", "mco",
    "bull", "bear", "ethbull", "ethbear", "eosbull", "eosbear", "xrpbull", "xrpbear", "strat", "bnbbull", "bnbbear",
    "xzc", "gxs", "lend", "bkrw", "dai", "xtzup", "xtzdown", "bzrx", "eosup", "eosdown", "ltcup", "ltcdown"]
basestablecoin = ["usdt", "tusd", "busd", "usdc", "eur"]
baseignore = [""]
baseignore = uppercase.(append!(baseignore, basestablecoin, basenottradable))
minimumquotevolume = 10  # USDT

MAXLIMITDELTA = 0.1
defaultcryptoexchange = "bybit"  # "binance"

"Returns a `(free, locked)` named tuple with the amount of `free` and `locked` amounts of coin in portfolio assets"
function _assetfreelocked(coin::String)
    coinix = findfirst(x -> x == uppercase(coin), xc.assets.coin)
    return isnothing(coinix) ? (free=0.0f0, locked=0.0f0) : (free=xc.assets.free[coinix], locked=xc.assets.locked[coinix])
end

function _updateasset!(coin::String, lockedqty, freeqty)
    coin = uppercase(coin)
    coinix = findfirst(x -> x == coin, xc.assets.coin)
    if isnothing(coinix)
        push!(xc.assets, (coin=coin, locked = lockedqty, free=freeqty))
    else
        xc.assets.free[coinix] += freeqty
        xc.assets.locked[coinix] += lockedqty
    end
end

function updatecache(;ohlcv=nothing, orders=nothing, assets=nothing)
    global xc
    if EnvConfig.configmode != production
        xc = isnothing(xc) ? XchCache(false, DataFrame(), DataFrame(), Dict(), DateTime("2000-01-01T00:00:00")) : xc
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
end

emptyassets()::DataFrame = DataFrame(coin=String[], locked=Float32[], free=Float32[])

"provides an empty dataframe for simulation (with lastcheck as extra column)"
emptyorders()::DataFrame = DataFrame(orderid=String[], symbol=String[], side=String[], baseqty=Float32[], ordertype=String[], timeinforce=String[], limitprice=Float32[], executedqty=Float32[], status=String[], created=DateTime[], updated=DateTime[], rejectreason=String[], lastcheck=DateTime[])

"""
Initializes the undrelying exchange.
If EnvConfig.configmode != production then set up the backtest/simulation.
Parameters are only relevant for != production mode.
"""
function init(; bases=["btc"], startdt=nothing, enddt=nothing, usdtbudget=10000)
    global xc
    xc = XchCache(emptyorders(), emptyassets(), Dict())
    for base in bases
        ohlcv = Ohlcv.defaultohlcv(base, "1m")
        Ohlcv.read!(ohlcv)
        startdt = isnothing(startdt) ? ohlcv.df.opentime[begin] : startdt
        enddt = isnothing(enddt) ? ohlcv.df.opentime[end] : enddt
        timerangecut!(ohlcv, startdt, enddt)
        ohlcv.ix = firstindex(ohlcv.df, 1)
        xc.bases[base] = ohlcv
    end
    if EnvConfig.configmode != production
        earlydummydate = DateTime("2000-01-01T00:00:00")
        push!(xc.assets, (coin=uppercase(EnvConfig.cryptoquote), locked = 0.0f0, free=usdtbudget))
    end
    Bybit.init()
end

"Set the *current* time in case of backtests/simulations, i.e. EnvConfig.configmode != production"
function setcacheindex(datetime)
    if EnvConfig.configmode != production
        if isnothing(xc)
            @error "cannot set cacheindex with uninitialized CryptoXch cache"
            return
        end
        for base in bases
            ohlcv = xc.bases[base]
            ohlcvdf = Ohlcv.dataframe(ohlcv)
            dt = floor(datetime, intervalperiod(ohlcv.interval))
            ix = ohlcv.ix
            if dt > ohlcvdf.opentime[ohlcv.ix]
                while ((ohlcv.ix + 1) <= lastindex(ohlcvdf.opentime)) && (dt > ohlcvdf.opentime[ohlcv.ix + 1])
                    ohlcv.ix += 1
                end
            elseif dt < ohlcvdf.opentime[ohlcv.ix]
                while ((ohlcv.ix - 1) >= firstindex(ohlcvdf.opentime)) && (dt < ohlcvdf.opentime[ohlcv.ix - 1])
                    ohlcv.ix -= 1
                end
            end
        end
    else
        @error "cannot work with CryptoXch cache for simulation with EnvConfig.configmode == production"
    end
end

symbolusdt(base) = isnothing(base) ? nothing : uppercase(base * EnvConfig.cryptoquote)


"""
Requests base/USDT from start until end (both including) in interval frequency but will return a maximum of 1000 entries.
Subsequent calls are required to get > 1000 entries.
Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""
function _ohlcfromexchange(base::String, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    df = nothing
    symbol = uppercase(base*cryptoquote)
    df = Bybit.getklines(symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
    Ohlcv.addpivot!(df)
    #TODO add TestOhlcv here for special test symbols - no need to allow them only in test mode
    return df
end

"Returns nothing but displays the latest klines/ohlcv data - test implementation"
function _getlastminutesdata()
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Minute(7)
    res = _ohlcfromexchange("BTCUSDT", startdt, enddt)
    # display(nrow(res))
    display(last(res, 3))
    # display(first(res, 3))
    enddt = Dates.now(Dates.UTC)
    res2 = _ohlcfromexchange("BTCUSDT", enddt - Dates.Second(1), enddt)
    # display(res)
    display(nrow(res2))
    display(last(res2, 3))
    println("dates equal? $(res[end, :opentime]==res2[end, :opentime])")
    # display(first(res2, 3))
    # display(res[:body][1:3, :])
    # display(res[:body][end-3:end, :])
end

"""
Requests base/USDT from start until end (both including) in interval frequency. If required Bybit is internally called several times to fill the request.

Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

time gaps will not be filled
"""
function _gethistoryohlcv(base::String, startdt::DateTime, enddt::DateTime=Dates.now(Dates.UTC), interval="1m")
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
        res = _ohlcfromexchange(base, startdt, enddt, interval)
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
function cryptoupdate!(ohlcv, startdt, enddt)
    base = ohlcv.base
    interval = ohlcv.interval
    println("Requesting $base $interval intervals from $startdt until $enddt")
    if enddt <= startdt
        Logging.@warn "Invalid datetime range: end datetime $enddt <= start datetime $startdt"
        return ohlcv
    end
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    olddf = Ohlcv.dataframe(ohlcv)
    if size(olddf, 1) > 0  # there is already data available
        if (startdt < olddf[begin, :opentime])
            # correct enddt in each case (gap between new and old range or range overlap) to avoid time range gaps
            tmpdt = olddf[begin, :opentime] - intervalperiod(interval)
            # get data of a timerange before the already available data
            newdf = _gethistoryohlcv(base, startdt, tmpdt, interval)
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
            newdf = _gethistoryohlcv(base, tmpdt, enddt, interval)
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
        newdf = _gethistoryohlcv(base, startdt, enddt, interval)
        Ohlcv.setdataframe!(ohlcv, newdf)
    end
    updatecache(ohlcv=ohlcv)
    return ohlcv
end

function timerangecut!(ohlcv, startdt, enddt)
    if isnothing(ohlcv) || isnothing(ohlcv.df)
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
    if ixdt <= startdt
        ohlcv.ix = firstindex(ohlcv.df.opentime)
    elseif ixdt >= enddt
        ohlcv.ix = lastindex(ohlcv.df.opentime)
    else
        ohlcv.ix = Ohlcv.rowix(ohlcv, ixdt)
        if isnothing(ohlcv.ix)
            @warn "unexpected missing opentime of $ixdt in ohlcv: $ohlcv"
            ohlcv.ix = firstindex(ohlcv.df.opentime)
        end
    end
end

"""
Returns the OHLCV data of the requested time range by first checking the stored cache data and if unsuccessful requesting it from the Exchange.

    - *base* identifier and interval specify what data is requested - the result will be returned as OhlcvData structure
    - startdt and enddt are DateTime stamps that specify the requested time range
    - any gap to chached data will be closed when asking for missing data from Bybit
"""
function cryptodownload(base, interval, startdt, enddt)::OhlcvData
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.setinterval!(ohlcv, interval)
    Ohlcv.read!(ohlcv)
    cryptoupdate!(ohlcv, startdt, enddt)
    return ohlcv
end

function downloadupdate!(bases, enddt, period=Dates.Year(4))
    count = length(bases)
    for (ix, base) in enumerate(bases)
        # break
        println()
        println("$(EnvConfig.now()) updating $base ($ix of $count)")
        startdt = enddt - period
        CryptoXch.cryptodownload(base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
    end
end

ceilbase(base, qty) = base == "usdt" ? ceil(qty, digits=3) : ceil(qty, digits=5)
floorbase(base, qty) = base == "usdt" ? floor(qty, digits=3) : floor(qty, digits=5)
roundbase(base, qty) = base == "usdt" ? round(qty, digits=3) : round(qty, digits=5)
# TODO read base specific digits from binance and use them base specific

onlyconfiguredsymbols(symbol) =
    endswith(symbol, uppercase(EnvConfig.cryptoquote)) &&
    !(uppercase(symbol[1:end-length(EnvConfig.cryptoquote)]) in baseignore)

"Returns pair of base and quotecoin if quotecoin == EnvConfig.cryptoquote (USDT) else `nothing` is returned"
function basequote(symbol)
    symbol = uppercase(symbol)
    range = findfirst(uppercase(EnvConfig.cryptoquote), symbol)
    return isnothing(range) ? nothing : (basecoin = symbol[1:end-length(EnvConfig.cryptoquote)], quotecoin = EnvConfig.cryptoquote)
end

_emptymarkets()::DataFrame = DataFrame(basecoin=String[], quotevolume24h=Float32[], pricechangepercent=Float32[], lastprice=Float32[], askprice=Float32[], bidprice=Float32[])
"""
Returns a dataframe with 24h values of all USDT quotecoin bases that are not in baseignore list with the following columns:

- basecoin
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

"""
function getUSDTmarket()
    if EnvConfig.configmode == production
        usdtdf = Bybit.get24h()
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
        return usdtdf
    else  # simulation
        #TODO get all canned data with common latest update and use those. For that purpose the OHLCV.OhlcvFiles iterator was created.
        usdtdf = _emptymarkets()
        if isnothing(xc)
            @error "cannot simulate getUSDTmarket() with uninitialized CryptoXch cache"
            return usdtdf
        end
        for base in keys(xc.bases)
            ohlcv = xc.bases[base]
            ohlcvdf = subset(ohlcv.df, :opentime => t -> (ohlcv.df.opentime[ohlcv.ix] - Dates.Day(1)) .<= t .<= ohlcv.df.opentime[ohlcv.ix], view=true)
            # println("usdtdf:$(typeof(usdtdf)), base:$(typeof(base)), $(sum(ohlcvdf.basevolume) * ohlcvdf.close[end]), $((ohlcvdf.close[end] - ohlcvdf.open[begin]) / ohlcvdf.open[begin] * 100), $(ohlcvdf.close[end]), $(ohlcvdf.high[end]), $(ohlcvdf.low[end])")
            push!(usdtdf, (base, sum(ohlcvdf.basevolume) * ohlcvdf.close[end], (ohlcvdf.close[end] - ohlcvdf.open[begin]) / ohlcvdf.open[begin] * 100, ohlcvdf.close[end], ohlcvdf.high[end], ohlcvdf.low[end]))
        end
        return usdtdf
    end
end

"Returns a DataFrame[:accounttype, :coin, :locked, :free] of wallet/portfolio balances"
function balances()
    if EnvConfig.configmode == production
        bdf = Bybit.balances()
        bdf.coin = lowercase.(bdf.coin)
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
function portfolio!(balancesdf=balances(), usdtdf=getUSDTmarket())
    balancesdf = leftjoin(balancesdf, usdtdf[!, [:basecoin, :lastprice]], on = :coin => :basecoin)
    balancesdf.lastprice = coalesce.(balancesdf.lastprice, 1.0f0)
    rename!(balancesdf, :lastprice => "usdtprice")
    return balancesdf
end

function downloadallUSDT(enddt, period=Dates.Year(4), minimumdayquotevolume = 10000000)
    df = getUSDTmarket()
    df = df[df.quotevolume24h .> minimumdayquotevolume , :]
    count = size(df, 1)
    for (ix, base) in enumerate(df[!, :base])
        break
        println()
        println("$(EnvConfig.now()) updating $base ($ix of $count)")
        startdt = enddt - period
        CryptoXch.cryptodownload(base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
    end
    return df
end

_openstatus(st::String)::Bool = st in ["New", "PartiallyFilled", "Untriggered"]
_openstatus(stvec::Vector{String})::Vector{Bool} = [st in ["New", "PartiallyFilled", "Untriggered"] for st in stvec]
_orderix(orderid) = findfirst(oid -> oid == orderid, xc.orders.orderid)
_orderbase(orderid) = (oix = _orderix(orderid); isnothing(oix) ? nothing : basequote(xc.orders[oix, :symbol])[1])
_orderohlcv(orderid) = (base = _orderbase(orderid); isnothing(base) ? nothing : xc.bases[base])
_ordercurrenttime(orderid) = (ohlcv = _orderohlcv(orderid); isnothing(ohlcv) ? nothing : (ot = Ohlcv.dataframe(ohlcv).opentime; length(ot) > 0 ? ot[Ohlcv.ix(ohlcv)] : nothing))
_ordercurrentprice(orderid) = (ohlcv = _orderohlcv(orderid); isnothing(ohlcv) ? nothing : (cl = Ohlcv.dataframe(ohlcv).close; length(cl) > 0 ? cl[Ohlcv.ix(ohlcv)] : nothing))

"Relevant for simulation/backtest: Changes all base ohlcv ix indices to be opentime[ix] == newcurrent or closest opentime[ix] < newcurrent. "
function setcurrenttime!(newcurrent::DateTime)
    if !isnothing(xc)
        for ohlcv in values(xc.bases)
            ix = Ohlcv.ix(ohlcv)
            ot = Ohlcv.dataframe(ohlcv).opentime
            if length(ot) > 0
                while (ix < lastindex(ot)) && (newcurrent >= ot[ix+1])
                    ix += 1
                end
                while (ix > firstindex(ot)) && (newcurrent < ot[ix-1])
                    ix -= 1
                end
                Ohlcv.setix!(ohlcv, ix)
            end
        end
    end
end

"Checks ohlcv since last check and marks order as executed if limitprice is exceeded"
function _updateorder!(orderix)
    if !_openstatus(xc.orders[orderix, :status])
        return
    end
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
    while oix <= Ohlcv.ix(ohlcv)
        if xc.orders[orderix, :side] == "Buy"
            if ohlcvdf.low[oix] <= xc.orders[orderix, :limitprice]
                xc.orders[orderix, :updated] = ohlcvdf.opentime[oix]
                xc.orders[orderix, :executedqty] = xc.orders[orderix, :baseqty]
                xc.orders[orderix, :status] = "Filled"
                _updateasset!(EnvConfig.cryptoquote, -(xc.orders[orderix, :executedqty] * xc.orders[orderix, :limitprice]), 0)
                _updateasset!(base, 0, xc.orders[orderix, :executedqty])
                break
            end
        else # sell side
            if ohlcvdf.high[oix] >= xc.orders[orderix, :limitprice]
                xc.orders[orderix, :updated] = ohlcvdf.opentime[oix]
                xc.orders[orderix, :executedqty] = xc.orders[orderix, :baseqty]
                xc.orders[orderix, :status] = "Filled"
                _updateasset!(base, -xc.orders[orderix, :executedqty], 0)
                _updateasset!(EnvConfig.cryptoquote, 0, (xc.orders[orderix, :executedqty] * xc.orders[orderix, :limitprice]))
                break
            end
        end
        oix += 1
    end
    xc.orders[orderix, :lastcheck] = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
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
function getopenorders(base=nothing)::AbstractDataFrame
    if EnvConfig.configmode == production
        return Bybit.openorders(symbol=symbolusdt(base))
    else  # simulation
        if isnothing(xc)
            @error "cannot simulate getopenorders() with uninitialized CryptoXch cache"
            return DataFrame()
        end
        for oix in eachindex(xc.orders.orderid)
             _updateorder!(oix)
        end
        orders = subset(xc.orders, :status => st -> _openstatus(st), view=true)
        # return isnothing(base) ? orders[!, Not(:lastcheck)] : orders[symbolusdt(base) .== orders.symbol, Not(:lastcheck)]
        return isnothing(base) ? orders[!, :] : orders[symbolusdt(base) .== orders.symbol, :]
    end
end

"Checks in sumulation buy or sell conditions and returns order index or nothign if not found"
function _orderrefresh(orderid)
    if isnothing(xc)
        @error "cannot simulate getorder() with uninitialized CryptoXch cache"
        return nothing
    end
    orderindex = nothing
    for oix in eachindex(xc.orders.orderid)
        _updateorder!(oix)
        orderindex = (orderid == xc.orders[oix, :orderid]) ? oix : orderindex
    end
    return orderindex
end

"Returns a named tuple with elements equal to columns of getopenorders() dataframe of the identified order or `nothing` if order is not found"
function getorder(orderid)
    if EnvConfig.configmode == production
        return Bybit.order(orderid)
    else  # simulation
        orderindex = _orderrefresh(orderid)
        return isnothing(orderindex) ? nothing : NamedTuple(xc.orders[orderindex, :])
    end
end

"Returns orderid in case of a successful cancellation"
function cancelorder(base, orderid)
    if EnvConfig.configmode == production
        return Bybit.cancelorder(symbolusdt(base), orderid)
    else  # simulation
        if isnothing(xc)
            @error "cannot simulate cancelorder() with uninitialized CryptoXch cache"
            return nothing
        end
        oix = findfirst(x -> x == orderid, xc.orders.orderid)
        if !isnothing(oix)
            base, _ = basequote(xc.orders[oix, :symbol])
            ohlcv = xc.bases[base]
            ohlcvdf = Ohlcv.dataframe(ohlcv)
            xc.orders[oix, :updated] = ohlcvdf.opentime[Ohlcv.ix(ohlcv)]
            xc.orders[oix, :status] = "Cancelled"
            if xc.orders[oix, :side] == "Buy"
                qteqty = (xc.orders[oix, :baseqty] - xc.orders[oix, :executedqty]) * xc.orders[oix, :limitprice]
                _updateasset!(EnvConfig.cryptoquote, -qteqty, qteqty)
            else # sell side
                baseqty = xc.orders[oix, :baseqty] - xc.orders[oix, :executedqty]
                _updateasset!(EnvConfig.cryptoquote, -baseqty, baseqty)
            end
            return orderid
        else
            return nothing
        end
    end
end

function _createordersimulation(base, side, baseqty, limitprice, freeasset, freeassetqty)
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
    orderid = "$side$baseqty*$limitprice$base$(Dates.now())"
    # println("_assetfreelocked($freeasset)=$(_assetfreelocked(freeasset)) >= freeassetqty=$freeassetqty")
    if _assetfreelocked(freeasset).free >= freeassetqty
        dtnow = ohlcv.df.opentime[ohlcv.ix]
        if ((side == "Buy") && (limitprice >= ohlcv.df.close[ohlcv.ix])) || ((side == "Sell") && (limitprice <= ohlcv.df.close[ohlcv.ix]))
            push!(xc.orders, (orderid=orderid, symbol=symbolusdt(base), side=side, baseqty=baseqty, ordertype="Limit", timeinforce="PostOnly", limitprice=limitprice, executedqty=0.0f0, status="Rejected", created=dtnow, updated=dtnow, rejectreason="Rejected because PostOnly prevents taker orders", lastcheck=dtnow))
        else
            push!(xc.orders, (orderid=orderid, symbol=symbolusdt(base), side=side, baseqty=baseqty, ordertype="Limit", timeinforce="PostOnly", limitprice=limitprice, executedqty=0.0f0, status="New", created=dtnow, updated=dtnow, rejectreason="NO ERROR", lastcheck=dtnow))
            _updateasset!(freeasset, freeassetqty, -freeassetqty)
        end
    else
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
function createbuyorder(base::String; limitprice, basequantity)
    base = uppercase(base)
    if EnvConfig.configmode == production
        return Bybit.createorder(symbolusdt(base), "Buy", basequantity, limitprice)
    else  # simulation
        return _createordersimulation(base, "Buy", basequantity, limitprice, EnvConfig.cryptoquote, basequantity * limitprice)
    end
end

"""
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` < current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createsellorder(base::String; limitprice, basequantity)
    base = uppercase(base)
    if EnvConfig.configmode == production
        return Bybit.createorder(symbolusdt(base), "Sell", basequantity, limitprice)
    else  # simulation
        return _createordersimulation(base, "Sell", basequantity, limitprice, base, basequantity)
    end
end

function changeorder(orderid; limitprice=nothing, basequantity=nothing)
    if EnvConfig.configmode == production
        oo = Bybit.order(orderid)
        if isnothing(oo)
            return nothing
        end
        return Bybit.amendorder(oo.symbol, orderid; quantity=basequantity, limitprice=limitprice)
    else  # simulation
        oix = _orderrefresh(orderid)
        if isnothing(oix) || !_openstatus(xc.orders[oix, :status]) || (xc.orders[oix, :baseqty] <= xc.orders[oix, :executedqty])
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
        freeasset, freeassetqty = xc.orders[oix, :side] == "Buy" ? (EnvConfig.cryptoquote, baseqty * limitdelta + limit * baseqtydelta) : (basequote(xc.orders[oix, :symbol])[1], baseqtydelta)
        _updateasset!(freeasset, freeassetqty, -freeassetqty)
        xc.orders[oix, :baseqty] = baseqty
        xc.orders[oix, :limitprice] = limit
        if baseqty <= xc.orders[oix, :executedqty] # close order
            xc.orders[oix, :status] = "Filled"
        end
        return xc.orders[oix, :orderid]
    end
end

end  # of module
