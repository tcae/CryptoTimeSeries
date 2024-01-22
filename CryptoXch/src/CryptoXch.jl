# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])


module CryptoXch

using Dates, DataFrames, DataAPI, JDF, CSV, Logging
using Bybit, EnvConfig, Ohlcv, TestOhlcv
import Ohlcv: intervalperiod


basenottradable = ["boba",
    "btt", "bcc", "ven", "pax", "bchabc", "bchsv", "usds", "nano", "usdsb", "erd", "npxs", "storm", "hc", "mco",
    "bull", "bear", "ethbull", "ethbear", "eosbull", "eosbear", "xrpbull", "xrpbear", "strat", "bnbbull", "bnbbear",
    "xzc", "gxs", "lend", "bkrw", "dai", "xtzup", "xtzdown", "bzrx", "eosup", "eosdown", "ltcup", "ltcdown"]
basestablecoin = ["usdt", "tusd", "busd", "usdc", "eur"]
baseignore = [""]
baseignore = uppercase.(append!(baseignore, basestablecoin, basenottradable))
minimumquotevolume = 10  # USDT


defaultcryptoexchange = "bybit"  # "binance"

symbolusdt(base) = isnothing(base) ? nothing : uppercase(base * EnvConfig.cryptoquote)


"""
Requests base/USDT from start until end (both including) in interval frequency but will return a maximum of 1000 entries.
Subsequent calls are required to get > 1000 entries.
Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""
function ohlcfromexchange(base::String, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    df = nothing
    symbol = uppercase(base*cryptoquote)
    df = Bybit.getklines(symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
    Ohlcv.addpivot!(df)
    return df
end

"Returns nothing but displays the latest klines/ohlcv data"
function getlastminutesdata()
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Minute(7)
    res = ohlcfromexchange("BTCUSDT", startdt, enddt)
    # display(nrow(res))
    display(last(res, 3))
    # display(first(res, 3))
    enddt = Dates.now(Dates.UTC)
    res2 = ohlcfromexchange("BTCUSDT", enddt - Dates.Second(1), enddt)
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
function gethistoryohlcv(base::String, startdt::DateTime, enddt::DateTime=Dates.now(Dates.UTC), interval="1m")
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
        res = ohlcfromexchange(base, startdt, enddt, interval)
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
            newdf = gethistoryohlcv(base, startdt, tmpdt, interval)
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
            newdf = gethistoryohlcv(base, tmpdt, enddt, interval)
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
        newdf = gethistoryohlcv(base, startdt, enddt, interval)
        Ohlcv.setdataframe!(ohlcv, newdf)
    end
    return ohlcv
end

timerangecut!(ohlcv, startdt, enddt) = subset!(ohlcv.df, :opentime => t -> floor(startdt, intervalperiod(ohlcv.interval)) .<= t .<= floor(enddt, intervalperiod(ohlcv.interval)))

"""
Returns the OHLCV data of the requested time range by first checking the stored cache data and if unsuccessful requesting it from the Exchange.

    - *base* identifier and interval specify what data is requested - the result will be returned as OhlcvData structure
    - startdt and enddt are DateTime stamps that specify the requested time range
    - any gap to chached data will be closed when asking for missing data from Bybit
    - beside returning the ohlcv data to the caller, it is also written to stored cache to reduce future slow data requests to Bybit
"""
function cryptodownload(base, interval, startdt, enddt)::OhlcvData
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.setinterval!(ohlcv, interval)
    Ohlcv.read!(ohlcv)
    cryptoupdate!(ohlcv, startdt, enddt)
    if size(Ohlcv.dataframe(ohlcv), 1) > 0
        Ohlcv.write(ohlcv)
    else
        @warn "cryptodownload: No $base OHLCV data written due to empty dataframe"
    end
    return ohlcv
end

function downloadupdate!(bases, enddt, period=Dates.Year(4))
    count = length(bases)
    for (ix, base) in enumerate(bases)
        # break
        println()
        println("$(EnvConfig.now()) updating $base ($ix of $count)")
        startdt = enddt - period
        ohlcv = CryptoXch.cryptodownload(base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
    end
end

ceilbase(base, qty) = base == "usdt" ? ceil(qty, digits=3) : ceil(qty, digits=5)
floorbase(base, qty) = base == "usdt" ? floor(qty, digits=3) : floor(qty, digits=5)
roundbase(base, qty) = base == "usdt" ? round(qty, digits=3) : round(qty, digits=5)
# TODO read base specific digits from binance and use them base specific

onlyconfiguredsymbols(symbol) =
    endswith(symbol, uppercase(EnvConfig.cryptoquote)) &&
    !(uppercase(symbol[1:end-length(EnvConfig.cryptoquote)]) in baseignore)

function basequote(symbol)
    symbol = lowercase(symbol)
    range = findfirst(lowercase(EnvConfig.cryptoquote), symbol)
    return isnothing(range) ? nothing : (basecoin = symbol[1:end-length(EnvConfig.cryptoquote)], quotecoin = EnvConfig.cryptoquote)
end

"""
Returns a dataframe with 24h values of all USDT quote bases with the following columns:

- basecoin
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

"""
function getUSDTmarket()
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
end

"Returns a DataFrame[:accounttype, :coin, :locked, :free] of wallet/portfolio balances"
balances() = Bybit.balances()

"""
Appends a balances DataFrame with the USDT value of the base asset using usdtdf[:lastprice] and returns it as DataFrame[:base, :locked, :free, :usdt].
"""
function portfolio!(balancesdf, usdtdf)
    balancesdf[:, :sym] = [symbolusdt(b) for b in balancesdf[!, :base]]
    balancesdf = leftjoin(balancesdf, usdtdf[!, [:symbol, :lastprice]], on = :sym => :symbol)
    balancesdf.lastprice = coalesce.(balancesdf.lastprice, 1.0f0)
    balancesdf[:, :usdt] = (balancesdf.locked + balancesdf.free) * balancesdf.lastprice
    balancesdf = balancesdf[!, Not([:lastprice, :sym])]
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

getopenorders(base=nothing) = Bybit.openorders(symbol=symbolusdt(base))

getorder(orderid) = Bybit.order(orderid)

"Returns orderid in case of a successful cancellation"
cancelorder(base, orderid) = Bybit.cancelorder(symbolusdt(base), orderid)

"""
Adapts `limitprice` and `usdtquantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` > current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
createbuyorder(base::String; limitprice, usdtquantity) = Bybit.createorder(symbolusdt(base), "Buy", usdtquantity / limitprice, limitprice)

"""
Adapts `limitprice` and `usdtquantity` according to symbol rules and executes order.
Order is rejected (but order created) if `limitprice` < current price in order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
createsellorder(base::String; limitprice, usdtquantity) = Bybit.createorder(symbolusdt(base), "Sell", usdtquantity / limitprice, limitprice)

function changeorder(base::String, orderid; limitprice=nothing, usdtquantity=nothing)
    oo = Bybit.order(orderid)
    if isnothing(oo)
        return nothing
    end

    Bybit.amendorder(symbolusdt(base), orderid, quantity=(isnothing(usdtquantity) ? nothing : usdtquantity / (isnothing(limitprice) ? oo.limitprice : limitprice)), limitprice=limitprice)
end

end  # of module
