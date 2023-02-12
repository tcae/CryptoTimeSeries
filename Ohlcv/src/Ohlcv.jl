# using Pkg;
# Pkg.add(["Dates", "DataFrames", "CategoricalArrays", "JDF", "CSV"])

"""
Use 'Config' to get canned train, eval, test data.

In future to add new OHLCV data from Binance.
"""
module Ohlcv

using Dates, DataFrames, CategoricalArrays, JDF, CSV, Logging, Statistics
using EnvConfig

export write, read!, OhlcvData

mutable struct OhlcvData
    df::DataFrames.DataFrame
    base::String
    qte::String  # instead of quote because *quote* is a Julia keyword
    xch::String  # exchange - also implies whether the asset type is crypto or stocks
    interval::String
end

function Base.show(io::IO, ohlcv::OhlcvData)
    print(io::IO, "ohlcv: base=$(ohlcv.base) interval=$(ohlcv.interval) size=$(size(ohlcv.df)) pivot: max=$(maximum(ohlcv.df[!, :pivot])) median=$(Statistics.median(ohlcv.df[!, :pivot])) min=$(minimum(ohlcv.df[!, :pivot]))")
end


save_cols = [:opentime, :open, :high, :low, :close, :basevolume]
testbases = ["sinus", "triplesinus"]
periods = Dict(
    "1m" => Dates.Minute(1),
    "3m" => Dates.Minute(3),
    "5m" => Dates.Minute(5),
    "15m" => Dates.Minute(15),
    "30m" => Dates.Minute(30),
    "1h" => Dates.Hour(1),
    "2h" => Dates.Hour(2),
    "4h" => Dates.Hour(4),
    "6h" => Dates.Hour(6),
    "8h" => Dates.Hour(8),
    "12h" => Dates.Hour(12),
    "1d" => Dates.Day(1),
    "3d" => Dates.Day(3),
    "1w" => Dates.Week(1),
    "1M" => Dates.Week(4)  # better to be able to calculate with this period
)

"Returns a Dates.Period that corresponds to a period string"
function intervalperiod(interval)
    @assert interval in keys(periods) "unknown $interval"
    return periods[interval]
end

"Returns an empty dataframe with all persistent columns"
function defaultohlcvdataframe(rows=0)::DataFrames.DataFrame
    # df = DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])
    df = DataFrame(opentime=zeros(DateTime, rows), open=zeros(Float32, rows), high=zeros(Float32, rows), low=zeros(Float32, rows), close=zeros(Float32, rows), basevolume=zeros(Float32, rows), pivot=zeros(Float32, rows))
    return df
end

" Returns an empty default crypto OhlcvData with quote=usdt, xch=binance, interval=1m"
function defaultohlcv(base, interval="1m", rows=0)::OhlcvData
    ohlcv = OhlcvData(defaultohlcvdataframe(rows), base, EnvConfig.cryptoquote, EnvConfig.cryptoexchange, interval)
    return ohlcv
end

basesymbol(ohlcv::OhlcvData) = ohlcv.base
quotesymbol(ohlcv::OhlcvData) = ohlcv.qte
exchange(ohlcv::OhlcvData) = ohlcv.xch
interval(ohlcv::OhlcvData) = ohlcv.interval
dataframe(ohlcv::OhlcvData) = ohlcv.df

# function copy(ohlcv::OhlcvData)
#     return OhlcvData(ohlcv.df, ohlcv.base, ohlcv.qte, ohlcv.xch, ohlcv.interval)
# end

function copy(ohlcv::OhlcvData)
    ohlcvcopy = OhlcvData(DataFrames.copy(ohlcv.df), ohlcv.base, ohlcv.qte, ohlcv.xch, ohlcv.interval)
    return ohlcvcopy
end

"""
if ohlcv overlaps with addohlcv thenadd ohlcv replaces content of ohlcv.
if there is a time gap between the two then ohlcv will not be modofied and an error is issued.
adds the pivot column to addohlcv if one is present in ohlcv. currently limited to *1m* interval

"""
function merge!(ohlcv::OhlcvData, addohlcv::OhlcvData)
    df1 = dataframe(ohlcv)
    df2 = dataframe(addohlcv)
    if haspivot(df1)
        addpivot!(df2)
    end
    @assert ohlcv.interval == addohlcv.interval == "1m"
    startgap = Int(Dates.Minute(df1[begin, :opentime] - df2[end, :opentime])/Dates.Minute(1))
    endgap = Int(Dates.Minute(df2[begin, :opentime] - df1[end, :opentime])/Dates.Minute(1))
    endoverlap = -startgap +1
    if (startgap > 1) || (endgap > 1)
        @error "cannot merge ohlcv data due to time gap" startgap endgap
        return
    end
    if (df2[begin, :opentime] <= df1[begin, :opentime]) && (df1[end, :opentime] <= df2[end, :opentime])
        # ohlcv is complete subset of addohlcv
        ohlcv.df = DataFrames.copy(addohlcv.df)
    elseif (df1[begin, :opentime] < df2[begin, :opentime]) && (df2[end, :opentime] < df1[end, :opentime])
        @warn "addohlcv is full subset of ohlcv - no action"
    elseif df2[begin, :opentime] <= df1[begin, :opentime] <= (df2[end, :opentime] + Dates.Minute(1)) <= df1[end, :opentime]
        startoverlap = Int(Dates.Minute(df2[end, :opentime] - df1[begin, :opentime])/Dates.Minute(1)) + 1
        ohlcv.df = vcat(df2[!, :], df1[begin+startoverlap:end, :])
    elseif df1[begin, :opentime] <= df2[begin, :opentime] <= (df1[end, :opentime] + Dates.Minute(1)) <= df2[end, :opentime]
        endoverlap = Int(Dates.Minute(df1[end, :opentime] - df2[begin, :opentime])/Dates.Minute(1)) + 1
        ohlcv.df = vcat(df1[begin:end-endoverlap, :], df2[!, :])
    end
    return ohlcv
end

function setbasesymbol!(ohlcv::OhlcvData, base::String)
    ohlcv.base = base
    return ohlcv
end

function setquotesymbol!(ohlcv::OhlcvData, qte::String)
    ohlcv.qte = qte
    return ohlcv
end

function setexchange!(ohlcv::OhlcvData, exchange::String)
    ohlcv.xch = exchange
    return ohlcv
end

function setinterval!(ohlcv::OhlcvData, interval::String)
    ohlcv.interval = interval
    return ohlcv
end

function setdataframe!(ohlcv::OhlcvData, df)
    ohlcv.df = df
    return ohlcv
end

function detectgapissues(ohlcv::OhlcvData)
    df = ohlcv.df
    ix = 2
    while ix <= size(df, 1)
        gap = round(df[ix, :opentime] - df[ix-1, :opentime], Dates.Minute)
        # gap = df[ix, :opentime] - df[ix-1, :opentime]
        if gap < intervalperiod(ohlcv.interval)
            ok = false
            iix = ix
            # gap = round(gap, Dates.Minute)
            while all([df[iix, c] == df[ix, c] for c in names(df) if c != "opentime"])
                iix += 1
            end
            println("ohlvc: interval issue detected $(ohlcv.base) gap between $(df[ix-1, :opentime]) and $(df[iix, :opentime]) of $(gap)")
            println(df[ix-2:iix+1, :])
        end
        ix += 1
    end
end

function fillgaps!(ohlcv::OhlcvData)
    if (ohlcv === nothing) || (ohlcv.df === nothing) || (size(ohlcv.df, 1) <= 1)
        return
    end
    println("$(EnvConfig.now()) starting fillgaps")
    ok = true
    df = ohlcv.df
    newdf = defaultohlcvdataframe()
    startix = 1
    for ix in 2:size(df, 1)
        # if ix % 200000 == 0
        #     println("working on row $ix")
        # end
        # gap = round(df[ix, :opentime] - df[ix-1, :opentime], Dates.Minute)
        gap = df[ix, :opentime] - df[ix-1, :opentime]
        if gap > intervalperiod(ohlcv.interval)  # then repair
            ok = false
            gap = round(gap, Dates.Minute)
            # println("ohlvc: detected $(ohlcv.base) gap between $(df[ix-1, :opentime]) and $(df[ix, :opentime]) of $(gap)")
            # println(df[ix-3:ix+3, :])
            startdt = floor(df[ix-1, :opentime], Dates.Minute)
            dffill = DataFrame()
            dffill[!, :opentime] = [startdt + intervalperiod(ohlcv.interval) * ix for ix in 1:(gap.value - 1)]
            # println("1 ix: $ix  startix: $startix  dffill size: $(size(dffill))  newdf size: $(size(newdf))")
            dffill[:, :basevolume] .= 0.0
            for cix in names(df)
                if (cix != "opentime") && (cix != "basevolume")
                    dffill[:, cix] .= df[ix-1, cix]
                end
            end
            append!(newdf, df[startix:ix-1, :])
            # println("2 ix: $ix  startix: $startix  dffill size: $(size(dffill))  newdf size: $(size(newdf))")
            append!(newdf, dffill)
            # println("3 ix: $ix  startix: $startix  dffill size: $(size(dffill))  newdf size: $(size(newdf))")
            startix = ix
            # println(dffill[[1,2,end-1,end], :])
        end
    end

    if !ok  # aftercare then check 2nd time after repair
        append!(newdf, df[startix:end, :])
        setdataframe!(ohlcv, newdf)
        df = dataframe(ohlcv)
        println("$(EnvConfig.now())) fillgaps repaired - checking again")
        ok = true
        for ix in 2:size(df, 1)
            gap = round(df[ix, :opentime] - df[ix-1, :opentime], Dates.Minute)
            # gap = df[ix, :opentime] - df[ix-1, :opentime]
            if gap > intervalperiod(ohlcv.interval)
                ok = false
                # gap = round(gap, Dates.Minute)
                println("ohlvc: detected $(ohlcv.base) gap between $(df[ix-1, :opentime]) and $(df[ix, :opentime]) of $(gap)")
            end

        end
    end
    println("$(EnvConfig.now()) fillgaps: $ok")
    detectgapissues(ohlcv)
    println("$(EnvConfig.now()) fillgaps ready")
    return ohlcv
end

"""
- returns the relative forward/backward looking gain
- deducts relativefee from both prices
"""
function relativegain(prices, baseix, gainix; relativefee=0.0)
    if baseix > gainix
        gain = (prices[baseix] - prices[gainix] - (prices[gainix] + prices[baseix]) * relativefee) / prices[baseix]
    else
        gain = (prices[gainix] - prices[baseix] - (prices[gainix] + prices[baseix]) * relativefee) / prices[baseix]
    end
    # println("forward gain(prices[$baseix]= $(prices[baseix]) prices[$gainix]= $(prices[gainix]))=$gain")
    return gain
end

"""
Divides all elements by the value of the last as reference point - if not otherwise specified - and subtracts 1.
If ``percent==true`` then the result is multiplied by 100
"""
normalize(ydata; ynormref=ydata[end], percent=false) = percent ? (ydata ./ ynormref .- 1) .* 100 : (ydata ./ ynormref .- 1)

function pivot(df::DataFrame)
    cols = names(df)
    p::Vector{Float32} = []
    if all([c in cols for c in ["open", "high", "low", "close"]])
        p = (df[!, :open] + df[!, :high] + df[!, :low] + df[!, :close]) ./ 4
    end
    return p
end

haspivot(df::DataFrame) = "pivot" in names(df)
haspivot(ohlcv::OhlcvData) = haspivot(ohlcv.df)

function addpivot!(df::DataFrame)
    if haspivot(df)
        return df
    else
        p = pivot(df)
        if size(p, 1) == size(df,1)
            df[:, :pivot] = p
        else
            @warn "$(@__MODULE__) unexpected difference of length(pivot)=$(size(p, 1)) != length(df)=$(size(df, 1))"
        end
        return df
    end
end

pivot!(df::DataFrame) = addpivot!(df)[!, :pivot]

function addpivot!(ohlcv::OhlcvData)
    addpivot!(ohlcv.df)
    return ohlcv
end

pivot!(ohlcv::OhlcvData) = pivot!(ohlcv.df)

function pivot_test()
    ohlcv = defaultohlcv("test")
    df = dataframe(ohlcv)
    push!(df, [Dates.now(), 1.0, 2.0, 3.0, 4.0, 5.0])
    push!(df, [Dates.now(), 1.0, 2.0, 3.0, 4.0, 5.0])
    push!(df, [Dates.now(), 1.0, 2.0, 3.0, 4.0, 5.0])
    p1 = pivot(df)
    println("$(typeof(p1)) $p1")
    println("$(typeof(ohlcv)) $ohlcv")
    p2 = pivot!(ohlcv)
    println("$(typeof(p2)) $p2")
    println("$(typeof(ohlcv)) $ohlcv")

end

"accumulates minute interval ohlcv dataframe into larger interval dataframe"
function accumulate(df::DataFrame, interval)
    # accumulates to day/hour/5min borders
    # e.g. 2022-04-25T21:50:00 == floor(2022-04-25T21:52:38.109, Dates.Minute(5))
    # e.g. 2022-04-25T21:51:00 == floor(2022-04-25T21:52:38.109, Dates.Minute(3))
    # e.g. 2022-04-25T00:00:00 == floor(2022-04-25T21:52:38.109, Dates.Day(1))
    if lowercase(interval) == "1m"
        return df  # no accumulation required because this is the smallest supported period
    end
    period = Ohlcv.intervalperiod(interval)
    periodm = Dates.Minute(period).value
    rows1m = size(df, 1)
    # adf = defaultohlcvdataframe()
    if rows1m == 0; return df; end
    firstdt = df[1, :opentime]
    startadd = Dates.Minute(ceil(firstdt, Dates.Minute(period)) - firstdt).value # minutes before period rounded start
    endadd = rows1m % periodm - startadd # minutes after period rounded end
    rowsperiod = rows1m ÷ periodm + (startadd > 0 ? 1 : 0) + (endadd > 0 ? 1 : 0)
    adfix = Array{Int64}(undef, rowsperiod + 1)
    adf = defaultohlcvdataframe(rowsperiod)
    aix = 0
    adfix[1] = offset = startadd > 0 ? 1 : 0
    for aix in (1 + offset):rowsperiod; adfix[aix] = (aix - 1 - offset) * periodm + startadd + 1; end
    adfix[rowsperiod + 1] = rows1m + 1
    Threads.@threads for aix in 1:rowsperiod
        ix = adfix[aix]
        nextblockix = adfix[aix+1]
        open = df[ix, :open]
        high = df[ix, :high]
        low = df[ix, :low]
        basevolume = df[ix, :basevolume]
        adf[aix, :opentime] = floor(df[ix, :opentime], period)
        ix += 1
        while ix < nextblockix
            high = max(high, df[ix, :high])
            low = min(low, df[ix, :low])
            basevolume += df[ix, :basevolume]
            ix += 1
        end
        ix -= 1
        adf[aix, :open] = open
        adf[aix, :high] = high
        adf[aix, :low] = low
        adf[aix, :close] = df[ix, :close]
        adf[aix, :basevolume] = basevolume
    end
    adf[:, :pivot] = pivot(adf)
    return adf
end


"""
Reads OHLCV data generated by python FollowUpward
"""
function readcsv!(ohlcv::OhlcvData)::OhlcvData
    io = CSV.File(EnvConfig.datafile(ohlcv.base * "_OHLCV", "_df.csv"), types=Dict(1=>String, 2=>Float32, 3=>Float32, 4=>Float32, 5=>Float32, 6=>Float32, 7=>Float32))
    df = DataFrame(io)
    df[!, :opentime] = DateTime.(DateTime.(df[!, :Column1], "y-m-d H:M:s+z"), UTC)
    df = df[!, Not(:Column1)]
    df = df[!, Cols(:opentime, :)]
    df = df[!, save_cols]
    # df = df[!, Not(:opentime)]
    addpivot!(df)
    setdataframe!(ohlcv::OhlcvData, df)
    return ohlcv

end

function setsplit()::DataFrame
    io = CSV.File(EnvConfig.setsplitfilename())
    iodf = DataFrame(io)
    len = size(iodf, 1)

    df = iodf[!, [:set_type]]
    df[!, :start] = Vector{DateTime}(undef, len)
    df[!, :end] = Vector{DateTime}(undef, len)
    [df[ix, :start] = DateTime.(DateTime.(String(iodf[ix, :start]), "y-m-d H:M:s+z"), UTC) for ix in 1:len]
    [df[ix, :end] = DateTime.(DateTime.(String(iodf[ix, :end]), "y-m-d H:M:s+z"), UTC) for ix in 1:len]
    # println(df)
    return df

end

function setassign!(ohlcv::OhlcvData)
    setname = ["NA" for ix in eachrow(ohlcv.df)]

    splitdf = setsplit()
    # display(splitdf)
    sort!(splitdf, [:start, :end])
    sizesplitdf = size(splitdf, 1)
    setix = 1
    opentimes = ohlcv.df[!, :opentime]
    for ix in eachindex(opentimes)
        while (setix <= sizesplitdf) && (splitdf[setix, :end] < opentimes[ix])
            setix += 1
        end
        if (setix <= sizesplitdf) && (splitdf[setix, :start] <= opentimes[ix])
            setname[ix] = splitdf[setix, :set_type]
        end
    end

    setcategory = CategoricalArrays.categorical(setname, compress=true)
    ohlcv.df[!, :set] = setcategory
end

"""
Selects the given columns and returns them as transposed array, i.e. values of one column belong to one sample and rows represent samples.
setname selects one of several disjunct subsets, e.g. :training, : test, as defined in the sets split csv file.
"""
function columnarray(ohlcv::OhlcvData, setname::String, cols::Array{Symbol,1})::Array{Float32,2}
    setassign!(ohlcv)
    gd = groupby(ohlcv.df, [:set])
    subdf = gd[(set=setname,)]
    stackarr = [subdf[:,sym] for sym in cols]
    n = size(stackarr, 1)
    colarray = zeros(eltype(stackarr[1]),(n, size(stackarr[1],1)))
    for i in 1:n
        colarray[i, :] .= stackarr[i]
    end
    return colarray
end

mnemonic(ohlcv::OhlcvData) = ohlcv.base * "_" * ohlcv.qte * "_" * ohlcv.xch * "_" * ohlcv.interval * "_OHLCV"

function write(ohlcv::OhlcvData)
    mnm = mnemonic(ohlcv)
    filename = EnvConfig.datafile(mnm)
    # println(filename)
    JDF.savejdf(filename, ohlcv.df[!, save_cols])  # without :pivot
    df = ohlcv.df
    println("saved $filename of $(ohlcv.base) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows at $(ohlcv.interval) interval")
end

function read!(ohlcv::OhlcvData)::OhlcvData
    mnm = mnemonic(ohlcv)
    filename = EnvConfig.datafile(mnm)
    df = DataFrame()
    # println(filename)
    if isdir(filename)
        try
            df = DataFrame(JDF.loadjdf(filename))
            println("loaded OHLCV data of $(ohlcv.base) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows at $(ohlcv.interval) interval")
        catch e
            Logging.@warn "exception $e detected"
        end
    end
    # display(first(df, 1))
    addpivot!(df)
    setdataframe!(ohlcv, df)
    return ohlcv
end

function delete(ohlcv::OhlcvData)
    mnm = mnemonic(ohlcv)
    filename = EnvConfig.datafile(mnm)
    # println(filename)
    if isdir(filename)
        rm(filename; force=true, recursive=true)
    end
end

# pivot_test()

end  # Ohlcv

