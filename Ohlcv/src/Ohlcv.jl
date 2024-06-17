# using Pkg;
# Pkg.add(["Dates", "DataFrames", "CategoricalArrays", "JDF", "CSV"])

"""
Provides facilities to work with OHLCV data sequences including storage and retrieval.
Use 'TestOhlcv' to get canned train, eval, test data.

In future to add new OHLCV data from Binance.
"""
module Ohlcv

using Dates, DataFrames, CategoricalArrays, JDF, CSV, Logging, Statistics, Base
using EnvConfig

export write, read!, OhlcvData

"""
Ohlcv data starts in the first row with the oldest data and adds new data at the end
"""
mutable struct OhlcvData
    df::DataFrames.DataFrame
    base::String
    quotecoin::String  # instead of quotecoin because *quotecoin* is a Julia keyword
    interval::String
    ix::Integer  # can be used to sync the current index between modules for a backtest
    latestloadeddt
end

function Base.show(io::IO, ohlcv::OhlcvData)
    print(io::IO, "ohlcv: base=$(ohlcv.base) interval=$(ohlcv.interval) size=$(size(ohlcv.df)) intervals=$(size(ohlcv.df, 1) > 0 ? round(Int, (ohlcv.df[end, :opentime] - (ohlcv.df[begin, :opentime] - Dates.Minute(1)))/intervalperiod(ohlcv.interval)) : 0) start=$(size(ohlcv.df, 1) > 0 ? ohlcv.df[begin, :opentime] : "no start datetime)") end=$(size(ohlcv.df, 1) > 0 ? ohlcv.df[end, :opentime] : "no end datetime)") ix=$(ohlcv.ix)")
end

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info, e.g. number of steps in rowix
"""
verbosity = 1

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
function intervalperiod(interval)::Dates.Period
    @assert interval in keys(periods) "unknown $interval"
    return periods[interval]
end

"Returns an empty dataframe with all persistent columns"
function defaultohlcvdataframe(rows=0)::DataFrames.DataFrame
    # df = DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])
    df = DataFrame(opentime=zeros(DateTime, rows), open=zeros(Float32, rows), high=zeros(Float32, rows), low=zeros(Float32, rows), close=zeros(Float32, rows), basevolume=zeros(Float32, rows), pivot=zeros(Float32, rows))
    return df
end

" Returns an empty default crypto OhlcvData with quotecoin=usdt, interval=1m"
function defaultohlcv(base, interval="1m", rows=0)::OhlcvData
    ohlcv = OhlcvData(defaultohlcvdataframe(rows), uppercase(base), uppercase(EnvConfig.cryptoquote), interval, 1, nothing)
    return ohlcv
end

basecoin(ohlcv::OhlcvData) = ohlcv.base
quotecoin(ohlcv::OhlcvData) = ohlcv.quotecoin
interval(ohlcv::OhlcvData) = ohlcv.interval
dataframe(ohlcv::OhlcvData) = ohlcv.df
ix(ohlcv::OhlcvData) = ohlcv.ix
function setix!(ohlcv::OhlcvData, ix::Integer)
    ix = size(ohlcv.df, 1) == 0 ? 0 : ix
    ix = ix < firstindex(ohlcv.df[!,:opentime]) ? firstindex(ohlcv.df[!,:opentime]) : ix
    ix = ix > lastindex(ohlcv.df[!,:opentime]) ? lastindex(ohlcv.df[!,:opentime]) : ix
    ohlcv.ix = ix
    return ix
end

"Returns the current ohlcv dataframe row of ohlcv.ix or nothing if no data."
current(ohlcv::OhlcvData) = size(ohlcv.df, 1) > 0 ? ohlcv.df[ohlcv.ix, :] : nothing

# function copy(ohlcv::OhlcvData)
#     return OhlcvData(ohlcv.df, ohlcv.base, ohlcv.quotecoin, ohlcv.interval)
# end

function copy(ohlcv::OhlcvData)
    ohlcvcopy = OhlcvData(DataFrames.copy(ohlcv.df), ohlcv.base, ohlcv.quotecoin, ohlcv.interval, ohlcv.ix, ohlcv.latestloadeddt)
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
        (verbosity >= 3) && @info "addohlcv is full subset of ohlcv - no action"
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
    ohlcv.base = uppercase(base)
    return ohlcv
end

function setquotesymbol!(ohlcv::OhlcvData, quotecoin::String)
    ohlcv.quotecoin = uppercase(quotecoin)
    return ohlcv
end

function setinterval!(ohlcv::OhlcvData, interval::String)
    ohlcv.interval = interval
    return ohlcv
end

function setdataframe!(ohlcv::OhlcvData, df)
    ohlcv.df = df
    ohlcv.ix = lastindex(df, 1)
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
        baseprice = prices[baseix] * (1 - relativefee)
        gainprice = prices[gainix] * (1 + relativefee)
        gain = (baseprice - gainprice) / prices[baseix]
    else
        baseprice = prices[baseix] * (1 + relativefee)
        gainprice = prices[gainix] * (1 - relativefee)
        gain = (gainprice - baseprice) / prices[baseix]
    end
    # println("forward gain(prices[$baseix]= $(prices[baseix]) prices[$gainix]= $(prices[gainix]))=$gain")
    return gain
end

"""
Divides all elements by the value of the last as reference point - if not otherwise specified - and subtracts 1.
If ``percent==true`` then the result is multiplied by 100
"""
normalize(ydata; ynormref=ydata[end], percent=false) = percent ? (ydata ./ ynormref .- 1) .* 100 : (ydata ./ ynormref .- 1)

function pivot(df::AbstractDataFrame)
    cols = names(df)
    p::Vector{Float32} = []
    if all([c in cols for c in ["open", "high", "low", "close"]])
        p = (df[!, :open] + df[!, :high] + df[!, :low] + df[!, :close]) ./ 4
    end
    return p
end

haspivot(df::AbstractDataFrame) = "pivot" in names(df)
haspivot(ohlcv::OhlcvData) = haspivot(ohlcv.df)

function addpivot!(df::AbstractDataFrame)
    if haspivot(df)
        return df
    else
        p = pivot(df)
        if size(p, 1) == size(df,1)
            df[:, :pivot] = p
        else
            @error "$(@__MODULE__) unexpected difference of length(pivot)=$(size(p, 1)) != length(df)=$(size(df, 1))"
        end
        return df
    end
end

pivot!(df::AbstractDataFrame) = addpivot!(df)[!, :pivot]

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

"Returns the row index within the timestamps vector or the closest ix if not found"
function rowix(timestamps::AbstractVector{DateTime}, dt::DateTime, interval::Union{Period, Nothing}=nothing)
    #TODO regression tests to be added
    ot = timestamps
    p = isnothing(interval) ? (lastindex(ot) - firstindex(ot) > 1 ? ot[firstindex(ot) + 1] - ot[firstindex(ot)] : intervalperiod("1m")) : interval
    # steps = 1
    dt = floor(dt, p)
    if length(ot) == 0
        return 0
    end
    if dt < ot[begin]
        return firstindex(ot)
    elseif dt > ot[end]
        return lastindex(ot)
    else
        ix = min(max(firstindex(ot), round(Int, ((dt - ot[begin]) / p + 1))), lastindex(ot))
        # println("ix1=$ix firstindex(ot)=$(firstindex(ot)) calc=$(round(Int, ((dt - ot[begin]) / p + 1)))")
        # println("dt=$dt ot[ix]=$(ot[ix]) dt==ot[ix]=$(dt == ot[ix]) firstindex(ot)=$(firstindex(ot)) calc=$(ix + round(Int, ((dt - ot[ix]) / p + 1)))")
        # ix should be very close unless there are missing minutes - hence a second try
        ix = dt == ot[ix] ? ix : min(max(firstindex(ot), ix + round(Int, ((dt - ot[ix]) / p + 1))), lastindex(ot))
        # ix should be very close unless it landed on teh wrong side of missing minutes - hence a third try
        ix = dt == ot[ix] ? ix : min(max(firstindex(ot), ix + round(Int, ((dt - ot[ix]) / p + 1))), lastindex(ot))
        # println("ix2=$ix")
        if dt > ot[ix]
            while ((ix + 1) <= lastindex(ot)) && (dt >= ot[ix + 1])
                ix += 1
                # steps += 1
            end
        elseif dt < ot[ix]
            while ((ix - 1) >= firstindex(ot)) && (dt <= ot[ix - 1])
                ix -= 1
                # steps += 1
            end
        end
        # println("rowix steps=$steps")
        return ix
    end
end

"Returns the row index within the ohlcv DataFrame of the given DateTime or the closest ix if not found"
rowix(ohlcv::OhlcvData, dt::Dates.DateTime) = rowix(dataframe(ohlcv).opentime, dt, intervalperiod(interval(ohlcv)))

"accumulates minute interval ohlcv dataframe into larger interval dataframe"
function accumulate(df::AbstractDataFrame, interval::AbstractString)
    # accumulates to day/hour/5min borders
    # e.g. 2022-04-25T21:50:00 == floor(2022-04-25T21:52:38.109, Dates.Minute(5))
    # e.g. 2022-04-25T21:51:00 == floor(2022-04-25T21:52:38.109, Dates.Minute(3))
    # e.g. 2022-04-25T00:00:00 == floor(2022-04-25T21:52:38.109, Dates.Day(1))
    if lowercase(interval) == "1m"
        return df  # no accumulation required because this is the smallest supported period
    end
    period = Ohlcv.intervalperiod(interval)
    return accumulate(df, period)
end

function accumulate(df::AbstractDataFrame, period::Period)
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
    io = CSV.File(EnvConfig.datafile(uppercase(ohlcv.base) * "_OHLCV", "_df.csv"), types=Dict(1=>String, 2=>Float32, 3=>Float32, 4=>Float32, 5=>Float32, 6=>Float32, 7=>Float32))
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

function setsplit()::AbstractDataFrame
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
setname selects one of several disjunct subsets, e.g. : , : test, as defined in the sets split csv file.
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

mnemonic(ohlcv::OhlcvData) = uppercase(ohlcv.base) * "_" * uppercase(ohlcv.quotecoin) * "_" * ohlcv.interval * "_OHLCV"

function file(ohlcv::OhlcvData)
    mnm = mnemonic(ohlcv)
    filename = EnvConfig.datafile(mnm, "OHLCV")
    if isdir(filename)
        return (filename=filename, existing=true)
    else
        return (filename=filename, existing=false)
    end
end

function write(ohlcv::OhlcvData)
    if EnvConfig.configmode == production
        if !isnothing(ohlcv.latestloadeddt) && (size(ohlcv.df, 1) > 0) && (ohlcv.latestloadeddt >= ohlcv.df[end, :opentime])
            (verbosity >= 3) && println("$(EnvConfig.now()) Ohlcv not written due to missing supplementations of already stored data")
            return
        end
        fn = file(ohlcv)
        try
            JDF.savejdf(fn.filename, ohlcv.df[!, save_cols])  # without :pivot
            df = ohlcv.df
            (verbosity >= 2) && println("$(EnvConfig.now()) saved $(fn.filename) of $(ohlcv.base) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows at $(ohlcv.interval) interval")
        catch e
            Logging.@error "exception $e detected"
        end
    else
        (verbosity >= 2) && println("no Ohlcv.write() if EnvConfig.configmode != production to prevent mixing testnet data with real canned data")
    end
end

function read!(ohlcv::OhlcvData)::OhlcvData
    fn = file(ohlcv)
    df = DataFrame()
    try
        if fn.existing
            (verbosity >= 3) && println("$(EnvConfig.now()) loading OHLCV data of $(ohlcv.base) from  $(fn.filename)")
            df = DataFrame(JDF.loadjdf(fn.filename))
            (verbosity >= 2) && println("$(EnvConfig.now()) loaded OHLCV data of $(ohlcv.base) from $(size(df, 1) > 0 ? df[1, :opentime] : "N/A due to empty") until $(size(df, 1) > 0 ? df[end, :opentime] : "N/A due to empty") with $(size(df, 1)) rows at $(ohlcv.interval) interval from  $(fn.filename)")
            ohlcv.latestloadeddt = size(df, 1) > 0 ? df[end, :opentime] : nothing
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(fn.filename)")
        end
    catch e
        Logging.@error "exception $e detected"
    end
    addpivot!(df)
    setdataframe!(ohlcv, df)
    return ohlcv
end

function delete(ohlcv::OhlcvData)
    if EnvConfig.configmode == production
        fn = file(ohlcv)
        if fn.existing
            rm(fn.filename; force=true, recursive=true)
        end
    else
        (verbosity >= 1) && @warn "no Ohlcv.delete() if EnvConfig.configmode != production to prevent loosing accidentially real canned data"
    end
end

"""
Removes ohlcv data rows that are outside the date boundaries (nothing= no boundary) and adjusts ohlcv.ix to stay within the new data range.
"""
function timerangecut!(ohlcv::OhlcvData, startdt, enddt)
    if isnothing(ohlcv) || isnothing(ohlcv.df) || (size(ohlcv.df, 1) == 0)
        return
    end
    ixdt = ohlcv.df[ohlcv.ix, :opentime]
    startdt = isnothing(startdt) ? ohlcv.df[begin, :opentime] : startdt
    startix = Ohlcv.rowix(ohlcv.df[!, :opentime], startdt)
    enddt = isnothing(enddt) ? ohlcv.df[end, :opentime] : enddt
    endix = Ohlcv.rowix(ohlcv.df[!, :opentime], enddt)
    ohlcv.df = ohlcv.df[startix:endix, :]
    ohlcv.ix = Ohlcv.rowix(ohlcv, ixdt)
end

function curetime!(ohlcvdf, rn)
    rmrow = nothing
    addrow = nothing
    if floor(ohlcvdf[rn, :opentime], Minute(1)) != ohlcvdf[rn, :opentime]  # found time that is not a full minute
        ohlcvdf[rn, :opentime] = floor(ohlcvdf[rn, :opentime], Minute(1))
    end
    # now the current row and all before should be full minutes
    if (rn > 1) && (ohlcvdf[rn, :opentime] == ohlcvdf[rn-1, :opentime]) # remove one of the 2 rows with same timestamp
        rmrow = ohlcvdf[rn-1, :basevolume] == 0.0f0 ? rn-1 : rn
    elseif (ohlcvdf[rn, :opentime] - ohlcvdf[rn-1, :opentime]) > Minute(1)
        addrow = rn
    end
    return addrow, rmrow
end

function cure!(ohlcvdf, defectdf)
    @assert size(ohlcvdf, 2) >= 6  # 6 columns without, 7 with pivot
    println("curing ...")
    gdf = groupby(defectdf, :category)
    rmrows = Int[]
    addrows = Int[]
    for (key, subdf) in pairs(gdf)
        println("Number of data points for $(key.category): $(nrow(subdf))")
        if key.category == "out-of-sequence"
            for rn in subdf.rownumber
                rnn = rn
                while true
                    addrow, rmrow = curetime!(ohlcvdf, rnn)
                    !isnothing(rmrow) ? push!(rmrows, rmrow) : 0
                    !isnothing(addrow) ? push!(addrows, addrow) : 0
                    rnn = rnn < lastindex(ohlcvdf.opentime) ? ((ohlcvdf[rn, :opentime] - ohlcvdf[rn-1, :opentime]) != Minute(1) ? rnn + 1 : break) : break
                end
            end
        end
    end
    deleteat!(ohlcvdf, rmrows)
    # insert!(ohlcvdf, 5, ohlcvdf[1, :])  - gaps will not be fixed because no negative impact - furthermore add and delete have to be done from end to beginning to maintain row numbers

end

"""
check a ohlcv for the following aspects:
    - rows are all 1 minute apart without gaps and repeats
    - data has no no negative values (i.e prices or voume)
    - data has no nothing, NaN, Inf, missing elements
    - expectation is that a trade break results in zero volume and no price moves
"""
function check(ohlcv::OhlcvData; cure=false)
    odf = dataframe(ohlcv)
    setvec = String[]
    cnamevec = String[]
    ixvec = Int[]
    for cname in names(odf)
        # println("analyzing column $cname")
        cvec = odf[!, cname]
        if cname == "opentime"
            inv = [ix for ix in eachindex(cvec) if isnothing(cvec[ix]) || ismissing(cvec[ix])]
            ixvec = vcat(ixvec, inv)
            setname = "invalid"
            setvec = vcat(setvec, [setname for _ in inv])
            cnamevec = vcat(cnamevec, [cname for _ in inv])

            nosequence = [ix for ix in eachindex(cvec) if (ix > firstindex(cvec)) && ((cvec[ix] - cvec[ix-1]) != Dates.Minute(1))]
            xnosequence = [ix-1 for ix in nosequence if !((ix-1) in nosequence)]
            nosequence = vcat(nosequence, xnosequence)
            ixvec = vcat(ixvec, nosequence)
            setname = "out-of-sequence"
            setvec = vcat(setvec, [setname for _ in nosequence])
            cnamevec = vcat(cnamevec, [cname for _ in nosequence])
        else
            inv = [ix for ix in eachindex(cvec) if isnothing(cvec[ix]) || ismissing(cvec[ix]) || isnan(cvec[ix])]
            ixvec = vcat(ixvec, inv)
            setname = "invalid"
            setvec = vcat(setvec, [setname for _ in inv])
            cnamevec = vcat(cnamevec, [cname for _ in inv])

            inv = [ix for ix in eachindex(cvec) if isinf(cvec[ix]) || (cvec[ix] < 0)]
            ixvec = vcat(ixvec, inv)
            setname = "out-of-range"
            setvec = vcat(setvec, [setname for _ in inv])
            cnamevec = vcat(cnamevec, [cname for _ in inv])
        end
    end
    if length(ixvec) > 0
        # sc = categorical(setvec; compress=true)
        rdf = DataFrame([setvec, cnamevec, ixvec], [:category, :colname, :rownumber])
        sort!(rdf, :rownumber)
        rodf = hcat(rdf, odf[rdf[!, :rownumber], :])
        @warn "Ohlcv DataFrame of $ohlcv is checked NOT OK" rodf
        if cure
            cure!(odf, rdf)
            write(ohlcv)
        end
    else
        println("Ohlcv DataFrame of $ohlcv is checked OK")
    end
end

function check(base::String; cure=false)
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.read!(ohlcv)
    check(ohlcv; cure)
    check(ohlcv; cure=false)
end

mutable struct OhlcvFiles
    filenames
    OhlcvFiles() = new(nothing)
end

# function Base.iterate(of::OhlcvFiles, state=1)
# end

function Base.iterate(of::OhlcvFiles, state=1)
    if isnothing(of.filenames)
        allff = readdir(EnvConfig.datafolderpath("OHLCV"), join=false, sort=false)
        fileixlist = findall(f -> endswith(f, "OHLCV.jdf"), allff)
        of.filenames = [allff[ix] for ix in fileixlist]
        if length(of.filenames) > 0
            state = firstindex(of.filenames)
        else
            return nothing
        end
    end
    if state > lastindex(of.filenames)
        return nothing
    end
    # fn = split(of.filenames[state], "/")[end]
    fnparts = split(of.filenames[state], "_")
    # return (basecoin=fnparts[1], quotecoin=fnparts[2], interval=fnparts[3]), state+1
    basecoin=fnparts[1]
    # quotecoin=fnparts[2]
    interval=fnparts[3]
    ohlcv = defaultohlcv(basecoin, interval)
    read!(ohlcv)
    return ohlcv, state+1
end

function _filenamechange(path)
    psegs = split(path, "/")
    fname = psegs[end]
    segs = split(fname, "_")
    segs[1] = uppercase(segs[1])
    segs[2] = uppercase(segs[2])
    fname = join(segs, "_")
    psegs[end] = fname
    path = join(psegs, "/")
    return path
end

function ohlcvrename()
    for m in [test]  #[test, production, training]
        EnvConfig.init(m)
        cd(EnvConfig.datafolderpath("OHLCV")) do
            allff = readdir(join=false, sort=false)
            println("$m dir=$(pwd())")
            fileixlist = findall(f -> endswith(f, "OHLCV.jdf"), allff)
            for ix in fileixlist
                fn = _filenamechange(allff[ix])
                println("$m $(allff[ix]) $fn")
                mv(allff[ix], fn * "1")
                mv(fn * "1", fn)
            end
            allff = readdir(EnvConfig.datafolderpath("OHLCV"), join=false, sort=false)
            fileixlist = findall(f -> endswith(f, "OHLCV.jdf"), allff)
            for ix in fileixlist
                println("$m $(allff[ix])")
            end
        end
        println("back to dir=$(pwd())")
    end
end

function ohlcvmove()
    for m in [test, production]  #[test, production, training]
        EnvConfig.init(m)
        cd(EnvConfig.datafolderpath()) do
            allff = readdir(join=false, sort=false)
            println("$m dir=$(pwd())")
            fileixlist = findall(f -> endswith(f, "OHLCV.jdf"), allff)
            for ix in fileixlist
                fn = "OHLCV/" * allff[ix]
                println("$m $(allff[ix]) $fn")
                mv(allff[ix], fn)
            end
            allff = readdir(EnvConfig.datafolderpath("OHLCV"), join=false, sort=false)
            fileixlist = findall(f -> endswith(f, "OHLCV.jdf"), allff)
            for ix in fileixlist
                println("$m $(allff[ix])")
            end
        end
        println("back to dir=$(pwd())")
    end
end

function testiterate()
    EnvConfig.init(production)
    # for (ix, fnp) in enumerate(OhlcvFiles)
    #     println("$ix: base=$(fnp.base) quote=$(fnp.quotecoin) interval=$(fnp.interval)")
    # for fnp in OhlcvFiles()
    #     println("base=$(fnp.basecoin) quote=$(fnp.quotecoin) interval=$(fnp.interval)")
    # end
    s=nothing
    for ohlcv in OhlcvFiles()
        check(ohlcv)
        println(ohlcv)
        # if s != size(ohlcv.df, 1)
        #     println("old size:$s new size:$(size(ohlcv.df, 1))")
        # end
        s = size(ohlcv.df, 1)
    end
end

function testrowix()
    EnvConfig.init(production)
    ohlcv = defaultohlcv("ZRX")
    Ohlcv.read!(ohlcv)
    # use indices before and after a gap
    println("ZRX opentime[2134911]=$(ohlcv.df[2134911, :opentime]) rowix=$(rowix(ohlcv, ohlcv.df[2134911, :opentime]))")
    println("ZRX opentime[2134912]=$(ohlcv.df[2134912, :opentime]) rowix=$(rowix(ohlcv, ohlcv.df[2134912, :opentime]))")

end

end  # Ohlcv

# Ohlcv.ohlcvrename()
# Ohlcv.testiterate()
# Ohlcv.testrowix()
# Ohlcv.ohlcvmove()
