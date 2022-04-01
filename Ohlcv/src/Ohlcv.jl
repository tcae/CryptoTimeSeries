# using Pkg;
# Pkg.add(["Dates", "DataFrames", "CategoricalArrays", "JDF", "CSV", "TimeZones"])

"""
Use 'Config' to get canned train, eval, test data.

In future to add new OHLCV data from Binance.
"""
module Ohlcv

using Dates, DataFrames, CategoricalArrays, JDF, CSV, TimeZones, Logging
using EnvConfig

export write, read!, OhlcvData

mutable struct OhlcvData
    df::DataFrames.DataFrame
    base::String
    qte::String  # instead of quote because *quote* is a Julia keyword
    xch::String  # exchange - also implies whether the asset type is crypto or stocks
    interval::String
end

save_cols = [:opentime, :open, :high, :low, :close, :basevolume]
testbases = ["sinus", "triplesinus"]

function intervalperiod(interval)
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
    if interval in keys(periods)
        return periods[interval]
    else
        @warn "unknown $interval"
        return Dates.Minute(1)
    end
end

"Returns an empty dataframe with all persistent columns"
function defaultohlcvdataframe()::DataFrames.DataFrame
    df = DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])
    return df
end

" Returns an empty default crypto OhlcvData with quote=usdt, xch=binance, interval=1m"
function defaultohlcv(base)::OhlcvData
    ohlcv = OhlcvData(defaultohlcvdataframe(), base, EnvConfig.cryptoquote, EnvConfig.cryptoexchange, "1m")
    return ohlcv
end

basesymbol(ohlcv::OhlcvData) = ohlcv.base
quotesymbol(ohlcv::OhlcvData) = ohlcv.qte
exchange(ohlcv::OhlcvData) = ohlcv.xch
interval(ohlcv::OhlcvData) = ohlcv.interval
dataframe(ohlcv::OhlcvData) = ohlcv.df

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
returns the relative forward/backward looking gain
"""
function relativegain(prices, baseix, gainix)
    gain = (prices[gainix] - prices[baseix]) / prices[baseix]
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

function addpivot!(df::DataFrame)
    if "pivot" in names(df)
        return df[:, :pivot]
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

pivot!(df::DataFrame) = addpivot!(df)[:, :pivot]

function addpivot!(ohlcv::OhlcvData)
    addpivot!(ohlcv.df)
    return ohlcv
end

pivot!(ohlcv::OhlcvData) = addpivot!(ohlcv.df)[:, :pivot]

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

"""
Reads OHLCV data generated by python FollowUpward
"""
function readcsv!(ohlcv::OhlcvData)::OhlcvData
    io = CSV.File(EnvConfig.datafile(ohlcv.base * "_OHLCV", "_df.csv"), types=Dict(1=>String, 2=>Float32, 3=>Float32, 4=>Float32, 5=>Float32, 6=>Float32, 7=>Float32))
    df = DataFrame(io)
    df[!, :opentime] = DateTime.(ZonedDateTime.(df[!, :Column1], "y-m-d H:M:s+z"), UTC)
    df = df[!, Not(:Column1)]
    df = df[!, Cols(:opentime, :)]
    df = df[!, save_cols]
    # df = df[!, Not(:opentime)]
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
    [df[ix, :start] = DateTime.(ZonedDateTime.(String(iodf[ix, :start]), "y-m-d H:M:s+z"), UTC) for ix in 1:len]
    [df[ix, :end] = DateTime.(ZonedDateTime.(String(iodf[ix, :end]), "y-m-d H:M:s+z"), UTC) for ix in 1:len]
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

