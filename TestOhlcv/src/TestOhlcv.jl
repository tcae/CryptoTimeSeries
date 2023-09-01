
module TestOhlcv

using Dates, DataFrames, Logging
using EnvConfig, Ohlcv

"""
Returns cumulative sine function samples by adding sines on each other described by parameters given as a tuple (periodsamples, offset, amplitude).
The parameter samples defines the length of the returned functioni samples and level the zero level of the function.
"""
function sinesamples(samples, level, sineparams)
    y = zeros(Float32, samples) .+ level
    x = collect(1:samples) .- 1
    for (periodsamples, offset, amplitude) in sineparams
        println("sinedata: periodsamples=$periodsamples, offset=$offset, level=$level, amplitude=$amplitude")
        # show(DataFrame(x=x, y=y))
        @. y += sin((x + offset) * 2 * pi / (periodsamples)) * amplitude
    end
    x .+= 1
    return x, y
end

"""
returns ohlcv data starting 2019-01-02 01:11 for - by default 5.7 years
"""
function sinedata(periodminutes, totalminutes=3000000, offset=0, overlayperiodmultiple = 1)
    price::Float32 = 200
    volumeconst::Float32 = 100
    amplitude::Float32 = 0.007  # 0.7% of price
    firstutc = DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s")
    firstutc = round(firstutc, Dates.Minute)
    # lastutc = round(lastutc, Dates.Minute)
    # first is the reference point to reproduce the pattern
    # minutes = Int((Dates.Minute(lastutc - firstutc) + Dates.Minute(1)) / Dates.Minute(1))
    # display(minutes)
    # minutes are used as degrees, i.e. 1 full sinus = 360 degree = 6h
    x1 = [(m + offset) * pi * 2 / periodminutes for m in 1:totalminutes]
    # x = [m * pi / (minutes/2) for m in 1:minutes]
    y1 = sin.(x1)
    variation = -cos.(x1) .* 0.01
    # display(y)
    timestamp = [firstutc + Dates.Minute(m) for m in 1:totalminutes]
    x2 = [(m + offset) * pi * 2 / (periodminutes * overlayperiodmultiple) for m in 1:totalminutes]
    y2 = overlayperiodmultiple > 0 ? sin.(x2) : 0
    y = y1 .* y2 + y2
    # display(timestamp)
    # open =   (y / 4)
    # high =   (y / 2)
    # low =    (y / 2)
    # close =  (y / 4)
    open::Vector{Float32} =  price .* (y .* amplitude .+ 1 .+ variation ./ 4)
    high::Vector{Float32} =  price .* (y .* amplitude .+ 1 .+ 0.01 ./ 2)
    low::Vector{Float32} =   price .* (y .* amplitude .+ 1 .- 0.01 ./ 2)
    close::Vector{Float32} = price .* (y .* amplitude .+ 1 .- variation ./ 4)
    @assert low <= open <= high "low $(low) <= open $(open) <= high $(high)"
    @assert low <= close <= high "low $(low) <= close $(close) <= high $(high)"
    volume::Vector{Float32} = (1.1 .- abs.(y1)) .* volumeconst
    df = DataFrame(opentime=timestamp, open=open, high=high, low=low, close=close, basevolume=volume)
    df[:, :pivot] = Ohlcv.pivot(df)
    return df
end

function oldsinedata(periodminutes, periods)
    price = 200
    volumeconst = 100
    amplitude = 0.007  # 0.5% of price
    firstutc = DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s")
    firstutc = round(firstutc, Dates.Minute)
    # lastutc = round(lastutc, Dates.Minute)
    # first is the reference point to reproduce the pattern
    # minutes = Int((Dates.Minute(lastutc - firstutc) + Dates.Minute(1)) / Dates.Minute(1))
    # display(minutes)
    # minutes are used as degrees, i.e. 1 full sinus = 360 degree = 6h
    minutes = periodminutes * periods
    x = [m * pi * 2 / periodminutes for m in 1:minutes]
    y = (sin.(x) + sin.(2 * x)) / 2
    variation = -cos.(x) .* 0.01
    # display(y)
    timestamp = [firstutc + Dates.Minute(m) for m in 1:minutes]
    # display(timestamp)
    # open =   (y / 4)
    # high =   (y / 2)
    # low =    (y / 2)
    # close =  (y / 4)
    open =  price .* (y .* amplitude .+ 1 .+ variation ./ 4)
    high =  price .* (y .* amplitude .+ 1 .+ 0.01 ./ 2)
    low =   price .* (y .* amplitude .+ 1 .- 0.01 ./ 2)
    close = price .* (y .* amplitude .+ 1 .- variation ./ 4)
    volume = (1.1 .- abs.(y)) .* volumeconst
    df = DataFrame(opentime=timestamp, open=open, high=high, low=low, close=close, basevolume=volume)
    df[:, :pivot] = Ohlcv.pivot(df)
    return df
end

singlesine(periodminutes, amplitude, initialoffset, level, totalminutes) =
    [sin((ix + initialoffset)/periodminutes*2*pi) * amplitude + level for ix in 1:totalminutes]

function triplesine(period1, period2, period3, amplitude, initialoffset, level, totalminutes)
    y1 = singlesine(period1, amplitude, initialoffset, 0, totalminutes)
    y2 = singlesine(period2, amplitude, initialoffset, 0, totalminutes)
    y3 = singlesine(period3, amplitude, initialoffset, 0, totalminutes)
    y = y1 + y2 + y3 .+ level
    return y
end

function singlesineohlcv(period, amplitude, initialoffset, level, totalminutes)
    open = singlesine(period, amplitude, initialoffset, level*0.98, totalminutes)
    close = singlesine(period, amplitude, initialoffset, level*1.02, totalminutes)
    high = singlesine(period, amplitude+0.1, initialoffset, level*1.05, totalminutes)
    low = singlesine(period, amplitude-0.1, initialoffset, level*0.95, totalminutes)
    y = singlesine(period, amplitude, initialoffset, 0, totalminutes)

    volume = (1.1 .+ abs.(y)) .* 1
    return open, high, low, close, volume
end

function triplesineohlcv(period1, period2, period3, amplitude, initialoffset, level, totalminutes)
    open = triplesine(period1, period2, period3, amplitude, initialoffset, level*0.98, totalminutes)
    close = triplesine(period1, period2, period3, amplitude, initialoffset, level*1.02, totalminutes)
    high = triplesine(period1, period2, period3, amplitude+0.1, initialoffset, level*1.05, totalminutes)
    low = triplesine(period1, period2, period3, amplitude-0.1, initialoffset, level*0.95, totalminutes)
    y = triplesine(period1, period2, period3, amplitude, initialoffset, 0, totalminutes)

    volume = (1.1 .+ abs.(y)) .* 1
    return open, high, low, close, volume
end


function sinedata_test()
    ohlcv = sinedata(20, 3)
    # display(ohlcv.df)
end

function singlesine(startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m")
    # totalminutes = Dates.value(ceil(enddt, Dates.Minute(1)) - floor(startdt, Dates.Minute(1)))
    df = sinedata(2*60, 3000000)
    # df.opentime = [startdt + Dates.Minute(m) for m in 1:totalminutes]
    df = df[startdt .<= df.opentime .<= enddt, :]
    # println("test single sinus $(size(df))")
    df = Ohlcv.accumulate(df, interval)
    return df
end

function doublesine(startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m")
    # totalminutes = Dates.value(ceil(enddt, Dates.Minute(1)) - floor(startdt, Dates.Minute(1)))
    df = sinedata(2*60, 3000000, 0, 10.5)
    # df.opentime = [startdt + Dates.Minute(m) for m in 1:totalminutes]
    df = df[startdt .<= df.opentime .<= enddt, :]
    # println("test double sinus $(size(df))")
    df = Ohlcv.accumulate(df, interval)
    return df
end

function testdataframe(base::String, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    dispatch = Dict(
        "sine" => singlesine,
        "doublesine" => doublesine
    )
    testbase = base
    if !(base in keys(dispatch))
        @info "unknown testohlcv test base: $base - fallback: using sine to fill $base"
        testbase = "sine"
    end
    df = dispatch[testbase](startdt, enddt, interval)
    if df === nothing
        @warn "unexpected missing df" base testbase startdt enddt interval
    # else
    #     println("testdataframe df size: $(size(df,1)) names: $(names(df))  $base $startdt $enddt $interval")
    end
    Ohlcv.addpivot!(df)
    return df
end

function testohlcv(base::String, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    ohlcv = Ohlcv.defaultohlcv(base)
    df = testdataframe(base, startdt, enddt, interval, cryptoquote)
    ohlcv = Ohlcv.setdataframe!(ohlcv, df)
    return ohlcv
end

end
