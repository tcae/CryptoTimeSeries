
module VolatilityTracker

using DataFrames, Dates, Statistics, CSV
using EnvConfig, TestOhlcv, Ohlcv, Features

const VOLATILITYSTUDYFILE = "volatilitystudy.csv"

emptytrackdf()::DataFrame = DataFrame(regr=[], direction=[], openix=[], closeix=[], tradelen=[], gain=[], gainthreshold=[], gap=[], dirregrwin=[])

function trackregression(gainthreshold, dirregrwin, regrwindow, gap, f2::Features.Features002)
    df = emptytrackdf()
    piv = Ohlcv.dataframe(Features.ohlcv(f2)).pivot[f2.firstix:f2.lastix]
    callopenix = Int32[]
    putopenix = Int32[]
    # gap = 5  # 5 minute gap between trades
    callcloseix = -gap
    putcloseix = -gap

    function closetrades!(openix, closeix, direction, handleall)
        while length(openix) > 0
            # handle open call positions
            gain = direction == "call" ? Ohlcv.relativegain(piv, openix[begin], closeix) : -Ohlcv.relativegain(piv, openix[begin], closeix)
            tl = closeix - openix[begin]
            push!(df, (regr=regrwindow, direction=direction, openix=openix[begin], closeix=closeix, tradelen=tl, gain=gain, gainthreshold=gainthreshold, gap=gap, dirregrwin=dirregrwin))
            openix = deleteat!(openix, 1)
            if !handleall
                break
            end
        end
        return openix
    end

    function opentrades!(openix, closeix)
        if ((f2.regr[regrwindow].std[closeix] / f2.regr[regrwindow].regry[closeix]) > gainthreshold) && ((length(openix) == 0) || (closeix - openix[end] >= gap))
            push!(openix, closeix)
        end
        return openix
    end

    #* idea: adapt gap with direction gradient
    for ix in eachindex(piv)
        relativedirgain = Features.relativegain(f2.regr[dirregrwin].regry[ix], f2.regr[dirregrwin].grad[ix], dirregrwin)
        if piv[ix] > f2.regr[regrwindow].regry[ix] + f2.regr[regrwindow].std[ix]
            if (relativedirgain < -gainthreshold) || (ix - putcloseix >= gap)
                callopenix = closetrades!(callopenix, ix, "call", (relativedirgain < -gainthreshold))
                callcloseix = ix
            end
            if !(relativedirgain > gainthreshold)
                opentrades!(putopenix, ix)
            end
        elseif piv[ix] < f2.regr[regrwindow].regry[ix] - f2.regr[regrwindow].std[ix]
            if (relativedirgain > gainthreshold) || (ix - putcloseix >= gap)
                putopenix = closetrades!(putopenix, ix, "put", (relativedirgain > gainthreshold))
                callcloseix = ix
            end
            if !(relativedirgain < -gainthreshold)
                opentrades!(callopenix, ix)
            end
        end
    end
    return df
end

function kpi(df, regr, dirregrwin, direction, gainthreshold, gap)
    # println(describe(df))
    if size(df, 1) > 0
        gainvec = df[!, :gain]
        tlvec = df[!, :tradelen]
        cumgain = round(sum(gainvec)*100, digits=3)
        count = length(gainvec)
        meangain = round(mean(gainvec)*100, digits=3)
        mediangain = round(median(gainvec)*100, digits=3)
        mediantl = round(median(tlvec), digits=0)
        maxtl = round(maximum(tlvec), digits=0)
        return (regr=regr, dir=direction, dirregrwin=dirregrwin, gap=gap, gainthreshold=gainthreshold, cumgain=cumgain, meangain=meangain, mediangain=mediangain, count=count, mediantradelen=mediantl, maxtradelen=maxtl)
    else
        return (regr=regr, dir=direction, dirregrwin=dirregrwin, gap=gap, gainthreshold=gainthreshold, cumgain=0.0, meangain=0.0, mediangain=0.0, count=0, mediantradelen=0.0, maxtradelen=0.0)
    end
end

function trackohlc(ohlcv::Ohlcv.OhlcvData)::DataFrame
    df = DataFrame()
    println("$(EnvConfig.now()) calculating F002 features")
    f2 = Features.Features002(ohlcv)
    rkeys = sort(collect(keys(f2.regr)))
    for kix in eachindex(rkeys)
        rw = rkeys[kix]
        println("\n$(EnvConfig.now()) processing regression window $rw")
        println(describe(DataFrame(grad=f2.regr[rw].grad, regry=f2.regr[rw].regry, std=f2.regr[rw].std), :all))
        if kix != firstindex(rkeys)
            for gainthreshold in [0.005, 0.01, 0.02]
                for gap in [2, 5]
                    # println("$(EnvConfig.now()) analyzing volatility performace of regression window $rw, gainthreshold=$gainthreshold, gap=$gap")
                    kdf = trackregression(gainthreshold, rkeys[kix-1], rw, gap, f2)
                    ckdf = filter(row -> row[:direction] == "call", kdf)
                    push!(df, kpi(filter(row -> row[:direction] == "call", kdf), rw, rkeys[kix-1], "call", gainthreshold, gap))
                    push!(df, kpi(filter(row -> row[:direction] == "put", kdf), rw, rkeys[kix-1], "put", gainthreshold, gap))
                end
            end
        end
    end
    return df
end

function trackasset(base::String, startdt::Dates.DateTime=DateTime("2017-07-02T22:54:00"), period=Dates.Year(10); select=nothing)
    ohlcv = Ohlcv.defaultohlcv(base)
    enddt = startdt + period
    Ohlcv.read!(ohlcv)
    subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
    println("loaded $ohlcv")
    df = trackohlc(ohlcv)
    println(df)
    pf = EnvConfig.logpath(VOLATILITYSTUDYFILE)
    CSV.write(pf, df, decimal='.', delim=';')
    println("done")
end

end  #of module

using EnvConfig, Dates

EnvConfig.init(production)
VolatilityTracker.trackasset("BTC")
# VolatilityTracker.trackasset("BTC", DateTime("2023-01-02T22:54:00"), Dates.Day(40))
