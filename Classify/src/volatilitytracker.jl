
module VolatilityTracker

using DataFrames, Dates, Statistics, CSV
using EnvConfig, TestOhlcv, Ohlcv, Features

const VOLATILITYSTUDYFILE = "volatilitystudy.csv"

function trackregression(f2::Features.Features002; asset, gainthreshold, dirregrwin, regrwindow, gap, dircontrol, selfmonitor)
    tradedf = DataFrame()
    calllastok = putlastok = true
    piv = Ohlcv.dataframe(Features.ohlcv(f2)).pivot[f2.firstix:f2.lastix]
    callopenix = Int32[]
    putopenix = Int32[]
    # gap = 5  # 5 minute gap between trades
    callcloseix = -gap
    putcloseix = -gap

    function closetrades!(openix, closeix, direction, handleall)
        while length(openix) > 0
            # handle open call positions
            tl = closeix - openix[begin]
            if direction == "call"
                gain = Ohlcv.relativegain(piv, openix[begin], closeix)
                calllastok = (gain >= selfmonitor)
                handleall = calllastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, direction=direction, openix=openix[begin], closeix=closeix, tradelen=tl, gain=gain, gainthreshold=gainthreshold, gap=gap, dirregrwin=dirregrwin, selfmonitor=selfmonitor))
            else
                gain = -Ohlcv.relativegain(piv, openix[begin], closeix)
                putlastok = (gain >= selfmonitor)
                handleall = putlastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, direction=direction, openix=openix[begin], closeix=closeix, tradelen=tl, gain=gain, gainthreshold=gainthreshold, gap=gap, dirregrwin=dirregrwin, selfmonitor=selfmonitor))
            end
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
                callopenix = closetrades!(callopenix, ix, "call", (relativedirgain < -gainthreshold) && dircontrol)
                callcloseix = ix
            end
            if (!(relativedirgain > gainthreshold) || !dircontrol) && putlastok
                opentrades!(putopenix, ix)
            end
        elseif piv[ix] < f2.regr[regrwindow].regry[ix] - f2.regr[regrwindow].std[ix]
            if (relativedirgain > gainthreshold) || (ix - putcloseix >= gap)
                putopenix = closetrades!(putopenix, ix, "put", (relativedirgain > gainthreshold) && dircontrol)
                callcloseix = ix
            end
            if (!(relativedirgain < -gainthreshold) || !dircontrol) && calllastok
                opentrades!(callopenix, ix)
            end
        end
    end
    return tradedf
end

function kpi(tradedf, asset, regr, dirregrwin, direction, gainthreshold, gap, dircontrol=dircontrol, selfmonitor=selfmonitor)
    # println(describe(tradedf))
    if size(tradedf, 1) > 0
        gainvec = tradedf[!, :gain]
        tlvec = tradedf[!, :tradelen]
        cumgain = round(sum(gainvec)*100, digits=3)
        count = length(gainvec)
        meangain = round(mean(gainvec)*100, digits=3)
        mediangain = round(median(gainvec)*100, digits=3)
        mediantl = round(median(tlvec), digits=0)
        maxtl = round(maximum(tlvec), digits=0)
        return (asset=asset, regr=regr, dir=direction, dirregrwin=dirregrwin, gap=gap, gainthreshold=gainthreshold, dircontrol=dircontrol, selfmonitor=selfmonitor, cumgain=cumgain, meangain=meangain, mediangain=mediangain, count=count, mediantradelen=mediantl, maxtradelen=maxtl)
    else
        return (asset=asset, regr=regr, dir=direction, dirregrwin=dirregrwin, gap=gap, gainthreshold=gainthreshold, dircontrol=dircontrol, selfmonitor=selfmonitor, cumgain=0.0, meangain=0.0, mediangain=0.0, count=0, mediantradelen=0.0, maxtradelen=0.0)
    end
end

function trackohlc(ohlcv::Ohlcv.OhlcvData)
    tradedf = DataFrame()
    kpidf = DataFrame()
    println("$(EnvConfig.now()) calculating F002 features")
    f2 = Features.Features002(ohlcv)
    asset = Ohlcv.basesymbol(Features.ohlcv(f2))
    rkeys = sort(collect(keys(f2.regr)))
    for kix in eachindex(rkeys)
        rw = rkeys[kix]
        println("\n$(EnvConfig.now()) processing regression window $rw")
        println(describe(DataFrame(grad=f2.regr[rw].grad, regry=f2.regr[rw].regry, std=f2.regr[rw].std), :all))
        if kix != firstindex(rkeys)
            for gainthreshold in [0.005, 0.01, 0.02]
                for gap in [2, 5]
                    for dircontrol in [true, false]
                        for selfmonitor in [-Inf, 0.0]  # [-Inf, 0.0]
                            # println("$(EnvConfig.now()) analyzing volatility performace of regression window $rw, gainthreshold=$gainthreshold, gap=$gap")
                            tdf = trackregression(f2, asset=asset, gainthreshold=gainthreshold, dirregrwin=rkeys[kix-1], regrwindow=rw, gap=gap, dircontrol=dircontrol, selfmonitor=selfmonitor)
                            push!(kpidf, kpi(filter(row -> row[:direction] == "call", tdf), asset, rw, rkeys[kix-1], "call", gainthreshold, gap, dircontrol, selfmonitor))
                            push!(kpidf, kpi(filter(row -> row[:direction] == "put", tdf), asset, rw, rkeys[kix-1], "put", gainthreshold, gap, dircontrol, selfmonitor))
                            if size(tradedf, 1) > 0
                                tradedf = vcat(tradedf, tdf)
                            end
                        end
                    end
                end
            end
        end
    end
    return kpidf, tradedf
end

function trackasset(base::String, startdt::Dates.DateTime=DateTime("2017-07-02T22:54:00"), period=Dates.Year(10); select=nothing)
    ohlcv = Ohlcv.defaultohlcv(base)
    enddt = startdt + period
    Ohlcv.read!(ohlcv)
    subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
    println("loaded $ohlcv")
    kpidf, tradedf = trackohlc(ohlcv)
    println(kpidf)
    pf = EnvConfig.logpath(VOLATILITYSTUDYFILE)
    CSV.write(pf, kpidf, decimal='.', delim=';')
    println("done")
end

end  #of module

using EnvConfig, Dates

EnvConfig.init(production)
EnvConfig.setlogpath("BTC_VolatilityTracker")
VolatilityTracker.trackasset("BTC")
# VolatilityTracker.trackasset("BTC", DateTime("2023-01-02T22:54:00"), Dates.Day(40))
