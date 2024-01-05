
module VolatilityTracker

using DataFrames, Dates
using EnvConfig, TestOhlcv, Ohlcv, Features

emptytrackdf()::DataFrame = DataFrame(regr=[], direction=[], openix=[], closeix=[], tradelen=[], gain=[], gainthreshold=[], gap=[], dirregrwin=[])

function trackregression(gainthreshold, dirregrwin, regrwindow, f2::Features.Features002)
    df = emptytrackdf()
    piv = Ohlcv.dataframe(Features.ohlcv(f2)).pivot[f2.firstix:f2.lastix]
    callopenix = Int32[]
    putopenix = Int32[]
    gap = 5  # 5 minute gap between trades
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
            # while length(callopenix) > 0
            #     # handle open call positions
            #     gain = Ohlcv.relativegain(piv, callopenix[begin], ix)
            #     tl = ix - callopenix[begin]
            #     push!(df, (regr=regrwindow, direction="call", openix=callopenix[begin], closeix=ix, tradelen=tl, gain=gain, gainthreshold=gainthreshold, gap=gap, dirregrwin=dirregrwin))
            #     callopenix = deleteat!(callopenix, 1)
            #     if (relativedirgain > -gainthreshold) && (ix - callcloseix < gap)
            #         callcloseix = ix
            #         break
            #     end
            #     callcloseix = ix
            # end
            # if ((f2.regr[regrwindow].std[ix] / f2.regr[regrwindow].regry[ix]) > gainthreshold) && ((length(putopenix) == 0) || (ix - putopenix[end] >= gap))
            #     push!(putopenix, ix)
            # end
        elseif piv[ix] < f2.regr[regrwindow].regry[ix] - f2.regr[regrwindow].std[ix]
            if (relativedirgain > gainthreshold) || (ix - putcloseix >= gap)
                putopenix = closetrades!(putopenix, ix, "put", (relativedirgain > gainthreshold))
                callcloseix = ix
            end
            if !(relativedirgain < -gainthreshold)
                opentrades!(callopenix, ix)
            end
            # if (length(putopenix) > 0) && ((relativedirgain > gainthreshold) || (ix - putcloseix >= gap))
            #     # handle open put positions
            #     gain = -Ohlcv.relativegain(piv, putopenix[begin], ix)
            #     tl = ix - putopenix[begin]
            #     push!(df, (regr=regrwindow, direction="put", openix=putopenix[begin], closeix=ix, tradelen=tl, gain=gain, gainthreshold=gainthreshold, gap=gap, dirregrwin=dirregrwin))
            #     putopenix = deleteat!(putopenix, 1)
            #     putcloseix = ix
            # end
            # if ((f2.regr[regrwindow].std[ix] / f2.regr[regrwindow].regry[ix]) > gainthreshold) && ((length(callopenix) == 0) || (ix - callopenix[end] >= gap))
            #     push!(callopenix, ix)
            # end
        end
    end
    return df
end

function trackohlc(ohlcv::Ohlcv.OhlcvData)::DataFrame
    df = emptytrackdf()
    println("$(EnvConfig.now()) calculating F002 features")
    f2 = Features.Features002(ohlcv)
    rkeys = sort(collect(keys(f2.regr)))
    gainthreshold = 0.005  # 0.5%
    println("$(EnvConfig.now()) starting evaluation using $(gainthreshold*100)% as gain threshold")
    for kix in eachindex(rkeys)
        rw = rkeys[kix]
        println("\n$(EnvConfig.now()) processing regression window $rw")
        println(describe(DataFrame(grad=f2.regr[rw].grad, regry=f2.regr[rw].regry, std=f2.regr[rw].std), :all))
        for gainthreshold in [0.005, 0.01, 0.02, 0.1]
            if kix != firstindex(rkeys)
                println("\n$(EnvConfig.now()) analyzing volatility performace of regression window $rw with gainthreshold=$gainthreshold")
                kdf = trackregression(gainthreshold, rkeys[kix-1], rw, f2)
                ckdf = filter(row -> row[:direction] == "call", kdf)
                pkdf = filter(row -> row[:direction] == "put", kdf)
                println(describe(ckdf))
                println(describe(pkdf))
                callgainvec = ckdf[!, :gain]
                if length(callgainvec) > 0
                    callgain = sum(callgainvec)
                    callcount = count(i -> typeof(i)<:AbstractFloat, callgainvec)
                    callavggain = round(callgain/callcount*100, digits=3)
                else
                    callgain = callavggain = 0.0
                    callcount = 0
                end
                putgainvec = pkdf[!, :gain]
                if length(callgainvec) > 0
                    putgain = sum(putgainvec)
                    putcount = count(i -> typeof(i)<:AbstractFloat, callgainvec)
                    putavggain = round(putgain/putcount*100, digits=3)
                else
                    putgain = putavggain = 0.0
                    putcount = 0
                end
                totalcount = callcount + putcount
                totalavggain = totalcount > 0 ? round((callgain + putgain)/totalcount*100, digits=3) : 0.0
                println("callgain=$callgain with $callcount trades = $callavggain% average gain per call trade")
                println("putgain=$putgain with $putcount trades = $putavggain% average gain per put trade")
                println("$totalavggain% average gain per trade \n")
                if size(df, 1) > 0
                    df = vcat(df, kdf)
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
    println("done")
end

end  #of module

using EnvConfig, Dates

EnvConfig.init(production)
# VolatilityTracker.trackasset("BTC")
VolatilityTracker.trackasset("BTC", DateTime("2022-01-02T22:54:00"), Dates.Day(40))
