using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test
using Logging, LoggingExtras
using EnvConfig, Features, Targets, TestOhlcv, Ohlcv, CryptoXch

# with_logger(TimestampTransformerLogger(current_logger(), BeginningMessageLocation();
#                                               format = "yyyy-mm-dd HH:MM:SSz")) do
Targets.verbosity = 1

EnvConfig.init(test)
# EnvConfig.init(production)
println("\nconfig mode = $(EnvConfig.configmode)")

function minitrenddisturbance!(ohlcv, startix, minutes, amplitude, delta)
    for ix in startix:(startix+minutes-1)
        ohlcv.df[ix, :pivot] = ohlcv.df[startix, :pivot] * (1 + amplitude + (ix-1) * delta)
    end
end

function noshorttrends(trd)
    lbl = trd.df[1, :label]
    startix = 1
    for ix in eachindex(trd.df[!, :label])
        if lbl != trd.df.label[ix]
            if (lbl in [longbuy, shortbuy]) && ((ix-startix) < trd.minwindow)
                if !((startix == 40) && (ix == 42))
                    (Targets.verbosity >= 3) && println("too small $(string(lbl)) trend detected at $startix:$(ix-1)")
                    return false
                end
            end
            startix = ix
            lbl = trd.df[ix, :label]
        end
    end
    return true
end

@testset "Targets::Trend tests" begin
    startdt = DateTime("2023-02-17T13:30:00")
    enddt = startdt + Hour(6)
    # EnvConfig.init(production)
    xc = CryptoXch.XchCache()
    ohlcv = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    # println(describe(ohlcv.df, :all))

    # trd = Targets.Trend(30, Targets.defaultlabelthresholds)
    thres = Targets.LabelThresholds(longbuy=0.11, longhold=0.01, shorthold=-0.01, shortbuy=-0.11)
    trd = Targets.Trend(3, 30, thres)
    minitrenddisturbance!(ohlcv, 100, 4, 0.9 * thres.shortbuy, 0.01 * thres.shortbuy)
    minitrenddisturbance!(ohlcv, 110, 4, 1.1 * thres.shortbuy, 0.01 * thres.shortbuy)
    minitrenddisturbance!(ohlcv, 40, 2, 0.9 * thres.longbuy, 0.01 * thres.longbuy)
    minitrenddisturbance!(ohlcv, 150, 4, 1.1 * thres.longbuy, 0.01 * thres.longbuy)
    Targets.setbase!(trd, ohlcv)

    # println("trade labels: $(Targets.tradelabels(trd))")
    # println(Targets.df(trd, DateTime("2023-02-17T13:31:00"), DateTime("2023-02-17T13:39:00")))
    # println(trd.df)
    # println(describe(trd.df, :all))

    # println("baseline trd")
    # println(Targets.relativegain(trd))
    # println(Targets.labels(trd))


    ixcheck = [ix - trd.maxwindow < trd.df[ix, :relix] <= ix for ix in eachindex(trd.df[!,:relix])]
    gaincheck = [(ohlcv.df[ix, :pivot] .- ohlcv.df[trd.df[ix, :relix], :pivot]) / ohlcv.df[trd.df[ix, :relix], :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    if (Targets.verbosity >= 3)
        df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=trd.df.opentime, relix=trd.df.relix, reldiff=trd.df.reldiff, labels=Targets.labels(trd), tmp2labels=trd.df.tmp2label, tmprelix=trd.df.tmprelix, tmpreldiff=trd.df.tmpreldiff, tmplabels=trd.df.tmplabel, ixcheck=ixcheck, relativegain=Targets.relativegain(trd), gaincheck=gaincheck, gainchecktest=(gaincheck .== Targets.relativegain(trd))))
        println(df)
    end
    @test noshorttrends(trd)
    @test all(ixcheck)
    @test all(gaincheck .== Targets.relativegain(trd))

    # cut time range at start and end
    Ohlcv.timerangecut!(ohlcv, startdt+Minute(5), enddt-Minute(5))
    Targets.timerangecut!(trd)
    # println("timerangecut $trd")
    # df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=trd.df.opentime, maxix=trd.df.maxix, relativegain=Targets.relativegain(trd), labels=Targets.labels(trd)))
    # println(df)
    # ixcheck = [ix - trd.window < trd.df[ix, :minix] <= ix for ix in eachindex(trd.df[!,:maxix])]
    # @test all(ixcheck)
    # gaincheck = [(ohlcv.df[ix, :pivot] .- ohlcv.df[trd.df[ix, :maxix], :pivot]) / ohlcv.df[trd.df[ix, :maxix], :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    # @test all(gaincheck .== Targets.relativegain(trd))

    # extend 5 minutes
    ohlcvxt = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt+Minute(5), enddt+Minute(5))
    ohlcv.df = ohlcvxt.df
    Ohlcv.timerangecut!(ohlcv, startdt+Minute(5), enddt+Minute(5))
    Targets.supplement!(trd)
    # ixcheck = [ix <= trd.df[ix, :maxix] <= ix + trd.window for ix in eachindex(trd.df[!, :maxix])]
    # @test all(ixcheck)
    # gaincheck = [(ohlcv.df[ix, :pivot] .- ohlcv.df[trd.df[ix, :maxix], :pivot]) / ohlcv.df[trd.df[ix, :maxix], :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    # @test all(gaincheck .== Targets.relativegain(trd))
    # df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=trd.df.opentime, maxix=trd.df.maxix, relativegain=Targets.relativegain(trd), labels=Targets.labels(trd), gaincheck=gaincheck, equal=(gaincheck .== Targets.relativegain(trd))))
    # println(df)

    @test length(Targets.relativegain(trd)) > 0
    Targets.removebase!(trd)
    @test length(Targets.relativegain(trd)) == 0
    ret="end"
end  # of testset
return 