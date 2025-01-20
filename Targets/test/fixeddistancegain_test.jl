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

@testset "Targets::FixedDistanceGain tests" begin
    startdt = DateTime("2023-02-17T13:30:00")
    enddt = startdt + Hour(5)
    # EnvConfig.init(production)
    xc = CryptoXch.XchCache()
    ohlcv = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    # println(describe(ohlcv.df, :all))

    # fdg = Targets.FixedDistanceGain(30, Targets.defaultlabelthresholds)
    fdg = Targets.FixedDistanceGain(30, Targets.LabelThresholds(longbuy=0.11, longhold=0.001, shorthold=-0.001, shortbuy=-0.11))
    Targets.setbase!(fdg, ohlcv)
    println(Targets.df(fdg, DateTime("2023-02-17T13:31:00"), DateTime("2023-02-17T13:39:00")))
    println(fdg.df)
    println(describe(fdg.df, :all))
    # println("baseline fdg")
    # println(Targets.relativegain(fdg))
    # println(Targets.labels(fdg))

    # maxix = [ix+fdg.df.maxix[ix]-1 for ix in eachindex(ohlcv.df.opentime)]

    ixcheck = [ix <= fdg.df[ix, :maxix] <= ix + fdg.window for ix in eachindex(fdg.df[!,:maxix])]
    gaincheck = [(ohlcv.df[fdg.df[ix, :maxix], :pivot] .- ohlcv.df[ix, :pivot]) / ohlcv.df[ix, :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    # df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=fdg.df.opentime, maxix=fdg.df.maxix, relativegain=Targets.relativegain(fdg), labels=Targets.labels(fdg), ixcheck=ixcheck, gaincheck=gaincheck, gainchecktest=(gaincheck .== Targets.relativegain(fdg))))
    # println(df)
    @test all(ixcheck)
    @test all(gaincheck .== Targets.relativegain(fdg))

    # cut time range at start and end
    Ohlcv.timerangecut!(ohlcv, startdt+Minute(5), enddt-Minute(5))
    Targets.timerangecut!(fdg)
    # println("timerangecut $fdg")
    # df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=fdg.df.opentime, maxix=fdg.df.maxix, relativegain=Targets.relativegain(fdg), labels=Targets.labels(fdg)))
    # println(df)
    ixcheck = [ix <= fdg.df[ix, :maxix] <= ix + fdg.window for ix in eachindex(fdg.df[!,:maxix])]
    @test all(ixcheck)
    gaincheck = [(ohlcv.df[fdg.df[ix, :maxix], :pivot] .- ohlcv.df[ix, :pivot]) / ohlcv.df[ix, :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    @test all(gaincheck .== Targets.relativegain(fdg))

    # extend 5 minutes
    ohlcvxt = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt+Minute(5), enddt+Minute(5))
    ohlcv.df = ohlcvxt.df
    Ohlcv.timerangecut!(ohlcv, startdt+Minute(5), enddt+Minute(5))
    Targets.supplement!(fdg)
    ixcheck = [ix <= fdg.df[ix, :maxix] <= ix + fdg.window for ix in eachindex(fdg.df[!, :maxix])]
    @test all(ixcheck)
    gaincheck = [(ohlcv.df[fdg.df[ix, :maxix], :pivot] .- ohlcv.df[ix, :pivot]) / ohlcv.df[ix, :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    @test all(gaincheck .== Targets.relativegain(fdg))
    # df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=fdg.df.opentime, maxix=fdg.df.maxix, relativegain=Targets.relativegain(fdg), labels=Targets.labels(fdg), gaincheck=gaincheck, equal=(gaincheck .== Targets.relativegain(fdg))))
    # println(df)

    @test length(Targets.relativegain(fdg)) > 0
    Targets.removebase!(fdg)
    @test length(Targets.relativegain(fdg)) == 0
end  # of testset
return nothing
