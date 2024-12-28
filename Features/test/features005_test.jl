using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, Features, CryptoXch, TestOhlcv

Features.verbosity = 1 # 3

EnvConfig.init(test)
@testset "Features005 tests" begin
    startdt = DateTime("2023-02-17T13:30:00")
    enddt = startdt + Day(22) -Minute(1) # DateTime("2023-02-28T13:29:00")
    EnvConfig.init(production)
    xc = CryptoXch.XchCache(true)
    ohlcv = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    # println("ohlcvdf=$(ohlcv)")
    requestedfeatures = ["rw_15_regry", "rw_15_std", "rw_15_gain", "mm_60_mindist", "mm_60_maxdist", "rv_5_60"]
    f5 = Features.Features005(requestedfeatures)
    Features.setbase!(f5, ohlcv, usecache=false)
    # println("fdf: $f5")

    @test size(f5.fdf, 1) == size(Ohlcv.dataframe(f5.ohlcv), 1) - Features.requiredminutes(f5) + 1
    @test size(Features.features(f5), 2) == length(requestedfeatures)
    # println(names(Features.features(f5)))

    ohlcvshort = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt, enddt-Hour(6))
    Ohlcv.timerangecut!(ohlcvshort, startdt, enddt-Hour(6))
    # println("short ohlcvdf=$(ohlcvshort)")
    f5short = Features.Features005(requestedfeatures)
    Features.setbase!(f5short, ohlcvshort, usecache=false)
    # println("write short fdf: $f5short")
    Features.write(f5short)
    f5checkshort = Features.Features005(requestedfeatures)
    Features.setbase!(f5checkshort, ohlcvshort, usecache=true)
    @test size(f5checkshort.fdf, 1) == size(Ohlcv.dataframe(f5checkshort.ohlcv), 1) - Features.requiredminutes(f5checkshort) + 1
    # println("read short fdf: $f5checkshort")
    @test all(Matrix(f5short.fdf) .== Matrix(f5checkshort.fdf))

    f5checklong = Features.Features005(requestedfeatures)
    Features.setbase!(f5checklong, ohlcv, usecache=true)
    # println("read/calc check long fdf: $f5checklong")
    @test all(Matrix(f5.fdf) .== Matrix(f5checklong.fdf))

    ohlcv_3 = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt+Day(25), enddt+Day(25)) # later than stored cache
    Ohlcv.timerangecut!(ohlcv_3, startdt+Day(25), enddt+Day(25))
    f5_3 = Features.Features005(requestedfeatures)
    Features.setbase!(f5_3, ohlcv_3, usecache=true)
    @test size(f5_3.fdf, 1) == size(Ohlcv.dataframe(f5_3.ohlcv), 1) - Features.requiredminutes(f5_3) + 1
    # println("fdf f5_3 outside cache: $f5_3")
    # println(describe(f5.fdf))

    Features.delete(f5short)

    # println("before timerangecut $f5")
    Ohlcv.timerangecut!(ohlcv, startdt+Minute(60), enddt-Minute(60))
    # cutting at start should not impact features
    Features.timerangecut!(f5)
    # println("after timerangecut $f5")
    @test size(f5.fdf, 1) == size(Ohlcv.dataframe(f5.ohlcv), 1) - Features.requiredminutes(f5) + 60 + 1
    @test all(Matrix(f5.fdf) .== Matrix(f5checklong.fdf[begin:end-60, :]))

    @test !isnothing(Features.ohlcvdfview(f5))
    Features.removebase!(f5)
    @test isnothing(Features.ohlcvdfview(f5))

    """
    - first creation of in memory reference
    - save, extend ohlcv, create new f5, compare new with pure in memory
    - timecut into features, create new f5, compare new with pure in memory
    - timecut beyond saved features, create new f5, compare new with pure in memory
    """

end # testset

