using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, Features, TestOhlcv

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 3
Features.verbosity = 3

EnvConfig.init(test)

function f6config(ohlcv)
    f6 = Features.Features006()
    Features.addregry!(f6, window=5, offset=2)
    Features.addgrad!(f6, window=5, offset=0)
    Features.addstd!(f6, window=10)
    Features.addmaxdist!(f6, window=10, offset=5)
    Features.addmindist!(f6, window=10, offset=10)
    Features.addrelvol!(f6, short=5, long=60)

    # println("ohlcvdf=$(ohlcv)")
    Features.setbase!(f6, ohlcv, usecache=false)
    return f6
end

#TODO supplement tests of a ohlcv view

@testset "Features006 tests" begin
    startdt = DateTime("2025-02-17T13:30:00")
    enddt = startdt + Hour(20) - Minute(1) 
    EnvConfig.init(production)
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
    odf = Ohlcv.dataframe(ohlcv)
    (verbosity >= 3) && println("stardt=$startdt enddt=$enddt ohlcv=$ohlcv")
    # Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    f6 = f6config(ohlcv)
    (verbosity >= 3) && println("f6 $f6, fdf size=$(size(f6.fdf)), fdfno size=$(size(f6.fdfno)), fdf describe $(describe(f6.fdf)), fdfno describe $(describe(f6.fdfno))")
    os = 2
    ryf = Features._regry(f6, window=5, offset=os)
    ryffdfcol = Features.fdfcol(f6, ryf)
    ryffdfnocol = Features.fdfnocol(f6, ryf)
    (verbosity >= 3) && println("f6.fdf[end, $ryffdfcol]=$(f6.fdf[end, ryffdfcol]), f6.fdfno[end-$os, $ryffdfnocol]=$(f6.fdfno[end-os, ryffdfnocol]), odf[end, :pivot]=$(odf[end, :pivot])")
    @test f6.fdf[end, ryffdfcol] == ((odf[end, :pivot] - f6.fdfno[end-os, ryffdfnocol]) / odf[end, :pivot]) # relative regression residual to pivot price

    @test size(f6.fdf, 1) == size(Ohlcv.dataframe(f6.ohlcv), 1) - f6.maxoffset
    @test size(f6.fdfno, 1) == size(Ohlcv.dataframe(f6.ohlcv), 1)
    # println(names(Features.features(f6)))

    ohlcvshort = TestOhlcv.testohlcv("SINE", startdt, enddt-Hour(1))
    (verbosity >= 3) && println("stardt=$startdt enddt=$enddt ohlcvshort=$ohlcvshort")
    # Ohlcv.timerangecut!(ohlcvshort, startdt, enddt-Hour(1))
    f6short = f6config(ohlcvshort)
    # println("short ohlcvdf=$(ohlcvshort)")
    (verbosity >= 3) && println("f6short $f6short, fdf size=$(size(f6short.fdf)), fdfno size=$(size(f6short.fdfno)), fdf describe $(describe(f6short.fdf)), fdfno describe $(describe(f6short.fdfno))")
    for n in names(f6.fdfno)
        (verbosity >= 3) && println("short $n test fdfno=$(all(view(f6.fdfno[!,n], 1:lastindex(f6short.fdfno[!,n])) .== f6short.fdfno[!,n]))")
        @test all(view(f6.fdfno[!,n], 1:lastindex(f6short.fdfno[!,n])) .== f6short.fdfno[!,n])
    end
    for n in names(f6.fdf)
        (verbosity >= 3) && println("short $n test fdf=$(all(view(f6.fdf[!,n], 1:lastindex(f6short.fdf[!,n])) .== f6short.fdf[!,n]))")
        @test all(view(f6.fdf[!,n], 1:lastindex(f6short.fdf[!,n])) .== f6short.fdf[!,n])
    end


    Ohlcv.setdataframe!(ohlcvshort, Ohlcv.dataframe(ohlcv))
    Features.supplement!(f6short)
    (verbosity >= 3) && println("f6short extended $f6short, fdf size=$(size(f6short.fdf)), fdfno size=$(size(f6short.fdfno)), fdf describe $(describe(f6short.fdf)), fdfno describe $(describe(f6short.fdfno))")
    for n in names(f6.fdfno)
        (verbosity >= 3) && println("short extended $n test fdfno=$(all(f6.fdfno[!,n] .== f6short.fdfno[!,n]))")
        if (verbosity >= 3) && !all(f6.fdfno[!,n] .== f6short.fdfno[!,n])
            rep = false
            for ix in eachindex(f6.fdfno[!,n])
                if !(f6.fdfno[ix, n] == f6short.fdfno[ix, n])
                    if !rep
                        println("NOK f6.fdfno[$ix, $n]=$(f6.fdfno[ix, n]) != f6short.fdfno[$ix, $n]=$(f6short.fdfno[ix, n]) ")
                        println("NOK f6.fdfno[$(max(ix-20,1)):$(ix+20), $n]=$(f6.fdfno[max(ix-20,1):ix+20, n])\nf6short.fdfno[$(max(ix-20,1)):$(ix+20), $n]=$(f6short.fdfno[max(ix-20,1):ix+20, n])")
                        rep=true
                    end
                else
                    if rep
                        println("OK! f6.fdfno[$ix, $n]=$(f6.fdfno[ix, n]) == f6short.fdfno[$ix, $n]=$(f6short.fdfno[ix, n]) ")
                    end
                    rep = false
                end
            end
        end
        @test all(f6.fdfno[!,n] .== f6short.fdfno[!,n])
    end
    for n in names(f6.fdf)
        (verbosity >= 3) && println("short extended $n test fdf=$(all(f6.fdf[!,n] .== f6short.fdf[!,n]))")
        @test all(f6.fdf[!,n] .== f6short.fdf[!,n])
    end
    featdf = Features.features(f6)
    (verbosity >= 3) && println("names(f6, features)=$(names(featdf))")
    @test length(names(featdf)) == length(names(f6.fdf)) - 1 # without opentime
    @test size(featdf, 1) == size(f6.fdf, 1)


    @test all(Matrix(f6.fdf) .== Matrix(f6short.fdf))

    (verbosity >= 3) && (Features.verbosity = 3)
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    f6 = Features.Features006()
    Features.addregry!(f6, window=4*60, offset=2)
    Features.setbase!(f6, ohlcv, usecache=true)
    (verbosity >= 3) && println("f6=$f6 size(f6.fdfno)=$(size(f6.fdfno))  size(f6.fdf)=$(size(f6.fdf)) names(f6.fdfno)=$(names(f6.fdfno)) names(f6.fdf)=$(names(f6.fdf)) ohlcv=$(f6.ohlcv)")
    @test size(f6.fdfno, 1) - f6.maxoffset == size(f6.fdf, 1)
    @test size(f6.fdfno, 1) == size(Ohlcv.dataframe(f6.ohlcv), 1)

    staleview = @view Ohlcv.dataframe(ohlcv)[21:end, :]
    stalef6 = Features.Features006()
    Features.addgrad!(stalef6, window=60, offset=0)
    stalef6.ohlcv = ohlcv
    stalef6.fdfno = DataFrame(
        "ry+60" => fill(1.0f0, size(staleview, 1) - 1),
        "rg+60" => fill(2.0f0, size(staleview, 1) - 1),
        "opentime" => staleview[1:(end-1), "opentime"],
    )
    staleotdf = DataFrame("rg+5" => fill(3.0f0, size(staleview, 1)))
    Features._opentime!(staleotdf, stalef6, staleview, nothing, nothing)
    @test size(staleotdf, 1) == size(staleview, 1)
    @test staleotdf[!, "opentime"] == staleview[!, "opentime"]

end # testset
return
