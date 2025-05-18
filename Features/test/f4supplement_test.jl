using Features, Test
Features.verbosity = 1
verbosity = 1

@testset "f4supplement" begin
    
function comp(f4long, f4short)
    if (size(first(values(f4long.rw)), 1) < size(first(values(f4short.rw)), 1))
        f = f4long
        f4long = f4short
        f4short = f
    end
    offset = f4short.ohlcvoffset - f4long.ohlcvoffset
    (verbosity >= 3) && println("offset=$offset, keys(f4short)=$(keys(f4short.rw)), keys(f4long)=$(keys(f4long.rw))")
    for rw in keys(f4short.rw)
        (verbosity >= 3) && println("names(f4short.rw[$rw])=$(names(f4short.rw[rw])) ($(f4short.rw[rw][begin, :opentime]) - $(f4short.rw[rw][end, :opentime])), names(f4long.rw[$rw])=$(names(f4long.rw[rw])) ($(f4long.rw[rw][begin, :opentime]) - $(f4long.rw[rw][end, :opentime]))")
        for col in names(f4short.rw[rw])
            @test all([f4short.rw[rw][ix, col] == f4long.rw[rw][ix+offset, col] for ix in eachindex(f4short.rw[rw][!, col])])
        end
    end
end

function testsupplement(;period, firstix, lastix)
    enddt = DateTime("2023-02-18T13:29:00")
    startdt = enddt - period
    EnvConfig.init(production)
    EnvConfig.setlogpath("F4StorageTest")
    xc = CryptoXch.XchCache()
    ohlcv = CryptoXch.cryptodownload(xc, "SINE", "1m", startdt, enddt)
    (verbosity >= 3) && println("ohlcv=$ohlcv")
    # f4 is larger reference
    f4a = Features.Features004(ohlcv; firstix=firstix, lastix=lastix, regrwindows=[15, 60], usecache=false)
    (verbosity >= 3) && println("f4a=$f4a")
    f4b = Features.Features004(ohlcv; firstix=firstix+2, lastix=lastix-2, regrwindows=[15, 60], usecache=false)
    (verbosity >= 3) && println("f4b=$f4b")
    comp(f4a, f4b)
    Features.supplement!(f4b, ohlcv, firstix=firstix, lastix=lastix)
    (verbosity >= 3) && println("f4b after supplement=$f4b")
    comp(f4a, f4b)
end

# testsupplement(period=Minute(70), firstix=70-8, lastix=70)
testsupplement(period=Minute(70), firstix=1, lastix=70)

end # testset
