using EnvConfig, Dates, DataFrames, CryptoXch, Features, Ohlcv
# startdt = DateTime("2022-05-15T22:54:00")
enddt = DateTime("2023-02-18T13:29:00")
period = Day(1)
startdt = enddt - period
EnvConfig.init(production)
EnvConfig.setlogpath("F4StorageTest")
xc = CryptoXch.XchCache()
ohlcv = CryptoXch.cryptodownload(xc, "SINEUSDT", "1m", startdt, enddt)
println(ohlcv)
println(describe(ohlcv.df))
f4 = Features.Features004(ohlcv; firstix=lastindex(ohlcv.df[!, "opentime"])-6, lastix=lastindex(ohlcv.df[!, "opentime"])-1, regrwindows=[15, 60], usecache=false)
dfendix = lastindex(ohlcv.df[!, :opentime])
println("ohlcv.df[end-8:end]=$(ohlcv.df[dfendix-8:dfendix, :])")
for (regr, df) in f4.rw
    println("regr=$regr df=$df")
end
f4s = Features.Features004(ohlcv; firstix=lastindex(ohlcv.df[!, "opentime"])-4, lastix=lastindex(ohlcv.df[!, "opentime"])-3, regrwindows=[15, 60], usecache=false)
for (regr, df) in f4s.rw
    println("regr=$regr df=$df")
end
println("before write: $(Features.file(f4s))")
Features.write(f4s)
println("after write: $(Features.file(f4s))")

f4r = Features.Features004(ohlcv; firstix=lastindex(ohlcv.df[!, "opentime"])-6, lastix=lastindex(ohlcv.df[!, "opentime"])-1, regrwindows=[15, 60], usecache=true)
for (regr, df) in f4r.rw
    println("regr=$regr df=$df")
    println("f4==f4r=$(f4.rw[regr]==select!(f4r.rw[regr], names(f4.rw[regr])))")
end
Features.delete(f4s)
println("after delete: $(Features.file(f4s))")

# create ohlcv.df view with last 2 missing and only caluclated as well as another view with 2 before, 2 as calculated in the limited view and 2 after
# for both calculate F4 with at least 2 regrwindowos
# store limited version, read it and compare against version that is fully calculated to ensure read and concat works as expected
