using TestOhlcv, Features, Ohlcv, Dates, DataFrames

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(2)
ohlcv = TestOhlcv.testohlcv("SINEUSDT", startdt, enddt)
df = Ohlcv.dataframe(ohlcv)
f2 = Features.Features002(ohlcv; regrwindows=[5, 15])
if true
    f12x = Features.features12x1m01(f2)
    println(f12x[1:12, ["open-p00", "open-p01"]])
else
    f12x = Features.regressionfeatures01(f2,  11, 5, [15, 60], 5, 4*60, "relminuteofday")
    println("f12x[6, :grad5m00]=$(f12x[6, :grad5m00])  f12x[1, :grad5m01]=$(f12x[1, :grad5m01])")
    println("f12x[6, :disty5m00]=$(f12x[6, :disty5m00])  f12x[1, :disty5m01]=$(f12x[1, :disty5m01])")
    println(f12x[1:12, [:grad5m00, :grad5m01]])
end
println("size(f12x)=$(size(f12x))")
f12xd = describe(f12x)
println(f12xd)
