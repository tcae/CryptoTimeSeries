using DataFrames, Dates
using Test
using EnvConfig
using Ohlcv

mode = EnvConfig.configmode
EnvConfig.init(production)

base = "test"
ov = Ohlcv.defaultohlcv(base)
dfmin = DataFrame(
    opentime=[DateTime("2022-01-02T22:54:00")+Dates.Minute(i) for i in 0:9],
    open=[2f0 for i in 0:9],
    high=[2f0 for i in 0:9],
    low= [2f0 for i in 0:9],
    close=[2f0 for i in 0:9],
    basevolume=[2f0 for i in 0:9]
)
dfmin[2:4, :basevolume] .= 4f0
dfmin[6:9, :basevolume] .= 4f0
Ohlcv.setdataframe!(ov, dfmin)
Ohlcv.pivot!(ov)
vv = Ohlcv.volumeohlcv(ov, 16, Minute(2), 14, Minute(2))::Vector{AbstractDataFrame}
println(ov)
for ovp in vv
    println(ovp)
end
