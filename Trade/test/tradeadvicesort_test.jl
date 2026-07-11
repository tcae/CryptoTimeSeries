using Test, Dates, DataFrames
using EnvConfig, Trade, Targets

println("$(EnvConfig.now()): started")

EnvConfig.init(test)
testdt = DateTime(2026, 1, 1)

function mkrow(base, label, probability)
    tdf = DataFrame(
        opentime=[testdt],
        lo_limit=Float32[100f0],
        lc_limit=Float32[101f0],
        so_limit=Float32[99f0],
        sc_limit=Float32[98f0],
        label=Targets.TradeLabel[label],
    )
    return (base=base, rowix=1, tradesdf=tdf, probability=(probability), configid=0)
end

rows = [
    mkrow("BTC", longopen, 0.2f0),
    mkrow("ETH", longclose, 0.1f0),
    mkrow("ADA", shortopen, 0.9f0),
]

sort!(rows, lt=Trade._strategyrows_lt)

@test rows[1].base == "ETH"
@test rows[2].base == "ADA"
@test rows[3].base == "BTC"
