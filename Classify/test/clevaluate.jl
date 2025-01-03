"""
Evaluates Classifers of subtype AbstractClassifier
- iterates over all recently available basecoins with USDT quotecoin
- baseline is to use all canned data but basecoins and time range is also configurable
- logs all simulated trades
- consecutive trades is reported at a mean price
- a configurable fee is deducted
- binned gain distribution per basecoin is provided
- overall mean gain per basecoin as well as across basecoins is provided
"""
module ClEvaluate

using Test, Dates, Logging, CSV, DataFrames, Statistics
using EnvConfig, Classify, CryptoXch, Ohlcv


EnvConfig.init(training)
# EnvConfig.init(training)
Ohlcv.verbosity = 1
# Features.verbosity = 2
EnvConfig.verbosity = 2
Classify.verbosity = 3

startdt = nothing # DateTime("2024-03-01T00:00:00")
enddt =   nothing # DateTime("2024-06-06T09:00:00")
enddt = DateTime("2024-12-31T14:15:00")
startdt = DateTime("2024-11-12T14:15:00")
# startdt = enddt - Year(10)
EnvConfig.setlogpath("250103-Classifier011_$(startdt)_$(enddt)")
coins = ["BTC", "ETH", "XRP", "ADA", "GOAT", "DOGE", "SOL", "APEX", "MNT", "ONDO"]
# coins = nothing # ["BTC"]
classifiertype = Classify.Classifier011
println("evaluating: $coins")
df = Classify.evaluateclassifiers([classifiertype], coins, startdt, enddt)
# df = Classify.readsimulation()
kpidf, gdf = Classify.kpioverview(df, classifiertype)
sort!(kpidf, [:gain_sum], rev=true)
println(kpidf)
# println(gdf[kpidf[1, :groupindex]])
# println(gdf[kpidf[2, :groupindex]])
println("done")

end  # module