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
EnvConfig.verbosity = 3
Classify.verbosity = 3
EnvConfig.setlogpath("2429-Classifier005-ShortTest_TrendawareVolatilityTracker")

startdt = DateTime("2024-03-01T00:00:00") # nothing
enddt =   DateTime("2024-06-06T09:00:00")
# coins = ["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"]
coins = ["BTC"]
coinsdf = Ohlcv.liquidcoins(liquidrangeminutes=108*24*60)
filtered_df = filter(row -> row.basecoin in coins, coinsdf)
println("evaluating: $coins \n coinsdf=$coinsdf \n filtered_df=$filtered_df")
df = Classify.evaluateclassifiers([Classify.Classifier005], filtered_df, startdt, enddt)
# df = Classify.readsimulation()
kpidf, gdf = Classify.kpioverview(df, Classify.Classifier005)
sort!(kpidf, [:gain_sum], rev=true)
println(kpidf)
# println(gdf[kpidf[1, :groupindex]])
# println(gdf[kpidf[2, :groupindex]])
println("done")

end  # module