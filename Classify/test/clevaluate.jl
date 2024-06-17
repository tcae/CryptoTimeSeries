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
EnvConfig.setlogpath("2424-6_TrendawareVolatilityTracker")

startdt = nothing  # DateTime("2024-01-01T00:00:00")
enddt =   nothing  # DateTime("2024-04-12T10:00:00")
coins = ["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"]
coins = ["BTC"]
df = Classify.evaluateclassifiers([Classify.Classifier005], coins, startdt, enddt)
# df = Classify.readsimulation()
kpidf = Classify.kpioverview(df, Classify.Classifier005)
println("done")

end  # module