module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit

EnvConfig.init(production)
xc = CryptoXch.XchCache(true)

assets = CryptoXch.portfolio!(xc)
println("assets: $assets")
end