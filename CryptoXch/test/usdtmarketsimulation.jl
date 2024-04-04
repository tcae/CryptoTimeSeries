module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit

EnvConfig.init(training)
xc = CryptoXch.XchCache(true)

usdtdf = CryptoXch.getUSDTmarket(xc)
println("getUSDTmarket: $usdtdf")
end