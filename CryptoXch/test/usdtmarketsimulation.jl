module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit

EnvConfig.init(training)
xc = CryptoXch.XchCache()

usdtdf = CryptoXch.getUSDTmarket(xc)
println("getUSDTmarket: $usdtdf")
end