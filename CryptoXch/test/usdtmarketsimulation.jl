module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit

EnvConfig.init(training)
xc = CryptoXch.XchCache()

EnvConfig.init(test)
usdtdf = CryptoXch.getUSDTmarket(xc)
println("\n$(EnvConfig.configmode) getUSDTmarket: $usdtdf")

EnvConfig.init(training)
usdtdf = CryptoXch.getUSDTmarket(xc)
println("\n$(EnvConfig.configmode) getUSDTmarket: $usdtdf")
end