module XchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, Xch, Bybit

EnvConfig.init(training)
xc = Xch.XchCache()

EnvConfig.init(test)
usdtdf = Xch.getUSDTmarket(xc)
println("\n$(EnvConfig.configmode) getUSDTmarket: $usdtdf")

EnvConfig.init(training)
usdtdf = Xch.getUSDTmarket(xc)
println("\n$(EnvConfig.configmode) getUSDTmarket: $usdtdf")
end