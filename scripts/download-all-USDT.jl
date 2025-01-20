using CryptoXch, Dates, EnvConfig

EnvConfig.init(production)
println("download all USDT crypto but those on CryptoXch.baseignore list")
CryptoXch.downloadallUSDT(CryptoXch.XchCache(), Dates.now(UTC), Year(10), 1000000)

