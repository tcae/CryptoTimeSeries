using Xch, Dates, EnvConfig

EnvConfig.init(production)
println("download all USDT crypto but those on Xch.baseignore list")
Xch.downloadallUSDT(Xch.XchCache(), Dates.now(UTC), Year(10), 1000000)

