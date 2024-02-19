using CryptoXch, EnvConfig
using Dates

bases =["BTC", "MATIC"]
enddt = Dates.now(UTC)
period = Year(10)
EnvConfig.init(production)
xc = CryptoXch.XchCache(true)
println("$(EnvConfig.now()) start")
CryptoXch.downloadupdate!(xc, bases, enddt, period)

println("$(EnvConfig.now()) finished")
