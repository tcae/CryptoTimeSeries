using Xch, EnvConfig
using Dates

bases =["BTC", "MATIC"]
enddt = Dates.now(UTC)
period = Year(10)
EnvConfig.init(production)
xc = Xch.XchCache()
println("$(EnvConfig.now()) start")
Xch.downloadupdate!(xc, bases, enddt, period)

println("$(EnvConfig.now()) finished")
