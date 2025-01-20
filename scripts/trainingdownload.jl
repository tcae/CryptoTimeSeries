
# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())

module AssetsTest
using Dates, DataFrames

using EnvConfig, Assets, CryptoXch, Ohlcv

EnvConfig.init(training)
bases = EnvConfig.trainingbases
xc = CryptoXch.XchCache()
# bases = ["btc"]
enddt = Dates.now(Dates.UTC)
startdt = enddt - Dates.Week(4.5 * 52)
for base in bases
    println("$(EnvConfig.now()): Loading $base from $startdt until $enddt as training data")
    ohlcv = CryptoXch.cryptodownload(xc, base, "1m", startdt, enddt)
    Ohlcv.write(ohlcv)
end

end  # module
