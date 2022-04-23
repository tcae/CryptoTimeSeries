
# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())

module AssetsTest
using Dates, DataFrames

using EnvConfig, Assets, CryptoXch, Ohlcv

EnvConfig.init(training)
bases = EnvConfig.trainingbases
# bases = ["btc"]
enddt = Dates.now()
startdt = enddt - Dates.Week(4.5 * 52)
for base in bases
    println("$(EnvConfig.now()): Loading $base from $startdt until $enddt as training data")
    ohlcv = CryptoXch.cryptodownload(base, "1m", startdt, enddt)
    Ohlcv.write(ohlcv)
end

end  # module
