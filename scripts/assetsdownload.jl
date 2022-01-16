
import Pkg: activate
cd("$(@__DIR__)/..")
println("activated $(pwd())")
activate(pwd())

include("../src/env_config.jl")
include("../src/cryptoxch.jl")
include("../src/assets.jl")

module AssetsTest
using Dates, DataFrames

using ..Config, ..Assets, ..CryptoXch

Config.init(production)
# Config.init(training)
if Config.configmode == production
    ad1 = Assets.loadassets(dayssperiod=Dates.Year(4), minutesperiod=Dates.Week(4))
elseif Config.configmode == training
    enddt = Dates.now()
    startdt = enddt - Dates.Week(4.5 * 52)
    for base in Config.trainingbases
        println("$(Dates.now()): Loading $base from $startdt until $enddt as training data")
        ohlcv = CryptoXch.cryptodownload(base, "1m", startdt, enddt)
    end
else
    println("no operation")
end

end  # module