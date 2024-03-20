# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())

module AssetsTest
using Dates

using EnvConfig, Assets, CryptoXch, Features, Ohlcv


EnvConfig.init(production)
Ohlcv.verbosity = 1
Features.verbosity = 2
ad1 = Assets.read!(Assets.AssetData())
println(ad1.basedf)
for coin in eachrow(ad1.basedf)
    ohlcv = Ohlcv.defaultohlcv(coin.base)
    ohlcv = Ohlcv.read!(ohlcv)
    f4 = Features.Features004(ohlcv, usecache=true)
    if !isnothing(f4)
        # println(f4)
        Features.write(f4)
    end
end
println("$(EnvConfig.now()) finished")
end  # module
