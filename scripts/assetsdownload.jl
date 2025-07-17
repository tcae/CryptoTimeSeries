# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())

module AssetsTest
using Dates

using EnvConfig, Assets, CryptoXch, Features, Ohlcv


EnvConfig.init(production)
Ohlcv.verbosity = 2
Features.verbosity = 2
ad1 = Assets.loadassets!(Assets.AssetData())
println(ad1.basedf)

coins = size(ad1.basedf, 1)
for (ix, coin) in enumerate(eachrow(ad1.basedf))
    ohlcv = Ohlcv.defaultohlcv(coin.base)
    print("\r$(EnvConfig.now()): ( $ix of $coins) F4 update $(ohlcv.base) ")
    ohlcv = Ohlcv.read!(ohlcv)
    f4 = Features.Features004(ohlcv, usecache=true)
    if !isnothing(f4)
        Features.write(f4)
    end
end
println("$(EnvConfig.now()) finished")

end  # module
