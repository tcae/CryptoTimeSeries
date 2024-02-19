# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())

module AssetsTest
using Dates

using EnvConfig, Assets, CryptoXch


EnvConfig.init(production)
ad1 = Assets.loadassets!(Assets.AssetData())
println(ad1.basedf)
end  # module
