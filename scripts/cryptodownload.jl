include("../src/env_config.jl")
include("../src/assets.jl")

using ..Assets

if false
    Config.init(Config.test)
    Assets.cryptolistdownload(["btc"])
else
    Config.init(Config.production)
    Assets.cryptomarketdownload()
end
