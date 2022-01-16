include("../src/env_config.jl")
include("../src/assets.jl")

using ..Assets

    # Config.init(Config.test)
    # Assets.cryptolistdownload(["btc"])

    Config.init(Config.production)
    Assets.cryptomarketdownload()

