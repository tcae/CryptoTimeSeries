include("../src/env_config.jl")
include("../src/assets.jl")

using ..Assets, ..EnvConfig

    # Config.init(Config.test)
    # Assets.cryptolistdownload(["btc"])

    EnvConfig.init(EnvConfig.production)
    Assets.cryptomarketdownload()

