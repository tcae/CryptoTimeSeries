using Ohlcv, EnvConfig, CryptoXch, Assets, Features
using Dates

function test()
    EnvConfig.init(EnvConfig.training)
    m1 = Features.regressionwindows001
    m2 = Features.regressionwindows002
    bases = EnvConfig.trainingbases
    # bases = ["btc"]

    for base in bases
        # println("$(EnvConfig.now()): Loading $base from $startdt until $enddt as training data")
        ohlcv = Ohlcv.defaultohlcv(base)
        ohlcv = Ohlcv.setinterval!(ohlcv, "1m")
        # Ohlcv.read!(ohlcv)
        # Ohlcv.fillgaps!(ohlcv)
        # Ohlcv.write(ohlcv)
    end
end

EnvConfig.init(EnvConfig.production)
Assets.loadassets()
