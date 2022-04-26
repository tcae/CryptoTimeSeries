using Ohlcv, EnvConfig
using Dates

EnvConfig.init(EnvConfig.training)
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
