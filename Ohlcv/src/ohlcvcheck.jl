# using Dates
using EnvConfig, Ohlcv

EnvConfig.init(production)
Ohlcv.check("BTC", cure=false)
