using Dates, Flux, DataFrames
using Ohlcv, Features, TestOhlcv, Targets, EnvConfig, ROC
# using Plots
using Classify

EnvConfig.init(production)
base = "BTC"
ohlcv = Ohlcv.defaultohlcv(base)
Ohlcv.read!(ohlcv)
labelthresholds = Targets.defaultlabelthresholds
Classify.evaluate( ohlcv, labelthresholds);

# Classify.evaluatetest()
