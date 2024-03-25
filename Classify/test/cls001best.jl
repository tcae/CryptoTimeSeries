module Cls001Best

using Test, Dates, Logging, CSV
using EnvConfig, Classify, CryptoXch, Assets, Features, Ohlcv


EnvConfig.init(production)
# EnvConfig.init(training)
Ohlcv.verbosity = 1
Features.verbosity = 1
Classify.verbosity = 2
Assets.verbosity = 2
println("$(EnvConfig.now()): started")
# messagelog = open(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt"), "w")
# logger = SimpleLogger(messagelog)
# defaultlogger = global_logger(logger)

update=true

cls = Classify.Classifier001()
xc= CryptoXch.XchCache(true)

enddt = nothing # nothing == latest;   DateTime("2024-03-22T20:40:00")
period = Day(10)  # Year(20)

bestdf = Classify.best!(cls, xc, 10, period, enddt, update, ["BTC"])
println("$(EnvConfig.now()) best! $bestdf")
println("$(EnvConfig.now()) size(cls.cfg))=$(size(cls.cfg))")


# @info "$(EnvConfig.now()): finished"
# global_logger(defaultlogger)
# close(cache.messagelog)
println("$(EnvConfig.now()): finished")

end  # module