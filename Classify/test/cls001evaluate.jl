module Cls001Evaluate

using Test, Dates, Logging
using EnvConfig, Classify, CryptoXch

println("$(EnvConfig.now()): started")
# messagelog = open(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt"), "w")
# logger = SimpleLogger(messagelog)
# defaultlogger = global_logger(logger)


EnvConfig.init(training)
# enddt = DateTime("2022-01-12T10:00:00")
# startdt = enddt - Day(30)

enddt = DateTime("2024-03-20T21:20:00")
startdt = enddt - Year(20)

cls = Classify.Classifier001()
xc= CryptoXch.XchCache(true)
# Classify.addconfig!(cls, "BTC", 1440, 0.02, true)
# Classify.addconfig!(cls, "MATIC", 1440, 0.02, true)
# Classify.evaluate!(cls, xc, ["BTC"], [1440], [0.01], startdt, enddt)
Classify.evaluate!(cls, xc, ["BTC"], Classify.STDREGRWINDOWSET, Classify.STDGAINTHRSHLDSET, startdt, enddt)
println("evaluate! result: $(cls.cfg)")

# @info "$(EnvConfig.now()): finished"
# global_logger(defaultlogger)
# close(cache.messagelog)
println("$(EnvConfig.now()): finished")

end  # module