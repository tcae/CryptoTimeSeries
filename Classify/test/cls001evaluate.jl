module Cls001Evaluate

using Test, Dates, Logging, CSV
using EnvConfig, Classify, CryptoXch, Assets


EnvConfig.init(production)
# EnvConfig.init(training)
# Ohlcv.verbosity = 1
# Features.verbosity = 2
Classify.verbosity = 2
println("$(EnvConfig.now()): started")
# messagelog = open(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt"), "w")
# logger = SimpleLogger(messagelog)
# defaultlogger = global_logger(logger)

test=false

cls = Classify.ClassifierSet001()
xc= CryptoXch.XchCache(true)
if test
    enddt = DateTime("2022-01-12T10:00:00")
    startdt = enddt - Day(30)
    startdt = DateTime("2022-01-01T00:00:00")

    # Classify.addreplaceconfig!(cls, "BTC", 1440, 0.02, 0, 0)
    # Classify.addreplaceconfig!(cls, "MATIC", 1440, 0.02, 0, 0)
    Classify.evaluate!(cls, xc, ["BTC"], [1440], [0.01], startdt, enddt)
    # Classify.evaluate!(cls, xc, ["BTC"], Classify.STDREGRWINDOWSET, Classify.STDGAINTHRSHLDSET, startdt, enddt)
else
    enddt = DateTime("2024-03-22T20:40:00")
    startdt = enddt - Day(10)  # Year(20)

    ad1 = Assets.read!(Assets.AssetData())
    println(ad1.basedf)
    cls = Classify.evaluate!(cls, xc, ad1.basedf[!, :base], Classify.STDREGRWINDOWSET, Classify.STDGAINTHRSHLDSET, startdt, enddt)
    kpifilename = EnvConfig.logpath("cls001evaluate.csv")
    EnvConfig.checkbackup(kpifilename)
    CSV.write(kpifilename, cls.cfg, decimal=',', delim=';')  # decimal as , to consume with European locale
end

println("$(EnvConfig.now()) evaluate! result: $(cls.cfg)")

# @info "$(EnvConfig.now()): finished"
# global_logger(defaultlogger)
# close(cache.messagelog)
println("$(EnvConfig.now()): finished")

end  # module