module TradeTest

using Test, Dates, Logging, LoggingExtras
using EnvConfig, Trade, Classify, Assets, Ohlcv, CryptoXch, Features

println("TradeTest trade_test")
println("$(EnvConfig.now()): started")
demux_logger = TeeLogger(
    MinLevelLogger(FileLogger(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt")), Logging.Info),
    MinLevelLogger(ConsoleLogger(stdout), Logging.Info)
)
defaultlogger = global_logger(demux_logger)

CryptoXch.verbosity = 1
Classify.verbosity = 2
Ohlcv.verbosity = 2
Features.verbosity = 2
EnvConfig.init(training)
# EnvConfig.init(production)
startdt = DateTime("2024-03-19T00:00:00")
enddt = DateTime("2024-03-29T10:00:00")
cache = Trade.TradeCache(xc=CryptoXch.XchCache( startdt=startdt, enddt=enddt))
CryptoXch.updateasset!(cache.xc, "USDT", 0f0, 10000f0)

# startdt = Dates.now(UTC)
# enddt = nothing
@info "$(EnvConfig.now()): finished"
global_logger(defaultlogger)
println("$(EnvConfig.now()): finished")

end  # module