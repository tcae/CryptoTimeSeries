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
reloadtimes = [Time("04:00:00")]
cache = Trade.TradeCache(xc=CryptoXch.XchCache(true, startdt=startdt, enddt=enddt), reloadtimes=reloadtimes)
CryptoXch.updateasset!(cache.xc, "USDT", 0f0, 10000f0)

# startdt = Dates.now(UTC)
# enddt = nothing
# cache = Trade.TradeCache(startdt=startdt, enddt=enddt, messagelog=messagelog)
# try
    Trade.tradeloop(cache)
    # Trade.tradeloop(Trade.TradeCache(startdt=Dates.now(UTC), enddt=Dates.now(UTC)+Minute(3)))
    # Trade.tradeloop(Trade.TradeCache(startdt=Dates.now(UTC), enddt=nothing))

    # Trade.tradelooptest(Trade.TradeCache(startdt=DateTime("2022-01-01T00:00:00"), enddt=DateTime("2022-02-01T01:00:00")))
    # Trade.tradelooptest(Trade.TradeCache(startdt=Dates.now(UTC), enddt=Dates.now(UTC)+Minute(3)))
    # Trade.tradelooptest(Trade.TradeCache(startdt=Dates.now(UTC), enddt=nothing))
# catch ex
#     if isa(ex, InterruptException)
#         println("Ctrl+C pressed by trade_test")
#     end
# finally
    @info "$(EnvConfig.now()): finished"
    global_logger(defaultlogger)
    println("$(EnvConfig.now()): finished")
# end

end  # module