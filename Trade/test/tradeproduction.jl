module TradeTest

using Test, Dates, Logging, LoggingExtras
using EnvConfig, Trade, Classify

println("$(EnvConfig.now()): started")
demux_logger = TeeLogger(
    MinLevelLogger(FileLogger(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt")), Logging.Info),
    MinLevelLogger(ConsoleLogger(stdout), Logging.Info)
)
defaultlogger = global_logger(demux_logger)

Classify.verbosity = 2
EnvConfig.init(production)
startdt = Dates.now(UTC)  # DateTime("2022-01-01T00:00:00")
enddt = nothing  # == continue endless
cache = Trade.TradeCache(bases=[], startdt=startdt, enddt=enddt, tradegapminutes=2, topx=10)

# EnvConfig.init(production)
# startdt = Dates.now(UTC)
# enddt = nothing
# cache = Trade.TradeCache(bases=["BTC", "MATIC"], startdt=startdt, enddt=enddt, messagelog=messagelog)
try
    Trade.tradeloop(cache)
    # Trade.tradeloop(Trade.TradeCache(bases=["BTC"], startdt=Dates.now(UTC), enddt=Dates.now(UTC)+Minute(3)))
    # Trade.tradeloop(Trade.TradeCache(bases=["BTC"], startdt=Dates.now(UTC), enddt=nothing))

    # Trade.tradelooptest(Trade.TradeCache(bases=["BTC"], startdt=DateTime("2022-01-01T00:00:00"), enddt=DateTime("2022-02-01T01:00:00")))
    # Trade.tradelooptest(Trade.TradeCache(bases=["BTC"], startdt=Dates.now(UTC), enddt=Dates.now(UTC)+Minute(3)))
    # Trade.tradelooptest(Trade.TradeCache(bases=["BTC"], startdt=Dates.now(UTC), enddt=nothing))
# catch ex
#     if isa(ex, InterruptException)
#         println("Ctrl+C pressed by trade_test")
#     end
finally
    @info "$(EnvConfig.now()): finished"
    global_logger(defaultlogger)
    println("$(EnvConfig.now()): finished")
end

end  # module