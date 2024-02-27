using Pkg


module TradeTest

using Test, Dates, Logging
using EnvConfig
using Trade

#
using Dates, EnvConfig, Trade

println("$(EnvConfig.now()): started")
messagelog = open(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt"), "w")
logger = SimpleLogger(messagelog)
defaultlogger = global_logger(logger)


EnvConfig.init(training)
startdt = DateTime("2022-01-01T00:00:00")
enddt = DateTime("2022-02-01T10:00:00")
cache = Trade.TradeCache(bases=["BTC", "MATIC"], startdt=startdt, enddt=enddt, messagelog=messagelog)

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
    close(cache.messagelog)
    println("$(EnvConfig.now()): finished")
end

end  # module