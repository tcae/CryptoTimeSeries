module TradeProduction

using Test, Dates, Logging, LoggingExtras
using EnvConfig, Trade, Classify, CryptoXch, Bybit

# # Disable the default behavior of exiting on Ctrl+C
# Base.exit_on_sigint(false)

# Redirect sigint to julia exception handling
ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

println("TradeProduction tradeproduction")
EnvConfig.setlogpath("Classifier010-production")
messagelogfn = EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt")
println("$(EnvConfig.now()): started - messages are logged in $messagelogfn")
demux_logger = TeeLogger(
    MinLevelLogger(FileLogger(messagelogfn, always_flush=true), Logging.Info),
    MinLevelLogger(ConsoleLogger(stdout), Logging.Info)
)
defaultlogger = global_logger(demux_logger)

CryptoXch.verbosity = 1
Classify.verbosity = 2
Trade.verbosity = 2
Bybit.verbosity = 3
EnvConfig.init(production)
enddt = nothing  # == continue endless
xc = CryptoXch.XchCache(true; enddt=enddt)
CryptoXch.setstartdt(xc, CryptoXch.tradetime(xc))
cl = Classify.Classifier010()
cfgnt = (regrwindow=3*24*60,trendthreshold=1f0, volatilitybuythreshold=-0.08f0, volatilitylongthreshold=0.02f0, volatilitysellthreshold=0.08f0, volatilityselltrendfactor=0f0)
cfgid = configurationid(cl, cfgnt)
println("cfgid=$cfgid for $cfgnt")
Classify.configureclassifier!(cl, cfgid, true)
reloadtimes = [Time("04:00:00")]
cache = Trade.TradeCache(xc=xc, cl=cl, reloadtimes=reloadtimes, trademode=Trade.buysell) # to exit , trademode=sellonly

# EnvConfig.init(production)
# startdt = Dates.now(UTC)
# enddt = nothing
# cache = Trade.TradeCache(startdt=startdt, enddt=enddt, messagelog=messagelog)
try
    Trade.tradeloop(cache)
    # Trade.tradeloop(startdt=Dates.now(UTC), enddt=Dates.now(UTC)+Minute(3)))
    # Trade.tradeloop(startdt=Dates.now(UTC), enddt=nothing))

    # Trade.tradelooptest(startdt=DateTime("2022-01-01T00:00:00"), enddt=DateTime("2022-02-01T01:00:00")))
    # Trade.tradelooptest(startdt=Dates.now(UTC), enddt=Dates.now(UTC)+Minute(3)))
    # Trade.tradelooptest(startdt=Dates.now(UTC), enddt=nothing))
# catch ex
#     if isa(ex, InterruptException)
#         println("Ctrl+C pressed by trade_test")
#     end
finally
    @info "$(EnvConfig.now()): finished"
    global_logger(defaultlogger)
end

end  # module