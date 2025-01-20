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
Trade.verbosity = 3
Bybit.verbosity = 3
EnvConfig.init(production)
enddt = nothing  # == continue endless
xc = CryptoXch.XchCache(enddt=enddt)
CryptoXch.setstartdt(xc, CryptoXch.tradetime(xc))
cl = Classify.Classifier011()
# cfgnt = (regrwindow=24*60,longtrendthreshold=0.02f0, shorttrendthreshold=-0.04f0, volatilitybuythreshold=-0.01f0, volatilitysellthreshold=0.01f0, volatilitylongthreshold=0.0f0, volatilityshortthreshold=-1f0)
cfgnt = (regrwindow=24*60,longtrendthreshold=0.02f0, shorttrendthreshold=-1f0, volatilitybuythreshold=-0.01f0, volatilitysellthreshold=0.01f0, volatilitylongthreshold=1f0, volatilityshortthreshold=-1f0)
cfgid = configurationid(cl, cfgnt)
println("cfgid=$cfgid for $cfgnt")
Classify.configureclassifier!(cl, cfgid, true)
cache = Trade.TradeCache(xc=xc, cl=cl, trademode=Trade.buysell) # buysell sellonly quickexit notrade


try
    Trade.tradeloop(cache)
finally
    @info "$(EnvConfig.now()): finished"
    global_logger(defaultlogger)
end

end  # module