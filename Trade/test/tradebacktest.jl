module TradeTest

using Test, Dates, Logging, LoggingExtras
using EnvConfig, Trade, Classify, Assets, Ohlcv, CryptoXch, TradingStrategy

# Redirect sigint to julia exception handling
ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)
for base in ["BTC"]
    for regrwindow in [24*60]
        for headwindow in [0]
            for trendwindow in [0]
                for gainthreshold in [0.01]
                    EnvConfig.setlogpath("Classifier001-$base-$headwindow-$regrwindow-$trendwindow")
                    println("TradeTest backtest")
                    println("$(EnvConfig.now()): started")
                    demux_logger = TeeLogger(
                        MinLevelLogger(FileLogger(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt")), Logging.Info),
                        MinLevelLogger(ConsoleLogger(stdout), Logging.Info)
                    )
                    defaultlogger = global_logger(demux_logger)

                    # CryptoXch.verbosity = 1
                    # TradingStrategy.verbosity = 1
                    # Classify.verbosity = 2
                    # Ohlcv.verbosity = 2
                    # Trade.verbosity = 1
                    EnvConfig.init(training)  # not production as this would result in real orders
                    # EnvConfig.init(production)
                    startdt = DateTime("2023-01-19T00:00:00")
                    enddt = DateTime("2024-05-20T10:00:00")
                    reloadtimes = []  # [Time("04:00:00")]
                    xc=CryptoXch.XchCache(true, startdt=startdt, enddt=enddt)
                    tc = TradingStrategy.train!(TradingStrategy.TradeConfig(xc), ["BTC"], datetime=startdt, assetonly=true)
                    tc.cfg[1, :headwindow] = headwindow
                    tc.cfg[1, :regrwindow] = regrwindow
                    tc.cfg[1, :trendwindow] = trendwindow
                    tc.cfg[1, :gainthreshold] = gainthreshold
                    @info "backtest trading config: $(tc.cfg)"
                    cache = Trade.TradeCache(xc=xc, tc=tc, reloadtimes=reloadtimes)
                    CryptoXch.updateasset!(cache.xc, "USDT", 0f0, 10000f0)
                    CryptoXch.writeassets(cache.xc, cache.xc.startdt)
                    Trade.tradeloop(cache)
                    CryptoXch.writeorders(cache.xc)
                    CryptoXch.writeassets(cache.xc, cache.xc.enddt)
                    @info "$(EnvConfig.now()): finished"
                    global_logger(defaultlogger)
                end
            end
        end
    end
end
println("$(EnvConfig.now()): finished")

end  # module