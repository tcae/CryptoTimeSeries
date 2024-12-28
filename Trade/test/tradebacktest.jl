module TradeTest

using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, Trade, Classify, Ohlcv, CryptoXch

function prepare()
    # Redirect sigint to julia exception handling
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)
    EnvConfig.init(training)  # not production as this would result in real orders
    # CryptoXch.verbosity = 1
    Classify.verbosity = 2
    # Ohlcv.verbosity = 2
    Trade.verbosity = 2
    # EnvConfig.init(production)
    startdt = DateTime("2024-03-01T00:00:00") # nothing
    enddt =   DateTime("2024-06-06T09:00:00")
    demux_logger = TeeLogger(
        MinLevelLogger(FileLogger(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt")), Logging.Info),
        MinLevelLogger(ConsoleLogger(stdout), Logging.Info)
    )
    defaultlogger = global_logger(demux_logger)
    println()
    return startdt, enddt, defaultlogger
end

function wrapup(defaultlogger)
    @info "$(EnvConfig.now()): finished"
    global_logger(defaultlogger)
    println("$(EnvConfig.now()): finished\n")
end

function backtestcl001()
    startdt, enddt, defaultlogger = prepare()
    for base in ["BTC"]
        for regrwindow in [24*60]
            for headwindow in [60]
                for trendwindow in [0]
                    for gainthreshold in [0.01]
                        EnvConfig.setlogpath("Classifier001-$base-$headwindow-$regrwindow-$trendwindow")
                        println("TradeTest backtest Classifier001-$base-$headwindow-$regrwindow-$trendwindow")
                        println("$(EnvConfig.now()): started")
                        xc=CryptoXch.XchCache(true, startdt=startdt, enddt=enddt)
                        cl = Classify.Classifier001()
                        cache = Trade.tradeselection!(Trade.TradeCache(xc=xc, cl=cl), [base], assetonly=true)
                        # @info "backtest trademode=$(cache.trademode) trading config: $(cache.cfg)"
                        Classify.addreplaceconfig!(cl, base, regrwindow, gainthreshold, headwindow, trendwindow)
                        CryptoXch.updateasset!(cache.xc, "USDT", 0f0, 10000f0)
                        CryptoXch.writeassets(cache.xc, cache.xc.startdt)
                        Trade.tradeloop(cache)
                        assets = CryptoXch.portfolio!(cache.xc)
                        totalusdt = sum(assets.usdtvalue)
                        println("finish total USDT = $totalusdt")
                        CryptoXch.writeorders(cache.xc)
                        CryptoXch.writeassets(cache.xc, cache.xc.enddt)
                    end
                end
            end
        end
    end
    wrapup(defaultlogger)
end

function backtestcl002()
    startdt, enddt, defaultlogger = prepare()
    for base in ["BTC"]
        EnvConfig.setlogpath("Classifier002-$base")
        for window in [[4*60, 8*60, 16*60, 32*60, 64*60]] #, [16*60, 32*60, 64*60]]  #[15, 30, 60, 2*60, 4*60, 8*60, 16*60]] #, [60, 2*60, 4*60, 8*60, 16*60]] #, [4*60, 8*60, 16*60]]
            for rbt in [0.05f0] # , 0.1f0]
                for trendwindow in [0] #, 2*60]
                    println("TradeTest backtest Classifier002-$base-$window-$rbt-$trendwindow")
                    println("$(EnvConfig.now()): started")

                    xc=CryptoXch.XchCache(true, startdt=startdt, enddt=enddt)
                    cl = Classify.Classifier002(window=window, rbt=rbt, trendwindow=trendwindow)
                    println("cl=$cl")
                    cache = Trade.tradeselection!(Trade.TradeCache(xc=xc, cl=cl), [base], assetonly=true)
                    # @info "backtest trademode=$(cache.trademode) trading config: $(cache.cfg)"
                    CryptoXch.updateasset!(cache.xc, "USDT", 0f0, 10000f0)
                    CryptoXch.writeassets(cache.xc, cache.xc.startdt)
                    assets = CryptoXch.portfolio!(cache.xc)
                    totalusdt = sum(assets.usdtvalue)
                    println("start total USDT = $totalusdt")
                    println("Trade.verbosity=$(Trade.verbosity)")

                    Trade.tradeloop(cache)
                    # println("$(cl.dbgdf) \n$(describe(cl.dbgdf, :all))")
                    # println(describe(cl.dbgdf))
                    assets = CryptoXch.portfolio!(cache.xc)
                    totalusdt = sum(assets.usdtvalue)
                    println("finish total USDT = $totalusdt")
                    CryptoXch.writeorders(cache.xc)
                    CryptoXch.writeassets(cache.xc, cache.xc.enddt)
                end
            end
        end
    end
    wrapup(defaultlogger)
end

function backtestcl003()
    # Redirect sigint to julia exception handling
    startdt, enddt, defaultlogger = prepare()
    for base in ["BTC"]
        EnvConfig.setlogpath("Classifier003-$base")
        for shortwindow in [[15, 30, 60, 2*60, 4*60]]
            for longwindow in [[12*60, 24*60, 2*24*60, 4*24*60, 8*24*60]]
                for rbt in [0.01f0] #, 0.02f0, 0.04f0]
                    for srnt in [0.01f0] #, 0.02f0, 0.04f0]
                        for lrnt in [0.005f0]
                            println("TradeTest backtest Classifier003-$base-$shortwindow-$longwindow-$rbt-$srnt-$lrnt")
                            println("$(EnvConfig.now()): started")

                            xc=CryptoXch.XchCache(true, startdt=startdt, enddt=enddt)
                            cl = Classify.Classifier003(shortwindow=shortwindow, longwindow=longwindow, rbt=rbt, srnt=srnt, lrnt=lrnt)
                            println("cl=$cl")
                            cache = Trade.tradeselection!(Trade.TradeCache(xc=xc, cl=cl), [base], assetonly=true)
                            # @info "backtest trademode=$(cache.trademode) trading config: $(cache.cfg)"
                            CryptoXch.updateasset!(cache.xc, "USDT", 0f0, 10000f0)
                            CryptoXch.writeassets(cache.xc, cache.xc.startdt)
                            assets = CryptoXch.portfolio!(cache.xc)
                            totalusdt = sum(assets.usdtvalue)
                            println("start total USDT = $totalusdt")
                            println("Trade.verbosity=$(Trade.verbosity)")

                            Trade.tradeloop(cache)
                            # println("$(cl.dbgdf) \n$(describe(cl.dbgdf, :all))")
                            # println(describe(cl.dbgdf))
                            assets = CryptoXch.portfolio!(cache.xc)
                            totalusdt = sum(assets.usdtvalue)
                            println("finish total USDT = $totalusdt")
                            CryptoXch.writeorders(cache.xc)
                            CryptoXch.writeassets(cache.xc, cache.xc.enddt)
                        end
                    end
                end
            end
        end
    end
    wrapup(defaultlogger)
end

# backtestcl001()
# backtestcl002()
backtestcl003()

end  # module