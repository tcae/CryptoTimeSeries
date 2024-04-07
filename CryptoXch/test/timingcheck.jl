module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit

CryptoXch.verbosity = 3

function sim()
    startdt = DateTime("2024-03-29T09:58:00")
    enddt = DateTime("2024-03-29T10:00:00")
    xc=CryptoXch.XchCache(true, startdt=startdt, enddt=enddt)
    println("starting $(EnvConfig.configmode) XchCache(true, startdt=$startdt, enddt=$enddt), tradetime=$(CryptoXch.tradetime(xc))")
    println(CryptoXch.tradetime(xc))
    for x in xc
        println("$(CryptoXch.tradetime(xc))")
    end
    println("ending simulation at $(CryptoXch.tradetime(xc))")
end

function prod()
    EnvConfig.init(production)
    minutes = 3
    enddt = nothing
    xc=CryptoXch.XchCache(true, enddt=enddt)
    println("starting $(EnvConfig.configmode) XchCache(true, startdt=$(xc.startdt), enddt=$enddt), tradetime=$(CryptoXch.tradetime(xc))")
    println(CryptoXch.tradetime(xc))
    for x in xc
        println("$(CryptoXch.tradetime(xc))")
        minutes -= 1
        (minutes <= 0) && break
    end
    println("ending production at $(CryptoXch.tradetime(xc))")
end

function prodbusy()
    EnvConfig.init(production)
    minutes = 3
    enddt = nothing
    xc=CryptoXch.XchCache(true, enddt=enddt)
    println("starting $(EnvConfig.configmode) XchCache(true, startdt=$(xc.startdt), enddt=$enddt), tradetime=$(CryptoXch.tradetime(xc))")
    println(CryptoXch.tradetime(xc))
    for x in xc
        println("$(CryptoXch.tradetime(xc))")
        minutes -= 1
        (minutes == 2) && (println("sleep 150s"), sleep(150))
        (minutes <= 0) && break
    end
    println("ending production at $(CryptoXch.tradetime(xc))")
end

EnvConfig.init(training)
sim()
println()
EnvConfig.init(production)
sim()
println()
prod()
println()
prodbusy()

end