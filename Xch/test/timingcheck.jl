module XchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, Xch, Bybit

Xch.verbosity = 3

function sim()
    startdt = DateTime("2024-03-29T09:58:00")
    enddt = DateTime("2024-03-29T10:00:00")
    xc=Xch.XchCache( startdt=startdt, enddt=enddt)
    println("starting $(EnvConfig.configmode) XchCache( startdt=$startdt, enddt=$enddt), tradetime=$(Xch.tradetime(xc))")
    println(Xch.tradetime(xc))
    for x in xc
        println("$(Xch.tradetime(xc))")
    end
    println("ending simulation at $(Xch.tradetime(xc))")
end

function prod()
    EnvConfig.init(production)
    minutes = 3
    enddt = nothing
    xc=Xch.XchCache( enddt=enddt)
    println("starting $(EnvConfig.configmode) XchCache( startdt=$(xc.startdt), enddt=$enddt), tradetime=$(Xch.tradetime(xc))")
    println(Xch.tradetime(xc))
    for x in xc
        println("$(Xch.tradetime(xc))")
        minutes -= 1
        (minutes <= 0) && break
    end
    println("ending production at $(Xch.tradetime(xc))")
end

function prodbusy()
    EnvConfig.init(production)
    minutes = 3
    enddt = nothing
    xc=Xch.XchCache( enddt=enddt)
    println("starting $(EnvConfig.configmode) XchCache( startdt=$(xc.startdt), enddt=$enddt), tradetime=$(Xch.tradetime(xc))")
    println(Xch.tradetime(xc))
    for x in xc
        println("$(Xch.tradetime(xc))")
        minutes -= 1
        (minutes == 2) && (println("sleep 150s"), sleep(150))
        (minutes <= 0) && break
    end
    println("ending production at $(Xch.tradetime(xc))")
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