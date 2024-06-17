module TradeCorrelation

using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, Trade, Classify, Ohlcv, CryptoXch

println("$(EnvConfig.now()): TradeCorrelation started")

CryptoXch.verbosity = 1
Classify.verbosity = 2
Ohlcv.verbosity = 1
EnvConfig.init(training)
# EnvConfig.init(production)
Classify.DEBUG = true

function compressdf(df)
    dfc = DataFrame()
    lp = p = lastrow = nothing
    count = 0
    for row in eachrow(df)
        p = (row.simtp, coalesce(row.side, "Noop"))
        if isnothing(lp) || (p != lp)
            if !isnothing(lastrow)
                push!(dfc, (lastrow..., count=count), promote=true)
            end
            lp = p
            lastrow = row
            count = 1
        else
            count += 1
        end
    end
    push!(dfc, (lastrow..., count=count))
    sort!(dfc, [:opentime])
    return dfc
end

startbudget = 100000f0
startdt = DateTime("2024-03-19T00:00:00")  # cl.ohlcv.df[begin, :opentime]
enddt = DateTime("2024-03-29T10:00:00")  # cl.ohlcv.df[end, :opentime]
base = "BTC"

xc=CryptoXch.XchCache(true, startdt=startdt, enddt=enddt)
cl = Classify.Classifier001()
cache = Trade.tradeselection!(Trade.TradeCache(xc=xc, cl=cl), [base], assetonly=true)
Classify.basetrain!(cache.cl.bd["BTC"]; regrwindows=[Classify.STDREGRWINDOW], gainthresholds=[Classify.STDGAINTHRSHLD], startdt, enddt)
println(cache.cl.bd["BTC"].cfg)

# @info "backtest trademode=$(cache.trademode) trading config: $(cache.cfg)"
# Classify.addreplaceconfig!(cl, base, Classify.STDREGRWINDOW, Classify.STDGAINTHRSHLD, 0, 0)
CryptoXch.updateasset!(cache.xc, "USDT", 0f0, 10000f0)
CryptoXch.writeassets(cache.xc, cache.xc.startdt)
Trade.tradeloop(cache)
assets = CryptoXch.portfolio!(cache.xc)
totalusdt = sum(assets.usdtvalue)
println("finish total USDT = $totalusdt")
CryptoXch.writeorders(cache.xc)
CryptoXch.writeassets(cache.xc, cache.xc.enddt)


endbudget = sum(assets[!, :usdtvalue])
gain = (endbudget - startbudget) / startbudget * 100
# println(cl.dbgdf)
println("tradeloop from $(cache.xc.startdt) until $(cache.xc.enddt) with gain=$gain%")
println("describe(cl.bd[BTC].cfg)=$(describe(cl.bd["BTC"].cfg, :all))")
println("describe(cl.cfg)=$(describe(cl.cfg, :all))")
println("describe(cache.cl.bd[BTC].dbgdf)=$(describe(cache.cl.bd["BTC"].dbgdf, :all))")
println("Classify.train! from $(startdt) until $(enddt) with gain=$(sum(cl.bd["BTC"].cfg[!, :simgain]))% of cl.cfg with size=$(size(cl.cfg))")

df = vcat(cache.xc.closedorders, cache.xc.orders)
println("describe(cache.xc.orders)=$(describe(df, :all))")
df = leftjoin(cache.cl.bd["BTC"].dbgdf, df, on = :opentime => :created, matchmissing = :notequal)
println("describe(df)=$(describe(df, :all))")
simtpcheck = coalesce.(df[!, :simtp], Classify.noop)
sidecheck = coalesce.(df[!, :side], "noop")
filtedf = DataFrame()
filtedf[:, :simsellcheck] = simtpcheck .== Classify.sell
filtedf[:, :simbuycheck] = simtpcheck .== Classify.buy
filtedf[:, :sidesellcheck] = sidecheck .== "Sell"
filtedf[:, :sidebuycheck] = sidecheck .== "Buy"
cnt = (simsellcount = sum(filtedf[!, :simsellcheck]), simbuycount = sum(filtedf[!, :simbuycheck]), sidesellcount = sum(filtedf[!, :sidesellcheck]), sidebuycount = sum(filtedf[!, :sidebuycheck]))
filter = (filtedf[!, 1] .|| filtedf[!, 2]) .|| (filtedf[!, 3] .|| filtedf[!, 4])
df = df[filter, :]
df = compressdf(df)
println("merged orders $df")
println(cnt)
println("$(EnvConfig.now()): TradeCorrelation finished")

end  # module