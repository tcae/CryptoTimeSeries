using EnvConfig, Ohlcv, Dates

EnvConfig.init(training)
Ohlcv.verbosity = 1

coinlist = Ohlcv.liquidcoins()
println(coinlist)


# for ohlcv in Ohlcv.OhlcvFiles()
#     otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
#     range = Ohlcv.liquidrange(ohlcv, 2*1000*1000f0, 1000f0)
#     if isnothing(range)
#         println("$(ohlcv.base) data: $(otime[begin])-$(otime[end])=$(Minute(otime[end]-otime[begin])) no time range with sufficient liquidity")
#     else
#         if range.endix == lastindex(otime)
#             if (range.endix - range.startix) > 2 * 10*24*60
#                 println("$(ohlcv.base) data: $(otime[begin])-$(otime[end])=$(Minute(otime[end]-otime[begin])) sufficient liquidity: $(otime[range.startix])-$(otime[range.endix])=$(Minute(otime[range.endix]-otime[range.startix]))  - current candidate - long enough for backtest")
#             else
#                 println("$(ohlcv.base) data: $(otime[begin])-$(otime[end])=$(Minute(otime[end]-otime[begin])) sufficient liquidity: $(otime[range.startix])-$(otime[range.endix])=$(Minute(otime[range.endix]-otime[range.startix]))  - current candidate but too short")
#             end
#         end
#     end
# end

# Ohlcv.verbosity = 3
# ohlcv = Ohlcv.defaultohlcv("BTC")
# ohlcv = Ohlcv.read!(ohlcv)
# otime = Ohlcv.dataframe(ohlcv)[!, :opentime]
# range = Ohlcv.liquidrange(ohlcv, 2*1000*1000f0, 1000f0)
# println("$(ohlcv.base) data: $(otime[begin])-$(otime[end])=$(Minute(otime[end]-otime[begin])) sufficient liquidity: $(otime[range.startix])-$(otime[range.endix])=$(Minute(otime[range.endix]-otime[range.startix]))")

