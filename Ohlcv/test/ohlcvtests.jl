using EnvConfig, Ohlcv, Dates

EnvConfig.init(training)
verbosity = 2
Ohlcv.verbosity = 1

res = Ohlcv.liquidcoins()
println("$(length(res)) liquid coins")
if verbosity >= 2
    for ix in eachindex(res)
        println("$ix coin=$(res[ix].basecoin) #ranges=$(length(res[ix].ranges)) ranges: $(res[ix].ranges)")
        # for r in baseranges.ranges
        #     println("$(baseranges.basecoin): range = $r")
        # end
    end
end
println("liquiddailyminimumquotevolume=$(Ohlcv.liquiddailyminimumquotevolume())")

