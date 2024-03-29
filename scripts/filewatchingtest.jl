using BetterFileWatching, Dates, DataFrames, CSV

# BetterFileWatching.watch_file(joinpath(@__DIR__, "..", "tradeconfig.csv")) do event
#     @info "Something changed!" event
# end

# BetterFileWatching.watch_task = @async watch_file(joinpath(@__DIR__, "..", "tradeconfig.csv")) do event
#     @info "Something changed!" event
# end

# sleep(5)

# # stop watching the folder
# schedule(watch_task, InterruptException(); error=true)
# println("istaskstarted(watch_task)=$(istaskstarted(watch_task))")

fnp = joinpath(@__DIR__, "..", "tradeconfig.csv")
mt = mtime(fnp)
mdt = Dates.unix2datetime(mt)
println("mt($(typeof(mt)))=$mt mdt($(typeof(mdt)))=$mdt ")
df = CSV.read(fnp, DataFrame, stripwhitespace=true, comment="#")
println(df)
tgm = gain = nothing
p1 = Ref(Symbol(df[1, :parameter]))
p1 = parse(eval(Symbol(df[1, :type])), df[1, :value])
p2 = Ref(Symbol(df[3, :parameter]))
p2 = parse(eval(Symbol(df[3, :type])), df[3, :value])
println("tgm{$(typeof(tgm))}=$tgm gain{$(typeof(gain))}=$gain ")
