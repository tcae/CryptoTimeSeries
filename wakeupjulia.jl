using Dates
using EnvConfig

dt = DateTime[]
println("$(Dates.now()) wake up julia $VERSION")
println(EnvConfig.checkfolders(false, false))

testx() = println("tstx: $(string(@__FUNCTION__))")
testx()
