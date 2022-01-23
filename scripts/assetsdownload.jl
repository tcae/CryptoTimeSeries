import Pkg: activate
cd("$(@__DIR__)/..")
println("activated $(pwd())")
activate(pwd())

include("../src/assets.jl")

module AssetsTest
using Dates

using ..EnvConfig, ..Assets

EnvConfig.init(production)
if EnvConfig.configmode == production
    ad1 = Assets.loadassets(dayssperiod=Dates.Year(4), minutesperiod=Dates.Week(4))
else
    println("no operation for config mode $(EnvConfig.configmode)")
end

end  # module
