
import Pkg: activate
cd("$(@__DIR__)/..")
println("activated $(pwd())")
activate(pwd())

include("../src/env_config.jl")
include("../src/assets.jl")

module AssetsTest
using Dates, DataFrames

using ..Config, ..Assets

Config.init(production)
ad1 = Assets.loadassets(dayssperiod=Dates.Year(4), minutesperiod=Dates.Week(4))

end  # module
