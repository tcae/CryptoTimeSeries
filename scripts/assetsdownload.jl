# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())

module AssetsTest
using Dates

using EnvConfig, Assets

EnvConfig.init(production)
# EnvConfig.init(test)
ad1 = Assets.loadassets()

end  # module
