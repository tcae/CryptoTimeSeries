# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())

module AssetsTest
using Dates

using EnvConfig, Assets

EnvConfig.init(production)
if EnvConfig.configmode == production
    ad1 = Assets.loadassets()
else
    println("no operation for config mode $(EnvConfig.configmode)")
end

end  # module
