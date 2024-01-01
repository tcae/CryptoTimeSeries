using Dates
using EnvConfig
# using Plots
using Classify

EnvConfig.init(production)
EnvConfig.setlogpath("Relu_200epochs_nosoftmax_ignore_oversample_nostandardizer_nocv_losscrossentropy")


Classify.evaluateclassifier("NNcombi_24-01-01_02-00-38_gitSHA-f2a9841a71a2c7fd5286917068fbd3b58f873484.bson")
println("done")