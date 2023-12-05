# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
Pkg.activate(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
# Pkg.upgrade_manifest()
Pkg.add([
    "Test",
    "JSON3",
    "DataFrames",
    "CSV",
    "CategoricalArrays", "CategoricalDistributions",
    "Logging", "ProfileView",
    "Statistics",
    # "MLJ", # "PartialLeastSquaresRegressor", "ScikitLearn", "MLJFlux",
    "Flux", "cuDNN", "StatisticalMeasures", "BSON", "ProgressMeter", "MLUtils",
    # "MLJGLMInterface", "GLM",
    # "MLJLinearModels", "MLJDecisionTreeInterface", # "MLJScikitLearnInterface",
    "Combinatorics", # MLJtest -
    "PrettyPrinting",
    "IJulia", "Plots", "PlotlyJS", "Colors"  # "WebIO", "Dash",   Regressionsim, CryptoCockpit, Notebooks
    ])

# Pkg.develop(path="/home/tor/TorProjects/CryptoTimeSeries")

# Pkg.resolve()
# Pkg.update()
Pkg.gc()
