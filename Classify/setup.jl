# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
Pkg.activate(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
# Pkg.upgrade_manifest()
# Pkg.add(url="https://github.com/tlienart/OpenSpecFun_jll")  # fix for MKL issue in Scikit-learn - see MLJ manual
# Pkg.add(url="add https://github.com/diegozea/ROC.jl")
Pkg.add([
    "Test",
    "JSON3",
    "DataFrames",
    "CSV",
    "CategoricalArrays",
    "Logging",
    "Statistics",
    # "MLJ", # "PartialLeastSquaresRegressor", "ScikitLearn", "MLJFlux",
    "Flux", "cuDNN", "StatisticalMeasures", "JLSO", "ProgressMeter",
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
