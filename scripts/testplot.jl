using Pkg
cd("/home/tor/TorProjects/CryptoTimeSeries")
Pkg.activate(".")
# cd("/home/tor/TorProjects/CryptoTimeSeries/notebooks")

# include(pwd() * "/" * "../src/features.jl")
include("../src/features.jl")

using PlotlyJS, WebIO
using ..Features

normpercent(ydata, ynormref) = (ydata ./ ynormref .- 1) .* 100

iny = [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
normref = iny[end]
regrwindow = size(iny, 1)

innormy = normpercent(iny, iny[end])
regry, grad = Features.rollingregression(iny, regrwindow)
println("regression y: $regry, size(y,1): $(size(regry,1)), y[end]: $(regry[end])")
println("grad: $grad")
intercept = regry[end] - size(regry,1) * grad[end]
println("intercept: $intercept")

regrliney = [regry[end] - grad[end] * (regrwindow - 1), regry[end]]
regrlinenormy = normpercent(regrliney, normref)
# lastregry = normpercent([regry[end]], normref)[1]
# lastgrad = normpercent([grad[end]], normref)[1]
# firstregry = lastregry - lastgrad * (regrwindow - 1)


traces = [
    scatter(;x=1:size(iny, 1), y=iny, mode="markers", name="input y")
    # scatter(;x=1:size(regry, 1), y=regry, mode="lines+markers", name="regression y")
    scatter(;x=1:size(innormy, 1), y=innormy, mode="markers", name="norm input y")
    # scatter(;x=[1,size(regry, 1)], y=[firstregry, lastregry], mode="lines+markers", name="regression y")
    scatter(;x=[1,size(regry, 1)], y=regrliney, mode="lines+markers", name="regression y")
    scatter(;x=[1,size(regry, 1)], y=regrlinenormy, mode="lines+markers", name="norm regression y")

]

std, mean, normy = Features.rollingregressionstdxt(iny, regry, grad, regrwindow)
traceadd = [scatter(;x=1:x1, y=normy[x1, 1:x1], mode="lines+markers", name="notrend y") for x1 in 1:regrwindow]
append!(traces, traceadd)
traceadd = [
    scatter(;x=1:regrwindow, y=std[1:regrwindow], mode="lines+markers", name="std notrend y")
    scatter(;x=1:regrwindow, y=mean[1:regrwindow], mode="lines+markers", name="mean notrend y")
]
append!(traces, traceadd)
plot(traces)
