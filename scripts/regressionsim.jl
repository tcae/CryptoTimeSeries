include("../src/features.jl")

using IJulia, PlotlyJS, WebIO
using ..Features

# ! TCAE: doesn't work in vscode --> use a IJulia notebook instead

y,grad = Features.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)
println("y: $y, size(y,1): $(size(y,1)), y[end]: $(y[end])")
println("grad: $grad")
intercept = y[end] - size(y,1) * grad[end]
println("intercept: $intercept")

# trace1 = scatter(;x=1:size(y, 1), y=y, mode="lines+markers")
# plot([trace1])

# p = PlotlyJS.plot(rand(10, 4));
p = PlotlyJS.plot(rand(10, 4), options=Dict(:staticPlot => true))
display(p)  # usually optional

IJulia.notebook()
on(p["hover"]) do data
    println("\nYou hovered over", data)
end


# using IJulia, PlotlyJS, WebIO


# notebook()
# p = PlotlyJS.plot(rand(10, 4), options=Dict(:staticPlot => true))
# display(p)  # usually optional