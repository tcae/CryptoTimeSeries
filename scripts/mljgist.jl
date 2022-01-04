
# include("../src/env_config.jl")
# include("../src/features.jl")
# include("../test/testohlcv.jl")
# include("../src/targets.jl")
# include("../src/ohlcv.jl")

# """
# abc
# """

using DataFrames, PlotlyJS, Dash

year=[i for i in 1995:2012]
y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263, 350, 430, 474, 526, 488, 537, 500, 439]

print(year)
print(y)
fig = PlotlyJS.plot(
    x=year, y=y,
    Layout(
        title_text="US Export of Plastic Scrap",
        legend=attr(x=0, y=1)
    )
)

app = dash()
app.layout = dcc_graph(figure=fig, style=Dict("height"=>300), id="my-graph")
run_server(app, "0.0.0.0", debug=true)


# using DataFrames, CSV, PlotlyJS, RDatasets
# using Dash, DashHtmlComponents, DashCoreComponents

# iris = dataset("datasets", "iris")  # dataset(DataFrame, "iris")
# fig = p1 = PlotlyJS.plot(
#     iris, x=:sepal_length, y=:sepal_width, color=:species,
#     mode="markers", marker_size=8
# )

# app = dash()

# app.layout = html_div() do
#     html_h4("Iris Sepal Length vs Sepal Width"),
#     dcc_graph(
#         id="example-graph-3",
#         figure=fig,
#     )
# end

# run_server(app, "0.0.0.0", debug=true)