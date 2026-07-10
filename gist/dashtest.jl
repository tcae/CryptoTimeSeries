import Dash: dcc_graph, html_h1, html_div, dash
import PlotlyJS: Plot, dataset

using CSV, DataFrames  # automatically loads these integrations

df = dataset(DataFrame, "tips")
print(df)

p2 = Plot(df, y=:tip, facet_row=:sex, facet_col=:smoker, color=:day, kind="violin")
# p2 = Plot(rand(10, 4))

app = dash()

app.layout = html_div() do
    html_h1("Hello Dash"),
    html_div("Dash: A web application framework for your data."),
    dcc_graph(
        id = "example-graph-1",
        figure = p2
    )
end

run_server(app, "0.0.0.0", debug=true)