cd("$(@__DIR__)/..")
println("activated $(pwd())")
activate(pwd())
cd(@__DIR__)
using Dash, Dates
import Dash: dash, callback!, run_server, Output, Input, State, callback_context
import Dash: dcc_graph, html_h1, html_div, dcc_checklist, html_button, dcc_dropdown, dash_datatable
import PlotlyJS: PlotlyBase, Plot, dataset, Layout, attr, scatter, candlestick, bar, heatmap


app =
    dash(external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"])

function getz()
    # z = [rand(3) for _ in 1:3]
    z = []
    for _ in 1: 3
        z = push!(z, rand(4))
    end
    t = [["t $i2 $i1 $(z[i1][i2])" for i2 in 1:size(z[1], 1)] for i1 in 1:size(z, 1)]
    println(z)
    println(t)
    return z, t
end


z, t = getz()
x = [d for d in DateTime(2022, 01, 12, 12, 00):Minute(1):DateTime(2022, 01, 12, 12, 04)]
println("$x $(size(x)) $(typeof(x))")
app.layout = html_div() do
    html_h1("Hello Dash"),
    html_div("Dash.jl: Julia interface for Dash"),
    dcc_dropdown(
        id="crypto_focus",
        options=[(label = i, value = i) for i in ["a", "b"]],
        value="a"
    ),
    html_div(id="targets4h"),
    dcc_graph(
        id = "example-graph",
        figure = (
            data = [
                (x=[d for d in x], z = z, text=t, type = "heatmap", name = "SF1"),
            ],
            layout = (title = "Dash Data Visualization",),
        ),
    )
end

callback!(
    app,
    Output("targets4h", "children"),
    Input("crypto_focus", "value")
    # prevent_initial_call=true
) do focus
    fig = nothing
    fig = Plot(
        # [heatmap(x=x, y=y, z=[z], text=t)],
        [heatmap(x=[d for d in x], z = z, text=t, name = "SF2")],
        Layout(xaxis_rangeslider_visible=false)
        )
    return dcc_graph(figure=fig)
end

run_server(app, "127.0.0.1", 8050, debug = true)
