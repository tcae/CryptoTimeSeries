include("../test/testohlcv.jl")
# include("../src/targets.jl")
include("../src/ohlcv.jl")
include("../src/assets.jl")


# include("../src/env_config.jl")

# include(srcdir("classify.jl"))

import Dash: dash, callback!, run_server, Output, Input, State, callback_context
import Dash: dcc_graph, html_h1, html_div, dcc_checklist, html_button, dcc_radioitems, dash_datatable
import PlotlyJS: Plot, dataset, Layout, attr, scatter, candlestick, bar

# using Dash, DashTable, PlotlyJS
using Dates, DataFrames
using ..Config, ..Ohlcv, ..Assets

# app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")
app = dash(external_stylesheets = ["dashboard.css"], assets_folder=(pwd() * "/scripts/"))

env_bases = ["BTC", "ETH"]
# env_bases = ["Test"]
indicator_opts = ["opt a", "opt b"]
ohlcv = TestOhlcv.sinedata(120, 3)
Ohlcv.addpivot!(ohlcv.df)
assets = Assets.read()

# println(first(ohlcv.df, 3))
# fig = Plot(
#     ohlcv.df, x=:opentime, y=:pivot,
# )

# assets =

trace1 = scatter(x=ohlcv.df.opentime, y=ohlcv.df.high, mode="lines", name="lines")
trace2 = scatter(x=ohlcv.df.opentime, y=ohlcv.df.low, mode="lines+markers", name="lines+markers")
trace3 = scatter(x=ohlcv.df.opentime, y=ohlcv.df.pivot, mode="markers", name="markers")
fig1 = Plot([trace1, trace2, trace3])

fig2 = Plot([candlestick(
        x=ohlcv.df.opentime,
        open=ohlcv.df.open,
        high=ohlcv.df.high,
        low=ohlcv.df.low,
        close=ohlcv.df.close, name="OHLC"
    ),
    bar(x=ohlcv.df.opentime, y=ohlcv.df.basevolume, name="basevolume", yaxis="y2")
    ],
    Layout(title_text="4hOHLCV", xaxis_title_text="time", yaxis_title_text="OHLCV%",
        yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.2],
        yaxis_domain=[0.3, 1.0])
    )


app.layout = html_div() do
    html_div(id="leftside", [
        dcc_checklist(
            id="crypto_select",
            options=[(label = i, value = i) for i in env_bases],
            value=[env_bases[1]],
            labelStyle=(:display => "inline-block")
        ),
        html_div(id="select_buttons", [
            html_button("all", id="all_button"),
            html_button("none", id="none_button"),
            html_button("update data", id="update_data"),
            html_button("reset selection", id="reset_selection")
        ]),
        # html_h1("Crypto Price"),  # style={"textAlign": "center"},
        html_div(id="myoutput1"),
        html_div(id="myoutput2"),
        dcc_graph(id="graph1day", figure=fig1),
        dcc_graph(id="graph10day"),
        dcc_graph(id="graph6month"),
        dcc_graph(id="graph_all"),
        html_div(id="focus"),
        html_div(id="div1day"),
        html_div(id="graph10day_end"),
        html_div(id="graph6month_end")
    ]),

    html_div(id="rightside", [
        dcc_radioitems(
            id="crypto_radio",
            options=[(label = i, value = i) for i in env_bases],
            value=env_bases[1],
            labelStyle=(:display => "inline-block")
        ),
        dcc_checklist(
            id="indicator_select",
            options=[(label = i, value = i) for i in indicator_opts],
            value=["opt a"],
            labelStyle=(:display => "inline-block")
        ),
        dcc_graph(id="graph4h", figure=fig2),
        dcc_graph(id="volume-signals-graph"),
        html_div(id="graph4h_end"),
        dash_datatable(id="kpi_table", editable=false,
            columns=[Dict("name" =>i, "id" => i) for i in names(assets.df)], data = Dict.(pairs.(eachrow(assets.df))),
            style_table=Dict("height" => "500px", "overflowY" => "auto"))
    ])
end

callback!(
    app,
    Output("crypto_select", "value"),
    Input("all_button", "n_clicks"),
    Input("none_button", "n_clicks"),
    State("crypto_select", "value")
    # prevent_initial_call=true
) do all, none, select
    s = "all = $all, none = $none, select = $select"
    println(s)
    # return s
    ctx = callback_context()
    if length(ctx.triggered) > 0
        button_id = split(ctx.triggered[1].prop_id, ".")[1]
        if button_id == "all_button"
            return env_bases
        else
            return [""]
        end
    else
        return select
    end
end

callback!(
    app,
    Output("myoutput2", "children"),
    Input("crypto_select", "value"),
    Input("crypto_radio", "value")
    # prevent_initial_call=true
) do select, radio
    s = "select = $select, radio = $radio"
    println(s)
    return s
end

run_server(app, "0.0.0.0", debug=true)
