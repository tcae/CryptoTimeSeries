using DrWatson
@quickactivate "CryptoTimeSeries"
using Pkg
Pkg.add(["LoggingFacilities", "NamedArrays"])

include("../test/testohlcv.jl")
# include("../src/targets.jl")
include("../src/ohlcv.jl")

# using DrWatson
# @quickactivate "CryptoTimeSeries"

# import Pkg; Pkg.add(["Dash", "DashCoreComponents", "DashHtmlComponents", "DashTable"])
# Pkg.status("Dash")
# Pkg.status("DashHtmlComponents")
# Pkg.status("DashCoreComponents")

# include("../src/env_config.jl")

# include(srcdir("classify.jl"))

using Dash, DashHtmlComponents, DashCoreComponents, DashTable
using Dates, DataFrames
using ..Config
using ..Ohlcv

app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")

env_bases = ["BTC", "ETH"]
env_bases = ["Test"]
indicator_opts = ["opt a", "opt b"]
app.layout = html_div() do
    html_div(id="leftside", [
        dcc_checklist(
            id="crypto_select",
            options=[(label = i, value = i) for i in env_bases],
            value=[env_bases[0]],
            labelStyle=(:display => "inline-block")
        ),
        html_div(id="select_buttons", [
            html_button("all", id="all_button"),
            html_button("none", id="none_button"),
            html_button("update data", id="update_data"),
            html_button("reset selection", id="reset_selection")
        ]),
        # html_h1("Crypto Price"),  # style={"textAlign": "center"},
        dcc_graph(id="graph1day"),
        dcc_graph(id="graph10day"),
        dcc_graph(id="graph6month"),
        dcc_graph(id="graph_all"),
        html_div(id="focus"),
        html_div(id="graph1day_end"),
        html_div(id="graph10day_end"),
        html_div(id="graph6month_end")
    ]),

    html_div(id="rightside", [
        dcc_radioitems(
            id="crypto_radio",
            options=[(label = i, value = i) for i in env_bases],
            labelStyle=(:display => "inline-block")
        ),
        dcc_checklist(
            id="indicator_select",
            options=[(label = i, value = i) for i in indicator_opts],
            value=["opt a"],
            labelStyle=(:display => "inline-block")
        ),
        dcc_graph(id="graph4h"),
        dcc_graph(id="volume-signals-graph"),
        html_div(id="graph4h_end"),
        dash_datatable(id="kpi_table", editable=false)
    ])
end


run_server(app, "0.0.0.0", debug=true)
ohlcv = TestOhlcv.sinedata(120, 3)
