
# using DrWatson
# @quickactivate "CryptoTimeSeries"

# import Pkg; Pkg.add(["Dash", "DashCoreComponents", "DashHtmlComponents", "DashTable"])
# Pkg.status("Dash")
# Pkg.status("DashHtmlComponents")
# Pkg.status("DashCoreComponents")

# include("../src/env_config.jl")

# include(srcdir("classify.jl"))

using Dash, DashHtmlComponents, DashCoreComponents, DashTable

# app = dash(external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"])
# app = dash(external_stylesheets = ["/home/tor/TorProjects/CryptoTimeSeries/scripts/dashboard.css"])
app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")
# app = dash()

# app.layout = dcc_checklist(
#     options=[
#         (label= "New York City", value= "NYC"),
#         (label= "Montréal", value= "MTL"),
#         (label= "San Francisco", value= "SF")
#     ],
#     value=["MTL", "SF"],
#     labelStyle=(:display => "inline_block")

# )

# ], style = (display = "inline-block")),


# html_div() do
#     html_h1("Hello Dash"),
#     html_div("Dash: A web application framework for Julia"),
#     dcc_graph(
#         id = "example-graph-1",
#         figure = (
#             data = [
#                 (x = ["giraffes", "orangutans", "monkeys"], y = [20, 14, 23], type = "bar", name = "SF"),
#                 (x = ["giraffes", "orangutans", "monkeys"], y = [12, 18, 29], type = "bar", name = "Montreal"),
#             ],
#             layout = (title = "Dash Data Visualization", barmode="group")
#         )
#     )
# end
env_bases = ["BTC", "ETH"]
indicator_opts = ["opt a", "opt b"]
app.layout = html_div() do
    html_div([
        html_div([
            dcc_checklist(
                id="crypto_select",
                options=[(label = i, value = i) for i in env_bases],
                labelStyle=(:display => "inline-block")
            ),
            html_button("all", id="all_button"),
            html_button("none", id="none_button"),
            html_button("update data", id="update_data"),
            html_button("reset selection", id="reset_selection"),
        ]),
        # html_h1("Crypto Price"),  # style={"textAlign": "center"},
        dcc_graph(id="graph1day"),
        dcc_graph(id="graph10day"),
        dcc_graph(id="graph6month"),
        dcc_graph(id="graph_all"),
        html_div(id="focus"),
        html_div(id="graph1day_end"),
        html_div(id="graph10day_end"),
        html_div(id="graph6month_end"),
    ], style=(
        borderBottom = "thin lightgrey solid",
        backgroundColor = "rgb(250, 250, 250)",
        padding = "10px 5px",
        width = "49%",
        display = "inline-block"
    )),

    html_div([
        dcc_radioitems(
            id="crypto_radio",
            options=[(label = i, value = i) for i in env_bases],
            labelStyle=(:display => "inline-block")
        ),
        dcc_checklist(
            id="indicator_select",
            options=[(label = i, value = i) for i in indicator_opts],
            value=["regression 1D"],
            labelStyle=(:display => "inline-block")
        ),
        dcc_graph(id="graph4h"),
        # dcc_graph(id="volume-signals-graph"),
        html_div(id="graph4h_end"),
        dash_datatable(
            id="kpi_table", editable=false,  # hidden_columns=["id"],
            style_header=(
                backgroundColor = "rgb(230, 230, 230)",
                fontWeight = "bold"
            )
        )
        # html_div(id="kpi_table_wrapper")
    ], style=(float = "right", width = "49%"))
end


run_server(app, "0.0.0.0", debug=true)
