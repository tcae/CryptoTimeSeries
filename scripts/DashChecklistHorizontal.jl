
# using DrWatson
# @quickactivate "CryptoTimeSeries"

# import Pkg; Pkg.add(["Dash", "DashCoreComponents", "DashHtmlComponents", "DashTable"])
# Pkg.status("Dash")
# Pkg.status("DashHtmlComponents")
# Pkg.status("DashCoreComponents")

# include("../src/env_config.jl")

# include(srcdir("classify.jl"))

using Dash, DashHtmlComponents, DashCoreComponents


app = dash()

app.layout = dcc_checklist(
    options=[
        (label= "New York City", value= "NYC"),
        (label= "Montréal", value= "MTL"),
        (label= "San Francisco", value= "SF")
    ],
    value=["MTL", "SF"],
    labelStyle=(:display => "inline_block")

)
# ], style = (display = "inline-block")),


# dcc.Checklist(
#     options=[
#         {"label": "New York City", "value": "NYC"},
#         {"label": "Montréal", "value": "MTL"},
#         {"label": "San Francisco", "value": "SF"}
#     ],
#     value=["MTL", "SF"],
#     label_style={"display": "inline-block"}
# )


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

run_server(app, "0.0.0.0", debug=true)
