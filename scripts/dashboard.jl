
using DrWatson
@quickactivate "CryptoTimeSeries"

# import Pkg; Pkg.add(["Dash", "DashCoreComponents", "DashHtmlComponents", "DashTable"])
# Pkg.status("Dash")
# Pkg.status("DashHtmlComponents")
# Pkg.status("DashCoreComponents")

include("../src/env_config.jl")

# include(srcdir("classify.jl"))

using Dash, DashHtmlComponents, DashCoreComponents
envbases = ["btc", "xrp", "eth"]

app = dash(external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])

app.layout = html_div(style = (clear = "both",)) do
    html_div([
        dcc_checklist(
            id = "crypto_select",
            options = [(label=i, value=i) for i in envbases],
            labelStyle = (:display => "inline-block")
        )
    ]),
    html_div([
        html_button("all", name="all", id="all_button"),
        html_button("none", id="none_button"),
        html_button("update data", id="update_data"),
        html_button("reset selection", id="reset_selection")
    ]),
    dcc_graph(id = "basic-interactions", figure = (
        data = [
            (
                x = [1,2,3,4],
                y = [4,1,3,5],
                text = ["a", "b", "c", "d"],
                customdata = ["c.a", "c.b", "c.c", "c.d"],
                name = "Trace 1",
                mode = "markers",
                marker = (size = 12,)
            ),
            (
                x = [1,2,3,4],
                y = [9,4,1,4],
                text = ["w", "x", "y", "z"],
                customdata = ["c.w", "c.x", "c.y", "c.z"],
                name = "Trace 2",
                mode = "markers",
                marker = (size = 12,)
            )
        ],
        layout = (clickmode = "event+select",)
    )),
    dcc_graph(id = "basic-interactions2", figure = (
        data = [
            (
                x = [1,2,3,4],
                y = [4,1,3,5],
                text = ["a", "b", "c", "d"],
                customdata = ["c.a", "c.b", "c.c", "c.d"],
                name = "Trace 1",
                mode = "markers",
                marker = (size = 12,)
            ),
            (
                x = [1,2,3,4],
                y = [9,4,1,4],
                text = ["w", "x", "y", "z"],
                customdata = ["c.w", "c.x", "c.y", "c.z"],
                name = "Trace 2",
                mode = "markers",
                marker = (size = 12,)
            )
        ],
        layout = (clickmode = "event+select",)
    ))
    # dcc_graph(id="graph1day"),
    # dcc_graph(id="graph10day"),
    # dcc_graph(id="graph6month"),
    # dcc_graph(id="graph_all"),
end

callback!(
    app,
    Output("hover-data", "children"),
    Input("basic-interactions", "hoverData"),
) do hover_data

    return JSON3.write(hover_data)
end


#=
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Checklist(
                id="crypto_select",
                options=[{"label": i, "value": i} for i in Env.bases],
                labelStyle={"display": "flow-root"}
            ),
            html.Button("all", id="all_button", style={"display": "flow-root"}),
            html.Button("none", id="none_button", style={"display": "flow-root"}),
            html.Button("update data", id="update_data", style={"display": "flow-root"}),
            html.Button("reset selection", id="reset_selection", style={"display": "flow-root"}),
        ], style={"display": "flow-root"}),
        # html.H1("Crypto Price"),  # style={"textAlign": "center"},
        dcc.Graph(id="graph1day"),
        dcc.Graph(id="graph10day"),
        dcc.Graph(id="graph6month"),
        dcc.Graph(id="graph_all"),
        html.Div(id="focus", style={"display": "none"}),
        html.Div(id="graph1day_end", style={"display": "none"}),
        html.Div(id="graph10day_end", style={"display": "none"}),
        html.Div(id="graph6month_end", style={"display": "none"}),
        html.Div(className="row", children=[
            html.Div([
                dcc.Markdown(dedent("""
                    **Hover Data**

                    Mouse over values in the graph.
                """)),
                html.Pre(id="hover-data", style=styles["pre"])
            ], className="three columns"),

            html.Div([
                dcc.Markdown(dedent("""
                    **Click Data**

                    Click on points in the graph.
                """)),
                html.Pre(id="click-data", style=styles["pre"]),
            ], className="three columns"),

            html.Div([
                dcc.Markdown(dedent("""
                    **Selection Data**

                    Choose the lasso or rectangle tool in the graph's menu
                    bar and then select points in the graph.

                    Note that if `layout.clickmode = 'event+select'`, selection data also
                    accumulates (or un-accumulates) selected data if you hold down the shift
                    button while clicking.
                """)),
                html.Pre(id="selected-data", style=styles["pre"]),
            ], className="three columns"),

            html.Div([
                dcc.Markdown(dedent("""
                    **Zoom and Relayout Data**

                    Click and drag on the graph to zoom or click on the zoom
                    buttons in the graph's menu bar.
                    Clicking on legend items will also fire
                    this event.
                """)),
                html.Pre(id="relayout-data", style=styles["pre"]),
            ], className="three columns")
        ]),
    ], style={
        "borderBottom": "thin lightgrey solid",
        "backgroundColor": "rgb(250, 250, 250)",
        "padding": "10px 5px",
        "width": "49%",
        "display": "flow-root"
    }),

    html.Div([
        dcc.RadioItems(
            id="crypto_radio",
            options=[{"label": i, "value": i} for i in Env.bases],
            labelStyle={"display": "flow-root"}
        ),
        dcc.Checklist(
            id="indicator_select",
            options=[{"label": i, "value": i} for i in indicator_opts],
            value=["regression 1D"],
            labelStyle={"display": "flow-root"}
        ),
        dcc.Graph(id="graph4h"),
        # dcc.Graph(id="volume-signals-graph"),
        html.Div(id="graph4h_end", style={"display": "none"}),
        dash_table.DataTable(
            id="kpi_table", editable=False,  # hidden_columns=["id"],
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold"
            }
        )
        # html.Div(id="kpi_table_wrapper")
    ], style={"display": "flow-root", "float": "right", "width": "49%"}),

])
=#
run_server(app, "0.0.0.0", debug=true)

module Dashboard
using Dates, DataFrames
using Test

using ..Config


end  # module