#=
to do list with prio in order of appearance:

- introduce regression lines
- introduce clickable graphs to zoom in with 4h candlestick
- format table to highlight quantities
- remove crypto select buttons by table row selection and apply all/none bottons to selected table rows
- remove crypto radio buttons by table cell click

=#

include("../test/testohlcv.jl")
# include("../src/targets.jl")
include("../src/ohlcv.jl")
include("../src/features.jl")
include("../src/assets.jl")


# include("../src/env_config.jl")

# include(srcdir("classify.jl"))

import Dash: dash, callback!, run_server, Output, Input, State, callback_context
import Dash: dcc_graph, html_h1, html_div, dcc_checklist, html_button, dcc_radioitems, dash_datatable
import PlotlyJS: Plot, dataset, Layout, attr, scatter, candlestick, bar

# using Dash, DashTable, PlotlyJS
using Dates, DataFrames
using ..Config, ..Ohlcv, ..Features, ..Assets

# app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")
app = dash(external_stylesheets = ["dashboard.css"], assets_folder=(pwd() * "/scripts/"))

indicator_opts = ["regression 1d", "targets", "signals", "features", "equal scale"]  # equal scale = vertical scale
# ohlcv = TestOhlcv.sinedata(120, 3)
# Ohlcv.addpivot!(ohlcv.df)
assets = Assets.read()
env_bases = assets.df[!, :base]
assets.df.id = assets.df[!, :base]
# println(first(assets.df, 2))
focusbase = "btc"
focus1m = Ohlcv.defaultohlcv(focusbase)
Ohlcv.read!(focus1m)
focus1d = Ohlcv.defaultohlcv(focusbase)
Ohlcv.setinterval!(focus1d, "1d")
Ohlcv.read!(focus1d)
ohlcvcache = Dict()

function updatecache(selectbases, radiobase)
    # println("update start: $([k for k in keys(ohlcvcache)]) select: $selectbases radio: $radiobase")
    bases = union(Set(selectbases), Set([radiobase]))
    loadedbases = Set(keys(ohlcvcache))
    # println("update to be removed: $([k for k in setdiff(loadedbases, bases)])")
    for base in setdiff(loadedbases, bases)  # to be removed bases
        delete!(ohlcvcache, base)
    end
    # println("update to be added: $([k for k in setdiff(bases, loadedbases)])")
    for base in setdiff(bases, loadedbases)  # to be added and loaded bases
        base1m = Ohlcv.defaultohlcv(base)
        Ohlcv.read!(base1m)
        Ohlcv.addpivot!(base1m)

        base1d = Ohlcv.defaultohlcv(base)
        Ohlcv.setinterval!(base1d, "1d")
        Ohlcv.read!(base1d)
        Ohlcv.addpivot!(base1d)
        ohlcvcache[base] = Dict("1m" => base1m, "1d" => base1d)
    end
    # println("update end: $([k for k in keys(ohlcvcache)])")
    return ohlcvcache
end

# trace1 = scatter(x=ohlcv.df.opentime, y=ohlcv.df.high, mode="lines", name="lines")
# trace2 = scatter(x=ohlcv.df.opentime, y=ohlcv.df.low, mode="lines+markers", name="lines+markers")
# trace3 = scatter(x=ohlcv.df.opentime, y=ohlcv.df.pivot, mode="markers", name="markers")
# fig1 = Plot([trace1, trace2, trace3])

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
        dcc_graph(id="graph1day"),
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
            value=["regression 1d"],
            labelStyle=(:display => "inline-block")
        ),
        dcc_graph(id="graph4h"),
        # dcc_graph(id="volume-signals-graph"),
        dash_datatable(id="kpi_table", editable=false,
            columns=[
                Dict("name" =>i, "id" => i) for i in names(assets.df) if i != "id"],  # exclude "id" to not display it
                data = Dict.(pairs.(eachrow(assets.df))),
            filter_action="native",
            row_selectable="multi",
            sort_action="native",
            sort_mode="multi",
            style_table=Dict("height" => "700px", "overflowY" => "auto")),
        html_div(id="tbl_action")
        ])
end

normpercent(ydata, ynormref) = (ydata ./ ynormref .- 1) .* 100

function linegraph(oc, interval, period)
    traces = nothing
    for base in keys(oc)
        df = oc[base][interval].df
        enddt = df[end, :opentime]
        startdt = enddt - period
        df = df[startdt .< df.opentime .<= enddt, :]
        normref = df[end, :pivot]
        if traces === nothing
            traces = [scatter(x=df.opentime, y=normpercent(df[!, :pivot], normref), mode="lines", name=base)]
        else
            append!(traces, [scatter(x=df.opentime, y=normpercent(df[!, :pivot], normref), mode="lines", name=base)])
        end

        # calc regression after df is reduced to relevant timerange
        regrwindow = size(df[!, :pivot], 1)
        regry, grad = Features.rollingregression(df[!, :pivot], regrwindow)
        lastregry = normpercent([regry[end]], normref)[1]
        lastgrad = normpercent([grad[end]], normref)[1]
        firstregry = lastregry - lastgrad * (regrwindow - 1)
        append!(traces, [scatter(x=[startdt, enddt], y=[firstregry, lastregry], mode="lines", name="$base $regrwindow regr")])
    end
    return traces
end

callback!(
    app,
    # Output("graph1day", "children"),
    Output("graph1day", "figure"),
    Output("graph10day", "figure"),
    Output("graph6month", "figure"),
    Output("graph_all", "figure"),
    Input("crypto_select", "value"),
    Input("crypto_radio", "value")
    # prevent_initial_call=true
) do select, radio
    ctx = callback_context()
    button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
    s = "select = $select, radio = $radio, button ID: $button_id"
    println(s)
    ohlcvcache = updatecache(select, radio)
    fig1d = Plot(linegraph(ohlcvcache, "1m", Dates.Hour(24)))
    fig10d = Plot(linegraph(ohlcvcache, "1m", Dates.Day(10)))
    fig6M = Plot(linegraph(ohlcvcache, "1d", Dates.Month(6)))
    figall = Plot(linegraph(ohlcvcache, "1d", Dates.Year(3)))

    return fig1d, fig10d, fig6M, figall
end

callback!(
    app,
    Output("crypto_select", "value"),
    Input("all_button", "n_clicks"),
    Input("none_button", "n_clicks"),
    State("crypto_select", "value")
    # prevent_initial_call=true
) do all, none, select
    ctx = callback_context()
    button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
    s = "all = $all, none = $none, select = $select, button ID: $button_id"
    println(s)
    res = Dict("all_button" => env_bases, "none_button" => [], "" => select)
    println("returning: $(res[button_id])")
    return res[button_id]
end

function candlestickgraph(ohlcv, period)
    df = ohlcv.df
    enddt = df[end, :opentime]
    startdt = enddt - period
    df = df[startdt .< df.opentime .<= enddt, :]

    normref = df[end, :close]
    fig = Plot([candlestick(
            x=df[!, :opentime],
            open=normpercent(df[!, :open], normref),
            high=normpercent(df[!, :high], normref),
            low=normpercent(df[!, :low], normref),
            close=normpercent(df[!, :close], normref),
            name="OHLC"
        ),
        bar(x=df.opentime, y=df.basevolume, name="basevolume", yaxis="y2")
        ],
        Layout(title_text="4hOHLCV", xaxis_title_text="time", yaxis_title_text="OHLCV%",
            yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.2],
            yaxis_domain=[0.3, 1.0])
        )
    return fig
end

callback!(
    app,
    # Output("graph1day", "children"),
    Output("graph4h", "figure"),
    Input("crypto_radio", "value"),
    State("crypto_select", "value")
    # prevent_initial_call=true
) do radio, select
    ctx = callback_context()
    button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
    s = "select = $select, radio = $radio, button ID: $button_id"
    println(s)
    ohlcvcache = updatecache(select, radio)
    fig4h = Plot(candlestickgraph(ohlcvcache[radio]["1m"], Dates.Hour(4)))

    return fig4h
end

callback!(app,
    Output("tbl_action", "children"),
    Input("kpi_table", "active_cell")
) do active_cell
  return active_cell === nothing ? "table click actions" : string(active_cell)
end


# callback!(
#     app,
#     Output("myoutput2", "children"),
#     Input("crypto_select", "value"),
#     Input("crypto_radio", "value")
#     # prevent_initial_call=true
# ) do select, radio
#     s = "select = $select, radio = $radio"
#     println(s)
#     return s
# end

run_server(app, "0.0.0.0", debug=true)
