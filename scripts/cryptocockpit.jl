include("../test/testohlcv.jl")
# include("../src/targets.jl")
include("../src/ohlcv.jl")
include("../src/features.jl")
include("../src/assets.jl")


# include("../src/env_config.jl")

# include(srcdir("classify.jl"))

import Dash: dash, callback!, run_server, Output, Input, State, callback_context
import Dash: dcc_graph, html_h1, html_div, dcc_checklist, html_button, dcc_dropdown, dash_datatable
import PlotlyJS: Plot, dataset, Layout, attr, scatter, candlestick, bar

# using Dash, DashTable, PlotlyJS
using Dates, DataFrames, JSON, JSON3
using ..Config, ..Ohlcv, ..Features, ..Assets

Config.init(Config.production)

# app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")
cssdir = Config.setprojectdir()  * "/scripts/"
# cssdir = pwd() * "/scripts/"
println("css dir: $cssdir")
app = dash(external_stylesheets = ["dashboard.css"], assets_folder=cssdir)

function updateassets!(assets, download=false)
    a = assets
    if download
        a = Assets.loadassets(dayssperiod=Dates.Year(4), minutesperiod=Dates.Week(4))
    else
        a = Assets.read()
    end
    if !(a === nothing)
        sort!(a.df, [:portfolio], rev=true)
        a.df.id = a.df.base
        println("updating table data")
        assets = a
    end
    return assets
end

# ohlcv = TestOhlcv.sinedata(120, 3)
# Ohlcv.addpivot!(ohlcv.df)
assets = updateassets!(nothing, false)
# println(first(assets.df, 2))
ohlcvcache = Dict()

function loadohlcv(base, interval)
    global ohlcvcache
    # println("loading $base interval: $interval")
    k = base * interval
    if !(k in keys(ohlcvcache))
        ohlcv = Ohlcv.defaultohlcv(base)
        Ohlcv.setinterval!(ohlcv, interval)
        Ohlcv.read!(ohlcv)
        Ohlcv.addpivot!(ohlcv)
        ohlcvcache[k] = ohlcv.df
    end
    return ohlcvcache[k]
end

function clearohlcvcache()
    global ohlcvcache
    ohlcvcache = Dict()
end

app.layout = html_div() do
    html_div(id="leftside", [
        html_div(id="select_buttons", [
            html_button("select all", id="all_button"),
            html_button("select none", id="none_button"),
            html_button("update data", id="update_data"),
            # html_button("reload data", id="reload_data"),
            html_button("reset selection", id="reset_selection")
        ]),
        # html_h1("Crypto Price"),  # style={"textAlign": "center"},
        dcc_graph(id="graph1day"),
        dcc_graph(id="graph10day"),
        dcc_graph(id="graph6month"),
        dcc_graph(id="graph_all"),
    ]),

    html_div(id="rightside", [
        dcc_dropdown(
            id="crypto_focus",
            options=[(label = i, value = i) for i in assets.df.base],
            value=assets.df[1, :base]
        ),
        dcc_checklist(
            id="indicator_select",
            options=[
                (label = "regression 1d", value = "regression 1d"),
                (label = "signals", value = "signals"),
                (label = "features", value = "features"),
                (label = "equal scale", value = "equal scale")],
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
            selected_rows = [ix-1 for ix in 1:size(assets.df, 1) if assets.df[ix, :portfolio]],
            selected_row_ids = [assets.df[ix, :base] for ix in 1:size(assets.df, 1) if assets.df[ix, :portfolio]],
            style_table=Dict("height" => "700px", "overflowY" => "auto")),
        html_div(id="graph_action"),
        html_div(id="messages")
            ])
end

function rangeselection(rl1d, rl10d, rl6M, rlall, rl4h)
    ctx = callback_context()
    graph_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
    res = (graph_id === nothing) ? "no range selection" : "$graph_id: "
    relayoutData = Dict(
        "graph1day" => rl1d,
        "graph10day" => rl10d,
        "graph6month" => rl6M,
        "graph_all" => rlall,
        "graph4h" => rl4h,
        "" => ""
    )
    if length(relayoutData[graph_id]) > 0
        range = relayoutData[graph_id]
        if !(range === nothing)
            if (try range.autosize catch end === nothing)
                if (try range[Symbol("xaxis.range[0]")]  catch end === nothing)
                    if (try range[Symbol("xaxis.autorange")]  catch end === nothing)
                        JSON3.pretty(JSON3.write(range))
                        res = res * "unexpected JSON - expecting xaxis.autorange"
                    else
                        res = res * "xaxis.autorange = $(range[Symbol("xaxis.autorange")])"
                    end
                else
                    xmin = range[Symbol("xaxis.range[0]")]
                    res = res * "xmin = $(range[Symbol("xaxis.range[0]")])"
                    if (try range[Symbol("xaxis.range[1]")]  catch end === nothing)
                        JSON3.pretty(JSON3.write(range))
                        res = res * ", unexpected JSON - expecting xaxis.range[1]"
                    else
                        xmax = range[Symbol("xaxis.range[1]")]
                        res = res * ", xmax = $(range[Symbol("xaxis.range[1]")])"
                    end
                end
            else
                # JSON3.pretty(JSON3.write(range))
                res = res * "relayout data: autosize = $(range[:autosize])"
            end
        end
    else
        res = res * "no relayout"
    end
    return res
end

    callback!(
    app,
    # Output("graph1day", "children"),
    Output("graph_action", "children"),
    Input("graph1day", "relayoutData"),
    Input("graph10day", "relayoutData"),
    Input("graph6month", "relayoutData"),
    Input("graph_all", "relayoutData"),
    Input("graph4h", "relayoutData")
    # prevent_initial_call=true
) do rl1d, rl10d, rl6M, rlall, rl4h
    res = rangeselection(rl1d, rl10d, rl6M, rlall, rl4h)
    println(res)
    return res
end
    """
Returns normalized percentage values related to `normref`, if `normref` is not `nothing`.
Otherwise the ydata input is returned.
"""
normpercent(ydata, ynormref) = (ynormref === nothing ) ? ydata : (ydata ./ ynormref .- 1) .* 100

"""
Returns start and end y coordinates of an input y coordinate vector or equidistant x coordinates.
These coordinates are retunred as normalized percentage values related to `normref`, if `normref` is not `nothing`.
"""
function regressionline(equiy, normref)
    regrwindow = size(equiy, 1)
    regry, grad = Features.rollingregression(equiy, regrwindow)
    regrliney = [regry[end] - grad[end] * (regrwindow - 1), regry[end]]
    return normpercent(regrliney, normref)
end

function linegraph(select, interval, period)
    if length(select) == 0
        return [scatter(x=[], y=[], mode="lines", name="no select")]
    end
    traces = nothing
    for base in select
        # println("linegraph base $base keys(oc): $(keys(oc))")
        df = loadohlcv(base, interval)
        enddt = df[end, :opentime]
        startdt = enddt - period
        days = Dates.Day(enddt - startdt)
        df = df[startdt .< df.opentime .<= enddt, :]
        startdt = df[begin, :opentime]  # in case there is less data than requested by period
        normref = df[end, :pivot]
        # normref = nothing
        if traces === nothing
            traces = [scatter(x=df.opentime, y=normpercent(df[!, :pivot], normref), mode="lines", name=base)]
        else
            append!(traces, [scatter(x=df.opentime, y=normpercent(df[!, :pivot], normref), mode="lines", name=base)])
        end

        append!(traces, [scatter(x=[startdt, enddt],
            y=regressionline(df[!, :pivot], normref),
            mode="lines", name="$base $days d")])
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
    Input("kpi_table", "selected_row_ids"),
    Input("kpi_table", "data"),
    Input("reset_selection", "n_clicks")
    # prevent_initial_call=true
) do select, tabledata, resetrange
    ctx = callback_context()
    button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
    s = "create linegraphs: select = $select, trigger: $(ctx.triggered[1].prop_id)"
    println(s)
    # println(keys(ohlcvcache))
    fig1d = Plot(linegraph(select, "1m", Dates.Hour(24)))
    fig10d = Plot(linegraph(select, "1m", Dates.Day(10)))
    fig6M = Plot(linegraph(select, "1d", Dates.Month(6)))
    figall = Plot(linegraph(select, "1d", Dates.Year(3)))

    return fig1d, fig10d, fig6M, figall
end

function candlestickgraph(base, interval, period)
    df = loadohlcv(base, interval)
    enddt = df[end, :opentime]
    startdt = enddt - period
    df = df[startdt .< df.opentime .<= enddt, :]

    normref = df[end, :pivot]
    fig = Plot([
        candlestick(
            x=df[!, :opentime],
            open=normpercent(df[!, :open], normref),
            high=normpercent(df[!, :high], normref),
            low=normpercent(df[!, :low], normref),
            close=normpercent(df[!, :close], normref),
            name="$base OHLC"),
        scatter(
            x=[startdt, enddt],
            y=regressionline(df[!, :pivot], normref),
            mode="lines", name="$base $(size(df, 1)/60) h"),
        bar(x=df.opentime, y=df.basevolume, name="basevolume", yaxis="y2")
        ],
        Layout(title_text="4hOHLCV", xaxis_title_text="time", yaxis_title_text="OHLCV % of last pivot",
            yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.2],
            yaxis_domain=[0.3, 1.0])
        )
    return fig
end

callback!(
    app,
    # Output("graph1day", "children"),
    Output("graph4h", "figure"),
    Input("crypto_focus", "value"),
    Input("kpi_table", "data"),
    Input("reset_selection", "n_clicks"),
    # prevent_initial_call=true
    ) do focus, tabledata, resetrange
        ctx = callback_context()
        button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
        s = "create candlestick graph: focus = $focus, trigger: $(ctx.triggered[1].prop_id)"
        println(s)
        if focus === nothing
            return Plot([scatter(x=[], y=[], mode="lines", name="no select")])
        else
            fig4h = Plot(candlestickgraph(focus, "1m", Dates.Hour(4)))
            return fig4h
        end
    end

callback!(
        app,
        Output("kpi_table", "data"),
        # Output("messages", "children"),
        Input("update_data", "n_clicks"),
        State("kpi_table", "data")
        # prevent_initial_call=true
    ) do update, olddata
        global assets
        if !(update === nothing)
            clearohlcvcache()
            assets = updateassets!(assets, true)
            if !(assets === nothing)
                return Dict.(pairs.(eachrow(assets.df)))
            end
        end
        println("staying with current table data")
        return olddata
    end

callback!(
        app,
        Output("kpi_table", "selected_rows"),
        Output("kpi_table", "selected_row_ids"),
        Input("all_button", "n_clicks"),
        Input("none_button", "n_clicks"),
        # Input("kpi_table", "active_cell"),
        State("kpi_table", "selected_rows"),
        State("kpi_table", "selected_row_ids")
        # prevent_initial_call=true
    ) do all, none, selectrows, selectids
        ctx = callback_context()
        button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : "kpi_table"
        s = "all = $all, none = $none, select = $selectids, button ID: $button_id"
        println(s)
        setselectrows = (selectrows === nothing) ? [] : [r for r in selectrows]  # convert JSON3 array to ordinary String array
        setselectids = (selectids === nothing) ? [] : [r for r in selectids]   # convert JSON3 array to ordinary Int64 array
        # if !(active_cell isa Nothing)
        #     push!(setselectrows, active_cell.row)
        #     push!(setselectids, active_cell.row_id)
        # end
        res = Dict(
            "all_button" => (0:(size(assets.df.base, 1)-1), assets.df.base),
            "none_button" => ([], []),
            "kpi_table" => (setselectrows, setselectids),
            "" => (setselectrows, setselectids))
        println("returning: $(res[button_id])")
        return res[button_id]
    end

    # function baseactivecell(active_cell)
#     if active_cell === nothing
#         return ""
#     else
#         ractive = String(active_cell)
#         ractive = replace(ractive, "\""=>"")
#         m = match(r"row_id: (?<base>\w+)", ractive)
#         if m === nothing
#             return ""
#         else
#             base = m["base"]
#             return base
#         end
#     end
# end

    callback!(app,
    Output("crypto_focus", "value"),
    Input("kpi_table", "active_cell"),
    State("crypto_focus", "value"),
    ) do active_cell, currentfocus
        if active_cell isa Nothing
            active_row_id = currentfocus
        else
            active_row_id = active_cell.row_id
        end
        return active_row_id
    end


run_server(app, "0.0.0.0", debug=true)
