import Pkg: activate
cd("$(@__DIR__)/..")
println("activated $(pwd())")
activate(pwd())
cd(@__DIR__)

# include("../test/testohlcv.jl")
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
using Dates, DataFrames, JSON, JSON3, Logging
using ..Config, ..Ohlcv, ..Features, ..Assets

dtf = "yyyy-mm-dd HH:MM"
Config.init(Config.production)

# app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")
cssdir = Config.setprojectdir()  * "/scripts/"
# cssdir = pwd() * "/scripts/"
println("css dir: $cssdir")
app = dash(external_stylesheets = ["dashboard.css"], assets_folder=cssdir)
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
        ohlcvcache[k] = ohlcv
    end
    return ohlcvcache[k]
end

function updateassets!(assets, download=false)
    global ohlcvcache

    a = assets
    if download
        ohlcvcache = Dict()
        a = Assets.loadassets(dayssperiod=Dates.Year(4), minutesperiod=Dates.Week(4))
    else
        a = Assets.read()
    end
    if !(a === nothing)
        # for (ix, base) in enumerate(a.df.base)
        #     ohlcv = loadohlcv(base, "1m")
        #     df = Ohlcv.dataframe(ohlcv)
        #     startdt = df[end, :opentime] - Dates.Day(10)
        #     df = df[startdt .< df.opentime, :]
        #     features = Features.features001set(ohlcv)
        #     for col in features.featuremask
        #         if !(col in names(a.df))
        #             a.df[:, col] .= 0.0
        #         end
        #         a.df[ix, col] = features.df[end, col]
        #     end
        # end

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
println("last assets update: $(assets.df[1, :update]) type $(typeof(assets.df[1, :update]))")
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
        html_div(id="graph1day_endtime", children=assets.df[1, :update]),
        dcc_graph(id="graph1day"),
        html_div(id="graph10day_endtime", children=assets.df[1, :update]),
        dcc_graph(id="graph10day"),
        html_div(id="graph6month_endtime", children=assets.df[1, :update]),
        dcc_graph(id="graph6month"),
        html_div(id="graph_all_endtime", children=assets.df[1, :update]),
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
        html_div(id="graph4h_endtime", children=assets.df[1, :update]),
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
            # selected_rows = [ix-1 for ix in 1:size(assets.df, 1) if assets.df[ix, :portfolio]],
            # selected_row_ids = [assets.df[ix, :base] for ix in 1:size(assets.df, 1) if assets.df[ix, :portfolio]],
            style_table=Dict("height" => "700px", "overflowY" => "auto")),
        html_div(id="graph_action"),
        html_div(id="click_action")
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

# callback!(
#     app,
#     # Output("graph1day", "children"),
#     Output("graph_action", "children"),
#     Input("graph1day", "relayoutData"),
#     Input("graph10day", "relayoutData"),
#     Input("graph6month", "relayoutData"),
#     Input("graph_all", "relayoutData"),
#     Input("graph4h", "relayoutData")
#     # prevent_initial_call=true
# ) do rl1d, rl10d, rl6M, rlall, rl4h
#     res = rangeselection(rl1d, rl10d, rl6M, rlall, rl4h)
#     println(res)
#     return res
# end

function clickx(graph_id, clk1d, clk10d, clk6M, clkall, clk4h)
    if (graph_id === nothing) || (graph_id == "")
        return ""
    end
    clickdata = Dict(
        "graph4h" => clk4h,
        "graph1day" => clk1d,
        "graph10day" => clk10d,
        "graph6month" => clk6M,
        "graph_all" => clkall,
    )
    clickdt = ""
    if length(clickdata[graph_id]) > 0
        clk = clickdata[graph_id]
        if !(clk === nothing)
            # JSON3.pretty(JSON3.write(clk))
            if (try clk.points catch end === nothing)
                println("no click points")
            else
                if (try clk.points[1] catch end === nothing)
                    println("no click points[1]")
                else
                    if (try clk.points[1].x catch end === nothing)
                        println("no click points[1].x")
                    else
                        clickdt = clk.points[1].x
                        # println("type of clickdata: $(typeof(clickdt)) clickdata: $(clickdt)")  # String with DateTime
                    end
                end
            end
        end
    end
    return clickdt
end

callback!(
    app,
    # Output("graph1day", "children"),
    Output("graph4h_endtime", "children"),
    Output("graph1day_endtime", "children"),
    Output("graph10day_endtime", "children"),
    Output("graph6month_endtime", "children"),
    Output("graph_all_endtime", "children"),
    Output("click_action", "children"),
    Input("graph4h", "clickData"),
    Input("graph1day", "clickData"),
    Input("graph10day", "clickData"),
    Input("graph6month", "clickData"),
    Input("graph_all", "clickData"),
    Input("reset_selection", "n_clicks"),
    State("crypto_focus", "value"),
    State("graph4h_endtime", "children"),
    State("graph1day_endtime", "children"),
    State("graph10day_endtime", "children"),
    State("graph6month_endtime", "children"),
    State("graph_all_endtime", "children")
    # prevent_initial_call=true
) do clk4h, clk1d, clk10d, clk6M, clkall, reset, focus, enddt4h, enddt1d, enddt10d, enddt6M, enddtall
    ctx = callback_context()
    graph_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : nothing
    res = (graph_id === nothing) ? "no click selection" : "$graph_id: "
    if graph_id == "reset_selection"
        updates = assets.df[assets.df[!, :base] .== focus, :update]
        clickdt = updates[1]
        println("reset_selection n_clicks value: $(ctx.triggered[1].value)")
        # println("focus update on $updates type $(typeof(updates)) using $clickdt type $(typeof(clickdt))")
    else
        clickdt = clickx(graph_id, clk1d, clk10d, clk6M, clkall, clk4h)
    end
    res = res * clickdt
    result = Dict(
        "graph4h" => (clickdt, enddt1d, enddt10d, enddt6M, enddtall, res),
        "graph1day" => (clickdt, enddt1d, enddt10d, enddt6M, enddtall, res),
        "graph10day" => (clickdt, clickdt, enddt10d, enddt6M, enddtall, res),
        "graph6month" => (clickdt, clickdt, clickdt, enddt6M, enddtall, res),
        "graph_all" => (clickdt, clickdt, clickdt, clickdt, enddtall, res),
        "reset_selection" => (clickdt, clickdt, clickdt, clickdt, clickdt, res),
        nothing => (enddt4h, enddt1d, enddt10d, enddt6M, enddtall, res)
    )
    return result[graph_id]
end

"""
Returns normalized percentage values related to `normref`, if `normref` is not `nothing`.
Otherwise the ydata input is returned.
"""
# normpercent(ydata, ynormref) = (ynormref === nothing ) ? ydata : (ydata ./ ynormref .- 1) .* 100
normpercent(ydata, ynormref) = Ohlcv.normalize(ydata, ynormref=ynormref, percent=true)

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

function linegraph(select, interval, period, enddt, boxperiod, boxenddt)
    if length(select) == 0
        return [scatter(x=[], y=[], mode="lines", name="no select")]
    end
    traces = nothing
    for base in select
        # println("linegraph base $base keys(oc): $(keys(oc))")
        df = Ohlcv.dataframe(loadohlcv(base, interval))
        # enddt = df[end, :opentime]
        startdt = enddt - period
        days = Dates.Day(enddt - startdt)
        enddt = enddt < df[begin, :opentime] ? df[begin, :opentime] : enddt
        enddt = enddt > df[end, :opentime] ? df[end, :opentime] : enddt
        startdt = startdt > df[end, :opentime] ? df[end, :opentime] : startdt
        startdt = startdt < df[begin, :opentime] ? df[begin, :opentime] : startdt
        df = df[startdt .< df.opentime .<= enddt, :]
        startdt = df[begin, :opentime]  # in case there is less data than requested by period
        normref = df[end, :pivot]
        # normref = nothing
        xarr = df[:, :opentime]
        append!(xarr, [enddt, startdt])
        yarr = normpercent(df[:, :pivot], normref)
        append!(yarr, reverse(regressionline(df[!, :pivot], normref)))
        if traces === nothing
            traces = [scatter(x=xarr, y=yarr, mode="lines", name=base)]
            # traces = [scatter(x=df.opentime, y=normpercent(df[!, :pivot], normref), mode="lines", name=base, color=base)]
        else
            append!(traces, [scatter(x=xarr, y=yarr, mode="lines", name=base)])
        end

        # append!(traces, [scatter(x=[startdt, enddt], y=regressionline(df[!, :pivot], normref), mode="lines", color=base, showlegend=false)])
    end
    boxstartdt = boxenddt - boxperiod
    append!(traces, [
        scatter(x=[boxstartdt, boxstartdt, boxenddt, boxenddt, boxstartdt], y=[0, 1, 1, 0, 0],
            mode="lines", showlegend=false, yaxis="y2", line=attr(color="grey"))
    ])
    return traces
end

callback!(
    app,
    # Output("graph1day", "children"),
    Output("graph1day", "figure"),
    Output("graph10day", "figure"),
    Output("graph6month", "figure"),
    Output("graph_all", "figure"),
    Input("crypto_focus", "value"),
    Input("kpi_table", "selected_row_ids"),
    Input("graph1day_endtime", "children"),
    Input("graph10day_endtime", "children"),
    Input("graph6month_endtime", "children"),
    Input("graph_all_endtime", "children"),
    State("graph4h_endtime", "children")
    # prevent_initial_call=true
) do focus, select, enddt1d, enddt10d, enddt6M, enddtall, enddt4h
    ctx = callback_context()
    button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
    s = "create linegraphs: focus = $focus, select = $select, trigger: $(ctx.triggered[1].prop_id)"
    println(s)
    println("enddt1d $enddt1d, enddt10d $enddt10d, enddt6M $enddt6M, enddtall $enddtall")
    # println(keys(ohlcvcache))
    drawselect = [focus]
    append!(drawselect, [s for s in select if s != focus])
    # Layout(title_text="$(Dates.Hour(24)) pivot", xaxis_title_text="time", yaxis_title_text="% of last pivot",
    fig1d = Plot(linegraph(drawselect, "1m", Dates.Hour(24), Dates.DateTime(enddt1d, dtf), Dates.Hour(4), Dates.DateTime(enddt4h, dtf)),
        Layout(yaxis_title_text="% of last pivot",
            yaxis2=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false)))
    fig10d = Plot(linegraph(drawselect, "1m", Dates.Day(10), Dates.DateTime(enddt10d, dtf), Dates.Hour(24), Dates.DateTime(enddt1d, dtf)),
        Layout(yaxis_title_text="% of last pivot",
            yaxis2=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false)))
    fig6M = Plot(linegraph(drawselect, "1d", Dates.Month(6), Dates.DateTime(enddt6M, dtf), Dates.Day(10), Dates.DateTime(enddt10d, dtf)),
        Layout(yaxis_title_text="% of last pivot",
            yaxis2=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false)))
    figall = Plot(linegraph(drawselect, "1d", Dates.Year(3), Dates.DateTime(enddtall, dtf), Dates.Month(6), Dates.DateTime(enddt6M, dtf)),
        Layout(yaxis_title_text="% of last pivot",
            yaxis2=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false)))

    return fig1d, fig10d, fig6M, figall
end

function candlestickgraph(base, interval, period, enddt)
    df = Ohlcv.dataframe(loadohlcv(base, interval))
    # enddt = df[end, :opentime]
    startdt = enddt - period
    enddt = enddt < df[begin, :opentime] ? df[begin, :opentime] : enddt
    startdt = startdt > df[end, :opentime] ? df[end, :opentime] : startdt
    df = df[startdt .< df.opentime .<= enddt, :]
    if size(df,1) == 0
        fig = Plot([scatter(x=[], y=[], mode="lines", name="no select")])
    else
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
                mode="lines", showlegend=false),
            bar(x=df.opentime, y=df.basevolume, name="basevolume", yaxis="y2")
            ],
            Layout(yaxis_title_text="% of last pivot",
                yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.2],
                yaxis_domain=[0.3, 1.0])
            )
    end
    return fig
end

callback!(
    app,
    # Output("graph1day", "children"),
    Output("graph4h", "figure"),
    Input("crypto_focus", "value"),
    Input("graph4h_endtime", "children")
    # prevent_initial_call=true
    ) do focus, enddt4h
        ctx = callback_context()
        button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
        s = "create candlestick graph: focus = $focus, trigger: $(ctx.triggered[1].prop_id)"
        println(s)
        if focus === nothing
            return Plot([scatter(x=[], y=[], mode="lines", name="no select")])
        else
            println("enddt4h $enddt4h")
            fig4h = Plot(candlestickgraph(focus, "1m", Dates.Hour(4), Dates.DateTime(enddt4h, dtf)))
            return fig4h
        end
    end

callback!(
        app,
        Output("reset_selection", "n_clicks"),
        Output("kpi_table", "data"),
        # Output("click_action", "children"),
        Input("update_data", "n_clicks"),
        State("kpi_table", "data")
        # prevent_initial_call=true
    ) do update, olddata
        global assets
        if !(update === nothing)
            assets = updateassets!(assets, true)
            if !(assets === nothing)
                return 1, Dict.(pairs.(eachrow(assets.df)))
            end
        end
        println("staying with current table data")
        return 0, olddata
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
