# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())
# cd(@__DIR__)

# TODO deviation spread and tracker visualization
# TODO deviation spread and tracker window clear color other regression windows opaque
# TODO yellow within x sigma range, green break out, red plunge
# TODO green marker = buy, black marker sell, red marker emergency sell in 4h Candlestick
# TODO load TestOhlcv dynamically instead of using a file cache to be able to adapt testdata
# TODO load test Assets dynamically instead of using a file cache to be able to add/remove testdata bases
# using Colors
import Dash: dash, callback!, run_server, Output, Input, State, callback_context
import Dash: dcc_graph, html_h1, html_div, dcc_checklist, html_button, dcc_dropdown, dash_datatable
import PlotlyJS: PlotlyBase, Plot, dataset, Layout, attr, scatter, candlestick, bar, heatmap
using Dates, DataFrames, Logging
using EnvConfig, Ohlcv, Features, Targets, Assets, Classify

include("../scripts/cockpitdatatablecolors.jl")

dtf = "yyyy-mm-dd HH:MM"
# EnvConfig.init(EnvConfig.production)
EnvConfig.init(EnvConfig.test)

# app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")
cssdir = EnvConfig.setprojectdir()  * "/scripts/"
println("css dir: $cssdir")
app = dash(external_stylesheets = ["dashboard.css"], assets_folder=cssdir)
ohlcvcache = Dict()

function loadohlcv(base, interval)
    global ohlcvcache
    k = base * interval
    if !(k in keys(ohlcvcache))
        ohlcv = Ohlcv.defaultohlcv(base)
        Ohlcv.setinterval!(ohlcv, interval)
        Ohlcv.read!(ohlcv)
        # Ohlcv.accumulate!(ohlcv, interval)
        ohlcvcache[k] = ohlcv
    end
    return ohlcvcache[k]
end

function updateassets(download=false)
    global ohlcvcache

    if !download
        a = Assets.read()
    end
    if download || (size(a.basedf, 1) == 0)
        ohlcvcache = Dict()
        a = Assets.loadassets()
        println(a)
    else
        a = Assets.read()
    end
    if !(a === nothing) && (size(a.basedf, 1) > 0)
        sort!(a.basedf, [:portfolio], rev=true)
        a.basedf.id = a.basedf.base
        println("updating table data of size: $(size(a.basedf))")
    end
    return a
end

assets = updateassets(false)
# println(assets)
println("last assets update: $(assets.basedf[1, :update]) type $(typeof(assets.basedf[1, :update]))")
app.layout = html_div() do
    html_div(id="leftside", [
        html_div(id="select_buttons", [
            html_button("select all", id="all_button"),
            html_button("select none", id="none_button"),
            html_button("update data", id="update_data"),
            # html_button("reload data", id="reload_data"),
            html_button("reset selection", id="reset_selection")
        ]),
        html_div(id="graph1day_endtime", children=assets.basedf[1, :update]),
        dcc_graph(id="graph1day"),
        html_div(id="graph10day_endtime", children=assets.basedf[1, :update]),
        dcc_graph(id="graph10day"),
        html_div(id="graph6month_endtime", children=assets.basedf[1, :update]),
        dcc_graph(id="graph6month"),
        html_div(id="graph_all_endtime", children=assets.basedf[1, :update]),
        dcc_graph(id="graph_all"),
    ]),

    html_div(id="rightside", [
        dcc_dropdown(
            id="crypto_focus",
            options=[(label = i, value = i) for i in assets.basedf.base],
            value=assets.basedf[1, :base]
        ),
        dcc_checklist(
            id="indicator_select",
            options=[
                (label = "regression 1d", value = "regression1d"),
                (label = "test", value = "test"),
                (label = "features", value = "features"),
                (label = "targets", value = "targets"),
                (label = "normalize", value = "normalize")],
                value=[EnvConfig.configmode == EnvConfig.test ? "test" : "normalize"],
                # value=["regression1d", "normalize"],
                labelStyle=(:display => "inline-block")
        ),
        dcc_checklist(
            id="spread_select",
            options=[(label = Features.periodlabels(window), value = window) for window in Features.regressionwindows002],
                # value=["test"],
                # value=[60],
                labelStyle=(:display => "inline-block")
        ),
        html_div(id="graph4h_endtime", children=assets.basedf[1, :update]),
        html_div(id="targets4h"),
        dcc_graph(id="graph4h"),
        dash_datatable(id="kpi_table", editable=false,
            columns=[Dict("name" =>i, "id" => i, "hideable" => true) for i in names(assets.basedf) if i != "id"],  # exclude "id" to not display it
            data = Dict.(pairs.(eachrow(assets.basedf))),
            style_data_conditional=discrete_background_color_bins(assets.basedf, n_bins=31, columns="pricechangepercent"),
            filter_action="native",
            row_selectable="multi",
            sort_action="native",
            sort_mode="multi",
            fixed_rows=Dict("headers" => true),
            # selected_rows = [ix-1 for ix in 1:size(assets.basedf, 1) if assets.basedf[ix, :portfolio]],
            # selected_row_ids = [assets.basedf[ix, :base] for ix in 1:size(assets.basedf, 1) if assets.basedf[ix, :portfolio]],
            style_table=Dict("height" => "700px", "overflowY" => "auto")),
        html_div(id="graph_action"),
        html_div(id="click_action")
    ])
end

# function rangeselection(rl1d, rl10d, rl6M, rlall, rl4h)
#     ctx = callback_context()
#     graph_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
#     res = (graph_id === nothing) ? "no range selection" : "$graph_id: "
#     relayoutData = Dict(
#         "graph1day" => rl1d,
#         "graph10day" => rl10d,
#         "graph6month" => rl6M,
#         "graph_all" => rlall,
#         "graph4h" => rl4h,
#         "" => ""
#     )
#     if length(relayoutData[graph_id]) > 0
#         range = relayoutData[graph_id]
#         if !(range === nothing)
#             if (try range.autosize catch end === nothing)
#                 if (try range[Symbol("xaxis.range[0]")]  catch end === nothing)
#                     if (try range[Symbol("xaxis.autorange")]  catch end === nothing)
#                         JSON3.pretty(JSON3.write(range))
#                         res = res * "unexpected JSON - expecting xaxis.autorange"
#                     else
#                         res = res * "xaxis.autorange = $(range[Symbol("xaxis.autorange")])"
#                     end
#                 else
#                     xmin = range[Symbol("xaxis.range[0]")]
#                     res = res * "xmin = $(range[Symbol("xaxis.range[0]")])"
#                     if (try range[Symbol("xaxis.range[1]")]  catch end === nothing)
#                         JSON3.pretty(JSON3.write(range))
#                         res = res * ", unexpected JSON - expecting xaxis.range[1]"
#                     else
#                         xmax = range[Symbol("xaxis.range[1]")]
#                         res = res * ", xmax = $(range[Symbol("xaxis.range[1]")])"
#                     end
#                 end
#             else
#                 # JSON3.pretty(JSON3.write(range))
#                 res = res * "relayout data: autosize = $(range[:autosize])"
#             end
#         end
#     else
#         res = res * "no relayout"
#     end
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
                        # println("$graph_id clickdata: $(clk.points[1]) clickdata.x: $(clickdt)")  # String with DateTime
                    end
                end
            end
        end
    end
    return clickdt
end

callback!(
    app,
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
        updates = assets.basedf[assets.basedf[!, :base] .== focus, :update]
        clickdt = length(updates) > 0 ? updates[1] : ""
        # println("reset_selection n_clicks value: $(ctx.triggered[1].value)")
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
normpercent(ydata, ynormref) = donormalize ? Ohlcv.normalize(ydata, ynormref=ynormref, percent=true) : ydata
donormalize = true

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

linegraphlayout() =
        Layout(yaxis_title_text="% of last pivot",
            yaxis3=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false))

function timebox!(traces, boxperiod, boxenddt)
    traces = traces === nothing ? PlotlyBase.GenericTrace{Dict{Symbol, Any}}[] : traces
    boxstartdt = boxenddt - boxperiod
    append!(traces,
        [scatter(x=[boxstartdt, boxstartdt, boxenddt, boxenddt, boxstartdt], y=[0, 1, 1, 0, 0],
            mode="lines", showlegend=false, yaxis="y3", line=attr(color="grey"))])
    return traces
end

function linegraph!(traces, select, interval, period, enddt, regression)
    traces = traces === nothing ? PlotlyBase.GenericTrace{Dict{Symbol, Any}}[] : traces
    if length(select) == 0
        traces = [scatter(x=[], y=[], mode="lines", name="no select")]
    end

    for base in select
        df = Ohlcv.dataframe(loadohlcv(base, interval))
        startdt = enddt - period
        days = Dates.Day(enddt - startdt)
        if size(df, 1) == 0
            continue
        end
        enddt = enddt < df[begin, :opentime] ? df[begin, :opentime] : enddt
        enddt = enddt > df[end, :opentime] ? df[end, :opentime] : enddt
        startdt = startdt > df[end, :opentime] ? df[end, :opentime] : startdt
        startdt = startdt < df[begin, :opentime] ? df[begin, :opentime] : startdt
        df = df[startdt .< df.opentime .<= enddt, :]
        if (size(df, 1) == 0) || (startdt == enddt)
            continue
        end
        startdt = df[begin, :opentime]  # in case there is less data than requested by period
        normref = df[end, :pivot]
        # normref = nothing
        xarr = df[:, :opentime]
        yarr = normpercent(df[:, :pivot], normref)
        # append!(traces, [scatter(x=xarr, y=yarr, mode="lines", name=base, legendgroup=base)])
        if regression
            # xarr = [enddt, startdt]
            # yarr = reverse(regressionline(df[!, :pivot], normref))
            # append!(traces, [scatter(x=xarr, y=yarr, mode="lines", name=base, legendgroup=base)])
            append!(xarr, [enddt, startdt])
            append!(yarr, reverse(regressionline(df[!, :pivot], normref)))
        end
        append!(traces, [scatter(x=xarr, y=yarr, mode="lines", name=base)])
    end
    return Plot(traces, linegraphlayout())
end

function targetfigure(base, period, enddt)
    fig = nothing
    if base === nothing
        return fig
    end
    ohlcv = loadohlcv(base, "1m")
    df = Ohlcv.dataframe(ohlcv)
    startdt = enddt - period
    enddt = enddt < df[begin, :opentime] ? df[begin, :opentime] : enddt
    startdt = startdt > df[end, :opentime] ? df[end, :opentime] : startdt
    subdf = df[startdt .< df.opentime .<= enddt, :]
    normref = subdf[end, :pivot]
    if size(subdf,1) > 0
        pivot = Ohlcv.pivot!(subdf)
        pivot = normpercent(pivot, normref)
        _, grad = Features.rollingregression(pivot, Features.regressionwindows001["1h"])
        labels, relativedistances, distances, regressionix, priceix = Targets.continuousdistancelabels(pivot, grad, Targets.defaultlabelthresholds)
        x = subdf[!, :opentime]
        y = ["distpeak4h"]
        z = [distances]
        t = [["p: $(Dates.format(x[priceix[ix]], "mm-dd HH:MM")) r: $(Dates.format(x[regressionix[ix]], "mm-dd HH:MM"))" for ix in 1:size(x, 1)]]
        dcg = dcc_graph(
            figure = (
                data = [
                    (x=x, z = z, text=t, type = "heatmap", name = "target distances"),
                ],
                layout = (title = "SF3 title",),
            )
        )
    end
    return dcg  # dcc_graph(figure=fig)
end

function addheatmap!(traces, ohlcv, subdf, normref)
    @assert size(subdf, 1) >= 1
    firstdt = subdf[begin, :opentime] - Dates.Minute(maximum(values(Features.regressionwindows001)))
    calcdf = ohlcv.df[firstdt .< ohlcv.df.opentime .<= subdf[end, :opentime], :]
    pivot = Ohlcv.pivot!(calcdf)
    pivot = normpercent(pivot, normref)
    fdf, featuremask = Features.getfeatures001(pivot)
    fdf = fdf[(end-size(subdf,1)+1):end, :]  # resize fdf to meet size of subdf
    x = subdf[!, :opentime]
    y = featuremask
    z = [[fdf[r, c] for c in y] for r in 1:size(x,1)]
    t = [["txt: $(fdf[r, c])" for r in 1:size(x,1)] for c in y]  # different z and text format -> Julia Dash heatmap() bug
    # println("x size = $(size(x)) y size = $(size(y)) z size = $(size(z))")
    # println("=$y")
    # println("z[1:2]=$(z[1:2][1:2])")
    # println("fdf[1:2]=$(fdf[1:2, y[1:2]])")

    traces = append!(traces, [
        heatmap(x=x, y=y, z=z, text=permutedims(t), yaxis="y4")
    ])
    fig = Plot(traces,
        Layout(xaxis_rangeslider_visible=false,
            yaxis=attr(title_text="% of last pivot", domain=[0.15, 0.65]),
            yaxis2=attr(title="vol", side="right", domain=[0.0, 0.1]),
            yaxis4=attr(visible =true, side="left", domain=[0.66, 1.0]),
            yaxis3=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false))
        )
    return fig
end

function featureset002(ohlcv, period, enddt)
    df = Ohlcv.dataframe(ohlcv)
    startdt = enddt - period - Dates.Minute(Features.requiredminutes)
    subdf = copy(df[startdt .< df.opentime .<= enddt, :], copycols=true)
    subohlcv = Ohlcv.copy(ohlcv)
    # println("len(subohlcv): $(size(subdf, 1)) len(ohlcv): $(size(df, 1))")
    # println("subohlcv===ohlcv?$(subohlcv===ohlcv) subdf===df?$(subdf===df)")
    subohlcv.df = subdf
    @assert size(subdf,1) > 0
    # pivot = Ohlcv.pivot!(calcdf)
    # pivot = normpercent(pivot, normref)
    f2 = Features.Features002(subohlcv)
    # println("F2=$f2")
    return f2
end

function logbreakix(df, brix)
    return
    dflen = size(df,1)
    println()
    println("df len=$dflen all in: $(all([1<=abs(bix)<=dflen  for bix in brix]))")
    println([(bix, df[abs(bix), :opentime]) for bix in brix])
end

function spreadtraces(f2, window, normref, period, enddt)
    startdt = enddt - period
    ohlcv = Features.ohlcv(f2)
    df = Ohlcv.dataframe(ohlcv)
    pivot = Ohlcv.pivot!(ohlcv)
    ftr = f2.regr[window]
    startix = endix = 0
    for ix in length(df.opentime):-1:1
        if (startix == 0) && (startdt > df[ix, :opentime])
            startix = ix
        end
        if (endix == 0) && (enddt > df[ix, :opentime])
            endix = ix
        end
    end
    # println("spreadtraces window=$window")
    @assert (startix > 0) && (endix > 0) && (startix <= endix)
    # println("startdt: $startdt, startix:$startix, enddt:$enddt, endix:$endix")
    x = [df[ix, :opentime] for ix in startix:endix]
    xarea = vcat(x, reverse(x))
    yarea = vcat([ftr.regry[ix] + ftr.medianstd[ix] for ix in startix:endix], [ftr.regry[ix] - ftr.medianstd[ix] for ix in endix:-1:startix])
    # println("regry x: size=$(size(xarea)) max=$(maximum(xarea)) min=$(minimum(xarea)) y: size=$(size(yarea)) max=$(maximum(yarea)) min=$(minimum(yarea)) ")
    s2 = scatter(x=xarea, y=normpercent(yarea, normref), fill="toself", fillcolor="rgba(0,100,80,0.2)", line=attr(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=false)
    # x = [df[ix, :opentime] for ix in startix:endix]

    # yarea = vcat([ftr.regry[ix] + ftr.std[ix] for ix in startix:endix], [ftr.regry[ix] - ftr.std[ix] for ix in endix:-1:startix])
    # s2 = scatter(x=xarea, y=normpercent(yarea, normref), fill="toself", fillcolor="rgba(20,100,80,0.2)", line=attr(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=false)

    y = [ftr.regry[ix] for ix in startix:endix]
    # println("regry x: size=$(size(x)) max=$(maximum(x)) min=$(minimum(x)) y: size=$(size(y)) max=$(maximum(y)) min=$(minimum(y)) ")
    s1 = scatter(name="regry", x=x, y=normpercent(y, normref), mode="lines", line=attr(color="rgb(31, 119, 180)", width=1))

    y = [pivot[ix] for ix in startix:endix]
    # println("regry x: size=$(size(x)) max=$(maximum(x)) min=$(minimum(x)) y: size=$(size(y)) max=$(maximum(y)) min=$(minimum(y)) ")
    s5 = scatter(name="pivot", x=x, y=normpercent(y, normref), mode="lines", line=attr(color="rgb(250, 250, 250)", width=1))

    breakoutix = Classify.breakoutextremesix001!(f2, window, 1.0, startix)
    logbreakix(df, breakoutix)
    xix = [ix for ix in breakoutix if startdt <= df[abs(ix), :opentime]  <= enddt]
    y = [df.high[ix] for ix in xix if ix > 0]
    x = [df[ix, :opentime] for ix in xix if ix > 0]
    s3 = scatter(name="breakout", x=x, y=normpercent(y, normref), mode="markers", marker=attr(size=10, line_width=2, symbol="arrow-down"))
    y = [df.low[abs(ix)] for ix in xix if ix < 0]
    x = [df[abs(ix), :opentime] for ix in xix if ix < 0]
    s4 = scatter(name="breakout", x=x, y=normpercent(y, normref), mode="markers", marker=attr(size=10, line_width=2, symbol="arrow-up"))
    return [s2, s1, s5, s3, s4]
end

function candlestickgraph(traces, base, interval, period, enddt, regression, heatmap, spread)
    traces = traces === nothing ? PlotlyBase.GenericTrace{Dict{Symbol, Any}}[] : traces
    fig = Plot([scatter(x=[], y=[], mode="lines", name="no select")])  # return an empty graph on failing asserts
    if base === nothing
        return fig
    end
    ohlcv = loadohlcv(base, interval)
    df = Ohlcv.dataframe(ohlcv)
    if !("opentime" in names(df))
        println("candlestickgraph $base len(df): $(size(df,1)) names: $(names(df))")
        return fig
    end
    startdt = enddt - period
    enddt = enddt < df[begin, :opentime] ? df[begin, :opentime] : enddt
    startdt = startdt > df[end, :opentime] ? df[end, :opentime] : startdt
    subdf = df[startdt .< df.opentime .<= enddt, :]
    if size(subdf,1) > 0
        normref = subdf[end, :pivot]
        traces = append!([
            candlestick(
                x=subdf[!, :opentime],
                open=normpercent(subdf[!, :open], normref),
                high=normpercent(subdf[!, :high], normref),
                low=normpercent(subdf[!, :low], normref),
                close=normpercent(subdf[!, :close], normref),
                name="$base OHLC")], traces)
        if regression
            traces = append!([
                scatter(
                    x=[startdt, enddt],
                    y=regressionline(subdf[!, :pivot], normref),
                    mode="lines", showlegend=false)], traces)
        end

        spread = isnothing(spread) ? [] : spread
        # println("typeof(spread): $(typeof(spread)) = $spread isempty(spread)=$(isempty(spread)); period=$period ")
        if !isempty(spread)
            f2 = featureset002(ohlcv, period, enddt)
            # println("priod=$period, enddt=$enddt")
            for win in spread
                # win = parse(Int64, window)
                #  visualization spreads
                traces = append!(spreadtraces(f2, win, normref, period, enddt), traces)
            end
            # println("length(traces)=$(length(traces))")
        end
        traces = append!([
            bar(x=subdf.opentime, y=subdf.basevolume, name="basevolume", yaxis="y2")], traces)
        # fig = addheatmap!(traces, ohlcv, subdf, normref)
        fig = Plot(traces,
            Layout(xaxis_rangeslider_visible=false,
                yaxis=attr(title_text="% of last pivot", domain=[0.3, 1.0]),
                yaxis2=attr(title="vol", side="right", domain=[0.0, 0.2]),
                yaxis3=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false))
            )
    end
    return fig
end

callback!(
    app,
    Output("graph1day", "figure"),
    Output("graph10day", "figure"),
    Output("graph6month", "figure"),
    Output("graph_all", "figure"),
    Output("graph4h", "figure"),
    Output("targets4h", "children"),
    Input("crypto_focus", "value"),
    Input("kpi_table", "selected_row_ids"),
    Input("graph1day_endtime", "children"),
    Input("graph10day_endtime", "children"),
    Input("graph6month_endtime", "children"),
    Input("graph_all_endtime", "children"),
    Input("graph4h_endtime", "children"),
    Input("spread_select", "value"),
    State("indicator_select", "value")
    # prevent_initial_call=true
) do focus, select, enddt1d, enddt10d, enddt6M, enddtall, enddt4h, spread, indicator
    ctx = callback_context()
    button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : ""
    s = "create linegraphs: focus = $focus, select = $select, trigger: $(ctx.triggered[1].prop_id)"
    # println(s)
    # println("enddt4h $enddt4h, enddt1d $enddt1d, enddt10d $enddt10d, enddt6M $enddt6M, enddtall $enddtall")
    drawbases = [focus]
    append!(drawbases, [s for s in select if s != focus])
    regression = "regression1d" in indicator
    heatmap = "features" in indicator

    fig4h = candlestickgraph(nothing, focus, "1m", Dates.Hour(4), Dates.DateTime(enddt4h, dtf), regression, heatmap, spread)
    targets4h = "targets" in indicator ? targetfigure(focus, Dates.Hour(4), Dates.DateTime(enddt4h, dtf)) : nothing
    fig1d = linegraph!(timebox!(nothing, Dates.Hour(4), Dates.DateTime(enddt4h, dtf)),
        drawbases, "1m", Dates.Hour(24), Dates.DateTime(enddt1d, dtf), regression)
    fig10d = linegraph!(timebox!(nothing, Dates.Hour(24), Dates.DateTime(enddt1d, dtf)),
        drawbases, "1m", Dates.Day(10), Dates.DateTime(enddt10d, dtf), regression)
    fig6M = candlestickgraph(timebox!(nothing, Dates.Day(10), Dates.DateTime(enddt10d, dtf)),
        focus, "1d", Dates.Month(6), Dates.DateTime(enddt6M, dtf), regression, false, [])
    # fig6M = linegraph!(timebox!(nothing, Dates.Day(10), Dates.DateTime(enddt10d, dtf)),
    #     drawbases, "1d", Dates.Month(6), Dates.DateTime(enddt6M, dtf), regression)
    figall = linegraph!(timebox!(nothing, Dates.Month(6), Dates.DateTime(enddt6M, dtf)),
        drawbases, "1d", Dates.Year(3), Dates.DateTime(enddtall, dtf), regression)

    return fig1d, fig10d, fig6M, figall, fig4h, targets4h
end

callback!(
        app,
        Output("reset_selection", "n_clicks"),
        Output("kpi_table", "data"),
        Output("crypto_focus", "value"),
        Output("crypto_focus", "options"),
        Input("kpi_table", "active_cell"),
        Input("update_data", "n_clicks"),
        Input("indicator_select", "value"),
        State("kpi_table", "data"),
        State("crypto_focus", "value"),
        State("crypto_focus", "options")
        # prevent_initial_call=true
    ) do active_cell, update, indicator, olddata, currentfocus, options
        global assets, donormalize

        ctx = callback_context()
        button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : "nothing"
        # println("trigger: $(button_id)  currentfocus $currentfocus  #options: $options")
        if active_cell isa Nothing
            active_row_id = currentfocus
        else
            active_row_id = active_cell.row_id
        end
        donormalize = "normalize" in indicator
        # if button_id == "indicator_select"
            if (EnvConfig.configmode == EnvConfig.production) && ("test" in indicator)  # switch from prodcution to test data
                EnvConfig.init(EnvConfig.test)
                assets = updateassets(false)
                active_row_id = assets.basedf[1, :base]
            elseif (EnvConfig.configmode == EnvConfig.test) && !("test" in indicator)  # switch from test to prodcution data
                EnvConfig.init(EnvConfig.production)
                assets = updateassets(false)
                active_row_id = assets.basedf[1, :base]
            end

        if button_id == "update_data"
            assets = updateassets(true)
        # else
        #     return 0, olddata, active_row_id, options
        end
        if !(assets === nothing)
            println("data update assets.basedf.size: $(size(assets.basedf))")
            return 1, Dict.(pairs.(eachrow(assets.basedf))), active_row_id, [(label = i, value = i) for i in assets.basedf.base]
        else
            @warn "found no assets"
            return 0, olddata, active_row_id, options
        end
    end

callback!(
        app,
        Output("kpi_table", "selected_rows"),
        Output("kpi_table", "selected_row_ids"),
        Input("all_button", "n_clicks"),
        Input("none_button", "n_clicks"),
        State("kpi_table", "selected_rows"),
        State("kpi_table", "selected_row_ids")
        # prevent_initial_call=true
    ) do all, none, selectrows, selectids
        ctx = callback_context()
        button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : "kpi_table"
        s = "all = $all, none = $none, select = $selectids, button ID: $button_id"
        # println(s)
        setselectrows = (selectrows === nothing) ? [] : [r for r in selectrows]  # convert JSON3 array to ordinary String array
        setselectids = (selectids === nothing) ? [] : [r for r in selectids]   # convert JSON3 array to ordinary Int64 array
        res = Dict(
            "all_button" => (0:(size(assets.basedf.base, 1)-1), assets.basedf.base),
            "none_button" => ([], []),
            "kpi_table" => (setselectrows, setselectids),
            "" => (setselectrows, setselectids))
        # println("select row returning: $(res[button_id])")
        return res[button_id]
    end

    # callback!(app,
    # Output("crypto_focus", "value"),
    # Input("kpi_table", "active_cell"),
    # State("crypto_focus", "value"),
    # ) do active_cell, currentfocus
    #     if active_cell isa Nothing
    #         active_row_id = currentfocus
    #     else
    #         active_row_id = active_cell.row_id
    #     end
    #     return active_row_id
    # end


run_server(app, "0.0.0.0", debug=true)
