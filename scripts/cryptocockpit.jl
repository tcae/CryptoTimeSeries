# import Pkg: activate
# cd("$(@__DIR__)/..")
# println("activated $(pwd())")
# activate(pwd())
# cd(@__DIR__)

# TODO treat orders as portfolio part, show sel limit as line
# TODO show % improvement of portfolio and whike portfolio
# TODO longbuy at next specific regresson minimum
# TODO migrate all windows to ohlcv graphs
# TODO deviation spread and tracker visualization
# TODO deviation spread and tracker window clear color other regression windows opaque
# TODO yellow within x sigma range, green break out, red plunge
# TODO green marker = longbuy, black marker longclose, red marker emergency longclose in 4h Candlestick
# TODO load TestOhlcv dynamically instead of using a file cache to be able to adapt testdata
# TODO load test Assets dynamically instead of using a file cache to be able to add/remove testdata bases
# using Colors
import Dash: dash, callback!, run_server, Output, Input, State, callback_context
import Dash: dcc_graph, html_h1, html_div, dcc_checklist, html_button, dcc_dropdown, dash_datatable
import PlotlyJS: PlotlyBase, Plot, dataset, Layout, attr, scatter, candlestick, bar, heatmap
using Dates, DataFrames, Logging
using EnvConfig, Ohlcv, Features, Targets, Assets, Classify, CryptoXch, Trade


function loadohlcv!(cp, base, interval)
    if !(base in keys(cp.coin))
        ohlcv = CryptoXch.ohlcv(cp.tc.xc, base)
        f4 = cp.tc.cl.bc[base].f4                      #* this is a Classifier011 hack !!!!! any other classifier in tc will derail it
        cp.coin[base] = CoinData(Dict(), nothing)
        cp.coin[base].ohlcv["1m"] = ohlcv
        cp.coin[base].f4 = f4
    end
    if !(interval in keys(cp.coin[base].ohlcv))
        @assert interval in ["5m", "1h", "1d", "3d"] "$interval not in [5m, 1h, 1d, 3d]"
        Threads.@threads for iv in ["5m", "1h", "1d", "3d"]
            ohlcv = Ohlcv.defaultohlcv(base)
            Ohlcv.setinterval!(ohlcv, iv)
            Ohlcv.setdataframe!(ohlcv, Ohlcv.accumulate(Ohlcv.dataframe(cp.coin[base].ohlcv["1m"]), iv))
            cp.coin[base].ohlcv[iv] = ohlcv
        end
    end
    return cp.coin[base].ohlcv[interval]
end

function updateassets!(cp, download=false)
    assets = CryptoXch.portfolio!(cp.tc.xc)
    cp.coin = Dict()
    if !download 
        Trade.read!(cp.tc)
    end
    if download || isnothing(cp.tc.cfg)
        Trade.tradeselection!(cp.tc, assets[!, :coin]; datetime=Dates.now(UTC), updatecache=true)
    end
    cp.tc.cfg = cp.tc.cfg[cp.tc.cfg[!, :classifieraccepted], :]
    Trade.addassetsconfig!(cp.tc, assets)
    # cp.update = Dates.now(UTC)
    # select!(cp.tc.cfg, Not(:update))
    if !isnothing(cp.tc.cfg) && (size(cp.tc.cfg, 1) > 0)
        cp.update = cp.tc.cfg[begin, :datetime]
        cp.tc.cfg.id = cp.tc.cfg[!, :basecoin]
        println("config + assets: $(cp.tc.cfg)")
        # println("updating table data of size: $(size(adf))")
        rows = size(cp.tc.cfg, 1)
        xcbases = CryptoXch.bases(cp.tc.xc)

        # # initial delay but quick switching between coins
        # for (ix, base) in enumerate(cp.tc.cfg[!, :basecoin])
        #     println("$(EnvConfig.now()) ($ix of $rows) loading $base")
        #     @assert base in xcbases "base=$base not in xcbases=$xcbases"
        #     ohlcv = loadohlcv!(cp, base, "1m")
        #     if size(ohlcv.df, 1) == 0
        #         println("skipping empty $(ohlcv.base)")
        #         continue
        #     end
        #     # if cp.update > ohlcv.df[end, :opentime] cp.update = ohlcv.df[end, :opentime] end
        #     Threads.@threads for interval in ["5m", "1h", "1d", "3d"]
        #         loadohlcv!(cp, base, interval)
        #     end
        #     # loadohlcv!(cp, base, "5m")
        #     # loadohlcv!(cp, base, "1h")
        #     # loadohlcv!(cp, base, "1d")
        # end
        println("$(EnvConfig.now()) all $rows coins loaded")
    else
        @warn "no config + assets found"
    end
    return cp.tc.cfg
end

mutable struct CoinData
    ohlcv  # Dict interval => OhlcvData
    f4     # Features004
end

mutable struct CockpitData
    tc               # Trade to use trade cache
    coin             # Dict of coin => CoinData
    update           # DateTime
    donormalize::Bool
    dtf::String      # DateTime display format
    cssdir::String
    function CockpitData()
        global cp
        dtf = "yyyy-mm-dd HH:MM"
        cssdir = EnvConfig.setprojectdir()  * "/scripts/"
        xc = CryptoXch.XchCache()
        cp = new(Trade.TradeCache(xc=xc), nothing, nothing, true, dtf, cssdir)
        updateassets!(cp, false)
        return cp
    end
end

include("../scripts/cockpitdatatablecolors.jl")

# EnvConfig.init(EnvConfig.production)
EnvConfig.init(production)
const CP = CockpitData()
# app = dash(external_stylesheets = ["dashboard.css"], assets_folder="/home/tor/TorProjects/CryptoTimeSeries/scripts/")
println("css dir: $(CP.cssdir)")
app = dash(external_stylesheets = ["dashboard.css"], assets_folder=CP.cssdir)


# println(CP.tc.cfg)
println("last assetsconfig update: $(CP.update)")
app.layout = html_div() do
    html_div(id="leftside", [
        html_div(id="select_buttons", [
            html_button("select all", id="all_button"),
            html_button("select none", id="none_button"),
            html_button("update data", id="update_data"),
            # html_button("reload data", id="reload_data"),
            html_button("reset selection", id="reset_selection")
        ]),
        html_div(id="graph1day_endtime", children=CP.update),
        dcc_graph(id="graph1day"),
        html_div(id="graph10day_endtime", children=CP.update),
        dcc_graph(id="graph10day"),
        html_div(id="graph6month_endtime", children=CP.update),
        dcc_graph(id="graph6month"),
        html_div(id="graph_all_endtime", children=CP.update),
        dcc_graph(id="graph_all"),
    ]),

    html_div(id="rightside", [
        dcc_dropdown(
            id="crypto_focus",
            options=[(label = i, value = i) for i in CP.tc.cfg[!, :basecoin]],
            value=CP.tc.cfg[1, :basecoin]
        ),
        dcc_checklist(
            id="indicator_select",
            options=[
                (label = "regression 1d", value = "regression1d"),
                (label = "test", value = "test"),
                (label = "features", value = "features"),
                (label = "targets", value = "targets"),
                (label = "normalize", value = "normalize")],
                value=[EnvConfig.configmode == EnvConfig.test ? "test" : "normalize", "regression1d"],
                # value=["regression1d", "normalize"],
                labelStyle=(:display => "inline-block")
        ),
        dcc_checklist(
            id="spread_select",
            options=[(label = label, value = value) for (label, value) in Features.regressionwindows004dict],
                # value=["test"],
                # value=[60],
                labelStyle=(:display => "inline-block")
        ),
        html_div(id="graph4h_endtime", children=CP.update),
        html_div(id="targets4h"),
        dcc_graph(id="graph4h"),
        dash_datatable(id="kpi_table", editable=false,
            columns=[Dict("name" =>i, "id" => i, "hideable" => true) for i in names(CP.tc.cfg) if !(i in ["id", "update"])],  # exclude "id" to not display it
            data = Dict.(pairs.(eachrow(CP.tc.cfg))),
            style_data_conditional=discrete_background_color_bins(CP.tc.cfg, n_bins=length(names(CP.tc.cfg)), columns="pricechangepercent"),
            filter_action="native",
            row_selectable="multi",
            sort_action="native",
            sort_mode="multi",
            fixed_rows=Dict("headers" => true),
            style_table=Dict("height" => "700px", "overflowY" => "auto")),
        html_div(id="graph_action"),
        html_div(id="click_action")
    ])
end
println("spread list: $([(label = label, value = value) for (label, value) in Features.regressionwindows004dict])")

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
        clickdt = Dates.format(CP.update, CP.dtf)
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
    println(result[graph_id])
    return result[graph_id]
end

"""
Returns normalized percentage values related to `normref`, if `normref` is not `nothing`.
Otherwise the ydata input is returned.
"""
# normpercent(cp, ydata, ynormref) = (ynormref === nothing ) ? ydata : (ydata ./ ynormref .- 1) .* 100
normpercent(cp, ydata, ynormref) = cp.donormalize ? Ohlcv.normalize(ydata, ynormref=ynormref, percent=true) : ydata

"""
Returns start and end y coordinates of an input y coordinate vector or equidistant x coordinates.
These coordinates are retunred as normalized percentage values related to `normref`, if `normref` is not `nothing`.
"""
function regressionline(cp, equiy, normref)
    regrwindow = size(equiy, 1)
    regry, grad = Features.rollingregression(equiy, regrwindow)
    regrliney = [regry[end] - grad[end] * (regrwindow - 1), regry[end]]
    return normpercent(cp, regrliney, normref)
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
        df = Ohlcv.dataframe(loadohlcv!(cp, base, interval))
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
        yarr = normpercent(cp, df[:, :pivot], normref)
        # append!(traces, [scatter(x=xarr, y=yarr, mode="lines", name=base, legendgroup=base)])
        if regression
            # xarr = [enddt, startdt]
            # yarr = reverse(regressionline(cp, df[!, :pivot], normref))
            # append!(traces, [scatter(x=xarr, y=yarr, mode="lines", name=base, legendgroup=base)])
            append!(xarr, [enddt, startdt])
            append!(yarr, reverse(regressionline(cp, df[!, :pivot], normref)))
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
    ohlcv = loadohlcv!(cp, base, "1m")
    df = Ohlcv.dataframe(ohlcv)
    startdt = enddt - period
    enddt = enddt < df[begin, :opentime] ? df[begin, :opentime] : enddt
    startdt = startdt > df[end, :opentime] ? df[end, :opentime] : startdt
    subdf = df[startdt .< df.opentime .<= enddt, :]
    normref = subdf[end, :pivot]
    if size(subdf,1) > 0
        pivot = Ohlcv.pivot!(subdf)
        pivot = normpercent(cp, pivot, normref)
        _, grad = Features.rollingregression(pivot, Features.regressionwindows004dict["1h"])
        labels, relativedistances, distances, priceix = Targets.continuousdistancelabels(pivot, Targets.defaultlabelthresholds)
        x = subdf[!, :opentime]
        y = ["distpeak4h"]
        z = [distances]
        t = [["pix=$(Dates.format(x[priceix[ix]], "mm-dd HH:MM")) " for ix in 1:size(x, 1)]]
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
    firstdt = subdf[begin, :opentime] - Dates.Minute(maximum(Features.regressionwindows004))
    calcdf = ohlcv.df[firstdt .< ohlcv.df.opentime .<= subdf[end, :opentime], :]
    pivot = Ohlcv.pivot!(calcdf)
    pivot = normpercent(cp, pivot, normref)
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

function spreadtraces(cp, base, ohlcvdf, window, normref)
    traces = []
    regr = (base in keys(cp.coin)) && (window in keys(cp.coin[base].f4.rw)) ? cp.coin[base].f4.rw[window] : nothing
    if isnothing(regr)
        @warn "no f4 regression data found for $base"
        return []
    end
    filter = falses(length(regr[!, :opentime]))
    ix = Ohlcv.rowix(regr[!, :opentime], ohlcvdf[begin, :opentime])
    for ts in ohlcvdf[!, :opentime]
        while regr[ix, :opentime] < ts
            ix += 1
        end
        filter[ix] = regr[ix, :opentime] == ts
    end
    if length(ohlcvdf[!, :opentime]) != count(filter)
        @warn "unexpected mismatch of length(ohlcvdf[!, :opentime])=$(length(ohlcvdf[!, :opentime])) != count(filter)=$(count(filter))"
        return []
    end
    regr = @view regr[filter, :]
    @assert size(ohlcvdf, 1) == size(regr, 1) "size(ohlcvdf, 1)=$(size(ohlcvdf, 1)) != size(regr, 1)=$(size(regr, 1))"

    xarea = vcat(ohlcvdf[!, :opentime], reverse(ohlcvdf[!, :opentime]))
    yarea = vcat(regr[!, :regry] * 1.01f0, reverse(regr[!, :regry] * 0.99f0))
    s2 = scatter(x=xarea, y=normpercent(cp, yarea, normref), fill="toself", fillcolor="rgba(0,100,80,0.2)", line=attr(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=false)
    yarea = vcat(regr[!, :regry] * 1.02f0, reverse(regr[!, :regry] * 0.98f0))
    s3 = scatter(x=xarea, y=normpercent(cp, yarea, normref), fill="toself", fillcolor="rgba(0,80,80,0.2)", line=attr(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=false)

    y = regr[!, :regry]
    s1 = scatter(name="regry", x=ohlcvdf[!, :opentime], y=normpercent(cp, y, normref), mode="lines", line=attr(color="rgb(31, 119, 180)", width=1))

    pivot = Ohlcv.pivot!(ohlcvdf)
    s5 = scatter(name="pivot", x=ohlcvdf[!, :opentime], y=normpercent(cp, pivot, normref), mode="lines", line=attr(color="rgb(250, 250, 250)", width=1))

    return [s3, s2, s1, s5]
end

function candlestickgraph(cp, traces, base, interval, period, enddt, regression, heatmap, spread)
    traces = traces === nothing ? PlotlyBase.GenericTrace{Dict{Symbol, Any}}[] : traces
    fig = Plot([scatter(x=[], y=[], mode="lines", name="no select")])  # return an empty graph on failing asserts
    if base === nothing
        return fig
    end
    ohlcv = loadohlcv!(cp, base, interval)
    df = Ohlcv.dataframe(ohlcv)
    if !("opentime" in names(df))
        println("candlestickgraph $base len(df): $(size(df,1)) names: $(names(df))")
        return fig
    end
    startdt = enddt - period
    enddt = enddt < df[begin, :opentime] ? df[begin, :opentime] : enddt
    startdt = startdt > df[end, :opentime] ? df[end, :opentime] : startdt
    subdf = @view df[startdt .< df.opentime .<= enddt, :]
    # startdt = startdt < df[begin, :opentime] ? df[begin, :opentime] : startdt
    # period = (enddt - startdt) + Dates.Minute(1)
    if size(subdf,1) > 0
        normref = subdf[end, :pivot]
        traces = append!([
            candlestick(
                x=subdf[!, :opentime],
                open=normpercent(cp, subdf[!, :open], normref),
                high=normpercent(cp, subdf[!, :high], normref),
                low=normpercent(cp, subdf[!, :low], normref),
                close=normpercent(cp, subdf[!, :close], normref),
                name="$base OHLC")], traces)
        if regression
            traces = append!([
                scatter(
                    x=[startdt, enddt],
                    y=regressionline(cp, subdf[!, :pivot], normref),
                    mode="lines", showlegend=false)], traces)
        end

        intervalminutes = interval == "1m" ? 1 : floor(Int, Ohlcv.periods[interval] / Minute(1))
        spread = isnothing(spread) ? [] : spread
        # println("typeof(spread): $(typeof(spread)) = $spread isempty(spread)=$(isempty(spread)); period=$period ")
        if !isempty(spread)
            # println("period=$period, enddt=$enddt")
            for win in spread
                # winminutes = win == "1m" ? 1 : Features.regressionwindows004dict[win]
                if win > intervalminutes
                    # win = parse(Int64, window)
                    #  visualization spreads
                    traces = append!(spreadtraces(cp, base, subdf, win, normref), traces)
                end
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
    s = "create linegraphs: focus = $focus, select = $select, trigger: $(ctx.triggered[1].prop_id) spread = $spread"
    println(s)
    # println("enddt4h $enddt4h, enddt1d $enddt1d, enddt10d $enddt10d, enddt6M $enddt6M, enddtall $enddtall")
    drawbases = [focus]
    append!(drawbases, [s for s in select if s != focus])
    regression = "regression1d" in indicator
    heatmap = "features" in indicator

    # println("enddt4h($(typeof(enddt4h)))=$(enddt4h) dtf=$(CP.dtf)")
    fig4h = candlestickgraph(CP, nothing, focus, "1m", Dates.Hour(4), Dates.DateTime(enddt4h, CP.dtf), regression, heatmap, spread)
    targets4h = "targets" in indicator ? targetfigure(focus, Dates.Hour(4), Dates.DateTime(enddt4h, CP.dtf)) : nothing
    # fig1d = linegraph!(timebox!(nothing, Dates.Hour(4), Dates.DateTime(enddt4h, CP.dtf)),
    #     drawbases, "1m", Dates.Hour(24), Dates.DateTime(enddt1d, CP.dtf), regression)
    fig1d = candlestickgraph(CP, timebox!(nothing, Dates.Hour(4), Dates.DateTime(enddt4h, CP.dtf)), focus, "5m", Dates.Hour(24), Dates.DateTime(enddt1d, CP.dtf), regression, false, spread)
    # fig10d = linegraph!(timebox!(nothing, Dates.Hour(24), Dates.DateTime(enddt1d, CP.dtf)),
    #     drawbases, "1m", Dates.Day(10), Dates.DateTime(enddt10d, CP.dtf), regression)
    fig10d = candlestickgraph(CP, timebox!(nothing, Dates.Hour(24), Dates.DateTime(enddt1d, CP.dtf)), focus, "1h", Dates.Day(10), Dates.DateTime(enddt10d, CP.dtf), regression, false, spread)
    # fig6M = linegraph!(timebox!(nothing, Dates.Day(10), Dates.DateTime(enddt10d, CP.dtf)),
    #     drawbases, "1d", Dates.Month(6), Dates.DateTime(enddt6M, CP.dtf), regression)
    fig6M = candlestickgraph(CP, timebox!(nothing, Dates.Day(10), Dates.DateTime(enddt10d, CP.dtf)), focus, "1d", Dates.Month(8), Dates.DateTime(enddt6M, CP.dtf), regression, false, spread)
    # figall = linegraph!(timebox!(nothing, Dates.Month(6), Dates.DateTime(enddt6M, CP.dtf)),
    #     drawbases, "1d", Dates.Year(3), Dates.DateTime(enddtall, CP.dtf), regression)
    figall = candlestickgraph(CP, timebox!(nothing, Dates.Year(1), Dates.DateTime(enddt6M, CP.dtf)), focus, "3d", Dates.Year(4), Dates.DateTime(enddtall, CP.dtf), false, false, nothing)

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

        ctx = callback_context()
        button_id = length(ctx.triggered) > 0 ? split(ctx.triggered[1].prop_id, ".")[1] : "nothing"
        # println("trigger: $(button_id)  currentfocus $currentfocus  #options: $options")
        if active_cell isa Nothing
            active_row_id = currentfocus
        else
            active_row_id = active_cell.row_id #TODO clear table before coin update otherwise this raises and error with row_id not found
        end
        CP.donormalize = "normalize" in indicator
        # if button_id == "indicator_select"
            if (EnvConfig.configmode == EnvConfig.production) && ("test" in indicator)  # switch from prodcution to test data
                EnvConfig.init(EnvConfig.test)
                updateassets!(CP, false)
                active_row_id = CP.tc.cfg[1, :basecoin]
            elseif (EnvConfig.configmode == EnvConfig.test) && !("test" in indicator)  # switch from test to prodcution data
                EnvConfig.init(EnvConfig.production)
                updateassets!(CP, false)
                active_row_id = CP.tc.cfg[1, :basecoin]
            end

        if button_id == "update_data"
            updateassets!(CP, true)
        # else
        #     return 0, olddata, active_row_id, options
        end
        if !isnothing(CP.tc.cfg)
            # println("data update CP.tc.cfg.size: $(size(CP.tc.cfg))")
            return 1, Dict.(pairs.(eachrow(CP.tc.cfg))), active_row_id, [(label = i, value = i) for i in CP.tc.cfg[!, :basecoin]]
        else
            @warn "found no assetsconfig"
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
            "all_button" => (0:(size(CP.tc.cfg[!, :basecoin], 1)-1), CP.tc.cfg[!, :basecoin]),
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


run_server(app, "0.0.0.0", debug=false)
