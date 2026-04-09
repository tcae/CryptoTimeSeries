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
using EnvConfig, Ohlcv, Features, Targets, Assets, Classify, CryptoXch, Trade, TradingStrategy

include("optimizationconfigs.jl")

const COCKPIT_TREND_REF = "025"
const COCKPIT_BOUNDS_REF = "001"
const COCKPIT_TRADEADVICE_REF = "025"
const DIAGNOSTIC_LABEL_ROWS = ["trend target", "trend pred", "tradepairs target", "lstm pred"]
const DIAGNOSTIC_LABEL_CODE = Dict{Any, Int}(
    missing => 0,
    Targets.shortbuy => 1,
    Targets.shorthold => 2,
    Targets.shortclose => 3,
    Targets.allclose => 4,
    Targets.longbuy => 5,
    Targets.longhold => 6,
    Targets.longclose => 7,
)
const DIAGNOSTIC_LABEL_COLOR = Dict{Any, String}(
    missing => "#808080",
    Targets.shortbuy => "#99000d",
    Targets.shorthold => "#ef3b2c",
    Targets.shortclose => "#fcbba1",
    Targets.allclose => "#dbeafe",
    Targets.longbuy => "#006d2c",
    Targets.longhold => "#31a354",
    Targets.longclose => "#a1d99b",
)
const DIAGNOSTIC_LABEL_TICKS = [
    "missing",
    "shortbuy",
    "shorthold",
    "shortclose",
    "allclose",
    "longbuy",
    "longhold",
    "longclose",
]
const DIAGNOSTIC_STATUS_COLOR = Dict(
    "within" => "#2ca02c",
    "late" => "#ffbf00",
    "opposite" => "#d62728",
    "pending" => "#9aa0a6",
)
const COCKPIT_DIAGNOSTIC_CACHE = Dict{String, Any}()

_cfgget(cfg, key::Symbol, default=nothing) = (cfg isa NamedTuple && (key in keys(cfg))) ? getfield(cfg, key) : (hasproperty(cfg, key) ? getproperty(cfg, key) : default)

function _cockpit_model_modes()
    modes = EnvConfig.Mode[]
    for mode in [EnvConfig.configmode, EnvConfig.training, EnvConfig.test, EnvConfig.production]
        mode in modes || push!(modes, mode)
    end
    return modes
end

function _cached_artifact(loader::Function, key::AbstractString)
    if haskey(COCKPIT_DIAGNOSTIC_CACHE, key)
        return COCKPIT_DIAGNOSTIC_CACHE[key]
    end
    value = loader()
    COCKPIT_DIAGNOSTIC_CACHE[key] = value
    return value
end

_cached_artifact(key::AbstractString, loader::Function) = _cached_artifact(loader, key)

function _resolve_trendconfig(ref::AbstractString)
    raw = lowercase(replace(strip(ref), r"config$" => ""))
    symbol = startswith(raw, "mk") ? Symbol(raw * "config") : Symbol("mk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown trend config ref=$ref; expected function $(symbol)"
    return getfield(@__MODULE__, symbol)()
end

function _resolve_boundsconfig(ref::AbstractString)
    raw = lowercase(replace(strip(ref), r"config$" => ""))
    symbol = startswith(raw, "boundsmk") ? Symbol(raw * "config") : startswith(raw, "mk") ? Symbol("bounds" * raw * "config") : Symbol("boundsmk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown bounds config ref=$ref; expected function $(symbol)"
    return getfield(@__MODULE__, symbol)()
end

function _resolve_tradeadviceconfig(ref::AbstractString)
    raw = lowercase(replace(strip(ref), r"config$" => ""))
    symbol = startswith(raw, "tradeadvicemk") ? Symbol(raw * "config") : startswith(raw, "mk") ? Symbol("tradeadvice" * raw * "config") : Symbol("tradeadvicemk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown trade advice config ref=$ref; expected function $(symbol)"
    return getfield(@__MODULE__, symbol)()
end

function _config_subfolder(prefix::AbstractString, cfg, mode::EnvConfig.Mode)
    folder = _cfgget(cfg, :folder, nothing)
    !isnothing(folder) && return String(folder)
    configname = string(_cfgget(cfg, :configname, "unknown"))
    return "$(prefix)-$(configname)-$(mode)"
end

function _with_log_subfolder(f::Function, folder::AbstractString)
    previous = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(folder)
    try
        return f()
    finally
        if previous == ""
            EnvConfig.setlogpath()
        else
            EnvConfig.setlogpath(previous)
        end
    end
end

_with_log_subfolder(folder::AbstractString, f::Function) = _with_log_subfolder(f, folder)

function _config_ref(kind::Symbol, sym::Symbol)::String
    raw = replace(String(sym), r"config$" => "")
    if kind == :trend
        return startswith(raw, "mk") ? raw[3:end] : raw
    elseif kind == :bounds
        return startswith(raw, "boundsmk") ? raw[(length("boundsmk") + 1):end] : raw
    elseif kind == :tradeadvice
        return startswith(raw, "tradeadvicemk") ? raw[(length("tradeadvicemk") + 1):end] : raw
    else
        return raw
    end
end

function _root_logfolder()
    previous = EnvConfig.logsubfolder()
    EnvConfig.setlogpath()
    root = EnvConfig.logfolder()
    if previous == ""
        EnvConfig.setlogpath()
    else
        EnvConfig.setlogpath(previous)
    end
    return root
end

function _config_tableexists(folder::AbstractString, filename::AbstractString)
    folderpath = joinpath(_root_logfolder(), folder)
    return EnvConfig.tableexists(filename; folderpath=folderpath, format=:auto)
end

function _load_cockpit_f4(ohlcv::Ohlcv.OhlcvData)
    f4 = Features.Features004(ohlcv; usecache=true)
    return isnothing(f4) ? Features.Features004(ohlcv.base, ohlcv.quotecoin) : f4
end

function _has_cached_output(kind::Symbol, sym::Symbol)
    try
        cfg = getfield(@__MODULE__, sym)()
        if kind == :trend
            return any(_cockpit_model_modes()) do mode
                folder = _config_subfolder("Trend", cfg, mode)
                _with_log_subfolder(folder) do
                    model = cfg.classifiermodel(Features.featurecount(cfg.featconfig), Targets.uniquelabels(cfg.targetconfig), "mix")
                    isfile(Classify.nnfilename(model.fileprefix)) || _config_tableexists(folder, "results")
                end
            end
        elseif kind == :bounds
            return any(_cockpit_model_modes()) do mode
                folder = _config_subfolder("Bounds", cfg, mode)
                _with_log_subfolder(folder) do
                    model = cfg.regressormodel(Features.featurecount(cfg.featconfig), ["center", "width"], "mix_relative_v1")
                    isfile(Classify.nnfilename(model.fileprefix)) || _config_tableexists(folder, "maxpredictions")
                end
            end
        elseif kind == :tradeadvice
            root = _root_logfolder()
            return any(mode -> EnvConfig.tableexists("lstm_predictions"; folderpath=joinpath(root, _config_subfolder("TradeAdviceLstm", cfg, mode)), format=:auto), _cockpit_model_modes()) ||
                   EnvConfig.tableexists("lstm_predictions"; folderpath=root, format=:auto)
        end
    catch err
        @debug "skipping config without usable cache" kind sym exception=(err, catch_backtrace())
    end
    return false
end

function _config_options(kind::Symbol)
    pattern = kind == :trend ? r"^mk.*config$" : kind == :bounds ? r"^boundsmk.*config$" : r"^tradeadvicemk.*config$"
    symbols = sort([s for s in names(@__MODULE__; all=true) if occursin(pattern, String(s))]; by=String)
    cached = [s for s in symbols if _has_cached_output(kind, s)]
    selected = isempty(cached) ? symbols : cached
    return [(label = "$(_config_ref(kind, s)) ($(String(s)))", value = _config_ref(kind, s)) for s in selected]
end

function _dropdownrow(label::AbstractString, id::AbstractString, options, value)
    return html_div([
        html_div(label, style=Dict("minWidth" => "118px", "fontWeight" => "600")),
        dcc_dropdown(id=id, options=options, value=value, clearable=false, style=Dict("flex" => "1")),
    ], style=Dict("display" => "flex", "alignItems" => "center", "gap" => "8px", "marginBottom" => "4px"))
end

function _active_row_id(active_cell, olddata, currentfocus)
    active_cell isa Nothing && return currentfocus

    row_id = _cfgget(active_cell, :row_id, nothing)
    !isnothing(row_id) && return String(row_id)

    row = _cfgget(active_cell, :row, nothing)
    if !isnothing(row)
        rowix = Int(row) + 1
        if (olddata !== nothing) && (1 <= rowix <= length(olddata))
            rowobj = olddata[rowix]
            fallback = _cfgget(rowobj, :basecoin, currentfocus)
            return String(_cfgget(rowobj, :id, fallback))
        end
    end

    return currentfocus
end

function _cockpit_contracts(; trendref::AbstractString=COCKPIT_TREND_REF, boundsref::AbstractString=COCKPIT_BOUNDS_REF, tradeadviceref::AbstractString=COCKPIT_TRADEADVICE_REF)
    tradecfg = _resolve_tradeadviceconfig(tradeadviceref)
    trendcfg = _resolve_trendconfig(trendref)
    boundscfg = _resolve_boundsconfig(boundsref)
    return (tradecfg=tradecfg, trendcfg=trendcfg, boundscfg=boundscfg, trendref=trendref, boundsref=boundsref, tradeadviceref=tradeadviceref)
end

function loadohlcv!(cp, base, interval)
    if !(base in keys(cp.coin))
        ohlcv = CryptoXch.ohlcv(cp.tc.xc, base)
        cp.coin[base] = CoinData(Dict(), nothing)
        cp.coin[base].ohlcv["1m"] = ohlcv
        cp.coin[base].f4 = _load_cockpit_f4(ohlcv)
    elseif isnothing(cp.coin[base].f4) || isempty(cp.coin[base].f4.rw)
        cp.coin[base].f4 = _load_cockpit_f4(cp.coin[base].ohlcv["1m"])
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
    assets = DataFrame((coin=String[], locked=Float32[], free=Float32[], borrowed=Float32[], accruedinterest=Float32[], usdtprice=Float32[], usdtvalue=Float32[])) # CryptoXch.portfolio!(cp.tc.xc)
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

EnvConfig.init(EnvConfig.test)
# EnvConfig.init(production)
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
        _dropdownrow("trend config", "trend_config_select", _config_options(:trend), COCKPIT_TREND_REF),
        _dropdownrow("bounds config", "bounds_config_select", _config_options(:bounds), COCKPIT_BOUNDS_REF),
        _dropdownrow("trade advice", "tradeadvice_config_select", _config_options(:tradeadvice), COCKPIT_TRADEADVICE_REF),
        dcc_checklist(
            id="indicator_select",
            options=[
                (label = "regression 1d", value = "regression1d"),
                (label = "test", value = "test"),
                (label = "features", value = "features"),
                (label = "targets / heatmap", value = "targets"),
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
price_axis_title(cp) = cp.donormalize ? "% of last pivot" : "price"

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

linegraphlayout(cp=CP) =
        Layout(yaxis_title_text=price_axis_title(cp),
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
    return Plot(traces, linegraphlayout(cp))
end

function _required_history_minutes(trendcfg, boundscfg)::Int
    minutes = max(Int(Features.requiredminutes(trendcfg.featconfig)), Int(Features.requiredminutes(boundscfg.featconfig)))
    if hasproperty(trendcfg.targetconfig, :maxwindow)
        minutes = max(minutes, Int(getproperty(trendcfg.targetconfig, :maxwindow)))
    end
    if hasproperty(boundscfg.targetconfig, :window)
        minutes = max(minutes, Int(getproperty(boundscfg.targetconfig, :window)))
    end
    return minutes
end

function _diagnostic_slice(ohlcv, period, enddt, history_minutes::Int)
    fulldf = Ohlcv.dataframe(ohlcv)
    view_enddt = min(enddt, fulldf[end, :opentime])
    view_startdt = max(fulldf[begin, :opentime], view_enddt - period + Dates.Minute(1))
    calc_startdt = max(fulldf[begin, :opentime], view_startdt - Dates.Minute(max(history_minutes, 0)))
    startix = Ohlcv.rowix(fulldf[!, :opentime], calc_startdt)
    endix = Ohlcv.rowix(fulldf[!, :opentime], view_enddt)
    sliceohlcv = Ohlcv.ohlcvview(ohlcv, startix:endix)
    calcdf = Ohlcv.dataframe(sliceohlcv)
    plotdf = calcdf[(view_startdt .<= calcdf[!, :opentime] .<= view_enddt), :]
    return (ohlcv=sliceohlcv, df=plotdf, startdt=plotdf[begin, :opentime], enddt=plotdf[end, :opentime])
end

function _discrete_colorscale()
    ordered = [missing, Targets.shortbuy, Targets.shorthold, Targets.shortclose, Targets.allclose, Targets.longbuy, Targets.longhold, Targets.longclose]
    boundaries = collect(range(0.0, 1.0; length=length(ordered) + 1))
    scale = Any[]
    for (ix, key) in enumerate(ordered)
        color = DIAGNOSTIC_LABEL_COLOR[key]
        push!(scale, Any[boundaries[ix], color])
        push!(scale, Any[boundaries[ix + 1], color])
    end
    return scale
end

function _safe_tradelabel(value)
    if ismissing(value)
        return missing
    elseif value isa Targets.TradeLabel
        return value
    else
        return Targets.tradelabel(String(value))
    end
end

_label_code(value) = get(DIAGNOSTIC_LABEL_CODE, _safe_tradelabel(value), 0)

function _load_classifier_from_interface(cfg)
    for mode in _cockpit_model_modes()
        folder = _config_subfolder("Trend", cfg, mode)
        nn = _cached_artifact("trend:" * folder) do
            _with_log_subfolder(folder) do
                model = cfg.classifiermodel(Features.featurecount(cfg.featconfig), Targets.uniquelabels(cfg.targetconfig), "mix")
                isfile(Classify.nnfilename(model.fileprefix)) ? Classify.loadnn(model.fileprefix) : nothing
            end
        end
        !isnothing(nn) && return nn
    end
    return nothing
end

function _load_regressor_from_interface(cfg)
    for mode in _cockpit_model_modes()
        folder = _config_subfolder("Bounds", cfg, mode)
        nn = _cached_artifact("bounds:" * folder) do
            _with_log_subfolder(folder) do
                model = cfg.regressormodel(Features.featurecount(cfg.featconfig), ["center", "width"], "mix_relative_v1")
                isfile(Classify.nnfilename(model.fileprefix)) ? Classify.loadnn(model.fileprefix) : nothing
            end
        end
        !isnothing(nn) && return nn
    end
    return nothing
end

function _load_lstm_overlay_from_interface(base::AbstractString, startdt, enddt, tradecfg)
    candidates = String[]
    for mode in _cockpit_model_modes()
        push!(candidates, _config_subfolder("TradeAdviceLstm", tradecfg, mode))
    end
    push!(candidates, "")
    seen = Set{String}()
    for folder in candidates
        folder in seen && continue
        push!(seen, folder)
        df = isempty(folder) ? _cached_artifact("lstm:root") do
            EnvConfig.readdf("lstm_predictions")
        end : _cached_artifact("lstm:" * folder) do
            _with_log_subfolder(folder) do
                EnvConfig.readdf("lstm_predictions")
            end
        end
        if !isnothing(df) && size(df, 1) > 0 && (:opentime in propertynames(df))
            mask = (string.(df[!, :coin]) .== base) .& (startdt .<= df[!, :opentime] .<= enddt)
            subdf = copy(df[mask, :])
            if size(subdf, 1) > 0
                sort!(subdf, :opentime)
                return subdf
            end
        end
    end
    return nothing
end

function _compute_trend_overlay(slice::NamedTuple, trendcfg)::DataFrame
    Features.setbase!(trendcfg.featconfig, slice.ohlcv, usecache=false)
    Targets.setbase!(trendcfg.targetconfig, slice.ohlcv)

    calcdf = Ohlcv.dataframe(slice.ohlcv)
    outdf = DataFrame(
        opentime=calcdf[!, :opentime],
        pivot=Float32.(calcdf[!, :pivot]),
        trend_target=collect(Targets.labels(trendcfg.targetconfig)),
        trend_pred=Vector{Any}(fill(missing, size(calcdf, 1))),
        trend_score=Vector{Union{Missing, Float32}}(fill(missing, size(calcdf, 1))),
    )

    nn = _load_classifier_from_interface(trendcfg)
    if !isnothing(nn)
        reqcols = Features.requestedcolumns(trendcfg.featconfig)
        featdf = Features.features(trendcfg.featconfig)
        X = Float32.(permutedims(Matrix(featdf[!, reqcols]), (2, 1)))
        probsdf = Classify.predictdf(nn, X)
        maxdf = Classify.maxpredictdf(nn, X)
        preddf = DataFrame(
            opentime=Features.opentime(trendcfg.featconfig),
            trend_pred=collect(maxdf[!, :label]),
            trend_score=Float32.(maxdf[!, :score]),
        )
        outdf = leftjoin(select(outdf, Not([:trend_pred, :trend_score])), preddf, on=:opentime)
        if !(:trend_pred in propertynames(outdf))
            outdf[!, :trend_pred] = Vector{Any}(fill(missing, size(outdf, 1)))
        end
        if !(:trend_score in propertynames(outdf))
            outdf[!, :trend_score] = Vector{Union{Missing, Float32}}(fill(missing, size(outdf, 1)))
        end
    end

    mask = (slice.startdt .<= outdf[!, :opentime] .<= slice.enddt)
    return outdf[mask, :]
end

function _compute_tradepair_targets(trenddf::AbstractDataFrame, tradecfg, trendcfg)
    tp = Targets.TradePairs(trendcfg.targetconfig; entryfraction=Float32(_cfgget(tradecfg, :entryfraction, 0.1f0)), exitfraction=Float32(_cfgget(tradecfg, :exitfraction, 0.1f0)))
    return Targets.tradepairlabels(tp, trenddf[!, :trend_target], Float32.(trenddf[!, :pivot]))
end

function _first_hit(rangeix, predicate)
    for ix in rangeix
        predicate(ix) && return ix
    end
    return nothing
end

function _denormalize_bounds(centerpred::AbstractVector{<:Real}, widthpred::AbstractVector{<:Real}, pivot::AbstractVector{<:Real})
    p = Float32.(pivot)
    centerabs = (1f0 .+ Float32.(centerpred)) .* p
    widthabs = Float32.(widthpred) .* p
    lowerabs = clamp.(centerabs .- widthabs ./ 2f0, 0f0, Inf32)
    upperabs = clamp.(centerabs .+ widthabs ./ 2f0, 0f0, Inf32)
    return lowerabs, upperabs
end

function _bound_status(df::AbstractDataFrame, predlow::AbstractVector{<:Real}, predhigh::AbstractVector{<:Real}, ix::Int, window::Int, side::Symbol)::NamedTuple
    n = size(df, 1)
    within_end = min(n, ix + max(window, 0))
    within_range = ix:within_end
    late_range = within_end < n ? ((within_end + 1):n) : ((n + 1):n)

    if side == :upper
        samehit = _first_hit(within_range, j -> Float32(df[j, :high]) >= Float32(predhigh[ix]))
        oppositehit = _first_hit(within_range, j -> Float32(df[j, :low]) <= Float32(predlow[ix]))
        latehit = _first_hit(late_range, j -> Float32(df[j, :high]) >= Float32(predhigh[ix]))
    else
        samehit = _first_hit(within_range, j -> Float32(df[j, :low]) <= Float32(predlow[ix]))
        oppositehit = _first_hit(within_range, j -> Float32(df[j, :high]) >= Float32(predhigh[ix]))
        latehit = _first_hit(late_range, j -> Float32(df[j, :low]) <= Float32(predlow[ix]))
    end

    if !isnothing(samehit) && (isnothing(oppositehit) || (samehit <= oppositehit))
        return (status="within", delay=samehit - ix, hitix=samehit)
    elseif !isnothing(oppositehit) && (isnothing(samehit) || (oppositehit < samehit))
        return (status="opposite", delay=oppositehit - ix, hitix=oppositehit)
    elseif !isnothing(latehit)
        return (status="late", delay=latehit - ix, hitix=latehit)
    else
        return (status="pending", delay=missing, hitix=missing)
    end
end

function _compute_bounds_overlay(slice::NamedTuple, boundscfg)
    Features.setbase!(boundscfg.featconfig, slice.ohlcv, usecache=false)
    Targets.setbase!(boundscfg.targetconfig, slice.ohlcv)
    featdf = Features.features(boundscfg.featconfig)
    reqcols = Features.requestedcolumns(boundscfg.featconfig)
    X = Float32.(permutedims(Matrix(featdf[!, reqcols]), (2, 1)))

    nn = _load_regressor_from_interface(boundscfg)
    isnothing(nn) && return nothing

    yraw = nn.model(X)
    centerpred = vec(Float32.(yraw[1, :]))
    widthpred = vec(clamp.(Float32.(yraw[2, :]), 0f0, Inf32))
    calcdf = Ohlcv.dataframe(slice.ohlcv)
    featuretimes = Features.opentime(boundscfg.featconfig)
    pivotmap = Dict(calcdf[ix, :opentime] => Float32(calcdf[ix, :pivot]) for ix in 1:size(calcdf, 1))
    predpivot = Float32[get(pivotmap, ts, 0f0) for ts in featuretimes]
    predlow, predhigh = _denormalize_bounds(centerpred, widthpred, predpivot)

    outdf = DataFrame(opentime=featuretimes, pred_low=Float32.(predlow), pred_high=Float32.(predhigh))
    outdf = leftjoin(outdf, select(calcdf, [:opentime, :high, :low, :close, :pivot]), on=:opentime)
    if (:lowtarget in propertynames(boundscfg.targetconfig.df)) && (:hightarget in propertynames(boundscfg.targetconfig.df))
        targetsubset = select(boundscfg.targetconfig.df, [:opentime, :lowtarget, :hightarget])
        rename!(targetsubset, :lowtarget => :target_low, :hightarget => :target_high)
        outdf = leftjoin(outdf, targetsubset, on=:opentime)
    else
        outdf[!, :target_low] = Float32.(predlow)
        outdf[!, :target_high] = Float32.(predhigh)
    end

    window = hasproperty(boundscfg.targetconfig, :window) ? Int(getproperty(boundscfg.targetconfig, :window)) : 0
    lowstatus = String[]
    highstatus = String[]
    lowdelay = Vector{Union{Missing, Int}}()
    highdelay = Vector{Union{Missing, Int}}()
    for ix in 1:size(outdf, 1)
        hi = _bound_status(outdf, outdf[!, :pred_low], outdf[!, :pred_high], ix, window, :upper)
        lo = _bound_status(outdf, outdf[!, :pred_low], outdf[!, :pred_high], ix, window, :lower)
        push!(highstatus, hi.status)
        push!(lowstatus, lo.status)
        push!(highdelay, hi.delay)
        push!(lowdelay, lo.delay)
    end

    outdf[!, :lowstatus] = lowstatus
    outdf[!, :highstatus] = highstatus
    outdf[!, :lowdelay] = lowdelay
    outdf[!, :highdelay] = highdelay
    mask = (slice.startdt .<= outdf[!, :opentime] .<= slice.enddt)
    return outdf[mask, :]
end

function _build_diagnostic_heatmap(trenddf::AbstractDataFrame, trade_targets, lstmdf)
    n = size(trenddf, 1)
    x = trenddf[!, :opentime]
    z = zeros(Int, length(DIAGNOSTIC_LABEL_ROWS), n)
    hovertext = fill("", length(DIAGNOSTIC_LABEL_ROWS), n)
    displaycolors = fill(DIAGNOSTIC_LABEL_COLOR[missing], length(DIAGNOSTIC_LABEL_ROWS), n)

    lstm_labels = isnothing(lstmdf) ? fill(missing, n) : begin
        mapping = Dict(row.opentime => row.label for row in eachrow(lstmdf))
        [get(mapping, ts, missing) for ts in x]
    end
    lstm_scores = isnothing(lstmdf) ? fill(missing, n) : begin
        mapping = Dict(row.opentime => row.score for row in eachrow(lstmdf))
        [get(mapping, ts, missing) for ts in x]
    end

    datasets = [
        (name="trend target", labels=trenddf[!, :trend_target], scores=fill(missing, n)),
        (name="trend pred", labels=trenddf[!, :trend_pred], scores=trenddf[!, :trend_score]),
        (name="tradepairs target", labels=trade_targets, scores=fill(missing, n)),
        (name="lstm pred", labels=lstm_labels, scores=lstm_scores),
    ]

    for (rowix, ds) in enumerate(datasets)
        for colix in 1:n
            lbl = _safe_tradelabel(ds.labels[colix])
            z[rowix, colix] = _label_code(lbl)
            scoretxt = ismissing(ds.scores[colix]) ? "" : "<br>score=$(round(Float32(ds.scores[colix]); digits=3))"
            labeltxt = ismissing(lbl) ? "missing" : string(lbl)
            hovertext[rowix, colix] = "field=$(ds.name)<br>time=$(x[colix])<br>label=$(labeltxt)$(scoretxt)"
            displaycolors[rowix, colix] = get(DIAGNOSTIC_LABEL_COLOR, lbl, DIAGNOSTIC_LABEL_COLOR[missing])
        end
    end

    heattrace = heatmap(
        x=x,
        y=DIAGNOSTIC_LABEL_ROWS,
        z=z,
        hoverinfo="skip",
        hoverongaps=false,
        zmin=-0.5,
        zmax=7.5,
        zsmooth=false,
        xgap=0,
        ygap=0,
        opacity=0.01,
        colorscale=_discrete_colorscale(),
        showscale=true,
        colorbar=attr(
            tickmode="array",
            tickvals=collect(0:7),
            ticktext=DIAGNOSTIC_LABEL_TICKS,
            tickfont=attr(size=9),
            len=0.17,
            y=0.88,
            x=1.02,
            xanchor="left",
            thickness=16,
            xpad=0,
        ),
        yaxis="y4",
        name="classifications",
        showlegend=false,
    )

    markersize = n <= 120 ? 16 : n <= 300 ? 12 : 9
    hovertraces = Any[heattrace]
    for rowix in length(DIAGNOSTIC_LABEL_ROWS):-1:1
        rowname = DIAGNOSTIC_LABEL_ROWS[rowix]
        push!(hovertraces,
            scatter(
                x=x,
                y=fill(rowname, n),
                mode="markers",
                text=collect(hovertext[rowix, :]),
                hoverinfo="text",
                marker=attr(
                    symbol="square",
                    size=markersize,
                    color=collect(displaycolors[rowix, :]),
                    line=attr(width=0.4, color="rgba(80,80,80,0.25)"),
                ),
                yaxis="y4",
                name=rowname,
                showlegend=false,
            )
        )
    end
    return hovertraces
end

function _status_line_trace(x, y, statusvec, delayvec, category::AbstractString, name::AbstractString; width::Int=2, hover::Bool=false)
    ycat = Vector{Union{Missing, Float32}}(undef, length(y))
    text = Vector{String}(undef, length(y))
    for ix in eachindex(y)
        if statusvec[ix] == category
            ycat[ix] = Float32(y[ix])
            delaytxt = ismissing(delayvec[ix]) ? "n/a" : string(delayvec[ix]) * "m"
            text[ix] = "$(name)<br>status=$(category)<br>delay=$(delaytxt)<br>value=$(round(Float32(y[ix]); digits=5))"
        else
            ycat[ix] = missing
            text[ix] = ""
        end
    end
    return scatter(x=x, y=ycat, mode="lines", name="$(name) $(category)", text=text, hoverinfo=hover ? "text" : "skip", line=attr(color=DIAGNOSTIC_STATUS_COLOR[category], width=width))
end

function _value_hover_trace(x, y, name::AbstractString)
    text = ["$(name)<br>value=$(round(Float32(val); digits=5))" for val in y]
    return scatter(
        x=x,
        y=Float32.(y),
        mode="markers",
        text=text,
        hoverinfo="text",
        marker=attr(symbol="circle", size=7, color="rgba(0,0,0,0.01)", line=attr(width=0)),
        name=name,
        showlegend=false,
    )
end

function _status_hover_trace(x, y, statusvec, delayvec, name::AbstractString)
    text = Vector{String}(undef, length(y))
    for ix in eachindex(y)
        delaytxt = ismissing(delayvec[ix]) ? "n/a" : string(delayvec[ix]) * "m"
        text[ix] = "$(name)<br>status=$(statusvec[ix])<br>delay=$(delaytxt)<br>value=$(round(Float32(y[ix]); digits=5))"
    end
    return scatter(
        x=x,
        y=Float32.(y),
        mode="markers",
        text=text,
        hoverinfo="text",
        marker=attr(symbol="circle", size=7, color="rgba(0,0,0,0.01)", line=attr(width=0)),
        name=name,
        showlegend=false,
    )
end

function _build_cockpit_diagnostics(base, period, enddt; trendref::AbstractString=COCKPIT_TREND_REF, boundsref::AbstractString=COCKPIT_BOUNDS_REF, tradeadviceref::AbstractString=COCKPIT_TRADEADVICE_REF)
    contracts = _cockpit_contracts(; trendref=trendref, boundsref=boundsref, tradeadviceref=tradeadviceref)
    ohlcv = loadohlcv!(cp, base, "1m")
    history_minutes = _required_history_minutes(contracts.trendcfg, contracts.boundscfg)
    slice = _diagnostic_slice(ohlcv, period, enddt, history_minutes)
    trenddf = _compute_trend_overlay(slice, contracts.trendcfg)
    trade_targets = _compute_tradepair_targets(trenddf, contracts.tradecfg, contracts.trendcfg)
    boundsdf = _compute_bounds_overlay(slice, contracts.boundscfg)
    lstmdf = _load_lstm_overlay_from_interface(base, slice.startdt, slice.enddt, contracts.tradecfg)
    return (slice=slice, trenddf=trenddf, trade_targets=trade_targets, boundsdf=boundsdf, lstmdf=lstmdf)
end

function _statuscountdict(df::AbstractDataFrame, col::Symbol)
    counts = Dict{String, Int}()
    if size(df, 1) == 0 || !(col in propertynames(df))
        return counts
    end
    for row in eachrow(combine(groupby(df, col), nrow => :rows))
        counts[string(row[col])] = Int(row.rows)
    end
    return counts
end

function _diagnostic_summary(diag)::String
    trenddist = Dict(string(lbl) => count(==(lbl), diag.trenddf[!, :trend_target]) for lbl in unique(diag.trenddf[!, :trend_target]))
    lstmcached = isnothing(diag.lstmdf) ? "no" : "yes"
    parts = [
        "trend_target=$(trenddist)",
        "lstm_cached=$(lstmcached)",
    ]
    if !isnothing(diag.boundsdf)
        push!(parts, "bounds_high=$(_statuscountdict(diag.boundsdf, :highstatus))")
        push!(parts, "bounds_low=$(_statuscountdict(diag.boundsdf, :lowstatus))")
    else
        push!(parts, "bounds=missing-model")
    end
    return join(parts, " | ")
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
            yaxis=attr(title_text=price_axis_title(cp), domain=[0.15, 0.65]),
            yaxis2=attr(title="vol", side="right", domain=[0.0, 0.1]),
            yaxis4=attr(visible =true, side="left", domain=[0.66, 1.0]),
            yaxis3=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false))
        )
    return fig
end

function spreadtraces(cp, base, ohlcvdf, window, normref)
    traces = []
    if (base in keys(cp.coin)) && (isnothing(cp.coin[base].f4) || !(window in keys(cp.coin[base].f4.rw)))
        cp.coin[base].f4 = _load_cockpit_f4(cp.coin[base].ohlcv["1m"])
    end
    regr = (base in keys(cp.coin)) && !isnothing(cp.coin[base].f4) && (window in keys(cp.coin[base].f4.rw)) ? cp.coin[base].f4.rw[window] : nothing
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
    #TODO replace 1% and 2% BAND AROUND REGRY WITH * STD AND 2 * STD
    # yarea = vcat(regr[!, :regry] * 1.01f0, reverse(regr[!, :regry] * 0.99f0))
    yarea = vcat(regr[!, :regry] + regr[!, :std], reverse(regr[!, :regry] - regr[!, :std]))
    s2 = scatter(x=xarea, y=normpercent(cp, yarea, normref), fill="toself", fillcolor="rgba(0,100,80,0.2)", line=attr(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=false)
    # yarea = vcat(regr[!, :regry] * 1.02f0, reverse(regr[!, :regry] * 0.98f0))
    yarea = vcat(regr[!, :regry] + 2 * regr[!, :std], reverse(regr[!, :regry] - 2 * regr[!, :std]))
    s3 = scatter(x=xarea, y=normpercent(cp, yarea, normref), fill="toself", fillcolor="rgba(0,80,80,0.2)", line=attr(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=false)

    y = regr[!, :regry]
    s1 = scatter(name="regry", x=ohlcvdf[!, :opentime], y=normpercent(cp, y, normref), mode="lines", line=attr(color="rgb(31, 119, 180)", width=1))

    pivot = Ohlcv.pivot!(ohlcvdf)
    s5 = scatter(name="pivot", x=ohlcvdf[!, :opentime], y=normpercent(cp, pivot, normref), mode="lines", line=attr(color="rgb(250, 250, 250)", width=1))

    return [s3, s2, s1, s5]
end

function candlestickgraph(cp, traces, base, interval, period, enddt, regression, heatmap, spread; diagnostics=nothing)
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

        if !isnothing(diagnostics)
            if !isnothing(diagnostics.boundsdf)
                x = diagnostics.boundsdf[!, :opentime]
                targethigh = normpercent(cp, diagnostics.boundsdf[!, :target_high], normref)
                targetlow = normpercent(cp, diagnostics.boundsdf[!, :target_low], normref)
                predhigh = normpercent(cp, diagnostics.boundsdf[!, :pred_high], normref)
                predlow = normpercent(cp, diagnostics.boundsdf[!, :pred_low], normref)
                append!(traces, [
                    scatter(x=x, y=targethigh, mode="lines", name="target high", hoverinfo="skip", line=attr(color="rgba(180,180,180,0.6)", dash="dot", width=1)),
                    scatter(x=x, y=targetlow, mode="lines", name="target low", hoverinfo="skip", line=attr(color="rgba(180,180,180,0.6)", dash="dot", width=1)),
                ])
                for category in ["within", "late", "opposite", "pending"]
                    append!(traces, [
                        _status_line_trace(x, predhigh, diagnostics.boundsdf[!, :highstatus], diagnostics.boundsdf[!, :highdelay], category, "pred high"; width=2, hover=false),
                        _status_line_trace(x, predlow, diagnostics.boundsdf[!, :lowstatus], diagnostics.boundsdf[!, :lowdelay], category, "pred low"; width=2, hover=false),
                    ])
                end
                append!(traces, [
                    _value_hover_trace(x, targetlow, "target low"),
                    _value_hover_trace(x, targethigh, "target high"),
                    _status_hover_trace(x, predlow, diagnostics.boundsdf[!, :lowstatus], diagnostics.boundsdf[!, :lowdelay], "pred low"),
                    _status_hover_trace(x, predhigh, diagnostics.boundsdf[!, :highstatus], diagnostics.boundsdf[!, :highdelay], "pred high"),
                ])
            end
            append!(traces, _build_diagnostic_heatmap(diagnostics.trenddf, diagnostics.trade_targets, diagnostics.lstmdf))
            fig = Plot(traces,
                Layout(xaxis_rangeslider_visible=false,
                    yaxis=attr(title_text=price_axis_title(cp), domain=[0.18, 0.80]),
                    yaxis2=attr(title="vol", side="right", domain=[0.0, 0.12]),
                    yaxis4=attr(title="", domain=[0.845, 0.915], automargin=true),
                    yaxis3=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false),
                    plot_bgcolor="rgba(238,238,238,1)",
                    legend=attr(
                        orientation="v",
                        x=1.02,
                        xanchor="left",
                        y=0.72,
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="rgba(160,160,160,0.7)",
                        borderwidth=1,
                        font=attr(size=10),
                    ),
                    margin=attr(l=60, r=220, t=40, b=40),
                    hovermode="x unified")
                )
        else
            # fig = addheatmap!(traces, ohlcv, subdf, normref)
            fig = Plot(traces,
                Layout(xaxis_rangeslider_visible=false,
                    yaxis=attr(title_text=price_axis_title(cp), domain=[0.3, 1.0]),
                    yaxis2=attr(title="vol", side="right", domain=[0.0, 0.2]),
                    yaxis3=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false))
                )
        end
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
    Input("trend_config_select", "value"),
    Input("bounds_config_select", "value"),
    Input("tradeadvice_config_select", "value"),
    Input("spread_select", "value"),
    State("indicator_select", "value")
    # prevent_initial_call=true
) do focus, select, enddt1d, enddt10d, enddt6M, enddtall, enddt4h, trendref, boundsref, tradeadviceref, spread, indicator
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
    showtargets = "targets" in indicator
    diagnostics4h = showtargets ? _build_cockpit_diagnostics(focus, Dates.Hour(4), Dates.DateTime(enddt4h, CP.dtf); trendref=string(trendref), boundsref=string(boundsref), tradeadviceref=string(tradeadviceref)) : nothing
    fig4h = candlestickgraph(CP, nothing, focus, "1m", Dates.Hour(4), Dates.DateTime(enddt4h, CP.dtf), regression, heatmap, spread; diagnostics=diagnostics4h)
    targets4h = isnothing(diagnostics4h) ? "enable 'targets / heatmap' to show lifecycle diagnostics" : "trend=$(trendref) | bounds=$(boundsref) | trade=$(tradeadviceref) | " * _diagnostic_summary(diagnostics4h)
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
    Output("trend_config_select", "value"),
    Output("bounds_config_select", "value"),
    Input("tradeadvice_config_select", "value"),
    State("trend_config_select", "value"),
    State("bounds_config_select", "value")
) do tradeadviceref, currenttrend, currentbounds
    tradecfg = _resolve_tradeadviceconfig(string(tradeadviceref))
    trendref = string(_cfgget(tradecfg, :trendconfigref, currenttrend))
    boundsref = string(_cfgget(tradecfg, :boundsconfigref, currentbounds))
    return trendref, boundsref
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
        active_row_id = _active_row_id(active_cell, olddata, currentfocus)
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
