"""
analyze.jl — Interactive Dash-based viewer for a Trades Arrow file.

Select a `*trades*.arrow` file (scanned recursively under `\$HOME/crypto/logs`),
pick a trading pair, and inspect the data as:
- a daily candlestick chart (1 candle = 1 day, built from the per-minute
  `close`/`high`/`low` columns of the Trades schema; there is no per-minute
  `open` field, so daily open/close use the first/last minute `close`),
- a minute-level bar chart for a clicked day, where each bar spans the
  minute `low..high` range, hovering a bar shows a short tooltip
  (`opentime`, `label`, `low`, `high`) and selects the matching row in the
  details table below the chart, executed trades are overlayed as
  triangles (green=long, red=short; placed above the bars for long and
  below for short; tip-up=open, tip-down=close), hovering a closing
  triangle shows the equity delta realized that minute,
- a gain% line (right axis) derived from the `equity` column relative to the
  first minute of the selected day,
- a details table below the minute chart with the full Trades v1 row for
  every minute of the selected day.

Usage:
    julia --project=scripts scripts/analyze.jl
"""

# Deliberately no `Pkg.activate` here: this must run under `--project=scripts` as-is.
# The workspace root environment resolves a different Dash.jl version without the
# PlotlyBase-aware serializer, which breaks multi-output callbacks returning a Plot.

using Dates, DataFrames, Arrow, Logging
import Dash: dash, callback!, run_server, Output, Input, State, callback_context
import Dash: dcc_graph, html_h3, html_div, html_button, dcc_dropdown, dash_datatable
import PlotlyJS: PlotlyBase, Plot, Layout, attr, scatter, bar, candlestick
using EnvConfig

# ─────────────────────────────────────────────────────────────────────────────
# Trades file discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

const TRADES_ROOT = EnvConfig.defaultlogfilespath

"Scan `root` recursively for `*trades*.arrow` files; label shows the path relative to `root` for experiment context."
function _scan_trades_files(root::AbstractString=TRADES_ROOT)
    options = NamedTuple{(:label, :value), Tuple{String, String}}[]
    isdir(root) || return options
    for (dirpath, _, filenames) in walkdir(root)
        for filename in filenames
            lowered = lowercase(filename)
            if occursin("trades", lowered) && endswith(lowered, ".arrow")
                fullpath = joinpath(dirpath, filename)
                push!(options, (label=relpath(fullpath, root), value=fullpath))
            end
        end
    end
    sort!(options; by=o -> o.label)
    return options
end

"Load one Trades Arrow file as a plain, sorted `DataFrame`."
function _load_trades_file(path::AbstractString)::DataFrame
    df = DataFrame(Arrow.Table(path); copycols=true)
    @assert :opentime in propertynames(df) "trades file $(path) missing :opentime column; names=$(names(df))"
    @assert :pair in propertynames(df) "trades file $(path) missing :pair column; names=$(names(df))"
    @assert :high in propertynames(df) && :low in propertynames(df) && :close in propertynames(df) "trades file $(path) missing high/low/close columns; names=$(names(df))"
    df[!, :pair] = string.(df[!, :pair])
    sort!(df, [:pair, :opentime])
    return df
end

"Mutable state shared across Dash callbacks (single active user session)."
mutable struct AnalyzeState
    filepath::Union{Nothing, String}
    rawdf::DataFrame
    pair::Union{Nothing, String}
    daydf::DataFrame
end
const AS = AnalyzeState(nothing, DataFrame(), nothing, DataFrame())

"Best-effort field access for both NamedTuple and JSON3.Object callback payloads (JSON3.Object does not reliably support `hasproperty`)."
function _cfgget(obj, key::Symbol, default=nothing)
    obj isa NamedTuple && return (key in keys(obj)) ? getfield(obj, key) : default
    try
        return getproperty(obj, key)
    catch
        return default
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Aggregation and figure builders
# ─────────────────────────────────────────────────────────────────────────────

"Return one row per day with open/high/low/close (open/close approximated from minute `close`) and start/end equity."
function _daily_aggregate(pairdf::AbstractDataFrame)::DataFrame
    tmp = DataFrame(pairdf)
    tmp[!, :date] = Dates.Date.(tmp[!, :opentime])
    grouped = groupby(tmp, :date)
    agg = combine(grouped,
        :close => first => :dayopen,
        :close => last => :dayclose,
        :high => maximum => :dayhigh,
        :low => minimum => :daylow,
        :equity => first => :equitystart,
        :equity => last => :equityend,
        nrow => :rows,
    )
    sort!(agg, :date)
    return agg
end

function _daily_figure(agg::AbstractDataFrame, pair::AbstractString)
    if nrow(agg) == 0
        return Plot([scatter(x=[], y=[], mode="lines", name="no data")])
    end
    trace = candlestick(x=agg[!, :date], open=agg[!, :dayopen], high=agg[!, :dayhigh], low=agg[!, :daylow], close=agg[!, :dayclose], name=pair)
    return Plot([trace], Layout(xaxis_rangeslider_visible=false, title="$(pair) daily candles (click a candle to inspect the day)"))
end

"Return the short hover text for one minute bar; the full row is shown in the details table below the chart instead."
function _row_hovertext(row)::String
    return "opentime=$(row[:opentime])<br>label=$(row[:label])<br>low=$(row[:low])<br>high=$(row[:high])"
end

"Return DataTable `columns`/`data` for the full Trades v1 rows of one day, all values stringified for safe JSON serialization."
function _table_columns_data(daydf::AbstractDataFrame)
    cols = names(daydf)
    columns = [Dict("name" => c, "id" => c) for c in cols]
    data = [Dict(c => string(daydf[ix, Symbol(c)]) for c in cols) for ix in 1:nrow(daydf)]
    return columns, data
end

"Return per-cell DataTable `tooltip_data`: hovering a cell shows opentime, column header, and cell content."
function _table_tooltip_data(daydf::AbstractDataFrame)
    cols = names(daydf)
    return [
        Dict(c => Dict("value" => "opentime=$(daydf[ix, :opentime])\ncolumn=$(c)\nvalue=$(daydf[ix, Symbol(c)])", "type" => "text") for c in cols)
        for ix in 1:nrow(daydf)
    ]
end

"Canonical Trades v1 schema column order, used to size the placeholder table before any file is loaded."
const TRADES_V1_COLUMNS = let
    cols = String["opentime", "pair", "close", "high", "low", "label", "score",
        "lp_amount", "sp_amount", "equity", "freemargin", "freequote",
        "lastopentrade", "set", "rangeid", "config", "tsmstate"]
    for lane in ("lo", "lc", "so", "sc")
        append!(cols, ["$(lane)_status", "$(lane)_id", "$(lane)_limit", "$(lane)_amount", "$(lane)_msg",
            "$(lane)l_status", "$(lane)l_id", "$(lane)l_filled", "$(lane)l_pavg", "$(lane)l_msg"])
    end
    cols
end

"Placeholder DataTable content sized like a real Trades v1 day (all schema columns, 24*60 blank rows); the dash_table JS renderer crashes on zero columns/rows."
function _placeholder_table_columns_data()
    columns = [Dict("name" => c, "id" => c) for c in TRADES_V1_COLUMNS]
    data = [Dict(c => "" for c in TRADES_V1_COLUMNS) for _ in 1:(24 * 60)]
    return columns, data
end

"Placeholder `tooltip_data` matching `_placeholder_table_columns_data()` row count."
function _placeholder_table_tooltip_data()
    return [Dict(c => Dict("value" => "", "type" => "text") for c in TRADES_V1_COLUMNS) for _ in 1:(24 * 60)]
end

"Return one point per executed (filled) trade lane for the given day, with side/action and the fill it belongs to."
function _executed_trade_points(daydf::AbstractDataFrame)
    lanes = (
        (statuscol=:lo_status, filledcol=:lol_filled, pavgcol=:lol_pavg, side=:long, action=:open),
        (statuscol=:lc_status, filledcol=:lcl_filled, pavgcol=:lcl_pavg, side=:long, action=:close),
        (statuscol=:so_status, filledcol=:sol_filled, pavgcol=:sol_pavg, side=:short, action=:open),
        (statuscol=:sc_status, filledcol=:scl_filled, pavgcol=:scl_pavg, side=:short, action=:close),
    )
    points = NamedTuple[]
    for lane in lanes
        (lane.statuscol in propertynames(daydf)) && (lane.filledcol in propertynames(daydf)) && (lane.pavgcol in propertynames(daydf)) || continue
        for ix in 1:nrow(daydf)
            status = lowercase(strip(string(daydf[ix, lane.statuscol])))
            status == "closed" || continue
            filled = ismissing(daydf[ix, lane.filledcol]) ? 0f0 : Float32(daydf[ix, lane.filledcol])
            avgprice = ismissing(daydf[ix, lane.pavgcol]) ? 0f0 : Float32(daydf[ix, lane.pavgcol])
            (filled > 0f0 && avgprice > 0f0) || continue
            equitydelta = ix > firstindex(daydf[!, :equity]) ? (daydf[ix, :equity] - daydf[ix - 1, :equity]) : missing
            push!(points, (
                opentime=daydf[ix, :opentime],
                side=lane.side,
                action=lane.action,
                filled=filled,
                avgprice=avgprice,
                high=daydf[ix, :high],
                low=daydf[ix, :low],
                equitydelta=equitydelta,
            ))
        end
    end
    return points
end

"Hover text for one executed-trade triangle; closing trades additionally show the equity delta realized that minute."
function _trade_hovertext(p)::String
    parts = ["side=$(p.side)", "action=$(p.action)", "opentime=$(p.opentime)", "filled=$(round(p.filled, digits=6))", "avgprice=$(round(p.avgprice, digits=6))"]
    if (p.action == :close) && !ismissing(p.equitydelta)
        push!(parts, "gain (equity Δ)=$(round(p.equitydelta, digits=4))")
    end
    return join(parts, "<br>")
end

const _LANE_MARKER_SPEC = Dict(
    (:long, :open) => (symbol="triangle-up", color="green", name="long open"),
    (:long, :close) => (symbol="triangle-down", color="green", name="long close"),
    (:short, :open) => (symbol="triangle-up", color="red", name="short open"),
    (:short, :close) => (symbol="triangle-down", color="red", name="short close"),
)

function _minute_figure(daydf::AbstractDataFrame, pair::AbstractString, date::Date)
    if nrow(daydf) == 0
        return Plot([scatter(x=[], y=[], mode="lines", name="no data")])
    end

    x = daydf[!, :opentime]
    high = Float64.(daydf[!, :high])
    low = Float64.(daydf[!, :low])
    equity = Float64.(daydf[!, :equity])
    equitystart = equity[begin] > 0.0 ? equity[begin] : 1.0
    gainpct = (equity ./ equitystart .- 1.0) .* 100.0
    hovertext = [_row_hovertext(daydf[ix, :]) for ix in 1:nrow(daydf)]

    bartrace = bar(x=x, y=(high .- low), base=low, text=hovertext, hoverinfo="text",
        marker=attr(color="rgba(100,120,200,0.35)"), name="minute range")
    gaintrace = scatter(x=x, y=gainpct, mode="lines", name="gain %", yaxis="y2", line=attr(color="rgb(230,140,0)", width=2))

    pricerange = max(maximum(high) - minimum(low), 1e-6)
    offset = pricerange * 0.03
    points = _executed_trade_points(daydf)
    bykey = Dict{Tuple{Symbol, Symbol}, Vector{NamedTuple}}()
    for p in points
        push!(get!(bykey, (p.side, p.action), NamedTuple[]), p)
    end

    lanetraces = PlotlyBase.AbstractTrace[]
    for (key, spec) in _LANE_MARKER_SPEC
        pts = get(bykey, key, NamedTuple[])
        isempty(pts) && continue
        xs = [p.opentime for p in pts]
        ys = [key[1] == :long ? (p.high + offset) : (p.low - offset) for p in pts]
        texts = [_trade_hovertext(p) for p in pts]
        push!(lanetraces, scatter(x=xs, y=ys, mode="markers", name=spec.name, text=texts, hoverinfo="text",
            marker=attr(symbol=spec.symbol, size=11, color=spec.color, line=attr(width=1, color="black"))))
    end

    traces = PlotlyBase.AbstractTrace[bartrace, lanetraces..., gaintrace]
    layout = Layout(
        title="$(pair) minute detail $(date)",
        xaxis=attr(title="time"),
        yaxis=attr(title="price", side="left"),
        yaxis2=attr(title="gain %", overlaying="y", side="right"),
        hovermode="closest",
    )
    return Plot(traces, layout)
end

# ─────────────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────────────

const CSSDIR = (@__DIR__) * "/"  # avoid EnvConfig.setprojectdir(), which calls Pkg.activate and can swap in a different Dash.jl
app = dash(external_stylesheets=["dashboard.css"], assets_folder=CSSDIR)

app.layout = html_div() do
    html_div([
        html_h3("Trades Analyzer"),
        html_div([
            html_div("trades file", style=Dict("minWidth" => "110px", "fontWeight" => "600")),
            dcc_dropdown(id="file_select", options=_scan_trades_files(), placeholder="select a trades .arrow file ($(TRADES_ROOT))", style=Dict("flex" => "1")),
            html_button("refresh files", id="refresh_files_button"),
        ], style=Dict("display" => "flex", "alignItems" => "center", "gap" => "8px", "marginBottom" => "6px")),
        html_div([
            html_div("pair", style=Dict("minWidth" => "110px", "fontWeight" => "600")),
            dcc_dropdown(id="pair_select", options=[], placeholder="select a trading pair", style=Dict("flex" => "1")),
        ], style=Dict("display" => "flex", "alignItems" => "center", "gap" => "8px", "marginBottom" => "10px")),
        dcc_graph(id="daily_chart"),
        html_div(id="selected_date_label", children="click a daily candle to inspect minute detail", style=Dict("margin" => "8px 0", "fontWeight" => "600")),
        dcc_graph(id="minute_chart"),
        dash_datatable(id="minute_table", columns=_placeholder_table_columns_data()[1], data=_placeholder_table_columns_data()[2],
            tooltip_data=_placeholder_table_tooltip_data(), tooltip_delay=0, tooltip_duration=nothing,
            row_selectable="single", selected_rows=[],
            filter_action="native", sort_action="native", fixed_rows=Dict("headers" => true),
            style_table=Dict("height" => "400px", "overflowY" => "auto", "overflowX" => "auto"),
            style_data_conditional=[]),
        html_div(id="scroll_sync", style=Dict("display" => "none")),
    ])
end

callback!(app, [Output("file_select", "options")], [Input("refresh_files_button", "n_clicks")]) do n_clicks
    return (_scan_trades_files(),)
end

# Scroll the selected (radio-checked) minute_table row into view; runs client-side since the table
# is row-virtualized and the active row is otherwise only marked by its (easy to miss) radio button.
callback!("""
function(selected_rows) {
    setTimeout(function() {
        var table = document.getElementById('minute_table');
        if (!table) { return; }
        var checked = table.querySelector('input[type=\"radio\"]:checked');
        if (checked) {
            var row = checked.closest('tr');
            if (row) { row.scrollIntoView({block: 'center', behavior: 'auto'}); }
        }
    }, 50);
    return window.dash_clientside.no_update;
}
""", app, [Output("scroll_sync", "title")], [Input("minute_table", "selected_rows")])

callback!(app, [Output("pair_select", "options"), Output("pair_select", "value")], [Input("file_select", "value")]) do filepath
    if isnothing(filepath) || isempty(String(filepath))
        AS.filepath = nothing
        AS.rawdf = DataFrame()
        AS.pair = nothing
        return [], nothing
    end
    df = _load_trades_file(String(filepath))
    AS.filepath = String(filepath)
    AS.rawdf = df
    pairs = sort(unique(df[!, :pair]))
    AS.pair = isempty(pairs) ? nothing : pairs[begin]
    return [(label=p, value=p) for p in pairs], AS.pair
end

callback!(app, [Output("daily_chart", "figure")], [Input("pair_select", "value")]) do pair
    if isnothing(pair) || isempty(String(pair)) || (nrow(AS.rawdf) == 0)
        return (Plot([scatter(x=[], y=[], mode="lines", name="no data")]),)
    end
    AS.pair = String(pair)
    pairdf = @view AS.rawdf[AS.rawdf[!, :pair] .== AS.pair, :]
    return (_daily_figure(_daily_aggregate(pairdf), AS.pair),)
end

callback!(app, [Output("minute_chart", "figure"), Output("selected_date_label", "children"),
        Output("minute_table", "columns"), Output("minute_table", "data"), Output("minute_table", "tooltip_data")], [Input("daily_chart", "clickData")]) do clickdata
    placeholder_columns, placeholder_data = _placeholder_table_columns_data()
    fallback = (Plot([scatter(x=[], y=[], mode="lines", name="no selection")]), "click a daily candle to inspect minute detail", placeholder_columns, placeholder_data, _placeholder_table_tooltip_data())
    (isnothing(clickdata) || isnothing(AS.pair) || (nrow(AS.rawdf) == 0)) && return fallback

    points = _cfgget(clickdata, :points, nothing)
    (isnothing(points) || isempty(points)) && return fallback
    xval = _cfgget(points[1], :x, nothing)
    isnothing(xval) && return fallback

    date = Date(String(xval)[1:10])
    pairdf = @view AS.rawdf[AS.rawdf[!, :pair] .== AS.pair, :]
    daydf = sort(DataFrame(pairdf[Dates.Date.(pairdf[!, :opentime]) .== date, :]), :opentime)
    AS.daydf = daydf
    fig = _minute_figure(daydf, AS.pair, date)
    label = "$(AS.pair) — $(date) ($(nrow(daydf)) minutes)"
    if (nrow(daydf) > 0) && all(==(0f0), daydf[!, :equity])
        label *= " ⚠ equity is flat/zero for this file — likely a TrendDetector gains-only replay (trades-td.arrow) that bypasses Xch account bookkeeping; load a tradesim-replay/trades-replay.arrow or trades-ts.arrow file for a populated equity/gain line"
    end
    tablecolumns, tabledata = _table_columns_data(daydf)
    tooltipdata = _table_tooltip_data(daydf)
    return fig, label, tablecolumns, tabledata, tooltipdata
end

callback!(app, [Output("minute_table", "selected_rows")], [Input("minute_chart", "hoverData")]) do hoverdata
    isnothing(hoverdata) && return ([],)
    points = _cfgget(hoverdata, :points, nothing)
    (isnothing(points) || isempty(points)) && return ([],)
    pt = points[1]
    # only the minute-range bar trace (curveNumber 0) drives table row selection
    curveix = _cfgget(pt, :curveNumber, nothing)
    (isnothing(curveix) || (Int(curveix) != 0)) && return ([],)
    pointix = _cfgget(pt, :pointNumber, nothing)
    isnothing(pointix) && return ([],)
    ix = Int(pointix)
    return ((0 <= ix < nrow(AS.daydf)) ? [ix] : [],)
end

# `style_data_conditional`'s `state: selected` only matches cell click-drag selection, not
# row_selectable rows, so the active row highlight has to be recomputed from row_index instead.
callback!(app, [Output("minute_table", "style_data_conditional")], [Input("minute_table", "selected_rows")]) do selected_rows
    rows = isnothing(selected_rows) ? Int[] : collect(selected_rows)
    return ([Dict("if" => Dict("row_index" => idx), "backgroundColor" => "rgba(255, 210, 110, 0.55)", "border" => "1px solid rgb(200, 140, 0)") for idx in rows],)
end

"Always serve on 127.0.0.1:8060."
function _run_analyze_server(app; host::String="127.0.0.1", port::Int=8060)
    println("$(EnvConfig.now()) starting analyze server on $(host):$(port)")
    run_server(app, host, port, debug=false)
end

_run_analyze_server(app)
