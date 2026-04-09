import Pkg
Pkg.activate(joinpath(@__DIR__, ".."), io=devnull)

using Dates
using DataFrames
using Statistics
using Logging
import PlotlyJS
import PlotlyJS: Plot, Layout, attr, scatter, candlestick, bar, heatmap, savefig
using EnvConfig, Ohlcv, TestOhlcv, Features, Targets, Classify, TradingStrategy

include("optimizationconfigs.jl")

const LABEL_ROWS_DEFAULT = ["trend target", "trend pred", "tradepairs target", "lstm pred"]
const LABEL_CODE = Dict{Any, Int}(
    missing => 0,
    Targets.shortbuy => 1,
    Targets.shorthold => 2,
    Targets.shortclose => 3,
    Targets.allclose => 4,
    Targets.longbuy => 5,
    Targets.longhold => 6,
    Targets.longclose => 7,
)
const LABEL_COLOR = Dict{Any, String}(
    missing => "#808080",
    Targets.shortbuy => "#99000d",
    Targets.shorthold => "#ef3b2c",
    Targets.shortclose => "#fcbba1",
    Targets.allclose => "#ffffff",
    Targets.longbuy => "#006d2c",
    Targets.longhold => "#31a354",
    Targets.longclose => "#a1d99b",
)
const LABEL_TICK_TEXT = [
    "missing",
    "shortbuy",
    "shorthold",
    "shortclose",
    "allclose",
    "longbuy",
    "longhold",
    "longclose",
]
const BOUNDS_STATUS_COLOR = Dict(
    "within" => "#2ca02c",
    "late" => "#ffbf00",
    "opposite" => "#d62728",
    "pending" => "#9aa0a6",
)

"""
    _argvalue(args, key, default=nothing)

Return the `key=value` CLI argument value or `default` when absent.
"""
function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)
    prefix = key * "="
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "="; limit=2)[2]
        end
    end
    return default
end

"""
    _parsesource(raw) -> Symbol

Normalize the data-source selector to one of `:auto`, `:test`, or `:cache`.
"""
function _parsesource(raw::AbstractString)::Symbol
    value = Symbol(lowercase(strip(raw)))
    @assert value in (:auto, :test, :cache) "source=$(raw) must be one of auto, test, cache"
    return value
end

"""
    _parsedatetime(raw)

Parse an ISO datetime string into `DateTime`.
"""
_parsedatetime(raw::AbstractString) = DateTime(strip(raw))

"""
    _resolve_trendconfig(ref)

Resolve a trend config reference like `"025"` or `"mk025"` to the corresponding
configuration NamedTuple from `optimizationconfigs.jl`.
"""
function _resolve_trendconfig(ref::AbstractString)
    raw = lowercase(replace(strip(ref), r"config$" => ""))
    symbol = startswith(raw, "mk") ? Symbol(raw * "config") : Symbol("mk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown trend config ref=$(ref); expected $(symbol) in optimizationconfigs.jl"
    return getfield(@__MODULE__, symbol)()
end

"""
    _resolve_boundsconfig(ref)

Resolve a bounds config reference like `"001"` or `"boundsmk001"`.
"""
function _resolve_boundsconfig(ref::AbstractString)
    raw = lowercase(replace(strip(ref), r"config$" => ""))
    symbol = startswith(raw, "boundsmk") ? Symbol(raw * "config") : startswith(raw, "mk") ? Symbol("bounds" * raw * "config") : Symbol("boundsmk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown bounds config ref=$(ref); expected $(symbol) in optimizationconfigs.jl"
    return getfield(@__MODULE__, symbol)()
end

"""
    _resolve_tradeadviceconfig(ref)

Resolve a trade-advice config reference like `"025"`.
"""
function _resolve_tradeadviceconfig(ref::AbstractString)
    raw = lowercase(replace(strip(ref), r"config$" => ""))
    symbol = startswith(raw, "tradeadvicemk") ? Symbol(raw * "config") : startswith(raw, "mk") ? Symbol("tradeadvice" * raw * "config") : Symbol("tradeadvicemk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown tradeadvice config ref=$(ref); expected $(symbol) in optimizationconfigs.jl"
    return getfield(@__MODULE__, symbol)()
end

"""
    parse_args(args) -> NamedTuple

Parse command-line arguments for `simulationcheck.jl`.

Examples:
- `julia --project=. scripts/simulationcheck.jl base=SINE source=test`
- `julia --project=. scripts/simulationcheck.jl base=BTC source=cache enddt=2025-08-10T15:00:00`
"""
function parse_args(args::Vector{String})::NamedTuple
    base = uppercase(_argvalue(args, "base", "SINE"))
    source = _parsesource(_argvalue(args, "source", "auto"))
    periodhours = parse(Int, _argvalue(args, "periodhours", "4"))
    @assert periodhours > 0 "periodhours=$(periodhours) must be > 0"
    startdt_raw = _argvalue(args, "startdt", nothing)
    enddt_raw = _argvalue(args, "enddt", nothing)
    trendref = _argvalue(args, "trend", "025")
    boundsref = _argvalue(args, "bounds", "001")
    tradeadviceref = _argvalue(args, "tradeadvice", "025")
    outfile = _argvalue(args, "outfile", nothing)
    inspectonly = "inspect" in args
    return (
        base=base,
        source=source,
        period=Hour(periodhours),
        startdt=isnothing(startdt_raw) ? nothing : _parsedatetime(startdt_raw),
        enddt=isnothing(enddt_raw) ? nothing : _parsedatetime(enddt_raw),
        trendref=trendref,
        boundsref=boundsref,
        tradeadviceref=tradeadviceref,
        outfile=outfile,
        inspectonly=inspectonly,
    )
end

"""
    _with_log_subfolder(folder, f)

Temporarily switch the active `EnvConfig` log folder for loading cached models/data.
"""
function _with_log_subfolder(folder::AbstractString, f::Function)
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

_with_log_subfolder(f::Function, folder::AbstractString) = _with_log_subfolder(folder, f)

"""
    load_slice(cfg, history_minutes=0) -> NamedTuple

Load a 1-minute OHLCV slice either from `TestOhlcv` synthetic patterns or from the
cached real-data store. The returned `ohlcv` includes the requested history padding,
while `df` is restricted to the visible display interval.
"""
function load_slice(cfg::NamedTuple, history_minutes::Int=0)::NamedTuple
    use_test = (cfg.source == :test) || ((cfg.source == :auto) && (cfg.base in TestOhlcv.testbasecoin()))
    if use_test
        EnvConfig.init(test)
        view_enddt = isnothing(cfg.enddt) ? DateTime("2025-08-01T09:31:00") : cfg.enddt
        view_startdt = isnothing(cfg.startdt) ? view_enddt - cfg.period + Minute(1) : cfg.startdt
        calc_startdt = view_startdt - Minute(max(history_minutes, 0))
        @assert calc_startdt <= view_enddt "invalid test slice with calc_startdt=$(calc_startdt) > enddt=$(view_enddt)"
        ohlcv = TestOhlcv.testohlcv(cfg.base, calc_startdt, view_enddt)
        calcdf = Ohlcv.dataframe(ohlcv)
        plotdf = calcdf[(view_startdt .<= calcdf[!, :opentime] .<= view_enddt), :]
        @assert size(plotdf, 1) > 0 "empty visible test slice for base=$(cfg.base) view_startdt=$(view_startdt) view_enddt=$(view_enddt)"
        return (ohlcv=ohlcv, df=plotdf, startdt=plotdf[begin, :opentime], enddt=plotdf[end, :opentime], calcstartdt=calcdf[begin, :opentime], datasrc="TestOhlcv", mode=EnvConfig.test)
    end

    EnvConfig.init(training)
    ohlcv = Ohlcv.read(cfg.base)
    fulldf = Ohlcv.dataframe(ohlcv)
    @assert size(fulldf, 1) > 0 "no cached OHLCV rows found for base=$(cfg.base)"
    view_enddt = isnothing(cfg.enddt) ? fulldf[end, :opentime] : cfg.enddt
    view_startdt = isnothing(cfg.startdt) ? max(fulldf[begin, :opentime], view_enddt - cfg.period + Minute(1)) : cfg.startdt
    calc_startdt = max(fulldf[begin, :opentime], view_startdt - Minute(max(history_minutes, 0)))
    startix = Ohlcv.rowix(fulldf[!, :opentime], calc_startdt)
    endix = Ohlcv.rowix(fulldf[!, :opentime], view_enddt)
    @assert startix <= endix "invalid cache slice for $(cfg.base): startix=$(startix) endix=$(endix)"
    slice = Ohlcv.ohlcvview(ohlcv, startix:endix)
    calcdf = Ohlcv.dataframe(slice)
    plotdf = calcdf[(view_startdt .<= calcdf[!, :opentime] .<= view_enddt), :]
    @assert size(plotdf, 1) > 0 "empty visible cache slice for base=$(cfg.base) view_startdt=$(view_startdt) view_enddt=$(view_enddt)"
    return (ohlcv=slice, df=plotdf, startdt=plotdf[begin, :opentime], enddt=plotdf[end, :opentime], calcstartdt=calcdf[begin, :opentime], datasrc="cache", mode=EnvConfig.training)
end

"""
    _discrete_colorscale() -> Vector

Create a Plotly discrete colorscale matching the `TradeLabel` color mapping.
"""
function _discrete_colorscale()
    ordered = [missing, Targets.shortbuy, Targets.shorthold, Targets.shortclose, Targets.allclose, Targets.longbuy, Targets.longhold, Targets.longclose]
    boundaries = collect(range(0.0, 1.0; length=length(ordered) + 1))
    scale = Vector{Any}()
    for (ix, key) in enumerate(ordered)
        color = LABEL_COLOR[key]
        push!(scale, Any[boundaries[ix], color])
        push!(scale, Any[boundaries[ix + 1], color])
    end
    return scale
end

"""
    _safe_tradelabel(value)

Normalize strings / `TradeLabel`s / `missing` into a single label-like representation.
"""
function _safe_tradelabel(value)
    if ismissing(value)
        return missing
    elseif value isa Targets.TradeLabel
        return value
    else
        return Targets.tradelabel(String(value))
    end
end

"""
    _label_code(value) -> Int

Return the integer heatmap code for a label-like value.
"""
_label_code(value) = get(LABEL_CODE, _safe_tradelabel(value), 0)

"""
    _load_trend_classifier(cfg, mode)

Load the cached trend-classifier neural network for the requested config, or return
`nothing` when no trained model is available.
"""
function _load_trend_classifier(cfg::NamedTuple, mode::EnvConfig.Mode)
    folder = "Trend-$(cfg.configname)-$(mode)"
    return _with_log_subfolder(folder) do
        nn = cfg.classifiermodel(Features.featurecount(cfg.featconfig), Targets.uniquelabels(cfg.targetconfig), "mix")
        return isfile(Classify.nnfilename(nn.fileprefix)) ? Classify.loadnn(nn.fileprefix) : nothing
    end
end

"""
    _load_bounds_regressor(cfg, mode)

Load the cached bounds regressor for the requested config, or return `nothing` when no
trained model is available.
"""
function _load_bounds_regressor(cfg::NamedTuple, mode::EnvConfig.Mode)
    folder = "Bounds-$(cfg.configname)-$(mode)"
    return _with_log_subfolder(folder) do
        nn = cfg.regressormodel(Features.featurecount(cfg.featconfig), ["center", "width"], "mix_relative_v1")
        return isfile(Classify.nnfilename(nn.fileprefix)) ? Classify.loadnn(nn.fileprefix) : nothing
    end
end

"""
    compute_trend_overlay(slice, trendcfg) -> DataFrame

Compute Trend04 targets on the selected slice and, when available, overlay the cached
trend-classifier predictions.
"""
function compute_trend_overlay(slice::NamedTuple, trendcfg::NamedTuple)::DataFrame
    featcfg = trendcfg.featconfig
    targetcfg = trendcfg.targetconfig
    Features.setbase!(featcfg, slice.ohlcv, usecache=false)
    Targets.setbase!(targetcfg, slice.ohlcv)

    calcdf = Ohlcv.dataframe(slice.ohlcv)
    outdf = DataFrame(
        opentime=calcdf[!, :opentime],
        pivot=Float32.(calcdf[!, :pivot]),
        trend_target=collect(Targets.labels(targetcfg)),
        trend_pred=Vector{Any}(fill(missing, size(calcdf, 1))),
        trend_score=Vector{Union{Missing, Float32}}(fill(missing, size(calcdf, 1))),
    )

    nn = _load_trend_classifier(trendcfg, slice.mode)
    if !isnothing(nn)
        reqcols = Features.requestedcolumns(featcfg)
        featdf = Features.features(featcfg)
        X = Float32.(permutedims(Matrix(featdf[!, reqcols]), (2, 1)))
        probsdf = Classify.predictdf(nn, X)
        maxdf = Classify.maxpredictdf(nn, X)
        preddf = DataFrame(
            opentime=Features.opentime(featcfg),
            trend_pred=collect(maxdf[!, :label]),
            trend_score=Float32.(maxdf[!, :score]),
        )
        for col in names(probsdf)
            preddf[!, Symbol(col)] = Float32.(probsdf[!, col])
        end
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

"""
    compute_tradepair_targets(trenddf, tradecfg, trendcfg) -> Vector{TradeLabel}

Convert dense Trend04 labels into sparse trade-pair lifecycle labels for the same slice.
"""
function compute_tradepair_targets(trenddf::AbstractDataFrame, tradecfg::NamedTuple, trendcfg::NamedTuple)
    tp = Targets.TradePairs(trendcfg.targetconfig; entryfraction=tradecfg.entryfraction, exitfraction=tradecfg.exitfraction)
    return Targets.tradepairlabels(tp, trenddf[!, :trend_target], Float32.(trenddf[!, :pivot]))
end

"""
    _first_hit(rangeix, predicate)

Return the first index in `rangeix` that satisfies `predicate`, or `nothing`.
"""
function _first_hit(rangeix, predicate)
    for ix in rangeix
        predicate(ix) && return ix
    end
    return nothing
end

"""
    _denormalize_bounds(centerpred, widthpred, pivot) -> Tuple

Convert pivot-relative center/width predictions into absolute lower/upper prices.
"""
function _denormalize_bounds(centerpred::AbstractVector{<:Real}, widthpred::AbstractVector{<:Real}, pivot::AbstractVector{<:Real})
    p = Float32.(pivot)
    centerabs = (1f0 .+ Float32.(centerpred)) .* p
    widthabs = Float32.(widthpred) .* p
    lowerabs = clamp.(centerabs .- widthabs ./ 2f0, 0f0, Inf32)
    upperabs = clamp.(centerabs .+ widthabs ./ 2f0, 0f0, Inf32)
    return lowerabs, upperabs
end

"""
    _bound_status(df, predlow, predhigh, ix, window, side) -> NamedTuple

Classify a predicted upper/lower bound sample as:
- `within`: exceeded within the target window
- `late`: exceeded only after the window
- `opposite`: opposite-side bound was hit first
- `pending`: neither event occurred in the visible slice
"""
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

"""
    compute_bounds_overlay(slice, boundscfg) -> Union{Nothing,DataFrame}

Compute predicted upper/lower bounds and their per-sample outcome classes for the
selected slice. The returned dataframe is aligned by `:opentime`.
"""
function compute_bounds_overlay(slice::NamedTuple, boundscfg::NamedTuple)::Union{Nothing, DataFrame}
    featcfg = boundscfg.featconfig
    targetcfg = boundscfg.targetconfig
    Features.setbase!(featcfg, slice.ohlcv, usecache=false)
    Targets.setbase!(targetcfg, slice.ohlcv)
    featdf = Features.features(featcfg)
    reqcols = Features.requestedcolumns(featcfg)
    X = Float32.(permutedims(Matrix(featdf[!, reqcols]), (2, 1)))

    nn = _load_bounds_regressor(boundscfg, slice.mode)
    if isnothing(nn)
        return nothing
    end

    yraw = nn.model(X)
    centerpred = vec(Float32.(yraw[1, :]))
    widthpred = vec(clamp.(Float32.(yraw[2, :]), 0f0, Inf32))
    calcdf = Ohlcv.dataframe(slice.ohlcv)
    featuretimes = Features.opentime(featcfg)
    pivotmap = Dict(calcdf[ix, :opentime] => Float32(calcdf[ix, :pivot]) for ix in 1:size(calcdf, 1))
    predpivot = Float32[get(pivotmap, ts, 0f0) for ts in featuretimes]
    predlow, predhigh = _denormalize_bounds(centerpred, widthpred, predpivot)

    outdf = DataFrame(opentime=featuretimes, pred_low=Float32.(predlow), pred_high=Float32.(predhigh))
    outdf = leftjoin(outdf, select(calcdf, [:opentime, :high, :low, :close, :pivot]), on=:opentime)
    if (:lowtarget in propertynames(targetcfg.df)) && (:hightarget in propertynames(targetcfg.df))
        targetsubset = select(targetcfg.df, [:opentime, :lowtarget, :hightarget])
        rename!(targetsubset, :lowtarget => :target_low, :hightarget => :target_high)
        outdf = leftjoin(outdf, targetsubset, on=:opentime)
    else
        outdf[!, :target_low] = Float32.(predlow)
        outdf[!, :target_high] = Float32.(predhigh)
    end

    window = hasproperty(targetcfg, :window) ? Int(getproperty(targetcfg, :window)) : 0
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

"""
    load_lstm_overlay(slice, tradecfg) -> Union{Nothing,DataFrame}

Load cached LSTM predictions when available. The script checks both the dedicated
trade-advice subfolder and the root log folder because earlier runs saved to both.
"""
function load_lstm_overlay(slice::NamedTuple, tradecfg::NamedTuple)::Union{Nothing, DataFrame}
    EnvConfig.setlogpath()
    root = EnvConfig.logfolder()
    candidates = [joinpath(root, "TradeAdviceLstm-$(tradecfg.configname)-$(slice.mode)"), root]
    for folderpath in candidates
        df = EnvConfig.readdf("lstm_predictions.jdf"; folderpath=folderpath)
        if !isnothing(df) && size(df, 1) > 0 && (:opentime in propertynames(df))
            mask = (string.(df[!, :coin]) .== slice.ohlcv.base) .& (slice.startdt .<= df[!, :opentime] .<= slice.enddt)
            subdf = DataFrame(df[mask, :])
            if size(subdf, 1) > 0
                sort!(subdf, :opentime)
                return subdf
            end
        end
    end
    return nothing
end

"""
    build_heatmap_panel(df, trade_targets, lstmdf) -> Vector{PlotlyJS.AbstractTrace}

Create the upper-right classification heatmap for trend targets, trend predictions,
trade-pair targets, and cached LSTM predictions, plus transparent hover overlays so
unified hover shows the textual class details reliably.
"""
function build_heatmap_panel(trenddf::AbstractDataFrame, trade_targets, lstmdf::Union{Nothing, AbstractDataFrame})
    n = size(trenddf, 1)
    x = trenddf[!, :opentime]
    rows = LABEL_ROWS_DEFAULT
    z = zeros(Int, length(rows), n)
    hovertext = fill("", length(rows), n)

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
        end
    end

    heattrace = heatmap(
        x=x,
        y=rows,
        z=z,
        hoverinfo="skip",
        hoverongaps=false,
        zmin=-0.5,
        zmax=7.5,
        zsmooth=false,
        xgap=1,
        ygap=1,
        colorscale=_discrete_colorscale(),
        showscale=true,
        colorbar=attr(
            title="class labels",
            tickmode="array",
            tickvals=collect(0:7),
            ticktext=LABEL_TICK_TEXT,
            len=0.24,
            y=0.88,
            x=1.03,
            xanchor="left",
            thickness=16,
            xpad=0,
        ),
        yaxis="y4",
        name="classifications",
        showlegend=false,
    )

    hovertraces = PlotlyJS.AbstractTrace[heattrace]
    for rowix in length(rows):-1:1
        rowname = rows[rowix]
        push!(hovertraces,
            scatter(
                x=x,
                y=fill(rowname, n),
                mode="markers",
                text=collect(hovertext[rowix, :]),
                hoverinfo="text",
                marker=attr(symbol="square", size=16, color="rgba(0,0,0,0.01)", line=attr(width=0)),
                yaxis="y4",
                name=rowname,
                showlegend=false,
            )
        )
    end
    return hovertraces
end

"""
    _status_line_trace(x, y, statusvec, delayvec, category, name; width=2)

Create one colored bounds trace for a single status category.
"""
function _status_line_trace(x, y, statusvec, delayvec, category::AbstractString, name::AbstractString; width::Int=2)
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
    return scatter(
        x=x,
        y=ycat,
        mode="lines",
        name="$(name) $(category)",
        text=text,
        hoverinfo="text",
        line=attr(color=BOUNDS_STATUS_COLOR[category], width=width),
    )
end

"""
    build_simulation_figure(slice, trenddf, trade_targets, boundsdf, lstmdf) -> Plot

Create the Plotly diagnostic figure modeled after the 4h cockpit view: candlesticks
and bounds in the main panel, classification heatmap in the upper-right panel.
"""
function build_simulation_figure(slice::NamedTuple, trenddf::AbstractDataFrame, trade_targets, boundsdf::Union{Nothing, AbstractDataFrame}, lstmdf::Union{Nothing, AbstractDataFrame})
    df = slice.df
    traces = PlotlyJS.AbstractTrace[]

    push!(traces,
        candlestick(
            x=df[!, :opentime],
            open=df[!, :open],
            high=df[!, :high],
            low=df[!, :low],
            close=df[!, :close],
            name="$(slice.ohlcv.base) OHLC",
            yaxis="y",
            showlegend=false,
        )
    )
    push!(traces,
        bar(
            x=df[!, :opentime],
            y=df[!, :basevolume],
            name="volume",
            yaxis="y2",
            marker=attr(color="rgba(120, 120, 120, 0.35)"),
            showlegend=false,
        )
    )

    if !isnothing(boundsdf)
        x = boundsdf[!, :opentime]
        push!(traces, scatter(x=x, y=boundsdf[!, :target_high], mode="lines", name="target high", line=attr(color="rgba(180,180,180,0.6)", dash="dot", width=1)))
        push!(traces, scatter(x=x, y=boundsdf[!, :target_low], mode="lines", name="target low", line=attr(color="rgba(180,180,180,0.6)", dash="dot", width=1)))
        for category in ["within", "late", "opposite", "pending"]
            push!(traces, _status_line_trace(x, boundsdf[!, :pred_high], boundsdf[!, :highstatus], boundsdf[!, :highdelay], category, "pred high"; width=2))
            push!(traces, _status_line_trace(x, boundsdf[!, :pred_low], boundsdf[!, :lowstatus], boundsdf[!, :lowdelay], category, "pred low"; width=2))
        end
    end

    append!(traces, build_heatmap_panel(trenddf, trade_targets, lstmdf))

    title = "SimulationCheck $(slice.ohlcv.base) ($(slice.datasrc))  $(slice.startdt) → $(slice.enddt)"
    layout = Layout(
        title=title,
        xaxis=attr(rangeslider=attr(visible=false)),
        yaxis=attr(title="price", domain=[0.20, 0.72]),
        yaxis2=attr(title="volume", domain=[0.0, 0.14]),
        yaxis4=attr(title="classifications", domain=[0.76, 1.0], automargin=true),
        legend=attr(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=0.72,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(160,160,160,0.7)",
            borderwidth=1,
        ),
        margin=attr(l=60, r=280, t=70, b=40),
        hovermode="x unified",
    )
    return Plot(traces, layout)
end

"""
    summary_text(slice, trenddf, boundsdf, lstmdf)

Return a short human-readable summary of the generated overlays.
"""
function summary_text(slice::NamedTuple, trenddf::AbstractDataFrame, boundsdf::Union{Nothing, AbstractDataFrame}, lstmdf::Union{Nothing, AbstractDataFrame})::String
    trenddist = Dict(string(lbl) => count(==(lbl), trenddf[!, :trend_target]) for lbl in unique(trenddf[!, :trend_target]))
    parts = [
        "data=$(slice.datasrc)",
        "rows=$(size(slice.df, 1))",
        "trend_target=$(trenddist)",
        "lstm_cached=$(isnothing(lstmdf) ? "no" : "yes")",
    ]
    if !isnothing(boundsdf)
        push!(parts, "bounds_high=$(combine(groupby(boundsdf, :highstatus), nrow => :rows))")
        push!(parts, "bounds_low=$(combine(groupby(boundsdf, :lowstatus), nrow => :rows))")
    else
        push!(parts, "bounds=missing-model")
    end
    return join(parts, "\n")
end

"""
    save_plot(fig, outfile)

Attempt to write the Plotly figure to an HTML file. The save is best-effort so the
script still succeeds when the HTML backend is unavailable.
"""
function save_plot(fig, outfile::AbstractString)
    mkpath(dirname(outfile))
    try
        savefig(fig, outfile)
        println("saved figure to $(outfile)")
    catch err
        @warn "failed to save Plotly figure to $(outfile)" exception=(err, catch_backtrace())
    end
end

"""
    required_history_minutes(trendcfg, boundscfg) -> Int

Estimate how much historical padding should be loaded before the visible window so the
feature and target generators can work on a short 4h inspection slice.
"""
function required_history_minutes(trendcfg::NamedTuple, boundscfg::NamedTuple)::Int
    minutes = 0
    minutes = max(minutes, Int(Features.requiredminutes(trendcfg.featconfig)))
    minutes = max(minutes, Int(Features.requiredminutes(boundscfg.featconfig)))
    if hasproperty(trendcfg.targetconfig, :maxwindow)
        minutes = max(minutes, Int(getproperty(trendcfg.targetconfig, :maxwindow)))
    end
    if hasproperty(boundscfg.targetconfig, :window)
        minutes = max(minutes, Int(getproperty(boundscfg.targetconfig, :window)))
    end
    return minutes
end

"""
    main(args=ARGS)

Entry point for the simulation diagnostic script.
"""
function main(args::Vector{String}=ARGS)
    cfg = parse_args(args)
    println("$(EnvConfig.now()) simulationcheck.jl ARGS=$(args)")

    trendcfg = _resolve_trendconfig(cfg.trendref)
    boundscfg = _resolve_boundsconfig(cfg.boundsref)
    tradecfg = _resolve_tradeadviceconfig(cfg.tradeadviceref)
    history_minutes = required_history_minutes(trendcfg, boundscfg)
    slice = load_slice(cfg, history_minutes)

    trenddf = compute_trend_overlay(slice, trendcfg)
    trade_targets = compute_tradepair_targets(trenddf, tradecfg, trendcfg)
    boundsdf = compute_bounds_overlay(slice, boundscfg)
    lstmdf = load_lstm_overlay(slice, tradecfg)

    fig = build_simulation_figure(slice, trenddf, trade_targets, boundsdf, lstmdf)
    summary = summary_text(slice, trenddf, boundsdf, lstmdf)
    println(summary)

    outpath = isnothing(cfg.outfile) ? begin
        EnvConfig.setlogpath("SimulationCheck")
        joinpath(EnvConfig.logfolder(), "simulationcheck_$(lowercase(slice.ohlcv.base))_$(lowercase(slice.datasrc))_$(Dates.format(slice.enddt, "yyyymmdd_HHMM")).html")
    end : cfg.outfile

    if !cfg.inspectonly
        save_plot(fig, outpath)
        display(fig)
    end

    println("done @ $(outpath)")
    return (cfg=cfg, slice=slice, trenddf=trenddf, trade_targets=trade_targets, boundsdf=boundsdf, lstmdf=lstmdf, fig=fig, outfile=outpath)
end

main()
