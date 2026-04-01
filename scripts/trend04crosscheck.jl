import Pkg
Pkg.activate(joinpath(@__DIR__, ".."), io=devnull)

using Dates
using DataFrames
using Ohlcv
using TestOhlcv
using Targets, EnvConfig

const SliceArg = Union{Nothing, DateTime, Int}

"""
Parse command line arguments.

Arguments in this order:
- `base` (default: `"BTC"`)
- `startdt` as ISO DateTime string (e.g. `"2026-03-28T12:00:00"`) or row index (e.g. `"24135"`) (optional)
- `enddt` as ISO DateTime string (e.g. `"2026-03-29T12:00:00"`) or row index (e.g. `"24137"`) (optional)

For real cached market data, omitting both dates processes the whole available range.
For TestOhlcv bases, omitted dates fall back to a fixed synthetic example window.
"""
function parse_slice_arg(arg::String)::Union{DateTime, Int}
    parsed_ix = tryparse(Int, arg)
    if !isnothing(parsed_ix)
        return parsed_ix
    end
    return DateTime(arg)
end

function parse_args(args::Vector{String})::NamedTuple
    base = length(args) >= 1 ? uppercase(args[1]) : "SINE"
    startdt = length(args) >= 3 ? parse_slice_arg(args[2]) : nothing
    enddt = length(args) >= 2 ? parse_slice_arg(args[3]) : nothing
    return (base=base, startdt=startdt, enddt=enddt)
end

"""
Load a requested OHLCV range.

Behavior:
- TestOhlcv bases (`SINE`, `DOUBLESINE`) use a fixed synthetic fallback window when
    dates are omitted.
- Real cached market data uses the full cached range when both `startdt` and `enddt`
    are omitted.
- If only `enddt` is provided for real cached data, the range starts at the first
    cached sample.
- If only `startdt` is provided for real cached data, the range ends at the last
    cached sample.
- `startdt` and `enddt` may be either DateTimes or absolute row indices in the
    full cached dataframe.
"""
function resolve_ix(opentimes::AbstractVector{<:DateTime}, arg::SliceArg, default_ix::Int)::Int
    if isnothing(arg)
        return default_ix
    end
    if arg isa Int
        return arg
    end
    return Ohlcv.rowix(opentimes, arg)
end

function load_slice(base::AbstractString, startdt::SliceArg, enddt::SliceArg)::NamedTuple
    if base in TestOhlcv.testbasecoin()
        EnvConfig.init(test)
        test_enddt = enddt isa DateTime ? enddt : DateTime("2025-08-01T09:31:00")
        if enddt isa Int || startdt isa Int
            @assert false "Integer index slicing is not supported for TestOhlcv base=$(base). enddt=$(enddt), startdt=$(startdt)."
        end
        test_startdt = startdt isa DateTime ? startdt : test_enddt - Hour(24) + Minute(1)
        if test_startdt > test_enddt
            @assert false "Invalid datetime range for TestOhlcv base=$(base): startdt=$(test_startdt), enddt=$(test_enddt)."
        end
        ohlcv = TestOhlcv.testohlcv(base, test_startdt, test_enddt)
        df = Ohlcv.dataframe(ohlcv)
        println("ohlcv=$ohlcv df size=$(size(df)) columns=$(names(df))")
        return (ohlcv=ohlcv, startix=1, endix=size(df, 1), startdt=df[1, :opentime], enddt=df[end, :opentime], datasrc="TestOhlcv")
    end
    EnvConfig.init(training)
    ohlcv = Ohlcv.read(base)
    datasrc = "cache"
    df = Ohlcv.dataframe(ohlcv)
    @assert size(df, 1) > 0 " no data found for base=$(base). "
    if isnothing(enddt)
        enddt = df[end, :opentime]
    end
    if isnothing(startdt)
        startdt = df[begin, :opentime]
    end
    startix = resolve_ix(df[!, :opentime], startdt, 1)
    endix = resolve_ix(df[!, :opentime], enddt, size(df, 1))
    @assert 1 <= startix <= size(df, 1) "start index out of bounds for base=$(base): startix=$(startix), valid=1:$(size(df, 1)), startdt=$(startdt)."
    @assert 1 <= endix <= size(df, 1) "end index out of bounds for base=$(base): endix=$(endix), valid=1:$(size(df, 1)), enddt=$(enddt)."
    @assert startix < endix "Invalid range for base=$(base): startix=$(startix), endix=$(endix), startdt=$(startdt), enddt=$(enddt)."

    Ohlcv.timerangecut!(ohlcv, startix, endix)
    return (ohlcv=ohlcv, startix=startix, endix=endix, startdt=df[startix, :opentime], enddt=df[endix, :opentime], datasrc=datasrc)
end

"""
Normalize `Targets.crosscheck` output into `(valid, issues)`.
"""
function normalize_crosscheck_result(checkresult)::NamedTuple
    if checkresult isa AbstractVector{<:AbstractString}
        issues = String.(checkresult)
        return (valid=isempty(issues), issues=issues)
    end

    if checkresult isa NamedTuple
        if haskey(checkresult, :issues)
            issues = String.(checkresult.issues)
            valid = haskey(checkresult, :valid) ? Bool(checkresult.valid) : isempty(issues)
            return (valid=valid, issues=issues)
        end
    end

    return (valid=false, issues=["Unexpected crosscheck return type: $(typeof(checkresult))"])
end

"""
Map a crosscheck issue string to a compact category.
"""
function issue_category(msg::AbstractString)::String
    if occursin("invalid label", msg)
        return "invalid-label"
    elseif occursin("preceded by", msg)
        return "hold-predecessor"
    elseif occursin("violates minwindow", msg)
        return "minwindow"
    elseif occursin("violates threshold", msg)
        return "threshold"
    elseif occursin("violates hold threshold", msg)
        return "hold-threshold"
    elseif occursin("violates buy exclusion", msg)
        return "buy-exclusion"
    elseif occursin("lacks longbuy confirmation", msg) || occursin("lacks shortbuy confirmation", msg)
        return "maxwindow-buy-confirmation"
    elseif occursin("violates maxwindow continuation", msg)
        return "maxwindow-continuation"
    elseif occursin("must start at a segment", msg)
        return "segment-start-extreme"
    elseif occursin("must end at a segment", msg)
        return "segment-end-extreme"
    elseif occursin("adjacent", msg) && occursin("same type", msg)
        return "segment-extension"
    elseif occursin("info", msg)
        return "info"
    end
    return "other"
end

"""
Analyze labels and crosscheck issues and print a concise summary.
"""
function analyze_result(trd::Targets.Trend04, labels::Vector{Targets.TradeLabel}, issues::Vector{String})::Nothing
    println("Trend04 Summary")
    println("- samples: $(length(labels))")
    println("- minwindow: $(trd.minwindow), maxwindow: $(trd.maxwindow)")
    println("- thresholds: $(Targets.thresholds(trd.thres))")

    unique_labels = unique(labels)
    println("- label distribution:")
    for lbl in unique_labels
        cnt = count(==(lbl), labels)
        pct = round(100 * cnt / length(labels), digits=2)
        println("  - $(lbl): $(cnt) ($(pct)%)")
    end

    if isempty(issues)
        println("Crosscheck: valid (no issues)")
        return
    end

    println("Crosscheck: invalid ($(length(issues)) issues)")

    category_counts = Dict{String, Int}()
    for msg in issues
        cat = issue_category(msg)
        category_counts[cat] = get(category_counts, cat, 0) + 1
    end

    println("- issue categories:")
    for (cat, cnt) in sort(collect(category_counts), by=x -> x[1])
        println("  - $(cat): $(cnt)")
    end

    println("- first issues:")
    for msg in issues[1:min(100, end)]
        println("  - $(msg)")
    end
end

"""
Run a Trend04 computation and crosscheck analysis on the requested or default range.
"""
function main(args::Vector)::Nothing
    if length(args) == 0
        args = ["SINE"]
    end
    Targets.verbosity = 3
    cfg = parse_args(args)
    slice = load_slice(cfg.base, cfg.startdt, cfg.enddt)

    println("$(EnvConfig.now()) Loaded samples for $(cfg.base)")
    println("- data source: $(slice.datasrc)")
    println("- row range: $(slice.startix):$(slice.endix)")
    println("- time range: $(slice.startdt) -> $(slice.enddt)")

    if cfg.base in TestOhlcv.testbasecoin()
        trd = Targets.Trend04(10, 4 * 60, Targets.LabelThresholds(0.1, 0.01, -0.01, -0.1))
    else
        trd = Targets.Trend04(10, 4 * 60, Targets.LabelThresholds(0.01, 0.005, -0.005, -0.01))
    end
    Targets.setbase!(trd, slice.ohlcv)

    labels = Vector{Targets.TradeLabel}(trd.df[!, :label])
    println("$(EnvConfig.now()) Targets calculated")

    checkresult = Targets.crosscheck(trd)
    println("$(EnvConfig.now()) Crosscheck completed. Normalizing results...")
    normalized = normalize_crosscheck_result(checkresult)

    println("$(EnvConfig.now()) Analysis:")
    analyze_result(trd, labels, normalized.issues)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
else
    main([])
end
