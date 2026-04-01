import Pkg
Pkg.activate(joinpath(@__DIR__, ".."), io=devnull)

using Dates
using DataFrames
using Ohlcv
using TestOhlcv
using Targets

"""
Parse command line arguments.

Arguments:
- `base` (default: `"BTC"`)
- `enddt` as ISO DateTime string, e.g. `"2026-03-29T12:00:00"` (default: latest sample)
"""
function parse_args(args::Vector{String})::NamedTuple
    base = length(args) >= 1 ? uppercase(args[1]) : "SINE"
    enddt = length(args) >= 2 ? DateTime(args[2]) : nothing
    return (base=base, enddt=enddt)
end

"""
Load one 24h slice ending at `enddt` (or latest sample if `nothing`).
"""
function load_24h_slice(base::AbstractString, enddt::Union{DateTime, Nothing})::NamedTuple
    if base in TestOhlcv.testbasecoin()
        ohlcv = TestOhlcv.testohlcv(base, enddt - Hour(24) + Minute(1), enddt)
        df = Ohlcv.dataframe(ohlcv)
        println("ohlcv=$ohlcv df size=$(size(df)) columns=$(names(df))")
        return (ohlcv=ohlcv, startix=1, endix=size(df, 1), startdt=df[1, :opentime], enddt=df[end, :opentime], datasrc="TestOhlcv")
    end
    ohlcv = Ohlcv.read(base)
    df = Ohlcv.dataframe(ohlcv)

    datasrc = "cache"
    if size(df, 1) == 0
        fallback_enddt = isnothing(enddt) ? Dates.now() : enddt
        fallback_startdt = fallback_enddt - Hour(24)
        ohlcv = Ohlcv.testohlcv("BTC", fallback_startdt, fallback_enddt, "1m")
        df = Ohlcv.dataframe(ohlcv)
        datasrc = "synthetic:SINE"
    end

    @assert size(df, 1) > 0 "No OHLCV rows found for base=$(base), and synthetic fallback also yielded no data."

    actual_enddt = isnothing(enddt) ? df[end, :opentime] : enddt
    startdt = actual_enddt - Hour(24)

    startix = Ohlcv.rowix(df[!, :opentime], startdt)
    endix = Ohlcv.rowix(df[!, :opentime], actual_enddt)

    @assert startix < endix "Invalid 24h range for base=$(base): startix=$(startix), endix=$(endix), startdt=$(startdt), enddt=$(actual_enddt)."

    viewohlcv = Ohlcv.ohlcvview(ohlcv, startix:endix)
    return (ohlcv=viewohlcv, startix=startix, endix=endix, startdt=df[startix, :opentime], enddt=df[endix, :opentime], datasrc=datasrc)
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
    end
    return "other"
end

"""
Analyze labels and crosscheck issues and print a concise summary.
"""
function analyze_result(trd::Targets.Trend02, labels::Vector{Targets.TradeLabel}, relgain::Vector{Float32}, issues::Vector{String})::Nothing
    println("Trend02 Summary")
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

    if !isempty(relgain)
        println("- relative gain stats: min=$(minimum(relgain)), mean=$(sum(relgain) / length(relgain)), max=$(maximum(relgain))")
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
    for msg in issues[1:min(10, end)]
        println("  - $(msg)")
    end
end

"""
Run a 24h Trend02 computation and crosscheck analysis.
"""
function main(args::Vector{String})::Nothing
    if length(args) == 0
        args = ["SINE", "2025-08-01T00:01:00"]
    end
    cfg = parse_args(args)
    slice = load_24h_slice(cfg.base, cfg.enddt)

    println("Loaded 24h slice for $(cfg.base)")
    println("- data source: $(slice.datasrc)")
    println("- row range: $(slice.startix):$(slice.endix)")
    println("- time range: $(slice.startdt) -> $(slice.enddt)")

    trd = Targets.Trend02(3, 30, Targets.defaultlabelthresholds)
    Targets.setbase!(trd, slice.ohlcv)

    labels = Vector{Targets.TradeLabel}(trd.df[!, :label])
    relgain = Float32.(trd.df[!, :reldiff])

    checkresult = Targets.crosscheck(trd)
    normalized = normalize_crosscheck_result(checkresult)

    analyze_result(trd, labels, relgain, normalized.issues)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
else
    main(["SINE", "2025-08-01T00:01:00"])
end
