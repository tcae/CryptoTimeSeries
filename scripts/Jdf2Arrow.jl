module Jdf2Arrow

using Dates, DataFrames
using EnvConfig, Ohlcv, Features, Targets, Classify, TradingStrategy

const DEFAULT_ARTIFACTS = String["features", "results", "maxpredictions"]
const DEFAULT_SHARED_ARTIFACTS = String["ohlcv", "f4"]
const DEFAULT_KEYCOLS = Symbol[:coin, :rangeid, :set, :target, :opentime, :rowix]

"""
    _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)

Return the string value of a `key=value` entry from `args`, or `default` if the
key is not present.
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
    _parse_list(raw::Union{Nothing,AbstractString}, default::Vector{String})::Vector{String}

Parse a comma-separated string into a cleaned vector of strings. If `raw` is
`nothing` or empty, return `default`.
"""
function _parse_list(raw::Union{Nothing,AbstractString}, default::Vector{String})::Vector{String}
    if isnothing(raw)
        return copy(default)
    end
    parsed = [strip(part) for part in split(String(raw), ",") if !isempty(strip(part))]
    return isempty(parsed) ? copy(default) : parsed
end

"""
    _parse_bool(raw::Union{Nothing,AbstractString}, default::Bool)::Bool

Parse a common textual boolean value. Returns `default` if `raw` is `nothing`.
"""
function _parse_bool(raw::Union{Nothing,AbstractString}, default::Bool)::Bool
    if isnothing(raw)
        return default
    end
    lowered = lowercase(strip(String(raw)))
    return lowered in ("1", "true", "yes", "y", "on")
end

"""
    _resolve_folderpath(folder::AbstractString)::String

Resolve `folder` to a concrete directory path. Relative folder names are treated
as subfolders of the active `EnvConfig` logs root; absolute paths are used as-is.
"""
function _resolve_folderpath(folder::AbstractString)::String
    return isabspath(folder) ? mkpath(folder) : EnvConfig.setlogpath(folder)
end

"""
    _default_output_stem(artifact::AbstractString)::String

Return the default subfolder-relative Arrow output stem for a monolithic pilot
artifact so the config folder remains uncluttered.
"""
function _default_output_stem(artifact::AbstractString)::String
    name = lowercase(strip(artifact))
    if name == "features"
        return joinpath("features", "all")
    elseif name == "targets"
        return joinpath("targets", "all")
    elseif name == "results"
        return joinpath("results", "all")
    elseif name == "maxpredictions"
        return joinpath("predictions", "maxpredictions")
    elseif name == "predictions"
        return joinpath("predictions", "all")
    elseif name == "trades"
        return joinpath("trades", "all")
    end
    return name
end

"""
    _matching_keycols(df::AbstractDataFrame; candidates::Vector{Symbol}=DEFAULT_KEYCOLS)

Return the subset of `candidates` that is present in `df`.
"""
function _matching_keycols(df::AbstractDataFrame; candidates::Vector{Symbol}=DEFAULT_KEYCOLS)::Vector{Symbol}
    return [col for col in candidates if col in propertynames(df)]
end

_keyvalue_string(value) = ismissing(value) ? missing : string(value)

function _normalized_keyvalues(column)
    T = Base.nonmissingtype(eltype(column))
    if T <: Enum || T <: Integer
        return [ismissing(value) ? missing : Int(value) for value in column]
    end
    return _keyvalue_string.(column)
end

"""
    _assert_conversion_matches(srcdf::AbstractDataFrame, destdf::AbstractDataFrame, artifact::AbstractString;
                               keycols::Vector{Symbol}=DEFAULT_KEYCOLS)

Verify that a converted Arrow table preserves row count, column count, column
names, and selected key columns.
"""
function _assert_conversion_matches(srcdf::AbstractDataFrame, destdf::AbstractDataFrame, artifact::AbstractString;
    keycols::Vector{Symbol}=DEFAULT_KEYCOLS)
    @assert size(srcdf) == size(destdf) "size mismatch after converting $(artifact): size(srcdf)=$(size(srcdf)) size(destdf)=$(size(destdf))"
    @assert names(srcdf) == names(destdf) "column mismatch after converting $(artifact): names(srcdf)=$(names(srcdf)) names(destdf)=$(names(destdf))"

    present = _matching_keycols(srcdf; candidates=keycols)
    for col in present
        @assert _normalized_keyvalues(srcdf[!, col]) == _normalized_keyvalues(destdf[!, col]) "key column $(col) mismatch after converting $(artifact)"
    end
    return present
end

"""
    convert_jdf_artifact(folder::AbstractString, artifact::AbstractString;
                         outputstem::Union{Nothing,AbstractString}=nothing,
                         keycols::Vector{Symbol}=DEFAULT_KEYCOLS)

Convert one JDF artifact inside `folder` to an Arrow copy stored in the default
config-scoped subfolder layout and return a summary named tuple.
"""
function convert_jdf_artifact(folder::AbstractString, artifact::AbstractString;
    outputstem::Union{Nothing,AbstractString}=nothing,
    keycols::Vector{Symbol}=DEFAULT_KEYCOLS)

    folderpath = _resolve_folderpath(folder)
    srcdf = EnvConfig.readtable(artifact; folderpath=folderpath, format=:jdf)
    @assert !isnothing(srcdf) "missing JDF source $(artifact) in $(folderpath)"

    targetstem = isnothing(outputstem) ? _default_output_stem(artifact) : String(outputstem)
    outpath = EnvConfig.savedf(srcdf, targetstem; folderpath=folderpath, format=:arrow)
    destdf = EnvConfig.readtable(targetstem; folderpath=folderpath, format=:arrow)
    @assert !isnothing(destdf) "failed to reload Arrow output $(targetstem) in $(folderpath)"

    verified_keys = _assert_conversion_matches(srcdf, destdf, artifact; keycols=keycols)
    return (
        artifact=String(artifact),
        rows=Int32(size(srcdf, 1)),
        cols=Int32(size(srcdf, 2)),
        output=String(outpath),
        verified_keys=join(string.(verified_keys), ","),
        converted_at=String(Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")),
    )
end

"""
    convert_folder(folder::AbstractString; artifacts::Vector{String}=DEFAULT_ARTIFACTS,
                   savesummary::Bool=true)

Convert the requested JDF `artifacts` inside `folder` to Arrow copies and return
an overview `DataFrame` of the completed conversions.
"""
function convert_folder(folder::AbstractString; artifacts::Vector{String}=DEFAULT_ARTIFACTS, savesummary::Bool=true)
    folderpath = _resolve_folderpath(folder)
    rows = NamedTuple[]
    for artifact in artifacts
        push!(rows, convert_jdf_artifact(folderpath, artifact))
    end
    summarydf = isempty(rows) ? DataFrame() : DataFrame(rows)
    if savesummary && (size(summarydf, 1) > 0)
        EnvConfig.savedf(summarydf, "jdf2arrow_conversion_summary"; folderpath=folderpath, format=:arrow)
    end
    return summarydf
end

"Resolve the root directory used for historical config-folder backfills."
function _resolve_rootpath(root::Union{Nothing,AbstractString}=nothing)::String
    logroot = EnvConfig.setlogpath(nothing)
    if isnothing(root) || isempty(strip(String(root)))
        return logroot
    elseif isabspath(String(root))
        return normpath(String(root))
    end
    return normpath(joinpath(logroot, String(root)))
end

"Return the concrete config folder paths selected for a historical backfill scan."
function _folderpaths_for_backfill(rootpath::AbstractString; folders::Vector{String}=String[])::Vector{String}
    if isempty(folders)
        return [normpath(joinpath(rootpath, entry)) for entry in readdir(rootpath; join=false, sort=true) if isdir(joinpath(rootpath, entry))]
    end
    resolved = String[]
    for folder in folders
        candidate = isabspath(folder) ? normpath(folder) : normpath(joinpath(rootpath, folder))
        isdir(candidate) && push!(resolved, candidate)
    end
    return unique(resolved)
end

"Inspect one config folder and report which requested JDF artifacts still need Arrow backfill copies."
function discover_folder_artifacts(folderpath::AbstractString; artifacts::Vector{String}=DEFAULT_ARTIFACTS)
    rows = NamedTuple[]
    for artifact in artifacts
        targetstem = _default_output_stem(artifact)
        has_jdf = EnvConfig.tableexists(artifact; folderpath=folderpath, format=:jdf)
        has_arrow = EnvConfig.tableexists(targetstem; folderpath=folderpath, format=:arrow)
        if has_jdf || has_arrow
            push!(rows, (
                folder = basename(folderpath),
                folderpath = String(folderpath),
                artifact = String(artifact),
                outputstem = String(targetstem),
                has_jdf = has_jdf,
                has_arrow = has_arrow,
                needs_conversion = has_jdf && !has_arrow,
            ))
        end
    end
    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

"Scan and optionally backfill historical config folders under the logs root."
function backfill_historical(root::Union{Nothing,AbstractString}=nothing; folders::Vector{String}=String[], artifacts::Vector{String}=DEFAULT_ARTIFACTS,
    missingonly::Bool=true, reportonly::Bool=false, savesummary::Bool=true)

    rootpath = _resolve_rootpath(root)
    folderpaths = _folderpaths_for_backfill(rootpath; folders=folders)
    status_parts = DataFrame[]
    for folderpath in folderpaths
        statusdf = discover_folder_artifacts(folderpath; artifacts=artifacts)
        if size(statusdf, 1) > 0
            push!(status_parts, statusdf)
        end
    end
    statusdf = isempty(status_parts) ? DataFrame() : reduce(vcat, status_parts; cols=:union)
    if missingonly && (size(statusdf, 1) > 0)
        statusdf = statusdf[statusdf[!, :needs_conversion], :]
    end

    if reportonly || (size(statusdf, 1) == 0)
        if savesummary && (size(statusdf, 1) > 0)
            EnvConfig.savedf(statusdf, "jdf2arrow_backfill_report"; folderpath=rootpath, format=:arrow)
        end
        return statusdf
    end

    rows = NamedTuple[]
    for row in eachrow(statusdf)
        converted = convert_jdf_artifact(row.folderpath, row.artifact)
        push!(rows, merge(converted, (folder=String(row.folder), outputstem=String(row.outputstem))))
    end
    summarydf = isempty(rows) ? DataFrame() : DataFrame(rows)
    if savesummary && (size(summarydf, 1) > 0)
        EnvConfig.savedf(summarydf, "jdf2arrow_backfill_summary"; folderpath=rootpath, format=:arrow)
    end
    return summarydf
end

_artifact_requested(artifacts::Vector{String}, names::Vararg{String}) = any(lowercase(strip(item)) in names for item in artifacts)

_shared_timestamp() = String(Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))

"""Return the recursive byte size of a file or directory path, or zero if it does not exist."""
function _path_bytes(path::AbstractString)::Int64
    if !ispath(path)
        return Int64(0)
    elseif isfile(path)
        return Int64(filesize(path))
    elseif isdir(path)
        total = Int64(0)
        for (root, _, files) in walkdir(path)
            for file in files
                total += Int64(filesize(joinpath(root, file)))
            end
        end
        return total
    end
    return Int64(0)
end

_shared_pairfolder(basecoin::AbstractString, quotecoin::AbstractString) = uppercase(basecoin) * "-" * uppercase(quotecoin)

function _shared_jdf_sources(subfolder::AbstractString)::Vector{String}
    folderpath = EnvConfig.datafolderpath(subfolder)
    isdir(folderpath) || return String[]
    return [normpath(joinpath(folderpath, entry)) for entry in readdir(folderpath; join=false, sort=true) if endswith(entry, ".jdf")]
end

function _parse_shared_basequote(source::AbstractString)
    stem = replace(basename(String(source)), r"\.jdf$" => "")
    parts = split(stem, "_")
    @assert length(parts) >= 2 "unexpected shared cache filename $(basename(String(source)))"
    return (base=uppercase(parts[1]), quotecoin=uppercase(parts[2]))
end

"""Cross-check one legacy shared OHLCV JDF cache from `crypto/Features/OHLCV` against its Arrow counterpart under `coins/<pair>/ohlcv.arrow`."""
function crosscheck_ohlcv_cache(source::AbstractString)
    parsed = _parse_shared_basequote(source)
    target = normpath(joinpath(EnvConfig.coinspath(), _shared_pairfolder(parsed.base, parsed.quotecoin), "ohlcv.arrow"))
    source_bytes = _path_bytes(source)
    target_bytes = _path_bytes(target)
    exists = isfile(target)
    isempty_source = source_bytes == 0
    status = isempty_source ? "empty-source" : (!exists ? "missing-arrow" : (target_bytes >= source_bytes ? "ok" : "size-mismatch"))
    return (
        scope="shared",
        coin=parsed.base * parsed.quotecoin,
        artifact="ohlcv",
        status=status,
        source=String(source),
        output=target,
        source_rows=missing,
        source_bytes=source_bytes,
        output_files=Int32(exists ? 1 : 0),
        output_bytes=target_bytes,
        counterpart_exists=exists,
        size_ok=isempty_source ? missing : (target_bytes >= source_bytes),
    )
end

"""Cross-check one legacy shared F4 JDF cache from `crypto/Features/Features004` against its single Arrow counterpart under `coins/<pair>/f4.arrow`."""
function crosscheck_f4_cache(source::AbstractString)
    parsed = _parse_shared_basequote(source)
    target = normpath(joinpath(EnvConfig.coinspath(), _shared_pairfolder(parsed.base, parsed.quotecoin), "f4.arrow"))
    output_bytes = _path_bytes(target)
    source_bytes = _path_bytes(source)
    isempty_source = source_bytes == 0
    exists = isfile(target)
    status = isempty_source ? "empty-source" : (!exists ? "missing-arrow" : (output_bytes >= source_bytes ? "ok" : "size-mismatch"))
    return (
        scope="shared",
        coin=parsed.base * parsed.quotecoin,
        artifact="f4",
        status=status,
        source=String(source),
        output=target,
        source_rows=missing,
        source_bytes=source_bytes,
        output_files=Int32(exists ? 1 : 0),
        output_bytes=output_bytes,
        counterpart_exists=exists,
        size_ok=isempty_source ? missing : (output_bytes >= source_bytes),
    )
end

"""Cross-check all legacy shared CloudStorage JDF caches against their Arrow counterparts and summarize coverage plus byte-size parity."""
function crosscheck_shared_data(; bases::Vector{String}=String[], artifacts::Vector{String}=DEFAULT_SHARED_ARTIFACTS, savesummary::Bool=true)
    EnvConfig.init(production)
    selectedbases = Set(uppercase.(bases))
    rows = NamedTuple[]

    if _artifact_requested(artifacts, "ohlcv")
        for source in _shared_jdf_sources("OHLCV")
            parsed = _parse_shared_basequote(source)
            if isempty(selectedbases) || (parsed.base in selectedbases)
                push!(rows, crosscheck_ohlcv_cache(source))
            end
        end
    end

    if _artifact_requested(artifacts, "f4", "features004")
        for source in _shared_jdf_sources("Features004")
            parsed = _parse_shared_basequote(source)
            if isempty(selectedbases) || (parsed.base in selectedbases)
                push!(rows, crosscheck_f4_cache(source))
            end
        end
    end

    summarydf = isempty(rows) ? DataFrame() : sort!(DataFrame(rows), [:artifact, :coin])
    if savesummary && (size(summarydf, 1) > 0)
        EnvConfig.savedf(summarydf, "jdf2arrow_shared_crosscheck"; folderpath=EnvConfig.coinspath(), format=:arrow)
    end
    return summarydf
end

"""Convert one legacy shared OHLCV cache from `crypto/Features/OHLCV` to the local Arrow cache under `coins/<pair>/ohlcv.arrow`."""
function convert_ohlcv_cache(ohlcv::Ohlcv.OhlcvData)
    source = String(Ohlcv.legacyfile(ohlcv).filename)
    rowcount = Int32(size(ohlcv.df, 1))
    colcount = Int32(length(names(ohlcv.df[!, Ohlcv.save_cols])))
    output = Ohlcv.write_arrow(ohlcv)
    isempty_note = rowcount == 0 ? "empty OHLCV cache" : ""
    return (
        scope="shared",
        coin=uppercase(ohlcv.base) * uppercase(ohlcv.quotecoin),
        artifact="ohlcv",
        status=isnothing(output) ? "skipped-empty" : "converted",
        rows=rowcount,
        cols=colcount,
        source=source,
        output=isnothing(output) ? "" : String(output),
        note=isempty_note,
        converted_at=_shared_timestamp(),
    )
end

"""Convert one legacy shared F4 cache from `crypto/Features/Features004` to a single Arrow file under `coins/<pair>/f4.arrow`."""
function convert_f4_cache(f4::Features.Features004)
    source = String(Features.legacyfile(f4).filename)
    rowcount = Int32(isempty(f4.rw) ? 0 : length(Features.opentime(f4)))
    outputs = Features.write_arrow(f4)
    isempty_note = rowcount == 0 ? "empty F4 cache" : ""
    return (
        scope="shared",
        coin=uppercase(f4.basecoin) * uppercase(f4.quotecoin),
        artifact="f4",
        status=isempty(outputs) ? "skipped-empty" : "converted",
        rows=rowcount,
        cols=Int32(isempty(f4.rw) ? 0 : length(names(Features._join(f4)))),
        source=source,
        output=join(outputs, ","),
        note=isempty_note,
        converted_at=_shared_timestamp(),
    )
end

"""
    convert_shared_data(; bases::Vector{String}=String[], artifacts::Vector{String}=DEFAULT_SHARED_ARTIFACTS,
                        savesummary::Bool=true)

Create non-destructive Arrow copies of any remaining legacy shared JDF caches
from `crypto/Features/OHLCV` and `crypto/Features/Features004` under `coins/`
while leaving the original legacy sources untouched.
"""
function convert_shared_data(; bases::Vector{String}=String[], artifacts::Vector{String}=DEFAULT_SHARED_ARTIFACTS, savesummary::Bool=true)
    EnvConfig.init(production)
    selectedbases = Set(uppercase.(bases))
    rows = NamedTuple[]

    if _artifact_requested(artifacts, "ohlcv")
        for source in _shared_jdf_sources("OHLCV")
            parsed = _parse_shared_basequote(source)
            if isempty(selectedbases) || (parsed.base in selectedbases)
                try
                    ohlcv = Ohlcv.defaultohlcv(parsed.base)
                    Ohlcv.read!(ohlcv)
                    push!(rows, convert_ohlcv_cache(ohlcv))
                catch e
                    push!(rows, (
                        scope="shared",
                        coin=parsed.base * parsed.quotecoin,
                        artifact="ohlcv",
                        status="error",
                        rows=Int32(0),
                        cols=Int32(0),
                        source=String(source),
                        output="",
                        note=sprint(showerror, e),
                        converted_at=_shared_timestamp(),
                    ))
                end
            end
        end
    end

    if _artifact_requested(artifacts, "f4", "features004")
        for source in _shared_jdf_sources("Features004")
            parsed = _parse_shared_basequote(source)
            if isempty(selectedbases) || (parsed.base in selectedbases)
                try
                    f4 = Features.Features004(parsed.base, parsed.quotecoin)
                    Features.read!(f4)
                    push!(rows, convert_f4_cache(f4))
                catch e
                    push!(rows, (
                        scope="shared",
                        coin=parsed.base * parsed.quotecoin,
                        artifact="f4",
                        status="error",
                        rows=Int32(0),
                        cols=Int32(0),
                        source=String(source),
                        output="",
                        note=sprint(showerror, e),
                        converted_at=_shared_timestamp(),
                    ))
                end
            end
        end
    end

    summarydf = isempty(rows) ? DataFrame() : DataFrame(rows)
    if savesummary && (size(summarydf, 1) > 0)
        EnvConfig.savedf(summarydf, "jdf2arrow_shared_summary"; folderpath=EnvConfig.coinspath(), format=:arrow)
    end
    return summarydf
end

"""
    main(args::Vector{String}=ARGS)

Command-line entry point.

Example:
```bash
julia --project=. scripts/Jdf2Arrow.jl folder=Trend-009-test
julia --project=. scripts/Jdf2Arrow.jl folder=Trend-009-training artifacts=features,results,maxpredictions
julia --project=. scripts/Jdf2Arrow.jl shared=true
julia --project=. scripts/Jdf2Arrow.jl shared=true artifacts=ohlcv,f4 bases=BTC,ETH
julia --project=. scripts/Jdf2Arrow.jl crosscheck=true artifacts=ohlcv,f4 bases=BTC,ETH savesummary=false
julia --project=. scripts/Jdf2Arrow.jl scan=true reportonly=true missingonly=true
julia --project=. scripts/Jdf2Arrow.jl scan=true folders=Trend-029-test,Bounds-001-test
```
"""
function main(args::Vector{String}=ARGS)
    crosscheck = _parse_bool(_argvalue(args, "crosscheck", nothing), false)
    shared = _parse_bool(_argvalue(args, "shared", nothing), false)
    scan = _parse_bool(_argvalue(args, "scan", nothing), false)
    reportonly = _parse_bool(_argvalue(args, "reportonly", nothing), false)
    missingonly = _parse_bool(_argvalue(args, "missingonly", "true"), true)
    use_shared_artifacts = shared || crosscheck
    artifacts = use_shared_artifacts ? _parse_list(_argvalue(args, "artifacts", join(DEFAULT_SHARED_ARTIFACTS, ",")), DEFAULT_SHARED_ARTIFACTS) : _parse_list(_argvalue(args, "artifacts", join(DEFAULT_ARTIFACTS, ",")), DEFAULT_ARTIFACTS)
    savesummary = _parse_bool(_argvalue(args, "savesummary", "true"), true)

    println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$(args)")
    if crosscheck
        bases = _parse_list(_argvalue(args, "bases", nothing), String[])
        summarydf = crosscheck_shared_data(; bases=bases, artifacts=artifacts, savesummary=savesummary)
        hasstatus = :status in propertynames(summarydf)
        oks = hasstatus ? count(==("ok"), summarydf[!, :status]) : 0
        empties = hasstatus ? count(==("empty-source"), summarydf[!, :status]) : 0
        missing = hasstatus ? count(==("missing-arrow"), summarydf[!, :status]) : 0
        mismatches = hasstatus ? count(==("size-mismatch"), summarydf[!, :status]) : 0
        println("$(EnvConfig.now()) cross-checked $(size(summarydf, 1)) shared artifact row(s) from $(EnvConfig.datafolderpath("OHLCV")) and $(EnvConfig.datafolderpath("Features004")) against $(EnvConfig.coinspath()) (ok=$(oks), empty=$(empties), missing=$(missing), size_mismatch=$(mismatches))")
    elseif shared
        bases = _parse_list(_argvalue(args, "bases", nothing), String[])
        summarydf = convert_shared_data(; bases=bases, artifacts=artifacts, savesummary=savesummary)
        hasstatus = :status in propertynames(summarydf)
        converted = hasstatus ? count(==("converted"), summarydf[!, :status]) : size(summarydf, 1)
        skipped = hasstatus ? count(==("skipped-empty"), summarydf[!, :status]) : 0
        errors = hasstatus ? count(==("error"), summarydf[!, :status]) : 0
        println("$(EnvConfig.now()) converted $(converted) shared artifact row(s) from $(EnvConfig.datafolderpath("OHLCV")) and $(EnvConfig.datafolderpath("Features004")) into $(EnvConfig.coinspath()) (skipped=$(skipped), errors=$(errors))")
    elseif scan
        root = _argvalue(args, "root", nothing)
        folders = _parse_list(_argvalue(args, "folders", nothing), String[])
        summarydf = backfill_historical(root; folders=folders, artifacts=artifacts, missingonly=missingonly, reportonly=reportonly, savesummary=savesummary)
        if reportonly
            println("$(EnvConfig.now()) discovered $(size(summarydf, 1)) historical artifact row(s) under $(_resolve_rootpath(root))")
        else
            println("$(EnvConfig.now()) converted $(size(summarydf, 1)) historical artifact row(s) under $(_resolve_rootpath(root))")
        end
    else
        folder = _argvalue(args, "folder", nothing)
        @assert !isnothing(folder) "missing required argument folder=<config-folder-or-absolute-path>"
        summarydf = convert_folder(String(folder); artifacts=artifacts, savesummary=savesummary)
        println("$(EnvConfig.now()) converted $(size(summarydf, 1)) artifact(s) in $(folder)")
    end
    if size(summarydf, 1) > 0
        show(stdout, MIME("text/plain"), summarydf; allrows=true, allcols=true, truncate=0)
        println()
    end
    return summarydf
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

end # module
