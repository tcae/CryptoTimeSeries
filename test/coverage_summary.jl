using Dates
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SUMMARY_PATH = joinpath(@__DIR__, "coverage_summary.md")
const LCOV_PATH = joinpath(@__DIR__, "coverage", "latest", "lcov.info")

function _coverage_ratio(covered::Int, total::Int)::Float64
    total == 0 && return 0.0
    return covered / total
end

function _parse_cov_file(path::AbstractString)::Tuple{Int, Int}
    executable = 0
    covered = 0
    open(path, "r") do io
        for raw in eachline(io)
            s = lstrip(raw)
            isempty(s) && continue
            token = split(s; limit=2)[1]
            token == "-" && continue
            count = tryparse(Int, token)
            isnothing(count) && continue
            executable += 1
            if count > 0
                covered += 1
            end
        end
    end
    return executable, covered
end

function _source_relpath(path::AbstractString)::Union{Nothing, String}
    full = isabspath(path) ? normpath(path) : normpath(joinpath(ROOT, path))
    startswith(full, ROOT) || return nothing
    return relpath(full, ROOT)
end

function _parse_lcov_file(path::AbstractString)::Dict{String, Tuple{Int, Int}}
    per_line = Dict{String, Dict{Int, Int}}()
    current_source = nothing

    open(path, "r") do io
        for raw in eachline(io)
            startswith(raw, "SF:") && begin
                src = strip(raw[4:end])
                current_source = _source_relpath(src)
                if !isnothing(current_source) && !haskey(per_line, current_source)
                    per_line[current_source] = Dict{Int, Int}()
                end
                continue
            end

            startswith(raw, "DA:") || continue
            isnothing(current_source) && continue

            parts = split(strip(raw[4:end]), ",")
            length(parts) >= 2 || continue
            line_no = tryparse(Int, parts[1])
            hits = tryparse(Int, parts[2])
            (isnothing(line_no) || isnothing(hits)) && continue

            line_hits = per_line[current_source]
            line_hits[line_no] = get(line_hits, line_no, 0) + hits
        end
    end

    parsed = Dict{String, Tuple{Int, Int}}()
    for (source, line_hits) in per_line
        executable = length(line_hits)
        covered = count(>(0), values(line_hits))
        parsed[source] = (executable, covered)
    end
    return parsed
end

function _logical_source_path(covpath::AbstractString)::String
    rel = relpath(covpath, ROOT)
    return replace(rel, r"\.\d+\.cov$" => "")
end

function _collect_latest_cov_files()::Dict{String, String}
    latest = Dict{String, Tuple{String, Float64}}()
    for (dir, _, files) in walkdir(ROOT)
        occursin("/.git", dir) && continue
        for file in files
            endswith(file, ".cov") || continue
            full = joinpath(dir, file)
            logical = _logical_source_path(full)
            mtime = stat(full).mtime
            prev = get(latest, logical, ("", -1.0))
            if mtime > prev[2]
                latest[logical] = (full, mtime)
            end
        end
    end
    return Dict(k => v[1] for (k, v) in latest)
end

function _collect_per_file_coverage()::Vector{NamedTuple{(:source, :package, :executable, :covered, :ratio), Tuple{String, String, Int, Int, Float64}}}
    rows = Vector{NamedTuple{(:source, :package, :executable, :covered, :ratio), Tuple{String, String, Int, Int, Float64}}}()

    if isfile(LCOV_PATH)
        for (source, (executable, covered)) in sort(collect(_parse_lcov_file(LCOV_PATH)); by=first)
            ratio = _coverage_ratio(covered, executable)
            push!(rows, (
                source=source,
                package=_package_name(source),
                executable=executable,
                covered=covered,
                ratio=ratio,
            ))
        end
        return rows
    end

    latest_files = _collect_latest_cov_files()
    for (source, covfile) in sort(collect(latest_files); by=first)
        executable, covered = _parse_cov_file(covfile)
        ratio = _coverage_ratio(covered, executable)
        push!(rows, (
            source=source,
            package=_package_name(source),
            executable=executable,
            covered=covered,
            ratio=ratio,
        ))
    end
    return rows
end

function _package_name(relpath::AbstractString)::String
    parts = split(relpath, '/')
    isempty(parts) && return "unknown"
    return parts[1]
end

function generate_coverage_summary()::Nothing
    per_file = _collect_per_file_coverage()

    per_package = Dict{String, Tuple{Int, Int}}()
    total_exec = 0
    total_cov = 0
    for row in per_file
        exec0, cov0 = get(per_package, row.package, (0, 0))
        per_package[row.package] = (exec0 + row.executable, cov0 + row.covered)
        total_exec += row.executable
        total_cov += row.covered
    end

    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    overall_pct = 100 * _coverage_ratio(total_cov, total_exec)

    lines = String[]
    push!(lines, "# Coverage Summary")
    push!(lines, "")
    push!(lines, "Generated: $(timestamp)")
    push!(lines, "")
    push!(lines, @sprintf("Overall executable lines covered: %d / %d (%.2f%%)", total_cov, total_exec, overall_pct))
    push!(lines, "")
    push!(lines, "## Per Package")
    push!(lines, "")
    push!(lines, "| Package | Covered | Executable | Coverage |")
    push!(lines, "|---|---:|---:|---:|")
    for pkg in sort(collect(keys(per_package)))
        exec, cov = per_package[pkg]
        pct = 100 * _coverage_ratio(cov, exec)
        push!(lines, @sprintf("| %s | %d | %d | %.2f%% |", pkg, cov, exec, pct))
    end
    push!(lines, "")
    push!(lines, "## Per File")
    push!(lines, "")
    push!(lines, "| File | Covered | Executable | Coverage |")
    push!(lines, "|---|---:|---:|---:|")
    for row in sort(per_file; by=r -> r.source)
        push!(lines, @sprintf("| %s | %d | %d | %.2f%% |", row.source, row.covered, row.executable, 100 * row.ratio))
    end

    write(SUMMARY_PATH, join(lines, "\n") * "\n")

    println(@sprintf("Coverage summary written to %s", relpath(SUMMARY_PATH, ROOT)))
    println(@sprintf("Overall: %d/%d lines (%.2f%%)", total_cov, total_exec, overall_pct))
    for pkg in sort(collect(keys(per_package)))
        exec, cov = per_package[pkg]
        pct = 100 * _coverage_ratio(cov, exec)
        println(@sprintf("  %s: %d/%d (%.2f%%)", pkg, cov, exec, pct))
    end
    return nothing
end

generate_coverage_summary()
