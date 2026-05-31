using Dates
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SUMMARY_PATH = joinpath(@__DIR__, "coverage_summary.md")

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

function _package_name(relpath::AbstractString)::String
    parts = split(relpath, '/')
    isempty(parts) && return "unknown"
    return parts[1]
end

function generate_coverage_summary()::Nothing
    latest_files = _collect_latest_cov_files()
    per_file = Vector{NamedTuple{(:source, :package, :executable, :covered, :ratio), Tuple{String, String, Int, Int, Float64}}}()

    for (source, covfile) in sort(collect(latest_files); by=first)
        executable, covered = _parse_cov_file(covfile)
        ratio = _coverage_ratio(covered, executable)
        push!(per_file, (
            source=source,
            package=_package_name(source),
            executable=executable,
            covered=covered,
            ratio=ratio,
        ))
    end

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
