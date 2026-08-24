#=
consist.jl — Unify shared third-party dependency versions across all local
package Manifests in this workspace.

Local packages already reference each other as `Pkg.develop`ed path
dependencies (see e.g. Classify/Manifest.toml's `path = "../Ohlcv"` entries).
Version drift instead happens for shared third-party deps (DataFrames, Arrow,
...): each package's own Project.toml/Manifest.toml is normally resolved in a
separate `julia --project=<pkg>` session, possibly against a different
registry snapshot, so the same dependency can end up pinned to different
versions in different packages.

This script re-`develop`s every local package into every other local
package/root/scripts environment that already declares it as a dependency,
then resolves and instantiates each environment back-to-back in one Julia
process. Running all resolutions against the same in-process registry
snapshot makes the resolver pick consistent shared-dependency versions
everywhere, given each package's own `[compat]` bounds allow it.

Usage:
    julia --project=. scripts/consist.jl
=#
using Pkg

const ROOTPATH = normpath(joinpath(@__DIR__, ".."))

"Folders under the workspace root that are not local packages and must be skipped."
const EXCLUDED_FOLDERS = Set(["gist", "docs", "papers", ".git", ".vscode"])

"Return the package name declared in one folder's Project.toml."
projectname(folderpath::AbstractString)::String = Pkg.TOML.parsefile(joinpath(folderpath, "Project.toml"))["name"]

"Return top-level workspace folders (excluding `scripts`) that hold one local package's Project.toml."
function localpackagefolders(rootpath::AbstractString)::Vector{String}
    folders = String[]
    for entry in sort(readdir(rootpath))
        (entry in EXCLUDED_FOLDERS || entry == "scripts") && continue
        folderpath = joinpath(rootpath, entry)
        isdir(folderpath) || continue
        isfile(joinpath(folderpath, "Project.toml")) || continue
        push!(folders, folderpath)
    end
    return folders
end

"Return whether one environment's Project.toml already declares `depname` as a dependency."
function hasdependency(envpath::AbstractString, depname::AbstractString)::Bool
    toml = Pkg.TOML.parsefile(joinpath(envpath, "Project.toml"))
    return haskey(toml, "deps") && haskey(toml["deps"], depname)
end

packagefolders = localpackagefolders(ROOTPATH)
namebyfolder = Dict(folder => projectname(folder) for folder in packagefolders)
println("$(length(packagefolders)) local packages: $(join(sort(collect(values(namebyfolder))), ", "))")

scriptspath = joinpath(ROOTPATH, "scripts")
environments = vcat([ROOTPATH, scriptspath], packagefolders)

for envpath in environments
    println("\n=== resolving $(envpath) ===")
    Pkg.activate(envpath, io=devnull)
    for (folder, name) in namebyfolder
        (folder == envpath) && continue
        hasdependency(envpath, name) || continue
        Pkg.develop(path=folder, io=devnull)
    end
    Pkg.resolve()
    Pkg.instantiate()
end

println("\n=== garbage collecting unused package versions ===")
Pkg.gc()
println("✓ consist complete: shared dependency versions unified across $(length(environments)) environments")
