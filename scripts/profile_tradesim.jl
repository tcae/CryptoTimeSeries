"""
profile_tradesim.jl — Runs scripts/tradesim.jl under Julia's sampling profiler
and stores a flat/tree profile report plus timing summary under
\$HOME/crypto/debug/tradesim-profile-<timestamp>/.

Usage:
    julia --project=. scripts/profile_tradesim.jl [same key=value args as tradesim.jl]
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."), io=devnull)

using Profile, Dates, Printf

global ARGS = copy(Base.ARGS)  # tradesim.jl reads the global ARGS

outroot = joinpath(homedir(), "crypto", "debug", "tradesim-profile-$(Dates.format(Dates.now(), Dates.DateFormat("yymmdd-HHMMSS")))")
mkpath(outroot)

Profile.clear()
elapsed = @elapsed begin
    Profile.@profile include(joinpath(@__DIR__, "tradesim.jl"))
end

@printf("profile_tradesim: tradesim.jl finished in %.1f s\n", elapsed)

open(joinpath(outroot, "profile-flat.txt"), "w") do io
    Profile.print(io; format=:flat, sortedby=:count, mincount=5)
end
open(joinpath(outroot, "profile-tree.txt"), "w") do io
    Profile.print(io; format=:tree, mincount=5, maxdepth=40)
end
open(joinpath(outroot, "summary.txt"), "w") do io
    println(io, "elapsed_seconds=$(elapsed)")
    println(io, "args=$(ARGS)")
    println(io, "n_profile_samples=$(length(Profile.fetch()))")
end

println("profile_tradesim: saved profile reports to $outroot")
