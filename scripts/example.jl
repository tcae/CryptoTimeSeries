import Pkg; Pkg.add(["BSON", "TimeseriesPrediction"])
using DrWatson
@quickactivate "CryptoTimeSeries"

module WatsonTest

using DrWatson
using DataFrames

function watsonfunctions()
    println(projectname())
    println(projectdir())
    println(datadir())
    println(plotsdir())
    println(srcdir())
    println(scriptsdir())
    # println(testdir())
end

function fakesim(a, b, v, method = "linear")
    if method == "linear"
        r = @. a + b * v
    elseif method == "cubic"
        r = @. a*b*v^3
    end
    y = sqrt(b)
    return r, y
end

function makesim(d::Dict)
    @unpack a, b, v, method = d
    r, y = fakesim(a, b, v, method)
    fulld = copy(d)
    fulld[:r] = r
    fulld[:y] = y
    return fulld
end

"""
```
barkley(T;
        tskip=0,
        periodic=true,
        ssize=(50,50),
        a=0.75, b=0.06, ε=0.08, D=1/50, h=0.1, Δt=0.1)
```
Simulate the Barkley model (nonlinear `u^3` term).
"""
function barkley(t, n, e; seed=1)
    return [t+n, t-n], e+seed
end

function sim2()
    ΔTs = [1.0, 0.5, 0.1] # resolution of the saved data
    Ns = [50, 150] # spatial extent
    for N ∈ Ns, ΔT ∈ ΔTs
        T = 10050 # we can offset up to 1000 units
        every = round(Int, ΔT/2)
        # every = round(Int, ΔT/barkley_Δt)
        seed = 1111

        simulation = @ntuple T N ΔT seed
        U, V = barkley(T, N, every; seed = seed)

        @tagsave(
            datadir("sim", "bk", savename(simulation, "bson")),
            @dict U V simulation
        )
    end
end

watsonfunctions()
a, b = 2, 3
v = rand(5)
method = "linear"
r, y = fakesim(a, b, v, method)
println(r,y)

params = Dict(:a => 2, :b => 3, :v => rand(5), :method => "linear")

display(makesim(params))

allparams = Dict(
    :a => [1, 2], # it is inside vector. It is expanded.
    :b => [3, 4],
    :v => [rand(5)],     # single element inside vector; no expansion
    :method => "linear", # not in vector = not expanded, even if naturally iterable
)

dicts = dict_list(allparams)
display(dicts)
for (i, d) in enumerate(dicts)
    f = makesim(d)
    # wsave(datadir("simulations", savename(d, "bson")), f)
    @tagsave(datadir("simulations", savename(d, "bson")), f)
end

println(readdir(datadir("simulations")))
firstsim = readdir(datadir("simulations"))[1]

d = wload(datadir("simulations", firstsim))
display(d)
df = collect_results(datadir("simulations"))
display(df.path)

sim2()
firstsim = readdir(datadir("sim", "bk"))[1]
d = wload(datadir("sim", "bk", firstsim))
display(d)
df = collect_results(datadir("sim", "bk"))
display(df.path)

end  # module

using Plots
x = 1:10; y = rand(10); # These are the plotting data
plot(x, y)
