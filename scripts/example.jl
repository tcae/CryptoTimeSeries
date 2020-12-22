import Pkg; Pkg.add("BSON")
using DrWatson
@quickactivate "CryptoTimeSeries"

module WatsonTest

using DrWatson

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
    wsave(datadir("simulations", savename(d, "bson")), f)
end

end  # module
