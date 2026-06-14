module TestTradeLabels
using Targets


println("Targets.uniquelabels() = $(Targets.uniquelabels()), typeof(Targets.uniquelabels()) = $(typeof(Targets.uniquelabels()))")
println("Targets.tradelabelstrings() = $(Targets.tradelabelstrings()), typeof(Targets.tradelabelstrings()) = $(typeof(Targets.tradelabelstrings()))")
println("Targets.tradelabel(\"longopen\") = $(Targets.tradelabel("longopen")). $(typeof(Targets.tradelabel("longopen")))")
println("longopen > shortclose = $(longopen > shortclose)")
println("shortstrongclose > longstrongclose = $(shortstrongclose > longstrongclose)")
println("tradelabelix(longopen) = $(Targets.tradelabelix(longopen))")
println("tradelabelix(\"longopen\") = $(Targets.tradelabelix("longopen"))")
println("tradelabelix(\"longopen\") = $(Targets.tradelabelix("longopen", [allclose, longopen]))")

myf() = println("this is $(@__FUNCTION__)")
end
