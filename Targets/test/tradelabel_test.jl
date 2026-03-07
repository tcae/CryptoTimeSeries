module TestTradeLabels
using Targets


println("Targets.tradelabels() = $(Targets.tradelabels()), typeof(Targets.tradelabels()) = $(typeof(Targets.tradelabels()))")
println("Targets.tradelabelstrings() = $(Targets.tradelabelstrings()), typeof(Targets.tradelabelstrings()) = $(typeof(Targets.tradelabelstrings()))")
println("Targets.tradelabel(\"longbuy\") = $(Targets.tradelabel("longbuy")). $(typeof(Targets.tradelabel("longbuy")))")
println("longbuy > shortclose = $(longbuy > shortclose)")
println("shortstrongclose > longstrongclose = $(shortstrongclose > longstrongclose)")
println("tradelabelix(longbuy) = $(Targets.tradelabelix(longbuy))")
println("tradelabelix(\"longbuy\") = $(Targets.tradelabelix("longbuy"))")
println("tradelabelix(\"longbuy\") = $(Targets.tradelabelix("longbuy", [allclose, longbuy]))")

myf() = println("this is $(@__FUNCTION__)")
end
