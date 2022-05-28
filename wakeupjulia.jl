println("wake up julia")
# for ix in (1:3); println(ix);end
# for ix in (3:-1:1); println(ix);end
# a = [ix for ix in vcat(1:3,3:-1:1)]; println(a)
# size(a, 1)

dd = Dict(1 => Dict("a" => 520, "b" => 521), 3 => Dict("x" => 670, "y" =>671))

mutable struct DD
    dd
end

function Base.iterate(elem::DD)
    for (e, ev) in elem.dd
        for (e1, ev1) in ev
            println("initial: $e / $e1")
            return (ev1, (e, e1))
        end
    end
    return nothing
end

function Base.iterate(elem::DD, state)
    found = false
    (laste, laste1) = state
    println("state: $laste / $laste1")
    for (e, ev) in elem.dd
        for (e1, ev1) in ev
            if found
                println("next: $e / $e1")
                return (ev1, (e, e1))
            end
            if (laste == e) && (laste1 == e1)
                println("found: $e / $e1")
                found = true
            end
        end
    end
    return nothing
end

mdd = DD(dd)
for (e, ev) in mdd.dd
    for (e1, ev1) in ev
        println("struct: $e / $e1 => $ev1")
    end
end

for e in mdd
    println("result: $e")
end
