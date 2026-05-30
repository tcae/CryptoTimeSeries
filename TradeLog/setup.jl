using Pkg

Pkg.activate(@__DIR__)
Pkg.add(["Dates", "JSON3", "Arrow"])
Pkg.develop(path="../EnvConfig")
Pkg.resolve()
Pkg.gc()