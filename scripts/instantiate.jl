using Pkg
# rootpath = ".."
rootpath = joinpath(@__DIR__, "..")
if Sys.islinux()
    # rootpath = joinpath(@__DIR__, "..")
    println("Linux, rootpath: $rootpath, homepath: $(homedir())")
elseif Sys.isapple()
    # rootpath = joinpath(@__DIR__, "..")
    println("Apple, rootpath: $rootpath, homepath: $(homedir())")
elseif Sys.iswindows()
    # rootpath = joinpath(@__DIR__, "..")
    println("Windows, rootpath: $rootpath, homepath: $(homedir())")
else
    # rootpath = joinpath(@__DIR__, "..")
    println("unknown OS, rootpath: $rootpath, homepath: $(homedir())")
end
Pkg.activate(rootpath)
cd(rootpath)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
mypackages = ["Trade", "Classify", "CryptoXch", "Bybit", "EnvConfig", "Ohlcv", "Features", "Targets", "TestOhlcv"]
rootpath = "."
for mypackage in mypackages
    folderpath = joinpath(rootpath, mypackage)
    println("preparing $folderpath")
    Pkg.develop(path=folderpath)
    Pkg.activate(folderpath)
    Pkg.update()
    Pkg.resolve()
    Pkg.instantiate()
    Pkg.precompile()
    Pkg.activate(rootpath)
    # Pkg.gc()
end

# Pkg.update()
Pkg.gc()
Pkg.instantiate()
