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

# Develop all packages first without resolving individually
for mypackage in mypackages
    folderpath = joinpath(rootpath, mypackage)
    println("developing $folderpath")
    Pkg.develop(path=folderpath)
end

# Resolve and instantiate once at the main level
println("resolving main environment...")
Pkg.resolve()
println("instantiating main environment...")
Pkg.instantiate()
println("garbage collecting...")
Pkg.gc()
println("✓ instantiation complete")
