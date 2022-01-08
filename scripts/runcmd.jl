cd("/home/tor/TorProjects/CryptoTimeSeries/scripts/")

println("updating data pwd: $(pwd())")
`julia assetsdownload.jl`
