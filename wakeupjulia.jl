using EnvConfig

# testdir = normpath(joinpath(@__DIR__, "..", "..", "crypto"))
EnvConfig.init(production)
println("EnvCOnfig=$(EnvConfig.configmode) DIR cryptopath: $(EnvConfig.cryptopath) isdir=$(isdir(EnvConfig.cryptopath))")
println("EnvCOnfig=$(EnvConfig.configmode) DIR authpath: $(EnvConfig.authpath) isdir=$(isdir(EnvConfig.authpath))")
EnvConfig.init(training)
println("EnvCOnfig=$(EnvConfig.configmode) DIR cryptopath: $(EnvConfig.cryptopath) isdir=$(isdir(EnvConfig.cryptopath))")
println("EnvCOnfig=$(EnvConfig.configmode) DIR authpath: $(EnvConfig.authpath) isdir=$(isdir(EnvConfig.authpath))")
EnvConfig.init(test)
println("EnvCOnfig=$(EnvConfig.configmode) DIR cryptopath: $(EnvConfig.cryptopath) isdir=$(isdir(EnvConfig.cryptopath))")
println("EnvCOnfig=$(EnvConfig.configmode) DIR authpath: $(EnvConfig.authpath) isdir=$(isdir(EnvConfig.authpath))")
println("wake up julia $VERSION")

