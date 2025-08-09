"""
Provides

- supported bases
- canned train, eval, test data
- authorization to Binance
- test and production mode

"""
module EnvConfig
using Logging, Dates, Pkg, JSON3, DataFrames, JDF
export authorization, test, production, training, now, timestr, AbstractConfiguration, configuration, configurationid, readconfigurations!

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info, e.g. number of steps in rowix
"""
verbosity = 1

@enum Mode test production training
cryptoquote = "USDT"
datetimeformat = "yymmdd HH:MM"
timezone = "Europe/Amsterdam"
symbolseperator = "_"  # symbol seperator
setsplitfname = "sets_split.csv"
testsetsplitfname = "test_sets_split.csv"
bases = String[]
trainingbases = String[]
cryptopath = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "crypto"))
if !isdir(cryptopath)
    @error "missing crypto folder $cryptopath"
end
if !isdir(cryptopath)
    @warn "cryptopath=$cryptopath is not an existing folder - now creating it"
    mkpath(cryptopath)
    @assert isdir(cryptopath)  "cannot create $cryptopath"
end
authpath = joinpath(cryptopath, "exchanges")  # ".catalyst/data/exchanges/bybit/"
if !isdir(cryptopath)
    @error "missing auth folder $authpath"
end
datafolder = "EnvConfig is not initialized"
configmode = production
authorization = nothing
# TODO file path checks to be added

struct Authentication
    name::String
    key::String
    secret::String

    function Authentication()
        auth = Dict()
        if configmode != test
            filename = normpath(joinpath(authpath, "auth.json"))
        else  # must be test
            filename = normpath(joinpath(authpath, "authtest.json"))
        end
        dicttxt = open(filename, "r") do f
            read(f, String)  # file information to string
        end
        auth = JSON3.read(dicttxt, Dict)  # parse and transform data
        # println(mode, auth)
        new(auth["name"], auth["key"], auth["secret"])
    end
end

timestr(dt) = isnothing(dt) ? "nodatetime" : Dates.format(dt, EnvConfig.datetimeformat)
now() = Dates.format(Dates.now(), EnvConfig.datetimeformat)

"returns string with timestamp and current git instance to reproduce the used source"
runid() = Dates.format(Dates.now(), "yy-mm-dd_HH-MM-SS") * "_gitSHA-" * read(`git log -n 1 --pretty=format:"%H"`, String)

logfilespath = "logs"

"extends the log path with folder or resets to default if folder=`nothing`"
function setlogpath(folder=nothing)
    global logfilespath
    if isnothing(folder) || (folder == "")
        logfilespath = "logs"
    else
        logfilespath = joinpath("logs", folder)
    end
    return mkpath(normpath(joinpath(cryptopath, logfilespath)))
end

"Returns the full path including filename of the given filename connected with the current log file path"
logpath(file) = normpath(joinpath(cryptopath, logfilespath, file))
logsubfolder() = logfilespath == "logs" ? "" : joinpath(splitpath(logfilespath)[2:end])
logfolder() = normpath(joinpath(cryptopath, logfilespath))

" set project dir as working dir "
function setprojectdir()
    cd("$(@__DIR__)/../..")  #! assumes a fixed folder structure with EnvConfig as a package within the project folder
    Pkg.activate(pwd())
    # println("activated $(pwd())")
    return pwd()
end

datafolderpath(subfolder=nothing) = isnothing(subfolder) ? normpath(joinpath(cryptopath, datafolder)) : normpath(joinpath(cryptopath, datafolder, subfolder))

function datafile(mnemonic::String, subfolder=nothing, extension=".jdf")
    p = datafolderpath(subfolder)
    if !isdir(p)
        println("EnvConfig $(now()): creating folder $p")
        mkpath(p)
    end
    # no file existence checks here because it may be new file
    return isnothing(subfolder) ? normpath(joinpath(cryptopath, datafolder, mnemonic * extension)) : normpath(joinpath(cryptopath, datafolder, subfolder, mnemonic * extension))
end

function getdatafolder(folder, newfolder=false)
    folder = newfolder ? folder * runid() : folder
    p = normpath(joinpath(cryptopath, folder))
    if !isdir(p)
        println("EnvConfig $(now()): creating folder $p")
        mkpath(p)
    end
    return folder
end

function init(mode::Mode; newdatafolder=false)
    global configmode = mode
    global bases, trainingbases, datafolder
    global authorization

    authorization = Authentication()
    if configmode == production
        bases = [
            "btc"]
        trainingbases = [
            "btc"]
        datafolder = getdatafolder("Features", newdatafolder)
    elseif  configmode == training
        # trainingbases = bases = ["btc"]
        # trainingbases = bases = ["btc", "xrp", "eos"]
        datafolder = getdatafolder("Features", newdatafolder)
        trainingbases = [
            "btc"]
        bases = [
            "btc"]
        # trainingbases = [
        #     "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx", "matic", "link", "theta"]
        # datafolder = "TrainingOHLCV"
    elseif configmode == test
        trainingbases = bases = ["SINEUSDT", "DOUBLESINEUSDT"]
        # trainingbases = ["SINEUSDT", "DOUBLESINEUSDT"]
        # bases = ["SINEUSDT", "DOUBLESINEUSDT"]
        datafolder = getdatafolder("TestFeatures", newdatafolder)
        # datafolder = "Features"
    else
        Logging.@error("invalid Config mode $configmode")
    end
    bases = uppercase.(bases)
    trainingbases = uppercase.(trainingbases)
end

function setsplitfilename()::String
    # println(configmode)
    if configmode == production
        return normpath(joinpath(cryptopath, datafolder, setsplitfname))
    else
        return normpath(joinpath(cryptopath, datafolder, testsetsplitfname))
    end
end

function backupfilename(filenameprefix, backupnbr, filenameextension)
    bfn = join([filenameprefix, string(backupnbr)], "_")
    if !isnothing(filenameextension)
        bfn = join([bfn, filenameextension], ".")
    end
    return bfn
end

function savebackup(filename; maxbackups=100)
    ret = []
    if isdir(filename) || isfile(filename)
        backupnbr = 1
        sfn1 = rsplit(filename, "."; limit=2)
        if length(sfn1) > 1
            filenameprefix = sfn1[1]
            filenameextension = sfn1[2]
        else # no filenameextension
            filenameprefix = filename
            filenameextension = nothing
        end
        bfn = backupfilename(filenameprefix, backupnbr, filenameextension)
        ret = push!(ret, bfn)
        while isdir(bfn) || isfile(bfn)
            if backupnbr == maxbackups
                rm(backupfilename(filenameprefix, 1, filenameextension)) # remove oldest backup
                (verbosity >= 3) && println("rm($(backupfilename(filenameprefix, 1, filenameextension)))")
                for bnr in 2:backupnbr
                    (verbosity >= 3) && println("mv($(backupfilename(filenameprefix, bnr, filenameextension)), $(backupfilename(filenameprefix, bnr-1, filenameextension)))")
                    mv(backupfilename(filenameprefix, bnr, filenameextension), backupfilename(filenameprefix, bnr-1, filenameextension))
                end
            else
                backupnbr += 1
                bfn = backupfilename(filenameprefix, backupnbr, filenameextension)
                ret = push!(ret, bfn)
            end
        end
        (verbosity >= 3) && println("folder=$(pwd()) filename=$filename, backupfilename=$bfn, sfn1=$sfn1, filenameprefix=$filenameprefix, backupnbr=$backupnbr, filenameextension=$filenameextension")
        mv(filename, bfn)
    end
    (verbosity >= 3) && println("maxbackups=$maxbackups, backupnbr=$backupnbr, ret=$ret")
    return ret
end

#region abstract-configuration

"""
Provides a facility to write a configuration into 1 row of a DataFrame and retrieve it for configuring types, e.g. Features, Targets, Classiifers.
The subtype shall implement a property `cfg` to longhold the DataFrame of all configurations of that subtype.
`cfgid` is an Integer identifier of a configuration but also used as direct access row index within the `cfg` DataFrame.
"""
abstract type AbstractConfiguration end

"Returns the configuration id in case the configuration is already registered or registers configuration persistently and returns cfgset configuration identifier"
function configurationid(cfgset::AbstractConfiguration, config::Union{NamedTuple, DataFrameRow})::Integer
    if !hasproperty(cfgset, :cfg)
        return 0
    end
    if size(cfgset.cfg, 1) > 0
        match = nothing
        for k in keys(config)
            if k != :cfgid
                match2 = isa(config[k], AbstractFloat) ? isapprox.(cfgset.cfg[!, k], config[k]) : cfgset.cfg[!, k] .== config[k]
                # (verbosity >=3) && println("$k = $match2")
                match = isnothing(match) ? match2 : match .& match2
            else
                # (verbosity >=3) && println("cfgid = $k")
            end
        end
        indices = findall(match)
    else
        indices = []
    end
    cfgid = 0
    if length(indices) == 0
        cfgid = length(cfgset.cfg[!, :cfgid]) > 0 ? maximum(cfgset.cfg[!, :cfgid]) + 1 : 1
        push!(cfgset.cfg, (;config..., cfgid))
        (verbosity >= 3) && println("raw config = $config config=$((;config..., cfgid))  cfg=$(cfgset.cfg)")
        writeconfigurations(cfgset)
    else
        @assert length(indices) == 1 "unexpected multiple entries $(indices) for the same configuration $(cfgset.cfg)"
        cfgid = cfgset.cfg[indices[1], :cfgid]
        @assert cfgid == indices[1] "config id ($cfgid) != index ($(indices[1]))"
    end
    @assert !isnothing(cfgid)
    return cfgid
end

"Returns the DataFrameRow that matches "
function configuration(cfgset::AbstractConfiguration, cfgid::Integer)
    if !hasproperty(cfgset, :cfg)
        return (cfgid=cfgid)
    end
    @assert cfgid <= size(cfgset.cfg, 1) "config id ($cfgid) > size(cfgset.cfg)=$(size(cfgset.cfg, 1)), config=$(cfgset.cfg)"
    @assert cfgset.cfg[cfgid, :cfgid] == cfgid "config id ($cfgid) != index ($(cfgset.cfg[cfgid, :cfgid]))"
    return cfgset.cfg[cfgid, :]
end


filename(cfgset::AbstractConfiguration) = EnvConfig.datafile(string(typeof(cfgset)) * "Config", "Config", ".jdf")

"Writes the cfgset DataFrame config file into field cfg"
function writeconfigurations(cfgset::AbstractConfiguration)
    if !hasproperty(cfgset, :cfg)
        return
    end
    fn = filename(cfgset)
    (verbosity >=3) && println("saved config in cfgfilename=$fn")
    JDF.savejdf(fn, cfgset.cfg)
end

"""
Reads the cfgset DataFrame config file from field cfg. If no config file then an empty DataFrame is created according to optparams.
`optparams` denotes for each dict pair a column as key and a vector of config parameters. The eltype of that vector is used as column type.
"""
function readconfigurations!(cfgset::AbstractConfiguration, optparams::Dict=Dict())
    if !hasproperty(cfgset, :cfg)
        return cfgset
    end
    fn = filename(cfgset)
    if isdir(fn)
        cfgset.cfg = DataFrame(JDF.loadjdf(fn))
        if isnothing(cfgset.cfg)
            @error "Loading $fn failed"
            cfgset.cfg = emptyconfigdf(cfgset, optparams)
        else
            (verbosity >= 2) && println("\r$(EnvConfig.now()) Loaded cfgset config from $fn")
        end
    else
        (verbosity >= 2) && println("\r$(EnvConfig.now()) No configuration file $fn found")
        cfgset.cfg = emptyconfigdf(cfgset, optparams)
        (verbosity >= 3) && println("\r$(EnvConfig.now()) empty cfgset config $(cfgset.cfg)")
    end
    return cfgset
end

function emptyconfigdf(cfgset::AbstractConfiguration, optparams::Dict=Dict())
    (verbosity >= 3) && println("AbstractConfiguration emptyconfigdf of $(typeof(cfgset))")
    df = DataFrame(
        cfgid=Int16[]  # config identificator
    )
    for (col, vec) in optparams
        coltype = eltype(vec)
        df[!, col] = Vector{coltype}()
    end
    return df
end

#endregion abstract-configuration

end # module


