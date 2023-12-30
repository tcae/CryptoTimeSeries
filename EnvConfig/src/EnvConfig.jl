# using Pkg;
# Pkg.add(["JSON"])

"""
Provides

- supported bases
- canned train, eval, test data
- authorization to Binance
- test and production mode

"""
module EnvConfig
using Logging, Dates, Pkg
export authorization, test, production, training, now, timestr

import JSON

@enum Mode test production training
cryptoquote = "usdt"
cryptoexchange = "binance"
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
        if configmode == production
            filename = normpath(joinpath(authpath, "auth.json"))
        else  # must be test
            filename = normpath(joinpath(authpath, "auth.json"))   # "auth_Tst1.json" no longer valid
        end
        dicttxt = open(filename, "r") do f
            read(f, String)  # file information to string
        end
        auth = JSON.parse(dicttxt)  # parse and transform data
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

logpath(file) = normpath(joinpath(cryptopath, logfilespath, file))
logsubfolder() = logfilespath == "logs" ? "" : joinpath(splitpath(logfilespath)[2:end])
logfolder() = logfilespath

" set project dir as working dir "
function setprojectdir()
    cd("$(@__DIR__)/../..")  #! assumes a fixed folder structure with EnvConfig as a package within the project folder
    Pkg.activate(pwd())
    # println("activated $(pwd())")
    return pwd()
end

function init(mode::Mode)
    global configmode = mode
    global bases, trainingbases, datafolder
    global authorization

    authorization = Authentication()
    if configmode == production
        bases = [
            "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx", "zrx", "bch",
            "etc", "link", "ada", "matic", "xtz", "zil", "omg", "xlm", "zec",
            "theta"]
        trainingbases = [
            "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx"]
        datafolder = "Features"
    elseif  configmode == training
        # trainingbases = bases = ["btc"]
        # trainingbases = bases = ["btc", "xrp", "eos"]
        datafolder = "Features"
        trainingbases = [
            "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx", "zrx", "bch",
            "etc", "link", "ada", "matic", "xtz", "zil", "omg", "xlm", "zec",
            "theta"]
        bases = [
            "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx",
            "link", "ada", "matic", "omg", "zec", "xlm",
            "theta"]
        # trainingbases = [
        #     "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx", "matic", "link", "theta"]
        # datafolder = "TrainingOHLCV"
    elseif configmode == test
        trainingbases = bases = ["sine", "doublesine"]
        # trainingbases = ["sine", "doublesine"]
        # bases = ["sine", "doublesine"]
        datafolder = "TestFeatures"
        # datafolder = "Features"
    else
        Logging.@error("invalid Config mode $configmode")
    end
end

function datafile(mnemonic::String, extension=".jdf")
    # no file existence checks here because it may be new file
    return normpath(joinpath(cryptopath, datafolder, mnemonic * extension))
end

function setsplitfilename()::String
    # println(configmode)
    if configmode == production
        return normpath(joinpath(cryptopath, datafolder, setsplitfname))
    else
        return normpath(joinpath(cryptopath, datafolder, testsetsplitfname))
    end
end

end # module


