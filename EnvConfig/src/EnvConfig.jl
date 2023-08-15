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
datapathprefix = "crypto/"
otherpathprefix = "crypto/"
authpathprefix = ".catalyst/data/exchanges/bybit/"
cachepath = datapathprefix * "cache/"
datapath = "Features/"
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
            filename = normpath(joinpath(homedir(), authpathprefix, "auth.json"))
        else  # must be test
            filename = normpath(joinpath(homedir(), authpathprefix, "auth.json"))   # "auth_Tst1.json" no longer valid
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

" set project dir as working dir "
function setprojectdir()
    cd("$(@__DIR__)/../..")  #! assumes a fixed folder structure with EnvConfig as a package within the project folder
    Pkg.activate(pwd())
    # println("activated $(pwd())")
    return pwd()
end

function init(mode::Mode)
    global configmode = mode
    global bases, trainingbases, datapath
    global authorization

    authorization = Authentication()
    if configmode == production
        bases = [
            "btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx", "zrx", "bch",
            "etc", "link", "ada", "matic", "xtz", "zil", "omg", "xlm", "zec",
            "tfuel", "theta", "ont", "vet", "iost"]
        trainingbases = [
            "btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx"]
        datapath = "Features/"
    elseif  configmode == training
        # trainingbases = bases = ["btc"]
        # trainingbases = bases = ["btc", "xrp", "eos"]
        datapath = "Features/"
        trainingbases = [
            "btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx", "zrx", "bch",
            "etc", "link", "ada", "matic", "xtz", "zil", "omg", "xlm", "zec",
            "tfuel", "theta", "ont", "vet", "iost"]
        bases = [
            "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx",
            "link", "ada", "matic", "omg", "zec", "xlm",
            "theta", "vet"]
        # trainingbases = [
        #     "btc", "xrp", "eos", "bnb", "eth", "ltc", "trx", "matic", "link", "theta"]
        # datapath = "TrainingOHLCV/"
    elseif configmode == test
        trainingbases = bases = ["sine", "doublesine"]
        # trainingbases = ["sine", "doublesine"]
        # bases = ["sine", "doublesine"]
        datapath = "TestFeatures/"
        # datapath = "Features/"
    else
        Logging.@error("invalid Config mode $configmode")
    end
end

function datafile(mnemonic::String, extension=".jdf")
    # no file existence checks here because it may be new file
    return normpath(joinpath(homedir(), datapathprefix, datapath, mnemonic * extension))
end

function setsplitfilename()::String
    # println(configmode)
    if configmode == production
        return normpath(joinpath(homedir(), datapathprefix, datapath, setsplitfname))
    else
        return normpath(joinpath(homedir(), datapathprefix, datapath, testsetsplitfname))
    end
end

end # module


