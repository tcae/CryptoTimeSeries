using DrWatson
@quickactivate "CryptoTimeSeries"

# using Pkg;
# Pkg.resolve()
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates","DataFrames","JSON"])

"""
Provides

- supported bases
- canned train, eval, test data
- authorization to Binance
- test and production mode

"""
module Config

export Authentication, test, production

using Dates
using DataFrames
using JSON

@enum Mode test production

datetimeformat = "%Y-%m-%d_%Hh%Mm"
quotesymbol = "usdt"
timezone = "Europe/Amsterdam"
symbolseperator = "_"  # symbol seperator
setsplitfname = "sets_split.csv"
testsetsplitfname = "test_sets_split.csv"
bases = String[]
trainingbases = String[]
home = "/home/tor/"
datapathprefix = "crypto/"
otherpathprefix = "crypto/"
authpathprefix = ".catalyst/data/exchanges/binance/"
cachepath = datapathprefix * "cache/"
datapath = "Features/"
configmode = production
authorization = nothing
# ! file path checks to be added

struct Authentication
    name::String
    key::String
    secret::String

    function Authentication()
        auth = Dict()
        if configmode == production
            filename = home * authpathprefix * "auth.json"
        else  # must be test
            filename = "/home/tor/.catalyst/data/exchanges/binance/auth_Tst1.json"
        end
        dicttxt = open(filename, "r") do f
            read(f, String)  # file information to string
        end
        auth = JSON.parse(dicttxt)  # parse and transform data
        # println(mode, auth)
        new(auth["name"], auth["key"], auth["secret"])
    end
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
    else  # must be test
        trainingbases = ["sinus"]
        bases = ["sinus"]
        datapath = "TestFeatures/"
        # datapath = "Features/"
    end
end

function datafile(mnemonic::String, extension=".jdf")
    # no file existence checks here because it may be new file
    return home * datapathprefix * datapath * mnemonic * extension
end

function setsplitfilename()::String
    # println(configmode)
    if configmode == production
        return home * datapathprefix * datapath * setsplitfname
    else
        return home * datapathprefix * datapath * testsetsplitfname
    end
end

function greet1()
    println("hello from greet1")
    x = Authentication(test)
    println(x)
end


function greet2()

    function greet3()
        println("greet3 start")
        greet1()
        println("hello from greet3")
    end

    a = 3
    println("hello from greet2")
    greet3()
end

# greet() = print("Hello World!")


# println("greetings from module env_config")
end # module

# println("greetings from top level env_config")

