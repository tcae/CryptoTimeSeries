"""
Provides

- supported bases
- canned train, eval, test data
- authorization to Binance
- test and production mode

"""
module EnvConfig
using Logging, Dates, Pkg, JSON3, DataFrames, JDF
export authorization, setauthorization!, test, production, training, now, timestr, AbstractConfiguration, configuration, configurationid, readconfigurations!

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


function checkfolders(doverbose, dodebug)
    # ----------------------------
    # Output helpers
    # ----------------------------
    function info(msg)
        doverbose && println("ℹ️  ", msg)
    end

    function debug(msg)
        dodebug && println("🐞 ", msg)
    end

    function fail(msg)
        println(stderr, "\n❌ ENVIRONMENT CHECK FAILED\n")
        println(stderr, msg)
        println(stderr)
        exit(1)
    end

    # ----------------------------
    # Checks start here
    # ----------------------------
    debug("ARGS = $(ARGS)")
    info("Running environment preflight check")

    # --- Check ONEDRIVE_ROOT ---
    root = get(ENV, "ONEDRIVE_ROOT", nothing)

    root === nothing && fail("""
    ONEDRIVE_ROOT is not set.

    This project requires access to a locally synced OneDrive folder.

    SETUP INSTRUCTIONS:

    Windows (PowerShell, run once):
    setx ONEDRIVE_ROOT "%UserProfile%\\OneDrive"

    macOS (Terminal):
    echo 'export ONEDRIVE_ROOT="$HOME/OneDrive"' >> ~/.zprofile
    launchctl setenv ONEDRIVE_ROOT "$HOME/OneDrive"

    Restart your terminal / VS Code / Julia afterward.
    """)

    debug("ONEDRIVE_ROOT = $root")
    info("ONEDRIVE_ROOT is set")

    # --- Validate path ---
    isdir(root) || fail("""
    ONEDRIVE_ROOT is set but does not exist or is not a directory:

    $root

    Check that:
    - OneDrive is installed and synced
    - The path is correct
    - Files are available locally (not online‑only placeholders)
    """)

    info("ONEDRIVE_ROOT directory exists")

    # --- Access test ---
    try
        files = readdir(root)
        debug("Found $(length(files)) entries in ONEDRIVE_ROOT")
    catch e
        fail("""
    Cannot read contents of ONEDRIVE_ROOT:

    $root

    Error:
    $e
    """)
    end

    info("ONEDRIVE_ROOT is readable")

    # --- Optional: required subdirectories ---
    required_subdirs = [
        "crypto", "crypto/exchanges"
    ]

    missing = String[]
    for d in required_subdirs
        p = joinpath(root, d)
        isdir(p) || push!(missing, p)
    end

    if !isempty(missing)
        fail("""
    Required project directories are missing:

    $(join("  - " .* missing, "\n"))
    """)
    end

    info("Required project directories present")

    # ----------------------------
    # Success
    # ----------------------------
    dodebug && println("\n✅ Environment check passed (debug mode)")
    doverbose && !dodebug && println("✅ Environment check passed")
    return root
end

cryptopath = normpath(joinpath(checkfolders(verbosity > 1, verbosity > 2), "crypto")) # OneDrive folder
# cryptopath = normpath(joinpath(homedir(), "crypto")) # local folder
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

    function Authentication(name::Union{Nothing, AbstractString}=nothing)
        filename = configmode != test ? normpath(joinpath(authpath, "auth.json")) : normpath(joinpath(authpath, "authtest.json"))
        authroot = open(filename, "r") do f
            JSON3.read(read(f, String), Dict)
        end
        entries, defaultname = _authentries(authroot)
        @assert length(entries) > 0 "no valid auth entries found in $filename"

        selectedname = isnothing(name) ? (isnothing(defaultname) ? String(entries[1]["name"]) : defaultname) : String(name)
        selectedix = findfirst(entry -> lowercase(String(entry["name"])) == lowercase(selectedname), entries)
        if isnothing(selectedix)
            available = [String(entry["name"]) for entry in entries]
            error("authentication tuple name=$selectedname not found in $filename. available=$available")
        end

        selected = entries[selectedix]
        @assert haskey(selected, "key") && haskey(selected, "secret") "selected auth entry $(selectedname) is missing key/secret"
        new(String(selected["name"]), String(selected["key"]), String(selected["secret"]))
    end
end

"""
Set global authorization credentials by tuple name from the auth file.
If `name` is `nothing`, default entry is selected.
"""
function setauthorization!(name::Union{Nothing, AbstractString}=nothing)
    global authorization
    authorization = Authentication(name)
    return authorization
end

"""
Parse auth root dictionary and return `(entries, defaultname)`.

Supports both legacy and extended formats:
- legacy: single top-level `name/key/secret`
- extended: top-level `default` + `credentials` (vector or dict)
"""
function _authentries(authroot::Dict)
    entries = Dict[]
    defaultname = haskey(authroot, "default") ? String(authroot["default"]) : nothing

    if haskey(authroot, "credentials")
        credentials = authroot["credentials"]
        if credentials isa AbstractVector
            for raw in credentials
                if raw isa AbstractDict
                    entry = Dict(raw)
                    haskey(entry, "name") || continue
                    haskey(entry, "key") || continue
                    haskey(entry, "secret") || continue
                    push!(entries, entry)
                end
            end
        elseif credentials isa AbstractDict
            for (tuple_name, raw) in Dict(credentials)
                if raw isa AbstractDict
                    entry = Dict(raw)
                    if !haskey(entry, "name")
                        entry["name"] = String(tuple_name)
                    end
                    haskey(entry, "key") || continue
                    haskey(entry, "secret") || continue
                    push!(entries, entry)
                end
            end
        end
    end

    if length(entries) == 0
        # Legacy single-entry format: {"name":...,"key":...,"secret":...}
        if haskey(authroot, "name") && haskey(authroot, "key") && haskey(authroot, "secret")
            push!(entries, Dict(
                "name" => authroot["name"],
                "key" => authroot["key"],
                "secret" => authroot["secret"],
            ))
        else
            # Alternative extended format: top-level map of tuples
            for (tuple_name, raw) in authroot
                if raw isa AbstractDict
                    entry = Dict(raw)
                    haskey(entry, "key") || continue
                    haskey(entry, "secret") || continue
                    if !haskey(entry, "name")
                        entry["name"] = String(tuple_name)
                    end
                    push!(entries, entry)
                end
            end
        end
    end

    return entries, defaultname
end

"""
Returns list of authentication tuple names found in the active auth file.
"""
function authenticationnames()::Vector{String}
    filename = configmode != test ? normpath(joinpath(authpath, "auth.json")) : normpath(joinpath(authpath, "authtest.json"))
    authroot = open(filename, "r") do f
        JSON3.read(read(f, String), Dict)
    end
    entries, _ = _authentries(authroot)
    return [String(entry["name"]) for entry in entries]
end

timestr(dt) = isnothing(dt) ? "nodatetime" : Dates.format(dt, EnvConfig.datetimeformat)
now() = Dates.format(Dates.now(), EnvConfig.datetimeformat)

"returns string with timestamp and current git instance to reproduce the used source"
runid() = Dates.format(Dates.now(), "yy-mm-dd_HH-MM-SS") * "_gitSHA-" * read(`git log -n 1 --pretty=format:"%H"`, String)

logfilesfolder = "logs"
# defaultlogfilespath = normpath(joinpath(cryptopath, logfilesfolder))
defaultlogfilespath = normpath(joinpath(homedir(), "crypto", logfilesfolder))
logfilespath = defaultlogfilespath

"extends the log path with folder or resets to default if folder=`nothing`"
function setlogpath(folder=nothing)
    global logfilespath
    if isnothing(folder) || (folder == "")
        logfilespath = defaultlogfilespath
    else
        logfilespath = normpath(joinpath(defaultlogfilespath, folder))
    end
    return mkpath(logfilespath)
end

"Returns the full path including filename of the given filename connected with the current log file path"
logpath(file) = normpath(joinpath(logfilespath, file))
logsubfolder() = splitpath(logfilespath)[end] == logfilesfolder ? "" : splitpath(logfilespath)[end]
logfolder() = logfilespath

"Saves a given dataframe df in the current log folder using the given filename"
function savedf(df, filename; folderpath=logfolder())
    filepath = normpath(joinpath(folderpath, filename))
    JDF.savejdf(filepath, df)
    (verbosity >= 3) && println("$(EnvConfig.now()) saved dataframe to $(filepath)")
end

"Reads and returns a dataframe from filename in the current log folder. Returns `nothing` if file does not exist or is no dataframe file."
function readdf(filename; folderpath=logfolder())
    df = nothing
    filepath = normpath(joinpath(folderpath, filename))
    if isdir(filepath)
        (verbosity >= 4) && print("$(EnvConfig.now()) loading dataframe from  $(filepath) ... ")
        df = DataFrame(JDF.loadjdf(filepath))
        (verbosity >= 4) && println("$(EnvConfig.now()) loaded $(size(df, 1)) rows successfully")
    else
        (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(filepath)")
    end
    return df
end

isfolder(filename) = isdir(EnvConfig.logpath(filename))

function deletefolder(filename; folderpath=logfolder())
    filepath = normpath(joinpath(folderpath, filename))
    if isdir(filepath)
        (verbosity >= 3) && println("$(EnvConfig.now()) deleting folder $(filepath)")
        rm(filepath; force=true, recursive=true)
    else
        (verbosity >= 3) && println("$(EnvConfig.now()) no folder deletion due to missing $(filepath)")
    end
end

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

function init(mode::Mode; newdatafolder=false, authname::Union{Nothing, AbstractString}=nothing)
    global configmode = mode
    global bases, trainingbases, datafolder
    global authorization

    authorization = Authentication(authname)
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
                rm(backupfilename(filenameprefix, 1, filenameextension), force=true, recursive=true) # remove oldest backup
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


