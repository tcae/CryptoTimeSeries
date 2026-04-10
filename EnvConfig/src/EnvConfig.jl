"""
Provides

- supported bases
- canned train, eval, test data
- authorization to Binance
- test and production mode

"""
module EnvConfig
using Logging, Dates, Pkg, JSON3, DataFrames, JDF, Arrow
using CategoricalArrays
export authorization, setauthorization!, test, production, training, now, timestr, AbstractConfiguration, configuration, configurationid, readconfigurations!, tablepath, tableexists, dfformat, setdfformat!, coinspath, coinfolderpath, coinfile

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
coinsfolder = "coins"
# defaultlogfilespath = normpath(joinpath(cryptopath, logfilesfolder))
defaultlogfilespath = normpath(joinpath(homedir(), "crypto", logfilesfolder))
defaultcoinspath = normpath(joinpath(dirname(defaultlogfilespath), coinsfolder))
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

const defaultdfformat = Ref{Symbol}(:jdf)

"Return the preferred default storage format for `savedf` / `readdf`."
dfformat() = defaultdfformat[]

"Set the preferred default storage format for `savedf` / `readdf`."
function setdfformat!(format::Symbol)
    @assert format in (:jdf, :arrow) "format=$(format) must be :jdf or :arrow"
    defaultdfformat[] = format
    return format
end

"Return the file extension for a supported table storage format."
function _table_extension(format::Symbol)::String
    if format == :jdf
        return ".jdf"
    elseif format == :arrow
        return ".arrow"
    end
    throw(ArgumentError("unsupported table format=$(format); expected :jdf or :arrow"))
end

"Normalize a table filename to the requested on-disk format."
function _table_filename(filename::AbstractString, format::Symbol)::String
    base, ext = splitext(filename)
    lowered = lowercase(ext)
    expected = _table_extension(format)
    return lowered == expected ? filename : (isempty(lowered) ? filename * expected : base * expected)
end

const _legacy_table_layout = Dict(
    "results" => joinpath("results", "all"),
    "features" => joinpath("features", "all"),
    "targets" => joinpath("targets", "all"),
    "predictions" => joinpath("predictions", "all"),
    "maxpredictions" => joinpath("predictions", "maxpredictions"),
    "gains" => joinpath("trades", "gains_all"),
    "distances" => joinpath("trades", "distances"),
    "lstm_gains" => joinpath("trades", "lstm_gains_all"),
    "lstm_transaction_pairs" => joinpath("trades", "lstm_transaction_pairs_all"),
)
const _modern_table_layout = Dict(v => k for (k, v) in _legacy_table_layout)

"Return the preferred modern/legacy storage stems to try for one logical artifact."
function _table_stems(filename::AbstractString)
    normalized = replace(normpath(String(filename)), '\\' => '/')
    base, _ = splitext(normalized)
    primary = get(_legacy_table_layout, base, base)
    stems = [primary]
    if haskey(_legacy_table_layout, base)
        push!(stems, base)
    elseif haskey(_modern_table_layout, base)
        push!(stems, _modern_table_layout[base])
    end
    return unique(stems)
end

"Return the full path for a stored table in the requested format. With `format=:auto`, return the first existing candidate path or the preferred default candidate when none exists yet."
function tablepath(filename::AbstractString; folderpath=logfolder(), format::Symbol=:jdf, preferred::Symbol=dfformat())
    if format == :auto
        candidates = _table_candidates(filename; preferred=preferred)
        for candidate in candidates
            candidate_format = endswith(lowercase(candidate), ".arrow") ? :arrow : :jdf
            filepath = normpath(joinpath(folderpath, candidate))
            if _table_storage_exists(filepath, candidate_format)
                return filepath
            end
        end
        return normpath(joinpath(folderpath, first(candidates)))
    end
    @assert format in (:jdf, :arrow) "format=$(format) must be :auto, :jdf or :arrow"
    return normpath(joinpath(folderpath, _table_filename(filename, format)))
end

function _table_candidates(filename::AbstractString; preferred::Symbol=dfformat())
    normalized = replace(normpath(String(filename)), '\\' => '/')
    _, ext = splitext(normalized)
    lowered = lowercase(ext)
    @assert preferred in (:jdf, :arrow) "preferred=$(preferred) must be :jdf or :arrow"

    candidates = String[]
    for stem in _table_stems(normalized)
        if lowered == ".arrow"
            append!(candidates, preferred == :jdf ? [stem * ".jdf", stem * ".arrow"] : [stem * ".arrow", stem * ".jdf"])
        elseif lowered == ".jdf"
            append!(candidates, preferred == :arrow ? [stem * ".arrow", stem * ".jdf"] : [stem * ".jdf", stem * ".arrow"])
        else
            append!(candidates, preferred == :arrow ? [stem * ".arrow", stem * ".jdf"] : [stem * ".jdf", stem * ".arrow"])
        end
    end
    return unique(candidates)
end

_table_storage_exists(filepath::AbstractString, format::Symbol)::Bool = format == :jdf ? isdir(filepath) : isfile(filepath)

_arrow_stringify(value) = ismissing(value) ? missing : string(value)

"Return `true` if a column element type can be stored in Arrow without coercion."
function _arrow_native_type(T::Type)::Bool
    return T <: Union{Missing, Bool, Integer, AbstractFloat, AbstractString, Dates.TimeType, AbstractChar}
end

"Return the smallest signed integer type that can represent the observed enum codes."
function _smallest_signed_type(lo::Int, hi::Int)::DataType
    if (typemin(Int8) <= lo) && (hi <= typemax(Int8))
        return Int8
    elseif (typemin(Int16) <= lo) && (hi <= typemax(Int16))
        return Int16
    elseif (typemin(Int32) <= lo) && (hi <= typemax(Int32))
        return Int32
    end
    return Int64
end

"Return the smallest unsigned integer type that can represent the observed values."
function _smallest_unsigned_type(hi::Int)::DataType
    if hi <= typemax(UInt8)
        return UInt8
    elseif hi <= typemax(UInt16)
        return UInt16
    elseif hi <= typemax(UInt32)
        return UInt32
    end
    return UInt64
end

"Determine a compact storage type for an enum-backed column."
function _enum_storage_type(values)::DataType
    seen = false
    lo = typemax(Int)
    hi = typemin(Int)
    for value in values
        ismissing(value) && continue
        seen = true
        code = Int(value)
        lo = min(lo, code)
        hi = max(hi, code)
    end
    return seen ? _smallest_signed_type(lo, hi) : Int8
end

"Determine a compact integer storage type for a numeric column based on its observed range."
function _integer_storage_type(values, T::Type)::DataType
    seen = false
    lo = typemax(Int)
    hi = typemin(Int)
    for value in values
        ismissing(value) && continue
        seen = true
        intval = Int(value)
        lo = min(lo, intval)
        hi = max(hi, intval)
    end
    if !seen
        return T <: Unsigned ? UInt8 : Int8
    elseif lo >= 0
        return _smallest_unsigned_type(hi)
    end
    return _smallest_signed_type(lo, hi)
end

"Convert enum-backed values to compact signed integer codes while preserving missings."
function _enum_codes(values)
    storagetype = _enum_storage_type(values)
    return [ismissing(value) ? missing : storagetype(Int(value)) for value in values]
end

"Convert integer columns to the smallest compatible signed or unsigned type while preserving missings."
function _compact_integer_values(values)
    storagetype = _integer_storage_type(values, Base.nonmissingtype(eltype(values)))
    return [ismissing(value) ? missing : storagetype(value) for value in values]
end

"Convert values to a categorical representation that Arrow stores as dictionary-encoded while preserving missings."
function _arrow_dictencoded_categorical(values)
    normalized = _arrow_stringify.(values)
    return CategoricalArray(normalized)
end

"Convert non-native Arrow columns to dictionary-encoded categorical or enum-backed values while preserving missings."
function _arrow_safe_table(table)
    df = DataFrame(table)
    for name in names(df)
        column = df[!, name]
        colsym = Symbol(name)
        T = Base.nonmissingtype(eltype(column))
        if T <: Enum
            df[!, name] = _arrow_dictencoded_categorical(column)
        elseif (column isa CategoricalArray) || (T <: CategoricalValue)
            df[!, name] = column
        elseif (colsym == :coin) && (T <: AbstractString)
            df[!, name] = _arrow_dictencoded_categorical(column)
        elseif (T <: Integer) && !(T <: Bool)
            df[!, name] = _compact_integer_values(column)
        elseif !_arrow_native_type(T)
            df[!, name] = _arrow_dictencoded_categorical(column)
        end
    end
    return df
end

"Return whether a stored table exists in the requested or auto-detected format."
function tableexists(filename::AbstractString; folderpath=logfolder(), format::Symbol=:auto)::Bool
    if format == :auto
        for candidate in _table_candidates(filename; preferred=dfformat())
            candidate_format = endswith(lowercase(candidate), ".arrow") ? :arrow : :jdf
            if _table_storage_exists(normpath(joinpath(folderpath, candidate)), candidate_format)
                return true
            end
        end
        return false
    end
    @assert format in (:jdf, :arrow) "format=$(format) must be :auto, :jdf or :arrow"
    return _table_storage_exists(tablepath(filename; folderpath=folderpath, format=format), format)
end

"Write a table in JDF or Arrow format and return the resulting path."
function writetable(table, filename::AbstractString; folderpath=logfolder(), format::Symbol=:jdf)
    @assert format in (:jdf, :arrow) "format=$(format) must be :jdf or :arrow"
    filepath = tablepath(filename; folderpath=folderpath, format=format)
    mkpath(dirname(filepath))
    if format == :jdf
        JDF.savejdf(filepath, DataFrame(table))
    else
        Arrow.write(filepath, _arrow_safe_table(table))
    end
    (verbosity >= 3) && println("$(EnvConfig.now()) saved $(format) table to $(filepath)")
    return filepath
end

"Read a stored table in JDF or Arrow format. With `format=:auto`, the preferred default format is tried first and the other format is used as fallback. Set `materialize=false` to keep Arrow tables in their lazy form, and `copycols=true` only when the caller needs to mutate the loaded dataframe."
function readtable(filename::AbstractString; folderpath=logfolder(), format::Symbol=:auto, preferred::Symbol=dfformat(), materialize::Bool=true, copycols::Bool=false)
    if format == :auto
        for candidate in _table_candidates(filename; preferred=preferred)
            candidate_format = endswith(lowercase(candidate), ".arrow") ? :arrow : :jdf
            filepath = normpath(joinpath(folderpath, candidate))
            if _table_storage_exists(filepath, candidate_format)
                return readtable(candidate; folderpath=folderpath, format=candidate_format, preferred=preferred, materialize=materialize, copycols=copycols)
            end
        end
        (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(normpath(joinpath(folderpath, filename)))")
        return nothing
    end

    @assert format in (:jdf, :arrow) "format=$(format) must be :auto, :jdf or :arrow"
    filepath = tablepath(filename; folderpath=folderpath, format=format)
    if !_table_storage_exists(filepath, format)
        (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(filepath)")
        return nothing
    end

    if format == :jdf
        (verbosity >= 4) && print("$(EnvConfig.now()) loading JDF dataframe from $(filepath) ... ")
        df = DataFrame(JDF.loadjdf(filepath))
        (verbosity >= 4) && println("$(EnvConfig.now()) loaded $(size(df, 1)) rows successfully")
        return df
    end

    (verbosity >= 4) && print("$(EnvConfig.now()) loading Arrow table from $(filepath) ... ")
    table = Arrow.Table(filepath)
    if materialize
        df = DataFrame(table; copycols=copycols)
        (verbosity >= 4) && println("$(EnvConfig.now()) loaded $(size(df, 1)) rows successfully")
        return df
    end
    return table
end

"Save a dataframe using the preferred or explicitly supplied storage format."
savedf(df, filename; folderpath=logfolder(), format::Symbol=dfformat()) = writetable(df, filename; folderpath=folderpath, format=format)

"Read a dataframe using the preferred or explicitly supplied storage format, with fallback to the alternate format when available. Use `copycols=true` only when the caller intends to mutate the returned dataframe."
readdf(filename; folderpath=logfolder(), format::Symbol=dfformat(), copycols::Bool=false) = readtable(filename; folderpath=folderpath, format=:auto, preferred=format, copycols=copycols)

"Backward-compatible cache existence check that works for both JDF directories and Arrow files."
isfolder(filename; folderpath=logfolder()) = tableexists(filename; folderpath=folderpath, format=:auto)

function deletefolder(filename; folderpath=logfolder())
    deleted = false
    candidates = unique(vcat([filename], _table_candidates(filename; preferred=dfformat())))
    for candidate in candidates
        filepath = normpath(joinpath(folderpath, candidate))
        if isdir(filepath) || isfile(filepath)
            (verbosity >= 3) && println("$(EnvConfig.now()) deleting $(filepath)")
            rm(filepath; force=true, recursive=true)
            deleted = true
        end
    end
    if !deleted
        (verbosity >= 3) && println("$(EnvConfig.now()) no deletion due to missing $(normpath(joinpath(folderpath, filename)))")
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

"""Return the local root folder for shared per-coin Arrow artifacts created during Phase 2, colocated with `logs/` under `~/crypto/coins`."""
function coinspath()
    isdir(defaultcoinspath) || mkpath(defaultcoinspath)
    return defaultcoinspath
end

"""
    coinfolderpath(basecoin::AbstractString, quotecoin::AbstractString=cryptoquote, subfolder=nothing)

Return the folder for a shared coin-pair artifact under `coins/`, creating it on
first use. Examples include `coins/BTCUSDT/ohlcv.arrow` and
`coins/BTCUSDT/f4/grad_15.arrow`.
"""
function coinfolderpath(basecoin::AbstractString, quotecoin::AbstractString=cryptoquote, subfolder=nothing)
    pairfolder = uppercase(basecoin) * "-" * uppercase(quotecoin)
    folderpath = isnothing(subfolder) ? normpath(joinpath(coinspath(), pairfolder)) : normpath(joinpath(coinspath(), pairfolder, subfolder))
    if !isdir(folderpath)
        println("EnvConfig $(now()): creating folder $folderpath")
        mkpath(folderpath)
    end
    return folderpath
end

"""
    coinfile(basecoin::AbstractString, quotecoin::AbstractString, artifact::AbstractString;
             subfolder=nothing, extension=".arrow")

Return the full path of a shared per-coin artifact file under `coins/` without
changing the legacy JDF storage paths.
"""
function coinfile(basecoin::AbstractString, quotecoin::AbstractString, artifact::AbstractString; subfolder=nothing, extension=".arrow")
    ext = startswith(extension, ".") ? extension : "." * extension
    return normpath(joinpath(coinfolderpath(basecoin, quotecoin, subfolder), artifact * ext))
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


