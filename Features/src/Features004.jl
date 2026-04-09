
#region Features004
"Features004 is a simplified subset of Features002 without regression extremes and relative volume but with save and read functions and implemented as DataFrame"

regressionwindows004 = [60, 4*60, 12*60, 24*60, 3*24*60, 10*24*60]
regressionwindows004dict = Dict("1h" => 1*60, "4h" => 4*60, "12h" => 12*60, "1d" => 24*60, "3d" => 3*24*60, "10d" => 10*24*60)

"""
Provides per regressionwindow gradient, regression line price, standard deviation.
"""
mutable struct Features004 <: AbstractFeatures
    basecoin::String
    quotecoin::String
    ohlcvoffset
    rw::Dict{Integer, AbstractDataFrame}  # keys: regression window in minutes, values: dataframe with columns :opentime, :regry, :grad, :std
    latestloadeddt  # nothing or latest DateTime of loaded data
    # opentime::Vector{DateTime}
    # grad::Vector{Float32} # rolling regression gradients; length == ohlcv - requiredminutes
    # regry::Vector{Float32}  # rolling regression price; length == ohlcv - requiredminutes
    # std::Vector{Float32}  # standard deviation of regression window; length == ohlcv - requiredminutes
    Features004(basecoin::String, quotecoin::String) = new(basecoin, quotecoin, nothing, Dict(), nothing)
end

"Provides Features004 of the given ohlcv within the requested time range. Canned data will be read and supplemented with calculated data."
function Features004(ohlcv; firstix=firstindex(ohlcv.df.opentime), lastix=lastindex(ohlcv.df.opentime), regrwindows=regressionwindows004, usecache=false)::Union{Nothing, Features004}
    startix = maxregrwindow = maximum(regrwindows)  # all dataframes to start at the same time to make performance comparable and f4 handling easier
    f4 = Features004(ohlcv.base, ohlcv.quotecoin)
    df = Ohlcv.dataframe(ohlcv)
    if !(firstindex(df[!, :opentime]) <= startix <= lastix <= lastindex(df[!, :opentime])) || ((lastix - (max(firstix, maxregrwindow)-maxregrwindow)) < maxregrwindow)
        (verbosity >= 2) && @warn "$(ohlcv.base): $(firstindex(df[!, :opentime])) <= $startix <= $lastix <= $(lastindex(df[!, :opentime])); size(dfv, 1)=$((lastix - (max(firstix, maxregrwindow)-maxregrwindow))) < maxregrwindow=$maxregrwindow"
        return nothing
    end
    dfv = view(df, (max(firstix, maxregrwindow)-maxregrwindow+1):lastix, :)
    if usecache
        f4 = read!(f4, dfv[startix, :opentime], dfv[end, :opentime])
    end
    for window in regrwindows
        if !(window in keys(f4.rw))
            f4.rw[window] = DataFrame(opentime=DateTime[], regry=Float32[], grad=Float32[], std=Float32[])
        end
    end
    return isnothing(supplement!(f4, ohlcv; firstix=firstix, lastix=lastix)) ? nothing : f4
end

# obsolete TODO add regression test
mnemonic(f4::Features004) = uppercase(f4.basecoin) * "_" * uppercase(f4.quotecoin) * "_" * "_F4"
ohlcvix(f4::Features004, featureix) = featureix + f4.ohlcvoffset
featureix(f4::Features004, ohlcvix) = ohlcvix - f4.ohlcvoffset
relativedaygain(f4::Features.Features004, regrminutes::Integer, featuresix::Integer) = relativegain(regry(f4, regrminutes)[featuresix], grad(f4, regrminutes)[featuresix], 24*60)

function consistent(f4::Features004, ohlcv::Ohlcv.OhlcvData)
    checkok = true
    if ohlcv.base != f4.basecoin
        @warn "bases of ohlcv=$(ohlcv.base) != f4=$(f4.basecoin)"
        checkok = false
    end
    for (rw, rwdf) in f4.rw
        if rwdf[end,:opentime] != ohlcv.df[end, :opentime]
            @warn "f4 of $(ohlcv.base) rw[$rw][end,:opentime]=$(rwdf[end,:opentime]) != ohlcv.df[end, :opentime]=$(ohlcv.df[end, :opentime])"
            checkok = false
        end
        if rwdf[begin,:opentime] - Minute(requiredminutes(f4)-1) != ohlcv.df[begin, :opentime]
            @warn "f4 of $(ohlcv.base) rw[$rw][begin,:opentime]-requiredminutes(f4)+1=$(rwdf[begin,:opentime]-Minute(requiredminutes(f4)-1)) != ohlcv.df[begin, :opentime]=$(ohlcv.df[begin, :opentime])"
            checkok = false
        end
        if isnothing(f4.ohlcvoffset)
            @warn "isnothing(f4.ohlcvoffset) of $(ohlcv.base) "
            checkok = false
        elseif rwdf[begin,:opentime] != ohlcv.df[ohlcvix(f4, firstindex(rwdf[!,:opentime])), :opentime]
            @warn "f4 of $(ohlcv.base) rw[$rw][begin,:opentime]=$(rwdf[begin,:opentime]) != ohlcv.df[ohlcvix(f4, firstindex(rwdf[!,:opentime])), :opentime]=$(ohlcv.df[ohlcvix(f4, firstindex(rwdf[!,:opentime])), :opentime])"
            checkok = false
        end
    end
    startequal = all([rwdf[begin,:opentime] == first(values(f4.rw))[begin,:opentime] for (rw, rwdf) in f4.rw])
    if !startequal
        @warn "different F4 start dates: $([(regr=rw, startdt=rwdf[begin,:opentime]) for (rw, rwdf) in f4.rw])"
        checkok = false
    end
    endequal = all([rwdf[end,:opentime] == first(values(f4.rw))[end,:opentime] for (rw, rwdf) in f4.rw])
    if !endequal
        @warn "different F4 end dates: $([(regr=rw, enddt=rwdf[end,:opentime]) for (rw, rwdf) in f4.rw])"
        checkok = false
    end
    lengthequal = all([size(rwdf, 1) == size(first(values(f4.rw)), 1) for (rw, rwdf) in f4.rw])
    if !lengthequal
        @warn "different F4 data length: $([(regr=rw, length=size(rwdf, 1)) for (rw, rwdf) in f4.rw])"
        checkok = false
    end
    return checkok
end

function featureoffset!(f4::Features004, ohlcv::Ohlcv.OhlcvData)
    f4.ohlcvoffset = nothing
    if (length(f4.rw) > 0) && (size(first(values(f4.rw)), 1) > 0) && (size(ohlcv.df, 1) > 0)
        fix = nothing
        f4df = first(values(f4.rw))
        # fix = (ohlcv.df[begin,:opentime] <= f4df[begin, :opentime]) && (ohlcv.df[end,:opentime] >= f4df[begin, :opentime]) ? firstindex(f4df[!, :opentime]) : (ohlcv.df[begin,:opentime] <= f4df[end, :opentime] && ohlcv.df[end,:opentime] >= f4df[end, :opentime] ?  lastindex(f4df[!, :opentime]) : nothing)
        if ohlcv.df[begin,:opentime] <= f4df[begin, :opentime] <= ohlcv.df[end,:opentime]
            fix = firstindex(f4df[!, :opentime])
        elseif ohlcv.df[begin,:opentime] <= f4df[end, :opentime] <= ohlcv.df[end,:opentime]
            fix = lastindex(f4df[!, :opentime])
        end
        if !isnothing(fix)
            oix = Ohlcv.rowix(ohlcv.df[!,:opentime], f4df[fix, :opentime])
            if f4df[fix, :opentime] == ohlcv.df[oix, :opentime]
                f4.ohlcvoffset = oix - fix
            end
        else
            (verbosity >= 3) && @warn "could not calc $(ohlcv.base) f4offset ohlcv.begin=$(ohlcv.df[begin,:opentime]), ohlcv.end=$(ohlcv.df[end,:opentime]), f4df.begin=$(f4df[begin, :opentime]), f4df.end=$(f4df[end, :opentime])"
        end
    end
    return f4.ohlcvoffset
end

function _equaltimes(f4)
    times = [(df[begin, :opentime], size(df, 1), df[end, :opentime]) for df in values(f4.rw)]
    if length(times) > 0
        if all([times[1] == t for t in times])
            return true
        else
            times = [(regr=regr, first=df[begin, :opentime], length=size(df, 1), last=df[end, :opentime]) for (regr, df) in f4.rw]
            @warn "F4 dataframes not equal: $times"
            return false
        end
    else
        @warn "F4 dataframes missing"
        return false
    end
end

function _join(f4)
    df = DataFrame()
    for (regr, rdf) in f4.rw
        for cname in names(rdf)
            if cname == "opentime"
                df[:, cname] = rdf[!, cname]
            else
                df[:, join([string(regr), cname], "_")] = rdf[!, cname]
            end
        end
    end
    return df
end

function timerangecut!(f4::Features004, startdt, enddt)
    (length(f4.rw) == 0) && @warn "empty f4 for $(f4.basecoin)"
    for (regr, rdf) in f4.rw
        if isnothing(rdf) || (size(rdf, 1) == 0)
            @warn "unexpected missing f4 data $f4"
            return
        end
        startdt = isnothing(startdt) ? rdf[begin, :opentime] : startdt
        startix = Ohlcv.rowix(rdf[!, :opentime], startdt)
        enddt = isnothing(enddt) ? rdf[end, :opentime] : enddt
        endix = Ohlcv.rowix(rdf[!, :opentime], enddt)
        f4.rw[regr] = rdf[startix:endix, :]
        # println("startdt=$startdt enddt=$enddt size(rdf)=$(size(rdf)) rdf=$(describe(rdf, :all)) ")
        # if !isnothing(startdt) && !isnothing(enddt)
        #     subset!(rdf, :opentime => t -> floor(startdt, Minute(1)) .<= t .<= floor(enddt, Minute(1)))
        # elseif !isnothing(startdt)
        #     subset!(rdf, :opentime => t -> floor(startdt, Minute(1)) .<= t)
        # elseif !isnothing(enddt)
        #     subset!(rdf, :opentime => t -> t .<= floor(enddt, Minute(1)))
        # end
    end
end

function _split!(f4, df)
    @assert length(f4.rw) == 0
    ot = nothing
    for cname in names(df)
        if cname == "opentime"
            ot = df[!, cname]
        else
            cnamevec = split(cname, "_")
            if length(cnamevec) != 2
                @error "unexpected f4.rw dataframe column name: $cnamevec"
            end
            regr = parse(Int, cnamevec[1])
            if !(regr in keys(f4.rw))
                f4.rw[regr] = DataFrame()
            end
            f4.rw[regr][:, cnamevec[2]] = df[!, cname]
        end
    end
    for (regr, rdf) in f4.rw
        rdf[:, "opentime"] = ot
    end
    return f4
end

function file(f4::Features004)
    mnm = mnemonic(f4)
    filename = EnvConfig.datafile(mnm, "Features004")
    if isdir(filename)
        return (filename=filename, existing=true)
    else
        return (filename=filename, existing=false)
    end
end

_f4_arrow_filename(name::AbstractString) = replace(String(name), r"[^A-Za-z0-9_.-]" => "_")

"""Write non-destructive per-column Arrow copies of shared F4 data to `coins/<pair>/f4/`."""
function _write_shared_f4_arrow(basecoin::AbstractString, quotecoin::AbstractString, df::AbstractDataFrame)
    if (size(df, 1) == 0) || !("opentime" in names(df))
        return String[]
    end
    folderpath = EnvConfig.coinfolderpath(basecoin, quotecoin, "f4")
    outpaths = String[]
    for col in names(df)
        if String(col) == "opentime"
            continue
        end
        colsym = col isa Symbol ? col : Symbol(col)
        push!(outpaths, EnvConfig.savedf(select(df, :opentime, colsym), _f4_arrow_filename(String(col)); folderpath=folderpath, format=:arrow))
    end
    return outpaths
end

"""Read shared F4 Arrow column files from `coins/<pair>/f4/` and rebuild the combined dataframe."""
function _read_shared_f4_arrow(basecoin::AbstractString, quotecoin::AbstractString; columns::Union{Nothing,AbstractVector}=nothing)
    folderpath = EnvConfig.coinfolderpath(basecoin, quotecoin, "f4")
    isdir(folderpath) || return DataFrame()

    stems = if isnothing(columns)
        [splitext(file)[1] for file in readdir(folderpath; join=false, sort=true) if endswith(file, ".arrow")]
    else
        [_f4_arrow_filename(String(col)) for col in columns if String(col) != "opentime"]
    end
    isempty(stems) && return DataFrame()

    pieces = DataFrame[]
    for stem in stems
        df = EnvConfig.readdf(stem; folderpath=folderpath, format=:arrow)
        if !isnothing(df) && (size(df, 1) > 0)
            push!(pieces, DataFrame(df))
        end
    end
    isempty(pieces) && return DataFrame()

    result = pieces[1]
    for piece in pieces[2:end]
        result = innerjoin(result, piece; on=:opentime)
    end
    sort!(result, :opentime)
    return result
end

"""Write non-destructive per-column Arrow copies of `Features004` shared cache data."""
write_arrow(f4::Features004) = _write_shared_f4_arrow(f4.basecoin, f4.quotecoin, _join(f4))

function write(f4::Features004)
    @assert _equaltimes(f4)
    df = _join(f4)
    if !isnothing(f4.latestloadeddt) && (f4.latestloadeddt >= df[end, :opentime])
        (verbosity >= 3) && println("$(EnvConfig.now()) F4 not written due to missing supplementations of already stored data")
        return
    end
    fn = file(f4)
    try
        JDF.savejdf(fn.filename, df[!, :])
        arrowpaths = _write_shared_f4_arrow(f4.basecoin, f4.quotecoin, df)
        (verbosity >= 2) && println("$(EnvConfig.now()) saved F4 data of $(f4.basecoin) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows to $(fn.filename)")
        (verbosity >= 3) && !isempty(arrowpaths) && println("$(EnvConfig.now()) saved $(length(arrowpaths)) shared F4 Arrow column files for $(f4.basecoin)")
    catch e
        Logging.@error "exception $e detected when writing $(fn.filename)"
    end
end

read!(f4::Features004)::Features004 = read!(f4, nothing, nothing)

function read!(f4::Features004, startdt, enddt)::Features004
    fn = file(f4)
    # try
        df = _read_shared_f4_arrow(f4.basecoin, f4.quotecoin)
        if size(df, 1) > 0
            startdt = isnothing(startdt) ? df[begin, :opentime] : startdt
            enddt = isnothing(enddt) ? df[end, :opentime] : enddt
            (verbosity >= 2) && println("$(EnvConfig.now()) loaded shared F4 Arrow data of $(f4.basecoin) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows")
            if (startdt <= df[end, :opentime]) && (enddt >= df[begin, :opentime])
                f4.latestloadeddt = df[end, :opentime]
                startix = Ohlcv.rowix(df[!, :opentime], startdt)
                startix = df[startix, :opentime] < startdt ? min(lastindex(df[!, :opentime]), startix+1) : startix
                endix = Ohlcv.rowix(df[!, :opentime], enddt)
                endix = df[endix, :opentime] > enddt ? max(firstindex(df[!, :opentime]), endix-1) : endix
                df = df[startix:endix, :]
                f4 = _split!(f4, df)
            end
        elseif fn.existing
            (verbosity >= 3) && println("$(EnvConfig.now()) start loading F4 data of $(f4.basecoin) from $(fn.filename)")
            df = DataFrame(JDF.loadjdf(fn.filename))
            startdt = isnothing(startdt) ? df[begin, :opentime] : startdt
            enddt = isnothing(enddt) ? df[end, :opentime] : enddt
            (verbosity >= 2) && println("$(EnvConfig.now()) loaded F4 data of $(f4.basecoin) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows from $(fn.filename)")
            if (size(df, 1) > 0) && (startdt <= df[end, :opentime]) && (enddt >= df[begin, :opentime])
                (verbosity >= 4) && println("f4cache $(fn.filename) names: $(names(df))")
                f4.latestloadeddt = df[end, :opentime]
                startix = Ohlcv.rowix(df[!, :opentime], startdt)
                startix = df[startix, :opentime] < startdt ? min(lastindex(df[!, :opentime]), startix+1) : startix
                endix = Ohlcv.rowix(df[!, :opentime], enddt)
                endix = df[endix, :opentime] > enddt ? max(firstindex(df[!, :opentime]), endix-1) : endix
                df = df[startix:endix, :]
                f4 = _split!(f4, df)
            end
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(fn.filename) and no shared F4 Arrow copy for $(f4.basecoin)")
        end
    # catch e
    #     Logging.@warn "exception $e detected"
    # end
    return f4
end

function delete(f4::Features004)
    fn = file(f4)
    if fn.existing
        rm(fn.filename; force=true, recursive=true)
    end
end

mutable struct Features004Files
    filenames
    Features004Files() = new(nothing)
end

# function Base.iterate(of::OhlcvFiles, state=1)
# end

function Base.iterate(f4f::Features004Files, state=1)
    if isnothing(f4f.filenames)
        allff = readdir(EnvConfig.datafolderpath("Features004"), join=false, sort=false)
        fileixlist = findall(f -> endswith(f, "_F4.jdf"), allff)
        f4f.filenames = [allff[ix] for ix in fileixlist]
        if length(f4f.filenames) > 0
            state = firstindex(f4f.filenames)
        else
            return nothing
        end
    end
    if state > lastindex(f4f.filenames)
        return nothing
    end
    # fn = split(of.filenames[state], "/")[end]
    fnparts = split(f4f.filenames[state], "_")
    # return (basecoin=fnparts[1], quotecoin=fnparts[2], interval=fnparts[3]), state+1
    basecoin=fnparts[1]
    quotecoin=fnparts[2]
    f4 = Features.Features004(String(basecoin), String(quotecoin))
    read!(f4, nothing, nothing)
    return f4, state+1
end

"Supplements Features004 with the newest ohlcv datapoints, i.e. datapoints newer than last(f4)"
function supplement!(f4::Features004, ohlcv::Ohlcv.OhlcvData; firstix=firstindex(ohlcv.df[!, :opentime]), lastix=lastindex(ohlcv.df[!, :opentime]))
    usecache = (length(f4.rw) > 0) && (size(first(values(f4.rw)), 1) > 0)
    Ohlcv.pivot!(ohlcv)
    df = Ohlcv.dataframe(ohlcv)
    maxregrwindow = maximum(regrwindows(f4))  # all dataframes to start at the same time to make performance comparable and f4 handling easier
    if !(firstindex(df[!, :opentime]) <= firstix <= lastix <= lastindex(df[!, :opentime])) || ((lastix - (max(firstix, maxregrwindow)-maxregrwindow)) < maxregrwindow)
        @warn "$(firstindex(df[!, :opentime])) <= $firstix <= $lastix <= $(lastindex(df[!, :opentime])); size(dfv, 1)=$((lastix - (max(firstix, maxregrwindow)-maxregrwindow))) < maxregrwindow=$maxregrwindow"
        return nothing
    end
    # dfv = view(df, (max(firstix, maxregrwindow)-maxregrwindow+1):lastix, :)
    # dfv = view(df, max(firstix-maxregrwindow+1, 1):lastix, :)
    pivot = df.pivot
    startafterix = endbeforeix = nothing
    ot = df[!, "opentime"]
    if usecache
        otstored = first(values(f4.rw))[!, "opentime"]
        endbeforeix = Ohlcv.rowix(ot, otstored[begin]) - 1
        endbeforeix = endbeforeix < firstix ? nothing : endbeforeix
        startafterix = Ohlcv.rowix(ot, otstored[end]) + 1
        startafterix = startafterix > lastindex(ot) ? nothing : startafterix
    end
    for window in regrwindows(f4)
        if usecache && (window in keys(f4.rw)) && (size(f4.rw[window], 1) > 0)
            if !isnothing(endbeforeix)
                dfv = view(df, 1:endbeforeix, :)
                (verbosity >= 3) && println("$(EnvConfig.now()) F4 endbeforeix=$endbeforeix with window=$window and firstix=$firstix for $(endbeforeix-firstix+1) rows")
                regry, grad = rollingregression(dfv.pivot, window, firstix)
                std = rollingregressionstdmv([dfv.open, dfv.high, dfv.low, dfv.close], regry, grad, window, firstix)
                dft = DataFrame(opentime=view(ot, firstix:endbeforeix), regry=regry, grad=grad, std=std)
                prepend!(f4.rw[window], dft)
            end
            if !isnothing(startafterix)
                dfv = view(df, 1:lastix, :)
                (verbosity >= 3) && println("$(EnvConfig.now()) F4 startafterix=$startafterix with window=$window for $(size(df, 1)-startafterix+1) rows")
                regry, grad = rollingregression(dfv.pivot, window, startafterix)
                std = rollingregressionstdmv([dfv.open, dfv.high, dfv.low, dfv.close], regry, grad, window, startafterix)
                dft = DataFrame(opentime=view(ot, startafterix:lastix), regry=regry, grad=grad, std=std)
                append!(f4.rw[window], dft)
            end
        else
            dfv = view(df, 1:lastix, :)
            (verbosity >= 3) && println("$(EnvConfig.now()) F4 full calc from firstix=$firstix until lastix=$lastix with window=$window for $(lastix-firstix+1) rows")
            regry, grad = rollingregression(dfv.pivot, window, firstix)
            std = rollingregressionstdmv([dfv.open, dfv.high, dfv.low, dfv.close], regry, grad, window, firstix)
            f4.rw[window] = DataFrame(opentime=view(ot, firstix:lastix), regry=regry, grad=grad, std=std)
        end
    end
    return isnothing(featureoffset!(f4, ohlcv)) ? nothing : f4
end

requiredminutes(f4::Features004) = maximum(regrwindows(f4))

function Base.show(io::IO, f4::Features004)
    print(io, "Features004 base=$(f4.basecoin) quote=$(f4.quotecoin) offset=$(f4.ohlcvoffset) from $(first(values(f4.rw))[begin, :opentime]) until $(first(values(f4.rw))[end, :opentime]), $(["$regr:size=$(size(df)) names=$(names(df)), " for (regr,df) in f4.rw])")
    # for (key, value) in f4.rw
    #     println(io, "Features004 base=$(f4.basecoin), regr key: $key of size=$(size(value))")
    #     (verbosity >= 3) && println(io, describe(value, :first, :last, :min, :mean, :max, :nuniqueall, :nnonmissing, :nmissing, :eltype))
    # end
end

grad(f4::Features004, regrminutes) =     f4.rw[regrminutes][!, :grad]
regry(f4::Features004, regrminutes) =    f4.rw[regrminutes][!, :regry]
std(f4::Features004, regrminutes) =      f4.rw[regrminutes][!, :std]
opentime(f4::Features004, regrminutes) = f4.rw[regrminutes][!, :opentime]
opentime(f4::Features004) = first(f4.rw)[2][!, :opentime]  # opentime array from all rw members shall start and end equally
regrwindows(f4::Features004) = keys(f4.rw)

function features(f4::Features004, firstix, lastix)::AbstractDataFrame
    #TODO
    @error "to be implemented"
end

#endregion Features004
