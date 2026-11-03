"""
Refresh shared feature and target stores and rebuild sets stores from full OHLCV history.

Usage:
  julia --project=. scripts/refresh.jl features=F19,F20 targets=T24,T25
  julia --project=. scripts/refresh.jl features=F19 targets=T24 pairs=BTC-USDT,ETH-USDT
  julia --project=. scripts/refresh.jl targets=T24 xch=KrakenSpot pairs=XBT-USDC
  julia --project=. scripts/refresh.jl partitions=P02_D01,P02_D00

Without `pairs`, all cached pairs under the selected exchange folder are processed.
Supported exchange folders are Bybit, BybitSim, IbkrSpot, KrakenFutures, and KrakenSpot.
"""
module Refresh

using Dates, DataFrames
using Classify, EnvConfig, Features, Ohlcv, Targets, TradingStrategy, Xch

const VALID_OPTIONS = ("features", "targets", "partitions", "pairs", "xch", "ts")
const HELP_FLAGS = ("help", "--help", "-h")
const DEFAULT_QUOTES = Dict(
    Xch.EXCHANGE_BYBIT => "USDT",
    Xch.EXCHANGE_BYBITSIM => "USDT",
    Xch.EXCHANGE_IBKRSPOT => "USD",
    Xch.EXCHANGE_KRAKENFUTURES => "USD",
    Xch.EXCHANGE_KRAKENSPOT => "USDC",
)

"""Return strict, unique key/value options from the CLI arguments."""
function parseoptions(args::Vector{String})::Dict{String,String}
    options = Dict{String,String}()
    for arg in args
        parts = split(arg, "="; limit=2)
        @assert length(parts) == 2 && first(parts) in VALID_OPTIONS "unknown argument=$arg; expected key=value with keys $(join(VALID_OPTIONS, ", "))"
        @assert !haskey(options, first(parts)) "duplicate argument=$(first(parts))"
        options[first(parts)] = strip(last(parts))
    end
    return options
end

"""Normalize and validate a comma-separated list of registry IDs."""
function configids(raw::AbstractString, kind::AbstractString)::Vector{String}
    ids = uppercase.(strip.(split(String(raw), ",")))
    @assert !isempty(ids) && all(!isempty, ids) && allunique(ids) "invalid $(kind) config list=$(raw)"
    for id in ids
        if kind == "features"
            TradingStrategy.featureconfig(id)
        elseif kind == "targets"
            TradingStrategy.targetconfig(id)
        else
            @assert kind == "partitions" "unsupported config kind=$kind"
            parts = split(id, "_")
            @assert length(parts) == 2 && all(!isempty, parts) "invalid partitions ID=$(id); expected Pxx_Dxx"
            TradingStrategy.partitionconfig(parts[1])
            TradingStrategy.daterangeconfig(parts[2])
        end
    end
    return ids
end

"""Validate unique comma-separated strategy configs supported by RegimeTraining."""
function regimeconfigs(raw::AbstractString)::Vector{String}
    tokens = strip.(split(String(raw),","))
    @assert !isempty(tokens) && all(!isempty,tokens) "invalid ts config list=$raw"
    configs = [TradingStrategy.tsconfig(token) for token in tokens]
    ids = [config.configname for config in configs]
    @assert allunique(ids) "duplicate ts configs=$ids"
    for config in configs
        classifier = TradingStrategy.classifierconfig(config.classifier)
        @assert classifier.model in ("M009","M011") "RegimeTraining requires SimpleChains: ts=$(config.configname) model=$(classifier.model)"
    end
    return ids
end

"""Return the canonical exchange name and its default quote currency."""
function exchangecfg(raw::AbstractString)::Tuple{String,String}
    key = lowercase(strip(String(raw)))
    matches = [exchange for exchange in keys(DEFAULT_QUOTES) if lowercase(exchange) == key]
    @assert length(matches) == 1 "unsupported xch=$(raw); expected one of $(join(sort!(collect(keys(DEFAULT_QUOTES))), ", "))"
    exchange = only(matches)
    return exchange, DEFAULT_QUOTES[exchange]
end

"""Find cached base/quote folders containing an OHLCV Arrow file."""
function cachedpairs(root::AbstractString)::Vector{String}
    @assert isdir(root) "exchange coin folder does not exist: $root"
    pairs = String[]
    for entry in readdir(root)
        folder = joinpath(root, entry)
        isdir(folder) && isfile(joinpath(folder, "ohlcv.arrow")) || continue
        parts = split(entry, "-"; limit=2)
        length(parts) == 2 && all(!isempty, parts) || continue
        push!(pairs, uppercase(entry))
    end
    return sort!(pairs)
end

"""Resolve user pair tokens to cached base/quote folders."""
function resolvepairs(raw::Union{Nothing,AbstractString}, available::Vector{String},
        defaultquote::AbstractString)::Vector{String}
    isnothing(raw) && return available
    tokens = uppercase.(strip.(split(String(raw), ",")))
    @assert !isempty(tokens) && all(!isempty, tokens) && allunique(tokens) "invalid pair list=$(raw)"
    pairs = [occursin("-", token) ? token : "$(token)-$(defaultquote)" for token in tokens]
    missing = sort!(collect(setdiff(Set(pairs), Set(available))))
    @assert isempty(missing) "requested pairs have no cached OHLCV files: $(join(missing, ", "))"
    return pairs
end

"""
Run `step` and continue on failure.
Any exception except `InterruptException` is logged, recorded in `failures` as
`label => error message`, and skipped. Return `true` on success.
"""
function guarded!(step::Function, failures::Vector{Pair{String,String}}, label::AbstractString)::Bool
    try
        step()
        return true
    catch error
        error isa InterruptException && rethrow()
        @error "$(EnvConfig.now()) $(label) failed; skipped" exception=(error, catch_backtrace())
        push!(failures, String(label) => sprint(showerror, error))
        return false
    end
end

"""Refresh and verify the shared feature store `id` covering the full OHLCV history of `pair`."""
function refreshfeatures!(ohlcv::Ohlcv.OhlcvData, pair::AbstractString, id::AbstractString)::Nothing
    featurestore = Features.updatefeatures!(ohlcv, id, TradingStrategy.featureconfig(id);
        confighash=TradingStrategy.featureshash(id))
    required = Features.requiredminutes(TradingStrategy.featureconfig(id))
    df = Ohlcv.dataframe(ohlcv)
    expectedrows = nrow(df) - required + 1
    @assert (nrow(featurestore) == expectedrows) &&
        (featurestore.opentime[begin] == df.opentime[required]) &&
        (featurestore.opentime[end] == df.opentime[end]) "feature store $(id) of $(pair) is incomplete: rows=$(nrow(featurestore)), expected=$(expectedrows)"
    println("$(EnvConfig.now()) $(pair) refreshed features $(id): $(nrow(featurestore)) rows")
    return nothing
end

"""Refresh and verify the shared target store `id` covering the full OHLCV history of `pair`."""
function refreshtargets!(ohlcv::Ohlcv.OhlcvData, pair::AbstractString, id::AbstractString)::Nothing
    targetstore = Targets.updatetargets!(ohlcv, id, TradingStrategy.targetconfig(id);
        confighash=TradingStrategy.targetshash(id))
    df = Ohlcv.dataframe(ohlcv)
    @assert (nrow(targetstore) == nrow(df)) &&
        (targetstore.opentime[begin] == df.opentime[begin]) &&
        (targetstore.opentime[end] == df.opentime[end]) "target store $(id) of $(pair) is incomplete: rows=$(nrow(targetstore)), expected=$(nrow(df))"
    println("$(EnvConfig.now()) $(pair) refreshed targets $(id): $(nrow(targetstore)) rows")
    return nothing
end

"""
Rebuild the sets store `id` (`Pxx_Dxx`) from complete OHLCV history using current registry
settings, explicitly replacing any existing store regardless of its config hash.
Return its assignment count, optionally capture assigned set types, and warn when the rebuilt store is empty.
"""
function refreshpartitions!(ohlcv::Ohlcv.OhlcvData, pair::AbstractString, id::AbstractString;
        assignedsets::Union{Nothing,Set{String}}=nothing)::Int
    id = only(configids(id, "partitions"))
    partitionid, daterangeid = split(id, "_")
    dr = TradingStrategy.daterangeconfig(daterangeid)
    hash = TradingStrategy.setshash(partitionid, daterangeid)
    store = Classify.updatesets!(ohlcv, id, TradingStrategy.partitionconfig(partitionid),
        dr.startdt, dr.enddt; confighash=hash, rebuild=true)
    @assert names(store) == ["opentime", "set", "rangeid"] "sets store $(id) of $(pair) has unexpected columns=$(names(store))"
    @assert issorted(store.opentime) && allunique(store.opentime) "sets store $(id) of $(pair) has unordered or duplicate timestamps: rows=$(nrow(store))"
    @assert all(dt -> dr.startdt <= dt <= dr.enddt, store.opentime) "sets store $(id) of $(pair) contains timestamps outside $(dr.startdt)..$(dr.enddt)"
    isnothing(assignedsets) || (empty!(assignedsets); union!(assignedsets,store.set))
    println("$(EnvConfig.now()) $(pair) rebuilt partitions $(id): $(nrow(store)) rows")
    if isempty(store.opentime)
        @warn "rebuilt sets store has zero assignments; no liquid range within the configured dates was long enough for partitioning" pair id startdt=dr.startdt enddt=dr.enddt
    end
    return nrow(store)
end

"""
Refresh the requested shared feature, target and partition artifacts for one complete OHLCV history.
A failing OHLCV load skips the pair; a failing artifact config skips only that
config. Failures are recorded in `failures` as `"<pair> <id>" => error message`.
Successfully rebuilt empty stores are recorded separately in `zeroassignments` as `pair => id`.
"""
function refreshpair!(failures::Vector{Pair{String,String}}, pair::AbstractString,
        features::Vector{String}, targets::Vector{String}, partitions::Vector{String};
        zeroassignments::Vector{Pair{String,String}}=Pair{String,String}[],
        partitionmemberships::Dict{Tuple{String,String},Set{String}}=Dict{Tuple{String,String},Set{String}}())::Nothing
    ohlcv = nothing
    guarded!(failures, "$(pair) OHLCV") do
        base, quotecoin = split(String(pair), "-"; limit=2)
        EnvConfig.setpairquote!(quotecoin)
        ohlcv = Ohlcv.read(base)
        @assert nrow(Ohlcv.dataframe(ohlcv)) > 1 "cannot refresh $(pair): OHLCV rows=$(nrow(Ohlcv.dataframe(ohlcv)))"
    end || return nothing
    for id in features
        guarded!(() -> refreshfeatures!(ohlcv, pair, id), failures, "$(pair) $(id)")
    end
    for id in targets
        guarded!(() -> refreshtargets!(ohlcv, pair, id), failures, "$(pair) $(id)")
    end
    for id in partitions
        guarded!(failures, "$(pair) $(id)") do
            assignedsets = Set{String}()
            count = refreshpartitions!(ohlcv, pair, id;assignedsets)
            partitionmemberships[(String(pair),id)] = assignedsets
            count == 0 && push!(zeroassignments, String(pair) => id)
        end
    end
    return nothing
end

"""Print copy-pasteable base-coin lists whose partition stores contain train, eval and test rows."""
function reportpartitioncoins(partitions::Vector{String},
        memberships::Dict{Tuple{String,String},Set{String}};io::IO=stdout)::Nothing
    requiredsets = Set(["train","eval","test"])
    println(io,"\nCopy-pasteable coin sets (coins with train, eval and test assignments):")
    for id in partitions
        pairs = sort!([pair for ((pair,partitionid),sets) in memberships
            if partitionid==id && issubset(requiredsets,sets)])
        quotecoins = sort!(unique([string(last(split(pair,"-";limit=2))) for pair in pairs]))
        for quotecoin in quotecoins
            coins = sort!(unique([string(first(split(pair,"-";limit=2))) for pair in pairs
                if last(split(pair,"-";limit=2))==quotecoin]))
            println(io,"\n# $(id), quote=$(quotecoin), $(length(coins)) coins")
            println(io,"traincoins() = ",repr(coins))
        end
        isempty(quotecoins) && println(io,"\n# $(id): no coin has train, eval and test assignments")
    end
    flush(io)
    return nothing
end

"""Report per-coin full regime/fold eligibility and a copy-pasteable complete-population list."""
function reportregimecoins(sets::AbstractDict, coins::Vector{String},
        ts::AbstractString, daterange::AbstractString;io::IO=stdout)::Nothing
    accepted = String[]
    println(io,"\nRegimeTraining eligibility: ts=$ts, daterange=$daterange, quote=USDT")
    for coin in sort(coins)
        if !haskey(sets,coin)
            println(io,"  $coin: excluded (insufficient window/context history)")
            continue
        end
        counts = sets[coin]
        complete = all(>(0),values(counts))
        complete && push!(accepted,coin)
        println(io,"  $coin: train=$(counts.train) eval=$(counts.eval) test=$(counts.test)",
            complete ? " eligible" : " excluded (missing regime/fold population)")
    end
    println(io,"# Full regime/fold preparation for the evaluated universe; changing the universe can change coverage.")
    println(io,"traincoins() = ",repr(accepted))
    flush(io)
    return nothing
end

"""Report each configuration and the intersection of full train/eval/test eligibility."""
function reportcommonregimecoins(populations::AbstractDict, configs::Vector{String},
        coins::Vector{String}, daterange::AbstractString;io::IO=stdout)::Nothing
    accepted = copy(coins)
    for ts in configs
        sets = populations[ts]
        reportregimecoins(sets,coins,ts,daterange;io)
        filter!(coin -> haskey(sets,coin) &&
            all(>(0),values(sets[coin])),accepted)
    end
    println(io,"\nCommon RegimeTraining eligibility: ts=$(join(configs,",")), daterange=$daterange")
    println(io,"# Eligible for every selected configuration in the evaluated universe.")
    println(io,"traincoins() = ",repr(sort!(accepted)))
    flush(io)
    return nothing
end

"""Read one window without allowing the full raw OHLCV table to escape this function."""
function regimehistory(coin::AbstractString, dr::NamedTuple, warmup::Int)::Union{Nothing,DataFrame}
    raw = EnvConfig.readtable("ohlcv";folderpath=EnvConfig.coinfolderpath(coin,"USDT"),format=:arrow)
    @assert !isnothing(raw) "missing OHLCV for $coin during RegimeTraining eligibility"
    return TradingStrategy.regimehistory(raw,coin,dr.startdt,dr.enddt,warmup)
end

"""Persist one coin's eligible timestamps, displaying its position in the full coin loop."""
function persistregimetimes!(history::DataFrame, coin::AbstractString, clcfg::NamedTuple,
        regime::TradingStrategy.RegimeConfig, folder::AbstractString;
        coinix::Int, ncoins::Int)::Nothing
    TradingStrategy.ensurefeaturetargetstores!(clcfg,[coin];
        progressoffset=coinix-1,progresstotal=ncoins)
    GC.gc()
    featurefactory = TradingStrategy.featurefactory(clcfg.features)
    targetfactory = () -> TradingStrategy.targetconfig(clcfg.targets)
    features = Features.readfeatures(coin,"USDT",clcfg.features;
        confighash=TradingStrategy.featureshash(clcfg.features))
    targets = Targets.readtargets(coin,"USDT",clcfg.targets;
        confighash=TradingStrategy.targetshash(clcfg.targets))
    @assert !isnothing(features) && !isnothing(targets) "missing shared feature/target stores for $coin"
    features,targets = TradingStrategy.sharedstorewindow(history,features,targets,
        featurefactory,targetfactory,coin)
    GC.gc()
    stores = (coins=[coin],featurestores=[features],targetstores=[targets])
    eligible = TradingStrategy.experimenttimes(Dict(coin=>history),stores,regime)
    stores = nothing
    features = nothing
    targets = nothing
    GC.gc()
    EnvConfig.writetable(DataFrame(opentime=eligible[coin]),coin;folderpath=folder,format=:arrow)
    return nothing
end

"""Process one coin at its full-loop position, retaining only observations and disk-backed timestamps."""
function preparecoin!(coin::AbstractString, dr::NamedTuple, warmup::Int,
        clcfg::NamedTuple, regime::TradingStrategy.RegimeConfig,
        times::AbstractVector{DateTime}, folder::AbstractString;
        coinix::Int, ncoins::Int)::Union{Nothing,DataFrame}
    history = regimehistory(coin,dr,warmup)
    isnothing(history) && return nothing
    GC.gc()
    persistregimetimes!(history,coin,clcfg,regime,folder;coinix,ncoins)
    GC.gc()
    return TradingStrategy.regimeobservations(history,times,regime)
end

"""Load one persisted eligible timestamp vector, failing explicitly on missing scratch data."""
function readregimetimes(coin::AbstractString, folder::AbstractString)::AbstractVector{DateTime}
    table = EnvConfig.readtable(coin;folderpath=folder,format=:arrow)
    @assert !isnothing(table) "missing eligible timestamps for $coin in $folder"
    return table.opentime
end

"""Stream full eligibility preparation and return only three sample counts per accepted coin."""
function regimecounts(config::AbstractString, dr::NamedTuple,
        coins::Vector{String};regime::TradingStrategy.RegimeConfig=TradingStrategy.RegimeConfig(),
        folds::TradingStrategy.FoldConfig=TradingStrategy.FoldConfig()
        )::Dict{String,typeof((train=0,eval=0,test=0))}
    clcfg = TradingStrategy.classifierconfig(TradingStrategy.tsconfig(config).classifier)
    TradingStrategy.checkregimeconfig(regime)
    warmup = TradingStrategy.regimewarmup(TradingStrategy.featurefactory(clcfg.features),
        () -> TradingStrategy.targetconfig(clcfg.targets),regime)
    times = collect(ceil(dr.startdt,Minute(regime.stepminutes)):Minute(regime.stepminutes):dr.enddt)
    return mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        acceptedcoins = String[]
        observations = DataFrame[]
        for (coinix,coin) in enumerate(coins)
            observation = preparecoin!(coin,dr,warmup,clcfg,regime,times,folder;
                coinix,ncoins=length(coins))
            if !isnothing(observation)
                push!(acceptedcoins,coin)
                push!(observations,observation)
            end
            observation = nothing
            GC.gc()
        end
        counts = Dict{String,typeof((train=0,eval=0,test=0))}()
        isempty(acceptedcoins) && return counts
        market = TradingStrategy.marketcalendar(observations,regime;
            startdt=dr.startdt,enddt=dr.enddt)
        empty!(observations)
        GC.gc()
        loader = coin -> readregimetimes(coin,folder)
        calendar = TradingStrategy.foldcalendar(market,loader,acceptedcoins,
            folds;enddt=dr.enddt,
            horizonminutes=Targets.lookaheadminutes(TradingStrategy.targetconfig(clcfg.targets)))
        market = nothing
        GC.gc()
        for coin in acceptedcoins
            eligible = loader(coin)
            counts[coin] = TradingStrategy.calendarcounts(eligible,calendar)
            eligible = nothing
            GC.gc()
        end
        return counts
    end
end

"""
Run full RegimeTraining store, liquidity, market-regime and fold preparation without training.
Uses the selected Bybit USDT universe for every classifier and requested date range.
Report the common coin list satisfying all configurations, not just a maximum warmup estimate.
Only one coin's source tables are retained at a time. Hourly observations are pooled for the
shared market calendar; eligible minute timestamps are stored temporarily on disk.
Only final per-coin counts are retained across configurations.
"""
function regimereport!(configs::Vector{String}, dateranges::Vector{String},
        pairs::Vector{String};io::IO=stdout)::Nothing
    configs = regimeconfigs(join(configs,","))
    coins = sort!([string(first(split(pair,"-";limit=2))) for pair in pairs
        if last(split(pair,"-";limit=2))=="USDT"])
    @assert !isempty(coins) "RegimeTraining eligibility requires Bybit USDT pairs: pairs=$pairs"
    EnvConfig.setpairquote!("USDT")
    for daterange in dateranges
        dr = TradingStrategy.daterangeconfig(daterange)
        populations = Dict{String,Dict{String,typeof((train=0,eval=0,test=0))}}()
        for ts in configs
            populations[ts] = regimecounts(ts,dr,coins)
            GC.gc()
        end
        reportcommonregimecoins(populations,configs,coins,daterange;io)
    end
    return nothing
end

"""Summarize successfully rebuilt empty sets stores, grouped by coin pair, to `io`."""
function reportzeroassignments(zeroassignments::Vector{Pair{String,String}}; io::IO=stderr)::Nothing
    isempty(zeroassignments) && return nothing
    pairs = sort!(unique(first.(zeroassignments)))
    println(io, "\nZero assignments in $(length(zeroassignments)) rebuilt sets stores across $(length(pairs)) coin pairs:")
    for pair in pairs
        ids = sort!(unique([id for (coinpair, id) in zeroassignments if coinpair == pair]))
        println(io, "  $(pair): $(join(ids, ", "))")
    end
    println(io, "Empty stores were saved; these pairs contribute no samples through the listed stores.")
    flush(io)
    return nothing
end

"""Print the skipped pair/config combinations with the first line of their error."""
function reportfailures(failures::Vector{Pair{String,String}})::Nothing
    isempty(failures) && return nothing
    println(stderr, "\nSkipped $(length(failures)) pair/config refreshes due to errors:")
    for (label, message) in failures
        println(stderr, "  $(label): $(first(split(message, '\n')))")
    end
    flush(stderr)
    return nothing
end

"""Print command usage."""
function helptext()::String
    return """
Usage: julia --project=. scripts/refresh.jl [features=<Fxx,...>] [targets=<Txx,...>] [partitions=<Pxx_Dxx,...>] [pairs=<BASE-QUOTE,...>] [xch=<exchange>] [ts=<id,...>]
       julia --project=. scripts/refresh.jl help

Provide at least one of features, targets, partitions or ts. Omitted pairs processes every cached pair
under the selected exchange folder. A pair may be given as BASE-QUOTE or as BASE,
which uses that exchange's default quote. Default xch=Bybit.
Supported xch values: $(join(sort!(collect(keys(DEFAULT_QUOTES))), ", ")).
Example: partitions=P02_D01,P02_D00
Partitions always rebuild the selected sets stores with current registry settings, even
when the stored config hash differs. Features and targets retain their normal reuse checks.
This does not rebuild classifiers, predictions or experiment-local fold assignments.
Changing memberships can invalidate earlier evaluations; use new config IDs for reproducibility.
Empty sets stores are saved, warned about and summarized by pair and partition ID.
At exit, each partition also prints a copy-pasteable `traincoins()` definition for
coins with at least one train, eval and test assignment.
Add ts=062,066,067 to run full RegimeTraining preparation and report actual regime/fold eligibility
at completion, separately from partition eligibility. Bybit only; uses selected USDT pairs,
each selected classifier and default detector/folds. Dates come from partitions, or default D01.
This refreshes required shared stores but does not train or modify classifiers.
The final common copy-pasteable list includes only coins with train, eval and test samples
under every selected configuration; per-config counts and exclusions are printed first.
Coverage depends on the evaluated universe; changing that list can change market regimes.
Preparation streams source tables and eligible timestamps one coin at a time.
Only hourly market observations and final sample counts are pooled, not full histories or stores.
"""
end

"""Refresh shared features/targets and rebuild requested partitions for the selected exchange and pairs."""
function main(args::Vector{String}=ARGS)::Nothing
    if any(in(HELP_FLAGS), args)
        println(helptext())
        return nothing
    end
    options = parseoptions(args)
    @assert any(key -> haskey(options, key), ("features", "targets", "partitions", "ts")) "provide at least one of features, targets, partitions or ts"
    features = haskey(options, "features") ? configids(options["features"], "features") : String[]
    targets = haskey(options, "targets") ? configids(options["targets"], "targets") : String[]
    partitions = haskey(options, "partitions") ? configids(options["partitions"], "partitions") : String[]
    configs = haskey(options,"ts") ? regimeconfigs(options["ts"]) : String[]
    exchange, defaultquote = exchangecfg(get(options, "xch", Xch.EXCHANGE_BYBIT))
    if haskey(options,"ts")
        @assert exchange==Xch.EXCHANGE_BYBIT "RegimeTraining eligibility supports Bybit only: xch=$exchange"
    end
    EnvConfig.init(EnvConfig.training)
    EnvConfig.setcoinspath!(exchange)
    EnvConfig.setdfformat!(:arrow)
    available = cachedpairs(EnvConfig.coinspath())
    pairs = resolvepairs(get(options, "pairs", nothing), available, defaultquote)
    @assert !isempty(pairs) "no cached OHLCV pairs found under $(EnvConfig.coinspath())"
    println("Refreshing $(length(pairs)) pairs from $(exchange): $(join(pairs, ", "))")
    failures = Pair{String,String}[]
    zeroassignments = Pair{String,String}[]
    partitionmemberships = Dict{Tuple{String,String},Set{String}}()
    reported = Ref(false)
    report = () -> begin
        reported[] && return nothing
        reportfailures(failures)
        reportzeroassignments(zeroassignments)
        isempty(partitions) || reportpartitioncoins(partitions,partitionmemberships)
        reported[] = true
        return nothing
    end
    # Ctrl+C terminates the process directly, even while blocked in download tasks; atexit still reports
    atexit(report)
    try
        for pair in pairs
            refreshpair!(failures, pair, features, targets, partitions;
                zeroassignments,partitionmemberships)
            GC.gc()
        end
    finally
        report()
    end
    if haskey(options,"ts")
        dateranges = isempty(partitions) ? [TradingStrategy.defaultdaterange()] :
            sort!(unique([string(last(split(id,"_"))) for id in partitions]))
        regimereport!(configs,dateranges,pairs)
    end
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        Refresh.main()
    catch error
        if error isa InterruptException
            println(stderr, "\nInterrupted. Completed stores are retained; rerun to continue.")
            exit(130)
        end
        rethrow()
    end
end
