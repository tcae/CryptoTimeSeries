"""
Refresh shared feature and target stores from full OHLCV history.

Usage:
  julia --project=. scripts/refresh.jl features=F19,F20 targets=T24,T25
  julia --project=. scripts/refresh.jl features=F19 targets=T24 pairs=BTC-USDT,ETH-USDT
  julia --project=. scripts/refresh.jl targets=T24 xch=KrakenSpot pairs=XBT-USDC

Without `pairs`, all cached pairs under the selected exchange folder are processed.
Supported exchange folders are Bybit, BybitSim, IbkrSpot, KrakenFutures, and KrakenSpot.
"""
module Refresh

using DataFrames
using EnvConfig, Features, Ohlcv, Targets, TradingStrategy, Xch

const VALID_OPTIONS = ("features", "targets", "pairs", "xch")
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
        else
            @assert kind == "targets" "unsupported config kind=$kind"
            TradingStrategy.targetconfig(id)
        end
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
Refresh the requested shared feature and target artifacts for one complete OHLCV history.
A failing OHLCV load skips the pair; a failing feature or target config skips only that
config. Failures are recorded in `failures` as `"<pair> <id>" => error message`.
"""
function refreshpair!(failures::Vector{Pair{String,String}}, pair::AbstractString,
        features::Vector{String}, targets::Vector{String})::Nothing
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
Usage: julia --project=. scripts/refresh.jl features=<Fxx,...> targets=<Txx,...> [pairs=<BASE-QUOTE,...>] [xch=<exchange>]
       julia --project=. scripts/refresh.jl help

Provide at least one of features or targets. Omitted pairs processes every cached pair
under the selected exchange folder. A pair may be given as BASE-QUOTE or as BASE,
which uses that exchange's default quote. Default xch=Bybit.
Supported xch values: $(join(sort!(collect(keys(DEFAULT_QUOTES))), ", ")).
"""
end

"""Run the shared feature/target refresh for the requested exchange and pairs."""
function main(args::Vector{String}=ARGS)::Nothing
    if any(in(HELP_FLAGS), args)
        println(helptext())
        return nothing
    end
    options = parseoptions(args)
    @assert haskey(options, "features") || haskey(options, "targets") "provide at least one of features or targets"
    features = haskey(options, "features") ? configids(options["features"], "features") : String[]
    targets = haskey(options, "targets") ? configids(options["targets"], "targets") : String[]
    exchange, defaultquote = exchangecfg(get(options, "xch", Xch.EXCHANGE_BYBIT))
    EnvConfig.init(EnvConfig.training)
    EnvConfig.setcoinspath!(exchange)
    EnvConfig.setdfformat!(:arrow)
    available = cachedpairs(EnvConfig.coinspath())
    pairs = resolvepairs(get(options, "pairs", nothing), available, defaultquote)
    @assert !isempty(pairs) "no cached OHLCV pairs found under $(EnvConfig.coinspath())"
    println("Refreshing $(length(pairs)) pairs from $(exchange): $(join(pairs, ", "))")
    failures = Pair{String,String}[]
    reported = Ref(false)
    report = () -> (reported[] || reportfailures(failures); reported[] = true; nothing)
    # Ctrl+C terminates the process directly, even while blocked in download tasks; atexit still reports
    atexit(report)
    try
        for pair in pairs
            refreshpair!(failures, pair, features, targets)
            GC.gc()
        end
    finally
        report()
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
