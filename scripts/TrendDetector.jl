module TrendDetector
using Dates, Statistics, DataFrames
using EnvConfig, Classify, Ohlcv, Features, Targets, TradingStrategy, Xch, Bybit, TSM

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 2

"""
Pipeline stages in execution order. Requesting a stage runs all previous stages, which only
calculate artifacts that are not yet stored:
- `stores`: per-coin regression, feature, target and sets stores
- `train`: training index and adapted classifier
- `predict`: per-coin predictions of the classifier
- `evaluate`: confusion matrices of the classifier
- `gain`: gain replay of the trading strategy context
"""
const STAGES = ["stores", "train", "predict", "evaluate", "gain"]

"""
Run configuration of one TrendDetector invocation.

- `tscfg`: trading strategy context (classifier base name + strategy id)
- `clcfg`: classifier config of `tscfg`
- `coinset`, `daterange`: training data identity of the classifier
- `coins`: coins whose stores, predictions, evaluation and gains are processed
- `stage`: last pipeline stage to run
- `retrain`, `reinforce_epochs`: classifier adaptation and soft-reinforcement options
- `replaystartdt`, `replayenddt`: optional gain replay window replacing the set partitions
- `folder`: log subfolder of the gain replay output
"""
Base.@kwdef struct RunConfig
    tscfg::NamedTuple
    clcfg::NamedTuple
    coinset::String
    daterange::String
    coins::Vector{String}
    stage::String = "gain"
    retrain::Bool = false
    reinforce_epochs::Int = 0
    replaystartdt::Union{Nothing, DateTime} = nothing
    replayenddt::Union{Nothing, DateTime} = nothing
    folder::String
end

classifiername(rc::RunConfig)::String = TradingStrategy.classifiername(rc.clcfg; coinset=rc.coinset, daterange=rc.daterange)

"Return whether `stage` is part of the stages that `rc` runs."
runsstage(rc::RunConfig, stage::AbstractString)::Bool = findfirst(==(stage), STAGES) <= findfirst(==(rc.stage), STAGES)

#region GainReplay

"""
Return `(ix, reason)` for the first invalid score where valid scores are finite
numbers within `(0.0, 1.0]`. Returns `(nothing, "")` when all scores are valid.

Zero is excluded: it is the live-path "not yet classified" sentinel that
`gain_limit_reversal!` acts on, so it must never appear in prediction output.
"""
function _first_invalid_score(scores)
    for ix in eachindex(scores)
        value = scores[ix]
        if ismissing(value)
            return ix, "missing"
        elseif isnan(value)
            return ix, "NaN"
        elseif !isfinite(value)
            return ix, "non-finite"
        elseif (value <= 0.0) || (value > 1.0)
            return ix, "out-of-range"
        end
    end
    return nothing, ""
end

"""Columns kept in the per-range Trades snapshot accumulated by `getgainsdf`.

These are exactly what `TSM.compilegainsdf` and `TradeAdviceCompare` consume. The
remaining ~45 Trades columns are order id/status/msg and account placeholders that stay
constant during gain replay; carrying them would roughly triple the snapshot memory that
accumulates over all range groups."""
const TRADES_SNAPSHOT_COLUMNS = Symbol[
    :opentime, :pair, :set, :rangeid, :lastopentrade,
    :close, :high, :low, :label, :score,
    :lp_amount, :sp_amount,
    :lol_pavg, :lcl_pavg, :sol_pavg, :scl_pavg,
    :lo_limit, :lc_limit, :so_limit, :sc_limit,
]

"""Stem of the compiled gain segments produced by gain replay."""
const COMPILED_GAINS_STEM = "tsmgains-td"

"""Concatenate per-pair compiled gains, persist them and derive the gain report."""
function _collectxchgains(gainparts::Vector{DataFrame}; gainsstem::AbstractString=COMPILED_GAINS_STEM, reportstem::AbstractString="xchgainsreport-td")
    folderpath = EnvConfig.logfolder()
    xchgainsdf = isempty(gainparts) ? DataFrame() : TSM.sortgainsdf!(reduce(vcat, gainparts))
    empty!(gainparts)
    EnvConfig.savedf(xchgainsdf, String(gainsstem); folderpath=folderpath)
    xchreportdf = TSM.gainsreport(instem=gainsstem, stem=reportstem, folderpath=folderpath)
    return xchgainsdf, xchreportdf
end

"""Log the compiled gains report of the current run."""
function _report_compiled_gains(xchreportdf::Union{AbstractDataFrame, Nothing})
    if !isnothing(xchreportdf) && (size(xchreportdf, 1) > 0)
        println("$(EnvConfig.now()) compiled gains report: $xchreportdf")
    end
    return nothing
end

"""Subfolder of the run log folder holding one Trades artifact per trading pair."""
const TRADES_TD_SUBFOLDER = "trades-td"

"""Return the per-pair Trades artifact folder of the current run."""
tradestdfolder(folderpath::AbstractString=EnvConfig.logfolder())::String = joinpath(String(folderpath), TRADES_TD_SUBFOLDER)

"""Persist one pair's accumulated Trades snapshots of one replay pass and compile its gains.

Consumes `tradeparts`. Each pass is stored and compiled separately because both replay the
same minutes with different scores/labels; mixing them in one partition would match an open
of one pass against a close of the other."""
function _flushpairtrades!(gainparts::Vector{DataFrame}, tradeparts::Vector{DataFrame}, folderpath::AbstractString, pass::AbstractString, predicted::Bool)
    isempty(tradeparts) && return nothing
    pairdf = reduce(vcat, tradeparts)
    empty!(tradeparts)
    pair = String(pairdf[1, :pair])
    opentimes = pairdf[!, :opentime]
    @assert issorted(opentimes) "expected opentime ordered Trades rows for pair=$(pair), pass=$(pass); rows=$(nrow(pairdf)), first=$(first(opentimes)), last=$(last(opentimes))"
    EnvConfig.savedf(pairdf, "$(pair)-$(pass)"; folderpath=folderpath)
    # Each range is replayed independently, so a position still open at a range end has no
    # close of its own; scoping to (set, rangeid) drops it instead of matching it against
    # the next range's first row.
    gdf = TSM.compilegains(pairdf; setpartitions=true)
    if nrow(gdf) > 0
        gdf[!, :predicted] = fill(predicted, nrow(gdf))
        push!(gainparts, gdf)
    end
    return nothing
end

"""
    getgainsdf(rc)

Replay the predictions (and as reference the targets) of every coin of `rc` with the trading
strategy context of `rc` and return the compiled gains. Each range of the set partitions (or the
explicit replay window) is replayed independently.
"""
function getgainsdf(rc::RunConfig)
    EnvConfig.setlogpath(rc.folder)
    strategy = TradingStrategy.strategyconfig(rc.tscfg; coinset=rc.coinset, daterange=rc.daterange)
    ts = TradingStrategy.TsCache(strategy=strategy, source=TradingStrategy.tsconfigsource(rc.tscfg))
    dr = TradingStrategy.daterangeconfig(rc.daterange)
    replay_startdt = isnothing(rc.replaystartdt) ? dr.startdt : rc.replaystartdt
    replay_enddt = isnothing(rc.replayenddt) ? dr.enddt : rc.replayenddt
    xc = Xch.XchCache(Bybit.BybitCache(); startdt=replay_startdt, enddt=replay_enddt)
    TSM.ensuretradesschema!(xc.tsm, TSM.tradesdf_all_contributors())
    tradesfolderpath = tradestdfolder()

    xchgainparts = DataFrame[]
    predparts = DataFrame[]
    truthparts = DataFrame[]
    totalranges = 0
    for (coinix, coin) in enumerate(rc.coins)
        coinresultsdf = TradingStrategy.replayinput(rc.clcfg, coin; coinset=rc.coinset, daterange=rc.daterange, startdt=rc.replaystartdt, enddt=rc.replayenddt)
        nrow(coinresultsdf) == 0 && continue
        badix, badreason = _first_invalid_score(coinresultsdf[!, :score])
        @assert isnothing(badix) "invalid score in gain input for coin=$(coin) at row $(badix) due to $(badreason)"
        # Replay feeds prepopulated label/score rows and never calls the classifier, so the
        # base has to be attached explicitly for algorithms reading classifier features.
        TradingStrategy.addclassifierbase!(ts, Ohlcv.read(coin); datetime=minimum(coinresultsdf[!, :opentime]))

        # Ranges are replayed in chronological order so the accumulated pair snapshot is built
        # from the beginning of time; sets interleave, so groups are ordered by first opentime.
        rangegroups = groupby(coinresultsdf, [:set, :rangeid])
        grouporder = sortperm([first(group[!, :opentime]) for group in rangegroups])
        totalranges += length(grouporder)

        for (rngix, groupix) in enumerate(grouporder)
            resultsview = rangegroups[groupix]
            rng = resultsview[begin, :rangeid]
            (verbosity >= 2) && print("$(EnvConfig.now()) calculating gains for $coin ($coinix/$(length(rc.coins))) range ($rngix/$(length(rangegroups))) $rng                             \r")
            # Replay has no account, so equity is a constant: one lane budget is always
            # fundable and repeated opens are prevented by the lane position amount.
            replaybudget = Dict{Symbol, Any}(
                :set => String(resultsview[begin, :set]),
                :rangeid => Int(rng),
                :freequote => strategy.maxbudgetquote,
                :freemargin => strategy.maxbudgetquote,
            )
            evaldt = resultsview[end, :opentime]
            tp = TradingStrategy.preparereplaytrades!(ts, xc, coin, resultsview, resultsview[!, :score], collect(resultsview[!, :label]); metadata=replaybudget, datetime=evaldt)
            TradingStrategy.processreplaygains!(tp; strategy=strategy)
            # Snapshot before the truth pass reuses and overwrites this same tradesdf.
            push!(predparts, select(tp.tradesdf, TRADES_SNAPSHOT_COLUMNS))

            tp = TradingStrategy.preparereplaytrades!(ts, xc, coin, resultsview, fill(1f0, nrow(resultsview)), collect(resultsview[!, :target]); metadata=replaybudget, datetime=evaldt)
            TradingStrategy.processreplaygains!(tp; strategy=strategy)
            push!(truthparts, select(tp.tradesdf, TRADES_SNAPSHOT_COLUMNS))

            # Release the full-width replay Trades DataFrame of this range.
            TradingStrategy.droppair!(ts, tp.pair)
            TSM.droppair!(xc.tsm, tp.pair)
        end
        TradingStrategy.dropbase!(ts, coin)
        # One coin is one pair, so its snapshots are complete once its ranges are processed.
        _flushpairtrades!(xchgainparts, predparts, tradesfolderpath, "predicted", true)
        _flushpairtrades!(xchgainparts, truthparts, tradesfolderpath, "truth", false)
    end
    (verbosity >= 2) && println()

    gaindf, xchreportdf = _collectxchgains(xchgainparts)
    _report_compiled_gains(xchreportdf)
    if size(gaindf, 1) > 0
        expected = Set(Xch.tradingpairkey(String(coin), EnvConfig.pairquote) for coin in rc.coins)
        missing_pairs = sort(collect(setdiff(expected, Set(String.(unique(gaindf[!, :pair]))))))
        isempty(missing_pairs) || @warn "missing pairs in gains output" missing_pairs
    end
    (verbosity >= 2) && println("$(EnvConfig.now()) calculated gains for $(totalranges) ranges")
    return gaindf
end

#endregion GainReplay

"""
    runpipeline(rc)

Run all pipeline stages of `rc` up to `rc.stage`. Stored artifacts are reused, i.e. only missing
stores, an untrained classifier or missing predictions are calculated.
"""
function runpipeline(rc::RunConfig)
    println("$(EnvConfig.now()) trading strategy context $(rc.tscfg.configname): classifier $(classifiername(rc)), strategy $(rc.tscfg.strategy), stage $(rc.stage), $(length(rc.coins)) coins")
    TradingStrategy.ensurecoinstores!(rc.clcfg, rc.coins; daterange=rc.daterange)
    runsstage(rc, "train") || return nothing
    TradingStrategy.trainclassifier!(rc.clcfg; coinset=rc.coinset, daterange=rc.daterange, retrain=rc.retrain, reinforce_epochs=rc.reinforce_epochs)
    runsstage(rc, "predict") || return nothing
    coins = TradingStrategy.predictcoins!(rc.clcfg, rc.coins; coinset=rc.coinset, daterange=rc.daterange)
    runsstage(rc, "evaluate") || return nothing
    cmdf, xcmdf = TradingStrategy.evaluateclassifier(rc.clcfg, coins; coinset=rc.coinset, daterange=rc.daterange)
    println("$(EnvConfig.now()) confusion matrix: $cmdf")
    println("$(EnvConfig.now()) extended confusion matrix: $xcmdf")
    runsstage(rc, "gain") || return nothing
    gaindf = getgainsdf(rc)
    if !isnothing(gaindf) && (size(gaindf, 1) > 0)
        cgaindf = combine(groupby(gaindf, [:set, :side, :predicted]), :gain => mean, nrow, :gain => sum, :gainquote => sum)
        sort!(cgaindf, [:set, :side])
        println("$(EnvConfig.now()) cgaindf=$cgaindf")
    end
    return nothing
end

"Print the stored artifacts of `rc` without calculating anything."
function inspect(rc::RunConfig)
    name = classifiername(rc)
    println("classifier $(name) stored=$(Classify.isclassifierstored(name)) meta=$(Classify.readmeta(name))")
    for coin in rc.coins
        fdf, tdf, sdf = TradingStrategy.coinstores(rc.clcfg, coin; daterange=rc.daterange)
        preddf = Classify.readpredictions(name, coin)
        rows(df) = isnothing(df) ? "missing" : "$(size(df, 1)) rows"
        println("$(coin): features $(rc.clcfg.features)=$(rows(fdf)), targets $(rc.clcfg.targets)=$(rows(tdf)), sets=$(rows(sdf)), predictions=$(rows(preddf))")
    end
end

#region CLI

function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)
    prefix = key * "="
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "="; limit=2)[2]
        end
    end
    return default
end

function _normalize_runid_token(value)::String
    token = replace(lowercase(strip(String(value))), r"[^a-z0-9._-]+" => "_")
    return isempty(token) ? "na" : token
end

function _set_deterministic_run_id!(args::Vector{String}, context::Vector{Pair{String, String}}=Pair{String, String}[])
    explicit = _argvalue(args, "runid", nothing)
    if !isnothing(explicit)
        runid = _normalize_runid_token(explicit)
        ENV["CTS_RUN_ID"] = runid
        println("$(EnvConfig.now()) CTS_RUN_ID=$(runid) (explicit)")
        return runid
    end

    argtokens = String[]
    for arg in args
        startswith(arg, "runid=") && continue
        if occursin("=", arg)
            parts = split(arg, "="; limit=2)
            push!(argtokens, "$( _normalize_runid_token(parts[1]) )=$( _normalize_runid_token(parts[2]) )")
        else
            push!(argtokens, _normalize_runid_token(arg))
        end
    end
    sort!(argtokens)
    ctxtokens = ["$( _normalize_runid_token(kv.first) )=$( _normalize_runid_token(kv.second) )" for kv in context]
    sort!(ctxtokens)
    runid = join(vcat(["trenddetector"], ctxtokens, argtokens), "__")
    ENV["CTS_RUN_ID"] = runid
    println("$(EnvConfig.now()) CTS_RUN_ID=$(runid)")
    return runid
end

function _parse_csv_tokens(raw::AbstractString)::Vector{String}
    tokens = [uppercase(strip(token)) for token in split(String(raw), ",") if !isempty(strip(token))]
    @assert !isempty(tokens) "coins=$(raw) must provide at least one non-empty token"
    return unique(tokens)
end

"""
Return whether the CLI arguments request the help output.
"""
function _wants_help(args::Vector{String})::Bool
    for arg in args
        normalized = lowercase(strip(arg))
        if normalized in ("help", "--help", "-h")
            return true
        elseif startswith(normalized, "help=")
            value = split(normalized, "="; limit=2)[2]
            return value in ("1", "true", "yes", "on")
        end
    end
    return false
end

const CLI_FLAGS = ("help", "--help", "-h", "test", "train", "inspect", "retrain")
const CLI_KEYS = ("help", "ts", "stage", "coinset", "daterange", "coins", "startdt", "enddt", "reinforce", "folder", "runid")

"Assert that every CLI argument is a known flag or `key=value` parameter to fail fast on typos and obsolete parameters."
function _assertknownargs(args::Vector{String})
    for arg in args
        if occursin("=", arg)
            key = first(split(arg, "="; limit=2))
            @assert key in CLI_KEYS "unknown parameter $(arg); known key=value parameters: $(join(CLI_KEYS, ", ")) - see `help`"
        else
            @assert arg in CLI_FLAGS "unknown flag $(arg); known flags: $(join(CLI_FLAGS, ", ")) - see `help`"
        end
    end
    return nothing
end

#endregion CLI

"""
    buildrunconfig(args)

Return the `RunConfig` of the CLI arguments `args`. Coin set and date range default to the
EnvConfig mode (test: `CS00`/`D00`, otherwise `CS01`/`D01`).
"""
function buildrunconfig(args::Vector{String})::RunConfig
    tscfg = TradingStrategy.tsconfig(_argvalue(args, "ts", "046"))
    coinset = _argvalue(args, "coinset", TradingStrategy.defaultcoinset())
    daterange = _argvalue(args, "daterange", TradingStrategy.defaultdaterange())
    coins_arg = _argvalue(args, "coins", nothing)
    coins = isnothing(coins_arg) ? copy(TradingStrategy.coinsetconfig(coinset).coins) : _parse_csv_tokens(String(coins_arg))
    stage = _argvalue(args, "stage", "gain")
    @assert stage in STAGES "stage=$(stage) must be one of $(STAGES)"
    startdt_arg = _argvalue(args, "startdt", nothing)
    enddt_arg = _argvalue(args, "enddt", nothing)
    @assert isnothing(startdt_arg) == isnothing(enddt_arg) "startdt=$(startdt_arg) and enddt=$(enddt_arg) must be given together"
    replaystartdt = isnothing(startdt_arg) ? nothing : DateTime(String(startdt_arg))
    replayenddt = isnothing(enddt_arg) ? nothing : DateTime(String(enddt_arg))
    @assert isnothing(replaystartdt) || (replaystartdt < replayenddt) "startdt=$(replaystartdt) must be < enddt=$(replayenddt)"
    modetag = lowercase(String(Symbol(EnvConfig.configmode)))
    folder = _argvalue(args, "folder", "TS-$(tscfg.configname)-$(modetag)")
    return RunConfig(
        tscfg=tscfg,
        clcfg=TradingStrategy.classifierconfig(tscfg.classifier),
        coinset=String(coinset),
        daterange=String(daterange),
        coins=coins,
        stage=String(stage),
        retrain=("retrain" in args),
        reinforce_epochs=parse(Int, _argvalue(args, "reinforce", "0")),
        replaystartdt=replaystartdt,
        replayenddt=replayenddt,
        folder=String(folder),
    )
end

"""
Return CLI help text for `TrendDetector.jl`.
"""
function trenddetectorhelp()::String
    return """
Usage:
    julia --project=. scripts/TrendDetector.jl [help] [test|train] [inspect] [retrain] [key=value ...]

Runs the artifact pipeline of a trading strategy context from `TS_CONFIGS` in
`TradingStrategy/src/tradingstrategyconfig.jl`. Every stage only calculates missing artifacts:
  stores   per-coin regression, feature (`Fxx`), target (`Txx`) and sets (`Pxx_Dxx`) stores
           under `<coinspath>/<BASE-QUOTE>/`
  train    training index and adapted classifier `Fxx-Txx-Pxx-Mxxx-Bx-CSxx-Dxx` under
           `EnvConfig.classifierspath()`
  predict  per-coin predictions in the classifier folder
  evaluate confusion matrices in the classifier folder
  gain     gain replay of the trading strategy context into `logs/<folder>`

Flag parameters:
  help, --help, -h
      Show this message and exit.
  test
      Use `EnvConfig.init(test)` (default): coin set `CS00`, date range `D00`.
  train
      Use `EnvConfig.init(training)`: coin set `CS01`, date range `D01`.
  inspect
      Print the stored artifacts of the selected context without calculating anything.
  retrain
      Continue training the stored classifier, preserving weights, optimizer state and loss history.
      Train a new classifier if none is stored. Can be combined with reinforce=<Int>.

Key=value parameters:
  ts=<configname>
      Trading strategy context from `TS_CONFIGS`. Default: `046`
  stage=<stores|train|predict|evaluate|gain>
      Last pipeline stage to run. Default: `gain`
  coinset=<CSxx>
      Coin set the classifier is trained on (`COINSET_CONFIGS`).
  daterange=<Dxx>
      Date range the classifier is trained on (`DATERANGE_CONFIGS`).
  coins=<CSV>
      Coins to process for stores, predictions, evaluation and gains. Default: coins of the coin set.
      Example: `coins=SINE,BTC,ETH`
  startdt=<DateTime> enddt=<DateTime>
      Replay window of the gain stage replacing the set partitions (ISO-8601).
      Example: `startdt=2025-07-01T01:00:00 enddt=2025-07-30T01:00:00`
  reinforce=<Int>
      Refresh interval in epochs, not the number of reinforced epochs. After every
      `reinforce`-th epoch the training-set misclassification flags are recomputed; the
      following epochs give misclassified samples 2× loss weight, normalized within each
      class, until the next refresh. Reinforcement starts after the first refresh and lasts
      until convergence or the epoch limit. All training samples remain in the shuffled batches.
      Example: `reinforce=3` refreshes after epochs 3, 6, 9, ... Default: `0` (off)
  folder=<name>
      Gain replay output subfolder. Default: `TS-<ts>-<mode>`
  runid=<name>
      Explicit run id exported as `CTS_RUN_ID`. Default: derived from mode, context and arguments

Unknown flags or parameters abort the run.

Examples:
    julia --project=. scripts/TrendDetector.jl test ts=050 stage=train
    julia --project=. scripts/TrendDetector.jl train ts=046 coins=BTC,ETH startdt=2025-07-01T00:00:00 enddt=2025-07-30T00:00:00
"""
# How reinforcement works, for example with reinforce=3:

# Epochs 1–3: normal training, using class weights only.
# After epoch 3: the training set is scored and every misclassified sample is flagged.
# Epochs 4–6: flagged samples get 2× loss weight, normalized within each class.
# After epoch 6: the flags are recomputed. This repeats until convergence or the epoch limit.
# So once it starts, reinforcement stays active for the rest of training. There is currently no parameter to limit how many epochs are reinforced.

end

"""
Run the `TrendDetector` script with the given CLI arguments.
"""
function main(args::Vector{String}=ARGS)
    if _wants_help(args)
        println(trenddetectorhelp())
        return nothing
    end
    _assertknownargs(args)
    println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$(args)")
    @assert !(("test" in args) && ("train" in args)) "mode flags are exclusive; use only one of test or train"
    EnvConfig.init("train" in args ? training : test)
    Ohlcv.verbosity = 1
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    EnvConfig.setcoinspath!("Bybit")
    EnvConfig.setdfformat!(:arrow)
    (verbosity >= 2) && println("coinspath: $(EnvConfig.coinspath()), classifierspath: $(EnvConfig.classifierspath())")
    rc = buildrunconfig(args)
    _set_deterministic_run_id!(args, [
        "mode" => lowercase(String(Symbol(EnvConfig.configmode))),
        "ts" => rc.tscfg.configname,
        "classifier" => classifiername(rc),
        "stage" => rc.stage,
    ])
    if "inspect" in args
        inspect(rc)
    else
        runpipeline(rc)
    end
    println("$(EnvConfig.now()) done")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end

end # of TrendDetector
