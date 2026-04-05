module TradeAdviceLstm
using Test, Dates, Logging, CSV, JDF, DataFrames, Statistics
using CategoricalArrays
using EnvConfig, Classify, Ohlcv, Features, Targets, TradingStrategy

include("optimizationconfigs.jl")

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 2

const LSTM_LABELS = ["longbuy", "longhold", "longclose", "shortbuy", "shorthold", "shortclose", "allclose"]

function _stripconfigsuffix(name::AbstractString)
    return replace(strip(name), r"config$" => "")
end

function _resolve_trendconfig(configref::AbstractString)
    raw = lowercase(_stripconfigsuffix(configref))
    symbol = startswith(raw, "mk") ? Symbol(raw * "config") : Symbol("mk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown trend config '$configref'; expected function $(symbol) in optimizationconfigs.jl"
    return getfield(@__MODULE__, symbol)()
end

function _resolve_boundsconfig(configref::AbstractString)
    raw = lowercase(_stripconfigsuffix(configref))
    symbol = startswith(raw, "boundsmk") ? Symbol(raw * "config") : startswith(raw, "mk") ? Symbol("bounds" * raw * "config") : Symbol("boundsmk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown bounds config '$configref'; expected function $(symbol) in optimizationconfigs.jl"
    return getfield(@__MODULE__, symbol)()
end

function _resolve_tradeadviceconfig(configref::AbstractString)
    raw = lowercase(_stripconfigsuffix(configref))
    symbol = startswith(raw, "tradeadvicemk") ? Symbol(raw * "config") : startswith(raw, "mk") ? Symbol("tradeadvice" * raw * "config") : Symbol("tradeadvicemk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown trade advice config '$configref'; expected function $(symbol) in optimizationconfigs.jl"
    return getfield(@__MODULE__, symbol)()
end

function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)
    prefix = key * "="
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "="; limit=2)[2]
        end
    end
    return default
end

_ntget(nt::Union{Nothing,NamedTuple}, key::Symbol, default) = isnothing(nt) ? default : (key in keys(nt) ? getfield(nt, key) : default)

function _parse_float32_list(raw::AbstractString)
    values = split(strip(raw), ",")
    parsed = Float32[]
    for value in values
        stripped = strip(value)
        if !isempty(stripped)
            push!(parsed, parse(Float32, stripped))
        end
    end
    return parsed
end

"""Configuration for the end-to-end LSTM trade-advice backtest pipeline."""
mutable struct TradeAdviceLstmConfig
    configname::String
    folder::String
    trendconfig::NamedTuple
    boundsconfig::NamedTuple
    seqlen::Int
    hidden_dim::Int
    maxepoch::Int
    batchsize::Int
    entryfraction::Float32
    exitfraction::Float32
    openthresholds::Vector{Float32}
    closethresholds::Vector{Float32}
    entrytimeout::Int
    exittimeout::Int
    exitstrategy::Symbol
    mode::EnvConfig.Mode
    function TradeAdviceLstmConfig(;configname="025", folder="TradeAdviceLstm-$configname-$(EnvConfig.configmode)", trendconfig=mk025config(), boundsconfig=boundsmk001config(), seqlen::Int=3, hidden_dim::Int=32, maxepoch::Int=200, batchsize::Int=64, entryfraction::AbstractFloat=0.1f0, exitfraction::AbstractFloat=0.1f0, openthresholds::Vector{Float32}=Float32[0.8f0, 0.7f0, 0.6f0], closethresholds::Vector{Float32}=Float32[0.6f0, 0.55f0, 0.5f0], entrytimeout::Int=2, exittimeout::Int=2, exitstrategy::Symbol=:opposite_signal_market, mode::EnvConfig.Mode=EnvConfig.configmode)
        @assert seqlen > 0 "seqlen=$seqlen must be > 0"
        @assert hidden_dim > 0 "hidden_dim=$hidden_dim must be > 0"
        @assert maxepoch > 0 "maxepoch=$maxepoch must be > 0"
        @assert batchsize > 0 "batchsize=$batchsize must be > 0"
        @assert 0f0 < entryfraction <= 1f0 "entryfraction=$(entryfraction) must satisfy 0 < entryfraction <= 1"
        @assert 0f0 < exitfraction <= 1f0 "exitfraction=$(exitfraction) must satisfy 0 < exitfraction <= 1"
        @assert entrytimeout >= 0 "entrytimeout=$entrytimeout must be >= 0"
        @assert exittimeout >= 0 "exittimeout=$exittimeout must be >= 0"
        @assert !isempty(openthresholds) "openthresholds must not be empty"
        @assert !isempty(closethresholds) "closethresholds must not be empty"
        @assert all(0f0 .<= openthresholds .<= 1f0) "expected openthresholds within [0, 1]; got openthresholds=$(openthresholds)"
        @assert all(0f0 .<= closethresholds .<= 1f0) "expected closethresholds within [0, 1]; got closethresholds=$(closethresholds)"
        @assert exitstrategy in (:opposite_signal_market,) "unsupported exitstrategy=$exitstrategy; expected :opposite_signal_market"
        EnvConfig.setlogpath(folder)
        return new(configname, folder, trendconfig, boundsconfig, seqlen, hidden_dim, maxepoch, batchsize, Float32(entryfraction), Float32(exitfraction), openthresholds, closethresholds, entrytimeout, exittimeout, exitstrategy, mode)
    end
end

resultsfilename() = "results.jdf"
featuresfilename() = "features.jdf"
predictionsfilename() = "maxpredictions.jdf"
lstmmergedfilename() = "lstm_merged_inputs.jdf"
lstmlossesfilename() = "lstm_losses.jdf"
lstmpredictionsfilename() = "lstm_predictions.jdf"
lstmconfusionfilename() = "lstm_confusion.jdf"
lstmxconfusionfilename() = "lstm_xconfusion.jdf"
lstmpairsfilename() = "lstm_transaction_pairs.jdf"
lstmgainsfilename() = "lstm_gains.jdf"
summaryfilename() = "summary.jdf"

trendfolder(cfg::TradeAdviceLstmConfig) = "Trend-$(cfg.trendconfig.configname)-$(cfg.mode)"
boundsfolder(cfg::TradeAdviceLstmConfig) = "Bounds-$(cfg.boundsconfig.configname)-$(cfg.mode)"

function _with_log_subfolder(folder::AbstractString, f::Function)
    previous = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(folder)
    try
        return f()
    finally
        if previous == ""
            EnvConfig.setlogpath()
        else
            EnvConfig.setlogpath(previous)
        end
    end
end

_with_log_subfolder(f::Function, folder::AbstractString) = _with_log_subfolder(folder, f)

function _folderpath(subfolder::AbstractString)
    EnvConfig.setlogpath()
    return normpath(joinpath(EnvConfig.logfolder(), subfolder))
end

function _load_df_or_assert(folder::AbstractString, filename::AbstractString)
    fp = _folderpath(folder)
    df = EnvConfig.readdf(filename; folderpath=fp)
    @assert !isnothing(df) && size(df, 1) > 0 "missing or empty $filename in $fp"
    return df
end

function _tradepairstarget(cfg::TradeAdviceLstmConfig)
    @assert cfg.trendconfig.targetconfig isa Targets.Trend04 "TradeAdviceLstm currently expects cfg.trendconfig.targetconfig to be Targets.Trend04; got $(typeof(cfg.trendconfig.targetconfig))"
    return Targets.TradePairs(cfg.trendconfig.targetconfig; entryfraction=cfg.entryfraction, exitfraction=cfg.exitfraction)
end

"""
Load optimized trend classifier outputs and return class probabilities aligned to trend results rows.
"""
function get_trend_probabilities(cfg::TradeAdviceLstmConfig)
    trfolder = trendfolder(cfg)
    resultsdf = _load_df_or_assert(trfolder, resultsfilename())
    featuresdf = _load_df_or_assert(trfolder, featuresfilename())

    @assert size(resultsdf, 1) == size(featuresdf, 1) "trend results/features row mismatch: $(size(resultsdf, 1)) != $(size(featuresdf, 1))"

    fcols = Features.requestedcolumns(cfg.trendconfig.featconfig)
    @assert all(c -> c in names(featuresdf), fcols) "trend features dataframe missing required columns: required=$(fcols) available=$(names(featuresdf))"
    X = permutedims(Matrix(featuresdf[!, fcols]), (2, 1))

    probsdf = _with_log_subfolder(trfolder) do
        nn = cfg.trendconfig.classifiermodel(Features.featurecount(cfg.trendconfig.featconfig), Targets.uniquelabels(cfg.trendconfig.targetconfig), "mix")
        @assert isfile(Classify.nnfilename(nn.fileprefix)) "optimized trend classifier file not found: $(Classify.nnfilename(nn.fileprefix)). Run scripts/TrendDetector.jl first."
        nn = Classify.loadnn(nn.fileprefix)
        Classify.predictdf(nn, X)
    end

    @assert size(probsdf, 1) == size(resultsdf, 1) "trend probabilities/results row mismatch"
    return resultsdf, probsdf
end

"""
Load optimized bounds estimator outputs aligned to trend rows using stable admin keys.
"""
function get_bounds_predictions(cfg::TradeAdviceLstmConfig)
    bfolder = boundsfolder(cfg)
    resultsdf = _load_df_or_assert(bfolder, resultsfilename())
    predictionsdf = _load_df_or_assert(bfolder, predictionsfilename())

    keycols = [:coin, :rangeid, :opentime]
    @assert all(c -> c in propertynames(resultsdf), keycols) "bounds results missing required join keys $(keycols)"
    @assert size(resultsdf, 1) == size(predictionsdf, 1) "bounds results/predictions row mismatch: $(size(resultsdf, 1)) != $(size(predictionsdf, 1))"

    centercol = :centerpred in propertynames(predictionsdf) ? :centerpred : (:pred_center in propertynames(predictionsdf) ? :pred_center : nothing)
    widthcol = :widthpred in propertynames(predictionsdf) ? :widthpred : (:pred_width in propertynames(predictionsdf) ? :pred_width : nothing)
    @assert !isnothing(centercol) && !isnothing(widthcol) "bounds predictions must contain center/width columns; names=$(names(predictionsdf))"

    bdf = copy(resultsdf[!, keycols])
    bdf[!, :centerpred] = Float32.(predictionsdf[!, centercol])
    bdf[!, :widthpred] = Float32.(predictionsdf[!, widthcol])
    return bdf
end

"""
Merge optimized trend + bounds outputs and build an LSTM contract dataframe.
"""
function build_lstm_input_df(cfg::TradeAdviceLstmConfig)
    trend_results, trend_probs = get_trend_probabilities(cfg)
    bounds_pred = get_bounds_predictions(cfg)

    keycols = [:coin, :rangeid, :opentime]
    cols = [:coin, :rangeid, :set, :target, :opentime, :high, :low, :close, :pivot]
    @assert all(c -> c in propertynames(trend_results), cols) "trend results missing required columns"

    mdf = innerjoin(trend_results[!, cols], bounds_pred, on=keycols)
    @assert size(mdf, 1) > 0 "empty merge between trend and bounds data"
    if (size(mdf, 1) != size(trend_results, 1)) || (size(mdf, 1) != size(bounds_pred, 1))
        (verbosity >= 1) && @warn "trend/bounds merge dropped non-overlapping rows" merged_rows=size(mdf, 1) trend_rows=size(trend_results, 1) bounds_rows=size(bounds_pred, 1)
    end

    trendprobcols = [:longbuy, :longhold, :shortbuy, :shorthold, :allclose]
    probdf = copy(trend_results[!, keycols])
    for c in trendprobcols
        if c in propertynames(trend_probs)
            probdf[!, c] = Float32.(trend_probs[!, c])
        else
            probdf[!, c] = zeros(Float32, size(probdf, 1))
        end
    end
    mdf = leftjoin(mdf, probdf, on=keycols)

    # Fallback if allclose is not directly available from classifier labels.
    if !(:allclose in propertynames(trend_probs))
        mdf[!, :allclose] = max.(1f0 .- mdf[!, :longbuy] .- mdf[!, :longhold] .- mdf[!, :shortbuy] .- mdf[!, :shorthold], 0f0)
    end

    # Keep deterministic order for target derivation, window generation, and persistence.
    sort!(mdf, [:coin, :rangeid, :opentime])
    pairtargets = _tradepairstarget(cfg)
    mdf[!, :target] = string.(Targets.tradepairlabels(pairtargets, mdf; labelcol=:target, pivotcol=:pivot, groupcols=[:coin, :rangeid]))
    EnvConfig.savedf(mdf, lstmmergedfilename())

    mdf = copy(mdf)
    mdf[!, :rowix] = Int32.(1:size(mdf, 1))
    return mdf
end

function train_lstm(cfg::TradeAdviceLstmConfig, mdf::AbstractDataFrame)
    contract = Classify.lstm_bounds_trend_features(
        mdf;
        trendprobcols=[:longbuy, :longhold, :shortbuy, :shorthold, :allclose],
        centercol=:centerpred,
        widthcol=:widthpred,
        targetcol=:target,
        setcol=:set,
        rangeidcol=:rangeid,
        rixcol=:rowix,
    )

    res = Classify.train_lstm_trade_signals!(contract, cfg.seqlen; hidden_dim=cfg.hidden_dim, maxepoch=cfg.maxepoch, batchsize=cfg.batchsize, labels=LSTM_LABELS)

    lossesdf = DataFrame(epoch=collect(1:length(res.losses)), train_loss=Float32.(res.losses), eval_loss=Float32.(res.eval_losses))
    EnvConfig.savedf(lossesdf, lstmlossesfilename())
    return contract, res
end

function evaluate_lstm(cfg::TradeAdviceLstmConfig, mdf::AbstractDataFrame, contract, trainres)
    windows = Classify.lstm_tensor_windows(contract; seqlen=cfg.seqlen)
    @assert size(windows.X, 3) > 0 "no windows generated for evaluation"

    probs = Classify.predict_lstm_trade_signals(trainres.model, windows.X)
    predix = vec(argmax(probs; dims=1))
    predlabel = [trainres.labels[ci[1]] for ci in predix]
    predscore = [Float32(probs[ci[1], ix]) for (ix, ci) in enumerate(predix)]

    admincols = [:rowix, :opentime, :high, :low, :close, :pivot, :set, :rangeid, :coin, :centerpred, :widthpred]
    admindf = mdf[!, admincols]
    evaldf = innerjoin(
        DataFrame(rowix=windows.endrix, target=windows.targets, pred_label=predlabel, score=predscore),
        admindf,
        on=:rowix,
    )
    @assert size(evaldf, 1) == length(windows.targets) "window/admin merge mismatch"
    select!(evaldf, Not(:rowix))

    evaldf[!, :label] = Targets.tradelabel.(evaldf[!, :pred_label])
    evaldf[!, :target] = Targets.tradelabel.(string.(evaldf[!, :target]))
    evaldf[!, :set] = CategoricalVector(string.(evaldf[!, :set]), levels=settypes())

    EnvConfig.savedf(evaldf, lstmpredictionsfilename())

    alltl = Targets.tradelabel.(trainres.labels)
    cmdf = Classify.confusionmatrix(evaldf, alltl)
    xcmdf = Classify.extendedconfusionmatrix(evaldf, alltl)
    EnvConfig.savedf(cmdf, lstmconfusionfilename())
    EnvConfig.savedf(xcmdf, lstmxconfusionfilename())

    strategy = cfg.trendconfig.tradingstrategy
    makerfee = Float32(strategy.makerfee)
    takerfee = Float32(strategy.takerfee)
    thresholdpairs = [(open, close) for open in cfg.openthresholds for close in cfg.closethresholds if close <= open]
    @assert !isempty(thresholdpairs) "no valid threshold pairs from openthresholds=$(cfg.openthresholds) and closethresholds=$(cfg.closethresholds)"

    gainparts = DataFrame[]
    for rid in unique(evaldf[!, :rangeid])
        rangedf = sort(evaldf[evaldf[!, :rangeid] .== rid, :], :opentime)
        for (openthreshold, closethreshold) in thresholdpairs
            gdf = TradingStrategy.simulate_limit_trade_pairs(
                rangedf,
                rangedf[!, :score],
                rangedf[!, :label];
                openthreshold=openthreshold,
                closethreshold=closethreshold,
                entrytimeout=cfg.entrytimeout,
                exittimeout=cfg.exittimeout,
                makerfee=makerfee,
                takerfee=takerfee,
                exitstrategy=cfg.exitstrategy,
            )
            if size(gdf, 1) > 0
                gdf[!, :rangeid] = fill(rid, size(gdf, 1))
                gdf[!, :coin] = fill(string(rangedf[begin, :coin]), size(gdf, 1))
                gdf[!, :set] = fill(string(rangedf[begin, :set]), size(gdf, 1))
                gdf[!, :predicted] = fill(true, size(gdf, 1))
                gdf[!, :openthreshold] = fill(Float32(openthreshold), size(gdf, 1))
                gdf[!, :closethreshold] = fill(Float32(closethreshold), size(gdf, 1))
                push!(gainparts, gdf)
            end
        end

        gdf = TradingStrategy.simulate_limit_trade_pairs(
            rangedf,
            fill(1f0, size(rangedf, 1)),
            rangedf[!, :target];
            openthreshold=0.9f0,
            closethreshold=0.9f0,
            entrytimeout=cfg.entrytimeout,
            exittimeout=cfg.exittimeout,
            makerfee=makerfee,
            takerfee=takerfee,
            exitstrategy=cfg.exitstrategy,
        )
        if size(gdf, 1) > 0
            gdf[!, :rangeid] = fill(rid, size(gdf, 1))
            gdf[!, :coin] = fill(string(rangedf[begin, :coin]), size(gdf, 1))
            gdf[!, :set] = fill(string(rangedf[begin, :set]), size(gdf, 1))
            gdf[!, :predicted] = fill(false, size(gdf, 1))
            gdf[!, :openthreshold] = fill(0.9f0, size(gdf, 1))
            gdf[!, :closethreshold] = fill(0.9f0, size(gdf, 1))
            push!(gainparts, gdf)
        end
    end

    gaindf = isempty(gainparts) ? DataFrame() : reduce(vcat, gainparts; cols=:union)
    if size(gaindf, 1) > 0
        EnvConfig.savedf(gaindf, lstmpairsfilename())
        EnvConfig.savedf(gaindf, lstmgainsfilename())
    end

    return evaldf, cmdf, xcmdf, gaindf
end

function _safe_mean_or_missing(values)
    return isempty(values) ? missing : Float32(mean(Float32.(values)))
end

function _safe_sum_or_zero(values)
    return isempty(values) ? 0f0 : Float32(sum(Float32.(values)))
end

function _best_eval_trade_summary(gaindf::AbstractDataFrame)
    if size(gaindf, 1) == 0
        return (openthreshold=missing, closethreshold=missing, eval_segments=0, eval_gainfee=0f0, test_segments=0, test_gainfee=0f0)
    end

    evalpred = gaindf[(gaindf[!, :set] .== "eval") .&& (gaindf[!, :predicted] .== true), :]
    if size(evalpred, 1) == 0
        return (openthreshold=missing, closethreshold=missing, eval_segments=0, eval_gainfee=0f0, test_segments=0, test_gainfee=0f0)
    end

    grouped = combine(
        groupby(evalpred, [:openthreshold, :closethreshold]),
        nrow => :eval_segments,
        :gainfee => _safe_sum_or_zero => :eval_gainfee,
    )
    sort!(grouped, [order(:eval_gainfee, rev=true), order(:eval_segments, rev=true)])
    best = grouped[1, :]

    testpred = gaindf[
        (gaindf[!, :set] .== "test") .&&
        (gaindf[!, :predicted] .== true) .&&
        (gaindf[!, :openthreshold] .== best.openthreshold) .&&
        (gaindf[!, :closethreshold] .== best.closethreshold),
        :,
    ]

    return (
        openthreshold=best.openthreshold,
        closethreshold=best.closethreshold,
        eval_segments=best.eval_segments,
        eval_gainfee=best.eval_gainfee,
        test_segments=size(testpred, 1),
        test_gainfee=_safe_sum_or_zero(size(testpred, 1) == 0 ? Float32[] : testpred[!, :gainfee]),
    )
end

function buildsummmary(cfg::TradeAdviceLstmConfig, trainres, cmdf::AbstractDataFrame, gaindf::AbstractDataFrame)
    ppvcol = Symbol("ppv%")
    evalcmdf = size(cmdf, 1) == 0 ? DataFrame() : @view cmdf[cmdf[!, :set] .== "eval", :]
    testcmdf = size(cmdf, 1) == 0 ? DataFrame() : @view cmdf[cmdf[!, :set] .== "test", :]
    besttrade = _best_eval_trade_summary(gaindf)
    evaltrue = size(gaindf, 1) == 0 ? DataFrame() : @view gaindf[(gaindf[!, :set] .== "eval") .&& (gaindf[!, :predicted] .== false), :]
    summarydf = DataFrame([(
        configname=cfg.configname,
        trendconfig=cfg.trendconfig.configname,
        boundsconfig=cfg.boundsconfig.configname,
        trendfolder=trendfolder(cfg),
        boundsfolder=boundsfolder(cfg),
        seqlen=cfg.seqlen,
        hidden_dim=cfg.hidden_dim,
        maxepoch=cfg.maxepoch,
        batchsize=cfg.batchsize,
        entryfraction=cfg.entryfraction,
        exitfraction=cfg.exitfraction,
        entrytimeout=cfg.entrytimeout,
        exittimeout=cfg.exittimeout,
        exitstrategy=String(cfg.exitstrategy),
        epochs=length(trainres.losses),
        final_train_loss=Float32(trainres.losses[end]),
        final_eval_loss=Float32(trainres.eval_losses[end]),
        eval_mean_ppv=_safe_mean_or_missing(size(evalcmdf, 1) == 0 ? Float32[] : evalcmdf[!, ppvcol]),
        test_mean_ppv=_safe_mean_or_missing(size(testcmdf, 1) == 0 ? Float32[] : testcmdf[!, ppvcol]),
        best_openthreshold=besttrade.openthreshold,
        best_closethreshold=besttrade.closethreshold,
        eval_pred_segments=besttrade.eval_segments,
        test_pred_segments=besttrade.test_segments,
        eval_true_segments=size(evaltrue, 1),
        eval_pred_gainfee_sum=besttrade.eval_gainfee,
        test_pred_gainfee_sum=besttrade.test_gainfee,
        eval_true_gainfee_sum=_safe_sum_or_zero(size(evaltrue, 1) == 0 ? Float32[] : evaltrue[!, :gainfee]),
    )])
    EnvConfig.savedf(summarydf, summaryfilename())
    return summarydf
end

function _list_tradeadviceconfigsymbols()
    syms = [sym for sym in names(@__MODULE__, all=true) if occursin(r"^tradeadvicemk\d+config$", String(sym))]
    sort!(syms; by=sym -> parse(Int, match(r"tradeadvicemk(\d+)config", String(sym)).captures[1]))
    return syms
end

function collect_tradeadvice_summaries(cfg::TradeAdviceLstmConfig)
    parts = DataFrame[]
    seenfolders = Set{String}()
    for sym in _list_tradeadviceconfigsymbols()
        tacfg = getfield(@__MODULE__, sym)()
        folder = "TradeAdviceLstm-$(tacfg.configname)-$(cfg.mode)"
        sdf = EnvConfig.readdf(summaryfilename(); folderpath=_folderpath(folder))
        if !isnothing(sdf) && size(sdf, 1) > 0
            push!(parts, sdf)
            push!(seenfolders, folder)
        end
    end
    currentsummary = EnvConfig.readdf(summaryfilename(); folderpath=_folderpath(cfg.folder))
    if !(cfg.folder in seenfolders) && !isnothing(currentsummary) && size(currentsummary, 1) > 0
        push!(parts, currentsummary)
    end
    if isempty(parts)
        return DataFrame()
    end
    comparison = vcat(parts...; cols=:union)
    sort!(comparison, [order(:eval_pred_gainfee_sum, rev=true), order(:final_eval_loss)])
    return comparison
end

function run_pipeline(cfg::TradeAdviceLstmConfig)
    (verbosity >= 2) && println("$(EnvConfig.now()) TradeAdviceLstm pipeline start with trend=$(trendfolder(cfg)) bounds=$(boundsfolder(cfg))")
    mdf = build_lstm_input_df(cfg)
    contract, trainres = train_lstm(cfg, mdf)
    evaldf, cmdf, xcmdf, gaindf = evaluate_lstm(cfg, mdf, contract, trainres)
    summarydf = buildsummmary(cfg, trainres, cmdf, gaindf)
    comparisondf = collect_tradeadvice_summaries(cfg)

    println("$(EnvConfig.now()) LSTM training epochs=$(length(trainres.losses)) final_train_loss=$(trainres.losses[end]) final_eval_loss=$(trainres.eval_losses[end])")
    println("$(EnvConfig.now()) LSTM confusion matrix rows=$(size(cmdf, 1)) extended rows=$(size(xcmdf, 1))")
    println("$(EnvConfig.now()) LSTM predictions rows=$(size(evaldf, 1)) gains rows=$(size(gaindf, 1))")
    println("$(EnvConfig.now()) TradeAdviceLstm summary: $summarydf")
    if size(comparisondf, 1) > 0
        println("$(EnvConfig.now()) TradeAdviceLstm comparison table: $comparisondf")
    end
    return (inputdf=mdf, trainres=trainres, evaldf=evaldf, cmdf=cmdf, xcmdf=xcmdf, gaindf=gaindf, summarydf=summarydf, comparisondf=comparisondf)
end

function buildcfg(args::Vector{String})
    tradeadviceref = _argvalue(args, "tradeadvice", nothing)
    tradeadvicecfg = isnothing(tradeadviceref) ? nothing : _resolve_tradeadviceconfig(tradeadviceref)
    trendref = _argvalue(args, "trend", isnothing(tradeadvicecfg) ? "025" : string(tradeadvicecfg.trendconfigref))
    boundsref = _argvalue(args, "bounds", isnothing(tradeadvicecfg) ? "001" : string(tradeadvicecfg.boundsconfigref))
    trendconfig = _resolve_trendconfig(trendref)
    boundsconfig = _resolve_boundsconfig(boundsref)
    configname = _argvalue(args, "configname", isnothing(tradeadvicecfg) ? "trend$(trendconfig.configname)_bounds$(boundsconfig.configname)" : string(tradeadvicecfg.configname))
    folder = _argvalue(args, "folder", "TradeAdviceLstm-$configname-$(EnvConfig.configmode)")
    seqlen = parse(Int, _argvalue(args, "seqlen", string(_ntget(tradeadvicecfg, :seqlen, 3))))
    hidden_dim = parse(Int, _argvalue(args, "hidden", _argvalue(args, "hidden_dim", string(_ntget(tradeadvicecfg, :hidden_dim, 32)))))
    maxepoch = parse(Int, _argvalue(args, "maxepoch", isnothing(tradeadvicecfg) ? ("train" in args ? "200" : "20") : string(tradeadvicecfg.maxepoch)))
    batchsize = parse(Int, _argvalue(args, "batchsize", string(_ntget(tradeadvicecfg, :batchsize, 64))))
    entryfraction = parse(Float32, _argvalue(args, "entryfraction", string(_ntget(tradeadvicecfg, :entryfraction, 0.1f0))))
    exitfraction = parse(Float32, _argvalue(args, "exitfraction", string(_ntget(tradeadvicecfg, :exitfraction, 0.1f0))))
    default_opens = join(string.(Float32.(_ntget(tradeadvicecfg, :openthresholds, Float32[0.8f0, 0.7f0, 0.6f0]))), ",")
    default_closes = join(string.(Float32.(_ntget(tradeadvicecfg, :closethresholds, Float32[0.6f0, 0.55f0, 0.5f0]))), ",")
    openthresholds = _parse_float32_list(_argvalue(args, "openthresholds", default_opens))
    closethresholds = _parse_float32_list(_argvalue(args, "closethresholds", default_closes))
    entrytimeout = parse(Int, _argvalue(args, "entrytimeout", string(_ntget(tradeadvicecfg, :entrytimeout, 2))))
    exittimeout = parse(Int, _argvalue(args, "exittimeout", string(_ntget(tradeadvicecfg, :exittimeout, 2))))
    exitstrategy = Symbol(_argvalue(args, "exitstrategy", string(_ntget(tradeadvicecfg, :exitstrategy, :opposite_signal_market))))
    return TradeAdviceLstmConfig(; configname=configname, folder=folder, trendconfig=trendconfig, boundsconfig=boundsconfig, seqlen=seqlen, hidden_dim=hidden_dim, maxepoch=maxepoch, batchsize=batchsize, entryfraction=entryfraction, exitfraction=exitfraction, openthresholds=openthresholds, closethresholds=closethresholds, entrytimeout=entrytimeout, exittimeout=exittimeout, exitstrategy=exitstrategy, mode=EnvConfig.configmode)
end

println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$ARGS")

testmode = "train" in ARGS ? false : true
inspectonly = "inspect" in ARGS
compareonly = "compare" in ARGS

if testmode
    EnvConfig.init(test)
else
    EnvConfig.init(training)
end

cfg = buildcfg(ARGS)

if inspectonly
    println("Using log folder $(EnvConfig.logfolder())")
    println("Selected trend config: $(cfg.trendconfig.configname)")
    println("Selected bounds config: $(cfg.boundsconfig.configname)")
    println("LSTM params: seqlen=$(cfg.seqlen) hidden_dim=$(cfg.hidden_dim) maxepoch=$(cfg.maxepoch) batchsize=$(cfg.batchsize) entryfraction=$(cfg.entryfraction) exitfraction=$(cfg.exitfraction) entrytimeout=$(cfg.entrytimeout) exittimeout=$(cfg.exittimeout)")
    println("Trend source folder: $(trendfolder(cfg))")
    println("Bounds source folder: $(boundsfolder(cfg))")
elseif compareonly
    comparisondf = collect_tradeadvice_summaries(cfg)
    println("$(EnvConfig.now()) available TradeAdvice summaries: $comparisondf")
else
    run_pipeline(cfg)
end

println("$(EnvConfig.now()) done @ $(cfg.folder)")

end # module
