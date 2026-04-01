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

const LSTM_LABELS = ["longbuy", "longclose", "shortbuy", "shortclose"]

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
    mode::EnvConfig.Mode
    function TradeAdviceLstmConfig(;configname="025", folder="TradeAdviceLstm-$configname-$(EnvConfig.configmode)", trendconfig=mk025config(), boundsconfig=boundsmk025config(), seqlen::Int=3, hidden_dim::Int=32, maxepoch::Int=200, batchsize::Int=64, mode::EnvConfig.Mode=EnvConfig.configmode)
        @assert seqlen > 0 "seqlen=$seqlen must be > 0"
        @assert hidden_dim > 0 "hidden_dim=$hidden_dim must be > 0"
        @assert maxepoch > 0 "maxepoch=$maxepoch must be > 0"
        @assert batchsize > 0 "batchsize=$batchsize must be > 0"
        EnvConfig.setlogpath(folder)
        return new(configname, folder, trendconfig, boundsconfig, seqlen, hidden_dim, maxepoch, batchsize, mode)
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

function _map_target_lstm(rawtarget::AbstractString, longprob::Float32, shortprob::Float32)::String
    t = lowercase(rawtarget)
    if occursin("long", t) && occursin("buy", t)
        return "longbuy"
    elseif occursin("short", t) && occursin("buy", t)
        return "shortbuy"
    elseif occursin("short", t) && (occursin("close", t) || occursin("hold", t))
        return "shortclose"
    elseif occursin("long", t) && (occursin("close", t) || occursin("hold", t))
        return "longclose"
    else
        return shortprob > longprob ? "shortclose" : "longclose"
    end
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
    @assert all(c -> c in names(featuresdf), fcols) "trend features dataframe missing required columns"
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
Load optimized bounds estimator outputs aligned to sample rows.
"""
function get_bounds_predictions(cfg::TradeAdviceLstmConfig)
    bfolder = boundsfolder(cfg)
    bdf = _load_df_or_assert(bfolder, predictionsfilename())
    required = [:sampleix, :pred_center, :pred_width]
    @assert all(c -> c in names(bdf), required) "bounds predictions missing required columns $(required)"
    return bdf
end

"""
Merge optimized trend + bounds outputs and build an LSTM contract dataframe.
"""
function build_lstm_input_df(cfg::TradeAdviceLstmConfig)
    trend_results, trend_probs = get_trend_probabilities(cfg)
    bounds_pred = get_bounds_predictions(cfg)

    cols = [:sampleix, :rangeid, :set, :target, :opentime, :high, :low, :close, :pivot]
    @assert all(c -> c in names(trend_results), cols) "trend results missing required columns"

    mdf = innerjoin(trend_results[!, cols], bounds_pred[!, [:sampleix, :pred_center, :pred_width]], on=:sampleix)
    @assert size(mdf, 1) > 0 "empty merge between trend and bounds data"

    for c in [:longbuy, :shortbuy, :allclose]
        if c in names(trend_probs)
            mdf[!, c] = Float32.(trend_probs[!, c])
        else
            mdf[!, c] = zeros(Float32, size(mdf, 1))
        end
    end

    # Fallback if allclose is not directly available from classifier labels.
    if !(:allclose in names(trend_probs))
        mdf[!, :allclose] = max.(1f0 .- mdf[!, :longbuy] .- mdf[!, :shortbuy], 0f0)
    end

    mapped_target = String[]
    rawtargets = string.(mdf[!, :target])
    for ix in eachindex(rawtargets)
        push!(mapped_target, _map_target_lstm(rawtargets[ix], mdf[ix, :longbuy], mdf[ix, :shortbuy]))
    end
    mdf[!, :target] = mapped_target

    # Keep deterministic order for window generation.
    sort!(mdf, [:rangeid, :sampleix])

    EnvConfig.savedf(mdf, lstmmergedfilename())
    return mdf
end

function train_lstm(cfg::TradeAdviceLstmConfig, mdf::AbstractDataFrame)
    contract = Classify.lstm_bounds_trend_features(
        mdf;
        trendprobcols=[:longbuy, :shortbuy, :allclose],
        centercol=:pred_center,
        widthcol=:pred_width,
        targetcol=:target,
        setcol=:set,
        rangeidcol=:rangeid,
        rixcol=:sampleix,
    )

    res = Classify.train_lstm_trade_signals!(contract, cfg.seqlen; hidden_dim=cfg.hidden_dim, maxepoch=cfg.maxepoch, batchsize=cfg.batchsize)

    lossesdf = DataFrame(epoch=collect(1:length(res.losses)), train_loss=Float32.(res.losses), eval_loss=Float32.(res.eval_losses))
    EnvConfig.savedf(lossesdf, lstmlossesfilename())
    return contract, res
end

function evaluate_lstm(cfg::TradeAdviceLstmConfig, mdf::AbstractDataFrame, contract, trainres)
    windows = Classify.lstm_tensor_windows(contract; seqlen=cfg.seqlen)
    @assert size(windows.X, 3) > 0 "no windows generated for evaluation"

    probs = Classify.predict_lstm_trade_signals(trainres.model, windows.X)
    predix = vec(argmax(probs; dims=1))
    predlabel = [LSTM_LABELS[ci[1]] for ci in predix]
    predscore = [Float32(probs[ci[1], ix]) for (ix, ci) in enumerate(predix)]

    admincols = [:sampleix, :opentime, :high, :low, :close, :pivot, :set, :rangeid]
    admindf = mdf[!, admincols]
    evaldf = innerjoin(
        DataFrame(sampleix=windows.endrix, target=windows.targets, pred_label=predlabel, score=predscore),
        admindf,
        on=:sampleix,
    )
    @assert size(evaldf, 1) == length(windows.targets) "window/admin merge mismatch"

    evaldf[!, :label] = Targets.tradelabel.(evaldf[!, :pred_label])
    evaldf[!, :target] = Targets.tradelabel.(string.(evaldf[!, :target]))
    evaldf[!, :set] = CategoricalVector(string.(evaldf[!, :set]), levels=settypes())

    EnvConfig.savedf(evaldf, lstmpredictionsfilename())

    alltl = [longbuy, longclose, shortbuy, shortclose]
    cmdf = Classify.confusionmatrix(evaldf, alltl)
    xcmdf = Classify.extendedconfusionmatrix(evaldf, alltl)
    EnvConfig.savedf(cmdf, lstmconfusionfilename())
    EnvConfig.savedf(xcmdf, lstmxconfusionfilename())

    gsegment = TradingStrategy.GainSegment(maxwindow=4*60, algorithm=TradingStrategy.algorithm02!, openthreshold=0.6f0, closethreshold=0.5f0, makerfee=0.0015f0, takerfee=0.002f0)
    gaindf = DataFrame()
    ranges = unique(evaldf[!, :rangeid])
    for rid in ranges
        rview = @view evaldf[evaldf[!, :rangeid] .== rid, :]
        sort!(rview, :opentime)
        for (openthreshold, closethreshold) in [(0.8f0, 0.5f0), (0.7f0, 0.5f0), (0.6f0, 0.5f0), (0.8f0, 0.6f0), (0.7f0, 0.6f0), (0.6f0, 0.55f0)]
            TradingStrategy.reset!(gsegment)
            gdf = TradingStrategy.getgains(gsegment, rview, rview[!, :score], rview[!, :label], true, openthreshold=openthreshold, closethreshold=closethreshold)
            if size(gdf, 1) > 0
                gdf[!, :rangeid] = fill(rid, size(gdf, 1))
                gdf[!, :set] = fill(string(rview[begin, :set]), size(gdf, 1))
                gdf[!, :predicted] = fill(true, size(gdf, 1))
                gdf[!, :openthreshold] = fill(openthreshold, size(gdf, 1))
                gdf[!, :closethreshold] = fill(closethreshold, size(gdf, 1))
                gaindf = size(gaindf, 1) == 0 ? gdf : vcat(gaindf, gdf)
            end
        end

        TradingStrategy.reset!(gsegment)
        gdf = TradingStrategy.getgains(gsegment, rview, fill(1f0, size(rview, 1)), rview[!, :target], true, openthreshold=0.9f0, closethreshold=0.9f0)
        if size(gdf, 1) > 0
            gdf[!, :rangeid] = fill(rid, size(gdf, 1))
            gdf[!, :set] = fill(string(rview[begin, :set]), size(gdf, 1))
            gdf[!, :predicted] = fill(false, size(gdf, 1))
            gdf[!, :openthreshold] = fill(0.9f0, size(gdf, 1))
            gdf[!, :closethreshold] = fill(0.9f0, size(gdf, 1))
            gaindf = size(gaindf, 1) == 0 ? gdf : vcat(gaindf, gdf)
        end
    end
    if size(gaindf, 1) > 0
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

function buildsummmary(cfg::TradeAdviceLstmConfig, trainres, cmdf::AbstractDataFrame, gaindf::AbstractDataFrame)
    ppvcol = Symbol("ppv%")
    evalcmdf = size(cmdf, 1) == 0 ? DataFrame() : @view cmdf[cmdf[!, :set] .== "eval", :]
    testcmdf = size(cmdf, 1) == 0 ? DataFrame() : @view cmdf[cmdf[!, :set] .== "test", :]
    evalpred = size(gaindf, 1) == 0 ? DataFrame() : @view gaindf[(gaindf[!, :set] .== "eval") .&& (gaindf[!, :predicted] .== true), :]
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
        epochs=length(trainres.losses),
        final_train_loss=Float32(trainres.losses[end]),
        final_eval_loss=Float32(trainres.eval_losses[end]),
        eval_mean_ppv=_safe_mean_or_missing(size(evalcmdf, 1) == 0 ? Float32[] : evalcmdf[!, ppvcol]),
        test_mean_ppv=_safe_mean_or_missing(size(testcmdf, 1) == 0 ? Float32[] : testcmdf[!, ppvcol]),
        eval_pred_segments=size(evalpred, 1),
        eval_true_segments=size(evaltrue, 1),
        eval_pred_gainfee_sum=_safe_sum_or_zero(size(evalpred, 1) == 0 ? Float32[] : evalpred[!, :gainfee]),
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
    boundsref = _argvalue(args, "bounds", isnothing(tradeadvicecfg) ? trendref : string(tradeadvicecfg.boundsconfigref))
    trendconfig = _resolve_trendconfig(trendref)
    boundsconfig = _resolve_boundsconfig(boundsref)
    configname = _argvalue(args, "configname", isnothing(tradeadvicecfg) ? "trend$(trendconfig.configname)_bounds$(boundsconfig.configname)" : string(tradeadvicecfg.configname))
    folder = _argvalue(args, "folder", "TradeAdviceLstm-$configname-$(EnvConfig.configmode)")
    seqlen = parse(Int, _argvalue(args, "seqlen", isnothing(tradeadvicecfg) ? "3" : string(tradeadvicecfg.seqlen)))
    hidden_dim = parse(Int, _argvalue(args, "hidden", _argvalue(args, "hidden_dim", isnothing(tradeadvicecfg) ? "32" : string(tradeadvicecfg.hidden_dim))))
    maxepoch = parse(Int, _argvalue(args, "maxepoch", isnothing(tradeadvicecfg) ? ("train" in args ? "200" : "20") : string(tradeadvicecfg.maxepoch)))
    batchsize = parse(Int, _argvalue(args, "batchsize", isnothing(tradeadvicecfg) ? "64" : string(tradeadvicecfg.batchsize)))
    return TradeAdviceLstmConfig(; configname=configname, folder=folder, trendconfig=trendconfig, boundsconfig=boundsconfig, seqlen=seqlen, hidden_dim=hidden_dim, maxepoch=maxepoch, batchsize=batchsize, mode=EnvConfig.configmode)
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
    println("LSTM params: seqlen=$(cfg.seqlen) hidden_dim=$(cfg.hidden_dim) maxepoch=$(cfg.maxepoch) batchsize=$(cfg.batchsize)")
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
