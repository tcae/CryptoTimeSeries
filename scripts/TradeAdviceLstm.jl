module TradeAdviceLstm
using Test, Dates, Logging, CSV, JDF, DataFrames, Statistics
using CategoricalArrays
using EnvConfig, Classify, Ohlcv, Features, Targets, TradingStrategy

const Tables = DataFrames.Tables

include("optimizationconfigs.jl")

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 2

const LSTM_LABELS = ["up", "down", "flat"]

function _stripconfigsuffix(name::AbstractString)
    return replace(strip(name), r"config$" => "")
end

function _resolve_trendconfig(configref::AbstractString)
    raw = lowercase(_stripconfigsuffix(configref))
    symbol = startswith(raw, "mk") ? Symbol(raw * "config") : Symbol("mk" * raw * "config")
    @assert isdefined(@__MODULE__, symbol) "unknown trend config '$configref'; expected function $(symbol) in optimizationconfigs.jl"
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

"""Configuration for the end-to-end LSTM trend-smoothing backtest pipeline."""
mutable struct TradeAdviceLstmConfig
    configname::String
    folder::String
    trendconfig::NamedTuple
    seqlen::Int
    hidden_dim::Int
    maxepoch::Int
    batchsize::Int
    openthresholds::Vector{Float32} # config of classifier score thresholds to be evaluated for opening trades; expected to be in [0, 1] 
    closethresholds::Vector{Float32} # config of classifier score thresholds to be evaluated for closing trades; expected to be in [0, 1]
    entrytimeout::Int  # currently unused
    exittimeout::Int # currently unused
    exitstrategy::Symbol # currently unused
    mode::EnvConfig.Mode
    function TradeAdviceLstmConfig(;configname="025", folder="TradeAdviceLstm-$configname-$(EnvConfig.configmode)", trendconfig=mk025config(), seqlen::Int=3, hidden_dim::Int=32, maxepoch::Int=200, batchsize::Int=64, openthresholds::Vector{Float32}=Float32[0.8f0, 0.7f0, 0.6f0], closethresholds::Vector{Float32}=Float32[0.6f0, 0.55f0, 0.5f0], entrytimeout::Int=2, exittimeout::Int=2, exitstrategy::Symbol=:opposite_signal_market, mode::EnvConfig.Mode=EnvConfig.configmode)
        @assert seqlen > 0 "seqlen=$seqlen must be > 0"
        @assert hidden_dim > 0 "hidden_dim=$hidden_dim must be > 0"
        @assert maxepoch > 0 "maxepoch=$maxepoch must be > 0"
        @assert batchsize > 0 "batchsize=$batchsize must be > 0"
        @assert entrytimeout >= 0 "entrytimeout=$entrytimeout must be >= 0"
        @assert exittimeout >= 0 "exittimeout=$exittimeout must be >= 0"
        @assert !isempty(openthresholds) "openthresholds must not be empty"
        @assert !isempty(closethresholds) "closethresholds must not be empty"
        @assert all(0f0 .<= openthresholds .<= 1f0) "expected openthresholds within [0, 1]; got openthresholds=$(openthresholds)"
        @assert all(0f0 .<= closethresholds .<= 1f0) "expected closethresholds within [0, 1]; got closethresholds=$(closethresholds)"
        @assert exitstrategy in (:opposite_signal_market,) "unsupported exitstrategy=$exitstrategy; expected :opposite_signal_market"
        EnvConfig.setlogpath(folder)
        return new(configname, folder, trendconfig, seqlen, hidden_dim, maxepoch, batchsize, openthresholds, closethresholds, entrytimeout, exittimeout, exitstrategy, mode)
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
lstmsequencesfilename() = "lstm_sequences.jdf"
lstmdistancesfilename() = joinpath("trades", "lstm_distances.jdf")
lstmpairsfilename() = joinpath("trades", "lstm_transaction_pairs_all.jdf")
lstmgainsfilename() = joinpath("trades", "lstm_gains_all.jdf")
summaryfilename() = "summary.jdf"

trendfolder(cfg::TradeAdviceLstmConfig) = "Trend-$(cfg.trendconfig.configname)-$(cfg.mode)"

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

function _logsroot()
    current = EnvConfig.logfolder()
    return EnvConfig.logsubfolder() == "" ? current : normpath(dirname(current))
end

function _folderpath(subfolder::AbstractString)
    return normpath(joinpath(_logsroot(), subfolder))
end

function _arrowartifactstem(filename::AbstractString)::String
    base = splitext(basename(filename))[1]
    if base == "features"
        return joinpath("features", "all")
    elseif base == "targets"
        return joinpath("targets", "all")
    elseif base == "results"
        return joinpath("results", "all")
    elseif base == "maxpredictions"
        return joinpath("predictions", "maxpredictions")
    elseif base == "predictions"
        return joinpath("predictions", "all")
    elseif base == "trades"
        return joinpath("trades", "all")
    end
    return base
end

function _table_rowcount(table)::Int
    table isa AbstractDataFrame && return size(table, 1)
    schema = Tables.schema(table)
    if isnothing(schema) || isempty(schema.names)
        return 0
    end
    return length(Tables.getcolumn(Tables.columns(table), first(schema.names)))
end

function _table_colnames(table)::Vector{String}
    table isa AbstractDataFrame && return names(table)
    schema = Tables.schema(table)
    return isnothing(schema) ? String[] : string.(collect(schema.names))
end

function _load_df_or_assert(folder::AbstractString, filename::AbstractString; format::Symbol=:jdf)
    fp = _folderpath(folder)
    stem = format == :arrow ? _arrowartifactstem(filename) : filename
    df = EnvConfig.readdf(stem; folderpath=fp, format=format)
    if isnothing(df) && (format != :jdf)
        df = EnvConfig.readdf(filename; folderpath=fp, format=:jdf)
    end
    @assert !isnothing(df) && size(df, 1) > 0 "missing or empty $(filename) in $(fp)"
    return df
end

function _load_table_or_assert(folder::AbstractString, filename::AbstractString; format::Symbol=:jdf, materialize::Bool=true)
    fp = _folderpath(folder)
    stem = format == :arrow ? _arrowartifactstem(filename) : filename
    table = EnvConfig.readtable(stem; folderpath=fp, format=:auto, preferred=format, materialize=materialize)
    if isnothing(table) && (format != :jdf)
        table = EnvConfig.readtable(filename; folderpath=fp, format=:jdf, preferred=:jdf, materialize=materialize)
    end
    @assert !isnothing(table) && (_table_rowcount(table) > 0) "missing or empty $(filename) in $(fp)"
    return table
end

@inline function _tradelabel_string(label)
    if label isa Targets.TradeLabel
        return string(label)
    elseif label isa Integer
        try
            return string(Targets.tradelabel(Int(label)))
        catch
            return string(label)
        end
    end
    raw = strip(string(label))
    parsed = tryparse(Int, raw)
    if !isnothing(parsed)
        try
            return string(Targets.tradelabel(parsed))
        catch
        end
    end
    return raw
end

@inline function _trendlabel2phase(label)::String
    lbl = lowercase(strip(_tradelabel_string(label)))
    if (lbl == "longbuy") || (lbl == "longhold")
        return "up"
    elseif (lbl == "shortbuy") || (lbl == "shorthold")
        return "down"
    elseif lbl in ("allclose", "longclose", "shortclose", "ignore")
        return "flat"
    end
    @warn "unexpected trend label $(label); mapping to flat"
    return "flat"
end

@inline _phase2trendphase(phase::AbstractString) = phase == "up" ? up : (phase == "down" ? down : flat)
@inline _phasegain(phase::AbstractString, startprice::Real, endprice::Real)::Float32 = phase == "up" ? Float32((endprice - startprice) / startprice) : (phase == "down" ? Float32(-(endprice - startprice) / startprice) : 0f0)

function _lstm_featurecols(mdf::AbstractDataFrame)
    cols = [Symbol(name) for name in names(mdf) if startswith(String(name), "lay3_")]
    @assert !isempty(cols) "expected lay3_* feature columns in merged LSTM dataframe; names=$(names(mdf))"
    return cols
end

_sequence_minutes(startdt::DateTime, enddt::DateTime)::Int = Int(div(Dates.value(enddt - startdt), 60000)) + 1

function _overlap_minutes(start1::DateTime, end1::DateTime, start2::DateTime, end2::DateTime)::Int
    ovstart = max(start1, start2)
    ovend = min(end1, end2)
    return ovstart > ovend ? 0 : _sequence_minutes(ovstart, ovend)
end

function _phase_score_table(evaldf::AbstractDataFrame, labels::Vector{String}; thresholdbins::Int=10)
    @assert thresholdbins > 0 "thresholdbins=$(thresholdbins) must be > 0"
    rows = NamedTuple[]
    setnames = sort!(collect(unique(string.(evaldf[!, :set]))))
    for setname in setnames
        sdf = @view evaldf[string.(evaldf[!, :set]) .== setname, :]
        for label in labels
            for bix in 1:thresholdbins
                low = Float32((bix - 1) / thresholdbins)
                high = Float32(bix / thresholdbins)
                predmask = (sdf[!, :label] .== label) .&& (Float32.(sdf[!, :score]) .>= low)
                tp = count(predmask .&& (sdf[!, :target] .== label))
                fp = count(predmask .&& (sdf[!, :target] .!= label))
                fn = count((sdf[!, :target] .== label) .&& .!predmask)
                ppv = (tp + fp) == 0 ? missing : Float32(round(tp / (tp + fp) * 100; digits=1))
                recall = (tp + fn) == 0 ? missing : Float32(round(tp / (tp + fn) * 100; digits=1))
                push!(rows, (set=setname, pred_label=label, bin="$bix/[$(round(low; digits=2))-$(round(high; digits=2))]", tp=tp, fp=fp, fn=fn, ppv=ppv, recall=recall))
            end
        end
    end
    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

function _phase_sequences(evaldf::AbstractDataFrame, labelcol::Symbol; predicted::Bool)
    rows = NamedTuple[]
    if size(evaldf, 1) == 0
        return DataFrame()
    end

    ordered = sort(copy(evaldf), [:coin, :rangeid, :opentime])
    for g in groupby(ordered, [:coin, :rangeid, :set])
        phases = string.(g[!, labelcol])
        scores = :score in propertynames(g) ? Float32.(g[!, :score]) : fill(1f0, size(g, 1))
        i = 1
        while i <= size(g, 1)
            phase = phases[i]
            j = i
            while (j < size(g, 1)) && (phases[j + 1] == phase)
                j += 1
            end
            startdt = g[i, :opentime]
            enddt = g[j, :opentime]
            startprice = Float32(g[i, :pivot])
            endprice = Float32(g[j, :pivot])
            push!(rows, (
                coin=String(g[i, :coin]),
                rangeid=Int32(g[i, :rangeid]),
                set=String(g[i, :set]),
                predicted=predicted,
                phase=phase,
                startdt=startdt,
                enddt=enddt,
                startix=Int(i),
                endix=Int(j),
                minutes=_sequence_minutes(startdt, enddt),
                startprice=startprice,
                endprice=endprice,
                gain=_phasegain(phase, startprice, endprice),
                meanscore=predicted ? Float32(mean(scores[i:j])) : missing,
            ))
            i = j + 1
        end
    end
    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

function _sequence_distances(seqdf::AbstractDataFrame)
    rows = NamedTuple[]
    if size(seqdf, 1) == 0
        return DataFrame()
    end

    directional = @view seqdf[seqdf[!, :phase] .!= "flat", :]
    if size(directional, 1) == 0
        return DataFrame()
    end

    for g in groupby(directional, [:coin, :rangeid, :set, :phase])
        pred = sort(g[g[!, :predicted] .== true, :], :startdt)
        truth = sort(g[g[!, :predicted] .== false, :], :startdt)
        matched_truth = falses(size(truth, 1))

        for prow in eachrow(pred)
            overlaps = size(truth, 1) == 0 ? Int[] : [_overlap_minutes(prow.startdt, prow.enddt, trow.startdt, trow.enddt) for trow in eachrow(truth)]
            bestoverlap = isempty(overlaps) ? 0 : maximum(overlaps)
            if bestoverlap > 0
                bestix = argmax(overlaps)
                matched_truth[bestix] = true
                trow = truth[bestix, :]
                push!(rows, (
                    coin=String(prow.coin),
                    rangeid=Int32(prow.rangeid),
                    set=String(prow.set),
                    phase=String(prow.phase),
                    source="predicted",
                    matched=true,
                    overlap_minutes=bestoverlap,
                    pred_minutes=Int(prow.minutes),
                    target_minutes=Int(trow.minutes),
                    minutesdiff=Int(prow.minutes - trow.minutes),
                    startdist=Minute(prow.startdt - trow.startdt).value,
                    enddist=Minute(prow.enddt - trow.enddt).value,
                    pred_gain=Float32(prow.gain),
                    target_gain=Float32(trow.gain),
                    gaindiff=Float32(prow.gain - trow.gain),
                    startdt=prow.startdt,
                    enddt=prow.enddt,
                    truestartdt=trow.startdt,
                    trueenddt=trow.enddt,
                ))
            else
                push!(rows, (
                    coin=String(prow.coin),
                    rangeid=Int32(prow.rangeid),
                    set=String(prow.set),
                    phase=String(prow.phase),
                    source="predicted",
                    matched=false,
                    overlap_minutes=0,
                    pred_minutes=Int(prow.minutes),
                    target_minutes=missing,
                    minutesdiff=missing,
                    startdist=missing,
                    enddist=missing,
                    pred_gain=Float32(prow.gain),
                    target_gain=missing,
                    gaindiff=missing,
                    startdt=prow.startdt,
                    enddt=prow.enddt,
                    truestartdt=missing,
                    trueenddt=missing,
                ))
            end
        end

        for truthix in eachindex(matched_truth)
            if !matched_truth[truthix]
                trow = truth[truthix, :]
                push!(rows, (
                    coin=String(trow.coin),
                    rangeid=Int32(trow.rangeid),
                    set=String(trow.set),
                    phase=String(trow.phase),
                    source="target_only",
                    matched=false,
                    overlap_minutes=0,
                    pred_minutes=0,
                    target_minutes=Int(trow.minutes),
                    minutesdiff=missing,
                    startdist=missing,
                    enddist=missing,
                    pred_gain=0f0,
                    target_gain=Float32(trow.gain),
                    gaindiff=missing,
                    startdt=trow.startdt,
                    enddt=trow.enddt,
                    truestartdt=trow.startdt,
                    trueenddt=trow.enddt,
                ))
            end
        end
    end

    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

function _sequence_summary(distdf::AbstractDataFrame, setname::AbstractString)
    if size(distdf, 1) == 0
        return (predicted_segments=0, matched_segments=0, mean_minutesdiff=missing, mean_abs_startdist=missing, mean_abs_enddist=missing, mean_gaindiff=missing)
    end

    setmask = (distdf[!, :set] .== setname) .&& (distdf[!, :source] .== "predicted")
    sdf = @view distdf[setmask, :]
    matched = @view sdf[sdf[!, :matched] .== true, :]
    return (
        predicted_segments=size(sdf, 1),
        matched_segments=size(matched, 1),
        mean_minutesdiff=_safe_mean_or_missing(size(matched, 1) == 0 ? Int[] : collect(skipmissing(matched[!, :minutesdiff]))),
        mean_abs_startdist=_safe_mean_or_missing(size(matched, 1) == 0 ? Int[] : abs.(collect(skipmissing(matched[!, :startdist])))),
        mean_abs_enddist=_safe_mean_or_missing(size(matched, 1) == 0 ? Int[] : abs.(collect(skipmissing(matched[!, :enddist])))),
        mean_gaindiff=_safe_mean_or_missing(size(matched, 1) == 0 ? Float32[] : Float32.(collect(skipmissing(matched[!, :gaindiff])))),
    )
end

"""
Load optimized TrendDetector hidden activations and return the penultimate
`model002` features aligned to the trend results rows.
"""
function get_trend_hidden_features(cfg::TradeAdviceLstmConfig)
    trfolder = trendfolder(cfg)
    resultsdf = _load_df_or_assert(trfolder, resultsfilename(); format=:arrow)
    featuretable = _load_table_or_assert(trfolder, featuresfilename(); format=:arrow, materialize=false)

    if (!(:label in propertynames(resultsdf)) || !(:score in propertynames(resultsdf)))
        predictionsdf = _load_df_or_assert(trfolder, predictionsfilename(); format=:arrow)
        @assert size(predictionsdf, 1) == size(resultsdf, 1) "trend results/predictions row mismatch: $(size(resultsdf, 1)) != $(size(predictionsdf, 1))"
        resultsdf = copy(resultsdf)
        resultsdf[!, :label] = predictionsdf[!, :label]
        resultsdf[!, :score] = Float32.(predictionsdf[!, :score])
    end

    @assert all(col -> col in propertynames(resultsdf), [:target, :label, :score]) "trend results missing required target/prediction columns; names=$(names(resultsdf))"
    @assert size(resultsdf, 1) == _table_rowcount(featuretable) "trend results/features row mismatch: $(size(resultsdf, 1)) != $(_table_rowcount(featuretable))"

    fcols = Features.requestedcolumns(cfg.trendconfig.featconfig)
    available = _table_colnames(featuretable)
    @assert all(c -> string(c) in available, fcols) "trend features table missing required columns: required=$(fcols) available=$(available)"

    hiddenmat = _with_log_subfolder(trfolder) do
        nn = cfg.trendconfig.classifiermodel(Features.featurecount(cfg.trendconfig.featconfig), Targets.uniquelabels(cfg.trendconfig.targetconfig), "mix")
        @assert isfile(Classify.nnfilename(nn.fileprefix)) "optimized trend classifier file not found: $(Classify.nnfilename(nn.fileprefix)). Run scripts/TrendDetector.jl first."
        nn = Classify.loadnn(nn.fileprefix)
        Classify.penultimatefeatures(nn, featuretable, fcols; batchsize=max(1024, cfg.batchsize * 64))
    end

    @assert size(hiddenmat, 2) == size(resultsdf, 1) "trend hidden features/results row mismatch"
    hiddennames = ["lay3_" * lpad(string(featix), 3, '0') for featix in 1:size(hiddenmat, 1)]
    GC.gc()
    return resultsdf, hiddenmat, hiddennames
end

function _rowperm_by_keycols(df::AbstractDataFrame, keycols::Vector{Symbol})
    perm = collect(1:size(df, 1))
    sort!(perm; by=ix -> Tuple(df[ix, col] for col in keycols))
    return perm
end

"""
Merge TrendDetector lay3 activations with the TrendDetector outputs and build the
LSTM input dataframe.

The LSTM is trained on the coarse phase target `up/down/flat` and later compared
against the TrendDetector phase baseline using close-price trade execution.
"""
function build_lstm_input_df(cfg::TradeAdviceLstmConfig)
    trend_results, hiddenmat, hiddennames = get_trend_hidden_features(cfg)

    cols = [:coin, :rangeid, :set, :target, :label, :score, :opentime, :high, :low, :close, :pivot]
    @assert all(c -> c in propertynames(trend_results), cols) "trend results missing required columns"

    mdf = copy(trend_results[!, cols])
    @assert size(mdf, 1) > 0 "empty trend results for LSTM input"

    perm = _rowperm_by_keycols(mdf, [:coin, :rangeid, :opentime])
    mdf = mdf[perm, :]
    hiddenmat = hiddenmat[:, perm]

    rename!(mdf, :label => :trend_label, :score => :trend_score)
    mdf[!, :trend_target] = _tradelabel_string.(mdf[!, :target])
    mdf[!, :trend_label] = _tradelabel_string.(mdf[!, :trend_label])
    mdf[!, :trend_phase] = _trendlabel2phase.(mdf[!, :trend_label])
    mdf[!, :trend_score] = Float32.(mdf[!, :trend_score])
    mdf[!, :target] = _trendlabel2phase.(mdf[!, :trend_target])
    mdf[!, :rowix] = Int32.(1:size(mdf, 1))

    contract = Classify.lstm_feature_contract(
        hiddenmat;
        feature_names=hiddennames,
        targets=mdf[!, :target],
        sets=mdf[!, :set],
        rangeids=mdf[!, :rangeid],
        rix=mdf[!, :rowix],
    )

    EnvConfig.savedf(mdf, lstmmergedfilename())
    GC.gc()
    return mdf, contract
end

function _save_lstm_losses(trainres)
    lossesdf = DataFrame(epoch=collect(1:length(trainres.losses)), train_loss=Float32.(trainres.losses), eval_loss=Float32.(trainres.eval_losses))
    EnvConfig.savedf(lossesdf, lstmlossesfilename())
    return lossesdf
end

"""Train a fresh LSTM classifier for the current `TradeAdviceLstm` run."""
function train_lstm(cfg::TradeAdviceLstmConfig, contract)
    res = Classify.train_lstm_trade_signals!(contract, cfg.seqlen; hidden_dim=cfg.hidden_dim, maxepoch=cfg.maxepoch, batchsize=cfg.batchsize, labels=LSTM_LABELS, resume=false)
    _save_lstm_losses(res)
    GC.gc()
    return merge(res, (source="retrained",))
end

"""Load the last saved LSTM classifier checkpoint for inference-only pipeline runs."""
function load_lstm(cfg::TradeAdviceLstmConfig)
    checkpoint = Classify.load_lstm_checkpoint()
    checkpointfile = Classify.lstm_checkpoint_filename()
    @assert !isnothing(checkpoint) "missing saved LSTM classifier at $(checkpointfile); rerun with `retrain` to fit one first"
    @assert Int(checkpoint.seqlen) == cfg.seqlen "saved LSTM checkpoint seqlen=$(checkpoint.seqlen) does not match cfg.seqlen=$(cfg.seqlen); rerun with `retrain`"
    @assert Int(checkpoint.hidden_dim) == cfg.hidden_dim "saved LSTM checkpoint hidden_dim=$(checkpoint.hidden_dim) does not match cfg.hidden_dim=$(cfg.hidden_dim); rerun with `retrain`"

    labels = String.(checkpoint.labels)
    @assert labels == LSTM_LABELS "saved LSTM checkpoint labels=$(labels) do not match expected labels=$(LSTM_LABELS); rerun with `retrain`"

    res = (
        model=checkpoint.model,
        optim=checkpoint.optim,
        losses=Float32.(checkpoint.losses),
        eval_losses=Float32.(checkpoint.eval_losses),
        labels=labels,
        checkpointfile=checkpointfile,
        source="checkpoint",
    )
    _save_lstm_losses(res)
    GC.gc()
    return res
end

function evaluate_lstm(cfg::TradeAdviceLstmConfig, mdf::AbstractDataFrame, contract, trainres)
    predres = Classify.predict_lstm_trade_signals(trainres.model, contract; seqlen=cfg.seqlen, batchsize=cfg.batchsize)
    @assert size(predres.probs, 2) > 0 "no windows generated for evaluation"

    predix = vec(argmax(predres.probs; dims=1))
    predlabel = [String(trainres.labels[ci[1]]) for ci in predix]
    predscore = [Float32(predres.probs[ci[1], ix]) for (ix, ci) in enumerate(predix)]

    admincols = [:rowix, :opentime, :high, :low, :close, :pivot, :set, :rangeid, :coin, :trend_target, :trend_label, :trend_phase, :trend_score]
    admindf = mdf[!, admincols]
    evaldf = innerjoin(
        DataFrame(rowix=predres.endrix, target=String.(predres.targets), label=predlabel, score=predscore),
        admindf,
        on=:rowix,
    )
    @assert size(evaldf, 1) == length(predres.targets) "window/admin merge mismatch"
    sort!(evaldf, [:coin, :rangeid, :opentime])
    evaldf[!, :set] = CategoricalVector(string.(evaldf[!, :set]), levels=settypes())

    EnvConfig.savedf(select(evaldf, Not(:rowix)), lstmpredictionsfilename())

    cmdf = Classify.confusionmatrix(evaldf, String.(trainres.labels))
    xcmdf = _phase_score_table(evaldf, String.(trainres.labels))
    size(cmdf, 1) > 0 && EnvConfig.savedf(cmdf, lstmconfusionfilename())
    size(xcmdf, 1) > 0 && EnvConfig.savedf(xcmdf, lstmxconfusionfilename())

    predseqdf = _phase_sequences(evaldf, :label; predicted=true)
    trueseqdf = _phase_sequences(evaldf, :target; predicted=false)
    if (size(predseqdf, 1) == 0) && (size(trueseqdf, 1) == 0)
        seqdf = DataFrame()
    elseif size(predseqdf, 1) == 0
        seqdf = trueseqdf
    elseif size(trueseqdf, 1) == 0
        seqdf = predseqdf
    else
        seqdf = vcat(predseqdf, trueseqdf; cols=:union)
    end
    size(seqdf, 1) > 0 && EnvConfig.savedf(seqdf, lstmsequencesfilename())

    distdf = _sequence_distances(seqdf)
    size(distdf, 1) > 0 && EnvConfig.savedf(distdf, lstmdistancesfilename())

    strategy = cfg.trendconfig.tradingstrategy
    makerfee = Float32(strategy.makerfee)
    takerfee = Float32(strategy.takerfee)
    thresholdpairs = [(open, close) for open in cfg.openthresholds for close in cfg.closethresholds if close <= open]
    @assert !isempty(thresholdpairs) "no valid threshold pairs from openthresholds=$(cfg.openthresholds) and closethresholds=$(cfg.closethresholds)"

    gainparts = DataFrame[]
    for grouped in groupby(evaldf, [:coin, :rangeid])
        rangedf = sort(DataFrame(grouped), :opentime)
        coin = string(rangedf[begin, :coin])
        setname = string(rangedf[begin, :set])
        rid = rangedf[begin, :rangeid]

        for (openthreshold, closethreshold) in thresholdpairs
            lstmgdf = TradingStrategy.simulate_market_trade_pairs(
                rangedf,
                rangedf[!, :score],
                rangedf[!, :label];
                openthreshold=openthreshold,
                closethreshold=closethreshold,
                makerfee=makerfee,
                takerfee=takerfee,
            )
            if size(lstmgdf, 1) > 0
                lstmgdf[!, :coin] = fill(coin, size(lstmgdf, 1))
                lstmgdf[!, :set] = fill(setname, size(lstmgdf, 1))
                lstmgdf[!, :source] = fill("lstm", size(lstmgdf, 1))
                lstmgdf[!, :predicted] = fill(true, size(lstmgdf, 1))
                lstmgdf[!, :rangeid] = fill(rid, size(lstmgdf, 1))
                lstmgdf[!, :openthreshold] = fill(Float32(openthreshold), size(lstmgdf, 1))
                lstmgdf[!, :closethreshold] = fill(Float32(closethreshold), size(lstmgdf, 1))
                push!(gainparts, lstmgdf)
            end

            trendgdf = TradingStrategy.simulate_market_trade_pairs(
                rangedf,
                rangedf[!, :trend_score],
                rangedf[!, :trend_phase];
                openthreshold=openthreshold,
                closethreshold=closethreshold,
                makerfee=makerfee,
                takerfee=takerfee,
            )
            if size(trendgdf, 1) > 0
                trendgdf[!, :coin] = fill(coin, size(trendgdf, 1))
                trendgdf[!, :set] = fill(setname, size(trendgdf, 1))
                trendgdf[!, :source] = fill("trend", size(trendgdf, 1))
                trendgdf[!, :predicted] = fill(true, size(trendgdf, 1))
                trendgdf[!, :rangeid] = fill(rid, size(trendgdf, 1))
                trendgdf[!, :openthreshold] = fill(Float32(openthreshold), size(trendgdf, 1))
                trendgdf[!, :closethreshold] = fill(Float32(closethreshold), size(trendgdf, 1))
                push!(gainparts, trendgdf)
            end
        end

        targetgdf = TradingStrategy.simulate_market_trade_pairs(
            rangedf,
            fill(1f0, size(rangedf, 1)),
            rangedf[!, :target];
            openthreshold=0.9f0,
            closethreshold=0.9f0,
            makerfee=makerfee,
            takerfee=takerfee,
        )
        if size(targetgdf, 1) > 0
            targetgdf[!, :coin] = fill(coin, size(targetgdf, 1))
            targetgdf[!, :set] = fill(setname, size(targetgdf, 1))
            targetgdf[!, :source] = fill("target", size(targetgdf, 1))
            targetgdf[!, :predicted] = fill(false, size(targetgdf, 1))
            targetgdf[!, :rangeid] = fill(rid, size(targetgdf, 1))
            targetgdf[!, :openthreshold] = fill(0.9f0, size(targetgdf, 1))
            targetgdf[!, :closethreshold] = fill(0.9f0, size(targetgdf, 1))
            push!(gainparts, targetgdf)
        end
    end

    gaindf = isempty(gainparts) ? DataFrame() : reduce(vcat, gainparts; cols=:union)
    if size(gaindf, 1) > 0
        TradingStrategy.savetrades(gaindf; stem="lstm_transaction_pairs")
        TradingStrategy.savetrades(gaindf; stem="lstm_gains")
    end

    return evaldf, cmdf, xcmdf, seqdf, distdf, gaindf
end

function _safe_mean_or_missing(values)
    return isempty(values) ? missing : Float32(mean(Float32.(values)))
end

function _transpose_summarydf(summarydf::AbstractDataFrame)
    if size(summarydf, 1) == 0
        return DataFrame(metric=String[], value=Any[])
    end
    @assert size(summarydf, 1) == 1 "expected single-row summary dataframe for display; got $(size(summarydf, 1)) rows"
    return DataFrame(metric=String.(names(summarydf)), value=Any[summarydf[1, col] for col in names(summarydf)])
end

function _print_df(label::AbstractString, df::AbstractDataFrame)
    println(label)
    show(stdout, MIME("text/plain"), df; allrows=true, allcols=true, truncate=0)
    println()
end

function _safe_sum_or_zero(values)
    return isempty(values) ? 0f0 : Float32(sum(Float32.(values)))
end

_safe_diff_or_missing(a, b) = (ismissing(a) || ismissing(b)) ? missing : Float32(a - b)

function _best_eval_trade_summary(gaindf::AbstractDataFrame, source::AbstractString)
    emptyres = (
        source=source,
        openthreshold=missing,
        closethreshold=missing,
        eval_segments=0,
        eval_mean_gain=missing,
        eval_mean_gainfee=missing,
        eval_gainfee_sum=0f0,
        test_segments=0,
        test_mean_gain=missing,
        test_mean_gainfee=missing,
        test_gainfee_sum=0f0,
    )
    if size(gaindf, 1) == 0 || (:source ∉ propertynames(gaindf))
        return emptyres
    end

    evalrows = gaindf[(gaindf[!, :set] .== "eval") .&& (gaindf[!, :source] .== source), :]
    if size(evalrows, 1) == 0
        return emptyres
    end

    grouped = combine(
        groupby(evalrows, [:openthreshold, :closethreshold]),
        nrow => :eval_segments,
        :gain => _safe_mean_or_missing => :eval_mean_gain,
        :gainfee => _safe_mean_or_missing => :eval_mean_gainfee,
        :gainfee => _safe_sum_or_zero => :eval_gainfee_sum,
    )
    sort!(grouped, [order(:eval_mean_gainfee, rev=true), order(:eval_mean_gain, rev=true), order(:eval_segments, rev=true)])
    best = grouped[1, :]

    testrows = gaindf[
        (gaindf[!, :set] .== "test") .&&
        (gaindf[!, :source] .== source) .&&
        (gaindf[!, :openthreshold] .== best.openthreshold) .&&
        (gaindf[!, :closethreshold] .== best.closethreshold),
        :,
    ]

    return (
        source=source,
        openthreshold=best.openthreshold,
        closethreshold=best.closethreshold,
        eval_segments=best.eval_segments,
        eval_mean_gain=best.eval_mean_gain,
        eval_mean_gainfee=best.eval_mean_gainfee,
        eval_gainfee_sum=best.eval_gainfee_sum,
        test_segments=size(testrows, 1),
        test_mean_gain=_safe_mean_or_missing(size(testrows, 1) == 0 ? Float32[] : testrows[!, :gain]),
        test_mean_gainfee=_safe_mean_or_missing(size(testrows, 1) == 0 ? Float32[] : testrows[!, :gainfee]),
        test_gainfee_sum=_safe_sum_or_zero(size(testrows, 1) == 0 ? Float32[] : testrows[!, :gainfee]),
    )
end

function buildsummmary(cfg::TradeAdviceLstmConfig, trainres, cmdf::AbstractDataFrame, distdf::AbstractDataFrame, gaindf::AbstractDataFrame)
    ppvcol = Symbol("ppv%")
    evalcmdf = size(cmdf, 1) == 0 ? DataFrame() : @view cmdf[cmdf[!, :set] .== "eval", :]
    testcmdf = size(cmdf, 1) == 0 ? DataFrame() : @view cmdf[cmdf[!, :set] .== "test", :]
    lstmtrade = _best_eval_trade_summary(gaindf, "lstm")
    trendtrade = _best_eval_trade_summary(gaindf, "trend")
    targettrade = _best_eval_trade_summary(gaindf, "target")
    evalseq = _sequence_summary(distdf, "eval")
    testseq = _sequence_summary(distdf, "test")
    summarydf = DataFrame([(
        configname=cfg.configname,
        trendconfig=cfg.trendconfig.configname,
        trendfolder=trendfolder(cfg),
        seqlen=cfg.seqlen,
        hidden_dim=cfg.hidden_dim,
        maxepoch=cfg.maxepoch,
        batchsize=cfg.batchsize,
        entrytimeout=cfg.entrytimeout,
        exittimeout=cfg.exittimeout,
        exitstrategy=String(cfg.exitstrategy),
        epochs=length(trainres.losses),
        final_train_loss=Float32(trainres.losses[end]),
        final_eval_loss=Float32(trainres.eval_losses[end]),
        eval_mean_ppv=_safe_mean_or_missing(size(evalcmdf, 1) == 0 ? Float32[] : evalcmdf[!, ppvcol]),
        test_mean_ppv=_safe_mean_or_missing(size(testcmdf, 1) == 0 ? Float32[] : testcmdf[!, ppvcol]),
        eval_phase_segments=evalseq.predicted_segments,
        test_phase_segments=testseq.predicted_segments,
        eval_phase_matches=evalseq.matched_segments,
        test_phase_matches=testseq.matched_segments,
        eval_mean_minutesdiff=evalseq.mean_minutesdiff,
        test_mean_minutesdiff=testseq.mean_minutesdiff,
        eval_mean_abs_startdist=evalseq.mean_abs_startdist,
        test_mean_abs_startdist=testseq.mean_abs_startdist,
        eval_mean_abs_enddist=evalseq.mean_abs_enddist,
        test_mean_abs_enddist=testseq.mean_abs_enddist,
        eval_mean_gaindiff=evalseq.mean_gaindiff,
        test_mean_gaindiff=testseq.mean_gaindiff,
        best_openthreshold=lstmtrade.openthreshold,
        best_closethreshold=lstmtrade.closethreshold,
        eval_pred_segments=lstmtrade.eval_segments,
        test_pred_segments=lstmtrade.test_segments,
        eval_true_segments=targettrade.eval_segments,
        eval_pred_gainfee_sum=lstmtrade.eval_gainfee_sum,
        test_pred_gainfee_sum=lstmtrade.test_gainfee_sum,
        eval_true_gainfee_sum=targettrade.eval_gainfee_sum,
        eval_lstm_mean_gain=lstmtrade.eval_mean_gain,
        test_lstm_mean_gain=lstmtrade.test_mean_gain,
        eval_trend_mean_gain=trendtrade.eval_mean_gain,
        test_trend_mean_gain=trendtrade.test_mean_gain,
        eval_target_mean_gain=targettrade.eval_mean_gain,
        test_target_mean_gain=targettrade.test_mean_gain,
        eval_lstm_mean_gainfee=lstmtrade.eval_mean_gainfee,
        test_lstm_mean_gainfee=lstmtrade.test_mean_gainfee,
        eval_trend_mean_gainfee=trendtrade.eval_mean_gainfee,
        test_trend_mean_gainfee=trendtrade.test_mean_gainfee,
        eval_target_mean_gainfee=targettrade.eval_mean_gainfee,
        test_target_mean_gainfee=targettrade.test_mean_gainfee,
        eval_gain_delta=_safe_diff_or_missing(lstmtrade.eval_mean_gain, trendtrade.eval_mean_gain),
        test_gain_delta=_safe_diff_or_missing(lstmtrade.test_mean_gain, trendtrade.test_mean_gain),
        eval_gainfee_delta=_safe_diff_or_missing(lstmtrade.eval_mean_gainfee, trendtrade.eval_mean_gainfee),
        test_gainfee_delta=_safe_diff_or_missing(lstmtrade.test_mean_gainfee, trendtrade.test_mean_gainfee),
        trend_best_openthreshold=trendtrade.openthreshold,
        trend_best_closethreshold=trendtrade.closethreshold,
        trend_eval_segments=trendtrade.eval_segments,
        trend_test_segments=trendtrade.test_segments,
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

function run_pipeline(cfg::TradeAdviceLstmConfig; retrain::Bool=false)
    return _with_log_subfolder(cfg.folder) do
        action = retrain ? "retrain" : "reuse-checkpoint"
        (verbosity >= 2) && println("$(EnvConfig.now()) TradeAdviceLstm pipeline start with trend=$(trendfolder(cfg)) action=$(action)")
        mdf, contract = build_lstm_input_df(cfg)
        trainres = retrain ? train_lstm(cfg, contract) : load_lstm(cfg)
        GC.gc()
        evaldf, cmdf, xcmdf, seqdf, distdf, gaindf = evaluate_lstm(cfg, mdf, contract, trainres)
        summarydf = buildsummmary(cfg, trainres, cmdf, distdf, gaindf)
        comparisondf = collect_tradeadvice_summaries(cfg)

        println("$(EnvConfig.now()) LSTM model source=$(trainres.source) epochs=$(length(trainres.losses)) final_train_loss=$(trainres.losses[end]) final_eval_loss=$(trainres.eval_losses[end])")
        println("$(EnvConfig.now()) LSTM confusion matrix rows=$(size(cmdf, 1)) extended rows=$(size(xcmdf, 1))")
        println("$(EnvConfig.now()) LSTM predictions rows=$(size(evaldf, 1)) sequences rows=$(size(seqdf, 1)) distances rows=$(size(distdf, 1)) gains rows=$(size(gaindf, 1))")
        _print_df("$(EnvConfig.now()) TradeAdviceLstm summary (transposed):", _transpose_summarydf(summarydf))
        if size(comparisondf, 1) > 0
            println("$(EnvConfig.now()) TradeAdviceLstm comparison table: $comparisondf")
        end
        return (inputdf=mdf, trainres=trainres, evaldf=evaldf, cmdf=cmdf, xcmdf=xcmdf, seqdf=seqdf, distdf=distdf, gaindf=gaindf, summarydf=summarydf, comparisondf=comparisondf)
    end
end

function buildcfg(args::Vector{String})
    retrainrequested = "retrain" in args
    tradeadviceref = _argvalue(args, "tradeadvice", nothing)
    tradeadvicecfg = isnothing(tradeadviceref) ? nothing : _resolve_tradeadviceconfig(tradeadviceref)
    trendref = _argvalue(args, "trend", isnothing(tradeadvicecfg) ? "025" : string(tradeadvicecfg.trendconfigref))
    trendconfig = _resolve_trendconfig(trendref)
    configname = _argvalue(args, "configname", isnothing(tradeadvicecfg) ? "trend$(trendconfig.configname)" : string(tradeadvicecfg.configname))
    folder = _argvalue(args, "folder", "TradeAdviceLstm-$configname-$(EnvConfig.configmode)")
    seqlen = parse(Int, _argvalue(args, "seqlen", string(_ntget(tradeadvicecfg, :seqlen, 30))))
    hidden_dim = parse(Int, _argvalue(args, "hidden", _argvalue(args, "hidden_dim", string(_ntget(tradeadvicecfg, :hidden_dim, 32)))))
    maxepoch = parse(Int, _argvalue(args, "maxepoch", isnothing(tradeadvicecfg) ? (retrainrequested ? "200" : "20") : string(tradeadvicecfg.maxepoch)))
    batchsize = parse(Int, _argvalue(args, "batchsize", string(_ntget(tradeadvicecfg, :batchsize, 64))))
    default_opens = join(string.(Float32.(_ntget(tradeadvicecfg, :openthresholds, Float32[0.8f0, 0.7f0, 0.6f0]))), ",")
    default_closes = join(string.(Float32.(_ntget(tradeadvicecfg, :closethresholds, Float32[0.6f0, 0.55f0, 0.5f0]))), ",")
    openthresholds = _parse_float32_list(_argvalue(args, "openthresholds", default_opens))
    closethresholds = _parse_float32_list(_argvalue(args, "closethresholds", default_closes))
    entrytimeout = parse(Int, _argvalue(args, "entrytimeout", string(_ntget(tradeadvicecfg, :entrytimeout, 2))))
    exittimeout = parse(Int, _argvalue(args, "exittimeout", string(_ntget(tradeadvicecfg, :exittimeout, 2))))
    exitstrategy = Symbol(_argvalue(args, "exitstrategy", string(_ntget(tradeadvicecfg, :exitstrategy, :opposite_signal_market))))
    return TradeAdviceLstmConfig(; configname=configname, folder=folder, trendconfig=trendconfig, seqlen=seqlen, hidden_dim=hidden_dim, maxepoch=maxepoch, batchsize=batchsize, openthresholds=openthresholds, closethresholds=closethresholds, entrytimeout=entrytimeout, exittimeout=exittimeout, exitstrategy=exitstrategy, mode=EnvConfig.configmode)
end

println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$ARGS")

retrain = "retrain" in ARGS
trainmode = ("train" in ARGS) || retrain
testmode = !trainmode
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
    println("LSTM params: seqlen=$(cfg.seqlen) hidden_dim=$(cfg.hidden_dim) maxepoch=$(cfg.maxepoch) batchsize=$(cfg.batchsize) entrytimeout=$(cfg.entrytimeout) exittimeout=$(cfg.exittimeout)")
    println("Run mode: $(trainmode ? "training" : "test") retrain=$(retrain)")
    println("Trend source folder: $(trendfolder(cfg))")
    summarydf = EnvConfig.readdf(summaryfilename(); folderpath=_folderpath(cfg.folder))
    if isnothing(summarydf) || (size(summarydf, 1) == 0)
        println("No TradeAdvice summary found yet in $(_folderpath(cfg.folder))")
    else
        _print_df("Saved TradeAdvice summary (transposed):", _transpose_summarydf(summarydf))
    end
elseif compareonly
    comparisondf = collect_tradeadvice_summaries(cfg)
    println("$(EnvConfig.now()) available TradeAdvice summaries: $comparisondf")
else
    run_pipeline(cfg; retrain=retrain)
end

println("$(EnvConfig.now()) done @ $(cfg.folder)")

end # module
