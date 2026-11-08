"""SHA-256 of a required file; missing inputs are never treated as a reusable cache."""
function filedigest(path::AbstractString)::String
    @assert isfile(path) "required gain artifact missing: $path"
    return bytes2hex(open(SHA.sha256,path))
end

"""Publish a table by atomic rename, keeping incomplete writes out of the committed path."""
function atomictable(table::AbstractDataFrame, stem::AbstractString, folder::AbstractString)::String
    mkpath(folder)
    path = EnvConfig.writetable(nativegaintable(table),stem*"-pending";
        folderpath=folder,format=:arrow,preserveschema=true)
    destination = joinpath(folder,stem*".arrow")
    mv(path,destination;force=true)
    return destination
end

"""Preserve native batch schemas, representing enum/categorical text without independent dictionaries."""
function nativegaintable(table::AbstractDataFrame)::DataFrame
    result = DataFrame(table;copycols=false)
    for name in names(result)
        column = result[!,name]
        kind = Base.nonmissingtype(eltype(column))
        if kind <: Enum || kind <: EnvConfig.CategoricalArrays.CategoricalValue
            result[!,name] = EnvConfig._arrow_stringify.(column)
        end
    end
    return result
end

"""Reconstruct configuration metadata without reading mutable market inputs."""
function gainprepared(opts::Dict{String,String}, rc::NamedTuple, folder::AbstractString)::NamedTuple
    input = inputs(opts,rc;saved=true)
    manifest = readpreparedtable(folder,"manifest")
    if !haskey(opts,"coins")
        storedcoins = match(r";coins=([^;]+);start=",only(manifest.description))
        @assert !isnothing(storedcoins) "saved experiment has no recorded universe: $(only(manifest.description))"
        coins = collect(split(storedcoins.captures[1],","))
        context = replace(input.context,r";coins=[^;]+;start="=>";coins=$(join(coins,","));start=";count=1)
        input = merge(input,(coins=coins,context=context,rc=merge(rc,(coins=coins,description=context))))
    end
    expected = preparationdescription(input)
    @assert only(manifest.description) == expected "saved experiment configuration differs: stored=$(only(manifest.description)), requested=$expected"
    identities = readpreparedtable(folder,"input-identities")
    @assert all(in(input.coins),identities.coin) "saved universe=$(identities.coin) is outside requested universe=$(input.coins)"
    coins = sort(filter(in(identities.coin),input.coins))
    @assert !isempty(coins) && allunique(coins) "invalid saved universe=$coins"
    input = merge(input,(coins=coins,histories=Dict{String,DataFrame}()))
    return (;input,identity=only(manifest.identity),folder,
        stores=(coins=coins,featurestores=DataFrame[],targetstores=DataFrame[]))
end

"""Verify the original training selection without rebuilding training/evaluation numeric sources."""
function gainsources(prepared::NamedTuple;folds::String="",regimes::String="",
        trainweights::String="class",folder::AbstractString=joinpath(prepared.folder,"balanced"))::NamedTuple
    foldids = isempty(folds) ? Int[] : parse.(Int,split(folds,","))
    regimetypes = isempty(regimes) ? String[] : collect(split(regimes,","))
    @assert trainweights in ("class","balanced") "invalid training weights=$trainweights"
    @assert all(in(["up","flat","down"]),regimetypes) "invalid selected regimes=$regimetypes"
    description = "$(prepared.identity);training=$trainweights;folds=$foldids;regimes=$regimetypes;eval=equal-cells-within-regimes-v1"
    manifest = readpreparedtable(folder,"manifest")
    @assert only(manifest.description) == description "saved training selection differs: stored=$(only(manifest.description)), requested=$description"
    labels = Targets.uniquelabels(prepared.input.targetfactory())
    return (;folder,identity=only(manifest.identity),labels,
        cols=Features.requestedcolumns(prepared.input.featurefactory()),folds=foldids,regimes=regimetypes)
end

"""Load and align one coin's reserved test rows from the experiment and shared stores."""
@noinline function gaincoin(prepared::NamedTuple,sources::NamedTuple,coin::AbstractString,
        coinix::Integer,index::AbstractDataFrame,windows::AbstractVector{Int})::DataFrame
    @assert nrow(index)>0 && all(==(coinix),index.coinix) && all(==(coin),index.coin) &&
        all(==("test"),index.set) "invalid test index for coin=$coin coinix=$coinix rows=$(nrow(index))"
    @assert issorted(index.opentime) && allunique(index.opentime) "test timestamps for $coin must be sorted and unique: rows=$(nrow(index))"
    @assert all(ix -> 1<=ix<=length(sources.labels),index.labelix) "invalid test labels for $coin: labels=$(length(sources.labels))"
    if prepared.input.mode=="market"
        clcfg = prepared.input.rc.clcfg
        features = Features.readfeatures(coin,"USDT",clcfg.features;
            confighash=TradingStrategy.featureshash(clcfg.features))
        @assert !isnothing(features) "missing shared features for $coin features=$(clcfg.features)"
    else
        storesfolder = joinpath(prepared.folder,"stores",coin)
        features = readpreparedtable(storesfolder,"features")
    end
    rows = index[!, :row]
    @assert all(ix -> 1<=rows[ix]<=nrow(features) && features.opentime[rows[ix]]==index.opentime[ix],
        eachindex(rows)) "test feature coordinates no longer match reserved timestamps for $coin"
    table = nativegaintable(DataFrame(index))
    for column in sources.cols
        table[!,column] = features[rows,column]
    end
    history = loadinputhistory(prepared.input,coin)
    historyrows = searchsortedfirst.(Ref(history.opentime),index.opentime)
    @assert all(ix -> historyrows[ix] <= nrow(history) && history.opentime[historyrows[ix]] == index.opentime[ix],
        eachindex(historyrows)) "test price timestamps missing for $coin"
    for column in (:high,:low,:close)
        table[!,column] = history[historyrows,column]
    end
    if !isempty(windows) && prepared.input.mode!="market"
        ohlcv = Ohlcv.defaultohlcv(coin)
        Ohlcv.setdataframe!(ohlcv,history)
        guidance = Features.Features006()
        for window in windows
            Features.addgrad!(guidance;window)
        end
        Features.setbase!(guidance,ohlcv;usecache=false)
        for window in windows
            aligned = TradingStrategy._alignregression(
                TradingStrategy._regressioncolumns(guidance,window),index.opentime)
            table[!,"regry$window"] = aligned.price
            table[!,"grad$window"] = aligned.gradient
        end
    elseif !isempty(windows)
        for window in windows
            regression = Features.readregression(coin,"USDT",window)
            @assert !isnothing(regression) "missing precalculated regression for coin=$coin window=$window"
            columns = TradingStrategy.RegressionColumns(regression.opentime,regression.regry,regression.grad)
            aligned = TradingStrategy._alignregression(columns,index.opentime)
            table[!,"regry$window"] = aligned.price
            table[!,"grad$window"] = aligned.gradient
        end
    end
    return table
end

"""Load a complete cache only if its committed content and execution identity match."""
function verifiedcache(folder::AbstractString,stem::AbstractString,identity::AbstractString)::Bool
    path = joinpath(folder,stem*"-complete.arrow")
    if !isfile(path)
        return false
    end
    marker = readpreparedtable(folder,stem*"-complete")
    @assert only(marker.identity) == identity "cache execution identity changed for $folder/$stem"
    @assert only(marker.digest) == filedigest(joinpath(folder,stem*".arrow")) "committed cache content changed for $folder/$stem"
    return true
end

"""Commit cache content before its identity/digest completion marker."""
function commitcache(table::AbstractDataFrame,folder::AbstractString,stem::AbstractString,
        identity::AbstractString)::Nothing
    path = atomictable(table,stem,folder)
    atomictable(DataFrame(identity=[identity],digest=[filedigest(path)]),stem*"-complete",folder)
    return nothing
end

"""Infer one coin's test rows in bounded batches, without compacting training sources."""
function gainpredictions(best,table::DataFrame,cols::AbstractVector{<:AbstractString};
        onprogress::Function)::DataFrame
    result = select(table,:coin,:opentime,:set,:regime,:foldid,:blockid)
    scores = Vector{Float32}(undef,nrow(table))
    labels = Vector{eltype(best.labels)}(undef,nrow(table))
    Classify.Flux.testmode!(best.model)
    for chunk in Iterators.partition(1:nrow(table),8192)
        features = permutedims(Matrix(table[chunk,cols]))
        chunkscores,chunklabels = Classify.maxpredict(best,features)
        scores[chunk] .= chunkscores
        labels[chunk] .= chunklabels
        onprogress(last(chunk),nrow(table))
    end
    result[!,:label] = labels
    result[!,:score] = scores
    result[!,:target] = best.labels[table.labelix]
    return result
end

"""Replay one coin/pass with canonical Trades schemas and append independent range snapshots."""
function gainreplay!(rows::DataFrame,strategy::TradingStrategy.StrategyConfig,
        windows::AbstractVector{Int},predicted::Bool,folder::AbstractString,stem::AbstractString)::DataFrame
    @assert nrow(rows)>0 "gain replay requires rows; stem=$stem rows=$(nrow(rows))"
    ranges = cumsum(vcat(1,[rows.blockid[ix]!=rows.blockid[ix-1] ||
        rows.opentime[ix]-rows.opentime[ix-1]!=Minute(1) for ix in 2:nrow(rows)]))
    rows[!,:rangeid] = ranges
    snapshots = [:opentime,:pair,:set,:rangeid,:lastopentrade,:close,:high,:low,:label,:score,
        :lp_amount,:sp_amount,:lol_pavg,:lcl_pavg,:sol_pavg,:scl_pavg,:lo_limit,:lc_limit,:so_limit,:sc_limit]
    gains = DataFrame[]
    emptygains = DataFrame()
    pending = stem*"-pending"
    for (groupix,group) in enumerate(groupby(rows,[:set,:rangeid];sort=false))
        trades = select(group,:opentime,:set,:regime,:foldid,:blockid,:score,:label,
            :high,:low,:close,:rangeid)
        trades[!,:pair] = fill(TradingStrategy.tspairkey(first(rows.coin)),nrow(trades))
        trades[!,:score] = predicted ? collect(group.score) : fill(1f0,nrow(group))
        trades[!,:label] = collect(predicted ? group.label : group.target)
        for contributor in TSM.tradesdf_all_contributors()
            contributor(trades)
        end
        trades.freequote .= strategy.maxbudgetquote
        trades.freemargin .= strategy.maxbudgetquote
        tp = TradingStrategy.TsTp(pair=first(trades.pair),tradesdf=trades)
        regressions = Dict(window=>DataFrame(opentime=group.opentime,
            regry=group[!,"regry$window"],grad=group[!,"grad$window"];copycols=false) for window in windows)
        TradingStrategy.processreplaygains!(tp;strategy,regressions)
        snapshot = select(trades,snapshots)
        EnvConfig.writetable(nativegaintable(snapshot),pending;folderpath=folder,format=:arrow,
            append=groupix>1,preserveschema=true)
        completed = TSM.compilegains(snapshot;setpartitions=true)
        emptygains = completed
        nrow(completed)>0 && push!(gains,completed)
    end
    mv(joinpath(folder,pending*".arrow"),joinpath(folder,stem*".arrow");force=true)
    return isempty(gains) ? emptygains : TSM.sortgainsdf!(vcat(gains...))
end

"""
Write gain-report.txt and gain-report.svg, separating predicted and target-reference trades.
The chart shows average completed-trade gain in percent, not compounded portfolio return.
Empty results replace previous reports with an explicit no-completed-trades message.
"""
function gainreadablereport!(gains::AbstractDataFrame,folder::AbstractString)::Nothing
    report = TSM.gainsreport(gains;groupcols=[:set,:predicted])
    @assert all(isfinite,report.avggain) "non-finite average gains in report: $(report.avggain)"
    mkpath(folder)
    open(joinpath(folder,"gain-report.txt"),"w") do io
        println(io,"Reserved test: strategy gains")
        println(io,"predicted=true: classifier predictions; predicted=false: target-label reference.")
        println(io,"avggain is a fractional average per completed trade, not portfolio return.")
        if isempty(report)
            println(io,"No completed trades.")
        else
            show(io,MIME("text/plain"),report;allrows=true,allcols=true)
            println(io)
        end
    end
    height = max(180,120+70nrow(report))
    limit = max(1e-6,maximum(abs,report.avggain;init=0.0)*100)
    open(joinpath(folder,"gain-report.svg"),"w") do io
        println(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"$height\" viewBox=\"0 0 1000 $height\">")
        println(io,"<style>text{font-family:sans-serif;font-size:14px}</style><rect width=\"100%\" height=\"100%\" fill=\"white\"/>")
        println(io,"<text x=\"20\" y=\"25\">Reserved test: average gain per completed trade (%)</text>")
        println(io,"<text x=\"20\" y=\"50\">Not portfolio return; target reference uses known target labels.</text>")
        if isempty(report)
            println(io,"<text x=\"20\" y=\"100\">No completed trades.</text>")
        else
            println(io,"<path d=\"M 600 75 V $(height-30)\" stroke=\"#777\"/><text x=\"600\" y=\"70\" text-anchor=\"middle\">0%</text>")
            for (ix,row) in enumerate(eachrow(report))
                label = "$(row.set): $(row.predicted ? "Predicted" : "Target reference")"
                label = replace(label,"&"=>"&amp;","<"=>"&lt;",">"=>"&gt;","\""=>"&quot;","'"=>"&apos;")
                gainpercent = 100row.avggain
                width = 260abs(gainpercent)/limit
                left = gainpercent < 0 ? 600-width : 600
                top = 85+70(ix-1)
                color = row.predicted ? "#267d36" : "#376bb5"
                println(io,"<text x=\"20\" y=\"$(top+20)\">$label</text>")
                println(io,"<rect x=\"$left\" y=\"$top\" width=\"$width\" height=\"28\" fill=\"$color\"><title>$label: $(round(gainpercent;digits=6))%; completed trades=$(row.segments)</title></rect>")
                println(io,"<text x=\"$(gainpercent < 0 ? left-8 : left+width+8)\" y=\"$(top+20)\" text-anchor=\"$(gainpercent < 0 ? "end" : "start")\">$(round(gainpercent;digits=4))%</text>")
                println(io,"<text x=\"20\" y=\"$(top+43)\">Completed trades: $(row.segments)</text>")
            end
        end
        println(io,"</svg>")
    end
    return nothing
end

"""Stream saved test selections from the stores; reuse committed predictions and replays."""
function directgainreport!(prepared::NamedTuple,sources::NamedTuple)::Nothing
    index = readpreparedtable(sources.folder,"test-index")
    @assert nrow(index)>0 && issorted(index.coinix) "saved test index must be nonempty and grouped by coin; rows=$(nrow(index))"
    groups = groupby(index,:coin;sort=false)
    testidentity = TradingStrategy.experimentmanifest!(joinpath(sources.folder,"test-identity"),
        Dict("test"=>index),sources.identity)
    indexdigest = filedigest(joinpath(sources.folder,"test-index.arrow"))
    best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
    @assert best.labels == sources.labels "checkpoint labels=$(best.labels) differ from test labels=$(sources.labels)"
    checkpoint = filedigest(joinpath(sources.folder,"eval-best.bson"))
    strategy = TradingStrategy.strategytemplate(prepared.input.rc.tscfg.strategy)
    windows = TradingStrategy._strategyregressionwindows(strategy.algorithmconfig)
    predictionidentity = TradingStrategy.confighash("gain-prediction-v2;$(sources.identity);$indexdigest;$checkpoint")
    replayidentity = TradingStrategy.confighash("gain-replay-v2;$predictionidentity;$(repr(strategy))")
    rc = prepared.input.rc
    if rc.mode == "market"
        EnvConfig.setlogpath(rc.folder)
        gainfolder = EnvConfig.logfolder()
    else
        gainfolder = mkpath(joinpath(prepared.folder,"gain",rc.folder))
    end
    tradesfolder = mkpath(joinpath(gainfolder,"trades-regime"))
    cachefolder = mkpath(joinpath(sources.folder,"gain-cache",predictionidentity))
    replayfolder = mkpath(joinpath(cachefolder,replayidentity))
    recordteststrategyidentity!(sources,testidentity,strategy)
    predictionpending = "test-predictions-pending"
    gainparts = DataFrame[]
    emptygains = DataFrame()
    started = time_ns()
    completedrows = 0
    totalrows = nrow(index)
    for (groupix,group) in enumerate(groups)
        coinix = first(group.coinix)
        @assert 1<=coinix<=length(prepared.stores.coins) "invalid test coin coordinate=$coinix coins=$(length(prepared.stores.coins))"
        coin = prepared.stores.coins[coinix]
        progressmessage("gain coin=$coin ($groupix/$(length(groups))) samples=$completedrows/$totalrows rss=$(Sys.maxrss())";started)
        table = gaincoin(prepared,sources,coin,coinix,group,windows)
        if verifiedcache(cachefolder,coin,predictionidentity)
            progressmessage("reusing verified predictions: $coin";started)
            predictions = readpreparedtable(cachefolder,coin)
        else
            progressmessage("inferring test rows: $coin";started)
            predictions = gainpredictions(best,table,sources.cols;
                onprogress=(done,total)->(done==total && progressmessage("inferred $coin samples=$done/$total";started)))
            commitcache(predictions,cachefolder,coin,predictionidentity)
        end
        EnvConfig.writetable(nativegaintable(predictions),predictionpending;folderpath=sources.folder,
            format=:arrow,append=groupix>1,preserveschema=true)
        rows = select(table,:coin,:opentime,:set,:regime,:foldid,:blockid,:high,:low,:close,
            [Symbol("regry$w") for w in windows]...,
            [Symbol("grad$w") for w in windows]...)
        rows[!,:label] = predictions.label
        rows[!,:score] = predictions.score
        rows[!,:target] = predictions.target
        for (pass,predicted) in (("predicted",true),("truth",false))
            stem = "$coin-$pass"
            if verifiedcache(replayfolder,stem,replayidentity)
                @assert verifiedcache(replayfolder,stem*"-gains",replayidentity) "missing completed gains for cached replay=$stem"
                progressmessage("reusing verified replay: $stem";started)
                gains = readpreparedtable(replayfolder,stem*"-gains")
            else
                progressmessage("replaying $stem rows=$(nrow(rows))";started)
                measured = @timed gainreplay!(rows,strategy,windows,predicted,replayfolder,stem)
                gains = measured.value
                progressmessage("replayed $stem seconds=$(round(measured.time;digits=2)) allocated_bytes=$(measured.bytes) gc_seconds=$(round(measured.gctime;digits=2)) rows_per_second=$(round(nrow(rows)/measured.time))";started)
                gains[!,:predicted] = fill(predicted,nrow(gains))
                commitcache(gains,replayfolder,stem*"-gains",replayidentity)
                atomictable(DataFrame(identity=[replayidentity],
                    digest=[filedigest(joinpath(replayfolder,stem*".arrow"))]),stem*"-complete",replayfolder)
            end
            cp(joinpath(replayfolder,stem*".arrow"),joinpath(tradesfolder,stem*".arrow");force=true)
            emptygains = gains
            nrow(gains)>0 && push!(gainparts,gains)
        end
        completedrows += nrow(table)
        GC.gc()
    end
    mv(joinpath(sources.folder,predictionpending*".arrow"),
        joinpath(sources.folder,"test-predictions.arrow");force=true)
    if isempty(gainparts)
        @warn "gain replay produced no completed trades" tradesfolder
        gains = emptygains
        EnvConfig.savedf(gains,"tsmgains-regime";folderpath=gainfolder)
        TSM.gainsreport(instem="tsmgains-regime",stem="xchgainsreport-regime",folderpath=gainfolder)
        atomictable(DataFrame(completed_trades=[0]),"gain-summary",gainfolder)
    else
        gains = TSM.sortgainsdf!(vcat(gainparts...))
        EnvConfig.savedf(gains,"tsmgains-regime";folderpath=gainfolder)
        TSM.gainsreport(instem="tsmgains-regime",stem="xchgainsreport-regime",folderpath=gainfolder)
        atomictable(DataFrame(completed_trades=[nrow(gains)]),"gain-summary",gainfolder)
    end
    gainreadablereport!(gains,gainfolder)
    atomictable(DataFrame(identity=[sources.identity],ts=[rc.tscfg.configname],
        strategy=[rc.tscfg.strategy],population=["test"],checkpoint=["eval-best.bson"],
        testindex=[indexdigest],replay=[replayidentity]),"gain-identity",gainfolder)
    progressmessage("gain complete: samples=$completedrows peak_rss=$(Sys.maxrss()) reports=$(joinpath(gainfolder,"gain-report.txt")), $(joinpath(gainfolder,"gain-report.svg"))";started)
    return nothing
end
