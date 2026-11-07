"""SHA-256 of a required file; missing inputs are never treated as a reusable cache."""
function filedigest(path::AbstractString)::String
    @assert isfile(path) "required snapshot artifact missing: $path"
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
function snapshotprepared(opts::Dict{String,String}, rc::NamedTuple, folder::AbstractString)::NamedTuple
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
function snapshotsources(prepared::NamedTuple;folds::String="",regimes::String="",
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

"""Verify and freeze one coin's reserved rows, without retaining full histories in the snapshot."""
@noinline function sealcoin!(prepared::NamedTuple,sources::NamedTuple,coin::AbstractString,
        coinix::Integer,index::AbstractDataFrame,expectedhistory::AbstractString,
        windows::AbstractVector{Int},folder::AbstractString)::NamedTuple
    history = loadinputhistory(prepared.input,coin)
    mktempdir(folder) do scratch
        digest = TradingStrategy._experimentdigest!(scratch, _ -> history, coin)
        @assert digest == expectedhistory "history for $coin changed since adaptation: stored=$expectedhistory current=$digest; cannot seal original experiment"
    end
    storesfolder = joinpath(prepared.folder,"stores",coin)
    features = readpreparedtable(storesfolder,"features")
    targets = readpreparedtable(storesfolder,"targets")
    sets = readpreparedtable(storesfolder,"sets")
    expected = TradingStrategy.experimentindex((coins=[coin],featurestores=[features],
        targetstores=[targets]),Dict(coin=>sets),sources.labels;settype="test",
        folds=sources.folds,regimes=sources.regimes)
    expected.coinix .= coinix
    @assert isequal(expected,DataFrame(index)) "reserved test membership for $coin changed since preparation"
    rows = index[!, :row]
    table = DataFrame(index)
    for column in sources.cols
        table[!,column] = features[rows,column]
    end
    historyrows = searchsortedfirst.(Ref(history.opentime),index.opentime)
    @assert all(ix -> historyrows[ix] <= nrow(history) && history.opentime[historyrows[ix]] == index.opentime[ix],
        eachindex(historyrows)) "snapshot price timestamps missing for $coin"
    for column in (:high,:low,:close)
        table[!,column] = history[historyrows,column]
    end
    if !isempty(windows)
        ohlcv = Ohlcv.defaultohlcv(coin)
        Ohlcv.setdataframe!(ohlcv,history)
        guidance = Features.Features006()
        for window in windows
            Features.addgrad!(guidance;window)
        end
        Features.setbase!(guidance,ohlcv;usecache=true)
        for window in windows
            aligned = TradingStrategy._alignregression(
                TradingStrategy._regressioncolumns(guidance,window),index.opentime)
            table[!,"regry$window"] = aligned.price
            table[!,"grad$window"] = aligned.gradient
        end
    end
    path = atomictable(table,coin,folder)
    return (coin=String(coin),rows=nrow(table),digest=filedigest(path))
end

"""Seal persisted test inputs once; existing seals must match instead of being silently replaced."""
function sealsnapshot!(prepared::NamedTuple,sources::NamedTuple)::Nothing
    folder = mkpath(joinpath(sources.folder,"gain-snapshot"))
    @assert !isfile(joinpath(folder,"manifest.arrow")) "snapshot already sealed in $folder; gain verifies/reuses it; do not overwrite an immutable snapshot"
    index = readpreparedtable(sources.folder,"test-index")
    @assert issorted(index.coinix) "saved test index must be grouped by coin; rows=$(nrow(index))"
    groups = groupby(index,:coin;sort=false)
    histories = readpreparedtable(prepared.folder,"input-identities")
    historydigests = Dict(zip(histories.coin,histories.digest))
    strategy = TradingStrategy.strategytemplate(prepared.input.rc.tscfg.strategy)
    windows = TradingStrategy._strategyregressionwindows(strategy.algorithmconfig)
    records = DataFrame(coin=String[],rows=Int[],digest=String[])
    started = time_ns()
    for (groupix,group) in enumerate(groups)
        coinix = first(group.coinix)
        coin = prepared.stores.coins[coinix]
        @assert prepared.stores.coins[coinix] == coin "saved coin coordinate=$coinix does not identify $coin"
        progressmessage("sealing test snapshot $coin ($groupix/$(length(groups))) rows=$(nrow(group))";started)
        push!(records,sealcoin!(prepared,sources,coin,coinix,group,historydigests[coin],windows,folder))
        GC.gc()
    end
    @assert sum(records.rows) == nrow(index) "snapshot rows=$(sum(records.rows)) differ from test rows=$(nrow(index))"
    testidentity = TradingStrategy.experimentmanifest!(joinpath(sources.folder,"test-identity"),
        Dict("test"=>index),sources.identity)
    atomictable(records,"coins",folder)
    manifest = DataFrame(identity=[sources.identity],preparation=[prepared.identity],
        description=[preparationdescription(prepared.input)],windows=[join(windows,",")],
        features=[join(sources.cols,",")],labels=[join(Targets.tradelabelcode.(sources.labels),",")],
        coinsdigest=[filedigest(joinpath(folder,"coins.arrow"))],testidentity=[testidentity])
    atomictable(manifest,"manifest",folder)
    progressmessage("sealed snapshot: coins=$(nrow(records)) samples=$(sum(records.rows)) output=$folder";started)
    return nothing
end

"""Verify the snapshot contract; per-coin content digests are checked immediately before use."""
function loadsnapshot(prepared::NamedTuple,sources::NamedTuple)::NamedTuple
    folder = joinpath(sources.folder,"gain-snapshot")
    @assert isfile(joinpath(folder,"manifest.arrow")) "no sealed gain snapshot in $folder; run the same command with stage=snapshot once (verifies original histories)"
    manifest = readpreparedtable(folder,"manifest")
    @assert only(manifest.identity) == sources.identity && only(manifest.preparation) == prepared.identity "snapshot identity does not match saved experiment=$(sources.identity)"
    @assert only(manifest.description) == preparationdescription(prepared.input) "snapshot configuration changed: $(only(manifest.description))"
    @assert only(manifest.features) == join(sources.cols,",") && only(manifest.labels) == join(Targets.tradelabelcode.(sources.labels),",") "snapshot feature/label order differs from requested model"
    @assert filedigest(joinpath(folder,"coins.arrow")) == only(manifest.coinsdigest) "snapshot coin manifest changed in $folder"
    records = readpreparedtable(folder,"coins")
    @assert nrow(records)>0 && allunique(records.coin) && issorted(records.coin) && all(>(0),records.rows) "invalid snapshot coin records: rows=$(nrow(records))"
    @assert all(in(prepared.stores.coins),records.coin) "snapshot coins outside the original universe=$(prepared.stores.coins)"
    required = TradingStrategy._strategyregressionwindows(
        TradingStrategy.strategytemplate(prepared.input.rc.tscfg.strategy).algorithmconfig)
    windows = isempty(only(manifest.windows)) ? Int[] : parse.(Int,split(only(manifest.windows),","))
    @assert all(in(windows),required) "strategy regression windows=$required unavailable in sealed snapshot windows=$windows"
    return (;folder,records,windows=required,testidentity=only(manifest.testidentity),
        digest=filedigest(joinpath(folder,"manifest.arrow")))
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

"""Infer one frozen coin in bounded batches, without creating training/evaluation/test mmap sources."""
function snapshotpredictions(best,table::DataFrame,cols::AbstractVector{<:AbstractString};
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
function snapshotreplay!(rows::DataFrame,strategy::TradingStrategy.StrategyConfig,
        windows::AbstractVector{Int},predicted::Bool,folder::AbstractString,stem::AbstractString)::DataFrame
    @assert nrow(rows)>0 "snapshot replay requires rows; stem=$stem rows=$(nrow(rows))"
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

"""Stream a verified saved test snapshot; reuse predictions and replays only with committed identities."""
function snapshotgainreport!(prepared::NamedTuple,sources::NamedTuple)::Nothing
    snapshot = loadsnapshot(prepared,sources)
    best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
    @assert best.labels == sources.labels "checkpoint labels=$(best.labels) differ from snapshot labels=$(sources.labels)"
    checkpoint = filedigest(joinpath(sources.folder,"eval-best.bson"))
    strategy = TradingStrategy.strategytemplate(prepared.input.rc.tscfg.strategy)
    predictionidentity = TradingStrategy.confighash("gain-prediction-v1;$(snapshot.digest);$checkpoint")
    replayidentity = TradingStrategy.confighash("gain-replay-v1;$predictionidentity;$(repr(strategy))")
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
    recordteststrategyidentity!(sources,snapshot.testidentity,strategy)
    predictionpending = "test-predictions-pending"
    gainparts = DataFrame[]
    started = time_ns()
    completedrows = 0
    totalrows = sum(snapshot.records.rows)
    for (coinix,record) in enumerate(eachrow(snapshot.records))
        coin = record.coin
        progressmessage("gain coin=$coin ($coinix/$(nrow(snapshot.records))) samples=$completedrows/$totalrows rss=$(Sys.maxrss())";started)
        path = joinpath(snapshot.folder,coin*".arrow")
        @assert filedigest(path) == record.digest "sealed snapshot content changed for coin=$coin"
        table = readpreparedtable(snapshot.folder,coin)
        @assert nrow(table) == record.rows && all(==(coin),table.coin) && all(==("test"),table.set) "invalid frozen population for $coin rows=$(nrow(table))"
        if verifiedcache(cachefolder,coin,predictionidentity)
            progressmessage("reusing verified predictions: $coin";started)
            predictions = readpreparedtable(cachefolder,coin)
        else
            progressmessage("inferring frozen test rows: $coin";started)
            predictions = snapshotpredictions(best,table,sources.cols;
                onprogress=(done,total)->(done==total && progressmessage("inferred $coin samples=$done/$total";started)))
            commitcache(predictions,cachefolder,coin,predictionidentity)
        end
        EnvConfig.writetable(nativegaintable(predictions),predictionpending;folderpath=sources.folder,
            format=:arrow,append=coinix>1,preserveschema=true)
        rows = select(table,:coin,:opentime,:set,:regime,:foldid,:blockid,:high,:low,:close,
            [Symbol("regry$w") for w in snapshot.windows]...,
            [Symbol("grad$w") for w in snapshot.windows]...)
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
                measured = @timed snapshotreplay!(rows,strategy,snapshot.windows,predicted,replayfolder,stem)
                gains = measured.value
                progressmessage("replayed $stem seconds=$(round(measured.time;digits=2)) allocated_bytes=$(measured.bytes) gc_seconds=$(round(measured.gctime;digits=2)) rows_per_second=$(round(nrow(rows)/measured.time))";started)
                gains[!,:predicted] = fill(predicted,nrow(gains))
                commitcache(gains,replayfolder,stem*"-gains",replayidentity)
                atomictable(DataFrame(identity=[replayidentity],
                    digest=[filedigest(joinpath(replayfolder,stem*".arrow"))]),stem*"-complete",replayfolder)
            end
            cp(joinpath(replayfolder,stem*".arrow"),joinpath(tradesfolder,stem*".arrow");force=true)
            nrow(gains)>0 && push!(gainparts,gains)
        end
        completedrows += nrow(table)
    end
    mv(joinpath(sources.folder,predictionpending*".arrow"),
        joinpath(sources.folder,"test-predictions.arrow");force=true)
    if isempty(gainparts)
        @warn "gain replay produced no completed trades" tradesfolder
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
        snapshot=[snapshot.digest],replay=[replayidentity]),"gain-identity",gainfolder)
    progressmessage("gain complete: samples=$completedrows peak_rss=$(Sys.maxrss()) reports=$(joinpath(gainfolder,"gain-report.txt")), $(joinpath(gainfolder,"gain-report.svg"))";started)
    return nothing
end
