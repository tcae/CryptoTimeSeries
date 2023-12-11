"""
Train and evaluate the trading signal classifiers
"""
module Classify

using DataFrames, Logging, Dates, PrettyPrinting
using BSON, JDF, Flux, Statistics, ProgressMeter, StatisticalMeasures, MLUtils
using CategoricalArrays
using CategoricalDistributions
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv, CryptoXch
using Plots

mutable struct AdaptSet
    targets
    predictions
    setranges
    symbol  #pair of base and quote
    startdt
    enddt
    function AdaptSet(targets, pred, setranges, f3)
        sym = Ohlcv.basesymbol(f3.f2.ohlcv) * Ohlcv.quotesymbol(f3.f2.ohlcv)
        startdt = Ohlcv.dataframe(f3.f2.ohlcv).opentime[Features.ohlcvix(f3, 1)]
        enddt = Ohlcv.dataframe(f3.f2.ohlcv).opentime[end]
        return new(targets, pred, setranges, sym, startdt, enddt)
    end
end

mutable struct AdaptContext
    featuresdecription
    targetsdecription
    losses
    adaptsets
end

function Base.show(io::IO, ac::AdaptContext)
    println(io, "NN AdaptContext: featuresdecription=$(ac.featuresdecription) targetsdecription=$(ac.targetsdecription) #adaptsets=$(length(ac.adaptsets)) len(losses)=$(length(ac.losses))")
    for ads in ac.adaptsets
        println(io, "symbol=$(ads.symbol) start=$(ads.startdt) end=$(ads.enddt) #targets=$(size(ads.targets)) #predictions=$(size(ads.predictions)) ")
        for (s,v) in ads.setranges
            println(io,"$s: length=$(length(v))")
        end
    end
end

mutable struct NN
    model
    optim
    lossfunc
    labels::Vector{String}  # in fixed sequence as index == class id
    description
    mnemonic::String
    fileprefix::String  # filename without suffix
    predecessors::Vector{String}  # filename vector without suffix
    featuresdescription::String
    targetsdescription::String
    losses::Vector
    predictions::Vector{String}  # filename vector without suffix of predictions
end

function NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix)
    return NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix, String[], "", "", [], String[])
end

function Base.show(io::IO, nn::NN)
    println(io, "NN: labels=$(nn.labels) #predecessors=$(length(nn.predecessors)) mnemonic=$(nn.mnemonic) fileprefix=$(nn.fileprefix) #losses=$(length(nn.losses))")
    println(io, "NN: featuresdecription=$(nn.featuresdescription) targetsdecription=$(nn.targetsdescription)")
    println(io, "NN: predecessors=$(nn.predecessors)")
    println(io, "NN: predictions=$(nn.predictions)")
    println(io, "NN: description=$(nn.description)")
end

"""
Returns the trade performance percentage of trade sigals given in `signals` applied to `prices`.
"""
function tradeperformance(prices, signals)
    fee = 0.002  # 0.2% fee for each trade
    initialcash = cash = 100.0
    startprice = 1.0
    asset = 0.0

    for ix in eachindex(prices)  # 1:size(prices, 1)
        if (signals[ix] == "long") && (cash > 0)
                asset = cash / prices[ix] * (1 - fee)
                cash = 0.0
        elseif (asset > 0) && ((signals[ix] == "close") || (signals[ix] == "short"))
                cash = asset * prices[ix] * (1 - fee)
                asset = 0.0
        # elseif enableshort && (signals[ix] == "short") && (cash > 0)
            # to be done
        end
    end
    if asset > 0
        cash = asset * prices[end] * (1 - fee)
    end
    return (cash - initialcash) / initialcash * 100
end



#region DataPrep
# folds(data, nfolds) = partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...)

"""
    - Returns a Dict of setname => vector of row ranges
    - input
        - `rowrange` is the range of rows that are split in subranges, e.g. 2000:5000
        - `samplesets` is a Dict of setname => relative setsize with sum of all setsizes == 1
        - `gapsize` is the number of rows between partitions of different sets that are not included in any partition
        - `relativesubrangesize` is the subrange size relative to `rowrange` of a consecutive range assigned to a setinterval
        - gaps will be removed from a subrange to avoid crosstalk bias between ranges
        - relativesubrangesize * length(rowrange) > 2 * gapsize
        - any relative setsize > 2 * relativesubrangesize
        - a mixture of ranges and individual indices in a vector can be unpacked into a index vector via `[ix for r in rr for ix in r]`
"""
function setpartitions(rowrange, samplesets::Dict, gapsize, relativesubrangesize)
    rowstart = rowrange[1]
    rowend = rowrange[end]
    rows = rowend - rowstart + 1
    gapsize = relativesubrangesize * rows > 2 * gapsize ? gapsize : floor(Int, relativesubrangesize * rows / 3)
    println("$(EnvConfig.now()) setpartitions rowrange=$rowrange, samplesets=$samplesets gapsize=$gapsize relativesubrangesize=$relativesubrangesize")
    # @assert relativesubrangesize * rows > 2 * gapsize
    @assert max(collect(values(samplesets))...) > 2 * relativesubrangesize "max(collect(values(samplesets))...)=$(max(collect(values(samplesets))...)) <= 2 * relativesubrangesize = $(2 * relativesubrangesize)"
    minrelsetsize = min(collect(values(samplesets))...)
    @assert minrelsetsize > 0.0
    sn = [setname for setname in keys(samplesets)]
    snl = length(sn)
    ix = rowstart
    aix = 1
    arr =[[] for _ in sn]
    while ix <= rowend
        subrangesize = round(Int, rows * relativesubrangesize * samplesets[sn[aix]] / minrelsetsize)
        push!(arr[aix], ((ix == rowstart ? ix : ix + gapsize), min(rowend, ix+subrangesize-1)))
        ix = ix + subrangesize
        aix = aix % snl + 1
    end
    res = Dict(sn[aix] => [t[1]:t[2]; for t in arr[aix]] for aix in eachindex(arr))
    return res
end

function test_setpartitions()
    # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 0, 1/9)
    # res = setpartitions(1:26, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/3), 1, 1/9)
    res = setpartitions(1:49, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 1, 3/50)
    # 3-element Vector{Vector{Any}}:
    #     [UnitRange{Int64}[1:3, 10:12, 19:21]]
    #     [UnitRange{Int64}[4:6, 13:15, 22:24]]
    #     [UnitRange{Int64}[7:9, 16:18, 25:26]]

    for (k, vec) in res
        # for rangeset in vec  # rangeset not necessary
            for range in vec  # rangeset
                for ix in range
                    # println("$k: $ix")
                end
            end
        # end
    end
    println(res)
end

function basecombitestpartitions(rowcount, micropartions)
    micropartionsize = Int(ceil(rowcount/micropartions, digits=0))
    ix = 1
    aix = 1
    arr =[[],[],[]]
    while ix <= rowcount
        push!(arr[aix], (ix, min(rowcount, ix+micropartionsize-1)))
        ix = ix + micropartionsize
        aix = aix % 3 + 1
    end
    res =[[],[],[]]
    for aix in 1:3
        res[aix] = [[t[1]:t[2]; for t in arr[aix]]] # with rangeset
        res[aix] = [t[1]:t[2]; for t in arr[aix]] # without rangeset
    end
    return res
end

function test_basecombitestpartitions()
    res = basecombitestpartitions(26, 9)
    # 3-element Vector{Vector{Any}}:
    #     [UnitRange{Int64}[1:3, 10:12, 19:21]]
    #     [UnitRange{Int64}[4:6, 13:15, 22:24]]
    #     [UnitRange{Int64}[7:9, 16:18, 25:26]]

    for vec in res
        # for rangeset in vec  # rangeset not necessary
            for range in vec  # rangeset
                for ix in range
                    println(ix)
                end
            end
        # end
    end
    println(res)
end

function predictionsfilename(fileprefix::String)
prefix = splitext(fileprefix)[1]
return prefix * ".jdf"
end

function loadpredictions(filename)
    filename = predictionsfilename(filename)
    df = DataFrame()
    try
        df = DataFrame(JDF.loadjdf(EnvConfig.logpath(filename)))
        println("loaded $filename predictions dataframe of size=$(size(df))")
    catch e
        Logging.@warn "exception $e detected"
    end
    return df
end

function savepredictions(df, fileprefix)
    filename = predictionsfilename(fileprefix)
    println("saving $filename predictions dataframe of size=$(size(df))")
    try
        JDF.savejdf(EnvConfig.logpath(filename), df)
    catch e
        Logging.@warn "exception $e detected"
    end
    return fileprefix
end

function predictionsdataframe(nn::NN, setranges, targets, predictions, f3::Features.Features003)
    df = DataFrame(permutedims(predictions, (2, 1)), nn.labels)
    # for (ix, label) in enumerate(nn.labels)
    #     df[label, :] = predictions[ix, :]
    # end
    setlabels = fill("unused", size(df, 1))
    for (setname, vec) in setranges
        for range in vec  # rangeset
            for ix in range
                setlabels[ix] = setname
            end
        end
    end
    sc = categorical(setlabels; compress=true)
    df.set = sc
    df.targets = categorical(targets; levels=nn.labels, compress=true)
    df.opentime = Ohlcv.dataframe(f3.f2.ohlcv).opentime[f3.firstix:end]
    println("Classify.predictionsdataframe size=$(size(df)) keys=$(names(df))")
    println(describe(df, :all))
    fileprefix = uppercase(Ohlcv.basesymbol(f3.f2.ohlcv) * Ohlcv.quotesymbol(f3.f2.ohlcv)) * "_" * nn.fileprefix
    savepredictions(df, fileprefix)
    return fileprefix
end

"returns features as dataframe and targets as CategorialArray (which is why this function resides in Classify and not in Targets)."
function featurestargets(regrwindow, f3::Features.Features003, pe::Dict)
    @assert regrwindow in keys(Features.featureslookback01) "regrwindow=$regrwindow not in keys(Features.featureslookback01)=$(keys(Features.featureslookback01))"
    @assert regrwindow in keys(pe)
    features, _ = Features.featureslookback01[regrwindow](f3)
    println(describe(features, :all))
    featuresdescription = "featureslookback01[$regrwindow](f3)"
    labels, _, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe[regrwindow])
    targets = labels[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    println(describe(DataFrame(reshape(targets, (length(targets), 1)), ["targets"]), :all))
    targetsdescription = "ohlcvlabels(ohlcv.pivot, PriceExtreme[$regrwindow])"
    features = Array(features)  # change from df to array
    @assert size(targets, 1) == size(features, 1)  "size(targets, 1)=$(size(targets, 1)) == size(features, 1)=$(size(features, 1))"
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    return features, targets, featuresdescription, targetsdescription
end

function predictionmatrix(df, labels)
    pred = Array{Float32, 2}(undef, length(labels), size(df, 1))
    pnames = names(df)
    @assert size(pred, 1) == length(labels)
    @assert size(pred, 2) == size(df, 1)
    for (ix, label) in enumerate(labels)
        if label in pnames
            pred[ix, :] = df[:, label]
        else
            pred[ix, :] .= 0.0f0
        end
    end
    return pred
end

function combifeaturestargets(nnvec::Vector{NN}, f3::Features.Features003, pe::Dict)
    @assert "combi" in keys(pe) "`combi` not in keys(pe)=$(keys(pe))"
    labels, _, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe["combi"])
    targets = labels[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    targetsdescription = "ohlcvlabels(ohlcv.pivot, PriceExtreme[combi])"
    features = nothing
    for nn in nnvec
        df = loadpredictions(nn.predictions[begin])
        pred = predictionmatrix(df, nn.labels)
        features = isnothing(features) ? pred : vcat(features, pred)
    end
    featuresdescription = "basepredictions[$([nn.mnemonic for nn in nnvec])]"
    @assert size(targets, 1) == size(features, 2)  "size(targets, 1)=$(size(targets, 1)) == size(features, 2)=$(size(features, 2))"
    return features, targets, featuresdescription, targetsdescription
end

"Returns the column (=samples) subset of featurestargets as given in ranges, which shall be a vector of ranges"
function subsetdim2(featurestargets::AbstractArray, ranges::AbstractVector)
    dim = length(size(featurestargets))
    @assert 0 < dim <= 2 "dim=$dim"

    # working with views
    ixvec = Int32[ix for range in ranges for ix in range]
    res = dim == 1 ? view(featurestargets, ixvec) : view(featurestargets, :, ixvec)
    return res

    # copying ranges - currently disabled
    res = nothing
    for range in ranges
        res = dim == 1 ? (isnothing(res) ? featurestargets[range] : vcat(res, featurestargets[range])) : (isnothing(res) ? featurestargets[:, range] : hcat(res, featurestargets[:, range]))
    end
    return res
end

#endregion DataPrep

#region LearningNetwork Flux

"""````
   lay_in = featurecount
    lay_out = length(labels)
    lay1 = 2 * lay_in
    lay2 = round(Int, (lay1 + lay_out) / 2)
    model = Chain(
        Dense(lay_in => lay1, tanh),   # activation function inside layer
        BatchNorm(lay1),
        Dense(lay1 => lay2, tanh),
        BatchNorm(lay2),
        Dense(lay2 => lay_out, tanh),
        softmax)
    optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.crossentropy```
"""
function model001(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 2 * lay_in
    lay2 = round(Int, (lay1 + lay_out) / 2)
    model = Chain(
        Dense(lay_in => lay1, tanh),   # activation function inside layer
        BatchNorm(lay1),
        Dense(lay1 => lay2, tanh),
        BatchNorm(lay2),
        Dense(lay2 => lay_out, tanh),
        softmax)
    optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.
    description = (@doc model001);
    lossfunc = Flux.crossentropy
    mnemonic = "NN" * (isnothing(mnemonic) ? "" : "$(mnemonic)")
    fileprefix = mnemonic * "_" * EnvConfig.runid()
    nn = NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix)
    return nn
end

function adaptmachine(features::AbstractMatrix, targets::AbstractVector, mnemonic=nothing)
    featurecount = size(features, 1)
    nn = model001(featurecount, Targets.all_labels, mnemonic)

    # The model encapsulates parameters, randomly initialised. Its initial output is:
    out1 = nn.model(features)  # size(classes, observations)
    onehottargets = Flux.onehotbatch(targets, Targets.all_labels)  # onehot class encoding of an observation as one column
    loader = Flux.DataLoader((features, onehottargets), batchsize=64, shuffle=true);

    # Training loop, using the whole data set 1000 times:
    losses = []
    @showprogress for epoch in 1:200  # 1:1000
    # for epoch in 1:200  # 1:1000
        for (x, y) in loader
            loss, grads = Flux.withgradient(nn.model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                nn.lossfunc(y_hat, y)  #TODO here the real gain/loss can be considered
            end
            Flux.update!(nn.optim, nn.model, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
    end

    nn.optim # parameters, momenta and output have all changed
    nn.losses = losses
    return nn
end

" Returns a predictions Float Array of size(classes, observations)"
predict(nn::NN, features) = nn.model(features)  # size(classes, observations)

function predictiondistribution(predictions, classifiertitle)
    maxindex = mapslices(argmax, predictions, dims=1)
    dist = zeros(Int, maximum(unique(maxindex)))
    for ix in maxindex
        dist[ix] += 1
    end
    println("$(EnvConfig.now()) prediction distribution with $classifiertitle classifier: $dist")
end

function adaptmachine(regrwindow, f3::Features.Features003, pe::Dict, setranges::Dict)
    println("$(EnvConfig.now()) preparing features and targets for regressionwindow $regrwindow")
    features, targets, featuresdescription, targetsdescription = featurestargets(regrwindow, f3, pe)
    trainfeatures = subsetdim2(features, setranges["base"])
    traintargets = subsetdim2(targets, setranges["base"])
    trainfeatures, traintargets = oversample(trainfeatures, traintargets)  # all classes are equally trained
    println("$(EnvConfig.now()) adapting machine for regressionwindow $regrwindow")
    nn = adaptmachine(trainfeatures, traintargets, Features.periodlabels(regrwindow))
    nn.featuresdescription = featuresdescription
    nn.targetsdescription = targetsdescription
    println("$(EnvConfig.now()) predicting with machine for regressionwindow $regrwindow")
    pred = predict(nn, features)
    push!(nn.predictions, predictionsdataframe(nn, setranges, targets, pred, f3))
    # predictiondistribution(pred, nn.mnemonic)

    println("saving adapted classifier $(nn.fileprefix)")
    # println(nn)
    savenn(nn)
    # println("$(EnvConfig.now()) load machine from file $(nn.fileprefix) for regressionwindow $regrwindow and predict")
    # nntest = loadnn(nn.fileprefix)
    # println(nntest)
    # predtest = predict(nntest, features)
    # @assert pred ≈ predtest  "NN results differ from loaded NN: pred[:, 1:5] = $(pred[:, begin:begin+5]) predtest[:, 1:5] = $(predtest[:, begin:begin+5])"
    return nn
end

function adaptcombi(nnvec::Vector{NN}, f3::Features.Features003, pe::Dict, setranges::Dict)
    println("$(EnvConfig.now()) preparing features and targets for combi classifier")
    features, targets, featuresdescription, targetsdescription = combifeaturestargets(nnvec, f3, pe)
    trainfeatures = subsetdim2(features, setranges["combi"])
    traintargets = subsetdim2(targets, setranges["combi"])
    trainfeatures, traintargets = oversample(trainfeatures, traintargets)  # all classes are equally trained
    println("$(EnvConfig.now()) adapting machine for combi classifier")
    nn = adaptmachine(trainfeatures, traintargets, "combi")
    nn.featuresdescription = featuresdescription
    nn.targetsdescription = targetsdescription
    nn.predecessors = [nn.fileprefix for nn in nnvec]
    println("$(EnvConfig.now()) predicting with combi classifier")
    pred = predict(nn, features)
    push!(nn.predictions, predictionsdataframe(nn, setranges, targets, pred, f3))
    # predictiondistribution(pred, nn.mnemonic)
    @assert size(pred, 2) == size(features, 2)  "size(pred[combi], 2)=$(size(pred, 2)) == size(features, 2)=$(size(features, 2))"
    @assert size(targets, 1) == size(features, 2)  "size(targets[combi], 1)=$(size(targets, 1)) == size(features, 2)=$(size(features, 2))"

    println("saving adapted classifier $(nn.fileprefix)")
    # println(nn)
    savenn(nn)
    # println("$(EnvConfig.now()) load machine from file $(nn.fileprefix) for regressionwindow combi and predict")
    # nntest = loadnn(nn.fileprefix)
    # println(nntest)
    # predtest = predict(nntest, features)
    # @assert pred ≈ predtest  "NN results differ from loaded NN: pred[:, 1:5] = $(pred[:, begin:begin+5]) predtest[:, 1:5] = $(predtest[:, begin:begin+5])"
    return nn
end


function evaluatepredictions(df::AbstractDataFrame, fileprefix)
    title = fileprefix
    for s in levels(df.set)
        sdf = filter(row -> row.set == s, df, view=true)
        if size(sdf, 1) > 0
            aucscores = Classify.aucscores(sdf)
            println("auc[$s, $title]=$(aucscores)")
            rc = Classify.roccurves(sdf)
            Classify.plotroccurves(rc, "$s / $title")
            # scores, labelindices = Classify.maxpredictions(pred)
            show(stdout, MIME"text/plain"(), Classify.confusionmatrix(sdf, sdf.targets))  # prints the table
        else
            @warn "no auc or roc data for [$s, $title] due to missing predictions"
        end
    end
end

function evaluateclassifier(nn::NN)
    title = nn.mnemonic
    if length(nn.predecessors) > 0
        println("$(EnvConfig.now()) evaluating $(length(nn.predecessors)) predecessors of $title")
    end
    for nnfileprefix in nn.predecessors
        evaluateclassifier(nnfileprefix)
    end
    println("$(EnvConfig.now()) evaluating classifier $title")
    for fileprefix in nn.predictions
        df = loadpredictions(fileprefix)
        evaluatepredictions(df, fileprefix)
    end
end

function evaluateclassifier(fileprefix::String)
    nn = loadnn(fileprefix)
    evaluateclassifier(nn)
end

function evaluate(ohlcv::Ohlcv.OhlcvData, labelthresholds; select=nothing)
    f3, pe = Targets.loaddata(ohlcv, labelthresholds)
    # println(f3)
    len = length(Ohlcv.dataframe(f3.f2.ohlcv).pivot) - Features.ohlcvix(f3, 1) + 1
    # println("$(EnvConfig.now()) len=$len  length(Ohlcv.dataframe(f3.f2.ohlcv).pivot)=$(length(Ohlcv.dataframe(f3.f2.ohlcv).pivot)) Features.ohlcvix(f3, 1)=$(Features.ohlcvix(f3, 1))")
    setranges = setpartitions(1:len, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/80)
    for (s,v) in setranges
        println("$s: length=$(length(v))")
    end
    nnvec = NN[]
    # Threads.@threads for regrwindow in f3.regrwindow
    for regrwindow in f3.regrwindow
        if isnothing(select) || (regrwindow in select)
            push!(nnvec, adaptmachine(regrwindow, f3, pe, setranges))
        else
            println("skipping $regrwindow classifier due to not selected")
        end
    end
    if isnothing(select) || ("combi" in select)
        nncombi = adaptcombi(nnvec, f3, pe, setranges)
        for nn in nncombi.predecessors
            evaluateclassifier(nn)
        end
        evaluateclassifier(nncombi)
    else
        for nn in nnvec
            evaluateclassifier(nn)
        end
        # println("skipping combi classifier due to not selected")
    end
    println("$(EnvConfig.now()) ready with adapting and evaluating classifier stack")
end

function evaluate(base::String, startdt::Dates.DateTime, period; select=nothing)
    ohlcv = Ohlcv.defaultohlcv(base)
    enddt = startdt + period
    Ohlcv.read!(ohlcv)
    subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
    println("loaded $ohlcv")
    labelthresholds = Targets.defaultlabelthresholds
    evaluate(ohlcv, labelthresholds, select=select);
end

function evaluate(base::String; select=nothing)
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.read!(ohlcv)
    println("loaded $ohlcv")
    println(describe(Ohlcv.dataframe(ohlcv)))
    labelthresholds = Targets.defaultlabelthresholds
    evaluate(ohlcv, labelthresholds, select=select);
end

function evaluatetest(startdt=DateTime("2022-01-02T22:54:00")::Dates.DateTime, period=Dates.Day(40); select=nothing)
    enddt = startdt + period
    ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
    labelthresholds = Targets.defaultlabelthresholds
    evaluate( ohlcv, labelthresholds, select=select)
end

function evaluatepredictions(filename)
    println("$(EnvConfig.now()) load predictions from file $(filename)")
    df = loadpredictions(filename)
    evaluatepredictions(df,filename)
end

function lossesfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return "losses_" * prefix * ".jdf"
end

function loadlosses!(nn)
    filename = lossesfilename(nn.fileprefix)
    df = DataFrame()
    try
        df = DataFrame(JDF.loadjdf(EnvConfig.logpath(filename)))
        nn.losses = df[!, "losses"]
        println("loaded $filename losses dataframe of size=$(size(df))")
    catch e
        Logging.@warn "exception $e detected"
    end
    return nn.losses
end

function savelosses(nn::NN)
    filename = lossesfilename(nn.fileprefix)
    df = DataFrame(reshape(nn.losses, (length(nn.losses), 1)), ["losses"])
    df.dummy .= 0.0f0
    println("saving $filename losses dataframe of size=$(size(df))")
    try
        JDF.savejdf(EnvConfig.logpath(filename), df)
    catch e
        Logging.@warn "exception $e detected"
    end
end

function nnfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return "NN" * prefix * ".bson"
end

function savenn(nn::NN)
    # savelosses(nn)
    lvec = nn.losses
    gap = floor(Int, length(lvec) / 100)
    start = length(lvec) % 100
    nn.losses = [l for (ix, l) in enumerate(lvec) if ((ix-start) % gap) == 0]
    @assert length(nn.losses) <= 100 "length(loses)=$(length(nn.losses))"
    BSON.@save EnvConfig.logpath(nnfilename(nn.fileprefix)) nn
    # @error "save machine to be implemented for pure flux" filename
    # smach = serializable(mach)
    # JLSO.save(filename, :machine => smach)
end

function loadnn(filename)
    nn = model001(1, Targets.all_labels, "dummy")  # dummy data struct
    BSON.@load EnvConfig.logpath(nnfilename(filename)) nn
    # loadlosses!(nn)
    return nn
    # @error "load machine to be implemented for pure flux" filename
    # loadedmach = JLSO.load(filename)[:machine]
    # Deserialize and restore learned parameters to useable form:
    # restore!(loadedmach)
    # return loadedmach
end

#endregion LearningNetwork

#region Evaluation

labelvec(labelindexvec, labels=Targets.all_labels) = [labels[i] for i in labelindexvec]

labelindexvec(labelvec, labels=Targets.all_labels) = [findfirst(x -> x == focuslabel, labels) for focuslabel in labelvec]

"returns a (scores, labelindices) tuple of best predictions"
function maxpredictions end
function maxpredictions(predictions::AbstractMatrix, labels=Targets.all_labels)
    @assert length(labels) == size(predictions, 1)
    maxindex = mapslices(argmax, predictions, dims=1)
    scores = [predictions[maxindex[i], i] for i in eachindex(maxindex)]
    return scores, maxindex
end

function maxpredictions(predictions::AbstractDataFrame, labels=Targets.all_labels)
    if size(predictions, 1) == 0
        return [],[]
    end
    pnames = names(predictions)
    scores = zeros32(size(predictions, 1))
    maxindex = ones(UInt32, size(predictions, 1))
    missinglabels = []
    for (lix, label) in enumerate(labels)
        if label in pnames
            vec = predictions[!, label]
            for ix in eachindex(vec)
                if scores[ix] < vec[ix]
                    scores[ix] = vec[ix]
                    maxindex[ix] = lix
                end
            end
        else
            push!(missinglabels, label)
        end
    end
    if length(missinglabels) > 0
        println("maxpredictions: no predictions for $missinglabels")
    end
    return scores, maxindex
end

"returns a (scores, booltargets) tuple of binary predictions of class `label`, i.e. booltargets[ix] == true if bestscore is assigned to focuslabel"
function binarypredictions end

function binarypredictions(predictions::AbstractMatrix, focuslabel::String, labels=Targets.all_labels)
    flix = findfirst(x -> x == focuslabel, labels)
    @assert !isnothing(flix) && (firstindex(labels) <= flix <= lastindex(labels)) "$focuslabel==$(isnothing(flix) ? "nothing" : flix) not found in $labels[$(firstindex(labels)):$(lastindex(labels))]"
    @assert length(labels) == size(predictions, 1) "length(labels)=$(length(labels)) == size(predictions, 1)=$(size(predictions, 1))"
    if size(predictions, 2) == 0
        return [],[]
    end
    maxindex = mapslices(argmax, predictions, dims=1)
    ixvec = [flix == maxindex[i] for i in eachindex(maxindex)]
    # labelvec = [flix == maxindex[i] ? focuslabel : "other" for i in eachindex(maxindex)]
    return (length(ixvec) > 0 ? predictions[flix, :] : []), ixvec
end

function binarypredictions(predictions::AbstractDataFrame, focuslabel::String, labels=Targets.all_labels)
    @assert focuslabel in labels
    if size(predictions, 1) == 0
        return [],[]
    end
    pnames = names(predictions)
    if !(focuslabel in pnames)
        return fill(0.0f0, size(predictions, 1)), fill(false, size(predictions, 1))
    end
    _, maxindex = maxpredictions(predictions, labels)
    flix = 0
    for (lix, label) in enumerate(labels)
        flix = focuslabel == label ? lix : flix
    end
    ixvec = [flix == maxindex[i] for i in eachindex(maxindex)]
    return predictions[:, focuslabel], ixvec
end

function smauc(scores, predlabels)
    y = categorical(predlabels, ordered=true)  # StatisticalMeasures package
    ŷ = UnivariateFinite([false, true], scores, augment=true, pool=y)   # StatisticalMeasures package
    return auc(ŷ, y)  # StatisticalMeasures package
end

function aucscores(pred, labels=Targets.all_labels)
    aucdict = Dict()
    if typeof(pred) == DataFrame ? (size(pred, 1) > 0) : (size(pred, 2) > 0)
        for focuslabel in labels
            scores, predlabels = binarypredictions(pred, focuslabel, labels)
            if length(unique(predlabels)) == 2
                aucdict[focuslabel] = smauc(scores, predlabels)
            else
                # @warn "no max score for $focuslabel => auc = 0"
                aucdict[focuslabel] = 0.0
            end
        end
    end
    return aucdict
end

# aucscores(pred, labels=Targets.all_labels) = Dict(String(focuslabel) => auc(binarypredictions(pred, focuslabel, labels)...) for focuslabel in labels)
    # auc_scores = []
    # for class_label in unique(targets)
    #     class_scores, class_events = binarypredictions(pred, targets, class_label)
    #     auc_score = auc(class_scores, class_events)
    #     push!(auc_scores, auc_score)
    # end
    # return auc_scores
# end

"Returns a Dict of class => roc tuple of vectors for false_positive_rates, true_positive_rates, thresholds"
function roccurves(pred, labels=Targets.all_labels)
    rocdict = Dict()
    if typeof(pred) == DataFrame ? (size(pred, 1) > 0) : (size(pred, 2) > 0)
        for focuslabel in labels
            scores, predlabels = binarypredictions(pred, focuslabel, labels)
            if length(unique(predlabels)) == 2
                y = categorical(predlabels, ordered=true)  # StatisticalMeasures package
                ŷ = UnivariateFinite([false, true], scores, augment=true, pool=y)   # StatisticalMeasures package
                rocdict[focuslabel] = roc_curve(ŷ, y)  # StatisticalMeasures package
            else
                # @warn "no max score for $focuslabel => roc = nothing"
                rocdict[focuslabel] = nothing
            end
        end
    end
    return rocdict
end

function plotroccurves(rc::Dict, customtitle)
    plotlyjs()
    default(legend=true)
    plt = plot()
    for (k, v) in rc
        if !isnothing(v)
            # println("$k = $(length(v[1]))  xxx $(length(v[2]))")
            # plot!(v, label=k)  # ROC package
            plot!(v[1], v[2], label=k)  # StatisticalMeasures package
        end
    end
    # plot!(xlab="false positive rate", ylab="true positive rate")
    plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black, label="chance")  # StatisticalMeasures package
    xlabel!("false positive rate")
    ylabel!("true positive rate")
    title!("receiver operator characteristic $customtitle")
    display(plt)
end

function confusionmatrix(pred, targets, labels=Targets.all_labels)
    _, maxindex = maxpredictions(pred, labels)
    targets = [String(targets[ix]) for ix in eachindex(targets)]
    predlabels = labelvec(maxindex, labels)
    StatisticalMeasures.ConfusionMatrices.confmat(predlabels, targets)
end

#endregion Evaluation

function fluxstart()
    noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
    truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

    # Define our model, a multi-layer perceptron with one hidden layer of size 3:
    model = Chain(
        Dense(2 => 3, tanh),   # activation function inside layer
        BatchNorm(3),
        Dense(3 => 2),
        softmax) |> gpu        # move model to GPU, if available

    # The model encapsulates parameters, randomly initialised. Its initial output is:
    out1 = model(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}

    # To train the model, we use batches of 64 samples, and one-hot encoding:
    target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
    loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true);
    # 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

    optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

    # Training loop, using the whole data set 1000 times:
    losses = []
    @showprogress for epoch in 1:1_000
        for (x, y) in loader
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                Flux.crossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
    end

    optim # parameters, momenta and output have all changed
    out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)

    mean((out2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!
end

end  # module
