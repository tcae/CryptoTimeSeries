"""
Train and evaluate the trading signal classifiers
"""
module Classify

abstract type AbstractClassifier end

using CSV, DataFrames, Logging, Dates, PrettyPrinting, PrettyTables, Plots
using BSON, JDF, Flux, Statistics, ProgressMeter, StatisticalMeasures, MLUtils
using CategoricalArrays
using CategoricalDistributions
# using Distributions
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv, CryptoXch, Assets
export noop, hold, sell, buy, strongsell, strongbuy

@enum InvestProposal noop hold sell buy strongsell strongbuy # noop = no statement possible, e.g. due to error, trading shall go on save side

const PREDICTIONLISTFILE = "predictionlist.csv"

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info, e.g. number of steps in rowix
"""
verbosity = 1

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

function predictionlist()
    df = nothing
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    pf = EnvConfig.logpath(PREDICTIONLISTFILE)
    if isfile(pf)
        df = CSV.read(pf, DataFrame, decimal='.', delim=';')
        # println(describe(df))
        # println(df)
    end
    EnvConfig.setlogpath(sf)
    return df
end

function registerpredictions(filename, evalperf=missing, testperf=missing)
    df = predictionlist()
    if isnothing(df)
        df = DataFrame(abbrev=String[], evalperf=[], testperf=[], comment=String[], subfolder=String[], filename=String[])
    else
        rows = findall(==(filename), df[!, "filename"])
        if length(rows) > 1
            @error "found $rows entries instead of one in $PREDICTIONLISTFILE"
            return
        elseif length(rows) == 1
            # no action - filename already registered
            return
        elseif length(rows) == 0
            # not found - don't return but add filename
        else
            println("found for $filename the follwing in $PREDICTIONLISTFILE: $rows")
        end
    end
    sf = EnvConfig.logsubfolder()
    push!(df, ("abbrev", round(evalperf, digits=3), round(testperf, digits=3), "comment", sf, filename))
    EnvConfig.setlogpath(nothing)
    pf = EnvConfig.logpath(PREDICTIONLISTFILE)
    CSV.write(pf, df, decimal='.', delim=';')
    EnvConfig.setlogpath(sf)
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
    if !("pivot" in names(df))
        ix = findfirst("USDT", filename)
        if isnothing(ix)
            @warn "cannot repair pivot because no USDT found in filename"
        else
            # println("ix=$ix typeof(ix)=$(typeof(ix))")  # ix is and index range
            base = SubString(filename, 1, (first(ix)-1))
            ohlcv = Ohlcv.defaultohlcv(base)
            startdt = minimum(df[!, "opentime"])
            if startdt != df[begin, "opentime"]
                @warn "startdt=$startdt != df[begin, opentime]=$(df[begin, "opentime"])"
            end
            enddt = maximum(df[!, "opentime"])
            if enddt != df[end, "opentime"]
                @warn "enddt=$enddt != df[begin, opentime]=$(df[end, "opentime"])"
            end
            Ohlcv.read!(ohlcv)
            subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
            df[:, "pivot"] = Ohlcv.dataframe(ohlcv).pivot
            println("succesful repair of pivot")
        end
    end
    # checktargetsequence(df[!, "targets"])
    checklabeldistribution(df[!, "targets"])
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
    df[:, "set"] = sc
    df[:, "targets"] = categorical(targets; levels=nn.labels, compress=true)
    df[:, "opentime"] = Ohlcv.dataframe(f3.f2.ohlcv).opentime[f3.firstix:end]
    df[:, "pivot"] = Ohlcv.dataframe(f3.f2.ohlcv).pivot[f3.firstix:end]
    println("Classify.predictionsdataframe size=$(size(df)) keys=$(names(df))")
    println(describe(df, :all))
    fileprefix = uppercase(Ohlcv.basecoin(f3.f2.ohlcv) * Ohlcv.quotecoin(f3.f2.ohlcv)) * "_" * nn.fileprefix
    savepredictions(df, fileprefix)
    return fileprefix
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

function checktargetsequence(targets::CategoricalArray)
    labels = levels(targets)
    ignoreix, longbuyix, longholdix, closeix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in ["ignore", "longbuy", "longhold", "close", "shorthold", "shortbuy"])
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    buy = levelcode(first(targets)) in [longholdix, closeix] ? longbuyix : (levelcode(first(targets)) == shortholdix ? shortbuyix : ignoreix)
    for (ix, tl) in enumerate(targets)
        if levelcode(tl) == longbuyix
            buy = longbuyix
        elseif (levelcode(tl) == longholdix)
            if (buy != longbuyix)
                @error "$ix: missed $longbuyix ($(labels[longbuyix])) before $longholdix ($(labels[longholdix]))"
            end
        elseif levelcode(tl) == shortbuyix
            buy = shortbuyix
        elseif (levelcode(tl) == shortholdix)
            if (buy != shortbuyix)
                @error "$ix: missed $shortbuyix ($(labels[shortbuyix])) before $shortholdix ($(labels[shortholdix]))"
            end
        elseif (levelcode(tl) == closeix)
            if (buy == ignoreix)
                @error "$ix: missed either $shortbuyix ($(labels[shortbuyix])) or $longbuyix ($(labels[longbuyix])) before $closeix ($(labels[closeix]))"
            end
        elseif (levelcode(tl) == ignoreix)
            buy = ignoreix
        else
            @error "$ix: unexpected $(levelcode(tl)) ($(labels[closeix]))"
        end
    end
    println("target sequence check ready")

end

function checklabeldistribution(targets::CategoricalArray)
    labels = levels(targets)
    cnt = zeros(Int, length(labels))
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    for tl in targets
        cnt[levelcode(tl)] += 1
    end
    targetcount = size(targets, 1)
    println("target label distribution in %: ", [(labels[i], round(cnt[i] / targetcount*100, digits=1)) for i in eachindex(labels)])

end

"returns features as dataframe and targets as CategorialArray (which is why this function resides in Classify and not in Targets)."
function featurestargets(regrwindow, f3::Features.Features003, pe::Dict)
    @assert regrwindow in keys(Features.featureslookback01) "regrwindow=$regrwindow not in keys(Features.featureslookback01)=$(keys(Features.featureslookback01))"
    @assert regrwindow in keys(pe)
    features, _ = Features.featureslookback01[regrwindow](f3)
    println(describe(features, :all))
    featuresdescription = "featureslookback01[$regrwindow](f3)"
    labels, relativedist, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe[regrwindow])
    targets = labels[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    relativedist = relativedist[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    println(describe(DataFrame(reshape(targets, (length(targets), 1)), ["targets"]), :all))
    targetsdescription = "ohlcvlabels(ohlcv.pivot, PriceExtreme[$regrwindow])"
    features = Array(features)  # change from df to array
    @assert size(targets, 1) == size(features, 1) == size(relativedist, 1)  "size(targets, 1)=$(size(targets, 1)) == size(features, 1)=$(size(features, 1)) == size(relativedist, 1)=$(size(relativedist, 1))"
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    return features, targets, featuresdescription, targetsdescription, relativedist
end

function combifeaturestargets(nnvec::Vector{NN}, f3::Features.Features003, pe::Dict)
    @assert "combi" in keys(pe) "`combi` not in keys(pe)=$(keys(pe))"
    labels, relativedist, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe["combi"])
    targets = labels[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    relativedist = relativedist[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    targetsdescription = "ohlcvlabels(ohlcv.pivot, PriceExtreme[combi])"
    features = nothing
    for nn in nnvec
        df = loadpredictions(nn.predictions[begin])
        pred = predictionmatrix(df, nn.labels)
        features = isnothing(features) ? pred : vcat(features, pred)
    end
    featuresdescription = "basepredictions[$([nn.mnemonic for nn in nnvec])]"
    @assert size(targets, 1) == size(features, 2) == size(relativedist, 1)  "size(targets, 1)=$(size(targets, 1)) == size(features, 2)=$(size(features, 2)) == size(relativedist, 1)=$(size(relativedist, 1))"
    return features, targets, featuresdescription, targetsdescription, relativedist
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

"""```
Neural Net description:
lay_in = featurecount
lay_out = length(labels)
lay1 = 3 * lay_in
lay2 = round(Int, lay1 * 2 / 3)
lay3 = round(Int, (lay2 + lay_out) / 2)
model = Chain(
    Dense(lay_in => lay1, relu),   # activation function inside layer
    BatchNorm(lay1),
    Dense(lay1 => lay2, relu),   # activation function inside layer
    BatchNorm(lay2),
    Dense(lay2 => lay3, relu),   # activation function inside layer
    BatchNorm(lay3),
    Dense(lay3 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
lossfunc = Flux.logitcrossentropy
```
"""
function model001(featurecount, labels, mnemonic)::NN
    lay_in = featurecount
    lay_out = length(labels)
    lay1 = 3 * lay_in
    lay2 = round(Int, lay1 * 2 / 3)
    lay3 = round(Int, (lay2 + lay_out) / 2)
    model = Chain(
        Dense(lay_in => lay1, relu),   # activation function inside layer
        BatchNorm(lay1),
        Dense(lay1 => lay2, relu),   # activation function inside layer
        BatchNorm(lay2),
        Dense(lay2 => lay3, relu),   # activation function inside layer
        BatchNorm(lay3),
        Dense(lay3 => lay_out))   # no activation function inside layer, no softmax in combination with logitcrossentropy instead of crossentropy with softmax
    optim = Flux.setup(Flux.Adam(0.001,(0.9, 0.999)), model)  # will store optimiser momentum, etc.
    lossfunc = Flux.logitcrossentropy

    description = (@doc model001);
    mnemonic = "NN" * (isnothing(mnemonic) ? "" : "$(mnemonic)")
    fileprefix = mnemonic * "_" * EnvConfig.runid()
    nn = NN(model, optim, lossfunc, labels, description, mnemonic, fileprefix)
    return nn
end

"""
creates and adapts a neural network using `features` with ground truth label provided with `targets` that belong to observation samples with index ix within the original sample sequence.
relativedist is a vector
"""
function adaptnn!(nn::NN, features::AbstractMatrix, targets::AbstractVector)
    onehottargets = Flux.onehotbatch(targets, Targets.all_labels)  # onehot class encoding of an observation as one column
    loader = Flux.DataLoader((features, onehottargets), batchsize=64, shuffle=true);

    # Training loop, using the whole data set 1000 times:
    nn.losses = Float32[]
    testmode!(nn, false)
    minloss = maxloss = missing
    breakmsg = ""
    @showprogress for epoch in 1:1000
    # for epoch in 1:200  # 1:1000
        losses = Float32[]
        for (x, y) in loader
            loss, grads = Flux.withgradient(nn.model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                #y_hat is a Float32 matrix , y is a boolean matrix both of size (classes, batchsize). The loss function returns a single Float32 number
                nn.lossfunc(y_hat, y)
            end
            minloss = ismissing(minloss) ? loss : min(minloss, loss)
            maxloss = ismissing(maxloss) ? loss : max(maxloss, loss)
            Flux.update!(nn.optim, nn.model, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
        epochloss = mean(losses)
        push!(nn.losses, epochloss)  # register only mean(losses) over a whole epoch
        if (epoch > 10) && (nn.losses[end-4] <= nn.losses[end-3] <= nn.losses[end-2] <= nn.losses[end-1] <= nn.losses[end])
            breakmsg = "stopping adaptation after epoch $epoch due to no loss reduction ($(nn.losses[end-4:end])) in last 4 epochs"
            break
        end
    end
    testmode!(nn, true)
    nn.optim # parameters, momenta and output have all changed
    println(breakmsg)  # print after showprogress loop to avoid cluttered text output
    println("minloss=$minloss  maxloss=$maxloss")
    return nn
end

" Returns a predictions Float Array of size(classes, observations)"
predict(nn::NN, features) = nn.model(features)  # size(classes, observations)

function predictiondistribution(predictions, classifiertitle)
    maxindex = mapslices(argmax, predictions, dims=1)
    dist = zeros(Int, maximum(unique(maxindex)))  # maximum == length
    for ix in maxindex
        dist[ix] += 1
    end
    println("$(EnvConfig.now()) prediction distribution with $classifiertitle classifier: $dist")
end

function adaptbase(regrwindow, f3::Features.Features003, pe::Dict, setranges::Dict)
    println("$(EnvConfig.now()) preparing features and targets for regressionwindow $regrwindow")
    features, targets, featuresdescription, targetsdescription, relativedist = featurestargets(regrwindow, f3, pe)
    # trainix = subsetdim2(collect(firstindex(targets):lastindex(targets)), setranges["base"])
    trainfeatures = subsetdim2(features, setranges["base"])
    traintargets = subsetdim2(targets, setranges["base"])
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    (trainfeatures), traintargets = oversample((trainfeatures), traintargets)  # all classes are equally trained
    # println("after oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    println("$(EnvConfig.now()) adapting machine for regressionwindow $regrwindow")
    nn = model001(size(trainfeatures, 1), Targets.all_labels, Features.periodlabels(regrwindow))
    nn = adaptnn!(nn, trainfeatures, traintargets)
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
    features, targets, featuresdescription, targetsdescription, relativedist = combifeaturestargets(nnvec, f3, pe)
    # trainix = subsetdim2(collect(firstindex(targets):lastindex(targets)), setranges["combi"])
    # println("size(features)=$(size(features)), size(targets)=$(size(targets))")
    trainfeatures = subsetdim2(features, setranges["combi"])
    traintargets = subsetdim2(targets, setranges["combi"])
    # println("before oversample size(trainfeatures)=$(size(trainfeatures)), size(traintargets)=$(size(traintargets))")
    (trainfeatures), traintargets = oversample((trainfeatures), traintargets)  # all classes are equally trained
    # println("after oversample size(trainfeatures)=$(size(trainfeatures)), size(traintargets)=$(size(traintargets))")
    println("$(EnvConfig.now()) adapting machine for combi classifier")
    nn = model001(size(trainfeatures, 1), Targets.all_labels, "combi")
    nn = adaptnn!(nn, trainfeatures, traintargets)
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
    println("saving $filename losses dataframe of size=$(size(df))")
    try
        JDF.savejdf(EnvConfig.logpath(filename), df)
    catch e
        Logging.@warn "exception $e detected"
    end
end

function nnfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return prefix * ".bson"
end

function compresslosses(losses)
    if length(losses) <= 1000
        return losses
    end
    @warn "nn.losses length=$(length(losses))"
    lvec = losses
    gap = floor(Int, length(lvec) / 100)
    start = length(lvec) % 100
    closses = [l for (ix, l) in enumerate(lvec) if ((ix-start) % gap) == 0]
    @assert length(closses) <= 100 "length(loses)=$(length(closses))"
    return closses
end

function savenn(nn::NN)
    # nn.losses = compresslosses(nn.losses)
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

"""
Groups trading pairs and stores them with the corresponding gain in a DataFrame that is returned.
Pairs that cross a set before closure are being forced closed at set border.
"""
function trades(predictions::AbstractDataFrame, thresholds::Vector)
    df = DataFrame(set=CategoricalArray(undef, 0; levels=levels(predictions.set), ordered=false), opentrade=Int32[], openix=Int32[], closetrade=Int32[], closeix=Int32[], gain=Float32[])
    if size(predictions, 1) == 0
        return df
    end
    scores, maxindex = maxpredictions(predictions)
    labels = levels(predictions.targets)
    ignoreix, longbuyix, longholdix, closeix, shortholdix, shortbuyix = (findfirst(x -> x == l, labels) for l in ["ignore", "longbuy", "longhold", "close", "shorthold", "shortbuy"])
    buytrade = (tradeix=closeix, predix=0, set=predictions.set[begin])  # tradesignal, predictions index
    holdtrade = (tradeix=closeix, predix=0, set=predictions.set[begin])  # tradesignal, predictions index

    function closetrade!(tradetuple, closetrade, ix)
        gain = (predictions.pivot[ix] - predictions.pivot[tradetuple.predix]) / predictions.pivot[tradetuple.predix] * 100
        gain = tradetuple.tradeix in [longbuyix, longholdix] ? gain : -gain
        push!(df, (tradetuple.set, tradetuple.tradeix, tradetuple.predix, closetrade, ix, gain))
        return (tradeix=closeix, predix=ix, set=predictions.set[ix])
    end

    function closeifneeded!(buychecktrades, holdchecktrades, closetrade, ix)
        buytrade = (buytrade.tradeix == buychecktrades) || ((buytrade.set != predictions.set[ix]) && (buytrade.tradeix in [longbuyix, shortbuyix])) ? closetrade!(buytrade, closetrade, ix) : buytrade
        holdtrade = (holdtrade.tradeix == holdchecktrades) || ((buytrade.set != predictions.set[ix]) && (buytrade.tradeix in [longholdix, shortholdix])) ? closetrade!(holdtrade, closetrade, ix) : holdtrade
        return buytrade, holdtrade
    end

    for ix in eachindex(maxindex)
        labelix = maxindex[ix]
        score = scores[ix]
        if (labelix == longbuyix) && (thresholds[longbuyix] <= score)
            buytrade, holdtrade = closeifneeded!(shortbuyix, shortholdix, longbuyix, ix)
            buytrade = buytrade.tradeix == closeix ? (tradeix=longbuyix, predix=ix, set=predictions.set[ix]) : buytrade
        elseif (labelix == longholdix) && (thresholds[longholdix] <= score)
            buytrade, holdtrade = closeifneeded!(shortbuyix, shortholdix, longholdix, ix)
            holdtrade = holdtrade.tradeix == closeix ? (tradeix=longholdix, predix=ix, set=predictions.set[ix]) : holdtrade
        elseif ((labelix == closeix) && (thresholds[closeix] <= score)) || ((labelix == ignoreix) && (thresholds[ignoreix] <= score))
            buytrade = buytrade.tradeix != closeix ? closetrade!(buytrade, closeix, ix) : buytrade
            holdtrade = holdtrade.tradeix != closeix ? closetrade!(holdtrade, closeix, ix) : holdtrade
        elseif (labelix == shortholdix) && (thresholds[shortholdix] <= score)
            buytrade, holdtrade = closeifneeded!(longbuyix, longholdix, shortholdix, ix)
            holdtrade = holdtrade.tradeix == closeix ? (tradeix=shortholdix, predix=ix, set=predictions.set[ix]) : holdtrade
        elseif (labelix == shortbuyix) && (thresholds[shortbuyix] <= score)
            buytrade, holdtrade = closeifneeded!(longbuyix, longholdix, shortbuyix, ix)
            buytrade = buytrade.tradeix == closeix ? (tradeix=shortbuyix, predix=ix, set=predictions.set[ix]) : buytrade
        end
    end
    return df
end

function tradeperformance(trades::AbstractDataFrame, labels::Vector)
    df = DataFrame(set=CategoricalArray(undef, 0; levels=labels, ordered=false), trade=[], tradeix=[], gainpct=[], cnt=[], gainpctpertrade=[])
    for tset in levels(trades.set)
        for ix in eachindex(labels)
            selectedtrades = filter(row -> (row.set == tset) && (row.opentrade == ix), trades, view=true)
            if size(selectedtrades, 1) > 0
                tcount = count(i-> !ismissing(i), selectedtrades.gain)
                tsum = sum(selectedtrades.gain)
                push!(df, (tset, labels[ix], ix, tsum, tcount, round(tsum/tcount, digits=3)))
            end
        end
        selectedtrades = filter(row -> (row.set == tset), trades, view=true)
        tcount = count(i-> !ismissing(i), selectedtrades.gain)
        tsum = sum(selectedtrades.gain)
        push!(df, (tset, "total", missing, tsum, tcount, tsum/tcount))
    end
    # println(df)
    return df
end

"maps a 0.0 <= score <= 1.0  to one of `thresholdbins` bins"
score2bin(score, thresholdbins) = max(min(floor(Int, score / (1.0/thresholdbins)) + 1, thresholdbins), 1)

"maps the index of one of `thresholdbins` bins to a score"
bin2score(binix, thresholdbins) = round((binix-1)*1.0/thresholdbins; digits = 2), round(binix*1.0/thresholdbins; digits = 2)

"""
generates summary statistics from predictions
"""
function extendedconfusionmatrix(predictions::AbstractDataFrame, thresholdbins=10)
    scores, maxindex = maxpredictions(predictions)
    labels = levels(predictions.targets)
    confcatsyms = [:tp, :tn, :fp, :fn]
    confcat = Dict(zip(confcatsyms, 1:length(confcatsyms)))
    # preallocate collection matrices with columns TP, TN, FP, FN and rows as bins with lower separation value x/thresholdbins per label per set
    setnames = levels(predictions.set)
    cmc = zeros(Int, length(setnames), length(labels), length(confcatsyms), thresholdbins)
    for ix in eachindex(maxindex)
        labelix = maxindex[ix]  # label of maxscore
        binix = score2bin(scores[ix], thresholdbins)
        if labelix == levelcode(predictions.targets[ix])
            cmc[levelcode(predictions.set[ix]), labelix, confcat[:tp], binix] += 1
        else  # labelix != levelcode(predictions.targets[ix])
            cmc[levelcode(predictions.set[ix]), labelix, confcat[:fp], binix] += 1
        end
    end
    cm = zeros(Int, length(setnames), length(labels), length(confcatsyms), thresholdbins)
    for six in eachindex(setnames)
        for lix in eachindex(labels)
            for bix in 1:thresholdbins
                for bix2 in bix:thresholdbins
                    cm[six, lix, confcat[:tp], bix] += cmc[six, lix, confcat[:tp], bix2]
                    cm[six, lix, confcat[:fp], bix] += cmc[six, lix, confcat[:fp], bix2]
                end
                for bix2 in 1:(bix-1)
                    cm[six, lix, confcat[:fn], bix] += cmc[six, lix, confcat[:tp], bix2]
                    cm[six, lix, confcat[:tn], bix] += cmc[six, lix, confcat[:fp], bix2]
                end
            end
        end
    end
    setnamevec = [setnames[six] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    sc = categorical(setnamevec; levels=setnames)
    labelsvec = [labels[lix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    lc = categorical(labelsvec; levels=labels)
    binvec = [(scr = bin2score(bix, thresholdbins); "$bix/[$(scr[1])-$(scr[2])]") for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    bc = categorical(binvec)
    tpvec = [cm[six, lix, confcat[:tp], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    tnvec = [cm[six, lix, confcat[:tn], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    fpvec = [cm[six, lix, confcat[:fp], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    fnvec = [cm[six, lix, confcat[:fn], bix] for six in eachindex(setnames) for lix in eachindex(labels) for bix in 1:thresholdbins]
    allvec = tpvec + tnvec + fpvec + fnvec
    tpprc = round.(tpvec ./ allvec .* 100.0; digits=2)
    tnprc = round.(tnvec ./ allvec .* 100.0; digits=2)
    fpprc = round.(fpvec ./ allvec .* 100.0; digits=2)
    fnprc = round.(fnvec ./ allvec .* 100.0; digits=2)
    tpr = round.(tpvec ./ (tpvec + fnvec); digits=2)
    fpr = round.(fpvec ./ (fpvec + tnvec); digits=2)
    xcdf = DataFrame("set" => sc, "pred_label" => lc, "bin" => bc, "tp" => tpvec, "tn" => tnvec, "fp" => fpvec, "fn" => fnvec, "tp%" => tpprc, "tn%" => tnprc, "fp%" => fpprc, "fn%" => fnprc, "tpr" => tpr, "fpr" => fpr)
    # println(xcdf)
    return xcdf
    #TODO next step: take only first of an equal trading signal sequence according to threshold -> how often is a sequence missed?
 end

 function confusionmatrix(predictions::AbstractDataFrame)
    scores, maxindex = maxpredictions(predictions)
    labels = levels(predictions.targets)
    confcatsyms = [:tp, :tn, :fp, :fn]
    confcat = Dict(zip(confcatsyms, 1:length(confcatsyms)))
    # preallocate collection matrices with columns TP, TN, FP, FN and rows as bins with lower separation value x/thresholdbins per label per set
    setnames = levels(predictions.set)
    cm = zeros(Int, length(setnames), length(labels), length(labels))
    for ix in eachindex(maxindex)
        cm[levelcode(predictions.set[ix]), maxindex[ix], levelcode(predictions.targets[ix])] += 1
    end
    setnamevec = [setnames[six] for six in eachindex(setnames) for lix in eachindex(labels)]
    sc = categorical(setnamevec; levels=setnames)
    labelsvec = ["pred_"*labels[lix] for six in eachindex(setnames) for lix in eachindex(labels)]
    plc = categorical(labelsvec)
    cdf = DataFrame(:set => sc, :predlabel => plc)
    pred_sum = zeros(Int, length(setnames) * length(labels))
    for (ix, l) in enumerate(labels)
        if l != "pred_invalid"
            lvec = [cm[six, lix, ix] for six in eachindex(setnames) for lix in eachindex(labels)]
            cdf[:, "truth_"*l] = lvec
            pred_sum += lvec
        end
    end
    # cdf[:, "truth_other"] = pred_sum - [cm[six, lix, lix] for six in eachindex(setnames) for lix in eachindex(labels)]
    cdf[:, "truth_all"] = pred_sum
    for (ix, l) in enumerate(labels)
        cdf[:, ("truth_"*l*"_%")] = round.(cdf[:, ("truth_"*l)] ./ pred_sum * 100; digits=2)
    end
    # println(cdf)
    return cdf
 end

function predictionsfilename(fileprefix::String)
    prefix = splitext(fileprefix)[1]
    return prefix * ".jdf"
end

labelvec(labelindexvec, labels=Targets.all_labels) = [labels[i] for i in labelindexvec]

labelindexvec(labelvec, labels=Targets.all_labels) = [findfirst(x -> x == focuslabel, labels) for focuslabel in labelvec]

"returns a (scores, labelindices) tuple of best predictions. Without labels the index is the index within levels(df.targets)."
function maxpredictions end
function maxpredictions(predictions::AbstractMatrix, labels=Targets.all_labels)
    @assert length(labels) == size(predictions, 1)
    maxindex = mapslices(argmax, predictions, dims=1)
    scores = [predictions[maxindex[i], i] for i in eachindex(maxindex)]
    return scores, maxindex
end

# function maxpredictions(predictions::AbstractDataFrame)
#     if size(predictions, 1) == 0
#         return [],[]
#     end
#     scores = zeros32(size(predictions, 1))
#     maxindex = zeros(UInt32, size(predictions, 1))
#     for (lix, label) in enumerate(names(predictions))
#         if eltype(predictions[!, label]) <: AbstractFloat
#             if !(label in Targets.all_labels)
#                 @warn "unexpected predictions class: $label"
#             end
#             vec = predictions[!, label]
#             for ix in eachindex(vec)
#                 if scores[ix] < vec[ix]
#                     scores[ix] = vec[ix]
#                     maxindex[ix] = lix
#                 end
#             end
#         end
#     end
#     return scores, maxindex
# end

function maxpredictions(predictions::AbstractDataFrame, labels=Targets.all_labels)
    if size(predictions, 1) == 0
        return [],[]
    end
    # labels = levels(predictions.targets)
    # @assert labels == levels(predictions.targets) "labels=$labels != levels(predictions.targets)=$(levels(predictions.targets))"
    pnames = names(predictions)
    scores = zeros32(size(predictions, 1))
    maxindex = zeros(UInt32, size(predictions, 1))
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
    # ts, tm = maxpredictionsfromcols(predictions)
    # @assert all([pnames[tm[ix]] == labels[maxindex[ix]] for ix in eachindex(maxindex)]) && (ts == scores) "maxindex check = $(tm == maxindex)  scores check = $(ts == scores)"
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


function evaluatepredictions(predictions::AbstractDataFrame, fileprefix)
    assetpair, nntitle = split(fileprefix, "_")[1:2]
    title = assetpair * "_" * nntitle
    if EnvConfig.configmode == EnvConfig.test
        cdf = confusionmatrix(predictions)
        xcdf = extendedconfusionmatrix(predictions)
    end
    labels = levels(predictions.targets)
    thresholds = [0.01f0 for l in labels]
    tdf = trades(predictions, thresholds)
    tpdf = tradeperformance(tdf, labels)

    selectedtrades = filter(row -> (row.set == "eval") && ((row.trade == "longbuy") || (row.trade == "shortbuy")), tpdf, view=true)
    if length(selectedtrades.gainpct) > 0
        tcount = sum(selectedtrades.cnt)
        tsum = sum(selectedtrades.gainpct)
        evalperf = round(tsum/tcount, digits=3)
    else
        evalperf = 0.0
    end
    selectedtrades = filter(row -> (row.set == "test") && ((row.trade == "longbuy") || (row.trade == "shortbuy")), tpdf, view=true)
    if length(selectedtrades.gainpct) > 0
        tcount = sum(selectedtrades.cnt)
        tsum = sum(selectedtrades.gainpct)
        testperf = round(tsum/tcount, digits=3)
    else
        testperf = 0.0
    end
    if EnvConfig.configmode == EnvConfig.production
        registerpredictions(fileprefix, evalperf, testperf)
    end

    for s in levels(predictions.set)
        if s == "unused"
            continue
        end
        sdf = filter(row -> row.set == s, predictions, view=true)
        if size(sdf, 1) > 0
            if EnvConfig.configmode == EnvConfig.test
                # println(title)
                # aucscores = Classify.aucscores(sdf)
                # println("auc[$s, $title]=$(aucscores)")
                # rc = Classify.roccurves(sdf)
                # Classify.plotroccurves(rc, "$s / $title")
                println(title)
                show(stdout, MIME"text/plain"(), Classify.confusionmatrix(sdf, sdf.targets))  # prints the table
                println(title)
                println(filter(row -> row.set == s, cdf, view=true))
                println(title)
                println(filter(row -> row.set == s, xcdf, view=true))
            end
            println(title)
            println(filter(row -> row.set == s, tpdf, view=true))
        else
            @warn "no auc or roc data for [$s, $title] due to missing predictions"
        end
    end
end

function evaluatepredictions(filename)
    println("$(EnvConfig.now()) load predictions from file $(filename)")
    df = loadpredictions(filename)
    evaluatepredictions(df,filename)
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
    packetsize = length(nn.losses) > 20 ? floor(Int, length(nn.losses) / 20) : 1  # only display 20 lines of loss summary
    startp = lastlosses = nothing
    for i in eachindex(nn.losses)
        if i > firstindex(nn.losses)
            if (i % packetsize == 0) || (i == lastindex(nn.losses))
                plosses = mean(nn.losses[startp:i])
                println("epoch $startp-$i loss: $plosses  lossdiff: $((plosses-lastlosses)/lastlosses*100)%")
                startp = i+1
                lastlosses = plosses
            end
        else
            println("loss: $(nn.losses[i])")
            startp = i+1
            lastlosses = nn.losses[i]
        end
    end
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
    nnvec = NN[]
    # push!(nnvec, loadnn("NN1d_23-12-18_15-58-18_gitSHA-b02abf01b3a714054ea6dd92d5b683648878b079"))
    f3, pe = Targets.loaddata(ohlcv, labelthresholds)
    # println(f3)
    len = length(Ohlcv.dataframe(f3.f2.ohlcv).pivot) - Features.ohlcvix(f3, 1) + 1
    # println("$(EnvConfig.now()) len=$len  length(Ohlcv.dataframe(f3.f2.ohlcv).pivot)=$(length(Ohlcv.dataframe(f3.f2.ohlcv).pivot)) Features.ohlcvix(f3, 1)=$(Features.ohlcvix(f3, 1))")
    setranges = setpartitions(1:len, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/80)
    for (s,v) in setranges
        println("$s: length=$(length(v))")
    end
    # Threads.@threads for regrwindow in f3.regrwindow
    for regrwindow in f3.regrwindow
        if isnothing(select) || (regrwindow in select)
            push!(nnvec, adaptbase(regrwindow, f3, pe, setranges))
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

function evaluate(base::String, startdt::Dates.DateTime=DateTime("2017-07-02T22:54:00"), period=Dates.Year(10); select=nothing)
    ohlcv = Ohlcv.defaultohlcv(base)
    enddt = startdt + period
    Ohlcv.read!(ohlcv)
    subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
    println("loaded $ohlcv")
    labelthresholds = Targets.defaultlabelthresholds
    evaluate(ohlcv, labelthresholds, select=select);
end

# function evaluate(base::String; select=nothing)
#     ohlcv = Ohlcv.defaultohlcv(base)
#     Ohlcv.read!(ohlcv)
#     println("loaded $ohlcv")
#     println(describe(Ohlcv.dataframe(ohlcv)))
#     labelthresholds = Targets.defaultlabelthresholds
#     evaluate(ohlcv, labelthresholds, select=select);
# end

function evaluatetest(startdt=DateTime("2022-01-02T22:54:00")::Dates.DateTime, period=Dates.Day(40); select=nothing)
    enddt = startdt + period
    ohlcv = TestOhlcv.testohlcv("SINEUSDT", startdt, enddt)
    labelthresholds = Targets.defaultlabelthresholds
    evaluate( ohlcv, labelthresholds, select=select)
end

#endregion Evaluation

#region Classifier-001

"cache and config: 1) receive ohlcv, 2) read config, 3) calculate f2"
mutable struct Cls001Cache
    ohlcv::Union{Nothing, Ohlcv.OhlcvData}    # ohlcv cache
    f4::Union{Nothing, Features.Features004}  # f4 cache that is shared with all regressionwindows
    regrwindow
    gainthreshold
end
STDREGRWINDOW = 1440
STDREGRWINDOWSET = [rw for rw in Features.regressionwindows004 if rw > 500 && rw < 10000]
STDGAINTHRSHLD = 0.01f0
STDGAINTHRSHLDSET = [0.005f0, 0.01f0, 0.02f0]
FEE = 0.1f0 / 100f0  # 0.1%
# FEE = 0f0  # no fee

"""
Returns a `buy` recommendation if the standard deviation of a given regression window relative to price of the regression line exceeds a threshold and
the current price is below regressionline minus standard deviation.
Corresondingly returns a `sell` recommendation if std/price > threshold and price > std + price. In all other price cases a `hold` is returned.
"""
mutable struct Classifier001 <: AbstractClassifier
    cfg::DataFrame  # configuration dataframe
    function Classifier001()::Classifier001
        return new(emptyconfigdf())
    end
end

CLASSIFIER001_CONFIGFILE = "Classifier001config"

function Base.show(io::IO, bc::Cls001Cache)
    println(io, "Classifier001[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f4=$(!isnothing(bc.f4))")  # \n$(bc.ohlcv) \n$(bc.f4)")
end

function Base.show(io::IO, cls::Classifier001)
    if size(cls.cfg, 1) > 0
        activedf = @view cls.cfg[cls.cfg[!, :active] .== true, :]
    else
        activedf = cls.cfg
    end
    println(io, "Classifier001: active config: $activedf")  # \n$(bc.ohlcv) \n$(bc.f4)")
end

function _cfgfilename(timestamp::Union{Nothing, DateTime}, ext)
    cfgfilename = EnvConfig.logpath(CLASSIFIER001_CONFIGFILE)
    if isnothing(timestamp)
        cfgfilename = join([cfgfilename, ext], ".")
    else
        cfgfilename = join([cfgfilename, Dates.format(timestamp, "yy-mm-dd_HH-MM"), ext], "_", ".")
    end
    return cfgfilename
end

"if timestamp=nothing then no extension otherwise timestamp extension"
function write(cls::Classifier001, timestamp::Union{Nothing, DateTime}=nothing)
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(timestamp, "jdf")
    EnvConfig.checkbackup(cfgfilename)
    JDF.savejdf(cfgfilename, cls.cfg[cls.cfg[!, :active] .== true, :])
    # CSV.write(cfgfilename, cls.cfg[cls.cfg[!, :active] .== true, :], decimal=',', delim=';')  # decimal as , to consume with European locale
    EnvConfig.setlogpath(sf)
end

function read!(cls::Classifier001, timestamp::Union{Nothing, DateTime}=nothing)
    df = emptyconfigdf()
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(timestamp, "jdf")
    # if isfile(cfgfilename)
    if isdir(cfgfilename)
        df = DataFrame(JDF.loadjdf(cfgfilename))
        # df = CSV.read(cfgfilename, DataFrame, decimal='.', delim=';')
        println("config df: $df, metadata: $(collect(metadatakeys(df)))")
    end
    EnvConfig.setlogpath(sf)
    if isnothing(df)
        @warn "Loading $cfgfilename failed"
    end
    cls.cfg = df
    return cls
end

function requiredminutes(cls::Union{Classifier001, Nothing})
    return maximum(Features.regressionwindows004)
    # igonre below, stay on safe side
    if isnothing(cls) || (size(cls.cfg, 1) == 0)
        return maximum(STDREGRWINDOWSET)
    else
        return maximum(cls.cfg[!, :regrwindow])
    end
end

"config DataFrame with columns: basecoin, regrwindow, gainthreshold, active, sellonly, minqteqty, startdt, enddt, totalcnt, sellcnt, buycnt, maxccbuycnt, medianccbuycnt, unresolvedcnt, totalgain, mediangain, meangain, cumgain, maxcumgain, mincumgain, maxgap, mediangap, usd10000, minusd10000, maxusd10000"
emptyconfigdf() = DataFrame(basecoin=String[], regrwindow=Int16[], gainthreshold=Float32[], active=Bool[], sellonly=Bool[], minqteqty=Float32[], startdt=DateTime[], enddt=DateTime[], totalcnt=Int32[], sellcnt=Int32[], buycnt=Int32[], maxccbuycnt=Int32[], medianccbuycnt=Float32[], unresolvedcnt=Int32[], totalgain=Float32[], mediangain=Float32[], meangain=Float32[], cumgain=Float32[], maxcumgain=Float32[], mincumgain=Float32[], maxgap=Int32[], mediangap=Float32[], usd10000=Float32[], minusd10000=Float32[], maxusd10000=Float32[])
dummyconfig = (basecoin="dummy", regrwindow=0, gainthreshold=0f0, active=false, sellonly=false, minqteqty=0f0, startdt=DateTime("2020-01-01T01:00:00"), enddt=DateTime("2020-01-01T01:00:00"), totalcnt=0, sellcnt=0, buycnt=0, maxccbuycnt=0, medianccbuycnt=0f0, unresolvedcnt=0, totalgain=0f0, mediangain=0f0, meangain=0f0, cumgain=0f0, maxcumgain=0f0, mincumgain=0f0, maxgap=0, mediangap=0f0, usd10000=10000f0, minusd10000=10000f0, maxusd10000=10000f0)

function _addconfig!(cls::Classifier001, base::String, regrwindow, gainthreshold, active, sellonly)
    push!(cls.cfg, dummyconfig)
    cls.cfg[end, :basecoin] = base
    cls.cfg[end, :regrwindow] = regrwindow
    cls.cfg[end, :gainthreshold] = gainthreshold
    cls.cfg[end, :active] = active
    cls.cfg[end, :sellonly] = sellonly
    return cls.cfg[size(cls.cfg, 1), :]
end

_active(row, base) = (row.basecoin == base) && row.active

# cfg is sorted 1) ascending for regrwindow 2) ascending for gainthreshold -> return the first buy/sell advice or hold if no such buy/sell signalled is identified
"Returns a DataFrameRow with the first active config or nothing if nothing is found"
baseclassifieractiveconfigs(cls::Classifier001, base) = baseclassifieractiveconfigs(cls.cfg, base)
baseclassifieractiveconfigs(cfg::AbstractDataFrame, base) = (ix = findfirst(row -> _active(row, base), eachrow(cfg)); isnothing(ix) ? nothing : cfg[ix, :])

function addreplaceconfig!(cls::Classifier001, base, regrwindow, gainthreshold, active, sellonly=false)
    @assert regrwindow in Features.regressionwindows004
    cfgix = findall((cls.cfg[!, :basecoin] .== base) .&& (cls.cfg[!, :regrwindow] .== regrwindow) .&& (cls.cfg[!, :gainthreshold] .== gainthreshold))
    cfg = nothing
    if length(cfgix) == 0
        cfg = _addconfig!(cls, base, regrwindow, gainthreshold, active, sellonly)
        # sort!(cls.cfg, [:basecoin, :regrwindow, :gainthreshold])
    else
        for ix in reverse(cfgix) # use reverse to maintain correct index in loop
            if ix == first(cfgix)
                # leave first entry
                cls.cfg[ix, :active] = active
                cls.cfg[ix, :sellonly] = sellonly
                cfg = cls.cfg[ix, :]
            else
                #remove all other entries
                deleteat!(cls.cfg, ix)
            end
        end
    end
    return cfg
end

function removeconfig!(cls::Classifier001, base, regrwindow, gainthreshold)
    cfgix = findall((cls.cfg[!, :basecoin] .== base) .&& (cls.cfg[!, :regrwindow] .== regrwindow) .&& (cls.cfg[!, :gainthreshold] .== gainthreshold))
    if length(cfgix) > 0
        for ix in reverse(cfgix) # use reverse to maintain correct index in loop
            deleteat!(cls.cfg, ix)
        end
    end
end

regrwindows(cls::Classifier001, base, activeonly=true) = activeonly ? unique(cls.cfg[cls.cfg[!, :basecoin] .== base .&& cls.cfg[!, :active] .== true, :regrwindow]) : unique(cls.cfg[cls.cfg[!, :basecoin] .== base, :regrwindow])

function _advice(bc::Cls001Cache, ohlcv::OhlcvData, oix)
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    fix = Features.f4ix(bc.f4, oix)
    dt = ohlcvdf[oix, :opentime]
    regrwindow = bc.regrwindow
    gainthreshold = bc.gainthreshold
    if fix <= 0
        return noop
    end
    if bc.f4.rw[regrwindow][fix, :opentime] == dt
        if ohlcvdf[oix, :pivot] > bc.f4.rw[regrwindow][fix, :regry] * (1 + gainthreshold)
            return sell
        elseif ohlcvdf[oix, :pivot] < bc.f4.rw[regrwindow][fix, :regry] * (1 - gainthreshold)
            if bc.f4.rw[14400][fix, :grad] >= 0f0
                return buy
            end
        end
    elseif bc.f4.rw[regrwindow][begin, :opentime] < dt
        @warn "expected $(ohlcv.base) ohlcv opentime[oix]=$dt not found in f4[$regrwindow] with start=$(bc.f4.rw[regrwindow][begin, :opentime]) end=$(bc.f4.rw[regrwindow][end, :opentime])"
    end
    return hold
end

"Returns `InvestProposal` recommendations for the timestamp given by `ohlcv.ix`
and all following until end of the ohlcv dataframe"
function advices(cls::Classifier001, ohlcv, regrwindow, gainthreshold)::Tuple{Vector{InvestProposal}, Union{Cls001Cache, Nothing}}
    f4 = Features.Features004(ohlcv, firstix=ohlcv.ix, usecache=true)
    bc = Cls001Cache(ohlcv, f4, regrwindow, gainthreshold)
    # @assert bc.f4.rw[bc.regrwindow][Features.f4ix(bc.f4, ohlcv.ix), :opentime] == ohlcv.df[ohlcv.ix, :opentime] "ohlcv.ix=$(ohlcv.ix) bc.f4.rw[$(bc.regrwindow)][$(Features.f4ix(bc.f4, ohlcv.ix))]=$(bc.f4.rw[bc.regrwindow][Features.f4ix(bc.f4, ohlcv.ix), :opentime]) ohlcv.df[ohlcv.ix, :opentime]=$(ohlcv.df[ohlcv.ix, :opentime])"
    ip = [_advice(bc, ohlcv, oix) for oix in ohlcv.ix:size(ohlcv.df, 1)]
    return ip, bc
end

"Returns a single `InvestProposal` recommendations for the timestamp given by `ohlcv.ix`"
function advice(cls::Classifier001, ohlcv)::InvestProposal
    ip = noop
    if ohlcv.ix < maximum(Features.regressionwindows004)
        return ip
    end
    f4 = Features.Features004(ohlcv, firstix=ohlcv.ix, lastix=ohlcv.ix)
    cdf = baseclassifieractiveconfigs(cls, ohlcv.base)
    if !isnothing(cdf)
        bc = Cls001Cache(ohlcv, f4, cdf.regrwindow, cdf.gainthreshold)
        @assert isa(cdf.gainthreshold, Float32) "typeof(cdf.gainthreshold)=$(typeof(cdf.gainthreshold))"
        @assert isa(cdf.regrwindow, Integer) "typeof(cdf.gainthreshold)=$(typeof(cdf.gainthreshold))"
        @assert bc.f4.rw[bc.regrwindow][Features.f4ix(bc.f4, ohlcv.ix), :opentime] == ohlcv.df[ohlcv.ix, :opentime] "ohlcv.ix=$(ohlcv.ix) bc.f4.rw[$(bc.regrwindow)][$(Features.f4ix(bc.f4, ohlcv.ix))]=$(bc.f4.rw[bc.regrwindow][Features.f4ix(bc.f4, ohlcv.ix), :opentime]) ohlcv.df[ohlcv.ix, :opentime]=$(ohlcv.df[ohlcv.ix, :opentime])"
        ip = _advice(bc, ohlcv, ohlcv.ix)
    else
        @error "no advices for $(ohlcv.base) due to missing configuration"
    end
    return ip
end

_ohlcvix(ohlcvstartix, adviceix) = adviceix + ohlcvstartix - 1

function _tradepairs(ipvec::Vector{InvestProposal}, bc::Cls001Cache, startix)
    ot = bc.ohlcv.df[!, :opentime]
    piv = Ohlcv.pivot!(bc.ohlcv)
    buystack = []
    usd10000stack = []
    usd10000 = (free=10000f0, used=0f0)
    usd10000rate = 1/1000
    tradepairs = DataFrame([name => [] for name in ["buyix", "buydt", "buyprice", "buycnt", "sellix", "selldt", "sellprice", "gain", "gap", "usd10000"]])
    for ipix in eachindex(ipvec)
        if ipvec[ipix] == buy

            usd10000part = sum(usd10000)*usd10000rate
            if usd10000part < usd10000.free
                push!(usd10000stack, usd10000part)
                usd10000 = (free=usd10000.free-usd10000part, used=usd10000.used+usd10000part)
            end

            push!(buystack, ipix)
        elseif ipvec[ipix] == sell
            sellix = _ohlcvix(startix, ipix)
            sellprice = piv[sellix]*(1-FEE)
            if length(buystack) > 0
                buyix = _ohlcvix(startix, popfirst!(buystack))
                buyprice = piv[buyix]*(1+FEE)
                gain = (sellprice - buyprice) / buyprice * 100

                if length(usd10000stack) > 0
                    usd10000part = popfirst!(usd10000stack)
                    usd10000 = (free=usd10000.free+usd10000part * (1 + gain/100), used=usd10000.used-usd10000part)
                end
                push!(tradepairs, (buyix=buyix, buydt=ot[buyix], buyprice=buyprice, buycnt=length(buystack), sellix=sellix, selldt=ot[sellix], sellprice=sellprice, gain=gain, gap=sellix-buyix, usd10000=sum(usd10000)), promote=true)
            else  # buyix = missing
                push!(tradepairs, (buyix=missing, buydt=missing, buyprice=missing, buycnt=0, sellix=sellix, selldt=ot[sellix], sellprice=sellprice, gain=missing, gap=missing, usd10000=sum(usd10000)), promote=true)
            end
        end
    end
    while length(buystack) > 0  # sellix = missing
        buyix = _ohlcvix(startix, popfirst!(buystack))
        buyprice = piv[buyix]*(1+FEE)
        push!(tradepairs, (buyix=buyix, buydt=ot[buyix], buyprice=buyprice, buycnt=length(buystack), sellix=missing, selldt=missing, sellprice=missing, gain=missing, gap=missing, usd10000=sum(usd10000)), promote=true)
    end
    return tradepairs
end

function _timecut(tradepairs::AbstractDataFrame, column, sellcheck::Bool, buycheck::Bool, startix, endix, invert::Bool=false)
    sellfilter = sellcheck ? coalesce.(startix .<= tradepairs[!, :sellix] .<= endix, false) : trues(length(tradepairs[!, :sellix]))
    buyfilter = buycheck ? coalesce.(startix .<= tradepairs[!, :buyix] .<= endix, false) : trues(length(tradepairs[!, :buyix]))
    filter = buyfilter .&& sellfilter
    filter = invert ? xor.(filter, trues(length(filter))) : filter
    if isnothing(column)
        return @view tradepairs[filter, :]
    else
        return @view tradepairs[filter, column]
    end
end

function _extremes(gains)
    cumgain = maxcumgain = mincumgain = 0.0f0
    # maxdrawdown = maxgaintmp = maxgain = mingaintmp = mingain = 0.0f0
    # maxgaintmpix = maxgainix = mingaintmpix = mingainix = firstindex(gains)
    for ix in eachindex(gains)
        cumgain += gains[ix]
        maxcumgain = max(maxcumgain, cumgain)
        mincumgain = min(mincumgain, cumgain)
        #=
        # now the maxdrawdown:
        if maxgaintmp < cumgain
            maxgaintmp = cumgain
            maxgaintmpix = ix
            # mingaintmp is reestablished with every maxgaintmp change -> mingaintmp is always equal or newer than maxgaintmp
            mingaintmp = cumgain
            mingaintmpix = ix
            if maxgain < maxgaintmp
                # safe drawdown before establishing a new candidate
                maxdrawdown = max(maxdrawdown, maxgain - mingain)
                maxgain = maxgaintmp
                maxgainix = maxgaintmpix
                # mingain is reestablished with every maxgain change -> mingain is always equal or newer than maxgain
                mingain = cumgain
                mingainix = ix
            end
        end
        if mingaintmp > cumgain
            mingaintmp = cumgain
            mingaintmpix = ix
            if mingain > mingaintmp
                mingain = cumgain
                mingainix = ix
            end
        end
        =#
    end
    return (cumgain=cumgain, maxcumgain=maxcumgain, mincumgain=mincumgain)
end

_count(f, itr::AbstractArray; init) = isempty(itr) ? init : count(f, itr)
_minimum(itr::AbstractArray; init) = isempty(itr) ? init : minimum(itr)
_maximum(itr::AbstractArray; init) = isempty(itr) ? init : maximum(itr)
_median(itr::AbstractArray; init) = isempty(itr) ? init : median(itr)
_mean(itr::AbstractArray; init) = isempty(itr) ? init : mean(itr)
_last(itr::AbstractArray; init) = isempty(itr) ? init : last(itr)

function _addconfig!(cls::Classifier001, bc::Cls001Cache, startix, endix, tradepairs)
    cfg = _addconfig!(cls, bc.ohlcv.base, bc.regrwindow, bc.gainthreshold, false, false) # returned cfg is a DataFrameRow and elements can be changed like a mutable NamedTuple
    cfg.startdt = bc.ohlcv.df[startix, :opentime]
    cfg.enddt = bc.ohlcv.df[endix, :opentime]
    cfg.totalcnt = endix - startix + 1

    cfg.sellcnt = _count(>(0), _timecut(tradepairs, :sellix, true, false, startix, endix), init=0)
    buytradepairs = _timecut(tradepairs, nothing, false, true, startix, endix)
    cfg.buycnt = _count(>(0), buytradepairs[!, :buyix], init=0)
    cfg.maxccbuycnt = _maximum(buytradepairs[!, :buycnt], init=0)
    cfg.medianccbuycnt = _median(buytradepairs[!, :buycnt], init=0f0)
    cfg.unresolvedcnt = _count(ismissing, _timecut(tradepairs, :sellix, true, false, startix, endix, true), init=0) + count(ismissing, _timecut(tradepairs, :buyix, false, true, startix, endix, true), init=0)
    filteredtradepairs = _timecut(tradepairs, nothing, true, true, startix, endix)
    cfg.totalgain = sum(filteredtradepairs[!, :gain], init=0f0)
    cfg.mediangain = _median(filteredtradepairs[!, :gain], init=0f0)
    cfg.meangain = _mean(filteredtradepairs[!, :gain], init=0f0)
    cfg.cumgain, cfg.maxcumgain, cfg.mincumgain = _extremes(filteredtradepairs[!, :gain])
    cfg.maxgap = _maximum(filteredtradepairs[!, :gap], init=0)
    cfg.mediangap = _median(filteredtradepairs[!, :gap], init=0f0)
    cfg.usd10000 = _last(filteredtradepairs[!, :usd10000], init=0f0)
    cfg.minusd10000 = _minimum(filteredtradepairs[!, :usd10000], init=0f0)
    cfg.maxusd10000 = _maximum(filteredtradepairs[!, :usd10000], init=0f0)
    return cfg
end

"""
Assessment of best tokens and best regressioin window per token backward from ohlcv.ix backwards for given period
- all previous configuration data will be replaced by a new dataframe
- creates a data frame with rows of the cartesian product of bases, regrwindows, gainthresholds as rows of cls.cfg (any previous cls.cfg dataframe is dropped)
- adds to each row the calculated evaluations
- all config rows are set to active=false
"""
function evaluate!(cls::Classifier001, xc::CryptoXch.XchCache, bases, regrwindows, gainthresholds, startdt, enddt)
    #TODO which tokens to use for trading?
    #TODO which regression windows to use? there should be only 1 at the same time active to avoid losses due to mixed strategies
    #TODO to be evaluated when all but less than minquantity of a token is sold to avoid losses in sell due to trading strategy change
    #TODO assess period as parameter
    @assert (length(bases) > 0) && (length(regrwindows) > 0) && (length(gainthresholds) > 0) "(length(bases) > 0)=$(length(bases) > 0) && (length(regrwindows) > 0)=$(length(regrwindows) > 0) && (length(gainthresholds) > 0)=$(length(gainthresholds) > 0)"
    @assert all([rw in Features.regressionwindows004 for rw in regrwindows]) "not all $regrwindows in $(Features.regressionwindows004)"
    cls.cfg = emptyconfigdf()
    f4offset = Minute(requiredminutes(cls))
    for base in bases
        ohlcv = CryptoXch.cryptodownload(xc, base, "1m", startdt-f4offset, enddt)
        CryptoXch.timerangecut!(ohlcv, startdt-f4offset, enddt)
        if size(ohlcv.df, 1) < requiredminutes(cls)
            # @warn "ohlcv size=$(size(ohlcv.df, 1)) of $base insufficient for required minutes $(requiredminutes(cls)) - $base will be skipped"
            continue
        end
        startix = Ohlcv.rowix(ohlcv.df.opentime, startdt-f4offset, Ohlcv.intervalperiod("1m"))
        endix = Ohlcv.rowix(ohlcv.df.opentime, enddt, Ohlcv.intervalperiod("1m"))
        ohlcvlen = endix - startix + 1
        f4 = Features.Features004(ohlcv, firstix=ohlcv.ix, usecache=true)
        for regr in regrwindows
            for gth in gainthresholds
                (verbosity >= 2) && print("\r$(EnvConfig.timestr(enddt)) $(ohlcv.base), $regr, $gth     ")
                if size(ohlcv.df, 1) > 0
                    bc = Cls001Cache(ohlcv, f4, regr, gth)
                    ipvec = [_advice(bc, ohlcv, oix) for oix in ohlcv.ix:size(ohlcv.df, 1)]
                    # ipvec, bc = advices(cls, ohlcv, regr, gth)

                    @assert length(ipvec) == ohlcvlen "length(ipvec)=$(length(ipvec)) == ohlcvlen=$ohlcvlen, ohlcv=$ohlcv"
                    tradepairs = _tradepairs(ipvec, bc, startix)
                    topxstartix = Ohlcv.rowix(ohlcv.df.opentime, startdt, Ohlcv.intervalperiod("1m"))  # now without f4offset
                    _addconfig!(cls, bc, topxstartix, endix, tradepairs)
                else
                    _addconfig!(cls, base, regr, gth, false, false)
                end
            end
        end
    end
    sort!(cls.cfg, [:basecoin, :regrwindow, :gainthreshold])  # beauty only
    metadata!(cls.cfg, "created", "$enddt", style=:note)
    (verbosity >= 2) && print("\r")
    return cls
end

"""
Loads coin candidates via Assets and evaluates the performance through a given period with end date enddt (nothing=now).
The result will be stored as configuration dataframe and any previous config dataframe will be replaced.
"""
function best!(cls::Classifier001, xc::CryptoXch.XchCache, topx, period, enddt=nothing, update=true, assetbases=[])
    ad = Assets.read!(Assets.AssetData())
    # use last saved data for all requests older than the last saved
    if (size(ad.basedf, 1)== 0) || ((isnothing(enddt) || (enddt > maximum(ad.basedf[!, :update]))) && update)
        ad = Assets.loadassets!(Assets.AssetData(), assetbases)
        for coin in eachrow(ad.basedf)
            ohlcv = Ohlcv.defaultohlcv(coin.base)
            ohlcv = Ohlcv.read!(ohlcv)
            f4 = Features.Features004(ohlcv, usecache=true)
            if !isnothing(f4)
                Features.write(f4)
            end
        end
    end
    enddt = isnothing(enddt) ? maximum(ad.basedf[!, :update]) : enddt
    startdt = enddt - period
    bases = unique(union(ad.basedf[!, :base], assetbases))
    (verbosity >= 3) && println("Loaded candidates: $(bases)")
    cls = Classify.evaluate!(cls, xc, bases, Classify.STDREGRWINDOWSET, Classify.STDGAINTHRSHLDSET, startdt, enddt)
    sort!(cls.cfg, [:usd10000], rev=true)  # prio1 in descending order
    activebases = []
    for row in eachrow(cls.cfg)
        if (row.usd10000 > 10500) && !(row.basecoin in activebases)  # at least 5% within 10 days added and each coin only with one config active
            row.active = true
            push!(activebases, row.basecoin)
            if length(activebases) >= topx
                break
            end
        end
    end
    topxdf = cls.cfg[cls.cfg[!, :active], :]
    (verbosity >= 3) && println("$(EnvConfig.timestr(enddt)) top-$topx trading coins: $(topxdf[!, :basecoin])")
    return topxdf
end

#endregion Classifier-001


end  # module
