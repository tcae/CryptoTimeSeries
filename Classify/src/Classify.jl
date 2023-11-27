"""
Train and evaluate the trading signal classifiers
"""
module Classify

using DataFrames, Logging, Dates, PrettyPrinting
using JLSO, Flux, Statistics, ProgressMeter, StatisticalMeasures, ROC
using EnvConfig, Ohlcv, Features, Targets, TestOhlcv
using Plots

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
        - gaps will be removed from a subrange
        - relativesubrangesize * length(rowrange) > 2 * gapsize
        - any relative setsize > 2 * relativesubrangesize
        - a mixture of ranges and individual indices in a vector can be unpacked into a index vector via `[ix for r in rr for ix in r]`
"""
function setpartitions(rowrange, samplesets::Dict, gapsize, relativesubrangesize)
    rowstart = rowrange[1]
    rowend = rowrange[end]
    rows = rowend - rowstart + 1
    @assert relativesubrangesize * rows > 2 * gapsize
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


"returns features as dataframe and targets as CategorialArray (which is why this function resides in Classify and not in Targets)."
function featurestargets(regrwindow, f3::Features.Features003, pe::Dict)
    @assert regrwindow in keys(Features.featureslookback01)
    @assert regrwindow in keys(pe)
    features, _ = Features.featureslookback01[regrwindow](f3)
    labels, _, _, _ = Targets.ohlcvlabels(Ohlcv.dataframe(f3.f2.ohlcv).pivot, pe[regrwindow])
    targets = labels[Features.ohlcvix(f3, 1):end]  # cut beginning from ohlcv observations to feature observations
    features = Array(features)  # change from df to array
    @assert size(targets, 1) == size(features, 1)
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    return features, targets
end

#endregion DataPrep

#region LearningNetwork Flux

struct FluxMachine
    model
    optim
    labels  # in fixed sequence as index == class id
    losses
end

function adaptmachine(features, targets)::FluxMachine
    model = Chain(
        Dense(36 => 64, tanh),   # activation function inside layer
        BatchNorm(64),
        Dense(64 => 32),
        BatchNorm(32),
        Dense(32 => length(Targets.all_labels)),
        softmax)

    # The model encapsulates parameters, randomly initialised. Its initial output is:
    out1 = model(features)  # size(classes, observations)
    onehottargets = Flux.onehotbatch(targets, Targets.all_labels)  # onehot class encoding of an observation as one column
    loader = Flux.DataLoader((features, onehottargets), batchsize=64, shuffle=true);
    optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

    # Training loop, using the whole data set 1000 times:
    losses = []
    @showprogress for epoch in 1:200  # 1:1000
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
    fm = FluxMachine(model, optim, Targets.all_labels, losses)
    return fm
end

" Returns a predictions Float Array of size(classes, observations)"
predict(fm::FluxMachine, features) = fm.model(features)  # size(classes, observations)

function savemach(mach, filename)
    @error "save machine to be implemented for pure flux" filename
    # smach = serializable(mach)
    # JLSO.save(filename, :machine => smach)
end

function loadmach(filename)
    @error "load machine to be implemented for pure flux" filename
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
function maxpredictions(predictions, labels=Targets.all_labels)
    @assert length(labels) == size(predictions, 1)
    maxindex = mapslices(argmax, predictions, dims=1)
    scores = [predictions[maxindex[i], i] for i in eachindex(maxindex)]
    return scores, maxindex
end

"returns a (scores, booltargets) tuple of binary predictions of class `label`, i.e. booltargets[ix] == true if bestscore is assigned to focuslabel"
function binarypredictions(predictions, focuslabel, labels=Targets.all_labels)
    flix = findfirst(x -> x == focuslabel, labels)
    @assert !isnothing(flix) && (firstindex(labels) <= flix <= lastindex(labels)) "$focuslabel==$(isnothing(flix) ? "nothing" : flix) not found in $labels[$(firstindex(labels)):$(lastindex(labels))]"
    @assert length(labels) == size(predictions, 1) "length(labels)=$(length(labels)) == size(predictions, 1)=$(size(predictions, 1))"
    maxindex = mapslices(argmax, predictions, dims=1)
    ixvec = [flix == maxindex[i] for i in eachindex(maxindex)]
    return predictions[flix, :], ixvec
end


aucscores(pred, labels=Targets.all_labels) = Dict(String(focuslabel) => ROC.AUC(ROC.roc(binarypredictions(pred, focuslabel, labels)...)) for focuslabel in labels)
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
roccurves(pred, labels=Targets.all_labels) = Dict(String(focuslabel) => ROC.roc(binarypredictions(pred, focuslabel, labels)...) for focuslabel in labels)
# roccurves(pred, labels=Targets.all_labels) = Dict(String(focuslabel) => roc_curve(binarypredictions(pred, focuslabel, labels)...) for focuslabel in labels)

function plotroccurves(rc::Dict)
    plotlyjs()
    default(legend=true)
    plt = plot()
    for (k, v) in rc
        # println("$k = $(length(v[1]))  xxx $(length(v[2]))")
        plot!(v, label=k)
    end
    # plot!(xlab="false positive rate", ylab="true positive rate")
    # plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black, label="chance")
    xlabel!("false positive rate")
    ylabel!("true positive rate")
    title!("receiver operator characteristic")
end

function confusionmatrix(pred, targets, labels=Targets.all_labels)
    _, maxindex = maxpredictions(pred, labels)
    StatisticalMeasures.ConfusionMatrices.confmat(labelvec(maxindex, labels), targets)
end

#endregion Evaluation

# EnvConfig.init(production)
# EnvConfig.init(test)

end  # module
