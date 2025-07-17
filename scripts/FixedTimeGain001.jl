using Classify

"""
- start with Testohlcv to check learning works
- find out how many blocks of a coin are formed by using only high volume coins
- then check adaptation per coin
- then adapt with all coins and retrain per coin
- one level base classifier (no combi)
- no lookback features

hyper parameters:
- gain: 0.5%, 1%, 2%, 4%
- window: 30min, 1h, 4h, 12h

classes: binary longbuy yes vs no (=longclose) with hysteresis using likelihood % of longbuy violate

"""
module FixedTimeGain001
using Test, Dates, Logging, CSV, DataFrames, Statistics, MLUtils
using EnvConfig, Classify, CryptoXch, Ohlcv, Features, Targets

allregressionfeatures() = [join(["rw", rw, rp], "_") for rw in [5, 15, 60, 4*60] for rp in ["grad", "regry"]] # see Features.regressionwindows005 and Features.savedregressionproperties for all options
popularminmaxfeatures() = [join(["mm", mmw, rp], "_") for mmw in [4*60] for rp in Features.minmaxproperties] # see Features.regressionwindows005 and Features.savedregressionproperties for all options
popularvolumefeatures() = [join(["rv", sw, lw], "_") for (sw, lw) in [(5, 24*60)]]

function calcfeatures(base::AbstractString, startdt::DateTime, enddt::DateTime, f5::Features.AbstractFeatures)
    println("$(EnvConfig.now()) loading $base") 
    xc = CryptoXch.XchCache()
    ohlcv = CryptoXch.cryptodownload(xc, base, "1m", startdt, enddt)
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)

    println("$(EnvConfig.now()) feature calculation")
    Features.setbase!(f5, ohlcv, usecache=true)
    # Features.write(f5)
    features = Features.features(f5)

    features = Array(features)  # change from df to array
    features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
    return features
end

"""
returns targets
feature base has to be set before calling because that determines the ohlcv and relevant time range
"""
function calctargets(f5::Features.AbstractFeatures, fdg::Targets.AbstractTargets)
    ohlcv = Features.ohlcv(f5)
    features = Features.features(f5)
    fot = Features.opentime(f5)
    println("$(EnvConfig.now()) target calculation fromm $(fot[begin]) until $(fot[end])")
    Targets.setbase!(fdg, ohlcv)
    # targets = Targets.labels(fdg, fot[begin], fot[end])
    targets = Targets.longbuybinarytargets(fdg, fot[begin], fot[end])
    @assert size(features, 1) == length(targets) "size(features, 1)=$(size(features, 1)) != length(targets)=$(length(targets))"
    print("Total ")
    Targets.labeldistribution(targets)
    # println(describe(fdg.df, :all))

    return targets
end

function evalclassifier(nn)
    println("evaluating classifier $(nn.fileprefix)")
    Classify.evaluateclassifier(nn)
    # println("$(EnvConfig.now()) load machine from file $(nn.fileprefix) for regressionwindow $regrwindow and predict")
    # nntest = loadnn(nn.fileprefix)
    # println(nntest)
    # predtest = predict(nntest, features)
    # @assert pred ≈ predtest  "NN results differ from loaded NN: pred[:, 1:5] = $(pred[:, begin:begin+5]) predtest[:, 1:5] = $(predtest[:, begin:begin+5])"
    println(nn)
end

function sinetest()
    # EnvConfig.init(production)
    startdt = DateTime("2023-02-17T13:30:00")
    enddt = startdt + Day(100) - Minute(1)
    requestedfeatures = vcat(allregressionfeatures(), popularminmaxfeatures(), popularvolumefeatures())
    f5 = Features.Features005(requestedfeatures)
    fdg = Targets.FixedDistanceGain(30, Targets.thresholds((longbuy=0.1, longhold=0.005, shorthold=-0.005, shortbuy=-0.1)))
    features = calcfeatures("SINE", startdt, enddt, f5)
    targets = calctargets(f5, fdg)
    # println("targets=$(fdg.df[1:300, :])")

    println("$(EnvConfig.now()) subset creation")
    setranges = Classify.setpartitions(1:length(targets), Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/13)
    println("setranges: $setranges")
    println("$(EnvConfig.now()) get train features subset")
    trainfeatures = Classify.subsetdim2(features, setranges["train"])
    println("$(EnvConfig.now()) get train targets subset")
    traintargets = Classify.subsetdim2(targets, setranges["train"])

    println("$(EnvConfig.now()) NN model cration")
    nn = Classify.model001(size(trainfeatures, 1), unique(targets), "FixedTimeGain10pct30minutes")
    nn.featuresdescription = Features.describe(f5)
    nn.targetsdescription = Targets.describe(fdg)
    println("$(EnvConfig.now()) adapting machine for targets $(nn.targetsdescription) \nusing features: $(nn.featuresdescription)")
    nn = Classify.adaptnn!(nn, trainfeatures, traintargets)
    println("$(EnvConfig.now()) predicting")
    pred = Classify.predict(nn, features)
    println("$(EnvConfig.now()) predictions to dataframe")
    push!(nn.predictions, Classify.predictionsdataframe(nn, setranges, targets, pred, f5))
    # predictiondistribution(pred, nn.mnemonic)

    println("saving adapted classifier $(nn.fileprefix)")
    # println(nn)
    Classify.savenn(nn)
    return nn
end

function BTCtest()
    # EnvConfig.init(production)
    enddt = DateTime("2024-12-20T22:58:00")
    startdt = enddt - Year(10)
    # startdt = enddt - Month(6)
    requestedfeatures = vcat(allregressionfeatures(), popularminmaxfeatures(), popularvolumefeatures())
    f5 = Features.Features005(requestedfeatures)
    features = calcfeatures("BTC", startdt, enddt, f5)
    for (window, lb) in [(60, 0.005), (60, 0.01), (60, 0.02), (4*60, 0.01), (4*60, 0.02), (4*60, 0.05)]
        fdg = Targets.FixedDistanceGain(window, Targets.thresholds((longbuy=lb, longhold=0.0005, shorthold=-0.0005, shortbuy=-0.01)))
        targets = calctargets(f5, fdg)
        # println("targets=$(fdg.df[1:300, :])")

        println("$(EnvConfig.now()) subset creation")
        setranges = Classify.setpartitions(1:length(targets), ["train", "test", "train", "train", "eval", "train"], partitionsize=20*24*60, gapsize=24*60)
        # setranges = Classify.setpartitions(1:length(targets), Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/13)
        println("setranges: $setranges")
        trainfeatures = Classify.subsetdim2(features, setranges["train"])
        println("$(EnvConfig.now()) got train features subset size=$(size(trainfeatures))")
        traintargets = Classify.subsetdim2(targets, setranges["train"])
        println("$(EnvConfig.now()) got train targets subset size=$(size(traintargets))")
        # trainfeatures, traintargets = oversample(trainfeatures, traintargets)  # all classes are equally trained
        trainfeatures, traintargets = undersample(trainfeatures, traintargets)  # all classes are equally trained
        println("$(EnvConfig.now()) after undersampling train features subset size=$(size(trainfeatures))")
        println("$(EnvConfig.now()) after undersampling train targets subset size=$(size(traintargets)) type=$(typeof(traintargets))")
        print("Training ")
        Targets.labeldistribution(traintargets)

        println("$(EnvConfig.now()) NN model cration")
        nn = Classify.model001(size(trainfeatures, 1), unique(targets), "FixedTimeGain$(round(Int, lb*1000))permille$(window)minutes")
        nn.featuresdescription = Features.describe(f5)
        nn.targetsdescription = Targets.describe(fdg)
        println("$(EnvConfig.now()) adapting machine for targets $(nn.targetsdescription) \nusing features: $(nn.featuresdescription)")
        nn = Classify.adaptnn!(nn, trainfeatures, traintargets)
        println("$(EnvConfig.now()) predicting")
        pred = Classify.predict(nn, features)
        println("$(EnvConfig.now()) predictions to dataframe")
        push!(nn.predictions, Classify.predictionsdataframe(nn, setranges, targets, pred, f5))
        # predictiondistribution(pred, nn.mnemonic)

        println("saving adapted classifier $(nn.fileprefix)")
        # println(nn)
        Classify.savenn(nn)
        evalclassifier(nn)
    end
    return nn
end

function liquidcoinstest()
    """
    hyper parameters:
    - target time window, extend to incease lieklihood of gain reached but be less focused on short term features
    - target gain, reduce gain to increase lieklihood but be more merged with noise
    - class balancing yes/no/factor 
    - how many situations observed, i.e. train on all coins and then zoom in on one
    """
    startdt = nothing # DateTime("2024-03-01T00:00:00")
    enddt =   nothing # DateTime("2024-06-06T09:00:00")
    # coins = ["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"]
    coins = nothing # ["BTC"]

    coinsdf = Ohlcv.liquidcoins()
    filtered_df = coinsdf # filter(row -> row.basecoin in coins, coinsdf)
    println("evaluating: $coins \n coinsdf=$coinsdf \n filtered_df=$filtered_df")
    df = Classify.evaluateclassifiers([Classify.Classifier005], filtered_df, startdt, enddt)
    # df = Classify.readsimulation()
    kpidf, gdf = Classify.kpioverview(df, Classify.Classifier005)
    sort!(kpidf, [:gain_sum], rev=true)
    println(kpidf)
    # println(gdf[kpidf[1, :groupindex]])
    # println(gdf[kpidf[2, :groupindex]])
end

EnvConfig.init(production)
# EnvConfig.init(test)
# EnvConfig.init(training)
Ohlcv.verbosity = 3
CryptoXch.verbosity = 3
Features.verbosity = 3
Targets.verbosity = 2
EnvConfig.verbosity = 3
Classify.verbosity = 3

EnvConfig.setlogpath("2452-FixedTimeGain001")
nn = sinetest()
# nn = Classify.loadnn("NNFixedTimeGain10pct30minutes_24-12-20_23-03-26_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb")

# nn = BTCtest()

# EnvConfig.setlogpath("2445-FixedTimeGain001")
# nn = Classify.loadnn("NNFixedTimeGain10pct30minutes_24-12-22_01-48-26_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # class balancing by undersampling - best generalization
# nn = Classify.loadnn("NNFixedTimeGain10pct30minutes_24-12-21_00-27-30_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # class balancing by oversampling
# nn = Classify.loadnn("NNFixedTimeGain10pct30minutes_24-12-21_21-53-16_gitSHA-083e1b7c51352cfd06775b0426632796d5e881eb") # no class balancing
# evalclassifier(nn)

println("done")
end # of FixedTimeGain001
