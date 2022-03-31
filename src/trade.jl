# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

include("../src/assets.jl")
include("../src/classify.jl")

"""
## problem statement

This module shall automatically follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.

Challenges:
- what is an uptrend, considering that we have more or less significant distortions during an uptrend? Once determined, history data need to get corresponding **`targetlabels`**
- when does a profitable uptrend starts?
- when does an uptrend ends? Independent whether or not it is a profitable uptrend we have to limit losses
- are there uptrends per granularity, e.g. regression window?
- are there cluster patterns (may be per granularity), e.g. fast rising peak, slow crawling increase, ..
- what are the most profitable currencies and how to split asset allocation?
- how to adapt the parameters of the trading strategy based on history data? **`trainstrategy`**
- how to evaluate the success of the trading strategy? **`assessstrategy`**
- how to learn from history concerning patterns and learning strategies?
    + visualize / report an overall status
    + in situations that need to be changed
        - understand what happened
        - how probable are comparable situations
- what is an efficient SW design for the **`tradeloop`**

All history data will be collected but a fixed subset **`historysubset`** will be used for training, evaluation and test. Such data is OHLCV data of a fixed set of crypto currencies that have proven to show sufficient liquidity.
"""
module Trade

using Dates, DataFrames
using ..EnvConfig, ..Ohlcv, ..Classify, ..CryptoXch, ..Assets

# using ..Binance

import JSON
counter = 0

function cb(data)
    global counter
    println("called")
    display(data)
    counter += 1
    counter < 10 ? true : false
end

"""
"""
function gettrainingohlcv(trainingbases=EnvConfig.trainingbases)
    splitdf = Ohlcv.setsplit()
    startdt = minimum(splitdf.start)
    enddt = maximum(splitdf.end)
    println("training bases: $(trainingbases)")
    for base in trainingbases
        println("$(Dates.now()): loading $base from exchange from $startdt until $enddt")
        df = CryptoXch.gethistoryohlcv(base, startdt, enddt)
        println("$(Dates.now()): saving $base")
        ohlcv = Ohlcv.defaultohlcv(base)
        Ohlcv.setdataframe!(ohlcv, df)
        Ohlcv.write(ohlcv)
        println("$(Dates.now()): saved $base from $(df[1, :opentime]) until $(df[end, :opentime])")
    end
end

"""
Returns a fixed subset of history data for one the following purposes:
    - to **`train`** the parameters of a trading strategy
    - to **`evaluate`** the success of a trading strategy to draw conclusions about further training
    - to **`test`** the success of a trading strategy without drawing conclusions concerning training
These subset are non overlapping. Each consists of OHLCV data from a number of contiguous time ranges. Both to avoid cross contamination, which biases the evaluation and test.

"""
function historysubset(setname)
end

"""
Assesses the success of a strategy measured against a given history OHLCV/feature dataset.
Features will always be created for the whole dataset before splitting into non overlapping subsets.
"""
function assessstrategy()
end

"""
Trains the parameters of a strategy on the basis of a given training OHLCV/feature dataset.
Features will always be created for the whole dataset before splitting into non overlapping subsets.
"""
function trainstrategy()
end

"""
read the recent ohlcv history to calculate the longest regressions required
"""
function readrecentohlcv(base)
    # the read ohlcv data shall be fomr time to time appended to the historic data
end

"""
**`tradeloop`** has to
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)

"""
function tradeloop()
    assets = Assets.loadassets(minutesperiod=Dates.Week(4))
    # the read ohlcv data shall be fomr time to time appended to the historic data
    readrecentohlcv(base)
end


end  # module