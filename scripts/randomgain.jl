using Ohlcv, Features

"""
The idea of this experiment is to leverage the random price changes around a regression line and not to follow the changes to detect a peak.
If the amplitude of such changes is large enough to get positive yield then it can be a viable trading strategy to buy x% blow the regression line and sell x% above it,
assuming a symetric distribution, which is likely because that's the urpose a of regression line.

Problems:

- what trading distance to choose?
- how to recognize that the assumption is invalid, e.g. massive sell, to stop losses and end trading?
- how to recognize a massive buy to buy and let run?
"""

module RandomGain

using Ohlcv, Features

"provides Ohlcv data and corresoonding regressions"
struct OhlcvRegr
    base::String
    ohlcv::Ohlcv.OhlcvData
    regr::Dict  # key = integer of number of regression window data points, value = vector
end

struct Measures

end

struct Thresholds

end

function getohlcvregressions(base)

end

"""
- ohlcv and regr have to have the same length and belong to the same asset.
- trainset is an integer vector of indices that are used from ohlcv and regr to measure distribution properties
- returns RGmeasure that is subsequently used for evaluation or trading execution
- shall measure the std deviation from regression of recent past of the last regrwindow data points and shall correlate this with the next regrwindow data points

Maesure means:

- how often is a distance touched after reversing from the other side of the regession line?
- how is looking back and forward correlated in this respect?
- is the standard deviation also correlated, i.e. can the std dev used to set distance thresholds for trading?
"""
function measure(rgset, trainset, regrwindow)::Measures

end

"""
Determines the trading thresholds based on measurements, e.g. distances to buy at low and sell at high
"""
function thresholds(rgm)::Thresholds

end

end

import .RandomGain: getohlcvregressions, measure, thresholds, evaluate

bases = EnvConfig.trainingbases
for base in bases
    rgset = getohlcvregressions(base)
    xcv = 6  # xcv = x * cross validation
    cvset = RandomGain.CV(6)
    for cv in 1:xcv
        rgm = measure(rgset, cvset[cv].train, 6)
        thrs = thresholds(rgm)
        traineval = evaluate(rgset, thrs, cvset[cv].train)
        testeval = evaluate(rgset, thrs, cvset[cv].test)
    end
end
