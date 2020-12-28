using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/features.jl")

module Targets

import Pkg; Pkg.add(["JDF", "RollingFunctions"])
using JDF, Dates, CSV, DataFrames
using ..Config

"""
    Targets are based on regressioin info, i.e. the gain between 2 horizontal regressions with at least 1% gain.
    However the underlining trend can already start before the horizontal regression is reached if the regressioin window is larger to smooth high frequent volatility.
    Therefore BUY shall be signaled before the minimum is reached as soon as the regression is monotonically increasing through the minimum.
    Correspondingly SELL shall be signaled before the maximum is reached as soon as the regression is monotonically decreasing through the maximum.

    Another aspect is that a signal label should represent an observable pattern, i.e. not solely based on a future pattern.
"""
function targets(prices, regressions)

end

end  # module
