"""
Async programming is advantegous to deal with Binance because IO is running in different tasks and
seperating order monitoring from OHLCV updates is a cleaner design as well.

Threads are advantegous to train and evaluate independent classifiers in parallel by using all CPU cores
as training data is the same and accessed read only.

ParallelGist will test the core concepts in a stripped down approach to understand shortcomings of the
implementation.
"""
module ParallelGist
# import Pkg; Pkg.add(["Dash", "DashCoreComponents", "DashHtmlComponents", "DashTable"])

end  # ParallelGist

