using Dates


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

struct A
    a
    b
    A(b) = new(2, b)
end

a1 = A(5)

datetime2epoch(::Type{Millisecond}, x::DateTime) = Millisecond(Dates.value(x) - Dates.UNIXEPOCH)
datetime2epoch(::Type{Second}, x::DateTime) = Dates.value(datetime2epoch(Millisecond, x)) / 1000
datetime2epoch(::Type{T}, x::DateTime) where {T <: TimePeriod} = datetime2epoch(Millisecond, x) |> T

dt = DateTime("2020-06-01T00:01:02.693", DateFormat("yyyy-mm-ddTHH:MM:SS.sssZ"))
println(Int64(floor(datetime2epoch(Second, dt))))  # float - seconds can be noninteger
# 1.590969662693e9
println(Dates.value(datetime2epoch(Millisecond, dt)))  # Millisecond and smaller give the corresponding period type
println(datetime2epoch(Nanosecond, dt))  # Millisecond and smaller give the corresponding period type
# 1590969662693000000 nanoseconds
