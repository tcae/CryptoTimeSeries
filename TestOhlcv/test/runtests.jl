module TestOhlcvTest

using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, TestOhlcv

ohlc = TestOhlcv.testohlcv(
    "SINE", Dates.DateTime("2022-01-01T00:00", dateformat"yyyy-mm-ddTHH:MM"),
    Dates.DateTime("2022-01-01T00:31", dateformat"yyyy-mm-ddTHH:MM"), "1m")
df = Ohlcv.dataframe(ohlc)


@testset begin
@test size(df) == (32, 7)

end # testset

end # of TestOhlcvTest
