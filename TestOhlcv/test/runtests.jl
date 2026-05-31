module TestOhlcvTest

using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, TestOhlcv

ohlc = TestOhlcv.testohlcv(
    "SINE", Dates.DateTime("2025-01-02T01:11", dateformat"yyyy-mm-ddTHH:MM"),
    Dates.DateTime("2025-01-02T01:42", dateformat"yyyy-mm-ddTHH:MM"), "1m")
df = Ohlcv.dataframe(ohlc)


@testset begin
@test size(df) == (31, 7)

end # testset

end # of TestOhlcvTest
