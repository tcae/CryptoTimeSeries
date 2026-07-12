module XchIssueLoggingTest
using Test
using EnvConfig, Xch

@testset "Xch issue logging" begin
    EnvConfig.init(test)
    xc = Xch.XchCache()

    msg1 = Xch.log_trading_issue(xc, "Trading", "insufficient funds for buy order")
    @test msg1 == "insufficient funds for buy order"

    msg2 = Xch.log_trading_issue(xc, Xch.exchange(xc), "order not found")
    @test msg2 == "order not found"
end

end