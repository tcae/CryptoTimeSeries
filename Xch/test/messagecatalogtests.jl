module XchMessageCatalogTest
using Test
using JSON3
using EnvConfig, Xch

@testset "Xch message catalogs" begin
    root = mktempdir()
    oldroot = EnvConfig.coinspath()
    EnvConfig.setcoinspath!(root)
    try
        mkpath(joinpath(root, "Trading"))
        mkpath(joinpath(root, "bybit"))
        write(joinpath(root, "Trading", "_errors.json"), """
        {
          "Trading": [
            {"id": 1, "code": "Trading-001", "message": "insufficient funds"},
            {"code": "Trading-002", "message": "private cooldown"}
          ]
        }
        """)
        write(joinpath(root, "bybit", "_errors.json"), """
        {
          "Bybit": [
            {"id": 50, "code": "Bybit-050", "message": "order not found"}
          ]
        }
        """)

        messages = Xch.load_messages(Xch.EXCHANGE_BYBIT; root=root)
        @test haskey(messages, "Trading")
        @test haskey(messages, Xch.EXCHANGE_BYBIT)
        @test messages["Trading"][1].id == UInt8(1)
        @test messages["Trading"][2].id == UInt8(2)
        @test messages[Xch.EXCHANGE_BYBIT][1].id == UInt8(50)

        xc = Xch.XchCache(exchange=Xch.EXCHANGE_BYBIT)
        xc.mc[:message_catalog_root] = root
        xc.messages = messages

        entry = Xch.log_trading_issue(xc, "Trading", "insufficient funds for buy order")
        @test entry.id == UInt8(1)
        @test entry.message == "insufficient funds"

        created = Xch.log_trading_issue(xc, "Trading", "new trading issue for temp test")
        @test created.id == UInt8(3)
        written = JSON3.read(read(joinpath(root, "Trading", "_errors.json"), String))
        @test written["Trading"][3]["message"] == "new trading issue for temp test"

        overflow_root = mktempdir()
        EnvConfig.setcoinspath!(overflow_root)
        mkpath(joinpath(overflow_root, "Trading"))
        entries = [Dict("id" => i, "code" => "Trading-$i", "message" => "msg-$i") for i in 1:49]
        write(joinpath(overflow_root, "Trading", "_errors.json"), JSON3.write(Dict("Trading" => entries)))
        overflow_messages = Xch.load_messages(Xch.EXCHANGE_BYBIT; root=overflow_root)
        overflow_xc = Xch.XchCache(exchange=Xch.EXCHANGE_BYBIT)
        overflow_xc.mc[:message_catalog_root] = overflow_root
        overflow_xc.messages = overflow_messages
        @test_throws ErrorException Xch.log_trading_issue(overflow_xc, "Trading", "one more trading issue")
    finally
        EnvConfig.setcoinspath!(oldroot)
    end
end

end