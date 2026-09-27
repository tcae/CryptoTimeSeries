using Test
using Xch

@testset "IbkrSpot Xch adapter registration" begin
    adapter = Xch._adaptercache(Xch.EXCHANGE_IBKRSPOT)
    @test Xch.exchange(Xch.XchCache(bc=adapter)) == "IbkrSpot"
    @test Xch._defaultquote(Xch.EXCHANGE_IBKRSPOT) == "USD"
end