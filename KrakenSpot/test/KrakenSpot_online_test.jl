using EnvConfig, KrakenSpot, Test

# Run online KrakenSpot tests only when explicitly enabled.
# Enable by setting environment variable: KRAKEN_ONLINE_TESTS=true
@testset "KrakenSpot online tests" begin
    enabled = lowercase(get(ENV, "KRAKEN_ONLINE_TESTS", "true")) in ["1", "true", "yes", "on"]
    if !enabled
        @info "Skipping KrakenSpot online tests. Set KRAKEN_ONLINE_TESTS=true to enable."
    else
        EnvConfig.init(EnvConfig.production)
        bc = KrakenSpot.KrakenSpotCache(autoloadexchangeinfo=false)

        balances = KrakenSpot.balances(bc)
        @test size(balances, 2) == 5
        @test all([name in names(balances) for name in ["coin", "locked", "free", "borrowed", "accruedinterest"]])
        @test size(balances, 1) > 0
        println("balances: ", balances)
    end
end
