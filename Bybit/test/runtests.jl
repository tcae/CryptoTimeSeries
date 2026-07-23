using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(production)  # test production

@testset "Bybit tests" begin
    bc = Bybit.BybitCache()
    syminfo = Bybit.exchangeinfo(bc)
    @test isa(syminfo, AbstractDataFrame)
    @test size(syminfo, 1) > 100

    @test (Dates.now(UTC) + Dates.Second(15)) > Bybit.servertime(bc) > (Dates.now(UTC) - Dates.Second(15))

    # acc = Bybit.account(bc)
    # @test acc["marginMode"] == "ISOLATED_MARGIN"  broken=true # "REGULAR_MARGIN"
    # @test isa(acc, AbstractDict)
    # @test length(acc) > 1

    syminfo = Bybit.symbolinfo(bc, "BTCUSDT")
    @test isa(syminfo, DataFrameRow)

    dayresult = Bybit.get24h(bc)
    @test isa(dayresult, AbstractDataFrame)
    @test size(dayresult, 1) > 100

    dayresult = Bybit.get24h(bc, "BTCUSDT")
    @test isa(dayresult, DataFrameRow)
    @test length(dayresult) >= 6
    @test all([s in ["askprice", "bidprice", "lastprice", "quotevolume24h", "pricechangepercent", "symbol"] for s in names(dayresult)])
    btcprice = dayresult.lastprice

    klines = Bybit.getklines(bc, "BTCUSDT")
    @test isa(klines, AbstractDataFrame)

    # BybitSim: TestOhlcv symbols must provide klines and support simulated trading.
    bc_sim = Bybit.BybitCache()
    Bybit._init_simulation!(bc_sim)
    Bybit.seedportfolio!(bc_sim, EnvConfig.pairquote, 1_000f0)

    sdt = DateTime("2025-01-05T00:00:00")
    edt = DateTime("2025-01-05T01:00:00")
    sine_klines = Bybit.getklines(bc_sim, "SINEUSDT"; startDateTime=sdt, endDateTime=edt, interval="1m")
    dsine_klines = Bybit.getklines(bc_sim, "DOUBLESINEUSDT"; startDateTime=sdt, endDateTime=edt, interval="1m")
    @test size(sine_klines, 1) > 0
    @test size(dsine_klines, 1) > 0

    o_sine = Bybit.createorder(bc_sim, "SINEUSDT", "Buy", 2.0f0, nothing, false)
    @test !isnothing(o_sine)
    @test o_sine.symbol == "SINEUSDT"
    @test o_sine.status == "Filled"

    sim_balances = Bybit.balances(bc_sim)
    @test any(sim_balances.coin .== "SINE")
    @test sim_balances[sim_balances.coin .== "SINE", :free][1] > 0f0

    sim_capacity = Bybit.accountcapacity(bc_sim)
    @test sim_capacity.available_opening_quote > 0.0
    @test sim_capacity.available_long_quote == sim_capacity.available_opening_quote
    @test sim_capacity.available_short_quote == sim_capacity.available_opening_quote
    @test sim_capacity.equity_quote > sim_capacity.available_opening_quote
    @test sim_capacity.source == "Bybit:sim_wallet"

    flip = Bybit.closebeforeopenflip!(
        bc_sim,
        "SINEUSDT",
        :long,
        0.5f0,
        nothing,
        false,
        false;
        open_basequantity=0.25f0,
        close_reduceonly=true,
        open_reduceonly=false,
    )
    @test !isnothing(flip.closeorderid)
    @test !isnothing(flip.openorderid)
    @test flip.closeorderid.side == "Sell"
    @test flip.openorderid.side == "Sell"

    # Pending maker orders in BybitSim should reserve balances, then fill only when
    # later candle ranges reach the limit price (checked since lastcheck).
    bc_pending = Bybit.BybitCache()
    Bybit._init_simulation!(bc_pending)
    Bybit.seedportfolio!(bc_pending, EnvConfig.pairquote, 1_000f0)
    bc_pending.simtime = DateTime("2025-01-05T00:10:00")

    lastpx = Bybit.get24h(bc_pending, "SINEUSDT").lastprice
    pending_limit = 0.5f0 * lastpx
    o_pending = Bybit.createorder(bc_pending, "SINEUSDT", "Buy", 1.0f0, pending_limit, true)
    @test o_pending.status == "New"
    @test size(Bybit.openorders(bc_pending), 1) == 1

    b_pending = Bybit.balances(bc_pending)
    qix_pending = findfirst(==(uppercase(EnvConfig.pairquote)), String.(b_pending.coin))
    @test !isnothing(qix_pending)
    @test b_pending[qix_pending, :locked] > 0f0

    # The first pending fill check should happen on the very next minute tick.
    bc_pending_fast = Bybit.BybitCache()
    Bybit._init_simulation!(bc_pending_fast)
    Bybit.seedportfolio!(bc_pending_fast, EnvConfig.pairquote, 1_000f0)
    bc_pending_fast.simtime = DateTime("2025-01-05T00:10:00")
    fast_lastpx = Bybit.get24h(bc_pending_fast, "SINEUSDT").lastprice
    fast_pending = Bybit.createorder(bc_pending_fast, "SINEUSDT", "Buy", 1.0f0, 2f0 * fast_lastpx, true)
    @test fast_pending.status == "New"
    bc_pending_fast.simtime = bc_pending_fast.simtime + Minute(1)
    _ = Bybit.balances(bc_pending_fast)
    fast_filled = Bybit.order(bc_pending_fast, String(fast_pending.orderid))
    @test !isnothing(fast_filled)
    @test fast_filled.status == "Filled"
    @test size(Bybit.openorders(bc_pending_fast), 1) == 0

    # Move time forward and amend to a guaranteed trigger level; processing should
    # sweep candles since lastcheck and fill the pending order.
    bc_pending.simtime = bc_pending.simtime + Minute(3)
    amended_pending = Bybit.amendorder(bc_pending, "SINEUSDT", String(o_pending.orderid); limitprice=2f0 * lastpx)
    @test !isnothing(amended_pending)

    bc_pending.simtime = bc_pending.simtime + Minute(3)
    _ = Bybit.balances(bc_pending)
    @test size(Bybit.openorders(bc_pending), 1) == 0

    filled_pending = Bybit.order(bc_pending, String(o_pending.orderid))
    @test !isnothing(filled_pending)
    @test filled_pending.status == "Filled"

    b_after_fill = Bybit.balances(bc_pending)
    qix_after = findfirst(==(uppercase(EnvConfig.pairquote)), String.(b_after_fill.coin))
    six_after = findfirst(==("SINE"), String.(b_after_fill.coin))
    @test !isnothing(qix_after)
    @test !isnothing(six_after)
    @test b_after_fill[qix_after, :locked] == 0f0
    @test b_after_fill[six_after, :free] > 0f0

    # Pending spot-sell maker order should lock base inventory and release it on cancel.
    bc_sell_pending = Bybit.BybitCache()
    Bybit._init_simulation!(bc_sell_pending)
    Bybit.seedportfolio!(bc_sell_pending, EnvConfig.pairquote, 1_000f0)
    Bybit.seedportfolio!(bc_sell_pending, "SINE", 2f0)
    bc_sell_pending.simtime = DateTime("2025-01-05T00:20:00")

    sell_pending = Bybit.createorder(bc_sell_pending, "SINEUSDT", "Sell", 1.5f0, Bybit.get24h(bc_sell_pending, "SINEUSDT").lastprice * 2f0, true; configside=:long)
    @test sell_pending.status == "New"
    b_sell_locked = Bybit.balances(bc_sell_pending)
    six_locked = findfirst(==("SINE"), String.(b_sell_locked.coin))
    @test !isnothing(six_locked)
    @test b_sell_locked[six_locked, :free] == 0.5f0
    @test b_sell_locked[six_locked, :locked] == 1.5f0

    cancelled_sell_oid = Bybit.cancelorder(bc_sell_pending, "SINEUSDT", String(sell_pending.orderid))
    @test cancelled_sell_oid == String(sell_pending.orderid)
    b_sell_released = Bybit.balances(bc_sell_pending)
    six_released = findfirst(==("SINE"), String.(b_sell_released.coin))
    @test !isnothing(six_released)
    @test b_sell_released[six_released, :free] == 2f0
    @test b_sell_released[six_released, :locked] == 0f0

    # Cancel-after-amend: lock deltas should track latest pending order reservation.
    bc_amend_cancel = Bybit.BybitCache()
    Bybit._init_simulation!(bc_amend_cancel)
    Bybit.seedportfolio!(bc_amend_cancel, EnvConfig.pairquote, 1_000f0)
    bc_amend_cancel.simtime = DateTime("2025-01-05T00:25:00")

    buy_pending = Bybit.createorder(bc_amend_cancel, "SINEUSDT", "Buy", 1f0, 100f0, true)
    @test buy_pending.status == "New"
    b_lock0 = Bybit.balances(bc_amend_cancel)
    qix0 = findfirst(==(uppercase(EnvConfig.pairquote)), String.(b_lock0.coin))
    @test !isnothing(qix0)
    @test b_lock0[qix0, :locked] == 100f0

    amended1 = Bybit.amendorder(bc_amend_cancel, "SINEUSDT", String(buy_pending.orderid); basequantity=1f0, limitprice=120f0)
    @test !isnothing(amended1)
    b_lock1 = Bybit.balances(bc_amend_cancel)
    qix1 = findfirst(==(uppercase(EnvConfig.pairquote)), String.(b_lock1.coin))
    @test b_lock1[qix1, :locked] == 120f0

    amended2 = Bybit.amendorder(bc_amend_cancel, "SINEUSDT", String(buy_pending.orderid); basequantity=0.5f0, limitprice=80f0)
    @test !isnothing(amended2)
    b_lock2 = Bybit.balances(bc_amend_cancel)
    qix2 = findfirst(==(uppercase(EnvConfig.pairquote)), String.(b_lock2.coin))
    @test b_lock2[qix2, :locked] == 40f0

    cancelled_buy_oid = Bybit.cancelorder(bc_amend_cancel, "SINEUSDT", String(buy_pending.orderid))
    @test cancelled_buy_oid == String(buy_pending.orderid)
    b_lock_end = Bybit.balances(bc_amend_cancel)
    qix_end = findfirst(==(uppercase(EnvConfig.pairquote)), String.(b_lock_end.coin))
    @test b_lock_end[qix_end, :locked] == 0f0
    @test b_lock_end[qix_end, :free] == 1_000f0

    # Explicit short-open trigger semantics: pending short sell fills on low<=limit.
    bc_short_open = Bybit.BybitCache()
    Bybit._init_simulation!(bc_short_open)
    Bybit.seedportfolio!(bc_short_open, EnvConfig.pairquote, 1_000f0)
    bc_short_open.simtime = DateTime("2025-01-05T00:30:00")
    short_open_limit = Bybit.get24h(bc_short_open, "SINEUSDT").lastprice * 2f0
    so = Bybit.createorder(bc_short_open, "SINEUSDT", "Sell", 0.5f0, short_open_limit, true; configside=:short)
    @test so.status == "New"
    bc_short_open.simtime = bc_short_open.simtime + Minute(2)
    _ = Bybit.balances(bc_short_open)
    so_filled = Bybit.order(bc_short_open, String(so.orderid))
    @test !isnothing(so_filled)
    @test so_filled.status == "Filled"

    # Explicit short-close trigger semantics: pending short buy fills on high>=limit.
    bc_short_close = Bybit.BybitCache()
    Bybit._init_simulation!(bc_short_close)
    Bybit.seedportfolio!(bc_short_close, EnvConfig.pairquote, 1_000f0)
    bc_short_close.simtime = DateTime("2025-01-05T00:40:00")
    opened_short = Bybit.createorder(bc_short_close, "SINEUSDT", "Sell", 0.4f0, nothing, false; configside=:short)
    @test opened_short.status == "Filled"
    short_close_limit = Bybit.get24h(bc_short_close, "SINEUSDT").lastprice * 0.01f0
    sc = Bybit.createorder(bc_short_close, "SINEUSDT", "Buy", 0.4f0, short_close_limit, true; configside=:short, reduceonly=true)
    @test sc.status == "New"
    bc_short_close.simtime = bc_short_close.simtime + Minute(2)
    _ = Bybit.balances(bc_short_close)
    sc_filled = Bybit.order(bc_short_close, String(sc.orderid))
    @test !isnothing(sc_filled)
    @test sc_filled.status == "Filled"

    # Adaptive maker limitprice=nothing should refresh around market spread each amend.
    bc_adaptive = Bybit.BybitCache()
    Bybit._init_simulation!(bc_adaptive)
    Bybit.seedportfolio!(bc_adaptive, EnvConfig.pairquote, 1_000f0)
    bc_adaptive.simtime = DateTime("2025-01-05T00:50:00")
    adaptive = Bybit.createorder(bc_adaptive, "SINEUSDT", "Buy", 0.3f0, nothing, true)
    @test adaptive.status == "New"
    adaptive_oid = String(adaptive.orderid)

    mkt1 = Bybit.get24h(bc_adaptive, "SINEUSDT")
    am1 = Bybit.amendorder(bc_adaptive, "SINEUSDT", adaptive_oid; limitprice=nothing)
    @test !isnothing(am1)
    syminfo_sine = Bybit.symbolinfo(bc_adaptive, "SINEUSDT")
    expected1 = mkt1.askprice - syminfo_sine.ticksize
    @test abs((am1.limitprice) - expected1) <= syminfo_sine.ticksize

    mkt2 = Bybit.get24h(bc_adaptive, "SINEUSDT")
    am2 = Bybit.amendorder(bc_adaptive, "SINEUSDT", adaptive_oid; limitprice=nothing)
    @test !isnothing(am2)
    expected2 = mkt2.askprice - syminfo_sine.ticksize
    @test abs((am2.limitprice) - expected2) <= syminfo_sine.ticksize

    # directsequence! should acknowledge valid pairs and fail fast on invalid chains.
    bc_sequence = Bybit.BybitCache()
    Bybit._init_simulation!(bc_sequence)
    Bybit.seedportfolio!(bc_sequence, EnvConfig.pairquote, 1_000f0)
    bc_sequence.simtime = DateTime("2025-01-05T01:00:00")
    seq_pre = Bybit.createorder(bc_sequence, "SINEUSDT", "Buy", 0.1f0, Bybit.get24h(bc_sequence, "SINEUSDT").lastprice * 0.5f0, true)
    seq_suc = Bybit.createorder(bc_sequence, "SINEUSDT", "Buy", 0.1f0, Bybit.get24h(bc_sequence, "SINEUSDT").lastprice * 0.4f0, true)
    seq_ok = Bybit.directsequence!(bc_sequence, String(seq_pre.orderid), String(seq_suc.orderid))
    @test seq_ok.acknowledged
    @test seq_ok.predecessor_orderid == String(seq_pre.orderid)
    @test seq_ok.successor_orderid == String(seq_suc.orderid)
    @test seq_ok.symbol == "SINEUSDT"

    @test_throws AssertionError Bybit.directsequence!(bc_sequence, "missing-order", String(seq_suc.orderid))

    seq_other = Bybit.createorder(bc_sequence, "DOUBLESINEUSDT", "Buy", 0.1f0, Bybit.get24h(bc_sequence, "DOUBLESINEUSDT").lastprice * 0.5f0, true)
    @test_throws AssertionError Bybit.directsequence!(bc_sequence, String(seq_pre.orderid), String(seq_other.orderid))

    spec_long = Bybit._executionorderspec(:long, "Buy", 0)
    @test spec_long.max_quote > 0
    @test_throws ArgumentError Bybit._enforce_maxquote_policy(spec_long, "SINEUSDT", 10.0, 100.0, false)
    @test_throws ArgumentError Bybit._enforce_maxquote_policy(spec_long, "SINEUSDT", 10.0, 100.0, true)
    @test Bybit._enforce_maxquote_policy(spec_long, "SINEUSDT", 1.0, 100.0, false) === nothing


    # oocreate = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00001, btcprice * 0.9, false)
    # oid = isnothing(oocreate) ? nothing : oocreate.orderid

    # oo = Bybit.order(bc, oid)
    # @test isa(oo, DataFrameRow)
    # @test length(oo) >= 13
    # @test oo.orderid == oid

    # ooamend = Bybit.amendorder(bc, "BTCUSDT", oid; basequantity=0.00011)
    # @test ooamend.orderid == oid

    # ooamend = Bybit.amendorder(bc, "BTCUSDT", oid; limitprice=btcprice * 0.8)
    # @test ooamend.orderid == oid

    # oo = Bybit.openorders(bc)
    # @test isa(oo, AbstractDataFrame)
    # @test (size(oo, 1) > 0)
    # @test (size(oo, 2) >= 13)

    # coid = Bybit.cancelorder(bc, "BTCUSDT", oid)
    # @test coid == oid

    # oo = Bybit.order(bc, oid)
    # @test oo.status == "Cancelled"

    # wb = Bybit.balances(bc)
    # @test isa(wb, AbstractDataFrame)
    # @test size(wb, 2) == 3

end
