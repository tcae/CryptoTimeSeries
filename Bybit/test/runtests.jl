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
    processed_dt = bc_pending_fast.lastpendingdecisiondt
    Bybit._simprocesspendingorders!(bc_pending_fast)
    @test bc_pending_fast.lastpendingdecisiondt == processed_dt == bc_pending_fast.simtime
    @test Bybit.order(bc_pending_fast, String(fast_pending.orderid)).status == "Filled"
    @test size(Bybit.openorders(bc_pending_fast), 1) == 0
    @test size(bc_pending_fast.orderbook, 1) == 1
    @test Bybit._sim_orderindex_for(bc_pending_fast)[String(fast_pending.orderid)] == 1

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

    # Explicit short-open trigger semantics: a non-stop sell only fills once price rises to the limit.
    bc_short_open = Bybit.BybitCache()
    Bybit._init_simulation!(bc_short_open)
    Bybit.seedportfolio!(bc_short_open, EnvConfig.pairquote, 1_000f0)
    bc_short_open.simtime = DateTime("2025-01-05T00:30:00")
    short_open_limit = Bybit.get24h(bc_short_open, "SINEUSDT").lastprice * 2f0
    so = Bybit.createorder(bc_short_open, "SINEUSDT", "Sell", 0.5f0, short_open_limit, true; configside=:short)
    @test so.status == "New"
    @test String(so.lane) == "so"
    bc_short_open.simtime = bc_short_open.simtime + Minute(2)
    _ = Bybit.balances(bc_short_open)
    so_filled = Bybit.order(bc_short_open, String(so.orderid))
    @test !isnothing(so_filled)
    @test so_filled.status == "New"
    short_open_balances = Bybit.balances(bc_short_open)
    quoteix = findfirst(==(EnvConfig.pairquote), short_open_balances[!, :coin])
    @test !isnothing(quoteix)
    short_open_cap = Bybit.accountcapacity(bc_short_open)
    posdf = Bybit.positionsnapshot(bc_short_open)
    pos_row = findfirst(==("SINE"), uppercase.(String.(posdf[!, :coin])))
    wallet_quote = short_open_balances[quoteix, :free] + short_open_balances[quoteix, :locked]
    if !isnothing(pos_row)
        short_qty = posdf[pos_row, :short_qty]
        short_price = Bybit.get24h(bc_short_open, "SINEUSDT").lastprice
        expected_equity = wallet_quote + (0f0 - short_qty * short_price)
        @test isapprox(short_open_cap.equity_quote, expected_equity; atol=1f-4)
    else
        @test isapprox(short_open_cap.equity_quote, wallet_quote; atol=1f-4)
    end
    @test short_open_balances[quoteix, :free] <= short_open_cap.equity_quote + 1f-6
    @test short_open_cap.available_opening_quote <= short_open_cap.equity_quote + 1f-6

    # A short must conserve account value: opening one leaves equity flat, holding it tracks
    # price, and closing it realizes exactly qty*(entry-exit) into free quote. Before the
    # sale proceeds were credited at open, the buyback at close spent cash the account had
    # never received, so every short permanently destroyed its own notional.
    bc_short_pnl = Bybit.BybitCache()
    Bybit._init_simulation!(bc_short_pnl)
    Bybit.seedportfolio!(bc_short_pnl, EnvConfig.pairquote, 1_000f0)
    pnl_start = DateTime("2025-01-27T20:55:00")
    bc_short_pnl.simtime = pnl_start
    pnl_qty = 125f0
    pnl_entry = Bybit._simcurrentprice(bc_short_pnl, "DOUBLESINEUSDT", pnl_start)
    @test !isnothing(pnl_entry)
    Bybit.createorder(bc_short_pnl, "DOUBLESINEUSDT", "Sell", pnl_qty, pnl_entry, false; configside=:short)
    bc_short_pnl.simtime = pnl_start + Minute(1)
    _ = Bybit.balances(bc_short_pnl)
    cap_open = Bybit.accountcapacity(bc_short_pnl)
    @test isapprox(cap_open.equity_quote, 1_000f0; atol=0.2f0)  # flat at open, only the 1-minute price tick moves it

    bc_short_pnl.simtime = pnl_start + Minute(45)
    _ = Bybit.balances(bc_short_pnl)
    pnl_exit = Bybit._simcurrentprice(bc_short_pnl, "DOUBLESINEUSDT", bc_short_pnl.simtime)
    @test pnl_exit < pnl_entry  # the short is in profit here
    cap_held = Bybit.accountcapacity(bc_short_pnl)
    @test isapprox(cap_held.equity_quote, 1_000f0 + pnl_qty * (pnl_entry - pnl_exit); atol=0.5f0)

    Bybit.createorder(bc_short_pnl, "DOUBLESINEUSDT", "Buy", pnl_qty, pnl_exit, false; configside=:short, reduceonly=true)
    bc_short_pnl.simtime = bc_short_pnl.simtime + Minute(1)
    _ = Bybit.balances(bc_short_pnl)
    cap_closed = Bybit.accountcapacity(bc_short_pnl)
    expected_pnl_equity = 1_000f0 + pnl_qty * (pnl_entry - pnl_exit)
    @test isapprox(cap_closed.equity_quote, expected_pnl_equity; atol=0.5f0)
    pnl_balances = Bybit.balances(bc_short_pnl)
    pnl_quoteix = findfirst(==(uppercase(EnvConfig.pairquote)), uppercase.(String.(pnl_balances[!, :coin])))
    @test !isnothing(pnl_quoteix)
    # the gain is realized as free cash, not left stranded in locked collateral
    @test isapprox(pnl_balances[pnl_quoteix, :free], expected_pnl_equity; atol=1f-1)
    @test isapprox(pnl_balances[pnl_quoteix, :locked], 0f0; atol=1f-3)

    # Explicit short-close trigger semantics: a non-stop buy only fills once price falls to the limit.
    bc_short_close = Bybit.BybitCache()
    Bybit._init_simulation!(bc_short_close)
    Bybit.seedportfolio!(bc_short_close, EnvConfig.pairquote, 1_000f0)
    bc_short_close.simtime = DateTime("2025-01-05T00:40:00")
    opened_short = Bybit.createorder(bc_short_close, "SINEUSDT", "Sell", 0.4f0, nothing, false; configside=:short)
    @test opened_short.status == "Filled"
    short_close_limit = Bybit.get24h(bc_short_close, "SINEUSDT").lastprice * 0.01f0
    sc = Bybit.createorder(bc_short_close, "SINEUSDT", "Buy", 0.4f0, short_close_limit, true; configside=:short, reduceonly=true)
    @test sc.status == "New"
    @test String(sc.lane) == "sc"
    bc_short_close.simtime = bc_short_close.simtime + Minute(2)
    _ = Bybit.balances(bc_short_close)
    sc_filled = Bybit.order(bc_short_close, String(sc.orderid))
    @test !isnothing(sc_filled)
    @test sc_filled.status == "New"

    # A short stop-loss (lane scsl, buy priced above market) waits for an adverse rise.
    bc_short_stoploss = Bybit.BybitCache()
    Bybit._init_simulation!(bc_short_stoploss)
    Bybit.seedportfolio!(bc_short_stoploss, EnvConfig.pairquote, 1_000f0)
    bc_short_stoploss.simtime = DateTime("2025-01-05T00:40:00")
    opened_short_sl = Bybit.createorder(bc_short_stoploss, "SINEUSDT", "Sell", 0.4f0, nothing, false; configside=:short)
    @test opened_short_sl.status == "Filled"
    short_stop_limit = Bybit.get24h(bc_short_stoploss, "SINEUSDT").lastprice * 2f0
    ss = Bybit.createorder(bc_short_stoploss, "SINEUSDT", "Buy", 0.4f0, short_stop_limit, true; configside=:short, reduceonly=true, lane="scsl")
    @test ss.status == "New"
    @test String(ss.lane) == "scsl"
    bc_short_stoploss.simtime = bc_short_stoploss.simtime + Minute(2)
    _ = Bybit.balances(bc_short_stoploss)
    ss_pending = Bybit.order(bc_short_stoploss, String(ss.orderid))
    @test !isnothing(ss_pending)
    @test ss_pending.status == "New"

    # A long stop-loss (lane lcsl, sell priced below market) must not fill immediately like a
    # marketable order; it should only trigger once price actually falls to reach it.
    bc_long_stoploss = Bybit.BybitCache()
    Bybit._init_simulation!(bc_long_stoploss)
    Bybit.seedportfolio!(bc_long_stoploss, EnvConfig.pairquote, 1_000f0)
    bc_long_stoploss.simtime = DateTime("2025-01-05T00:45:00")
    opened_long = Bybit.createorder(bc_long_stoploss, "SINEUSDT", "Buy", 0.4f0, nothing, false)
    @test opened_long.status == "Filled"
    stoploss_limit = Bybit.get24h(bc_long_stoploss, "SINEUSDT").lastprice * 0.5f0
    ls = Bybit.createorder(bc_long_stoploss, "SINEUSDT", "Sell", 0.4f0, stoploss_limit, true; configside=:long, reduceonly=true, lane="lcsl")
    @test ls.status == "New"
    @test String(ls.lane) == "lcsl"
    bc_long_stoploss.simtime = bc_long_stoploss.simtime + Minute(2)
    _ = Bybit.balances(bc_long_stoploss)
    ls_pending = Bybit.order(bc_long_stoploss, String(ls.orderid))
    @test !isnothing(ls_pending)
    @test ls_pending.status == "New"

    # A long take-profit (reduce-only sell priced above market at creation) keeps the
    # standard direction and still fills once price rises to reach it.
    bc_long_takeprofit = Bybit.BybitCache()
    Bybit._init_simulation!(bc_long_takeprofit)
    Bybit.seedportfolio!(bc_long_takeprofit, EnvConfig.pairquote, 1_000f0)
    bc_long_takeprofit.simtime = DateTime("2025-01-05T00:45:00")
    opened_long_tp = Bybit.createorder(bc_long_takeprofit, "SINEUSDT", "Buy", 0.4f0, nothing, false)
    @test opened_long_tp.status == "Filled"
    lt = Bybit.createorder(bc_long_takeprofit, "SINEUSDT", "Sell", 0.4f0, Bybit.get24h(bc_long_takeprofit, "SINEUSDT").lastprice * 100f0, true; configside=:long, reduceonly=true)
    @test lt.status == "New"

    # A close bracket reserves the position once: both legs cover the same quantity.
    bc_bracket = Bybit.BybitCache()
    Bybit._init_simulation!(bc_bracket)
    Bybit.seedportfolio!(bc_bracket, EnvConfig.pairquote, 1_000f0)
    bc_bracket.simtime = DateTime("2025-01-05T00:45:00")
    br_open = Bybit.createorder(bc_bracket, "SINEUSDT", "Buy", 0.4f0, nothing, false)
    @test br_open.status == "Filled"
    br_price = Bybit.get24h(bc_bracket, "SINEUSDT").lastprice
    br_tp = Bybit.createorder(bc_bracket, "SINEUSDT", "Sell", 0.4f0, br_price * 100f0, true; configside=:long, reduceonly=true, lane="lc")
    br_sl = Bybit.createorder(bc_bracket, "SINEUSDT", "Sell", 0.4f0, br_price * 0.5f0, true; configside=:long, reduceonly=true, lane="lcsl")
    @test br_tp.status == "New"
    @test br_sl.status == "New"
    br_assets = bc_bracket.assets
    br_pix = findfirst((br_assets[!, :coin] .== "SINE") .& (br_assets[!, :side] .== "long"))
    @test !isnothing(br_pix)
    # Without sharing, two reduce-only legs would have locked 0.8 of a 0.4 position.
    @test br_assets[br_pix, :locked] == 0.4f0
    @test br_assets[br_pix, :free] == 0f0

    # Cancelling one leg leaves the shared reservation with the surviving leg.
    Bybit.cancelorder(bc_bracket, "SINEUSDT", String(br_tp.orderid))
    @test br_assets[br_pix, :locked] == 0.4f0
    Bybit.cancelorder(bc_bracket, "SINEUSDT", String(br_sl.orderid))
    @test br_assets[br_pix, :locked] == 0f0
    @test br_assets[br_pix, :free] == 0.4f0

    # When one bracket leg fills, its sibling is cancelled rather than left resting.
    bc_oco = Bybit.BybitCache()
    Bybit._init_simulation!(bc_oco)
    Bybit.seedportfolio!(bc_oco, EnvConfig.pairquote, 1_000f0)
    bc_oco.simtime = DateTime("2025-01-05T00:45:00")
    oco_open = Bybit.createorder(bc_oco, "SINEUSDT", "Buy", 0.4f0, nothing, false)
    @test oco_open.status == "Filled"
    oco_price = Bybit.get24h(bc_oco, "SINEUSDT").lastprice
    # take-profit just below market fills on the next candle, stop far away stays untriggered
    oco_tp = Bybit.createorder(bc_oco, "SINEUSDT", "Sell", 0.4f0, oco_price * 0.9f0, true; configside=:long, reduceonly=true, lane="lc")
    oco_sl = Bybit.createorder(bc_oco, "SINEUSDT", "Sell", 0.4f0, oco_price * 0.5f0, true; configside=:long, reduceonly=true, lane="lcsl")
    bc_oco.simtime = bc_oco.simtime + Minute(2)
    _ = Bybit.balances(bc_oco)
    oco_tp_after = Bybit.order(bc_oco, String(oco_tp.orderid))
    oco_sl_after = Bybit.order(bc_oco, String(oco_sl.orderid))
    @test oco_tp_after.status == "Filled"
    @test oco_sl_after.status == "Cancelled"
    @test String(oco_sl_after.rejectreason) == "bracket sibling filled"

    # Both legs triggering in the same candle resolves to the protective stop.
    bc_tie = Bybit.BybitCache()
    Bybit._init_simulation!(bc_tie)
    Bybit.seedportfolio!(bc_tie, EnvConfig.pairquote, 1_000f0)
    bc_tie.simtime = DateTime("2025-01-05T00:45:00")
    tie_open = Bybit.createorder(bc_tie, "SINEUSDT", "Buy", 0.4f0, nothing, false)
    @test tie_open.status == "Filled"
    tie_price = Bybit.get24h(bc_tie, "SINEUSDT").lastprice
    # both priced so the very next candle reaches them: sell tp below market, stop above low
    tie_tp = Bybit.createorder(bc_tie, "SINEUSDT", "Sell", 0.4f0, tie_price * 0.9f0, true; configside=:long, reduceonly=true, lane="lc")
    tie_sl = Bybit.createorder(bc_tie, "SINEUSDT", "Sell", 0.4f0, tie_price * 1.1f0, true; configside=:long, reduceonly=true, lane="lcsl")
    bc_tie.simtime = bc_tie.simtime + Minute(2)
    _ = Bybit.balances(bc_tie)
    tie_tp_after = Bybit.order(bc_tie, String(tie_tp.orderid))
    tie_sl_after = Bybit.order(bc_tie, String(tie_sl.orderid))
    @test tie_sl_after.status == "Filled"
    @test tie_tp_after.status == "Cancelled"

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

    # Margin-call liquidation must cancel the stale pending close/stop-loss order (it never
    # filled at its own price) and record the forced close as a separate "Filled" order plus
    # a queued liquidation event, so Xch/TSM sync can reflect lc_status=cancelled alongside
    # lcl_status=closed/lcl_pavg/lcl_filled for the same position.
    bc_liquidate = Bybit.BybitCache()
    Bybit._init_simulation!(bc_liquidate)
    Bybit.seedportfolio!(bc_liquidate, EnvConfig.pairquote, 1_000f0)
    bc_liquidate.simtime = DateTime("2025-01-05T00:45:00")
    opened_liquidate = Bybit.createorder(bc_liquidate, "SINEUSDT", "Sell", 0.4f0, nothing, false; configside=:short)
    @test opened_liquidate.status == "Filled"
    liquidate_close_limit = Bybit.get24h(bc_liquidate, "SINEUSDT").lastprice * 0.01f0
    pending_close = Bybit.createorder(bc_liquidate, "SINEUSDT", "Buy", 0.4f0, liquidate_close_limit, true; configside=:short, reduceonly=true)
    @test pending_close.status == "New"
    pending_close_oid = String(pending_close.orderid)

    liquidate_price = Bybit.get24h(bc_liquidate, "SINEUSDT").lastprice
    liquidated = Bybit._simforceliquidateposition!(bc_liquidate, "SINEUSDT", :short, liquidate_price, bc_liquidate.simtime)
    @test liquidated

    liquidated_balances = Bybit.balances(bc_liquidate)
    sineix_liquidated = findfirst((liquidated_balances.coin .== "SINE") .& (liquidated_balances.side .== "short"))
    @test !isnothing(sineix_liquidated)
    @test liquidated_balances[sineix_liquidated, :free] == 0f0
    @test liquidated_balances[sineix_liquidated, :locked] == 0f0

    @test size(Bybit.openorders(bc_liquidate; orderid=pending_close_oid), 1) == 0
    cancelled_pending = Bybit.order(bc_liquidate, pending_close_oid)
    @test !isnothing(cancelled_pending)
    @test cancelled_pending.status == "Cancelled"

    liquidation_events = Bybit.drainliquidations!(bc_liquidate)
    @test length(liquidation_events) == 1
    liquidation_event = liquidation_events[1]
    @test liquidation_event.symbol == "SINEUSDT"
    @test liquidation_event.positionside == :short
    @test liquidation_event.qty == 0.4f0
    @test liquidation_event.price == Float32(liquidate_price)
    @test liquidation_event.hadpendingorder
    @test liquidation_event.reason == "liquidation"
    @test isempty(Bybit.drainliquidations!(bc_liquidate))

end
