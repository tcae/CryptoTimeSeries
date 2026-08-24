
# Known issues

- always set a OCO bracket order for an open amount in trade!
- trade! to handle rejects (FMEA)
- exit program on liquidation
- **l_status also to set to reject if this happens
- The SOLUDT open is smooth but I still see jumps. See attached screenshot from a recent detailed BTCUSDT graph
/Users/torsten/Library/Application Support/Code/User/workspaceStorage/vscode-chat-images/image-1787490346920.png
- websocket close completion and open issueing of an order on async websocket basis is not yet done
	- What I have not done: true sub-minute, websocket-event-triggered immediate resubmission of the open order the instant a close-fill event arrives (fully avoiding the "wait 	    another minute" for live KrakenSpot/KrakenFutures trading). That requires a background async callback that submits live orders outside the normal per-minute tick loop — a materially riskier change (concurrent order submission touching real capital, hard to test without live exchange access). Given the current fix already guarantees correctness (never opens before the close resolves, checked via websocket-first data), it only costs at most one extra tick of latency versus true event-driven immediacy. I'd like to discuss the design (where to store pending-successor order specs, how to avoid double-submission races with the normal tick loop) before implementing that piece — want me to 	proceed with a concrete design proposal?
- implement OCO bracket order instead of own timing coordination with timeing gaps
	- please check for KrakenSpot and KrakenFutures whether this can be mapped to an OCO bracket order API call, which would push the timing to teh exchange side  
	- I checked both APIs directly against Kraken's docs. Findings differ significantly between the two:
	- KrakenSpot — yes, native OCO/bracket support exists. AddOrder accepts close[ordertype], close[price], close[price2] parameters on the same request as the entry order. Per Kraken's docs: "Conditional close orders are triggered by execution of the primary order in the same quantity and opposite direction, but once triggered are independent orders." This means:Submit the entry (buy/sell) with close[ordertype]=stop-loss, close[price]=<stop price> in the same AddOrder call. Kraken automatically fires the stop-loss the instant the entry fills — no polling, no "is the position filled yet" race, no need for our own _ensurestoploss! deferral logic at all. This completely sidesteps the exact bug we just fixed (submitting a reduce-only close before the position exists) — the exchange itself guarantees correct sequencing. Currently KrakenSpot.createorder/_addorderparams don't set any close[...] params — this would be new wiring, not something already there.
	- KrakenFutures — no native bracket/OCO. SendOrder only supports standalone conditional order types (stp, take_profit, trailing_stop) with their own triggerPrice/triggerSignal/reduceOnly, each submitted as an independent order — there's no parameter linking a stop to a specific entry order the way Spot's close[...] does. We'd still need our own "submit once entry is confirmed filled" logic for Futures (i.e., keep the _ensurestoploss! approach we just built), though a Futures stop order being a dormant trigger (not an immediate reduce-only action) means it may tolerate being placed slightly earlier than Spot's model requires — worth confirming empirically before relying on it.
	- Want me to implement the native close[...] bracket for KrakenSpot's open-order path (so Spot pushes stop timing to the exchange), while leaving KrakenFutures on the current post-fill _ensurestoploss! mechanism?
- Known gap: the generic KrakenSpot/KrakenFutures fallback uses the current bar's close price as an approximation of the real liquidation fill price, since getting the exact fill would require querying exchange trade-history endpoints (not attempted — I didn't have enough certainty about the exact Kraken API response shapes to implement that safely for live trading). Noted in memory for a future, more precise pass if needed.
- check memory usage, which seems at least with tradesim very high
	- memory break down
	- is classifier loaded once or per symbol?
	- what is the length of OHLCV history that is maintained in memeory?
- multiple orders per symbol
- frequent order change
- what is the log semantic of opened/closed? Is it order placed or order filled?
- why are so many warnings? I don't believe those are all race conditions
- liquidity check too relaxed?
- Boundsestimator does not adhere to teh rangeid numbering scheme that applies 10000 steps to liquidity range and subranges to sets ranges within teh liquidity range

# To be observed

- positions shall apply to constraints as documented in docs/tradereal-risk-constraints-overview-2026-06-03.md

# intent

- max 1 long and 1 short order per symbol

# to be validated

- implemented strategy shall be robust against multi day trend changes, i.e. a short trend shall not significantly reduce equity 

# test use cases

- close and opposite open in same minute
- extreme volatility forcing stop loss and liquidation
- gradual losses shall result in gradual reduction of corresponding position but as hysteresis to avoid high frequent buy/sell trades resulting in significant fees
- tradesim and TSM reporting shall be able to consider or ignore pair, set, rangeid grouping but should consider all pairs per minute processing
- stop loss adaptation every time a new open portion is added or if the close limit is adapted
- trade! to handle **l_rejects: what are failure modes and appropriate reactions
