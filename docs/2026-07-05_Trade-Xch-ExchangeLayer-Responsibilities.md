# Clarification of intended Trade, Xch, Exchange Layer responsibilities

## Summary

Trade should remain the policy and orchestration layer. Xch should remain the normalized execution interface. The exchange layer, meaning the concrete adapters, should own venue-specific mechanics such as quantity correction, minimum and maximum quantity handling, reduce-only mapping, and same-symbol close-before-open sequencing. Small but positive requests that are below venue minimums should be handled as soft rejects in Xch or the exchange layer, while negative amounts and not-tradable pairs remain hard errors.

## Refactor Checklist

- [x] Keep Trade focused on selection, portfolio-level limits, and cross-symbol ordering.
- [~] Quantity validation and correction are functionally complete, but ownership is still shared between Xch (pre-validation/normalization) and adapters (venue-specific rules).
- [x] Treat below-minimum positive amounts as soft rejects instead of Trade errors.
- [x] Move same-symbol close-before-open sequencing into the exchange layer.
- [x] Normalize reduce-only and close intent in the adapters.
- [x] Keep Trade tests focused on policy and intent mapping, not venue math.
- [x] Add exchange-layer tests for quantity correction and flip sequencing. (KrakenFutures sequence progression plus KrakenSpot and BybitSim adapter-level sequencing/quantity-policy coverage are now in place.)

## Implementation Status (2026-07-05)

### Implemented

- Xch API uses explicit open/close intent: `createopenorder` and `createcloseorder`.
- Adapter-specific reduce-only and close handling are wired for KrakenSpot and KrakenFutures.
- KrakenSpot uses native iceberg support for oversized limit orders.
- KrakenFutures uses adapter-managed sequential iceberg splitting and advancement via private order state updates.
- Exchange-layer tests now cover adapter sequencing and quantity-policy paths for KrakenFutures, KrakenSpot, and BybitSim.
- Bybit oversized-order handling is separated as its own path (currently conservative policy branch).
- Side-specific execution configuration is externalized in exchange `execution_config.json` files and consumed by adapters.
- Auth selection supports `purpose` (`trading` / `testing`) and test-mode KrakenFutures credential selection.
- KrakenSpot supports validate-only order submission (`validate=true`) for non-executing processing checks.

### Partially Implemented

- Quantity validation ownership: adapters normalize venue constraints, but Xch still performs validation and normalization before dispatch.
- Endpoint routing ownership: KrakenFutures `derivatives` endpoint can be selected from auth when present; websocket and charts endpoints remain adapter constants.

### Remaining / Clarifications Needed

- Decide final ownership boundary for quantity validation (pure adapter-owned vs shared Xch+adapter).
- If desired, unify KrakenFutures environment endpoint switching for charts and websocket endpoints, not only derivatives REST.
- Complete Bybit adapter-side sequencing implementation (private worker-based sequential oversized execution) if/when API/runtime path is available.
- Expand adapter tests for additional quantity-correction edge-cases (broader min quantity, precision, and max quote boundary matrices) across KrakenSpot, KrakenFutures, and Bybit paths.

## Trade

### Selection of Trading Pairs to be traded

Selects trading pairs according to configuration and selection rules implemented in tradeselection!

- trading pairs of black listed bases are not considered for opening positions, only for reduction of positions
- trading pairs shall have sufficient liquidity
- trading pair bases that are currently owned with an amount above minimum tradable quantity shall be at least able to be reduced

### Allocation of assets to orders

Allocates assets to orders over time, which is implemented by trade!

- considers maxassetfraction of a trading pair as a ratio to all assets expressed in quote
- does not exceed a quote allocation of maxbudgetquote to sum(positions + assets)
- considers the TradingStrategy advice to close and open positions by creating Xch orders
    - uses exchange specific configuration separately for long and short to determine spot versus margin + leverage
    - spreads asset allocation to holdings over time for risk consideration 
- considers the TradingStrategy advice to adapt order limits
- considers trade mode configuration in order creation
    - buysell is the normal trade mode
    - closeonly disables opening trades and only closes existing long/short positions
    - quickexit sells all assets as soon as possible
    - notrade for testing

### Refactor _tradestep! and all called functions including trade! with the following intent:

- work in tradestep through cache.cfg, which should cover all relevant trading pairs
- for a specific trading pair get TradingStrategy.gettradesrow! but reduce the returned tuple to tradesdf + rowix
- provide trade! the Cache, the tradesdf and tradesdf_rowix. All relevant info should be included there
- trade! shall update the order information within the tradesdf row
- eventually trade! shall call Xch with create, amend, close order functions to execute what was noted in teh tradsdf row
- in contrast to previous implementatons orders shall not be closed every minutes by default and then newly created

## Xch

Acts as abstraction layer towards Trade such that Trade allocates assets to trading pairs and provides limits and considers holdings in a uniform way towards all supported exchanges.

- defines the normalized execution contract that Trade calls
- forwards venue-specific order mechanics to the adapters
- exposes normalized reduce-only and order state so Trade does not infer venue behavior from leverage
- distinguishes soft rejects, such as below-minimum positive amounts, from hard errors like negative amounts or non-tradable pairs
- assets (spot trading) and positions (futures and margin trading) both require asset allocation. Xch shall unify these representation in holdings such that Trade has only a view on the holdings and does not need to distinguish between futures, margin, and spot

## Exchange Layer: Kraken Spot, Kraken Futures, Bybit (spot?)

Receive through Xch directions via orders to open or close holdings (assets or positions). They shall handle internally the following tasks.

- in case of parallel close and open order for positions towards opposite trends, ensure to first close and then open as quick as possible, i.e. within the async web socket handling
- change order quantities to meet quantity resolution or minimum / maximum quantity constraints or reject orders if quantities are too small
- use exchange-specific JSON config files to define
    - which instruments to use for long and short trades, e.g. spot versus margin + leverage
    - the maximum quote amount allowed per order for each exchange order side
    - if a Trade order is larger than that maximum, split it into multiple maximum-size orders and execute them one after another in the adapter websocket as an iceberg-style sequence
    - remove leverage parameters from order-creation functions once the side-specific maximum-order handling is in place
- break up large orders into an iceberg order sequence of maximum quote amount orders, which shall be executed as quick as possible, i.e. within the async web socket handling
- resolve positions below tradable quantity autonomously
    - don't report them back as holdings
    - resolve them under the hood without involving Trade e.g. by buying a minimum amount and close the combined amount
- when a requested amount is positive but below venue minimum, return a soft reject and log the event instead of raising an error to Trade
- when a requested amount is negative or the pair is not tradable, fail fast as a hard error

### Kraken Futures

- closeorder with reduceonly: The Futures sendorder API has reduceOnly, and the Futures openorders response also returns reduceOnly. So this API native reduceonly will be used.
- split Trade orders above configured max_quote into exchange side iceberg order: adapter-managed sequencing in the private websocket/order-update worker, because there is no native API iceberg support.

### Kraken Spot

- closeorder with reduceonly: The Spot AddOrder API has a reduce_only parameter for margin orders. So API reduceonly will be used for margin orders and a sell order for spot orders.
- split Trade orders above configured max_quote into exchange side iceberg order: native API iceberg support does exist and will be used.

### Bybit

Bybit offers only API to 3rd party trading programs and not to homebrewed programs for European Union customers.

### Bybit Simulation
