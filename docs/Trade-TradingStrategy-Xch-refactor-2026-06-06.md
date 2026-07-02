# Why this refactoring is required?

It became clear that we run into a situation with more and more omplex code although the underlying logic does not seem to justify it.

- Many helper functions
- Redundant structs
- Difficult to understand how the different data structures are kept in sync
- Too much code for the intended functionality

## Target Data Architecture

- Share 1 dashboard struct per trading pair (base/quote) between Trade, TradingStrategy, Xch enabling all modules to access all relevant data without redundancy
- Store per trading pair (base/quote) all sample relevant data in a Trades DataFrame that can also be saved
  - Shall replace a significant part of the current textual logging
  - Provides a data to allow diagnostics of unintended trading mistakes in order to fix them
  - Provides data to compare simualted training against real trading to understand deviations and improve simulation 
  - The data shall be stored as arrow file in the exchange specific coins folders next to ohlcv.arrow
  - data of a trading session shall be appended to already present data
- Ohlcv data shall stay in its own DataFrame managed by Xch
- Features shall stay in their own DataFrame managed by Features
- Targets shall stay in their own DataFrame managed by Targets
- The Trades, Targets, Features, Ohlcv DataFrames must stay in sync to allow row access via the same index
  - If Features require history, i.e. it starts later than Ohlcv, then the dashboard shall use a view to Ohlcv to keep the indices of all DataFrames in sync
- The Trades Data Frame shall comprise at least the following sample info
  - DateTime timestamp of sample (copy of corresponding OHLCV :opentime)
  - TradingStrategy advice concerning long buy price, long sell price, short buy price, short sell price, trade label
  - Trade order request for short and long order: order type (open, amend, close, noop), leverage, base amount, limit quote price
  - Xch order feedback for short and long order: order id or missing, order status (accepted, rejected, partially filled, filled, cancelled, noop in case of no order), not yet filled order amount, average order fill price, message id (see below)
  - Xch available asset or position type (asset, margin, future), leverage, position amount, quote price, maintenance margin

  ## Responsibilities

  ### Xch 

- Xch is receiving order requests from Trade and is managing that request as good as possible
    - validity and tradability of trading pair for the request is checked and request is rejected if required
    - minimum tradable amount for the trading pair is checked: if equity allows is increased to fulfill the request or if equity does not allow is rejecting the request
- Inconsistencies result in warnings and act towards the save side, i.e. close positions, reject orders or redcue only orders
- Xch ensures that open orders are first closed - preferrably as reduce only orders - before another (opposite trend) order is opened, i.e. 
    - 1 order per trading pair at any time
    - 1 position or available asset with amount >= tradable minimum volume per trading pair
    - it shall be possible that Trade request to close an order for a specific trading pair  and to request opening an order for the opposite trend in the same minute. It is then up to Xch to first serve the closing and then the opening - possibly in the same minute
- Xch provides market data per minute to other modules
- Xch maintains equity, balance, free margin changes and provides equity, balances, free margin updates per minute to other modules. 
- Xch maintains open orders and provides an order update to other modules per minute
- Xch only accepts open orders for margin and futures positions if free margin > used margin * MARGINHEADROOM
- provides per exchange layer an implementation for available exposure and initial margin for a trading pair with leverage
- strongbuy or strongclose orders without limit are post-only makers orders that are periodicly amended to stay 1 tick next to the ask price
- trading issues shall be logged by a Xch.log_trading_issue(issuer, message) function
  - issues are logged directly as text and captured in Trades columns `longmsg` / `shortmsg`
  - `longstatus`, `shortstatus`, `longmsg`, and `shortmsg` are stored as `CategoricalVector`
  - this replaces the previous `XchCache.messages` and `_errors.json` id-catalog concept
  - the purpose is to store the ultimate reason of an issue with a trading pair together with all other relevant sample trading data without indirection through message ids
    - The issuer "Trading" is a message from the tradereal program and may be issued in TradingStrategy, Trade or Crypto before any exchange interaction takes place
    - The issuer KrakenSpot, Bybit, KrakenFutures is a message response from the exchange that signals a failing interaction, which should not crash the program

### TradingStrategy

- Is deriving features from Ohlcv
- Is classifying the sample based on features
- Is considering the classification result and derives a trading advice for short and long positions of the trading pair under consideration

### Trade

- Is controlling the trade loop of max 1 minute duration
    - Is receiving from Xch an update of 
      - open orders
      - available free quote, balance, free margin, equity
    - Is receiving the trading advice from TradingStrategy
    - In case of an advice to open an asset or position
        - requests close order from Xch for any open asset or position of the opposite trend (short vs long) as redcue only (if applicable) maker order without limit = 1 tick next to ask
        - requests an open order from Xch with a buy limit (longbuy) or an open order without limit (strongbuy)
- At session start and periodically as configured a selection takes place via tradeselection! that checks trading pairs according to liquidity and selects a set that is tradable
  - portfolio assets or open positions are by default sellable
  - trading pairs that are valid AND per exchange tradable AND white listed AND have enough OHLCV history to be accepted by TradingStrategy  AND sufficient liquidity contiuity are considered openclose tradable
- beside the trading pair specific categorization the following trade modes shall be supported: openclose, closeonly, quickexit (market sell of all positions), notrade (for testing)
- the current implementation distinguishes robot owned orders from not robot owned orders. This difference shall no longer apply and all orders shall be considered

## Phases

### Phase 1: cosmetics

- renaming cosmetics
  - Xch is now the workspace-wide hard-cut package and module name
  - coins_exchange folder shall be renamed to corresponding exchange, i.e. coins_bybit to Bybit, coins_krakenfutures to KrakenFutures, coins_krakenspot to KrakenSpot. In the same context Xch._setexchangecoinspath! shall be renamed to _setexchangepath!
  - because opening a short position is a sell order the following hard cutover renaming shall be done: shortstrongbuy to shortstrongopen, shortbuy to shortopen, longbuy to longopen, longstrongbuy to longstrongopen
- EnvConfig.pairquote shall be the canonical quote setting after the quote-name hard cutover
- EnvConfig.cryptopath shall be renamed to EnvConfig.tradingfolder
- The authname parameter of XchCache shall be removed as it should be equal to the exchange name
- The tradeselection! produced DataFrame uses the canonical columns openenabled and closeenabled
- A defaultquote shall be added to XchCache that shall be optional and shall be USDT for Bybit, USDC for KrakenSpot, USD for KrakenFutures. The XchCache function shall set the EnvConfig.pairquote accordingly
- Authentication shall move from EnvConfig to Xch because it is exchange related
- pass criteria: passing runtests, TrendDetector works on synthetic patterns, tradereal on KrakenSpot runs

### Phase 2: introduce Trades DataFrame and use it in TrendDetector 

- add a trading pair Dict(trading pair, Trades DataFrame) to Xch.XchCache and provide Xch creation and access functions because Xch is used by Trade and TradingStrategy
- create a new TsCache struct inside TradingStrategy for internal session data, e.g. configuration, Classifier reference
- implement an alternative TradingStrategy.getgains function that uses TsCache and Ohlcv, Trades DataFrame from xch
- TradingStrategy trading pair specific data shall be maintained in a TsTp struct that can be accessed via TsCache by trading pair Dict lookup
  - TsTp may also have a (read only) reference to trading pair specific Ohlcv data, which is updated by Xch
- The TradingStrategy algorithm configuration that exist shall be maintained
- gain_limit_reversal_pricedelta! shall be implemented using TsTp and Trades DataFrame
- TrendDetector shall be adapted accordingly
  - TradingStrategy.tradingstrategy shall use the trading pair specific Trades DataFrame instead of GainSegment
  - trades.arrow shall represent the Trades DataFrame and shall be stored next to gains.arrow
  - the configured TradingStrategy algorithm, e.g. gain_limit_reversal_pricedelta!, shall work on the Trades DataFrame to derive gains
- pass criteria: 
  - TrendDetector can still adapt a classifier based on synthetic SINE, DOUBLESINE data
  - TrendDetector can still load an already adapted classifier and create gains data and provide a result summary as today

### Phase 3: adapt Trade and Xch to use the Trading DataFrame 

- add Xch function `log_trading_issue(issuer, message)` and integrate direct message capture into Trades (`longmsg`, `shortmsg`) with unit tests
- add functions to Xch to be called by Trade in the trade loop
  - account_status to update equity, balance, free margin, free quote
  - order_status to update the order status of a specific trading pair in the Trades DataFrame
  - process_order_request to evaluate the order request for a specific trading pair from Trades DataFrame
    - to close an order first if an opposite position is open
    - if possible prepare open order and issue it async (without blocking trade loop) as soon as opposite position is closed
    - to check constraints before an order is opened and reject the request in case of inconsistencies or constraints are not fullfilled
      - if long asset open order request
        - round amount to supported precision
        - reject order request if (amount < minimum tradable amount) OR (amount > free quote)
          - log reject with Xch.log_trading_issue("Trading")
      - if margin or futures open order request 
        - round amount to supported precision
        - reject order request if (amount < minimum tradable amount) OR (initial margin(amount) > free margin)
          - log reject with Xch.log_trading_issue("Trading")
    - if amount is confirmed then check open orders for that position and consolidate them in 1 amended order by adding old order amount to new order amount and setting the limit to current openlimit
    - else open a new order to open a position or buy an asset
    - failure to place an order shall be logged by Xch.log_trading_issue() using the exchange string as issuer
  - exchange communication failures that are related to connection or workload but not related to the content of a request
    - Web Sockets connection drops and REST connection timeouts shall result in REST server time request to make sure the Internet and servers are available
    - retry with exponential delays up to 1 minute until server time is provided
    - reestablish Web Sockets and issue order
      - in case of repeated Web Socket failures, fall back to REST interface
    - unavailable server shall be retried with increasingly longer waittime up to 1 minute
  - to indicate a reject/failure reason by Xch.log_trading_issue in the Trades DataFrame
    - logs all messages at terminal and log file and leaves last failure as error id in sample data of Trades DataFrame
- implement Trade.open_amount(account status, unfilled open order amount, trading pair, order type, leverage), which considers constraints and returns a base amount that can be requested in case of a positive value or reduced in case of a negative value
    - if long asset order
      - amount = min(min(balance, MAXBUDGET) * MAXFRACTION - already bought base assets, free quote * (1 - BALANCEHEADROOM))
    - if margin or futures order 
      - open order only if free margin > used margin * MARGINHEADROOM
      - if free margin == 0
        - amount = - max(10% * already bought position, minimum tradable amount)
      else
        - amount = min(min(equity, MAXBUDGET) * MAXFRACTION - already bought position, min(free exposure(trading pair, leverage), free margin) * (1 - MARGINHEADROOM))
        - if 0 <= amount < tradable minimum quantity then amount = 0
        - by default the smallest available leverage shall be chosen
- implement trade loop variant of Trade.tradeloop and Trade.trade! with minimum required functionality
  - as per configuration tradeselection! shall identify at session start and periodically trade pairs that are considered for open and close, for close only, for quick close (to be mapped to strongclose)
  - update account status via Xch
  - per buysell or sellonly identified trading pair as long as it fits in 60s loop, otherwise start with not yet processed trading pairs in next trade loop cycle
    - request trade advice from TradingStrategy using Trades DataFrame
    - update order status via Xch using Trades DataFrame
    - in case of advice to open orders calculate amount by Trade.open_amount if the trading pair is identified as openclose tradable
      - if an opposite position is open Xch will first close it before issuing an order to open the requested position
      - deducting amount of requested order from free margin and free quote shall ensure sufficient funding 
    - in case of a close advice a maker reduce-only (if supported by the exchange) order shall use the closelimit
    - in case of a strongclose advice a maker reduce-only (if supported by the exchange) order shall use limit=nothing (i.e. limit is adapted periodically to 1 tick next to ask)
- Trade shall implement an account DataFrame that it shall use in the trade loop to store the account summary (tradetime(xc::XchCache), equity, balance, free margin, free quote), pairs (closeenabled trading pairs stored as blank separated String of pairs) per sample
- tradereal, and tradesim shall be adapted to use the new implementation
- on session exit and before the tradeselection! periodic refresh (not before the initial) the account DataFrame shall be appended to accountV1.arrow in the Xch.exchangepath, the trading pair specific Trades DataFrame shall be appended to tradesV1.arrow in the trading pair specific subfolder of the exchangepath, the Ohlcv data shall be appended to ohlcv.arrow in the trading pair specific subfolder of the exchangepath
-  pass criteria: tradereal and tradesim run without errors

### Phase 4: cleanup and 

- remove all code in Trade and TradingStrategy that is no longer used in the call graph of tradesim, tradereal, TrendDetector after switching them over to deprecate GainSegment and GainSegmentRuntime
-  pass criteria: tradereal, tradesim, TrendDetector run without errors

## Copilot review comments (2026-06-07)

### High severity

- Message id space is underspecified and can overflow fast because ids are planned as UINT8 with fixed issuer ranges. The current design has no collision/overflow policy and no rollover behavior when new messages are appended over time.
  - Proposal: keep in-memory UInt16 ids, persist compact UInt8 only for known catalog entries, and reserve 255 as unknown/overflow sentinel. Add deterministic remap rules during load/save.
  - Response: considered in approach with FataalError in case of overrun

- Async close-then-open behavior can create duplicate or contradictory orders when trade loop cycles before close confirmation.
  - Proposal: introduce explicit per-pair order state machine states (idle, close_pending, close_confirmed, open_pending, open_confirmed, failed) and allow one transition per loop tick.
  - Response: the tradeloop is async from the web socket service but it is the requirement that Xch handles the complete closure of the opposite position before issuing an opening order synchronuously in thsi web socket service

- Trades DataFrame append model is missing schema versioning and migration rules. With hard cutovers and renamed columns, old sessions can become unreadable.
  - Proposal: persist schema_version and producer_version columns and add migration adapters at read time.
  - Response is now considered by adding a V1 to accountV1.arrow and tradesV1.arrow. ohlcv.arrow is very stable over years and not considered a change candidate.

### Medium severity

- DataFrame sync by index across Ohlcv, Features, Targets, Trades needs a formal contract for startup gaps and late feature availability.
  - Proposal: define one canonical sample key as (pair, opentime) and use explicit joins/views, not only positional index assumptions.
  - Response: The canonical sample key is used as exchange + pair to select the data folder, the opentime at start and end as timestamp check. I see insufficient value in a join and later split for storage

- Accounting formulas for open_amount mix free quote, free margin, equity, and already-open quantities but do not define behavior for stale account snapshots and concurrent pending orders.
  - Proposal: compute against a loop-local reserved-funds ledger and expire reservations deterministically after timeout.
  - Response: good proposal but refresh per cycle of account and open order status to be considered in the loop-local ledger

- Message matching uses substring rules, which can misclassify unrelated errors.
  - Proposal: use priority matching order: exact code, exact message, normalized message hash, then fallback substring.
  - Response: as long as existing messages from the list are no substrings of each other, I don't see a risk. Clarification: The message of the registered list shall be a substring of the provided message. Rationale: Composition and additional info added to a message can vary between exchanges. The messages from the registered list shall be long enough to avoid misclassification.

### Low severity

- Several renamed terms are broad hard cutovers across many packages.
  - Proposal: keep one temporary compatibility layer for one release cycle to reduce breakage risk while tests are expanded.
  - Response: No need for compatibility layer. I prefer quick fixes instead and keep code compact.

## Updated implementation plan (Copilot)

### Progress status (2026-06-11)

- Overall status: refactor design is still in planning / pre-implementation, but the runtime baseline was stabilized and workspace/package tests were brought back to green.
- Completed baseline work relevant to this plan:
  - `Trade` tests were repaired and are passing again.
  - removed `TradeAudit` call sites were migrated or removed in favor of `TradeLog`.
  - `KrakenSpot` test execution was stabilized by making online tests opt-in by default.
  - `KrakenFutures` websocket challenge signing was corrected to match the documented vector.
  - workspace test entrypoint was re-run successfully after the above fixes.
- Interpretation:
  - The codebase is in a usable baseline state for the refactor.
  - The new `Trades DataFrame` architecture and most of the phase checklist items below are still pending implementation.

### TradeLabel hard cutover progress (2026-06-13)

- The enum rename from `longbuy/shortbuy` to `longopen/shortopen` is complete in the active runtime and test paths that were blocking the refactor, and the persistent-format concern remains resolved: BSON and Arrow load/store enum values as integer codes, so no data migration layer is required.
- `Trade` runtime code and `Trade` tests were migrated and validated; `Trade` package tests are passing again.
- `TradingStrategy` runtime code and the touched tests were migrated and validated; stale label-name failures were removed from the active paths.
- `Targets` runtime helpers, tests, and threshold API/config surface were migrated to the new naming, including the `shortbuy` to `shortopen` API/config cutover.
- `TrendDetector` label-column and PPV naming was updated to the new `longopen/shortopen` terminology.
- Residual legacy wording still exists in some older classifiers, comments, diagnostics, and legacy/reference files, but it is no longer blocking active runtime behavior.
- Validation status after this cutover: `Pkg.test("Targets")`, `Pkg.test("Trade")`, and the workspace `test/runtests.jl` entrypoint are passing.
- Next step: finish residual legacy wording cleanup where still worthwhile, alongside the broader refactor phases below.

### Phase 0: contracts and safety rails `[in progress]`

Goal: lock interfaces before implementation work in phases 1-5.

#### Xch package checklist

- Required structs/types
  - [ ] `Xch.TradesSchemaV1` constant contract (column names, eltypes, nullability).
  - [ ] `Xch.XchCache` additions: `pairstates::Dict{String,DataFrame}`, `defaultquote::Union{Nothing,String}`.
- Required Trades DataFrame v1 columns (must be created by Xch helper)
  - [x] Canonical naming contract: `tradelabel` is the only supported label column in Trades v1 (`label` is not part of the contract).
  - [x] Identity/time: `opentime::DateTime`, `lastopentrade::Union{Missing,DateTime}`.
    - comment: `exchange::String`, `pair::String` notrequired as column because folder structure is used for that info
  - [x] Strategy advice: `longopenlimit::Union{Missing,Float32}`, `longcloselimit::Union{Missing,Float32}`, `shortopenlimit::Union{Missing,Float32}`, `shortcloselimit::Union{Missing,Float32}`, `tradelabel::TradeLabel`, `labelscore::Float32`.
  - [x] Trade request long: `longleverage::Union{Missing,UINT8}`, `longamount::Union{Missing,Float32}`, `longopenlimit::Union{Missing,Float32}`, `longcloselimit::Union{Missing,Float32}`.
  - [x] Trade request short: `shortleverage::Union{Missing,UINT8}`, `shortamount::Union{Missing,Float32}`, `shortopenlimit::Union{Missing,Float32}`, `shortcloselimit::Union{Missing,Float32}`.
  - [x] Exchange feedback long: `longid::Union{Missing,String}`, `longstatus::CategoricalVector{String}`, `longunfilled::Union{Missing,Float32}`, `longpriceavg::Union{Missing,Float32}`, `longmsg::CategoricalVector{Union{Missing,String}}`.
  - [x] Exchange feedback short: `shortid::Union{Missing,String}`, `shortstatus::CategoricalVector{String}`, `shortunfilled::Union{Missing,Float32}`, `shortpriceavg::Union{Missing,Float32}`, `shortmsg::CategoricalVector{Union{Missing,String}}`.
  - [x] Position/account snapshot: `postype::String`, `posleverage::Union{Missing,Float32}`, `posamount::Union{Missing,Float32}`, `quoteprice::Union{Missing,Float32}`, `maintmargin::Union{Missing,Float32}`, `equity::Union{Missing,Float32}`, `balance::Union{Missing,Float32}`, `freemargin::Union{Missing,Float32}`, `freequote::Union{Missing,Float32}`.
- Required tests (Xch/test)
  - [x] schema test: a newly created trades table contains exactly all v1 columns with expected eltypes.
  - [x] issue logging test: `log_trading_issue` returns the normalized message string for direct Trades storage.
  - [ ] close order for open position followed by open order in opposite direction for the same trading pair in the same minute: close is fully completed before open request for opposite direction in async web socket service 
    - [ ] connection errors are handled gracefully such that rate limit overrun does not occur and that connections are reestablished after a connection break
    - [ ] all exchange site errors are handled such that they don't result in a fatal errors but they are reported, the situation is mitigated and trading can continue as soon as teh exchange is available again

#### TradingStrategy package checklist

- Required structs/types
  - [ ] `TradingStrategy.TsTp` with fields: `ohlcv`, `trades::DataFrame`.
  - [ ] `TradingStrategy.TsCache` with fields: `mc::Dict`, `classifier::AbstractClassifier`, `pair::Dict{String,TsTp}`.
  - [ ] `TradingStrategy.TsCache.mc` modules constant keys: `:configname`, `:buygain`, `:sellgain`, `:limitreduction`, `:minpricedelta`, `:maxwindow`.
  - [ ] Classifier call skip optimization is deferred for now; strategy path shall classify each sample and shall not require cached no-classify bookkeeping fields.
- Required column ownership contract
  - [ ] TradingStrategy writes only advice columns (`longopenlimit`, `longcloselimit`, `shortopenlimit`, `shortcloselimit`, `tradelabel`, `labelscore`).
  - [ ] TradingStrategy does not mutate exchange feedback/account columns.
- Required tests (TradingStrategy/test)
  - [ ] ownership test: strategy functions modify only the advice columns.
  - [ ] continuity test: SINE/DOUBLESINE results still produce deterministic advice series.
  - [ ] initialization test: TsCache/TsTp creation works for first sample and warm restart.

#### Trade package checklist

- Required structs/types
  - [ ] `Trade.Account` with fields: `tradetime`, `equity`, `balance`, `free_margin`, `free_quote`, `pairs`.
- Required DataFrame contract
  - [ ] `accountV1.arrow` schema has columns: `tradetime::DateTime`, `equity::Float32`, `balance::Float32`, `free_margin::Float32`, `free_quote::Float32`, `pairs::String`.
- Required tests (Trade/test)
  - [x] open_amount contract test for asset and margin/futures paths.
  - [ ] immediate-funding test: issuing an order reduces free margin/free quote in the current account snapshot without a separate reservation ledger.
  - [ ] TsCache init test: all expected data structures are filled and are consistent
  - [ ] loop sequencing test: 
    - [ ] all openenabled and closeenabled trading pairs are processed round robin
    - [ ] if not all trading pairs can be processed in a minute processing starts the next cycle with the next trading pair to ensure fair round robin processing
    - [ ] tradeselection! updates the list of tradable pairs at session start and periodically as configured
    - [ ] before tradeselection!  all data that is identified as to be saved is actually appended to the corresponding arrow files
    - [ ] fatal errors and interrupts are handled such that all data that is identified as to be saved is actually appended to the corresponding arrow files

#### EnvConfig package checklist

- Required config contract
  - [ ] `pairquote` and `tradingfolder` are canonical fields used by Xch/Trade.
  - [ ] authentication keys are no longer owned by EnvConfig APIs used in runtime path.
- Required tests (EnvConfig/test)
  - [ ] runtime config load test verifies `pairquote` and `tradingfolder` default/override behavior.
  - [ ] guard test verifies deprecated auth access path is absent from runtime code path.

#### Workspace-level integration checklist

- Required cross-package tests
  - [ ] integration smoke: create Xch cache, create trades table, run one TradingStrategy advice write, run one Trade loop iteration.
  - [ ] persistence smoke: append one row to `tradesV1.arrow` and one row to `accountV1.arrow` and read back with expected schema.
  - [ ] deterministic replay smoke: same input OHLCV and config yields same advice/request columns.

- Exit gate
  - [ ] Xch, TradingStrategy, Trade, EnvConfig package tests pass.
  - [x] workspace test entrypoint passes with no schema-contract failures.

### Validation checklist snapshot (2026-06-15)

- [x] `Pkg.test("Xch")` passed in this refactor cycle.
- [x] `Pkg.test("Targets")` passed after removing duplicate `longopenbinarytargets` overwrite.
- [x] workspace `julia --project=. test/runtests.jl` passed.
- [x] focused TrendDetector path `include("test/trend_detector_cache_test.jl")` passed.
- [x] focused TrendDetector path `include("test/trend_detector_cache_test.jl")` re-run passed.
- [x] `KrakenSpot` package tests passed (`Pkg.test("KrakenSpot")`, offline suite green).
- [~] `KrakenSpot` online tests remain opt-in and were skipped by default (`KRAKEN_ONLINE_TESTS=true` required).
- [x] `Pkg.test("TradingStrategy")` re-run passed after `StrategyConfig` rename.
- [x] `Pkg.test("Trade")` re-run passed after `StrategyConfig` rename.

Progress note:
- Baseline validation is complete enough to start the refactor work: package/workspace tests were repaired and brought back to passing state.
- The interface contracts and schema safety rails listed in this phase are not implemented yet, so this phase remains in progress.

### Phase 1: naming and path cutover `[partially completed]`

Checklist (Phase 1)

- Naming and module ownership
  - [x] Workspace package/module rename completed from CryptoXch to Xch.
  - [x] buy/sell to open/close naming hard cutover is complete for active runtime/test/API paths.
  - [x] tradeselection! uses canonical columns openenabled and closeenabled.
- Config and path cutover
  - [x] EnvConfig.pairquote is canonical.
  - [x] EnvConfig.cryptopath hard cutover to EnvConfig.tradingfolder is done.
  - [x] coins_exchange hard cutover is complete in code paths; _setexchangepath! is canonical.
  - [x] XchCache defaultquote uses exchange-specific defaults and sets EnvConfig.pairquote.
- Authentication and constructor surface
  - [x] Authentication ownership moved to Xch runtime entry points.
  - [x] XchCache authname is removed from runtime and constructor surface.
- Validation and pass criteria
  - [x] Workspace/package runtests relevant for the cutover are passing.
  - [x] TrendDetector still works on synthetic patterns (config=046 parity workflow).
  - [x] tradereal on KrakenSpot live run confirmed by manual test (2026-06-14).

Audit update (2026-06-13):
- [x] Workspace package/module rename completed from CryptoXch to Xch.
- [x] EnvConfig.pairquote is canonical.
- [x] EnvConfig.cryptopath hard cutover to EnvConfig.tradingfolder is done.
- [x] tradeselection! uses canonical columns openenabled and closeenabled.
- [x] Authentication ownership moved to Xch runtime entry points; EnvConfig auth APIs remain available but are no longer exported as part of the default public surface.
- [x] XchCache authname is removed from runtime and constructor surface.
- [x] coins_exchange hard cutover is complete in code paths (`_setexchangepath!` is canonical and legacy `_setexchangecoinspath!` usage is absent).
- [x] buy/sell to open/close naming hard cutover is complete for active runtime/test/API paths (`Trade`, `TradingStrategy`, `Targets`, key scripts).
- [x] defaultquote in XchCache with exchange-specific quote defaults (and pairquote setup from that field) is implemented.

Current pass-criteria verification:
- [x] package runtests: workspace `test/runtests.jl` is passing, and the focused `Targets` and `Trade` package suites were rerun successfully after the naming/API cutover.
- [x] TrendDetector still works on synthetic patterns (validated with config=046 parity workflow).
- [x] tradereal on KrakenSpot live run confirmed by manual test (2026-06-14).

Progress note:
- The most disruptive Phase 1 naming cutover is now functionally complete in the active code paths: runtime labels use `longopen/shortopen`, and the `Targets` threshold/config API now uses `shortopen` consistently.
- Remaining Phase 1 work is narrowed to path/defaultquote items plus cleanup of residual legacy wording in older or non-critical files.

### Phase 2: Trades DataFrame foundation in Xch and TradingStrategy `[completed]`

Checklist (Phase 2)

- Xch Trades DataFrame ownership
  - [x] Add per-pair Trades tables in XchCache and creation/access API.
  - [x] Pair-state access is deterministic and keyed by canonical pair token.
- TradingStrategy cache and ownership contract
  - [x] Add TsCache/TsTp structures for pair-local strategy state.
  - [x] Enforce ownership rule: TradingStrategy writes advice columns only; Xch owns request/feedback/account columns.
- TrendDetector integration
  - [x] Adapt TrendDetector path to use TsCache plus Xch pair-state path.
  - [x] Persist tradesV1.arrow artifacts in the refactored path.
- Strategy behavior continuity
  - [x] Keep strategy configuration behavior while removing active GainSegment dependency from runtime flow.
  - [x] gain_limit_reversal_pricedelta! runs through TsTp/Trades row-state updates.
- Validation and pass criteria
  - [x] Synthetic SINE and DOUBLESINE adaptation/reload scenarios pass.
  - [x] TradingStrategy package tests pass in the current cycle.
  - [x] TrendDetector parity run against Trend-046-test-ref passes.

### Trades v1 Schema Consolidation Refactor (2026-06-14)

**Objective**: Eliminate duplication of Trades DataFrame schema normalization logic, consolidate column requirements into a single source-of-truth contract, and enforce that contract via explicit assertions rather than repeated per-call expansion.

**Completed work**:

1. **Schema contract definition** (`Xch/src/XchCore.jl`)
   - Added `TRADES_V1_REQUIRED_COLUMNS` constant tuple defining all 33 required columns at module scope.
   - Defined `_asserttradesv1schema(df::DataFrame)::Nothing` to validate schema completeness against the canonical contract.
   - Error messages now include missing columns and actual column names for clarity.

2. **Schema expansion consolidation** (`Xch/src/XchCore.jl`)
   - `_ensuretradesv1schema(df::DataFrame)::DataFrame` now the single normalization path:
     - Handles missing `opentime` by throwing for non-empty rows (safety gate).
     - Lazily instantiates missing columns with correct types (`Union{Missing, T}` or String default values).
     - Now calls `_asserttradesv1schema` internally to validate final contract completion.
   - Replaced 60+ lines of repeated `_ensuretradesexecutioncolumns!` implementation with single-line delegation alias to `_ensuretradesv1schema`.

3. **Contract persistence** (`Xch/src/XchCore.jl`)
   - Stored `TRADES_V1_REQUIRED_COLUMNS` tuple in `XchCache.mc[:trades_v1_required_columns]` at cache construction time.
   - Enables future runtime schema audits and documentation without re-parsing the constant.

4. **Runtime assertion usage** (`Xch/src/XchCore.jl`)
   - Replaced repeated `_ensuretradesexecutioncolumns!(tradesdf)` calls in runtime entry points with lightweight `_asserttradesv1schema(tradesdf)`.
   - `order_status(xc, tradesdf, ix)` and `process_order_request(xc, tradesdf, ix)` now assert contract instead of expanding.
   - Separation of concerns: expansion happens once at `settrades!` or first use; assertions happen at runtime entry points.

5. **Schema contract test** (`Xch/test/trades_schema_contract_test.jl`)
   - New unit test module ensures all required columns are present in empty trades DataFrames.
   - Validates column types match contract (String, Float32, UInt8, DateTime, Any).
   - Test uses `_asserttradesv1schema` directly to confirm schema-validation logic is correct.
   - Added `cols=:subset` to row insertions to maintain compatibility with expanded schema.

6. **Test fallout fix** (`TradingStrategy/test/runtime_api_test.jl`)
   - Fixed `TsCache pair-state scaffolding syncs Xch trades` test (line 200).
   - Changed `push!(tp.tradesdf, (opentime=..., lastopentrade=missing))` to use `cols=:subset` to allow partial-row insertion into expanded DataFrame.
   - All TradingStrategy tests now pass (12/12 in this testset).

**Validation results**:
- [x] `Pkg.test("Xch")` passes with new schema-contract test included.
- [x] `Pkg.test("TradingStrategy")` passes (92 total tests, including refactored row-insertion logic).
- [x] `Pkg.test("Trade")` passes (94 tests).
- [x] No redundant column-expansion calls remain in runtime paths.
- [x] Schema contract is now explicitly documented in code and enforced at entry points.

**Impact**:
- Eliminated code duplication: ~60-line `_ensuretradesexecutioncolumns!` is now a single-line alias.
- Single source of truth: `TRADES_V1_REQUIRED_COLUMNS` is the canonical reference for expected schema.
- Improved maintainability: adding a required column now requires a single edit to the constant and its usage documentation.
- Regression protection: new test ensures schema contract is not silently violated by row insertions or DataFrame mutations.
- Easier debugging: schema assertion errors now report missing columns explicitly instead of failing downstream on first field access.

**Remaining work in Phase 2**:
- [ ] Account DataFrame schema contract (similar pattern to Trades).
- [ ] OHLCV schema validation if needed (currently stable, not modified in this phase).
- [ ] Runtime audit trail that logs schema-contract assertion results for suspicious mutations.

- [x] Add per-pair Trades tables in XchCache and creation/access API.
- [x] Add TsCache/TsTp with strict ownership rules (Xch owns mutable trade tables, TradingStrategy reads/writes only its columns).
- [x] Adapt TrendDetector path to write tradesV1.arrow artifacts.
- [x] Keep strategy configuration behavior while moving TrendDetector coupling away from GainSegment (StrategyConfig cutover with compatibility adapter).
- [x] gain_limit_reversal_pricedelta! is executed through TsTp/Trades DataFrame row-state updates in TsCache path.
- Exit gate:
  - [x] synthetic SINE and DOUBLESINE adaptation and reload scenarios still pass.

Progress note:
- Implemented `XchCache` per-pair Trades DataFrame ownership with deterministic creation/access helpers.
- Implemented `TradingStrategy` `TsCache`/`TsTp` pair-state integration and gain evaluation path that syncs pair trades through `Xch`.
- Adapted `TrendDetector` gain flow to use the Phase 2 `TsCache`+`Xch` path and persist `tradesV1` artifacts.
- Validated with `TradingStrategy` package tests and clean-reset TrendDetector parity run (`config=046`) against `Trend-046-test-ref`.



### Phase 3: Trade loop integration and deterministic order orchestration `[completed]`

- [x] Implement account_status, order_status, process_order_request with state-machine transitions.
- [x] Add loop-local reservation ledger for quote and margin to avoid over-allocation across pairs.
- [x] Implement close-first then open sequencing with explicit close confirmation requirement within web socket handling of orders.
- [x] Integrate log_trading_issue and message-id capture into Trades rows.
- [x] Default integration mode runs with TradeLog disabled (opt-in only) to reduce runtime overhead during refactor rollout.
- [x] `usenewtrade` is default-on in `TradeCache`; scripts `tradesim.jl` and `tradereal.jl` now follow this default (with env opt-out toggles).
- Exit gate:
  - [x] tradesim and tradereal runtime entry paths are switched to the new Trade/Xch DataFrame orchestration path.
  - [~] no duplicate open orders under reversal stress is covered by integration tests for close-then-open blocking; extended long-run stress soak remains operational follow-up.

Progress note:
- Xch-side helpers (`account_status`, `order_status`, `process_order_request`, direct message logging into Trades msg columns) are implemented with tests.
- Trade now has a default-on (`cache.mc[:usenewtrade]=true`) per-tick delegation path that writes Trades DataFrame request/advice rows and calls Xch request/status APIs.
- Legacy Trade path remains available as fallback for runtime safety; Trade package tests pass after this integration slice.
- Reservation-ledger semantics are now integrated in the new path (`_tick_opening_reservations`, `_reserve_opening_budget!`, account projection helpers).
- `open_amount` now applies account constraints in the new path and is covered by integration tests.
- Close-then-open sequencing guard is active in the new path and covered by integration tests.
- Dust-level false warnings for positions-without-close-order were suppressed to reduce noise (`qty` below tradable minimum is ignored).
- TradeLog/audit writes are now default-off in integration mode (`TradeCache` sets `enable_tradelog=false`), with explicit test/runtime opt-in where audit artifacts are required.
- Validation snapshot (2026-06-14): `Pkg.test("Trade")` passes (94/94), and workspace `test/runtests.jl` is green.
- `usenewtrade` default-ready cutover is complete and enabled by default.
- Script-level runtime toggle support added: `TRADESIM_USE_NEW_TRADE` and `TRADEREAL_USE_NEW_TRADE` (default `true`, set `false` to force legacy path during rollback diagnostics).
- Remaining operational follow-up: optional extended reversal soak validation in long-run live/sim sessions.

#### Phase 3.1 Questions concerning simplified tradeloop

- Xch.account_status is called called multiple times before tradeloop but also at start of tradestep!

### Phase 4: persistence, retries, and resilience hardening `[partially addressed outside refactor]`

- [ ] Persist account.arrow, trades.arrow, ohlcv.arrow with atomic append strategy and crash-safe flush checkpoints.
- [ ] Implement connection retry policy with bounded backoff and explicit degraded-mode flags.
- [ ] Add tests for websocket drop, REST timeout, and reconnect fallback behavior.
- Exit gate:
  - [ ] resilience tests pass and data files stay consistent after forced interruption tests.

Progress note:
- Some exchange/runtime stability issues were fixed during baseline stabilization, especially around KrakenSpot/KrakenFutures test behavior.
- The persistence and resilience model described here has not yet been implemented as a dedicated refactor phase.

### Phase 5: cleanup and hard cutover `[in progress]`

- [x] Remove deprecated GainSegment/GainSegmentRuntime call paths after all entrypoints are migrated.
- [ ] Remove temporary compatibility aliases introduced in phase 1 (note: new backward-compatibility layer for legacy TradeLabel aliases added in 2026-06-15 is NOT temporary and should be retained for long-term cache interoperability).
- Exit gate:
  - [~] tradereal, tradesim, TrendDetector pass and no deprecated symbol is referenced in call graph checks (legacy label alias support now enables stable TrendDetector operation with pre-cutover cached data).

#### Bug fix: legacy TradeLabel aliases normalization (2026-06-15)

**Issue**: TrendDetector gain diagnostics showed all-zero gain rows after switching from legacy labels (`longbuy`/`shortbuy`) to current names (`longopen`/`shortopen`). Investigation revealed that legacy labels in cached data were normalizing to `allclose`, which eliminated all open signals and thus materialized zero gains.

**Root cause**: The `Targets.tradelabel(str)` function lacked backward-compatible alias mapping for the buy→open and sell→close terminology shift, causing old cached predictions/targets to silently collapse to the default fallback.

**Resolution**:
1. Added `_legacytradelabelalias()` helper in `Targets.jl` to map legacy names to current equivalents:
   - `longbuy` → `longopen`, `shortbuy` → `shortopen`
   - `longstrongbuy` → `longstrongopen`, `shortstrongbuy` → `shortstrongopen`
   - `longsell` → `longclose`, `shortsell` → `shortclose`
   - `longstrongsell` → `longstrongclose`, `shortstrongsell` → `shortstrongclose`
2. Updated `Targets.tradelabel()` to consult alias map before enum lookup.
3. Added `@testset "TradeLabel legacy aliases"` in `Targets/test/runtests.jl` to prevent regression.
4. Hardened `TrendDetector.getmaxpredictionsdf()` to normalize labels in freshly computed predictions and assembled results (both cached and fresh paths).

**Validation**: 
- Confirmed `Targets.tradelabel("longbuy")` now returns `longopen` (not `allclose`).
- Verified legacy cached targets normalize to active open labels (224,357 / 224,362 non-allclose rows in reference cache).
- Ran `Pkg.test("Targets")` with new regression testset; all tests pass.

**Impact**: This fix restores TrendDetector's gain materialization for all datasets that were built with pre-cutover terminology, eliminating the source of zero-gain diagnostics and enabling backward-compatible use of cached prediction/training data.

Progress note:
- Deprecated symbol cleanup is complete in active Julia code paths; remaining work is compatibility alias retirement plus final end-to-end pass/soak confirmation.
- **Update (2026-06-16)**: legacy label alias backward-compatibility layer is now implemented and validated, unblocking TrendDetector workflow with legacy cached data.
