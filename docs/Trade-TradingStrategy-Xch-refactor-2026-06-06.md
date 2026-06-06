# Motivation

## Observations

- Many helper functions
- Redundant structs
- Difficult to understand how the different data structures are kept in sync
- Too much code for teh intended functionality

## Cosmetics

- CryptoXch shall be renamed as workspace wide hard cutover from CryptoXch to Xch
- coins_exchange folder shall be renamed to corresponding exhange, i.e. coins_bybit to bybit, coins_krakenfutures to krakenfutures, coins_krakenspot to krakenspot
- because opening a short position is a sell order the following hard cutover renaming shall be done: shortstrongbuy to shortstrongopen, shortbuy to shortopen, longbuy to longopen, longstrongbuy to longstrongopen

## Target Data Architecture

- Share 1 dashboard struct per trading pair (base/quote) between Trade, TradingStrategy, Xch enabling all modules to access all relevant data without redundancy
- Store per trading pair (base/quote) all sample relevant data in a Trading DataFrame that can also be saved
  - Shall replace a significant part of the current textual logging
  - Provides a data to allow diagnostics of unintended trading mistakes in order to fix them
  - Provides data to compare simualted training against real trading to understand deviations and improve simulation 
  - The data shall be stored as arrow file in the exchange specific coins folders next to ohlcv.arrow
  - data of a trading session shall be appended with an outerjoin to already present data
- Ohlcv data shall stay in its own DataFrame managed by Xch
- Features shall stay in their own DataFrame managed by Features
- Targets shall stay in their own DataFrame managed by Targets
- The Trading, Targets, Features, Ohlcv DataFrames must stay in sync to allow row access via the same index
  - If Features require history, i.e. it starts later than Ohlcv, then the dashboard shall use a view to Ohlcv to keep the indices of all DataFrames in sync
- The Trading Data Frame shall comprise at least the following sample info
  - DateTime timestamp of sample (copy of corresponding OHLCV :opentime)
  - TradingStrategy advice concerning long buy price, long sell price, short buy price, short sell price, trade label
  - Trade order request for short and long order: order type (open, amend, close, noop), leverage, base amount, limit quote price
  - Xch order feedback for short and long order: order id or missing, order status (accepted, rejected, partially filled, filled, cancelled, noop in case of no order), not yet filled order amount, average order fill price, categorized order error from exchange or internal signals (e.g. coin pair not tradable, order amount smaller than minimum amount)
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

### TradingStrategy

- Is deriving features from Ohlcv
- Is classifying the sample based on features
- Is considering the classification result and derives a trading advice for short and long positions of the trading pair under consideration

### Trade

- Is controlling the trade loop of max 1 minute duration
    - Is receiving from Xch an update of 
    - open orders
    - available assets, positions, maintenance margin, equity
    - Is receiving the trading advice from TradingStrategy
    - In case of an advice to open an asset or position
        - requests close order for any open asset or position of the opposite trend (short vs long) as redcue only (if applicable) maker order without limit = 1 tick next to ask
        - if long asset order
        - amount = min(min(balance, MAXBUDGET) * MAXFRACTION - already bought base assets, free quote)
        - if margin or futures order 
        - open order only if free margin > used margin * MARGINHEADROOM
        - amount = min(min(equity, MAXBUDGET) * MAXFRACTION - already bought position, free exposure(trading pair, leverage))
        - no order if amount < tradable minimum quantity
        - requests an open order with a buy limit (longbuy) or an open order without limit (strongbuy)
- At session start and periodically as configured a selection takes place that checks trading pairs according to liquidity and selects a set that is tradable
  - portfolio assets or open positions are by default sellable
  - trading pairs that are valid AND per exchange tradable AND white listed AND have enough OHLCV history to be accepted by TradingStrategy  AND sufficient liquidity contiuity are considered openclose tradable
- beside the trading pair specific categorization the following trade modes shall be supported: openclose, closeonly, quickexit (market sell of all positions), notrade (for testing)
