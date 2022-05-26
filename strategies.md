# Strategies

General thoughts:

- Problem: what to label as a trade signal, i.e. what is a noteworthy deviation from a trend that is worth to signal a trade?
  - An option is to label everything that has a future gain or loss above a threshold. This will result in relatively poor classifers because it takes some time before a trend is established as such and distinguished from noise. Hm, that can be mitigated by setting the threshold high enough (e.g. 5% while 1% should be already profitable)- the advantage is that labelling remains straightforward.
  - To address noise, signal labels may only be sent after a number of samples in the same direction and then still exceeding a gain/loss threshold. That can be achieved by using a regression window of x minutes.
- Follow trends: when is a trend broken? One may consider trend stability a criteria (less volatility but reliability in direction) versus a trend with high volatilty to catch extrema for trading and assume that the trend repairs a missed extreme. Trend reversal detection is crucial.
  - a trend is considered broken when the regression gradient changes sign but this should be detected earlier.
  - measure deviations of intact trends and accept  up to x sigma deviations before declared broken. This also enables break outs of a e.g. positive trend and to switch monitoring to a shorter regression window. Having a cascade of regression windows (e.g. 5m, 30m, 3h, 18h, 36h) the signal shall be bound to one of those windows with buy in a minimum and sell in a maximum. [Straight forward evaluations](data/gradientgainhisto.data) showed that a 24h showed good results while shorter windows returned losses. This can be explained with the observation that short windows also have more low amplitude noise signals. On the other hand in volatile times a long range window misses the fall back. Therefore a short range regression window is preferred in volatile times with high frequent, high amplitude prices changes while longer regression windows are preferred in low volatile times. In both cases a minimum gain is needed to avoid losses.
  - switch between a set of regression windows by using always the shortest regression window that shows recelty sufficient amplitude to realize profits
    - evaluate the profits of the last X hours for all regression windows and choose the most profitable
    - classify which regression window to use - can be implemented as classifier tree with leaf classifiers per regression window and one summarizing classifier to decide

## Target labels

Target labels are crucial as learning and evaluation reference of success (= performance).
With a perfect classifier at hand a maximum performant target signals buy at a minimum with a gain more than 2*fee, leveraging the high frequent volatility. However, classifiers are not perfect, which means that an appraoch is needed that ignores local extremes with too less gain in between.

The following heuristics may help here:

- select a regression window and calculate the linear regression line
  - a local minimum as **buy** signal is detected if the regression gradient becomes >= 0 after a downslope
  - a local maximum as **sell** signal is detected if the regression gradient becomes <>= 0 after an upslope
  - once a local regression maximum is reached search back for the price maximum between regression minimum and regression maximum
  - vice versa to search for the price maximum between a regression maximum and a regression minimum
  - perform this approach with various regression windows
  - use only those results that satisfy a certain gain/loss threshold
    - to be considered as risk for later refinements: short whale spikes may fool the approach
  - select the shortest regression window that meets above criteria because a short regression window has a higher chance of higher frequent gains

## Adaptive regression

- calculate potential slope gain by *regression slope with price extremes spreads* as follows
  - look for next minimum of regression slope
  - then look back for global minimum between last regression maximum and the just identified regression minimum, which estbalishes the price minimum spread
    - ceveat: spikes may confuse the approach -> risk reduction: further limit the price minimum search range back to the last gradient inflection point
  - look for next maximum of regression slope
  - then look back for global maximum between last regression minimum and the just identified regression minimum, which estbalishes the price maximum spread
    - ceveat: spieks may confuse the approach -> risk reduction: further limit the price maximum search range back to the last gradient inflection point
  - as target function connect the price extreme spreads with straights lines, which comes close to the regression line (but the regression line does does hit the extremes in most cases)
    - use the distance to the extremes from the straight lines as target function (resulting in jumps at extremes)
  - start the training window when the regression is still going down but regression is constantly improving, which also creates a smooth target function
    - as long as price minimum spread is not yet reached it should be classified *close*
    - as soon as price minimum spread is passed and price difference to price maximum spread is below minimum threshold it should be classified *buy*
    - as soon as price minimum spread is passed and price difference to price maximum spread is above minimum threshold it should be classified *hold*
    - as soon as price maximum spread is passed it should be classified *close*
- select most performant *regression slope with price extremes spreads* via selection classification as follows
  - calculate *regression slope with price extremes spreads* from a set of defined regressions
  - for training: use the shortest regression that meets or exceeds the minimum amplitude threshold by leveraging knowledge about the actual amplitude of the current slope at the point under consideration
  - for production: use the nearest regression result of the selection classifier predict, which can be a prediction between valid regressions

## Straight forward

Common part:

- choose a couple of regression windows and cryptos that have enough liquidity
  - enough liquidity = 1h median of liquidity per minute >= 3 * maximum volume to invest
- evaluate sepatately for each crypto and each regression window the gain when buying at regression minimum and selling at regression maximum

### Single crypto, single regression window

- Compare results without merging gains of cryptos or regression windows

Results:

- [Measurements regression windows [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]](data/gradientgainhisto.data) show that 24h performs best and >= 24h in general better. That is supported by a 2nd measurement with 18h, 24h, 30h, 36h (results appended to the same file as above).

### Single crypto, multiple regression windows

- per crypto on each maximum and minimum of the shortest regression window calculate the gain of all regression windows and select the regression window with the best gain
- compare since last longest complete slope start spread timestamp

Results:

- [best gradient gain](data/bestgradientgain.data) show no improved results at gain thresholds:
  - -1000 = no gain threshold, i.e. change always to the best
  - 0.0 = only change to the best if the gain since the beginning of the longest complete up slope is better then 0.0
  - 0.01 = explanation like 0.0 but gain improvement has to meet of exceed 1% in order to change the favorite regression window
  - 0.02 = explanation like 0.0 but gain improvement has to meet of exceed 2% in order to change the favorite regression window

#### Single crypto, multiple regression target label strategy

- focus on shortest regression window that match one of the following criteria
  - current downslope decreased < x% (e.g. 1%) between regression window maximum (gradient =0) actual price and current actual price
  - current upslope gradient exceeds regression window specific buy threshold and next larger regression window has also gradient > 0% and remaining upslope gain >y%, e.g. 1% (last criteria is only applicable in training due to view into the future).
  - focus was on this regression window and call-long was traded and only larger regression windows meet above criteria
  - focus was on this regression window and no other regression window matches above criteria
- with focus regression window do
  - trade call-close if gradient <= 0%
  - trade call-long upslope gradient exceeds regression window specific buy threshold and next larger regression window has also gradient > 0% and remaining upslope gain >y%, e.g. 1% (last criteria is only applicable in training due to view into the future).
  - no trade but status call-hold when predecessor is call-long or call-hold and gradient is still positive but less than 0.5% from gradient maximum price (i.e. gradient become 0 but the actual price and not the regression price is used)
  - no trade but status flat-hold if none of the above

### Multiple cryptos, single regression windows

- change invested crypto if gain over the N * last regression window minutes improved more than a threshold

### Multiple cryptos, multiple regression windows

- per crypto on each maximum and minimum of the shortest regression window calculate the gain of all regression windows and select the regression window with the best gain
  - over last n period of the shortest regression window
  - over the period of the longest bought candidate period
- compare the gain between the different cryptos and select the cryptos with
  - at least 1% gain
  - are within 10% of the most successful crypto
  - over the lat 1h (2h, 3h, ...) - if no crypto selected chose the next longer time window

## Experiments

[simplegainperformance](scripts/simplegainperformance.jl) measures the performace on a trinaing set of cryptocurrencies when buying at the begin of an up slope and selling at the end of an up slope at various regression time window lenghts.

- Use 24h as main investment guideline and add performance if volatility around a positive regression gradient exceeds threshold to add buy and sell within that volatility.
  Assumption: Thresholds of such approach shall be significantly higher than 1%.

  What are the gain distributions per regression window? Result: longer regression windows have a broader distribution and are less focused on the small <1% gains, i.e. also more extreme gains.

  Thesis: a regression qualifies if its low/high gain exceeds y% of threshold (e.g. 1%) and the next longer regression is positive.
  Distribution of regression gain > 1% in relation to last gain per regression and in relation to longer positive regression y/n?
  The same for losses: per regression window distribution of regression loss < -0.5% in relation to last loss

  include 2min regression

  2 windows (e.g. 5min, 15min) deviating in the same direction with increasing gradient is a signal to follow. Gradient thesholds may to be considered.

 aspect1 thesis: maximum slope gradients are correlated with trade control. The higher the gradient the shorter teh regression window.

 aspect2:  label strategy: if the real upslope gain is larger than x% then label according to that slope with real maximum backtrack.
  assumption: the higher the gain threshold the less frequent gains but the more reliable gain trade signals.

  to be investigated: is a larger window trend supervisor beneficial? assumption = yes
  consider factor 6 between windows: 5min, 30min, 3h, 18h, 36h (1,5d)

  approach: use the last n=3 predecessor slopes to determine by majority vote how the mnext slope likely works out in terms of gain
do that for different windows and switch after each sell to the best performing window

Break out handling approach: if smaller regression windows are within x \* standard deviation (rollstd) or within a fixed threshold (e.g. 1%) with higher likelihood that the change is significant then follow the shorter outlier.
    As soon as the shortest outlier is again within the x * standard deviation of a longer then switch back to that one.

Consider out of spread deviations down from a reached price level as trade criteria.


## Notes backup from Trade.jl

Challenges:
- what is an uptrend, considering that we have more or less significant distortions during an uptrend? Once determined, history data need to get corresponding **`targetlabels`**
- when does a profitable uptrend starts?
- when does an uptrend ends? Independent whether or not it is a profitable uptrend we have to limit losses
- are there uptrends per granularity, e.g. regression window?
- are there cluster patterns (may be per granularity), e.g. fast rising peak, slow crawling increase, ..
- what are the most profitable currencies and how to split asset allocation?
- how to adapt the parameters of the trading strategy based on history data? **`trainstrategy`**
- how to evaluate the success of the trading strategy? **`assessstrategy`**
- how to learn from history concerning patterns and learning strategies?
    + visualize / report an overall status
    + in situations that need to be changed
        - understand what happened
        - how probable are comparable situations
- what is an efficient SW design for the **`tradeloop`**

## Notes for using volatility and trend tracker in combination
- General thoughts
  - less volatility = more stable direction -> tracking preferred
  - high volatility = instable direction -> catch statistic outliers for yield
  - stabilze direction by longer regression window
- use regressionlines of different regression time windows as basis
- select the shortest window with a median standard deviation 2 * 1 sigma over longest considered regression window that is sufficient to satisfy the minimum profit requirement == spread window
- buy
  - only buy if spread gradient is positive
  - if prices decrease below regression line - 1 sigma then track with a tracker window that is shorter than spread window but has the longest history for the last direction change to filter out irrelevant small volatility
  - if price increases and tracker gradient becomes positive then "buy"
  - if after a "buy" prices fall then "sell" if prices fall below spread regression line - 3 sigma
- sell
  - sell independent of spread gradient
  - if prices increase above regression line + 1 sigma then track with a tracker window that is shorter than spread window but has the longest history for the last direction change to filter out irrelevant small volatility
  - if price decreases and tracker gradient becomes negative then "sell"
  - *(if after a "sell" prices rise then "buy" if prices rise above spread regression line + 3 sigma and (prices rise above tracker line + 2 sigma or tracker trend becomes positive))*
- spread window change
  - in general spread window is
    - the shortest regression window that satisfies profitabilityrequirements with deviation catching
    - has positive trend
    - has the most catches * normal deviation range (e.g. 2 * 1 sigma) = gain wihtin 24h among spread window candidates
    - spread sigma calculation = median of this window over the last 24h
  - for every buy the spread is newly calculated and stays with the trade
- tracker window change
  - always has to be shorter than spread window
  - should be long enough to suppress irrelevant regression xtremes
  - option 1: whenever another window is closer with its last regression xtreme to the last trade, i.e. no irrelevant xtremes in between
  - option 2: shortest window with least in between extremes since last trade
- tracker sigma calculation = median of this window since last trade signal
- parameters:
  - minimum profit = minimum profitability requirement a normal deviation range has to exceed to consider trading
  - spread minutes of regression window --> dynamically adapted
  - tracker minutes of regression window --> dynamically adapted
  - spread breakout sigma = factor * sigma to consider breakout of normal deviations, e.g. 2.0
  - spread minimum gradient to do any buy, e.g. 0.0
  - spread surprise sigma = factor * sigma to signal surprise plunge (sell) or raise (buy)
  - spread sigma minutes = longest considered regression window = minutes used to calculate the median of all regression window sigmas


## Notes for using volatility and trend tracker side by side
- General thoughts
  - less volatility = more stable direction -> tracking preferred
  - high volatility = instable direction -> catch statistic outliers for yield
  - stabilze direction by longer regression window with disadvantage that they follow less agile
- use regressionlines of different regression time windows as basis
- for each of the regression lines use different standard deviation factors to define a band around the rolling regression line
- the best gain over the minimum timespan wins as combi of (regression window, standard deviation factor)
- 2 strategy flavors:
- tracker:
  - buy at regression minimum + hysteresis (gradient | gain | period of positive gradient)
  - sell at regression maximum
- spread:
  - buy at lower breakout of `factor * standard deviation`
  - sell at upper breakout of `factor * standard deviation`
  - emergency sell at buy price - `emergency_factor * standard deviation`
- in either case only buy if window gradient is positive
-
- parameters:
  - minimum profit = minimum profitability requirement a normal deviation range has to exceed to consider trading
  - gain backtest minutes to determine best tracker | spread regression window and standard deviation factor that spans the tracker | spread band
  - set of regression windows to choose from
  - set of standard deviation factors to choose from
  - emergency standard deviation factor for emergency spread sell
  - hysteresis criteria to start tracker
  - spread minimum gradient to do any buy, e.g. 0.0

