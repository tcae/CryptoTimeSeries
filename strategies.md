# Strategies

General thoughts:

- Problem: what to label as a trade signal, i.e. what is a noteworthy deviation from a trend that is worth to signal a trade?
  - An option is to label everything that has a future gain or loss above a threshold. This will result in relatively poor classifers because it takes some time before a trend is established as such and distinguidhed from noise. Hm, that can be mitigated by setting the threshold high enough (e.g. 5% while 1% should be already profitable)- the advantage that labelling remains straightforward.
  - To address noise, signals labels may only be sent after a number of samples in the same direction and then still exceeding a gain/loss threshold. That can be achieved by using a regression window of x minutes.
- Follow trends: when is a trend broken? One may consider trend stability a criteria (less volatility but reliability in direction) versus a trend with high volatilty to catch extrema for trading and assume that the trend repairs a missed extreme. Trend reversal detection is crucial.
  - a trend is considered broken when the regression gradient changes sign but this should be detected earlier.
  - measure deviations of intact trends and accept  up to x sigma deviations before declared broken. This also enables break outs of a e.g. positive trend and to switch monitoring to a shorter regression window. Having a cascade of regression windows (e.g. 5m, 30m, 3h, 18h, 36h) the signal shall be bound to one of those windows with buy in a minimum and sell in a maximum. [Straight forward evaluations](data/gradientgainhisto.data) showed that a 24h showed good results while shorter windows returned losses. This can be explained with the observation that short windows also have more low amplitude noise signals. On the other hand in volatile times a long range window misses the fall back. Therefore a short range regression window is preferred in volatile times with high frequent, high amplitude prices changes while longer regression windows are preferred in low volatile times. In both cases a minimum gain is needed to avoid losses.
  - switch between a set of regression windows by using always the shortest regression window that shows recelty sufficient amplitude to realize profits
    - evaluate the profits of the last X hours for all regression windows and choose the most profitable
    - classify which regression window to use - can be implemented as classifier tree with leaf classifiers per regression window and one summarizing classifier to decide

## Target labels

Target labels are crucial as learning and evaluation reference of success (= performance).
The ideal target is the most performant, i.e. one that signals buy at a minimum with a gain more than 2*fee. However, that can not be reached with only a historical view. That means as a consequence that a target shall reflect, which has a chance to be recognized with only historical data at hand.

The following heuristics may help here:

- select a regression window and calculate the linear regression line
  - a local minimum as **buy** signal is detected if the regression gradient becomes >= 0 after a downslope
  - a local maximum as **sell** signal is detected if the regression gradient becomes <>= 0 after an upslope
  - use only regression windows that show a minimum difference of X% (e.g. 1%) in recent N slopes
  - use only regression windows with a supporting trend regression line of X*(regression window), e.g. X = 6
  - ignore local minima above and local maxima below the supporting tend regression line, which should prevent local disturbances (to be investigated because it bears the risk of missing extremes)
- use a set of different regression windows for above apprach
  - select the shortest regression window that meets above criteria because a short regression window has a higher chance of higher frequent gains
  - fall back to a shorter (or the shortest?) regression window in case of extraordinary deviation from the regression (e.g. 3 \* $\sigma$) to cover outbreaks
- open challenge: when to switch between regression windows?

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
- compare since last longest complete slope start anchor timestamp

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