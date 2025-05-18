# Here are just unordered notes

## ToDo prio

1) Lern- and Trainset across coins and balanced across targets
2) regression featurtes down to regrwindow 2 minutes (diff of consecutive pivots
3) Targets
    - follow longest regression window as long as pivot does not deviate more than 0.5%
    - shorten regression window to meet above max deviation
    - in case of a trend change, it starts after 0.5% deviation + 1 minute to react with buy or sell if and only if it continues at least another 0.5% considering previous mentioned trend change rules of 0.5% change + 1 minute

## Adaptation approach

- trade reactions have to be proportional to amplitude changes
- regression focus has to be proportional to frequency of significant (0.5%) amplitude changes
- trade reactions shall not be digital but proportional to price changes
- once a trade reaction is required consider using the next opposite foclogus regression extreme to trade
- features
  - use within minute amplitude (H-L) relative to median of last 15 min amplitude
  - use within minute volume relative to median of last 15 min volume
  - use consecutive minute movements piv(x) - piv(x-1) for the last 5 minutes
  - use last 3 5min regr grad
  - use the last 3 15min regr grad
  - use the last 1h regr grad
  - use the last 4h regr grad
  - use the mindist maxdist of the last 4h
  - use last relative price max after 5min regr max (relative price in relation to most recent piv)
  - use last relative price min after 5min regr min (relative price in relation to most recent piv)
  - use last relative price max after 15min regr max (relative price in relation to most recent piv)
  - use last relative price min after 15min regr min (relative price in relation to most recent piv)
- long targets
  - deviation threshold = DTHRS (e.g. 0,5%)
  - buy 
    - after DTHRS price increase after last minimum + 1 minute to buy
    - and still DTHRS gain to expect compared to next sell signal
  - sell
    - after DTHRS decrease after last maximum + one minute to sell

### what is a trend?

- a trend is a price development that allows to earn profit
- it has to be long enough to be detected and earn profit
- it has to be identified
  - a steep price development requires a quick investment reaction
  - a moderate price development enables looking for proper conditions to deinvest
- investments need to take place, which takes time while price slips
- the end of a trend need to be identified
  - can be identification of an opposite trend, which requires quick deinvestment actions
  - can be the absence of earning enough over a period of time, which enables looking for proper conditions to deinvest
- deinevstment needs to take place, which takes time while price slips

### option 1

- classifier has to have a chance to recognize a target situation
- for longbuy: let have a gain of x% (e.g. 1%) within a timerange (e.g. 1h) and then label a target of an additional gain (within the same timerange/ without going beyond the already gained) as target
  - what is the best timerange?
    - Identify the next x% gain over all coins with sufficient liquidity
    - check that is approx normal distribution (required?)
    - define forward looking timerange as 90% of distribution in that range
    - the timerange can be different between x% gain versus x% loss
  - what is the best grad per regression window?
    - with a look ahead of above timerange
    - distribution of grad without target in range
    - distribution of grad with target in range

### longbuy classifier option 2

- consider regression windows 5, 15, 60 minutes with regry and grad to folllow also short term changes under the assumption that y delta and grad are most important features
- consider the following constellations as longbuy targets:
  - a regression window can theoretically catch a gain of at least 1% (compare later with the constraint within 3h), which gives 0.5% to identify the trend
  - the gain starts 1 minute after regression line minimum and ends one minute after regression line maximum, which likely excludes a few minute peak that falls back afterwards
  - the gain period can start and end with a short regression window even if it has to leverage in between longer regression lines for supporting the trend to ignore intermediate local down bumps that
    - are smaller 1% fall back
    - don't fall lower than the start point
    - is bridged by one regression line with a positive grad during that period

### follow with short regression window as regressor

- construct targets to achieve, i.e. peaks to harvest of x% (e.g. 1%)
  - estimate with regressor the best suited regression window that ignores small peaks that cannot been detected but swings in on larger peaks that a regression wondow can follow 
  - if trend changes, follow trend and include in peaks when above target gain otherwise discard last local extreme
  - not only peaks of x% should be taget points but also changes that require a suddenchaneg in regression window (e.g. 4h required to ignore flat peaks but then change to 2minutes to follow a peak investment)
- adapt at every time mark to the shortest regression window that catches the next peak with its local extreme exceeding target gain
  - TO BE ELABORATED: loss = 0 if in correct direction and negative % delta if in opposite direction (e.g. falling instead of increasing) as well as full fee penalty for peak below target
  - advantage of many more learning examples -> avoid overadaptation by limiting training to extremes
  - using regressor instead of classifier

## Swing in approach

- always start with the smallest 5min regression and if no swing in then contunue to the next larger regression
- check whether one of the previous 2 swings between extremes had sufficient gain/loss to be trade worthy
  - 2 swings to catch large declines or gains with steps in between
- if so then trade the next swing of extremes and if not continue teh check with the next larger regression window
