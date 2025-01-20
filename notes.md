# Here are just unordered notes

## 2021-08-19 How to go forward?

The project shows for some time for various private reasons no progress. Coming back to the source code shows me that I have to be careful not to over-engineer resutling in a too complex approach.
My target is short term:

- simple target labelling
- set up machine learning on the basis of artificial data (e.g. sine data) to check the base mechanism
- establish a visualization - probably one of: Plots, Dash, VegaLite, Makie

## Adaptation approach

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

### option 2

- construct targets to achieve, i.e. peaks to harvest
  - if trend changes, follow trend and include in peaks when above target gain otherwise discard last local extreme
- adapt at every time mark to the shortest regression window that catches the next peak with its local extreme exceeding target gain
  - advantage of many more learning examples -> avoid overadaptation by limiting training to extremes
  - using regressor instead of classifier
