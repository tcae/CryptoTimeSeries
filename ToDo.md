
# Known issues

- /Users/tc/github/CryptoTimeSeries/CryptoXch/test/productionruntests.jl fails rpobably because it is still attached to Bybit.com. Mitigation: complete move to Bybit.eu and Kraken.com
- Complete package tests because CryptoXch is failing and Kraken not yet in

# 2026-04-22 Next steps
	- mk038 as basis for a decision classifier that only decides the longbuy/shortbuy decision
	- longbuy and shortbuy cancel each other, i.e. closes the opposite trend
	- the decision classifier will be trained on sequences of mk038 and the y distance from a window regression 
	- target is a trend that meets or exceeds x% gain
    - experiments can entail different gain levels or criteria concernign the y-position (e.g. below/above with y%) or regression window length