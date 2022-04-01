# CryptoTimeSeries

## Getting started

<!-- This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) -->
This code base is using the Julia Language to make a reproducible project named
> CryptoTimeSeries

It is authored by tcae.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:

   ```julia
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

## What is it all about?

This is a playground Machine Learning project that aims to predict the best trades using popular cryptocurrencies on Binance.

Be cautious and don't use it for real trading unless you know what to do. There is no backward compatibility intention, restructuring of project parts may happen without announcement. Any trade signal from this software is no recommendation to trade. Trading cryptocurrencies is high risk.

[Trading strategy considerations can be found here](strategies.md).

## Entry points

Following the DrWatson approach, each module is located in the src folder and has a corresponding '_test' unti test in the test folder.

- Main module is [Trade](src/trade.jl)
- Machine learning is made available by [Classify](src/classify.jl).
- [Targets](Targets/src/Targets.jl) provides target labels to train, evaluate and test machine learning approaches.
- [Features](Features/src/Features.jl) provides features as input for machine learning.
- [Ohlcv](Ohlcv/src/Ohlcv.jl) provides means to get locally stored historic Open High Low Close Volume data.
- [Exchange](src/exchange.jl) is the abstract layer to [MyBinance](src/Binance.jl)
- [Config](EnvConfig/src/EnvConfig.jl) provides common configuratioin items, like the set of cryptos to use or common folders

Scripts for investigations as well as the GUI are located in the scripts folder without a unit test

## Under construction

1. [simplegainperformance](scripts/simplegainperformance.jl) nbestgradientgain
2. [targets](src/targets.jl)
3. [notes](notes.md)

## to do

- don't cache day ohlcv in files but aggregate them from miute data : is probaly faster, reduces code and storage
  - change in assets.jl and cryptocockpit.jl

