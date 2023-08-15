# CryptoTimeSeries

## Getting started

<!-- This code base is using the Julia Language ) -->
This code base is using the Julia Language to make a reproducible project named
> CryptoTimeSeries

It is authored by tcae.

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:

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

Each module is located in the src folder and has a corresponding '_test' unti test in the test folder.

- The main module is [Trade](Trade/src/Trade.jl) that is setup via [trade_test](Trade/test/trade_test.jl), which runs a tradeloop but applies [TradingStrategy](TradingStrategy/src/TradingStrategy.jl) to determine buy and sell actions
- Machine learning is made available by [Classify](Classify/src/Classify.jl).
- [Targets](Targets/src/Targets.jl) provides target labels to train, evaluate and test machine learning approaches.
- [Features](Features/src/Features.jl) provides OHLCV derived features as input for machine learning and trading.
- [Ohlcv](Ohlcv/src/Ohlcv.jl) provides means to get locally stored historic Open High Low Close Volume data.
- [CryptoXch](CryptoXch/src/CryptoXch.jl) is the abstract layer to [Bybit](Bybit/src/Bybit.jl)
- [Assets](Assets/src/Assets.jl) is selecting those assets from the exchange that are compliant to certain criteria that makes them a trading candidate.
- [EnvConfig](EnvConfig/src/EnvConfig.jl) provides common configuratioin items, like the set of cryptos to use or common folders
- [TestOhlcv](TestOhlcv/src/TestOhlcv.jl) provides periodic OHLCV patterns to test implementations.

Scripts for investigations as well as the GUI ([cryptocockpit](scripts/cryptocockpit.jl)) are located in the scripts folder without a unit test

## Under construction

1. [simplegainperformance](scripts/simplegainperformance.jl) nbestgradientgain
2. [targets](src/targets.jl)
3. [notes](notes.md)

## to do

- Trade/preparetradecache! should use Assets.jl instead of creating it's own asset selection

## OHLCV file transfer

- create tar file of ohlcv files from ~/crypto/Features:
  - cd ~/crypto/Features
  - tar -czvf ohlcv.tar.gz *.jdf
  - or
  - tar -czvf ohlcv.tar.gz ~/crypto/Features/*.jdf
- list files in tar file ohlcv.tar.gz:
  - tar -ztvf ohlcv.tgz
- extract files from ohlcv.tar.gz to ~/crypto/Features:
  - tar -xzvf ohlcv.tar.gz -C ~/crypto/Features
- split a big file into 1.5GB chunks
  - split -b 700m ohlcv.tar.gz ohlcv.tar.gz.parts_
  - creates:
    - -rw-rw-r--   1 tor tor 2955474895 jun 10 17:40  ohlcv.tar.gz
    - -rw-rw-r--   1 tor tor 1572864000 jun 10 17:43  ohlcv.tar.gz.parts_aa
    - -rw-rw-r--   1 tor tor 1382610895 jun 10 17:43  ohlcv.tar.gz.parts_ab
- combine split files
  - cat ohlcv.tar.gz.parts_* > ohlcv.tar.gz
