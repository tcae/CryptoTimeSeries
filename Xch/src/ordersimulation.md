# Order simulation design

## maintain assetsa and orders as different structures

- assetsvolume is more often requested than order creation
  - assets as simple table
  - faster: order creation/change/cancellation need to update assets
  - less mistake prone: no redundancy, i.e. locked and borrowed amounts of open orders are not part of assets
  - approach: less mistake design and measure differnce of order volume requests versus order changes

## maintain only investmets as order pairs - go for it

- quote as an always open investment
- closed investments, i.e. buy and sell cycle done, are moved to closed investments
- investments challenge: buy - close is a n:m relation

### design

- only closed orders will be represented in the assets
  - there are only free and borrowed assets (all locked assets are open orders)
- the df gets 2 additional columns as index vectors (index = searchsortedfirst(sorted_strings, search_string) == 0 means not found)
  - basecoin as String15
  - tradestatus as enums with one of the following states
    - freeasset (close spot orders update the freeasset row and move into closed orders dataframe)
    - borrowedasset (close margin orders update the borrowedasset row and move into closed orders dataframe)
    - spotbuyopen
    - sportbuyclosed
    - sportbuycancelled
    - spotsellopen
    - spotsellclosed
    - spotsellcancelled
    - marginbuyopen - only accepted with sufficient free quote coverage or if it is reducing a borrowed amount
    - marginbuyclosed
    - marginbuycancelled
    - marginsellopen - only accepted with sufficient free quote coverage or if it is reducing a borrowed amount
    - marginsellclosed
    - marginsellcancelled
- an order change will result in a closed order an a new order with the same oid
- an order close will result
  - in an update of freeasset or borrowedasset
  - a order record move to closed orders dataframe
- an order cancellation will
  - NOT result in an update of freeasset or borrowedasset
  - a order record move to closed orders dataframe
- an assetvolume request will result
  - in a selection via basecoin and tradestatus (assuming a fixed quotecoin == USDT)
  - borrowed = sum(baseqty, basecoin, tradestatus in [marginbuyopen,marginsellopen] )
  - locked = sum(baseqty, basecoin, tradestatus == spotsellopen)
  - free(basecoin) = sum(baseqty, basecoin, tradestatus == freeasset) - locked
  - free(quotecoin) = sum(baseqty, quotecoin, tradestatus == freeasset) - sum((baseqty - executedqty) * limitprice, coin, tradestatus == spotbuyopen)
