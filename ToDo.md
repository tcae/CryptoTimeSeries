
# Known issues

- check memory usage, which seems at least with tradesim very high
	- memory break down
	- is classifier loaded once or per symbol?
	- what is the length of OHLCV history that is maintained in memeory?
- multiple orders per symbol
- frequent order change
- what is the log semantic of opened/closed? Is it order placed or order filled?
- why are so many warnings? I don't believe those are all race conditions
- liquidity check too relaxed?

# To be observed

- positions shall apply to constraints as documented in docs/tradereal-risk-constraints-overview-2026-06-03.md

# intent

- max 1 long and 1 short order per symbol

