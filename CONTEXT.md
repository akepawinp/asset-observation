# Domain Context

Glossary and ubiquitous language for asset observation, quantitative analysis, and trading research.

## Terms

### Candidate Assets
The seven target financial instruments evaluated for grid trading:
- **BTC/USDT** (proxy: `BTC-USD` on Yahoo Finance)
- **ETH/USDT** (proxy: `ETH-USD` on Yahoo Finance)
- **XRP/USDT** (proxy: `XRP-USD` on Yahoo Finance)
- **Gold Spot** (proxy: `GC=F` Gold Futures)
- **Silver Spot** (proxy: `SI=F` Silver Futures)
- **US Oil** (proxy: `CL=F` WTI Crude Oil Futures)
- **UK Oil** (proxy: `BZ=F` Brent Crude Oil Futures)

### Grid Trading
An algorithmic trading strategy that places laddered buy and sell limit orders above and below a baseline price within a predetermined range. It profits from repetitive oscillatory price swings and suffers severe inventory drawdowns in sustained non-reverting trends.

### Mean Reversion
The quantitative property of an asset's price or returns to return to a mean or central equilibrium value over time.

### Hurst Exponent ($H$)
A statistical metric measuring time-series memory:
- $H < 0.5$: Mean-reverting (anti-persistent) series
- $H \approx 0.5$: Geometric random walk / Brownian motion
- $H > 0.5$: Trending (persistent) series

### Half-Life of Mean Reversion
The expected duration (in trading days) required for a price shock or deviation from the mean to decay by 50%, estimated via an Ornstein-Uhlenbeck AR(1) process.

### Augmented Dickey-Fuller (ADF) Test
A statistical hypothesis test assessing unit root presence ($p < 0.05$ rejects unit root, indicating stationarity / mean-reversion).

### Daily and Annualized Standard Deviation (SD)
Measures of price return dispersion. Daily SD ($\sigma_{\text{daily}}$) is the standard deviation of logarithmic daily returns. Annualized SD ($\sigma_{\text{annual}}$) scales daily SD by $\sqrt{252}$ for traditional commodities or $\sqrt{365}$ for 24/7 crypto markets.

### Composite Grid Score
A composite ranking index designed to evaluate asset suitability for grid trading, scaling favorably with higher return volatility ($\sigma$) and unfavorably with persistence ($H$) and long mean-reversion half-life.
