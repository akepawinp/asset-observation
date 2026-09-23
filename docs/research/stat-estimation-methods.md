# Survey of Statistical Estimation Methods

This document surveys statistical estimation methods for evaluating mean reversion, random walk, or trending behaviors in daily financial data. It specifically addresses:
1. Hurst Exponent ($H$)
2. Half-Life of Mean Reversion
3. Augmented Dickey-Fuller (ADF) Stationarity Test

---

## 1. Hurst Exponent ($H$)

### Mathematical Formula / Model
The Hurst exponent is a measure of the long-term memory of a time series. A common estimation approach is Rescaled Range (R/S) analysis. 
The expected value of the rescaled range follows the power law:
$$ \mathbb{E}\left[\frac{R(n)}{S(n)}\right] = C n^H $$
Where:
- $R(n)$ is the range of the first $n$ cumulative deviations from the mean.
- $S(n)$ is the standard deviation of the series of size $n$.
- $n$ is the observation window size.
- $H$ is the Hurst exponent.

Assumptions: Assumes a stationary underlying process in terms of variance over the analyzed sub-windows. 

Interpretation:
- $H < 0.5$: Mean-reverting (anti-persistent)
- $H \approx 0.5$: Random walk (Brownian motion)
- $H > 0.5$: Trending (persistent)

### Preprocessing
Inputs should typically be the **log prices** ($ \ln(P_t) $) if the method internally calculates differences (as with `kind='price'` in the `hurst` package). Using log prices mitigates the exponential growth and variance scaling typical of financial assets over 3-5 year horizons.

### Python Implementation
**Library:** `hurst`
```python
from hurst import compute_Hc
import numpy as np

# Assuming `prices` is a pandas Series of daily closing prices
log_prices = np.log(prices.dropna())

# kind='price' indicates the function should compute the returns/differences internally
H, c, data = compute_Hc(log_prices, kind='price', simplified=True)

print(f"Hurst Exponent: {H:.4f}")
```

### Robustness Notes
- **Sample Size Bias:** Small samples can severely bias $H$ towards 0.5. At least 100-250 observations are recommended. For a 3-5 year horizon, daily data yields ~750 to 1800 points, which is sufficiently robust.
- **Structural Breaks:** Shifts in market regimes (e.g., Covid-19 crash) can distort $H$. Calculating a rolling Hurst exponent (e.g., 100-252 day window) is often more informative than a static lifetime metric.
- **Crypto vs Commodities:** Crypto markets operate 365 days a year, while commodities operate ~252. Adjust rolling window parameters to match calendar equivalents (e.g., 365 for 1-year crypto, 252 for 1-year commodity).

### Alternatives Considered
- **Detrended Fluctuation Analysis (DFA):** More robust to non-stationarity and local trends than R/S, but slightly more complex to implement and interpret.
- **Higuchi's Fractal Dimension:** Better for high-frequency or noisy data, less common in traditional finance contexts compared to R/S.

---

## 2. Half-Life of Mean Reversion

### Mathematical Formula / Model
The half-life measures how long it takes for a process to revert halfway back to its mean following a shock. It is commonly derived by modeling the price series as an Ornstein-Uhlenbeck (OU) process.

The discrete-time equivalent of an OU process is an AR(1) model:
$$ y_t - y_{t-1} = \alpha + \beta y_{t-1} + \epsilon_t $$
Where $y_t$ is the price series (or log price).
The speed of mean reversion $\theta \approx -\ln(1+\beta)$ (assuming $\Delta t = 1$ day).
The Half-Life is:
$$ \text{Half-Life} = \frac{\ln(2)}{\theta} = -\frac{\ln(2)}{\ln(1+\beta)} $$

### Preprocessing
The model should be fit on **log prices** ($ \ln(P_t) $) or a **spread** if analyzing pairs. The dependent variable is the difference in log prices (log returns), and the independent variable is the lagged log price. Detrending is not strictly necessary unless there is a strong deterministic trend that overshadows the stochastic reversion.

### Python Implementation
**Library:** `statsmodels` (via OLS)
```python
import numpy as np
import statsmodels.api as sm

# y is a pandas Series of log prices
y_lag = y.shift(1).dropna()
y_diff = y.diff().dropna()

# Align data
y_lag, y_diff = y_lag.align(y_diff, join='inner')

# Add intercept
X = sm.add_constant(y_lag)
model = sm.OLS(y_diff, X)
res = model.fit()

# Beta is the coefficient on the lagged price
beta = res.params.iloc[1]

# Calculate Half-Life
if beta < 0:
    half_life = -np.log(2) / np.log(1 + beta)
else:
    half_life = np.inf # Not mean reverting

print(f"Half-Life: {half_life:.2f} days")
```

### Robustness Notes
- **Intercept Necessity:** Omitting the intercept forces the mean reversion target to 0, which breaks the estimate. `sm.add_constant` is mandatory.
- **Beta bounds:** If $\beta \ge 0$, the series is not mean-reverting, rendering half-life undefined or infinite.
- **Economic viability:** A half-life < 1 day is usually just microstructure noise. A half-life > 1 year suggests structural non-stationarity rather than tradable mean-reversion.

### Alternatives Considered
- **Kalman Filter:** Allows for dynamic/time-varying mean reversion rates and means. Much more robust for pairs trading, but computationally intensive and overkill for simple univariate asset screening.
- **Variance Ratio Test:** Non-parametric alternative but doesn't directly yield a "days to revert" metric.

---

## 3. Augmented Dickey-Fuller (ADF) Stationarity Test

### Mathematical Formula / Model
The ADF test checks for the presence of a unit root (non-stationarity). The test fits the regression:
$$ \Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t $$
- Null Hypothesis ($H_0$): $\gamma = 0$ (the series has a unit root, is non-stationary).
- Alternative Hypothesis ($H_1$): $\gamma < 0$ (the series is stationary or trend-stationary).

### Preprocessing
Input should be **log prices** ($ \ln(P_t) $) if you want to test if the asset price itself is mean-reverting. However, pure asset prices almost always fail to reject $H_0$. Thus, ADF is typically applied to **spreads** (in pairs trading) or **log returns** to confirm stationarity of the returns process. If analyzing structural stationarity of the asset, input the log price directly. 

### Python Implementation
**Library:** `statsmodels.tsa.stattools.adfuller`
```python
from statsmodels.tsa.stattools import adfuller
import numpy as np

# y is a pandas Series of log prices or log returns
result = adfuller(y.dropna(), autolag='AIC')

adf_stat = result[0]
p_value = result[1]
critical_values = result[4]

print(f"ADF Statistic: {adf_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H0: Series is stationary (mean-reverting).")
else:
    print("Fail to reject H0: Series is non-stationary (has a unit root).")
```

### Robustness Notes
- **Lag Selection:** By using `autolag='AIC'`, the test automatically determines the optimal number of lagged differences ($p$) to include, avoiding arbitrary lag selection which can heavily skew results.
- **Sensitivity:** The ADF test is notorious for low statistical power (high Type II error). It may fail to reject the null hypothesis for a series that is only weakly mean-reverting.
- **Trend Spec:** Be mindful of the regression specification. Default `adfuller` uses a constant but no trend (`regression='c'`). If testing a strongly trending asset, a constant + trend (`regression='ct'`) might be necessary.

### Alternatives Considered
- **Phillips-Perron (PP) Test:** Robust to unspecified autocorrelation and heteroskedasticity in the errors without needing to specify lag lengths. 
- **KPSS Test:** Reverses the hypotheses ($H_0$ is stationarity). Often used in tandem with ADF to cross-validate results (confirming both rejects unit root and fails to reject stationarity).

---

## Final Recommendation Table

| Metric | Library & Function | Preprocessing | Key Parameters |
|--------|---------------------|---------------|----------------|
| **Hurst ($H$)** | `hurst.compute_Hc` | Log Prices ($ \ln(P) $) | `kind='price'`, `simplified=True` |
| **Half-Life** | `statsmodels.api.OLS` | Lagged Log Prices & Log Returns | Use `sm.add_constant()`, fit AR(1) |
| **ADF Test** | `statsmodels.tsa.stattools.adfuller` | Log Prices ($ \ln(P) $) | `autolag='AIC'` |
