"""Volatility and mean-reversion metrics for asset screening.

Implements the methods selected in the statistical estimation survey
(ticket #3 / branch research/survey-stat-methods):
  - Hurst Exponent via R/S analysis (hurst.compute_Hc)
  - Half-Life of mean reversion via OLS AR(1) on log prices
  - ADF p-value via statsmodels adfuller(autolag='AIC')
  - Daily and annualised standard deviation of log returns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from hurst import compute_Hc
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

# Minimum observations required for reliable Hurst estimation (per research survey).
_MIN_OBS_HURST = 100


class MetricResult(NamedTuple):
    """Container for a single asset × lookback metric set."""

    ticker: str
    lookback: str  # e.g. "1Y", "3Y", "5Y"
    rows: int
    daily_sd: float
    annual_sd: float
    hurst: float
    half_life: float  # days; inf when β ≥ 0
    adf_stat: float
    adf_pvalue: float


# ---------------------------------------------------------------------------
# Value Formatting Utilities
# ---------------------------------------------------------------------------

def format_half_life(hl: float) -> str:
    """Format half-life duration into readable string representation."""
    if np.isnan(hl) or np.isinf(hl) or hl > 9999 or hl <= 0:
        return "∞"
    return f"{hl:.1f}d"


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def daily_log_return_sd(prices: pd.Series) -> float:
    """Standard deviation of daily log returns."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return float(log_returns.std(ddof=1))


def annualised_sd(daily_sd: float, trading_days: int) -> float:
    """Annualise a daily SD given the asset's trading-day count per year.

    Crypto: 365, commodities: 252.
    """
    return daily_sd * np.sqrt(trading_days)


def hurst_exponent(prices: pd.Series) -> float:
    """Hurst exponent via R/S analysis on log prices.

    Returns NaN when the series is too short (< 100 observations).
    """
    log_prices = np.log(prices.dropna())
    if len(log_prices) < _MIN_OBS_HURST:
        logger.warning("Series too short for Hurst (%d < %d)", len(log_prices), _MIN_OBS_HURST)
        return float("nan")
    try:
        H, _c, _data = compute_Hc(log_prices.values, kind="price", simplified=True)
        return float(H)
    except (FloatingPointError, ValueError):
        pass
    # Fallback: full R/S (more sub-windows, avoids edge-case log10(0))
    try:
        H, _c, _data = compute_Hc(log_prices.values, kind="price", simplified=False)
        logger.info("Hurst fallback (simplified=False) succeeded: H=%.4f", H)
        return float(H)
    except (FloatingPointError, ValueError) as exc:
        logger.warning("Hurst computation failed on both modes: %s — returning NaN", exc)
        return float("nan")


def half_life(prices: pd.Series) -> float:
    """Half-life of mean reversion via OLS AR(1) on log prices.

    Returns inf when β ≥ 0 (no mean-reversion detected).
    """
    y = np.log(prices.dropna())
    y_lag = y.shift(1).dropna()
    y_diff = y.diff().dropna()

    # Align after shifting / differencing
    y_lag, y_diff = y_lag.align(y_diff, join="inner")

    X = sm.add_constant(y_lag)
    result = sm.OLS(y_diff, X).fit()

    beta = result.params.iloc[1]
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + beta))


def adf_test(prices: pd.Series) -> tuple[float, float]:
    """ADF unit-root test on log prices.

    Returns (adf_statistic, p_value).
    """
    log_prices = np.log(prices.dropna())
    result = adfuller(log_prices, autolag="AIC", result_object=False)
    return float(result[0]), float(result[1])


# ---------------------------------------------------------------------------
# Lookback slicing
# ---------------------------------------------------------------------------

_CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "XRP-USD"}

# Trading days per year used for annualisation.
TRADING_DAYS = {
    "crypto": 365,
    "commodity": 252,
}


def _asset_class(ticker: str) -> str:
    return "crypto" if ticker.upper() in _CRYPTO_TICKERS else "commodity"


def _calendar_days_for_lookback(lookback: str) -> int:
    """Approximate calendar days for a lookback label."""
    mapping = {"1Y": 365, "3Y": 3 * 365, "5Y": 5 * 365}
    return mapping.get(lookback, 365)


def slice_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    """Return the trailing *lookback* window from a DatetimeIndex DataFrame."""
    cal_days = _calendar_days_for_lookback(lookback)
    cutoff = df.index.max() - pd.Timedelta(days=cal_days)
    return df.loc[df.index > cutoff]


# ---------------------------------------------------------------------------
# Full matrix computation
# ---------------------------------------------------------------------------

CANDIDATE_TICKERS = ["BTC-USD", "ETH-USD", "XRP-USD", "GC=F", "SI=F", "CL=F", "BZ=F"]
LOOKBACKS = ["1Y", "3Y", "5Y"]


def compute_metric_row(ticker: str, prices: pd.Series, lookback: str) -> MetricResult:
    """Compute the full metric set for one ticker × lookback combination."""
    ac = _asset_class(ticker)
    td = TRADING_DAYS[ac]

    d_sd = daily_log_return_sd(prices)
    a_sd = annualised_sd(d_sd, td)
    h = hurst_exponent(prices)
    hl = half_life(prices)
    adf_s, adf_p = adf_test(prices)

    return MetricResult(
        ticker=ticker,
        lookback=lookback,
        rows=len(prices),
        daily_sd=d_sd,
        annual_sd=a_sd,
        hurst=h,
        half_life=hl,
        adf_stat=adf_s,
        adf_pvalue=adf_p,
    )


def compute_all(
    tickers: Sequence[str] | None = None,
    lookbacks: Sequence[str] | None = None,
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    """Compute the metric matrix across specified tickers and lookbacks.

    Returns a DataFrame with one row per (ticker, lookback).
    """
    from src.downloader import load_ticker_data

    target_tickers = tickers if tickers is not None else CANDIDATE_TICKERS
    target_lookbacks = lookbacks if lookbacks is not None else LOOKBACKS

    rows: list[MetricResult] = []
    for ticker in target_tickers:
        try:
            df = load_ticker_data(ticker, data_dir=data_dir)
        except Exception as exc:
            logger.warning("Failed to load data for ticker %s: %s", ticker, exc)
            continue

        for lb in target_lookbacks:
            window = slice_lookback(df, lb)
            if window.empty or len(window) < 10:
                logger.warning("Insufficient data for %s at lookback %s — skipping", ticker, lb)
                continue
            row = compute_metric_row(ticker, window["Close"], lb)
            rows.append(row)
            logger.info(
                "%s %s: daily_sd=%.6f annual_sd=%.4f H=%.4f HL=%.1f ADF_p=%.4f",
                ticker, lb, row.daily_sd, row.annual_sd, row.hurst, row.half_life, row.adf_pvalue,
            )

    return pd.DataFrame(rows)
