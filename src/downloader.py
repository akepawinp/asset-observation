"""Downloader module to fetch asset prices from yfinance and save locally as CSV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe index to clean, tz-naive dates named 'Date'."""
    if df.empty:
        return df

    cleaned = df.copy()
    if isinstance(cleaned.index, pd.DatetimeIndex):
        if cleaned.index.tz is not None:
            cleaned.index = cleaned.index.tz_localize(None)
        cleaned.index = cleaned.index.normalize()
    cleaned.index.name = "Date"
    return cleaned


def fetch_ticker_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "max",
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Fetch historical price data for a ticker using yfinance.

    Args:
        ticker: Symbol of the asset (e.g. 'AAPL', 'MSFT', 'BTC-USD').
        start: Start date string (YYYY-MM-DD). If provided, period is ignored.
        end: End date string (YYYY-MM-DD).
        period: Data period to download if start is None (e.g. '1y', '5y', 'max').
        interval: Data interval ('1d', '1wk', '1mo', etc.).
        auto_adjust: Whether to adjust OHLC automatically. Defaults to False
            so that both 'Close' and 'Adj Close' columns are retained.

    Returns:
        pd.DataFrame with normalized Date index.
    """
    yf_ticker = yf.Ticker(ticker.strip().upper())

    if start:
        df = yf_ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )
    else:
        df = yf_ticker.history(
            period=period,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )

    return _normalize_dataframe(df)


def get_ticker_csv_path(ticker: str, data_dir: str | Path = "data") -> Path:
    """Return the canonical CSV path for a ticker."""
    target_dir = Path(data_dir)
    return target_dir / f"{ticker.strip().upper()}.csv"


def load_ticker_data(ticker: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Load local CSV price data for a ticker if it exists.

    Args:
        ticker: Asset ticker symbol.
        data_dir: Directory containing CSV files.

    Returns:
        pd.DataFrame with parsed Date index.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    file_path = get_ticker_csv_path(ticker, data_dir)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file for ticker '{ticker}' not found at {file_path}")

    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    return _normalize_dataframe(df)


def save_ticker_data(
    ticker: str,
    data_dir: str | Path = "data",
    start: str | None = None,
    end: str | None = None,
    period: str | None = "max",
    interval: str = "1d",
    incremental: bool = True,
    force_full: bool = False,
    auto_adjust: bool = False,
) -> Path:
    """Download price data from yfinance and save to CSV in data_dir.

    If incremental=True and an existing CSV is found, only newer data since the
    latest saved date is fetched and merged (with deduplication), keeping existing history intact.

    Args:
        ticker: Asset symbol (e.g. 'AAPL', 'SPY').
        data_dir: Target directory path (default: 'data').
        start: Specific start date (YYYY-MM-DD). If supplied, incremental logic is bypassed.
        end: Specific end date (YYYY-MM-DD).
        period: Period when downloading full history (default: 'max').
        interval: Data interval (default: '1d').
        incremental: If True and file exists, fetch from latest date onward and merge.
        force_full: If True, ignore existing CSV and fetch full period/start.
        auto_adjust: Whether yfinance should auto-adjust OHLC.

    Returns:
        Path to the saved CSV file.

    Raises:
        ValueError: If no data could be fetched for the ticker.
    """
    symbol = ticker.strip().upper()
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = get_ticker_csv_path(symbol, target_dir)

    existing_df: pd.DataFrame | None = None
    if file_path.exists() and not force_full:
        try:
            existing_df = load_ticker_data(symbol, target_dir)
        except Exception as exc:
            logger.warning("Failed to read existing data for %s: %s. Fetching fresh data.", symbol, exc)
            existing_df = None

    if incremental and existing_df is not None and not existing_df.empty and start is None:
        last_date = existing_df.index.max()
        # Fetch from the last recorded date to catch potentially updated closing prices or new days
        fetch_start = last_date.strftime("%Y-%m-%d")
        logger.info("Incrementally updating %s from %s", symbol, fetch_start)

        new_df = fetch_ticker_history(
            symbol,
            start=fetch_start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )

        if new_df.empty:
            logger.info("No newer data found for %s; file is up-to-date.", symbol)
            return file_path

        # Combine existing and new data, deduplicating on Date index (preferring newest data)
        combined_df = pd.concat([existing_df, new_df])
        final_df = combined_df[~combined_df.index.duplicated(keep="last")].sort_index()
    else:
        logger.info("Fetching full data for %s (period=%s, start=%s, end=%s)", symbol, period, start, end)
        final_df = fetch_ticker_history(
            symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
        )

        if final_df.empty:
            raise ValueError(f"No price data found for ticker '{symbol}'. Check symbol or network connection.")

    final_df.to_csv(file_path, date_format="%Y-%m-%d")
    logger.info("Saved %d rows for %s to %s", len(final_df), symbol, file_path)
    return file_path


def update_ticker_data(ticker: str, data_dir: str | Path = "data") -> Path:
    """Convenience function to incrementally update an existing ticker's CSV data."""
    return save_ticker_data(ticker=ticker, data_dir=data_dir, incremental=True)


def fetch_and_save_tickers(
    tickers: Sequence[str],
    data_dir: str | Path = "data",
    start: str | None = None,
    end: str | None = None,
    period: str | None = "max",
    interval: str = "1d",
    incremental: bool = True,
    force_full: bool = False,
) -> dict[str, Path]:
    """Fetch and save data for multiple tickers.

    Args:
        tickers: Sequence of ticker symbols.
        data_dir: Target directory path.
        start: Optional start date string.
        end: Optional end date string.
        period: Optional period string.
        interval: Data interval.
        incremental: Whether to use incremental updates for existing files.
        force_full: Whether to overwrite existing files completely.

    Returns:
        Dict mapping ticker symbol to saved Path.
    """
    results: dict[str, Path] = {}
    for ticker in tickers:
        sym = ticker.strip().upper()
        try:
            path = save_ticker_data(
                ticker=sym,
                data_dir=data_dir,
                start=start,
                end=end,
                period=period,
                interval=interval,
                incremental=incremental,
                force_full=force_full,
            )
            results[sym] = path
        except Exception as exc:
            logger.error("Failed to save data for ticker '%s': %s", sym, exc)
            raise
    return results
