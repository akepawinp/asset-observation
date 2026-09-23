# Asset Observation

Tools for downloading, persisting, and observing historical asset prices using [yfinance](https://github.com/ranaroussi/yfinance).

## Features

- **Local CSV Storage**: Automatically persists asset price data into `data/{TICKER}.csv`.
- **OHLCV + Adjusted Close**: Keeps standard historical columns (`Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, etc.) with normalized `Date` indices (`YYYY-MM-DD`).
- **Incremental Updates**: Avoids re-downloading entire histories by querying only newer dates and merging/deduplicating existing data.
- **Python Module & CLI**: Reusable Python package `src` with both programmatic functions and command-line execution.

## Installation

Using `uv`:
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install -r pyproject.toml
```

## CLI Usage

Run `main.py` directly with one or more ticker symbols:

```bash
# Download historical data for single or multiple tickers
uv run python main.py AAPL MSFT SPY

# Specify historical period (default: max)
uv run python main.py NVDA --period 1y

# Specify a custom date range
uv run python main.py TSLA --start 2024-01-01 --end 2025-01-01

# Specify custom target folder
uv run python main.py BTC-USD --data-dir ./data/crypto

# Force full re-download (overwrite without incremental merge)
uv run python main.py AAPL --force-full
```

## Python API Usage

```python
from src import (
    save_ticker_data,
    update_ticker_data,
    fetch_and_save_tickers,
    load_ticker_data,
)

# 1. Fetch and save single ticker data
csv_path = save_ticker_data("AAPL", data_dir="data", period="max")
print(f"Saved to {csv_path}")

# 2. Incremental update (fetches new days and updates latest close)
update_ticker_data("AAPL", data_dir="data")

# 3. Batch download multiple tickers
paths = fetch_and_save_tickers(["AAPL", "MSFT", "GOOGL"], data_dir="data")

# 4. Load CSV data back into pandas DataFrame
df = load_ticker_data("AAPL", data_dir="data")
print(df.tail())
```

## Running Tests

Run the test suite using Python's built-in `unittest`:

```bash
uv run python -m unittest discover tests
```
