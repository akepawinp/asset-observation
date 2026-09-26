# Asset Observation

Quantitative Asset Observation & Mean-Reversion Grid Screening Pipeline.

Downloads, processes, evaluates, and allocates candidate assets into operational strategy lanes (Lane A Mean-Reversion Grid, Lane B Zone-Concentration Entry, or Observed-Only) using statistical memory estimation and Two-Pillar Multiplicative Composite Grid Scoring.

## Features

- **Automated Data Ingestion**: Daily OHLCV data fetched and incrementally synced from Yahoo Finance.
- **Statistical Estimator Suite**: Hurst Exponent (R/S analysis), Ornstein-Uhlenbeck Half-Life AR(1) decay, ADF stationarity test, and annualized volatility.
- **Two-Pillar Multiplicative Scoring**: Multiplicative volatility-gating scoring model prioritizing oscillating assets and penalizing runaway trends.
- **Multi-Horizon Regime Stability Analysis**: Cross-lookback (1Y, 3Y, 5Y) consistency evaluation detecting regime flips.
- **Operational Strategy Lane Allocation**: Deterministic routing into Lane A, Lane B, or Observed-Only.
- **Unified CLI & Reporting**: CLI runner with rich stdout tables, CSV outputs, and Markdown reports.

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

### 1. End-to-End Pipeline Runner (`run`)

Execute the full asset observation, scoring, and allocation workflow:

```bash
# Run complete pipeline with default candidate tickers and lookbacks
uv run python main.py run

# Run with fresh Yahoo Finance download before analysis
uv run python main.py run --fetch

# Run with custom scoring weights (e.g. 60% Hurst, 30% Half-Life, 10% ADF)
uv run python main.py run --w-hurst 0.60 --w-half-life 0.30 --w-adf 0.10

# Run with custom lookbacks and tickers
uv run python main.py run --tickers CL=F BZ=F XRP-USD --lookbacks 1Y 3Y

# Custom output destination and reference volatility benchmark
uv run python main.py run --reports-dir custom_reports --ref-sd 0.025
```

### 2. Historical Data Downloader (`fetch`)

Download or update price series:

```bash
# Fetch default candidate assets
uv run python main.py fetch

# Fetch specific tickers
uv run python main.py fetch AAPL MSFT BTC-USD --period 2y
```

### 3. Metric Computation (`metrics`)

Compute volatility and statistical memory metrics:

```bash
uv run python main.py metrics --output reports/metric_matrix.csv
```

### 4. Scoring & Rankings (`score`)

Evaluate Composite Grid Scores from an existing metric matrix:

```bash
uv run python main.py score --input reports/metric_matrix.csv
```

## Python API Usage

```python
from src.pipeline import PipelineConfig, run_pipeline
from src.scoring import ScoringConfig

# Configure and run the complete pipeline
config = PipelineConfig(
    scoring_config=ScoringConfig(
        w_hurst=0.50,
        w_half_life=0.35,
        w_adf=0.15,
        ref_sd=0.03,
    )
)
result = run_pipeline(config)

print(result.aggregate_rankings_df)
for ticker, eval_obj in result.regime_allocations.items():
    print(f"{ticker}: {eval_obj.lane.value} ({eval_obj.role.value}) - {eval_obj.verdict}")
```

## Running Tests

Run the test suite:

```bash
uv run python -m unittest discover tests
```
