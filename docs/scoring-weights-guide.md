# Composite Grid Score: Weight Customization Guide

This guide explains how to customize and adjust the weights, benchmarks, and decay parameters for the Composite Grid Score formula.

---

## 1. Where Weights Are Defined

All default weights and tuning parameters live in [`src/scoring.py`](file:///home/akepawinp/Documents/Asset%20Observation/src/scoring.py) inside the [`ScoringConfig`](file:///home/akepawinp/Documents/Asset%20Observation/src/scoring.py#L26-L56) dataclass:

```python
@dataclass
class ScoringConfig:
    # --- Mean-Reversion Quality (Q_MR) Weights (Must sum to 1.0) ---
    w_hurst: float = 0.50       # Weight for Hurst anti-persistence vs trend
    w_half_life: float = 0.35   # Weight for Half-Life decay speed
    w_adf: float = 0.15         # Weight for ADF unit-root rejection confidence

    # --- Volatility Benchmark & Cap ---
    ref_sd: float = 0.03        # 3.0% daily SD benchmark (100% VolScore)
    max_vol_score: float = 150.0  # Cap on volatility score

    # --- Hurst Quality Thresholds ---
    h_opt: float = 0.40         # H <= h_opt receives 1.0 (strong anti-persistence)
    h_max: float = 0.70         # H >= h_max receives 0.0 (persistent trending regime)

    # --- Half-Life Decay Scale ---
    tau_half_life: float = 90.0  # Characteristic decay scale in days: exp(-HL / tau)

    # --- Multi-Horizon Aggregation Weights ---
    horizon_weights: dict[str, float] = field(
        default_factory=lambda: {"1Y": 0.30, "3Y": 0.50, "5Y": 0.20}
    )
```

---

## 2. Option A: Command-Line Flags (Fastest for Experiments)

You can override weights directly when running [`scripts/run_scoring_prototype.py`](file:///home/akepawinp/Documents/Asset%20Observation/scripts/run_scoring_prototype.py):

```bash
# Example 1: Prioritize Hurst memory more heavily (60% Hurst, 30% Half-Life, 10% ADF)
PYTHONPATH=. .venv/bin/python scripts/run_scoring_prototype.py \
  --w-hurst 0.60 \
  --w-half-life 0.30 \
  --w-adf 0.10

# Example 2: Adjust volatility benchmark (e.g. 2.5% daily SD benchmark)
PYTHONPATH=. .venv/bin/python scripts/run_scoring_prototype.py \
  --ref-sd 0.025

# Example 3: Change horizon aggregation weights (e.g. 60% 3Y, 20% 1Y, 20% 5Y)
PYTHONPATH=. .venv/bin/python scripts/run_scoring_prototype.py \
  --w-3y 0.60 \
  --w-1y 0.20 \
  --w-5y 0.20
```

---

## 3. Option B: In Python Code

You can instantiate a custom `ScoringConfig` object and pass it to any scoring function:

```python
from src.scoring import ScoringConfig, evaluate_candidate_models, compute_aggregate_rankings
import pandas as pd

df = pd.read_csv("reports/metric_matrix.csv")

# Create custom configuration
my_config = ScoringConfig(
    w_hurst=0.60,
    w_half_life=0.30,
    w_adf=0.10,
    ref_sd=0.025,
    tau_half_life=60.0,
    horizon_weights={"1Y": 0.25, "3Y": 0.50, "5Y": 0.25},
)

# Evaluate
scored_df = evaluate_candidate_models(df, config=my_config)
rankings = compute_aggregate_rankings(scored_df, config=my_config)
print(rankings)
```

---

## 4. Option C: Modifying Defaults Permanently

To change project defaults permanently:
1. Open [`src/scoring.py`](file:///home/akepawinp/Documents/Asset%20Observation/src/scoring.py).
2. Edit the field default values in `ScoringConfig`.
3. Run `PYTHONPATH=. .venv/bin/python scripts/run_scoring_prototype.py` to regenerate `reports/candidate_scores.csv` and `reports/aggregate_rankings.csv`.
