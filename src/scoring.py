"""Composite Grid Score formulas and metric weighting prototypes.

Combines volatility (daily / annualised SD) with mean-reversion metrics
(Hurst exponent, Half-Life, ADF p-value) into single actionable scores
for ranking grid trading candidate assets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Central Configuration for Scoring Weights and Parameters
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """Configurable weights and tuning parameters for Composite Grid Score.

    Users can customize these weights anytime via Python code or CLI flags.
    """

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

    def validate(self) -> None:
        """Validate that weights are valid probabilities."""
        mr_sum = self.w_hurst + self.w_half_life + self.w_adf
        if not np.isclose(mr_sum, 1.0, atol=1e-3):
            raise ValueError(f"Q_MR weights must sum to 1.0 (got {mr_sum:.4f})")
        h_sum = sum(self.horizon_weights.values())
        if not np.isclose(h_sum, 1.0, atol=1e-3):
            raise ValueError(f"Horizon weights must sum to 1.0 (got {h_sum:.4f})")


DEFAULT_CONFIG = ScoringConfig()


# ---------------------------------------------------------------------------
# Individual Component Scaling Functions
# ---------------------------------------------------------------------------

def hurst_quality_factor(
    h: float,
    h_opt: float = 0.40,
    h_max: float = 0.70,
) -> float:
    """Evaluate mean-reversion quality from Hurst exponent.

    Returns a factor in [0.0, 1.0]:
      - H <= h_opt (0.40): 1.0 (strong anti-persistence)
      - H = 0.50: 0.67 (random walk threshold)
      - H >= h_max (0.70): 0.0 (persistent trending regime, toxic for grid)
    """
    if np.isnan(h):
        return 0.0
    if h <= h_opt:
        return 1.0
    if h >= h_max:
        return 0.0
    return float((h_max - h) / (h_max - h_opt))


def half_life_quality_factor(hl: float, tau: float = 90.0) -> float:
    """Evaluate mean-reversion speed from Half-Life (days).

    Uses exponential decay exp(-HL / tau).
      - HL = 20d: 0.80
      - HL = 45d: 0.61
      - HL = 90d: 0.37
      - HL = 300d: 0.036
      - HL = inf: 0.0
    """
    if np.isnan(hl) or hl <= 0.0 or np.isinf(hl):
        return 0.0
    return float(np.exp(-hl / tau))


def adf_quality_factor(p_value: float) -> float:
    """Evaluate stationarity confidence from ADF test p-value.

    Returns 1 - p_value, clipped to [0.0, 1.0].
      - p = 0.01: 0.99
      - p = 0.05: 0.95
      - p = 0.50: 0.50
      - p = 0.90: 0.10
    """
    if np.isnan(p_value):
        return 0.0
    return float(np.clip(1.0 - p_value, 0.0, 1.0))


def mean_reversion_quality(
    hurst: float,
    half_life: float,
    adf_pvalue: float,
    config: ScoringConfig | None = None,
    w_h: float | None = None,
    w_hl: float | None = None,
    w_adf: float | None = None,
) -> float:
    """Composite Mean-Reversion Quality Index Q_MR in [0.0, 1.0]."""
    cfg = config or DEFAULT_CONFIG
    weight_h = cfg.w_hurst if w_h is None else w_h
    weight_hl = cfg.w_half_life if w_hl is None else w_hl
    weight_adf = cfg.w_adf if w_adf is None else w_adf

    q_h = hurst_quality_factor(hurst, h_opt=cfg.h_opt, h_max=cfg.h_max)
    q_hl = half_life_quality_factor(half_life, tau=cfg.tau_half_life)
    q_adf = adf_quality_factor(adf_pvalue)
    return float(weight_h * q_h + weight_hl * q_hl + weight_adf * q_adf)


# ---------------------------------------------------------------------------
# Candidate Scoring Models
# ---------------------------------------------------------------------------

def score_power_ratio(
    sd: float,
    hurst: float,
    half_life: float,
    scale: float = 1000.0,
) -> float:
    """Candidate Model 1: Power-Law Ratio Index."""
    if np.isinf(half_life) or half_life <= 0 or np.isnan(half_life):
        return 0.0
    if np.isnan(hurst) or hurst >= 1.0 or hurst <= 0.0:
        return 0.0
    return float((sd * ((1.0 - hurst) ** 2) / np.sqrt(half_life)) * scale)


def score_exponential_ratio(
    sd: float,
    hurst: float,
    half_life: float,
    k: float = 4.0,
    scale: float = 1000.0,
) -> float:
    """Candidate Model 2: Exponential Penalty Ratio."""
    if np.isinf(half_life) or half_life <= 0 or np.isnan(half_life):
        return 0.0
    if np.isnan(hurst) or np.isnan(sd):
        return 0.0
    penalty = np.exp(-k * (hurst - 0.5))
    return float((sd * penalty / np.log1p(half_life)) * scale)


def score_two_pillar(
    sd: float,
    hurst: float,
    half_life: float,
    adf_pvalue: float,
    config: ScoringConfig | None = None,
    ref_sd: float | None = None,
) -> tuple[float, float, float]:
    """Candidate Model 3: Two-Pillar Multiplicative Model (Recommended).

    Pillar 1: Volatility Yield Potential (sd / ref_sd * 100, capped at max_vol_score)
    Pillar 2: Mean-Reversion Quality Q_MR in [0, 1].

    Composite Score = Volatility Potential * Q_MR.
    Returns (composite_score, vol_potential, q_mr).
    """
    cfg = config or DEFAULT_CONFIG
    r_sd = cfg.ref_sd if ref_sd is None else ref_sd
    if np.isnan(sd) or sd <= 0:
        return 0.0, 0.0, 0.0

    vol_potential = min(cfg.max_vol_score, (sd / r_sd) * 100.0)
    q_mr = mean_reversion_quality(hurst, half_life, adf_pvalue, config=cfg)
    composite = vol_potential * q_mr
    return float(composite), float(vol_potential), float(q_mr)


def score_weighted_additive_row(
    daily_sd: float,
    hurst: float,
    half_life: float,
    adf_pvalue: float,
    config: ScoringConfig | None = None,
    ref_sd: float | None = None,
    w_vol: float = 0.40,
    w_mr: float = 0.60,
) -> float:
    """Candidate Model 4: Weighted Additive Model (0 to 100)."""
    cfg = config or DEFAULT_CONFIG
    r_sd = cfg.ref_sd if ref_sd is None else ref_sd
    vol_score = min(100.0, (daily_sd / r_sd) * 100.0)
    q_mr = mean_reversion_quality(hurst, half_life, adf_pvalue, config=cfg)
    return float(w_vol * vol_score + w_mr * (q_mr * 100.0))


# ---------------------------------------------------------------------------
# Batch Evaluation & Comparison
# ---------------------------------------------------------------------------

def evaluate_candidate_models(
    df: pd.DataFrame,
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    """Evaluate all prototype models across the metric matrix DataFrame."""
    cfg = config or DEFAULT_CONFIG
    results: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        d_sd = row["daily_sd"]
        a_sd = row["annual_sd"]
        h = row["hurst"]
        hl = row["half_life"]
        adf_p = row["adf_pvalue"]

        s_power = score_power_ratio(d_sd, h, hl)
        s_exp = score_exponential_ratio(d_sd, h, hl)
        s_two_pillar, vol_pot, q_mr = score_two_pillar(d_sd, h, hl, adf_p, config=cfg)
        s_additive = score_weighted_additive_row(d_sd, h, hl, adf_p, config=cfg)

        results.append({
            "ticker": row["ticker"],
            "lookback": row["lookback"],
            "daily_sd": d_sd,
            "annual_sd": a_sd,
            "hurst": h,
            "half_life": hl,
            "adf_pvalue": adf_p,
            "q_mr": q_mr,
            "vol_potential": vol_pot,
            "score_two_pillar": s_two_pillar,
            "score_additive": s_additive,
            "score_power_ratio": s_power,
            "score_exp_ratio": s_exp,
        })

    return pd.DataFrame(results)


def compute_aggregate_rankings(
    scored_df: pd.DataFrame,
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    """Compute multi-horizon aggregate scores."""
    cfg = config or DEFAULT_CONFIG
    weights = cfg.horizon_weights

    records = []
    tickers = scored_df["ticker"].unique()
    for t in tickers:
        t_df = scored_df[scored_df["ticker"] == t].set_index("lookback")
        if not all(lb in t_df.index for lb in weights):
            continue
        agg_two_pillar = sum(t_df.loc[lb, "score_two_pillar"] * w for lb, w in weights.items())
        agg_additive = sum(t_df.loc[lb, "score_additive"] * w for lb, w in weights.items())
        agg_q_mr = sum(t_df.loc[lb, "q_mr"] * w for lb, w in weights.items())
        mean_daily_sd = t_df["daily_sd"].mean()
        mean_hurst = t_df["hurst"].mean()
        mean_hl = t_df["half_life"].median()

        records.append({
            "ticker": t,
            "agg_two_pillar": agg_two_pillar,
            "agg_additive": agg_additive,
            "agg_q_mr": agg_q_mr,
            "mean_daily_sd": mean_daily_sd,
            "mean_hurst": mean_hurst,
            "median_half_life": mean_hl,
        })

    agg_df = pd.DataFrame(records).sort_values(by="agg_two_pillar", ascending=False).reset_index(drop=True)
    agg_df["rank"] = agg_df.index + 1
    return agg_df
