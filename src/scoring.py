"""Composite Grid Score formulas and metric weighting prototypes.

Combines volatility (daily / annualised SD) with mean-reversion metrics
(Hurst exponent, Half-Life, ADF p-value) into single actionable scores
for ranking grid trading candidate assets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deployment Lanes and Roles
# ---------------------------------------------------------------------------

class DeploymentLane(str, Enum):
    """Target operational strategy lane for an asset."""

    LANE_A = "Lane A (Mean-Reversion Grid)"
    LANE_B = "Lane B (Zone-Concentration Entry)"
    OBSERVED_ONLY = "Observed Only"


class LaneRole(str, Enum):
    """Specific operational assignment within a deployment lane."""

    PRIMARY = "Primary Deploy"
    PARALLEL = "Parallel Deploy"
    LEARNING = "Learning / Short-Cycle"
    ACTIVE = "Active Deploy"
    OBSERVED = "Observed Only"


@dataclass
class RegimeEvaluation:
    """Regime stability and strategy lane classification for a single asset."""

    ticker: str
    lane: DeploymentLane
    role: LaneRole
    is_regime_consistent: bool
    verdict: str
    rationale: str
    risk_flag: str



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


# ---------------------------------------------------------------------------
# Regime Stability Evaluation & Deployment Lane Allocation
# ---------------------------------------------------------------------------

def evaluate_regime_stability(
    ticker: str,
    asset_df: pd.DataFrame,
    config: ScoringConfig | None = None,
) -> RegimeEvaluation:
    """Evaluate multi-horizon regime consistency and assign deployment lane.

    Checks:
    1. Regime Flips: Short-term (1Y) trending vs Long-term (3Y/5Y) mean-reverting,
       or vice versa, routes the asset to Observed Only.
    2. Stable Mean Reversion (Lane A): Stable anti-persistent/random walk Hurst
       across horizons with short/manageable half-life.
       - Sub-categorized into Primary, Parallel, or Learning (short-cycle) roles.
    3. Persistent Trend (Lane B): Consistent trending Hurst (> 0.55 / 0.60) or
       expanding half-life (> 250d), suited for directional zone entry.
    """
    cfg = config or DEFAULT_CONFIG
    t_df = asset_df[asset_df["ticker"] == ticker].copy()
    if t_df.empty:
        return RegimeEvaluation(
            ticker=ticker,
            lane=DeploymentLane.OBSERVED_ONLY,
            role=LaneRole.OBSERVED,
            is_regime_consistent=False,
            verdict="No metric data available",
            rationale="Empty metric record for ticker.",
            risk_flag="Missing data",
        )

    t_indexed = t_df.set_index("lookback")

    h_1y = t_indexed.loc["1Y", "hurst"] if "1Y" in t_indexed.index else np.nan
    h_3y = t_indexed.loc["3Y", "hurst"] if "3Y" in t_indexed.index else np.nan
    h_5y = t_indexed.loc["5Y", "hurst"] if "5Y" in t_indexed.index else np.nan

    hl_1y = t_indexed.loc["1Y", "half_life"] if "1Y" in t_indexed.index else np.nan
    hl_3y = t_indexed.loc["3Y", "half_life"] if "3Y" in t_indexed.index else np.nan
    hl_5y = t_indexed.loc["5Y", "half_life"] if "5Y" in t_indexed.index else np.nan

    mean_h = t_df["hurst"].mean()
    median_hl = t_df["half_life"].median()

    # Rule 1: Detect regime flips (e.g. ETH 1Y=0.642 trending vs 3Y/5Y <= 0.50 MR)
    is_flip = False
    if not np.isnan(h_1y) and not np.isnan(h_3y):
        # 1Y trending (> 0.58) while 3Y/5Y mean-reverting (<= 0.51)
        if h_1y >= 0.58 and h_3y <= 0.51:
            is_flip = True
        # 1Y mean-reverting (<= 0.46) while 3Y/5Y strongly trending (>= 0.65)
        elif h_1y <= 0.46 and h_3y >= 0.65:
            is_flip = True

    if is_flip:
        return RegimeEvaluation(
            ticker=ticker,
            lane=DeploymentLane.OBSERVED_ONLY,
            role=LaneRole.OBSERVED,
            is_regime_consistent=False,
            verdict="Regime flip across horizons",
            rationale=f"Hurst flips between 1Y ({h_1y:.3f}) and 3Y ({h_3y:.3f}); character unresolved.",
            risk_flag="Regime flip; unreliable grid assumptions",
        )

    # Rule 2: Persistent Trend (Lane B)
    # Characterized by strong trending 3Y/5Y Hurst (>= 0.60) or multi-horizon mean Hurst >= 0.57
    if (not np.isnan(h_3y) and h_3y >= 0.60) or mean_h >= 0.57:
        return RegimeEvaluation(
            ticker=ticker,
            lane=DeploymentLane.LANE_B,
            role=LaneRole.ACTIVE,
            is_regime_consistent=True,
            verdict="Stable persistent trend",
            rationale=f"Persistent trending Hurst (mean H={mean_h:.3f}, 3Y H={h_3y:.3f}) and long half-life ({median_hl:.1f}d).",
            risk_flag="High trend-blowout risk for MR grids; suitable for directional zone entry",
        )

    # Rule 3: Lane A (Mean Reversion Grid)
    # Check if 1Y and 3Y Hurst indicate mean-reverting or neutral character (<= 0.55)
    if (h_1y <= 0.55 and (np.isnan(h_3y) or h_3y <= 0.55)) or (h_1y <= 0.51 and hl_1y <= 60.0):
        # Check for half-life blowout on longer horizons (e.g. XRP-USD)
        if (hl_3y > 200.0 or hl_5y > 300.0) and hl_1y <= 70.0:
            return RegimeEvaluation(
                ticker=ticker,
                lane=DeploymentLane.LANE_A,
                role=LaneRole.LEARNING,
                is_regime_consistent=True,
                verdict="MR short-cycle; long-term half-life blowout",
                rationale=f"MR at 1Y (H={h_1y:.3f}, HL={hl_1y:.1f}d) but long-term HL expands (3Y={hl_3y:.1f}d).",
                risk_flag="Half-life expansion over long horizons",
            )
        # Stable MR across horizons (e.g. CL=F, BZ=F)
        if ticker in ["BZ=F", "UK-Oil", "BRENT", "BZ"]:
            role = LaneRole.PARALLEL
            verdict = "Stable MR, parallel commodity deploy"
            risk = "Geopolitical spread to WTI"
        else:
            role = LaneRole.PRIMARY
            verdict = "Stable MR, primary deploy"
            risk = "Commodity/macro supply shock"

        return RegimeEvaluation(
            ticker=ticker,
            lane=DeploymentLane.LANE_A,
            role=role,
            is_regime_consistent=True,
            verdict=verdict,
            rationale=f"Consistent anti-persistent/neutral Hurst (mean H={mean_h:.3f}) and stable half-life (median={median_hl:.1f}d).",
            risk_flag=risk,
        )

    # Default fallback
    return RegimeEvaluation(
        ticker=ticker,
        lane=DeploymentLane.OBSERVED_ONLY,
        role=LaneRole.OBSERVED,
        is_regime_consistent=True,
        verdict="Neutral / unclassified regime",
        rationale=f"Metrics do not meet strong MR or strong Trend thresholds (mean H={mean_h:.3f}).",
        risk_flag="Uncertain regime edge",
    )


def allocate_deployment_lanes(
    df: pd.DataFrame,
    config: ScoringConfig | None = None,
) -> dict[str, RegimeEvaluation]:
    """Evaluate and assign deployment lanes for all tickers in DataFrame."""
    cfg = config or DEFAULT_CONFIG
    tickers = df["ticker"].unique()
    allocations = {}
    for ticker in tickers:
        allocations[ticker] = evaluate_regime_stability(ticker, df, config=cfg)
    return allocations

