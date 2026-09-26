"""Unified Multi-Horizon Analysis and Reporting Pipeline Orchestrator.

Orchestrates the quantitative asset selection workflow:
1. (Optional) Ingest / update daily OHLCV price series via yfinance.
2. Compute volatility and mean-reversion metrics across 1Y, 3Y, and 5Y horizons.
3. Evaluate Two-Pillar Multiplicative and benchmark scoring models.
4. Calculate multi-horizon aggregate scores and rankings.
5. Classify regime stability and allocate candidate assets into operational strategy lanes.
6. Export structured CSV files and human-readable Markdown summary reports.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.downloader import fetch_and_save_tickers, load_ticker_data
from src.metrics import (
    CANDIDATE_TICKERS,
    LOOKBACKS,
    compute_metric_row,
    slice_lookback,
)
from src.scoring import (
    DEFAULT_CONFIG,
    DeploymentLane,
    RegimeEvaluation,
    ScoringConfig,
    allocate_deployment_lanes,
    compute_aggregate_rankings,
    evaluate_candidate_models,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration options for pipeline execution."""

    tickers: list[str] = field(default_factory=lambda: list(CANDIDATE_TICKERS))
    lookbacks: list[str] = field(default_factory=lambda: list(LOOKBACKS))
    data_dir: str | Path = "data"
    reports_dir: str | Path = "reports"
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)
    fetch_fresh: bool = False
    start_date: str | None = None
    end_date: str | None = None
    force_full_download: bool = False
    generate_markdown: bool = True
    metric_matrix_filename: str = "metric_matrix.csv"
    candidate_scores_filename: str = "candidate_scores.csv"
    aggregate_rankings_filename: str = "aggregate_rankings.csv"
    report_filename: str = "final_asset_selection_report.md"


@dataclass
class PipelineResult:
    """Artifacts and structured results produced by pipeline execution."""

    metric_matrix_df: pd.DataFrame
    candidate_scores_df: pd.DataFrame
    aggregate_rankings_df: pd.DataFrame
    regime_allocations: dict[str, RegimeEvaluation]
    markdown_report: str
    saved_files: dict[str, Path]


def compute_pipeline_metrics(
    tickers: Sequence[str],
    lookbacks: Sequence[str],
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    """Compute the multi-horizon metric matrix for specified tickers."""
    rows = []
    for ticker in tickers:
        try:
            df = load_ticker_data(ticker, data_dir=data_dir)
        except Exception as exc:
            logger.warning("Failed to load data for ticker %s: %s", ticker, exc)
            continue

        for lb in lookbacks:
            window = slice_lookback(df, lb)
            if window.empty or len(window) < 10:
                logger.warning("Insufficient data for %s at lookback %s", ticker, lb)
                continue
            row = compute_metric_row(ticker, window["Close"], lb)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def generate_markdown_report(
    metric_matrix_df: pd.DataFrame,
    candidate_scores_df: pd.DataFrame,
    aggregate_rankings_df: pd.DataFrame,
    regime_allocations: dict[str, RegimeEvaluation],
    scoring_config: ScoringConfig,
    report_date: str | None = None,
) -> str:
    """Format and generate the human-readable Markdown summary report."""
    now_str = report_date or datetime.date.today().strftime("%Y-%m-%d")

    # Group allocations by lane
    lane_a = [eval_obj for eval_obj in regime_allocations.values() if eval_obj.lane == DeploymentLane.LANE_A]
    lane_b = [eval_obj for eval_obj in regime_allocations.values() if eval_obj.lane == DeploymentLane.LANE_B]
    observed = [eval_obj for eval_obj in regime_allocations.values() if eval_obj.lane == DeploymentLane.OBSERVED_ONLY]

    lane_a_str = ", ".join(e.ticker for e in lane_a) if lane_a else "None"
    lane_b_str = ", ".join(e.ticker for e in lane_b) if lane_b else "None"
    observed_str = ", ".join(e.ticker for e in observed) if observed else "None"

    # Multi-horizon weights display
    hw_str = " + ".join(f"{int(w * 100)}% × {lb}" for lb, w in scoring_config.horizon_weights.items())

    lines: list[str] = [
        "# Asset Selection & Strategy Allocation Report",
        "**Quantitative Asset Observation & Mean-Reversion Grid Screening**",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "Candidate assets have been quantitatively evaluated across volatility, mean-reversion persistence, half-life decay, and regime stability across 1Y, 3Y, and 5Y horizons:",
        "",
        "| Strategy Tier | Assets | Operational Objective |",
        "|---|---|---|",
        f"| **Lane A (Mean-Reversion Grid)** | {lane_a_str} | Laddered limit orders in range; profits from oscillatory swings |",
        f"| **Lane B (Zone-Concentration Entry)** | {lane_b_str} | Single/few entries at support zones; captures directional trend |",
        f"| **Observed Only** | {observed_str} | Disqualified from automated execution pending regime stabilization |",
        "",
        "---",
        "",
        "## 1. Scoring Methodology",
        "",
        "### Two-Pillar Multiplicative Model",
        "```",
        "CompositeGridScore = VolPotential × Q_MR",
        "",
        f"VolPotential = min((daily_σ / {scoring_config.ref_sd:.2f}) × 100, {scoring_config.max_vol_score:.0f})",
        f"Q_MR         = {scoring_config.w_hurst:.2f} × HurstScore + {scoring_config.w_half_life:.2f} × HalfLifeScore + {scoring_config.w_adf:.2f} × ADFScore",
        "```",
        "",
        "- **HurstScore**: Linear decay between $H \\le 0.40 \\rightarrow 1.0$ (strong MR) and $H \\ge 0.70 \\rightarrow 0.0$ (strong trend).",
        f"- **HalfLifeScore**: $\\exp(-HL / {scoring_config.tau_half_life:.0f}d)$ exponential decay factor.",
        "- **ADFScore**: $1 - \\min(p, 1.0)$ unit root rejection confidence.",
        f"- **Multi-Horizon Aggregation**: {hw_str}",
        "",
        "---",
        "",
        "## 2. Volatility & Mean-Reversion Metric Matrix",
        "",
        "| Asset | Lookback | Daily σ | Annual σ | Hurst (H) | Half-Life | ADF p-value |",
        "|---|---|---|---|---|---|---|",
    ]

    for _, row in metric_matrix_df.iterrows():
        hl_str = "∞" if (np.isinf(row["half_life"]) or row["half_life"] > 9999) else f"{row['half_life']:.1f}d"
        lines.append(
            f"| {row['ticker']} | {row['lookback']} | {row['daily_sd'] * 100:.2f}% | "
            f"{row['annual_sd'] * 100:.1f}% | {row['hurst']:.3f} | {hl_str} | {row['adf_pvalue']:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Composite Grid Scores (Per Horizon)",
        "",
    ])

    lookbacks = candidate_scores_df["lookback"].unique() if not candidate_scores_df.empty else []
    for lb in lookbacks:
        lines.append(f"### {lb} Horizon Rankings")
        lines.append("")
        lines.append("| Rank | Asset | Daily σ | Hurst | Half-Life | Q_MR | Grid Score |")
        lines.append("|---|---|---|---|---|---|---|")
        lb_df = candidate_scores_df[candidate_scores_df["lookback"] == lb].sort_values(
            by="score_two_pillar", ascending=False
        ).reset_index(drop=True)
        for rank, (_, r) in enumerate(lb_df.iterrows(), start=1):
            hl_str = "∞" if (np.isinf(r["half_life"]) or r["half_life"] > 9999) else f"{r['half_life']:.1f}d"
            lines.append(
                f"| {rank} | {r['ticker']} | {r['daily_sd'] * 100:.2f}% | {r['hurst']:.3f} | "
                f"{hl_str} | {r['q_mr']:.3f} | {r['score_two_pillar']:.1f} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Multi-Horizon Aggregate Rankings",
        "",
        "| Rank | Asset | Aggregate Grid Score | Additive Score | Aggregate Q_MR | Mean Daily σ | Mean Hurst | Median Half-Life |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for _, r in aggregate_rankings_df.iterrows():
        hl_str = "∞" if (np.isinf(r["median_half_life"]) or r["median_half_life"] > 9999) else f"{r['median_half_life']:.1f}d"
        lines.append(
            f"| {int(r['rank'])} | {r['ticker']} | **{r['agg_two_pillar']:.1f}** | {r['agg_additive']:.1f} | "
            f"{r['agg_q_mr']:.3f} | {r['mean_daily_sd'] * 100:.2f}% | {r['mean_hurst']:.3f} | {hl_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Deployment Lane Allocation & Regime Scorecard",
        "",
        "| Asset | Target Lane | Role | Regime Stability | Allocation Verdict | Primary Risk Factor |",
        "|---|---|---|---|---|---|",
    ])

    for ticker, eval_obj in regime_allocations.items():
        stability_icon = "Consistent" if eval_obj.is_regime_consistent else "Regime Flip"
        lines.append(
            f"| **{ticker}** | {eval_obj.lane.value} | {eval_obj.role.value} | {stability_icon} | "
            f"{eval_obj.verdict} | {eval_obj.risk_flag} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Operational Recommendations & Guardrails",
        "",
        "1. **Primary Deployment (Lane A):** Focus automated grid execution on assets with verified multi-horizon mean-reversion (`CL=F`, `BZ=F`).",
        "2. **Learning / Short-Cycle Deployment:** Assets with short-horizon mean-reversion but long-horizon half-life expansion (`XRP-USD`) must use strict dynamic cycle timeouts and tightened inventory caps.",
        "3. **Directional Zone Deployment (Lane B):** High-volatility trending assets (`GC=F`, `SI=F`, `BTC-USD`) should be traded with directional pullback entries rather than continuous bi-directional grids.",
        "4. **Observed Quarantine:** Assets with regime flips across horizons (`ETH-USD`) must remain observed-only until 1Y and 3Y memory metrics re-align.",
        "",
    ])

    return "\n".join(lines)


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Execute the end-to-end quantitative asset analysis and reporting pipeline."""
    cfg = config or PipelineConfig()
    cfg.scoring_config.validate()

    data_dir = Path(cfg.data_dir)
    reports_dir = Path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Optional fresh data download
    if cfg.fetch_fresh:
        logger.info("Fetching data for %d tickers...", len(cfg.tickers))
        fetch_and_save_tickers(
            tickers=cfg.tickers,
            data_dir=data_dir,
            start=cfg.start_date,
            end=cfg.end_date,
            incremental=not cfg.force_full_download,
            force_full=cfg.force_full_download,
        )

    # Step 2: Compute multi-horizon metric matrix
    logger.info("Computing metrics across tickers and lookbacks...")
    metric_matrix_df = compute_pipeline_metrics(
        tickers=cfg.tickers,
        lookbacks=cfg.lookbacks,
        data_dir=data_dir,
    )

    if metric_matrix_df.empty:
        raise ValueError("Metric computation yielded no results. Please check data files and tickers.")

    # Step 3: Evaluate scoring models
    logger.info("Evaluating candidate scoring models...")
    candidate_scores_df = evaluate_candidate_models(
        metric_matrix_df,
        config=cfg.scoring_config,
    )

    # Step 4: Compute multi-horizon aggregate rankings
    logger.info("Computing multi-horizon aggregate rankings...")
    aggregate_rankings_df = compute_aggregate_rankings(
        candidate_scores_df,
        config=cfg.scoring_config,
    )

    # Step 5: Evaluate regime stability & deployment lanes
    logger.info("Evaluating regime stability and allocating deployment lanes...")
    regime_allocations = allocate_deployment_lanes(
        metric_matrix_df,
        config=cfg.scoring_config,
    )

    # Step 6: Generate Markdown summary report
    markdown_report = ""
    if cfg.generate_markdown:
        logger.info("Generating Markdown summary report...")
        markdown_report = generate_markdown_report(
            metric_matrix_df=metric_matrix_df,
            candidate_scores_df=candidate_scores_df,
            aggregate_rankings_df=aggregate_rankings_df,
            regime_allocations=regime_allocations,
            scoring_config=cfg.scoring_config,
        )

    # Step 7: Export files to disk
    saved_files: dict[str, Path] = {}

    metric_file = reports_dir / cfg.metric_matrix_filename
    metric_matrix_df.to_csv(metric_file, index=False, float_format="%.6f")
    saved_files["metric_matrix"] = metric_file

    scores_file = reports_dir / cfg.candidate_scores_filename
    candidate_scores_df.to_csv(scores_file, index=False, float_format="%.6f")
    saved_files["candidate_scores"] = scores_file

    rankings_file = reports_dir / cfg.aggregate_rankings_filename
    aggregate_rankings_df.to_csv(rankings_file, index=False, float_format="%.6f")
    saved_files["aggregate_rankings"] = rankings_file

    if cfg.generate_markdown and markdown_report:
        report_file = reports_dir / cfg.report_filename
        report_file.write_text(markdown_report, encoding="utf-8")
        saved_files["markdown_report"] = report_file

    logger.info("Pipeline execution finished successfully. Output files: %s", list(saved_files.values()))

    return PipelineResult(
        metric_matrix_df=metric_matrix_df,
        candidate_scores_df=candidate_scores_df,
        aggregate_rankings_df=aggregate_rankings_df,
        regime_allocations=regime_allocations,
        markdown_report=markdown_report,
        saved_files=saved_files,
    )
