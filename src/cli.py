"""CLI interface and configurable execution runner for quantitative asset observation.

Provides a unified command-line tool supporting:
- Complete end-to-end quantitative asset selection and strategy allocation workflow (`run`)
- Data download and incremental ingestion (`fetch`)
- Volatility and statistical memory metric calculation (`metrics`)
- Multi-horizon Composite Grid Score evaluation and ranking (`score`)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.downloader import fetch_and_save_tickers
from src.metrics import (
    CANDIDATE_TICKERS,
    LOOKBACKS,
    compute_all,
    format_half_life,
)
from src.pipeline import (
    PipelineConfig,
    PipelineResult,
    compute_pipeline_metrics,
    run_pipeline,
)
from src.scoring import (
    DEFAULT_CONFIG,
    DeploymentLane,
    LaneRole,
    RegimeEvaluation,
    ScoringConfig,
    allocate_deployment_lanes,
    compute_aggregate_rankings,
    evaluate_candidate_models,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatted Stdout Summary Tables
# ---------------------------------------------------------------------------

def format_metric_matrix_table(df: pd.DataFrame) -> str:
    """Format the volatility and mean-reversion metric matrix into an ASCII table."""
    if df.empty:
        return "No metric data available."

    lines: list[str] = [
        "=" * 105,
        "VOLATILITY & MEAN-REVERSION METRIC MATRIX",
        "=" * 105,
    ]
    disp = df.copy()
    disp["daily_sd"] = (disp["daily_sd"] * 100).map("{:.2f}%".format)
    disp["annual_sd"] = (disp["annual_sd"] * 100).map("{:.1f}%".format)
    disp["hurst"] = disp["hurst"].map("{:.3f}".format)
    disp["half_life"] = disp["half_life"].apply(format_half_life)
    disp["adf_pvalue"] = disp["adf_pvalue"].map("{:.3f}".format)

    cols = ["ticker", "lookback", "daily_sd", "annual_sd", "hurst", "half_life", "adf_pvalue"]
    if "adf_stat" in disp.columns:
        disp["adf_stat"] = disp["adf_stat"].map("{:.3f}".format)
        cols.append("adf_stat")

    lines.append(disp[cols].to_string(index=False))
    return "\n".join(lines)


def format_candidate_scores_table(df: pd.DataFrame, lookback: str | None = None) -> str:
    """Format candidate scoring models comparison table."""
    if df.empty:
        return "No scoring data available."

    sub_df = df if lookback is None else df[df["lookback"] == lookback]
    if sub_df.empty:
        return f"No scoring data for lookback {lookback}."

    lines: list[str] = [
        "=" * 115,
        f"CANDIDATE GRID SCORES (HORIZON: {lookback or 'ALL'})",
        "=" * 115,
    ]
    disp = sub_df.copy().sort_values(by="score_two_pillar", ascending=False)
    disp["daily_sd"] = (disp["daily_sd"] * 100).map("{:.2f}%".format)
    disp["hurst"] = disp["hurst"].map("{:.3f}".format)
    disp["half_life"] = disp["half_life"].apply(format_half_life)
    disp["q_mr"] = disp["q_mr"].map("{:.3f}".format)
    disp["score_two_pillar"] = disp["score_two_pillar"].map("{:.1f}".format)
    disp["score_additive"] = disp["score_additive"].map("{:.1f}".format)
    disp["score_power_ratio"] = disp["score_power_ratio"].map("{:.2f}".format)
    disp["score_exp_ratio"] = disp["score_exp_ratio"].map("{:.2f}".format)

    cols = [
        "ticker",
        "lookback",
        "daily_sd",
        "hurst",
        "half_life",
        "q_mr",
        "score_two_pillar",
        "score_additive",
        "score_power_ratio",
        "score_exp_ratio",
    ]
    lines.append(disp[cols].to_string(index=False))
    return "\n".join(lines)


def format_aggregate_rankings_table(df: pd.DataFrame) -> str:
    """Format multi-horizon aggregate rankings table."""
    if df.empty:
        return "No aggregate ranking data available."

    lines: list[str] = [
        "=" * 105,
        "MULTI-HORIZON AGGREGATE RANKINGS",
        "=" * 105,
    ]
    disp = df.copy()
    disp["agg_two_pillar"] = disp["agg_two_pillar"].map("{:.1f}".format)
    disp["agg_additive"] = disp["agg_additive"].map("{:.1f}".format)
    disp["agg_q_mr"] = disp["agg_q_mr"].map("{:.3f}".format)
    disp["mean_daily_sd"] = (disp["mean_daily_sd"] * 100).map("{:.2f}%".format)
    disp["mean_hurst"] = disp["mean_hurst"].map("{:.3f}".format)
    disp["median_half_life"] = disp["median_half_life"].apply(format_half_life)

    cols = [
        "rank",
        "ticker",
        "agg_two_pillar",
        "agg_additive",
        "agg_q_mr",
        "mean_daily_sd",
        "mean_hurst",
        "median_half_life",
    ]
    lines.append(disp[cols].to_string(index=False))
    return "\n".join(lines)


def format_lane_allocations_table(allocations: dict[str, RegimeEvaluation]) -> str:
    """Format deployment lane allocation scorecard."""
    if not allocations:
        return "No allocation data available."

    lines: list[str] = [
        "=" * 135,
        "DEPLOYMENT LANE ALLOCATION & REGIME SCORECARD",
        "=" * 135,
    ]

    rows = []
    for ticker, eval_obj in allocations.items():
        stability = "Consistent" if eval_obj.is_regime_consistent else "Regime Flip"
        rows.append({
            "ticker": ticker,
            "lane": eval_obj.lane.value,
            "role": eval_obj.role.value,
            "stability": stability,
            "verdict": eval_obj.verdict,
            "risk_flag": eval_obj.risk_flag,
        })

    alloc_df = pd.DataFrame(rows)
    lines.append(alloc_df.to_string(index=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ScoringConfig Builder Helper
# ---------------------------------------------------------------------------

def _build_scoring_config(
    args: argparse.Namespace,
    lookbacks: Sequence[str] | None = None,
) -> ScoringConfig:
    """Build and validate ScoringConfig from CLI args, aligning horizon weights with active lookbacks."""
    active_lbs = list(lookbacks) if lookbacks is not None else list(LOOKBACKS)
    
    # Check if user explicitly provided horizon weights
    w_1y = getattr(args, "w_1y", None)
    w_3y = getattr(args, "w_3y", None)
    w_5y = getattr(args, "w_5y", None)

    horizon_weights: dict[str, float] = {}

    # If user provided specific weights, respect them for active lookbacks
    raw_weights = {"1Y": w_1y, "3Y": w_3y, "5Y": w_5y}
    filtered_weights = {lb: w for lb, w in raw_weights.items() if lb in active_lbs and w is not None and w > 0}

    if filtered_weights:
        total_w = sum(filtered_weights.values())
        if total_w > 0:
            # Re-normalize if sum is not exactly 1.0 due to subsetting lookbacks
            horizon_weights = {lb: w / total_w for lb, w in filtered_weights.items()}
    else:
        # Default equal or standard weights across active lookbacks
        std_weights = {"1Y": 0.30, "3Y": 0.50, "5Y": 0.20}
        active_std = {lb: std_weights.get(lb, 1.0 / len(active_lbs)) for lb in active_lbs}
        total_std = sum(active_std.values())
        horizon_weights = {lb: w / total_std for lb, w in active_std.items()}

    config = ScoringConfig(
        w_hurst=args.w_hurst,
        w_half_life=args.w_half_life,
        w_adf=args.w_adf,
        ref_sd=args.ref_sd,
        max_vol_score=args.max_vol_score,
        h_opt=args.h_opt,
        h_max=args.h_max,
        tau_half_life=args.tau_hl,
        horizon_weights=horizon_weights,
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Argument Parser Construction
# ---------------------------------------------------------------------------

def _add_scoring_weight_arguments(parser: argparse.ArgumentParser) -> None:
    """Add reusable scoring weight and parameter options to a subparser."""
    group = parser.add_argument_group("Scoring Weights & Parameters")
    group.add_argument(
        "--w-hurst",
        type=float,
        default=DEFAULT_CONFIG.w_hurst,
        help=f"Weight for Hurst anti-persistence in Q_MR (default: {DEFAULT_CONFIG.w_hurst:.2f}).",
    )
    group.add_argument(
        "--w-half-life",
        type=float,
        default=DEFAULT_CONFIG.w_half_life,
        help=f"Weight for Half-Life decay speed in Q_MR (default: {DEFAULT_CONFIG.w_half_life:.2f}).",
    )
    group.add_argument(
        "--w-adf",
        type=float,
        default=DEFAULT_CONFIG.w_adf,
        help=f"Weight for ADF stationarity in Q_MR (default: {DEFAULT_CONFIG.w_adf:.2f}).",
    )
    group.add_argument(
        "--ref-sd",
        type=float,
        default=DEFAULT_CONFIG.ref_sd,
        help=f"Reference daily SD benchmark (default: {DEFAULT_CONFIG.ref_sd:.3f} = 3.0%%).",
    )
    group.add_argument(
        "--max-vol-score",
        type=float,
        default=DEFAULT_CONFIG.max_vol_score,
        help=f"Cap on volatility potential score (default: {DEFAULT_CONFIG.max_vol_score:.1f}).",
    )
    group.add_argument(
        "--h-opt",
        type=float,
        default=DEFAULT_CONFIG.h_opt,
        help=f"Optimal Hurst threshold for 1.0 MR quality (default: {DEFAULT_CONFIG.h_opt:.2f}).",
    )
    group.add_argument(
        "--h-max",
        type=float,
        default=DEFAULT_CONFIG.h_max,
        help=f"Max Hurst threshold above which MR quality is 0.0 (default: {DEFAULT_CONFIG.h_max:.2f}).",
    )
    group.add_argument(
        "--tau-hl",
        "--tau-half-life",
        dest="tau_hl",
        type=float,
        default=DEFAULT_CONFIG.tau_half_life,
        help=f"Half-Life decay constant in days (default: {DEFAULT_CONFIG.tau_half_life:.1f}).",
    )
    group.add_argument(
        "--w-1y",
        type=float,
        default=DEFAULT_CONFIG.horizon_weights.get("1Y", 0.30),
        help="Multi-horizon weight for 1Y horizon (default: 0.30).",
    )
    group.add_argument(
        "--w-3y",
        type=float,
        default=DEFAULT_CONFIG.horizon_weights.get("3Y", 0.50),
        help="Multi-horizon weight for 3Y horizon (default: 0.50).",
    )
    group.add_argument(
        "--w-5y",
        type=float,
        default=DEFAULT_CONFIG.horizon_weights.get("5Y", 0.20),
        help="Multi-horizon weight for 5Y horizon (default: 0.20).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the unified command-line parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="asset-observation",
        description="Quantitative Asset Observation & Mean-Reversion Grid Screening CLI.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logs.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Workflow subcommand to execute.")

    # 1. Pipeline execution runner ('run')
    run_parser = subparsers.add_parser(
        "run",
        help="Execute the end-to-end asset selection and strategy allocation pipeline.",
    )
    run_parser.add_argument(
        "--tickers",
        "-t",
        nargs="+",
        default=list(CANDIDATE_TICKERS),
        help=f"Tickers to analyze (default: {' '.join(CANDIDATE_TICKERS)}).",
    )
    run_parser.add_argument(
        "--lookbacks",
        "-l",
        nargs="+",
        default=list(LOOKBACKS),
        help=f"Lookback horizons (default: {' '.join(LOOKBACKS)}).",
    )
    run_parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("data"),
        help="Directory containing daily price CSVs (default: data).",
    )
    run_parser.add_argument(
        "--reports-dir",
        "-r",
        "--output-dir",
        dest="reports_dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for CSVs, JSON, and Markdown reports (default: reports).",
    )
    run_parser.add_argument(
        "--fetch",
        "--fetch-fresh",
        dest="fetch_fresh",
        action="store_true",
        help="Download/update price data from Yahoo Finance before running pipeline.",
    )
    run_parser.add_argument(
        "--start",
        "-s",
        default=None,
        help="Start date YYYY-MM-DD for fresh data fetch.",
    )
    run_parser.add_argument(
        "--end",
        "-e",
        default=None,
        help="End date YYYY-MM-DD for fresh data fetch.",
    )
    run_parser.add_argument(
        "--force-full",
        action="store_true",
        help="Force full re-download of price data without incremental merge.",
    )
    run_parser.add_argument(
        "--json",
        "--export-json",
        dest="export_json",
        action="store_true",
        help="Export machine-readable JSON files alongside CSVs.",
    )
    run_parser.add_argument(
        "--no-markdown",
        action="store_false",
        dest="generate_markdown",
        help="Skip generating human-readable Markdown summary report.",
    )
    run_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress terminal summary tables.",
    )
    _add_scoring_weight_arguments(run_parser)

    # 2. Data downloader ('fetch')
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download or update daily OHLCV price series from Yahoo Finance.",
    )
    fetch_parser.add_argument(
        "tickers",
        nargs="*",
        default=list(CANDIDATE_TICKERS),
        help=f"Tickers to download (default: {' '.join(CANDIDATE_TICKERS)}).",
    )
    fetch_parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("data"),
        help="Target directory to save CSV files (default: data).",
    )
    fetch_parser.add_argument(
        "--period",
        "-p",
        default="max",
        help="Historical period to download (default: max).",
    )
    fetch_parser.add_argument(
        "--interval",
        "-i",
        default="1d",
        help="Data interval (default: 1d).",
    )
    fetch_parser.add_argument(
        "--start",
        "-s",
        default=None,
        help="Start date in YYYY-MM-DD format (overrides period).",
    )
    fetch_parser.add_argument(
        "--end",
        "-e",
        default=None,
        help="End date in YYYY-MM-DD format.",
    )
    fetch_parser.add_argument(
        "--force-full",
        action="store_true",
        help="Force full download and overwrite existing CSV files without incremental merging.",
    )

    # 3. Metric computation ('metrics')
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Compute volatility and mean-reversion metric matrix.",
    )
    metrics_parser.add_argument(
        "--tickers",
        "-t",
        nargs="+",
        default=list(CANDIDATE_TICKERS),
        help=f"Tickers to compute (default: {' '.join(CANDIDATE_TICKERS)}).",
    )
    metrics_parser.add_argument(
        "--lookbacks",
        "-l",
        nargs="+",
        default=list(LOOKBACKS),
        help=f"Lookback horizons (default: {' '.join(LOOKBACKS)}).",
    )
    metrics_parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("data"),
        help="Directory with CSV price files (default: data).",
    )
    metrics_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("reports/metric_matrix.csv"),
        help="Output CSV path for metric matrix (default: reports/metric_matrix.csv).",
    )
    metrics_parser.add_argument(
        "--json",
        action="store_true",
        dest="export_json",
        help="Export JSON format alongside CSV.",
    )

    # 4. Scoring and ranking ('score')
    score_parser = subparsers.add_parser(
        "score",
        help="Evaluate Composite Grid Scores and rankings from a metric matrix CSV.",
    )
    score_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("reports/metric_matrix.csv"),
        help="Input metric matrix CSV path (default: reports/metric_matrix.csv).",
    )
    score_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("reports/candidate_scores.csv"),
        help="Output CSV path for candidate scores (default: reports/candidate_scores.csv).",
    )
    score_parser.add_argument(
        "--agg-output",
        "-a",
        type=Path,
        default=Path("reports/aggregate_rankings.csv"),
        help="Output CSV path for aggregate rankings (default: reports/aggregate_rankings.csv).",
    )
    score_parser.add_argument(
        "--json",
        action="store_true",
        dest="export_json",
        help="Export JSON format alongside CSV.",
    )
    _add_scoring_weight_arguments(score_parser)

    return parser


# ---------------------------------------------------------------------------
# Subcommand Execution Handlers
# ---------------------------------------------------------------------------

def handle_run(args: argparse.Namespace) -> int:
    """Handle 'run' subcommand executing the full analysis pipeline."""
    scoring_config = _build_scoring_config(args, lookbacks=args.lookbacks)

    pipeline_config = PipelineConfig(
        tickers=args.tickers,
        lookbacks=args.lookbacks,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        scoring_config=scoring_config,
        fetch_fresh=args.fetch_fresh,
        start_date=args.start,
        end_date=args.end,
        force_full_download=args.force_full,
        generate_markdown=args.generate_markdown,
        export_json=args.export_json,
    )

    result: PipelineResult = run_pipeline(pipeline_config)

    if not args.quiet:
        print("\n" + format_metric_matrix_table(result.metric_matrix_df))
        print("\n" + format_aggregate_rankings_table(result.aggregate_rankings_df))
        print("\n" + format_lane_allocations_table(result.regime_allocations))
        print("\n" + "=" * 105)
        print("EXECUTION SUMMARY & SAVED ARTIFACTS")
        print("=" * 105)
        print(f"Status: Completed successfully ({len(result.aggregate_rankings_df)} assets analyzed)")
        print("Output artifacts saved to:")
        for name, path in result.saved_files.items():
            print(f"  - {name}: {path}")

    return 0


def handle_fetch(args: argparse.Namespace) -> int:
    """Handle 'fetch' subcommand downloading price data."""
    tickers = args.tickers if isinstance(args.tickers, list) and args.tickers else list(CANDIDATE_TICKERS)
    print(f"Fetching data for {len(tickers)} ticker(s): {', '.join(tickers)}")
    results = fetch_and_save_tickers(
        tickers=tickers,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        period=args.period,
        interval=args.interval,
        incremental=not args.force_full,
        force_full=args.force_full,
    )
    print("\nSuccessfully saved:")
    for ticker, path in results.items():
        print(f"  - {ticker}: {path}")
    return 0


def handle_metrics(args: argparse.Namespace) -> int:
    """Handle 'metrics' subcommand computing volatility and memory metrics."""
    print(f"Computing metrics for {len(args.tickers)} ticker(s) across lookbacks: {', '.join(args.lookbacks)}")
    df = compute_pipeline_metrics(
        tickers=args.tickers,
        lookbacks=args.lookbacks,
        data_dir=args.data_dir,
    )
    if df.empty:
        raise ValueError("Metric computation yielded no results. Check tickers and data directory.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")

    if getattr(args, "export_json", False):
        json_path = out_path.with_suffix(".json")
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved metric matrix JSON to: {json_path}")

    print("\n" + format_metric_matrix_table(df))
    print(f"\nSaved metric matrix to: {out_path}")
    return 0


def handle_score(args: argparse.Namespace) -> int:
    """Handle 'score' subcommand evaluating scoring models."""
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input metric matrix file '{in_path}' does not exist.")

    df = pd.read_csv(in_path)
    available_lookbacks = df["lookback"].unique() if "lookback" in df.columns else list(LOOKBACKS)
    config = _build_scoring_config(args, lookbacks=available_lookbacks)

    scored = evaluate_candidate_models(df, config=config)
    agg = compute_aggregate_rankings(scored, config=config)
    allocations = allocate_deployment_lanes(df, config=config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False, float_format="%.6f")

    agg_path = Path(args.agg_output)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(agg_path, index=False, float_format="%.6f")

    if getattr(args, "export_json", False):
        scored_json = out_path.with_suffix(".json")
        scored.to_json(scored_json, orient="records", indent=2)
        agg_json = agg_path.with_suffix(".json")
        agg.to_json(agg_json, orient="records", indent=2)
        print(f"Saved candidate scores JSON to: {scored_json}")
        print(f"Saved aggregate rankings JSON to: {agg_json}")

    print("\n" + format_aggregate_rankings_table(agg))
    print("\n" + format_lane_allocations_table(allocations))
    print(f"\nSaved candidate scores to: {out_path}")
    print(f"Saved aggregate rankings to: {agg_path}")
    return 0


# ---------------------------------------------------------------------------
# Command Dispatch Table
# ---------------------------------------------------------------------------

_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": handle_run,
    "fetch": handle_fetch,
    "metrics": handle_metrics,
    "score": handle_score,
}


# ---------------------------------------------------------------------------
# Main CLI Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        return 0

    handler = _COMMAND_HANDLERS.get(args.command)
    if not handler:
        parser.print_help()
        return 0

    try:
        return handler(args)
    except Exception as exc:
        if getattr(args, "verbose", False):
            logger.exception("CLI execution failed: %s", exc)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
