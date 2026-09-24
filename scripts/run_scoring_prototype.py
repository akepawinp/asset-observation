#!/usr/bin/env python
"""Prototype runner for Composite Grid Score formula evaluation.

Loads reports/metric_matrix.csv, evaluates candidate scoring formulas with
configurable weights and benchmarks, and outputs reports.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_scoring_prototype.py [options]

Examples:
    # Run with defaults
    PYTHONPATH=. .venv/bin/python scripts/run_scoring_prototype.py

    # Customize weights: 60% Hurst, 30% Half-Life, 10% ADF
    PYTHONPATH=. .venv/bin/python scripts/run_scoring_prototype.py --w-hurst 0.60 --w-half-life 0.30 --w-adf 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.scoring import (
    ScoringConfig,
    compute_aggregate_rankings,
    evaluate_candidate_models,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate grid score prototypes.")
    parser.add_argument("--input", "-i", default="reports/metric_matrix.csv", help="Path to metric matrix CSV.")
    parser.add_argument("--output", "-o", default="reports/candidate_scores.csv", help="Output candidate scores CSV.")
    parser.add_argument("--agg-output", "-a", default="reports/aggregate_rankings.csv", help="Output aggregate rankings CSV.")
    
    # Weight configuration arguments
    parser.add_argument("--w-hurst", type=float, default=0.50, help="Weight for Hurst exponent in Q_MR (default: 0.50)")
    parser.add_argument("--w-half-life", type=float, default=0.35, help="Weight for Half-Life in Q_MR (default: 0.35)")
    parser.add_argument("--w-adf", type=float, default=0.15, help="Weight for ADF p-value in Q_MR (default: 0.15)")
    parser.add_argument("--ref-sd", type=float, default=0.03, help="Reference daily SD benchmark (default: 0.03 = 3%)")
    parser.add_argument("--tau-hl", type=float, default=90.0, help="Half-Life decay constant in days (default: 90.0)")
    
    # Horizon weights
    parser.add_argument("--w-1y", type=float, default=0.30, help="Horizon weight for 1Y (default: 0.30)")
    parser.add_argument("--w-3y", type=float, default=0.50, help="Horizon weight for 3Y (default: 0.50)")
    parser.add_argument("--w-5y", type=float, default=0.20, help="Horizon weight for 5Y (default: 0.20)")
    
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: {in_path} does not exist. Run src.compute_metrics first.")
        return 1

    config = ScoringConfig(
        w_hurst=args.w_hurst,
        w_half_life=args.w_half_life,
        w_adf=args.w_adf,
        ref_sd=args.ref_sd,
        tau_half_life=args.tau_hl,
        horizon_weights={"1Y": args.w_1y, "3Y": args.w_3y, "5Y": args.w_5y},
    )
    config.validate()

    df = pd.read_csv(in_path)
    scored = evaluate_candidate_models(df, config=config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False, float_format="%.4f")

    agg = compute_aggregate_rankings(scored, config=config)
    agg.to_csv(Path(args.agg_output), index=False, float_format="%.4f")

    print("\n" + "=" * 90)
    print("COMPOSITE GRID SCORE PROTOTYPE: CANDIDATE MODEL COMPARISON")
    print(f"Weights: Hurst={config.w_hurst:.2f}, Half-Life={config.w_half_life:.2f}, ADF={config.w_adf:.2f} | ref_sd={config.ref_sd:.3f}")
    print("=" * 90)

    for lb in ["1Y", "3Y", "5Y"]:
        print(f"\n--- Lookback: {lb} ---")
        sub = scored[scored["lookback"] == lb].copy()
        sub = sub.sort_values(by="score_two_pillar", ascending=False)
        cols_display = [
            "ticker",
            "daily_sd",
            "hurst",
            "half_life",
            "adf_pvalue",
            "q_mr",
            "score_two_pillar",
            "score_additive",
            "score_power_ratio",
            "score_exp_ratio",
        ]
        disp = sub[cols_display].copy()
        disp["daily_sd"] = (disp["daily_sd"] * 100).map("{:.2f}%".format)
        disp["hurst"] = disp["hurst"].map("{:.3f}".format)
        disp["half_life"] = disp["half_life"].apply(
            lambda x: "inf" if x == float("inf") else f"{x:.1f}d"
        )
        disp["adf_pvalue"] = disp["adf_pvalue"].map("{:.3f}".format)
        disp["q_mr"] = disp["q_mr"].map("{:.3f}".format)
        disp["score_two_pillar"] = disp["score_two_pillar"].map("{:.1f}".format)
        disp["score_additive"] = disp["score_additive"].map("{:.1f}".format)
        disp["score_power_ratio"] = disp["score_power_ratio"].map("{:.2f}".format)
        disp["score_exp_ratio"] = disp["score_exp_ratio"].map("{:.2f}".format)
        print(disp.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"MULTI-HORIZON AGGREGATE RANKINGS ({int(args.w_3y*100)}% 3Y + {int(args.w_1y*100)}% 1Y + {int(args.w_5y*100)}% 5Y)")
    print("=" * 90)
    disp_agg = agg.copy()
    disp_agg["agg_two_pillar"] = disp_agg["agg_two_pillar"].map("{:.1f}".format)
    disp_agg["agg_additive"] = disp_agg["agg_additive"].map("{:.1f}".format)
    disp_agg["agg_q_mr"] = disp_agg["agg_q_mr"].map("{:.3f}".format)
    disp_agg["mean_daily_sd"] = (disp_agg["mean_daily_sd"] * 100).map("{:.2f}%".format)
    disp_agg["mean_hurst"] = disp_agg["mean_hurst"].map("{:.3f}".format)
    disp_agg["median_half_life"] = disp_agg["median_half_life"].map("{:.1f}d".format)
    print(disp_agg[["rank", "ticker", "agg_two_pillar", "agg_additive", "agg_q_mr", "mean_daily_sd", "mean_hurst", "median_half_life"]].to_string(index=False))

    print(f"\nSaved detailed scores to {out_path}")
    print(f"Saved aggregate rankings to {args.agg_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
