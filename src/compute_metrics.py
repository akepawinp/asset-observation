#!/usr/bin/env python
"""Compute the volatility & mean-reversion metric matrix and save to CSV + stdout.

Usage:
    python -m src.compute_metrics [--data-dir data] [--output reports/metric_matrix.csv]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute asset metric matrix.")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV price files.")
    parser.add_argument("--output", "-o", default="reports/metric_matrix.csv", help="Output CSV path.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from src.metrics import compute_all

    df = compute_all(data_dir=args.data_dir)

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")

    # Pretty-print to stdout
    print("\n" + "=" * 100)
    print("VOLATILITY & MEAN-REVERSION METRIC MATRIX")
    print("=" * 100)

    # Format for display
    display = df.copy()
    display["daily_sd"] = display["daily_sd"].map("{:.6f}".format)
    display["annual_sd"] = display["annual_sd"].map("{:.4f}".format)
    display["hurst"] = display["hurst"].map("{:.4f}".format)
    display["half_life"] = display["half_life"].apply(
        lambda x: "∞" if x == float("inf") else f"{x:.1f}"
    )
    display["adf_stat"] = display["adf_stat"].map("{:.4f}".format)
    display["adf_pvalue"] = display["adf_pvalue"].map("{:.4f}".format)

    print(display.to_string(index=False))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
