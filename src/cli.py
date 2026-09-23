"""CLI interface for asset price downloader."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from src.downloader import fetch_and_save_tickers
except ImportError:
    from .downloader import fetch_and_save_tickers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download historical asset price data from Yahoo Finance and save to CSV in /data."
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="One or more asset tickers (e.g. AAPL MSFT SPY BTC-USD).",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data",
        type=Path,
        help="Target folder to save CSV files (default: data).",
    )
    parser.add_argument(
        "--period",
        "-p",
        default="max",
        help="Historical period to download (default: max). Valid: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.",
    )
    parser.add_argument(
        "--interval",
        "-i",
        default="1d",
        help="Data interval (default: 1d).",
    )
    parser.add_argument(
        "--start",
        "-s",
        default=None,
        help="Start date in YYYY-MM-DD format (overrides period).",
    )
    parser.add_argument(
        "--end",
        "-e",
        default=None,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Force downloading full data and overwrite existing CSV files without incremental merging.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print(f"Fetching data for {len(args.tickers)} ticker(s): {', '.join(args.tickers)}")
    try:
        results = fetch_and_save_tickers(
            tickers=args.tickers,
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
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
