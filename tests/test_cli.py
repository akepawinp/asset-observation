"""Unit and integration tests for src.cli interface and configurable execution runner."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.cli import (
    _build_scoring_config,
    build_parser,
    format_aggregate_rankings_table,
    format_lane_allocations_table,
    format_metric_matrix_table,
    main,
)
from src.metrics import CANDIDATE_TICKERS, LOOKBACKS, format_half_life
from src.pipeline import PipelineConfig, run_pipeline
from src.scoring import DeploymentLane, LaneRole, RegimeEvaluation, ScoringConfig


class TestCLIParser(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_build_parser_has_subcommands(self):
        subparsers_actions = [
            action for action in self.parser._actions if action.dest == "command"
        ]
        self.assertEqual(len(subparsers_actions), 1)
        subcommands = subparsers_actions[0].choices
        self.assertIn("run", subcommands)
        self.assertIn("fetch", subcommands)
        self.assertIn("metrics", subcommands)
        self.assertIn("score", subcommands)

    def test_parse_run_defaults(self):
        args = self.parser.parse_args(["run"])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.tickers, list(CANDIDATE_TICKERS))
        self.assertEqual(args.lookbacks, list(LOOKBACKS))
        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.reports_dir, Path("reports"))
        self.assertEqual(args.w_hurst, 0.50)
        self.assertEqual(args.w_half_life, 0.35)
        self.assertEqual(args.w_adf, 0.15)
        self.assertEqual(args.ref_sd, 0.03)
        self.assertEqual(args.max_vol_score, 150.0)
        self.assertEqual(args.tau_hl, 90.0)
        self.assertEqual(args.w_1y, 0.30)
        self.assertEqual(args.w_3y, 0.50)
        self.assertEqual(args.w_5y, 0.20)
        self.assertFalse(args.fetch_fresh)
        self.assertFalse(args.export_json)
        self.assertTrue(args.generate_markdown)

    def test_parse_run_custom_arguments(self):
        args = self.parser.parse_args([
            "run",
            "--tickers", "CL=F", "BZ=F",
            "--lookbacks", "1Y", "3Y",
            "--data-dir", "custom_data",
            "--reports-dir", "custom_reports",
            "--w-hurst", "0.60",
            "--w-half-life", "0.30",
            "--w-adf", "0.10",
            "--ref-sd", "0.025",
            "--max-vol-score", "200.0",
            "--tau-hl", "60.0",
            "--w-1y", "0.40",
            "--w-3y", "0.60",
            "--w-5y", "0.00",
            "--fetch-fresh",
            "--json",
            "--no-markdown",
        ])
        self.assertEqual(args.tickers, ["CL=F", "BZ=F"])
        self.assertEqual(args.lookbacks, ["1Y", "3Y"])
        self.assertEqual(args.data_dir, Path("custom_data"))
        self.assertEqual(args.reports_dir, Path("custom_reports"))
        self.assertEqual(args.w_hurst, 0.60)
        self.assertEqual(args.w_half_life, 0.30)
        self.assertEqual(args.w_adf, 0.10)
        self.assertEqual(args.ref_sd, 0.025)
        self.assertEqual(args.max_vol_score, 200.0)
        self.assertEqual(args.tau_hl, 60.0)
        self.assertTrue(args.fetch_fresh)
        self.assertTrue(args.export_json)
        self.assertFalse(args.generate_markdown)

    def test_build_scoring_config_adjusts_lookbacks(self):
        args = self.parser.parse_args(["run", "--lookbacks", "1Y", "3Y"])
        config = _build_scoring_config(args, lookbacks=["1Y", "3Y"])
        self.assertIn("1Y", config.horizon_weights)
        self.assertIn("3Y", config.horizon_weights)
        self.assertNotIn("5Y", config.horizon_weights)
        # Verify weights re-normalized to 1.0
        self.assertAlmostEqual(sum(config.horizon_weights.values()), 1.0)
        self.assertAlmostEqual(config.horizon_weights["1Y"], 0.30 / 0.80)
        self.assertAlmostEqual(config.horizon_weights["3Y"], 0.50 / 0.80)

    def test_parse_fetch_arguments(self):
        args = self.parser.parse_args([
            "fetch", "AAPL", "MSFT",
            "--data-dir", "my_data",
            "--period", "1y",
            "--interval", "1d",
            "--force-full",
        ])
        self.assertEqual(args.command, "fetch")
        self.assertEqual(args.tickers, ["AAPL", "MSFT"])
        self.assertEqual(args.data_dir, Path("my_data"))
        self.assertEqual(args.period, "1y")
        self.assertEqual(args.interval, "1d")
        self.assertTrue(args.force_full)

    def test_parse_metrics_arguments(self):
        args = self.parser.parse_args([
            "metrics",
            "--tickers", "BTC-USD",
            "--lookbacks", "1Y",
            "--data-dir", "custom_data",
            "--output", "custom_reports/metrics.csv",
            "--json",
        ])
        self.assertEqual(args.command, "metrics")
        self.assertEqual(args.tickers, ["BTC-USD"])
        self.assertEqual(args.lookbacks, ["1Y"])
        self.assertEqual(args.data_dir, Path("custom_data"))
        self.assertEqual(args.output, Path("custom_reports/metrics.csv"))
        self.assertTrue(args.export_json)

    def test_parse_score_arguments(self):
        args = self.parser.parse_args([
            "score",
            "--input", "my_metrics.csv",
            "--output", "my_scores.csv",
            "--agg-output", "my_agg.csv",
            "--w-hurst", "0.45",
            "--w-half-life", "0.40",
            "--w-adf", "0.15",
            "--json",
        ])
        self.assertEqual(args.command, "score")
        self.assertEqual(args.input, Path("my_metrics.csv"))
        self.assertEqual(args.output, Path("my_scores.csv"))
        self.assertEqual(args.agg_output, Path("my_agg.csv"))
        self.assertEqual(args.w_hurst, 0.45)
        self.assertTrue(args.export_json)


class TestCLIFlowAndTables(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.data_dir = self.base_path / "data"
        self.reports_dir = self.base_path / "reports"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy synthetic data
        dates = pd.date_range("2021-01-01", "2026-01-01", freq="D")
        np.random.seed(42)

        # Mean reverting series for TEST-A
        p_a = 100.0 + np.sin(np.linspace(0, 50, len(dates))) * 10.0 + np.random.normal(0, 1, len(dates))
        df_a = pd.DataFrame({"Open": p_a, "High": p_a + 1, "Low": p_a - 1, "Close": p_a, "Adj Close": p_a, "Volume": 1000}, index=dates)
        df_a.index.name = "Date"
        df_a.to_csv(self.data_dir / "TEST-A.csv")

        # Trending series for TEST-B
        p_b = 50.0 + np.linspace(10, 200, len(dates)) + np.random.normal(0, 0.5, len(dates))
        df_b = pd.DataFrame({"Open": p_b, "High": p_b + 1, "Low": p_b - 1, "Close": p_b, "Adj Close": p_b, "Volume": 2000}, index=dates)
        df_b.index.name = "Date"
        df_b.to_csv(self.data_dir / "TEST-B.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_format_half_life_helper(self):
        self.assertEqual(format_half_life(float("nan")), "∞")
        self.assertEqual(format_half_life(float("inf")), "∞")
        self.assertEqual(format_half_life(10000.0), "∞")
        self.assertEqual(format_half_life(-5.0), "∞")
        self.assertEqual(format_half_life(45.67), "45.7d")

    def test_main_no_args_shows_help(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            exit_code = main([])
            self.assertEqual(exit_code, 0)
            self.assertIn("usage:", mock_out.getvalue())

    def test_main_run_end_to_end_with_json(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            exit_code = main([
                "run",
                "--tickers", "TEST-A", "TEST-B",
                "--lookbacks", "1Y", "3Y",
                "--data-dir", str(self.data_dir),
                "--reports-dir", str(self.reports_dir),
                "--json",
            ])
            self.assertEqual(exit_code, 0)
            output = mock_out.getvalue()
            # Verify stdout tables
            self.assertIn("VOLATILITY & MEAN-REVERSION METRIC MATRIX", output)
            self.assertIn("MULTI-HORIZON AGGREGATE RANKINGS", output)
            self.assertIn("DEPLOYMENT LANE ALLOCATION & REGIME SCORECARD", output)
            self.assertIn("TEST-A", output)
            self.assertIn("TEST-B", output)
            self.assertIn("Output artifacts saved to:", output)

            # Verify CSV & JSON files created on disk
            self.assertTrue((self.reports_dir / "metric_matrix.csv").exists())
            self.assertTrue((self.reports_dir / "candidate_scores.csv").exists())
            self.assertTrue((self.reports_dir / "aggregate_rankings.csv").exists())
            self.assertTrue((self.reports_dir / "final_asset_selection_report.md").exists())
            self.assertTrue((self.reports_dir / "metric_matrix.json").exists())
            self.assertTrue((self.reports_dir / "candidate_scores.json").exists())
            self.assertTrue((self.reports_dir / "aggregate_rankings.json").exists())
            self.assertTrue((self.reports_dir / "regime_allocations.json").exists())

    def test_main_run_invalid_weights_error(self):
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            exit_code = main([
                "run",
                "--tickers", "TEST-A",
                "--data-dir", str(self.data_dir),
                "--reports-dir", str(self.reports_dir),
                "--w-hurst", "0.80",
                "--w-half-life", "0.50",
                "--w-adf", "0.10",  # Sum = 1.4 != 1.0
            ])
            self.assertEqual(exit_code, 1)
            self.assertIn("Error", mock_err.getvalue())

    @patch("src.cli.fetch_and_save_tickers")
    def test_main_fetch_invokes_downloader(self, mock_fetch):
        mock_fetch.return_value = {"AAPL": Path("data/AAPL.csv")}
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            exit_code = main([
                "fetch", "AAPL",
                "--data-dir", str(self.data_dir),
                "--period", "6mo",
            ])
            self.assertEqual(exit_code, 0)
            mock_fetch.assert_called_once()
            output = mock_out.getvalue()
            self.assertIn("Successfully saved", output)

    def test_main_metrics_command_with_json(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            metrics_out = self.reports_dir / "custom_metrics.csv"
            exit_code = main([
                "metrics",
                "--tickers", "TEST-A", "TEST-B",
                "--lookbacks", "1Y",
                "--data-dir", str(self.data_dir),
                "--output", str(metrics_out),
                "--json",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(metrics_out.exists())
            self.assertTrue(metrics_out.with_suffix(".json").exists())
            output = mock_out.getvalue()
            self.assertIn("METRIC MATRIX", output)

    def test_main_score_command_with_json(self):
        # Generate metric matrix first
        cfg = PipelineConfig(
            tickers=["TEST-A", "TEST-B"],
            lookbacks=["1Y", "3Y"],
            data_dir=self.data_dir,
            reports_dir=self.reports_dir,
            scoring_config=ScoringConfig(horizon_weights={"1Y": 0.5, "3Y": 0.5}),
        )
        run_pipeline(cfg)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            exit_code = main([
                "score",
                "--input", str(self.reports_dir / "metric_matrix.csv"),
                "--output", str(self.reports_dir / "scored.csv"),
                "--agg-output", str(self.reports_dir / "agg.csv"),
                "--json",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((self.reports_dir / "scored.csv").exists())
            self.assertTrue((self.reports_dir / "agg.csv").exists())
            self.assertTrue((self.reports_dir / "scored.json").exists())
            self.assertTrue((self.reports_dir / "agg.json").exists())
            output = mock_out.getvalue()
            self.assertIn("MULTI-HORIZON AGGREGATE RANKINGS", output)


if __name__ == "__main__":
    unittest.main()
