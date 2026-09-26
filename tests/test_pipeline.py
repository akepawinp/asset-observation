"""Unit and integration tests for src.pipeline orchestrator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import (
    PipelineConfig,
    PipelineResult,
    generate_markdown_report,
    run_pipeline,
)
from src.scoring import DeploymentLane, LaneRole, ScoringConfig


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.data_dir = self.base_path / "data"
        self.reports_dir = self.base_path / "reports"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy daily CSVs for 2 test tickers
        dates = pd.date_range("2021-01-01", "2026-01-01", freq="D")
        np.random.seed(42)

        # Ticker A: Mean-reverting synthetic series
        price_a = 100.0 + np.sin(np.linspace(0, 50, len(dates))) * 10.0 + np.random.normal(0, 1, len(dates))
        df_a = pd.DataFrame(
            {
                "Open": price_a,
                "High": price_a + 1.0,
                "Low": price_a - 1.0,
                "Close": price_a,
                "Adj Close": price_a,
                "Volume": 1000,
            },
            index=dates,
        )
        df_a.index.name = "Date"
        df_a.to_csv(self.data_dir / "TEST-A.csv")

        # Ticker B: Trending synthetic series
        price_b = 50.0 + np.linspace(10, 200, len(dates)) + np.random.normal(0, 0.5, len(dates))
        df_b = pd.DataFrame(
            {
                "Open": price_b,
                "High": price_b + 1.0,
                "Low": price_b - 1.0,
                "Close": price_b,
                "Adj Close": price_b,
                "Volume": 2000,
            },
            index=dates,
        )
        df_b.index.name = "Date"
        df_b.to_csv(self.data_dir / "TEST-B.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_pipeline_end_to_end_synthetic(self):
        config = PipelineConfig(
            tickers=["TEST-A", "TEST-B"],
            lookbacks=["1Y", "3Y"],
            data_dir=self.data_dir,
            reports_dir=self.reports_dir,
            scoring_config=ScoringConfig(
                horizon_weights={"1Y": 0.40, "3Y": 0.60}
            ),
            export_json=True,
        )

        result: PipelineResult = run_pipeline(config)

        # Check returned DataFrames
        self.assertIsInstance(result.metric_matrix_df, pd.DataFrame)
        self.assertIsInstance(result.candidate_scores_df, pd.DataFrame)
        self.assertIsInstance(result.aggregate_rankings_df, pd.DataFrame)
        self.assertEqual(len(result.metric_matrix_df), 4)  # 2 tickers * 2 lookbacks
        self.assertEqual(len(result.candidate_scores_df), 4)
        self.assertEqual(len(result.aggregate_rankings_df), 2)

        # Check regime allocations
        self.assertIn("TEST-A", result.regime_allocations)
        self.assertIn("TEST-B", result.regime_allocations)

        # Check generated files on disk (CSV + JSON)
        metric_file = self.reports_dir / "metric_matrix.csv"
        scores_file = self.reports_dir / "candidate_scores.csv"
        rankings_file = self.reports_dir / "aggregate_rankings.csv"
        report_file = self.reports_dir / "final_asset_selection_report.md"
        metric_json = self.reports_dir / "metric_matrix.json"
        scores_json = self.reports_dir / "candidate_scores.json"
        rankings_json = self.reports_dir / "aggregate_rankings.json"
        alloc_json = self.reports_dir / "regime_allocations.json"

        self.assertTrue(metric_file.exists())
        self.assertTrue(scores_file.exists())
        self.assertTrue(rankings_file.exists())
        self.assertTrue(report_file.exists())
        self.assertTrue(metric_json.exists())
        self.assertTrue(scores_json.exists())
        self.assertTrue(rankings_json.exists())
        self.assertTrue(alloc_json.exists())

        # Validate JSON contents
        with open(alloc_json, encoding="utf-8") as f:
            alloc_data = json.load(f)
            self.assertIn("TEST-A", alloc_data)
            self.assertIn("lane", alloc_data["TEST-A"])

        # Validate CSV contents
        df_metrics = pd.read_csv(metric_file)
        self.assertIn("daily_sd", df_metrics.columns)
        self.assertIn("hurst", df_metrics.columns)
        self.assertIn("half_life", df_metrics.columns)

        df_rankings = pd.read_csv(rankings_file)
        self.assertIn("agg_two_pillar", df_rankings.columns)
        self.assertIn("rank", df_rankings.columns)

        # Validate Markdown report content
        report_text = report_file.read_text()
        self.assertIn("# Asset Selection & Strategy Allocation Report", report_text)
        self.assertIn("## Executive Summary", report_text)
        self.assertIn("TEST-A", report_text)
        self.assertIn("TEST-B", report_text)

    def test_run_pipeline_with_real_candidate_data(self):
        # Run pipeline using the actual repository data files in data/
        config = PipelineConfig(
            data_dir="data",
            reports_dir=self.reports_dir,
        )
        result = run_pipeline(config)

        self.assertEqual(len(result.aggregate_rankings_df), 7)
        self.assertEqual(len(result.regime_allocations), 7)

        # Assert specific allocation verdicts match domain rules
        self.assertEqual(result.regime_allocations["ETH-USD"].lane, DeploymentLane.OBSERVED_ONLY)
        self.assertEqual(result.regime_allocations["CL=F"].lane, DeploymentLane.LANE_A)
        self.assertEqual(result.regime_allocations["BTC-USD"].lane, DeploymentLane.LANE_B)

        # Verify saved files dictionary
        self.assertIn("metric_matrix", result.saved_files)
        self.assertIn("candidate_scores", result.saved_files)
        self.assertIn("aggregate_rankings", result.saved_files)
        self.assertIn("markdown_report", result.saved_files)
        for path in result.saved_files.values():
            self.assertTrue(path.exists())

    def test_invalid_scoring_weights_raises(self):
        config = PipelineConfig(
            tickers=["TEST-A"],
            data_dir=self.data_dir,
            reports_dir=self.reports_dir,
            scoring_config=ScoringConfig(w_hurst=0.8, w_half_life=0.5, w_adf=0.1),  # Sum != 1.0
        )
        with self.assertRaises(ValueError):
            run_pipeline(config)

    def test_empty_tickers_raises_value_error(self):
        config = PipelineConfig(
            tickers=["NON_EXISTENT"],
            data_dir=self.data_dir,
            reports_dir=self.reports_dir,
        )
        with self.assertRaises(ValueError):
            run_pipeline(config)

    def test_generate_markdown_report_custom_date(self):
        config = PipelineConfig(
            tickers=["TEST-A"],
            lookbacks=["1Y"],
            data_dir=self.data_dir,
            reports_dir=self.reports_dir,
        )
        result = run_pipeline(config)
        md = generate_markdown_report(
            metric_matrix_df=result.metric_matrix_df,
            candidate_scores_df=result.candidate_scores_df,
            aggregate_rankings_df=result.aggregate_rankings_df,
            regime_allocations=result.regime_allocations,
            scoring_config=config.scoring_config,
            report_date="2026-10-01",
        )
        self.assertIn("*Generated: 2026-10-01*", md)
        self.assertIn("## 1. Scoring Methodology", md)
        self.assertIn("## 4. Multi-Horizon Aggregate Rankings", md)


if __name__ == "__main__":
    unittest.main()
