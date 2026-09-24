"""Unit tests for src.scoring."""

import unittest
import numpy as np
import pandas as pd

from src.scoring import (
    adf_quality_factor,
    compute_aggregate_rankings,
    evaluate_candidate_models,
    half_life_quality_factor,
    hurst_quality_factor,
    mean_reversion_quality,
    score_exponential_ratio,
    score_power_ratio,
    score_two_pillar,
    score_weighted_additive_row,
)


class TestScoring(unittest.TestCase):
    def test_hurst_quality_factor(self):
        # Optimal anti-persistent
        self.assertEqual(hurst_quality_factor(0.35), 1.0)
        self.assertEqual(hurst_quality_factor(0.40), 1.0)
        # Random walk
        self.assertAlmostEqual(hurst_quality_factor(0.50), 2.0 / 3.0, places=3)
        # Toxic trending
        self.assertEqual(hurst_quality_factor(0.70), 0.0)
        self.assertEqual(hurst_quality_factor(0.85), 0.0)
        # Edge cases
        self.assertEqual(hurst_quality_factor(float("nan")), 0.0)

    def test_half_life_quality_factor(self):
        # Very fast reversion
        q20 = half_life_quality_factor(20.0, tau=90.0)
        self.assertAlmostEqual(q20, np.exp(-20.0 / 90.0), places=4)
        # Infinite half life
        self.assertEqual(half_life_quality_factor(float("inf")), 0.0)
        self.assertEqual(half_life_quality_factor(0.0), 0.0)
        self.assertEqual(half_life_quality_factor(-5.0), 0.0)
        self.assertEqual(half_life_quality_factor(float("nan")), 0.0)

    def test_adf_quality_factor(self):
        self.assertAlmostEqual(adf_quality_factor(0.01), 0.99, places=4)
        self.assertAlmostEqual(adf_quality_factor(0.50), 0.50, places=4)
        self.assertEqual(adf_quality_factor(1.2), 0.0)
        self.assertEqual(adf_quality_factor(float("nan")), 0.0)

    def test_mean_reversion_quality_bounds(self):
        # Worst case: H=0.8, HL=inf, ADF=1.0 -> 0.0
        q_worst = mean_reversion_quality(0.8, float("inf"), 1.0)
        self.assertEqual(q_worst, 0.0)

        # Best case: H=0.35, HL=10, ADF=0.001 -> near 1.0
        q_best = mean_reversion_quality(0.35, 10.0, 0.001)
        self.assertGreater(q_best, 0.90)

    def test_score_two_pillar_behavior(self):
        # High vol but zero mean-reversion -> composite score is 0
        comp_bad, vol_bad, q_bad = score_two_pillar(
            sd=0.05, hurst=0.75, half_life=float("inf"), adf_pvalue=1.0
        )
        self.assertEqual(q_bad, 0.0)
        self.assertEqual(comp_bad, 0.0)
        self.assertGreater(vol_bad, 100.0)

        # High vol AND high mean-reversion -> high composite
        comp_good, vol_good, q_good = score_two_pillar(
            sd=0.035, hurst=0.45, half_life=45.0, adf_pvalue=0.10
        )
        self.assertGreater(comp_good, 60.0)
        self.assertGreater(q_good, 0.50)

    def test_evaluate_candidate_models(self):
        df = pd.DataFrame([{
            "ticker": "TEST",
            "lookback": "3Y",
            "daily_sd": 0.03,
            "annual_sd": 0.45,
            "hurst": 0.48,
            "half_life": 50.0,
            "adf_pvalue": 0.15,
        }])
        res = evaluate_candidate_models(df)
        self.assertEqual(len(res), 1)
        self.assertIn("score_two_pillar", res.columns)
        self.assertIn("score_additive", res.columns)
        self.assertIn("score_power_ratio", res.columns)
        self.assertIn("score_exp_ratio", res.columns)

    def test_compute_aggregate_rankings(self):
        scored_df = pd.DataFrame([
            {"ticker": "AAA", "lookback": "1Y", "daily_sd": 0.03, "hurst": 0.48, "half_life": 50.0, "adf_pvalue": 0.1, "score_two_pillar": 70.0, "score_additive": 75.0, "q_mr": 0.7},
            {"ticker": "AAA", "lookback": "3Y", "daily_sd": 0.03, "hurst": 0.48, "half_life": 50.0, "adf_pvalue": 0.1, "score_two_pillar": 70.0, "score_additive": 75.0, "q_mr": 0.7},
            {"ticker": "AAA", "lookback": "5Y", "daily_sd": 0.03, "hurst": 0.48, "half_life": 50.0, "adf_pvalue": 0.1, "score_two_pillar": 70.0, "score_additive": 75.0, "q_mr": 0.7},
            {"ticker": "BBB", "lookback": "1Y", "daily_sd": 0.01, "hurst": 0.70, "half_life": 500.0, "adf_pvalue": 0.8, "score_two_pillar": 10.0, "score_additive": 30.0, "q_mr": 0.1},
            {"ticker": "BBB", "lookback": "3Y", "daily_sd": 0.01, "hurst": 0.70, "half_life": 500.0, "adf_pvalue": 0.8, "score_two_pillar": 10.0, "score_additive": 30.0, "q_mr": 0.1},
            {"ticker": "BBB", "lookback": "5Y", "daily_sd": 0.01, "hurst": 0.70, "half_life": 500.0, "adf_pvalue": 0.8, "score_two_pillar": 10.0, "score_additive": 30.0, "q_mr": 0.1},
        ])
        agg = compute_aggregate_rankings(scored_df)
        self.assertEqual(len(agg), 2)
        self.assertEqual(agg.iloc[0]["ticker"], "AAA")
        self.assertEqual(agg.iloc[1]["ticker"], "BBB")


if __name__ == "__main__":
    unittest.main()
