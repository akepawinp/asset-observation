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
    def test_evaluate_regime_stability_and_lane_allocation(self):
        from src.scoring import (
            DeploymentLane,
            LaneRole,
            allocate_deployment_lanes,
            evaluate_regime_stability,
        )

        # 1. Stable MR asset (e.g. CL=F)
        df_clf = pd.DataFrame([
            {"ticker": "CL=F", "lookback": "1Y", "daily_sd": 0.035, "annual_sd": 0.55, "hurst": 0.531, "half_life": 53.8, "adf_pvalue": 0.60},
            {"ticker": "CL=F", "lookback": "3Y", "daily_sd": 0.026, "annual_sd": 0.41, "hurst": 0.488, "half_life": 44.9, "adf_pvalue": 0.15},
            {"ticker": "CL=F", "lookback": "5Y", "daily_sd": 0.026, "annual_sd": 0.41, "hurst": 0.453, "half_life": 54.1, "adf_pvalue": 0.12},
        ])
        eval_clf = evaluate_regime_stability("CL=F", df_clf)
        self.assertTrue(eval_clf.is_regime_consistent)
        self.assertEqual(eval_clf.lane, DeploymentLane.LANE_A)
        self.assertEqual(eval_clf.role, LaneRole.PRIMARY)

        # 2. Parallel MR asset (e.g. BZ=F)
        df_bzf = pd.DataFrame([
            {"ticker": "BZ=F", "lookback": "1Y", "daily_sd": 0.034, "annual_sd": 0.55, "hurst": 0.522, "half_life": 53.2, "adf_pvalue": 0.53},
            {"ticker": "BZ=F", "lookback": "3Y", "daily_sd": 0.025, "annual_sd": 0.39, "hurst": 0.493, "half_life": 48.5, "adf_pvalue": 0.25},
            {"ticker": "BZ=F", "lookback": "5Y", "daily_sd": 0.025, "annual_sd": 0.39, "hurst": 0.457, "half_life": 56.8, "adf_pvalue": 0.16},
        ])
        eval_bzf = evaluate_regime_stability("BZ=F", df_bzf)
        self.assertTrue(eval_bzf.is_regime_consistent)
        self.assertEqual(eval_bzf.lane, DeploymentLane.LANE_A)
        self.assertEqual(eval_bzf.role, LaneRole.PARALLEL)

        # 3. Learning/short-cycle MR asset with long-term HL blowout (e.g. XRP-USD)
        df_xrp = pd.DataFrame([
            {"ticker": "XRP-USD", "lookback": "1Y", "daily_sd": 0.035, "annual_sd": 0.67, "hurst": 0.499, "half_life": 50.2, "adf_pvalue": 0.22},
            {"ticker": "XRP-USD", "lookback": "3Y", "daily_sd": 0.040, "annual_sd": 0.76, "hurst": 0.558, "half_life": 273.5, "adf_pvalue": 0.59},
            {"ticker": "XRP-USD", "lookback": "5Y", "daily_sd": 0.042, "annual_sd": 0.80, "hurst": 0.548, "half_life": 423.8, "adf_pvalue": 0.72},
        ])
        eval_xrp = evaluate_regime_stability("XRP-USD", df_xrp)
        self.assertEqual(eval_xrp.lane, DeploymentLane.LANE_A)
        self.assertEqual(eval_xrp.role, LaneRole.LEARNING)

        # 4. Persistent trending asset (e.g. GC=F, SI=F, BTC-USD)
        df_gold = pd.DataFrame([
            {"ticker": "GC=F", "lookback": "1Y", "daily_sd": 0.018, "annual_sd": 0.29, "hurst": 0.502, "half_life": 19.7, "adf_pvalue": 0.14},
            {"ticker": "GC=F", "lookback": "3Y", "daily_sd": 0.014, "annual_sd": 0.22, "hurst": 0.732, "half_life": 379.3, "adf_pvalue": 0.71},
            {"ticker": "GC=F", "lookback": "5Y", "daily_sd": 0.012, "annual_sd": 0.19, "hurst": 0.666, "half_life": 8274.3, "adf_pvalue": 0.96},
        ])
        eval_gold = evaluate_regime_stability("GC=F", df_gold)
        self.assertEqual(eval_gold.lane, DeploymentLane.LANE_B)
        self.assertEqual(eval_gold.role, LaneRole.ACTIVE)

        # 5. Regime flip asset (e.g. ETH-USD) -> Observed Only
        df_eth = pd.DataFrame([
            {"ticker": "ETH-USD", "lookback": "1Y", "daily_sd": 0.034, "annual_sd": 0.64, "hurst": 0.642, "half_life": 53.7, "adf_pvalue": 0.30},
            {"ticker": "ETH-USD", "lookback": "3Y", "daily_sd": 0.035, "annual_sd": 0.66, "hurst": 0.497, "half_life": 73.0, "adf_pvalue": 0.12},
            {"ticker": "ETH-USD", "lookback": "5Y", "daily_sd": 0.036, "annual_sd": 0.69, "hurst": 0.500, "half_life": 132.4, "adf_pvalue": 0.21},
        ])
        eval_eth = evaluate_regime_stability("ETH-USD", df_eth)
        self.assertFalse(eval_eth.is_regime_consistent)
        self.assertEqual(eval_eth.lane, DeploymentLane.OBSERVED_ONLY)
        self.assertEqual(eval_eth.role, LaneRole.OBSERVED)

        # Batch allocation across all candidate dataframes
        combined_df = pd.concat([df_clf, df_bzf, df_xrp, df_gold, df_eth])
        allocations = allocate_deployment_lanes(combined_df)
        self.assertEqual(len(allocations), 5)
        self.assertEqual(allocations["CL=F"].lane, DeploymentLane.LANE_A)
        self.assertEqual(allocations["ETH-USD"].lane, DeploymentLane.OBSERVED_ONLY)

    def test_actual_candidate_metric_matrix_allocations(self):
        from pathlib import Path
        from src.scoring import (
            DeploymentLane,
            LaneRole,
            allocate_deployment_lanes,
        )

        matrix_path = Path("reports/metric_matrix.csv")
        if not matrix_path.exists():
            return

        df = pd.read_csv(matrix_path)
        allocations = allocate_deployment_lanes(df)

        # 7 candidate assets
        self.assertEqual(len(allocations), 7)

        # Lane A assets
        self.assertEqual(allocations["CL=F"].lane, DeploymentLane.LANE_A)
        self.assertEqual(allocations["CL=F"].role, LaneRole.PRIMARY)

        self.assertEqual(allocations["BZ=F"].lane, DeploymentLane.LANE_A)
        self.assertEqual(allocations["BZ=F"].role, LaneRole.PARALLEL)

        self.assertEqual(allocations["XRP-USD"].lane, DeploymentLane.LANE_A)
        self.assertEqual(allocations["XRP-USD"].role, LaneRole.LEARNING)

        # Lane B assets
        self.assertEqual(allocations["GC=F"].lane, DeploymentLane.LANE_B)
        self.assertEqual(allocations["GC=F"].role, LaneRole.ACTIVE)

        self.assertEqual(allocations["SI=F"].lane, DeploymentLane.LANE_B)
        self.assertEqual(allocations["SI=F"].role, LaneRole.ACTIVE)

        self.assertEqual(allocations["BTC-USD"].lane, DeploymentLane.LANE_B)
        self.assertEqual(allocations["BTC-USD"].role, LaneRole.ACTIVE)

        # Observed only asset
        self.assertEqual(allocations["ETH-USD"].lane, DeploymentLane.OBSERVED_ONLY)
        self.assertEqual(allocations["ETH-USD"].role, LaneRole.OBSERVED)
        self.assertFalse(allocations["ETH-USD"].is_regime_consistent)


if __name__ == "__main__":
    unittest.main()
