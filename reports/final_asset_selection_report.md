# Final Asset Selection Ranking Report
**Grid Trading Strategy — Asset Observation Project**
*Generated: 2026-09-25 | Horizon: 2021-09-23 to 2026-09-23 (up to 5Y daily OHLCV)*

---

## Executive Summary

All 7 candidate assets have been evaluated across volatility, mean-reversion strength, regime consistency, and composite grid score. They are assigned to one of three deployment tiers:

| Tier | Assets | Purpose |
|---|---|---|
| **Lane A — Mean-Reversion Grid** | CL=F, BZ=F, XRP-USD | Laddered orders within range; profits from oscillation |
| **Lane B — Zone-Concentration Entry** | GC=F, SI=F, BTC-USD | One/few entries at key zone; rides directional persistence |
| **Observed Only** | ETH-USD | Regime flip disqualifies from either lane pending stabilisation |

---

## 1. Scoring Methodology

### Formula: Two-Pillar Multiplicative Model

```
CompositeGridScore = VolPotential × Q_MR

VolPotential = (daily_σ / ref_σ) × 100  [ref_σ = 3.0% daily SD; cap 150]
Q_MR         = 0.50 × HurstScore + 0.35 × HalfLifeScore + 0.15 × ADFScore
```

**Mean-Reversion Quality (Q_MR) sub-scores:**
- **HurstScore**: Linear interpolation: H ≤ 0.40 → 1.0 (strong MR); H ≥ 0.70 → 0.0 (strong trend)
- **HalfLifeScore**: `exp(−HL / 90d)` — exponential decay; short half-life scores near 1.0
- **ADFScore**: `1 − min(p, 1)` — lower p-value (more stationary) scores higher

**Horizon aggregation:** 30% × 1Y + 50% × 3Y + 20% × 5Y

The multiplicative structure is intentional: an asset trending at Q_MR ≈ 0 collapses to near-zero regardless of volatility, providing a hard gate against grid deployment in persistent trends.

---

## 2. Metric Matrix (All 7 Assets × 3 Horizons)

| Asset | Lookback | Daily σ | Annual σ | Hurst | Half-Life | ADF p |
|---|---|---|---|---|---|---|
| BTC-USD | 1Y | 2.38% | 45.5% | 0.611 | 53d | 0.309 |
| BTC-USD | 3Y | 2.47% | 47.2% | 0.634 | 103d | 0.025 |
| BTC-USD | 5Y | 2.70% | 51.6% | 0.593 | 766d | 0.814 |
| ETH-USD | 1Y | 3.38% | 64.5% | 0.642 | 54d | 0.301 |
| ETH-USD | 3Y | 3.48% | 66.5% | 0.497 | 73d | 0.117 |
| ETH-USD | 5Y | 3.61% | 68.9% | 0.500 | 132d | 0.208 |
| XRP-USD | 1Y | 3.53% | 67.4% | 0.499 | 50d | 0.217 |
| XRP-USD | 3Y | 3.98% | 76.0% | 0.558 | 274d | 0.589 |
| XRP-USD | 5Y | 4.17% | 79.6% | 0.548 | 424d | 0.720 |
| GC=F | 1Y | 1.84% | 29.1% | 0.502 | 20d | 0.142 |
| GC=F | 3Y | 1.36% | 21.7% | 0.732 | 379d | 0.711 |
| GC=F | 5Y | 1.20% | 19.1% | 0.666 | 8,274d | 0.965 |
| SI=F | 1Y | 4.38% | 69.6% | 0.605 | 21d | 0.149 |
| SI=F | 3Y | 2.94% | 46.7% | 0.693 | 289d | 0.781 |
| SI=F | 5Y | 2.55% | 40.4% | 0.601 | 848d | 0.932 |
| CL=F | 1Y | 3.52% | 55.8% | 0.531 | 54d | 0.605 |
| CL=F | 3Y | 2.61% | 41.4% | 0.488 | 45d | 0.147 |
| CL=F | 5Y | 2.63% | 41.7% | 0.453 | 54d | 0.123 |
| BZ=F | 1Y | 3.47% | 55.0% | 0.522 | 53d | 0.533 |
| BZ=F | 3Y | 2.51% | 39.8% | 0.493 | 48d | 0.255 |
| BZ=F | 5Y | 2.50% | 39.8% | 0.457 | 57d | 0.164 |

---

## 3. Composite Grid Scores

### Per-Horizon Rankings (Two-Pillar Multiplicative Model)

**1-Year Horizon:**

| Rank | Asset | Daily σ | Hurst | Half-Life | Q_MR | Grid Score |
|---|---|---|---|---|---|---|
| 1 | SI=F | 4.38% | 0.605 | 21d | 0.564 | 82.3 |
| 2 | XRP-USD | 3.53% | 0.499 | 50d | 0.653 | 76.8 |
| 3 | BZ=F | 3.47% | 0.522 | 53d | 0.560 | 64.7 |
| 4 | CL=F | 3.52% | 0.531 | 54d | 0.533 | 62.5 |
| 5 | GC=F | 1.84% | 0.502 | 20d | 0.739 | 45.2 |
| 6 | ETH-USD | 3.38% | 0.642 | 54d | 0.394 | 44.3 |
| 7 | BTC-USD | 2.38% | 0.611 | 53d | 0.446 | 35.4 |

**3-Year Horizon:**

| Rank | Asset | Daily σ | Hurst | Half-Life | Q_MR | Grid Score |
|---|---|---|---|---|---|---|
| 1 | ETH-USD | 3.48% | 0.497 | 73d | 0.626 | 72.6 |
| 2 | CL=F | 2.61% | 0.488 | 45d | 0.693 | 60.3 |
| 3 | BZ=F | 2.51% | 0.493 | 48d | 0.662 | 55.3 |
| 4 | XRP-USD | 3.98% | 0.558 | 274d | 0.316 | 41.9 |
| 5 | BTC-USD | 2.47% | 0.634 | 103d | 0.368 | 30.3 |
| 6 | SI=F | 2.94% | 0.693 | 289d | 0.058 | 5.7 |
| 7 | GC=F | 1.36% | 0.732 | 379d | 0.048 | 2.2 |

**5-Year Horizon:**

| Rank | Asset | Daily σ | Hurst | Half-Life | Q_MR | Grid Score |
|---|---|---|---|---|---|---|
| 1 | CL=F | 2.63% | 0.453 | 54d | 0.735 | 64.3 |
| 2 | ETH-USD | 3.61% | 0.500 | 132d | 0.533 | 64.0 |
| 3 | BZ=F | 2.50% | 0.457 | 57d | 0.717 | 59.9 |
| 4 | XRP-USD | 4.17% | 0.548 | 424d | 0.298 | 41.4 |
| 5 | BTC-USD | 2.70% | 0.593 | 766d | 0.206 | 18.5 |
| 6 | SI=F | 2.55% | 0.601 | 848d | 0.175 | 14.8 |
| 7 | GC=F | 1.20% | 0.666 | 8,274d | 0.062 | 2.5 |

### Multi-Horizon Aggregate Rankings (30% 1Y + 50% 3Y + 20% 5Y)

| Rank | Asset | Agg. Grid Score | Agg. Q_MR | Mean Daily σ | Mean Hurst | Median Half-Life |
|---|---|---|---|---|---|---|
| 1 | ETH-USD | **62.4** | 0.538 | 3.49% | 0.546 | 73d |
| 2 | CL=F | **61.7** | 0.654 | 2.92% | 0.491 | 54d |
| 3 | BZ=F | **59.0** | 0.642 | 2.83% | 0.491 | 53d |
| 4 | XRP-USD | **52.3** | 0.413 | 3.89% | 0.535 | 274d |
| 5 | SI=F | **30.5** | 0.233 | 3.29% | 0.633 | 289d |
| 6 | BTC-USD | **29.5** | 0.359 | 2.52% | 0.613 | 103d |
| 7 | GC=F | **15.2** | 0.258 | 1.47% | 0.634 | 379d |

> **Note on ETH-USD rank 1:** The aggregate score places ETH-USD first, but raw aggregate score is overridden by regime consistency as the governing criterion. ETH-USD's Hurst flip from 0.642 (1Y trending) to 0.497 (3Y MR) to 0.500 (5Y MR) renders it unreliable for deployment. CL=F and BZ=F, with stable Hurst in the 0.45–0.53 range across all horizons, are the true primary MR candidates.

---

## 4. Regime Consistency Analysis

**Regime consistency** — stability of Hurst and half-life character across 1Y/3Y/5Y horizons — is the primary deployment criterion. An asset with a high aggregate score but an unstable regime is more dangerous than one with a moderate but consistent score, because grid sizing assumptions made at deployment can be invalidated mid-run.

| Asset | 1Y Hurst | 3Y Hurst | 5Y Hurst | Median HL | Regime verdict | Risk flag |
|---|---|---|---|---|---|---|
| **CL=F** | 0.531 | 0.488 | 0.453 | 54d | ✅ Stable MR, improving | None |
| **BZ=F** | 0.522 | 0.493 | 0.457 | 53d | ✅ Stable MR, improving | None |
| **XRP-USD** | 0.499 | 0.558 | 0.548 | 274d | ⚠️ MR at 1Y, HL blowout long-term | HL unstable |
| **GC=F** | 0.502 | 0.732 | 0.666 | 379d | ✅ Stable strong trend | Grid blowout |
| **SI=F** | 0.605 | 0.693 | 0.601 | 289d | ✅ Stable strong trend | Grid blowout |
| **BTC-USD** | 0.611 | 0.634 | 0.593 | 103d | ✅ Stable trend | Grid blowout |
| **ETH-USD** | 0.642 | 0.497 | 0.500 | 73d | ❌ Regime flip 1Y→3Y | Unreliable |

**Trend blowout risk:** GC=F, SI=F, and BTC-USD display persistent Hurst (> 0.59 across all horizons) and long half-lives that create structural inventory blowout risk in a mean-reversion grid. They are ruled out of Lane A and retained in Lane B where directional persistence is an asset.

---

## 5. Final Deployment Assignments

### Lane A — Mean-Reversion Grid

*Strategy: Laddered buy/sell limit orders within a price range. Profits from oscillatory price swings. Requires sustained MR character and manageable half-life.*

| Asset | Role | Rationale |
|---|---|---|
| **CL=F (WTI Crude Oil)** | 🟢 Primary deploy | Most regime-consistent MR asset. Hurst 0.45–0.53, HL 44–54d stable across all three windows. Composite score 61.7. |
| **BZ=F (Brent Crude Oil)** | 🟢 Parallel deploy | Near-identical profile to WTI (Hurst 0.46–0.52, HL 48–57d). Deployed alongside CL=F rather than redundantly; Brent and WTI occasionally diverge on geopolitical shocks, providing complementary coverage. |
| **XRP-USD** | 🟡 Learning / short-cycle | Genuine MR at 1Y (H=0.499, HL=50d, score=76.8). Half-life blows out severely at 3Y/5Y (274d/424d), ranking drops sharply. Kept active in a limited, short-cycle capacity to observe intraday oscillation behaviour and gather live data. Not a primary commitment. |

### Lane B — Zone-Concentration Entry

*Strategy: One or a few entries at a key support/resistance zone, riding directional trend momentum. Requires persistent directional character (H > 0.5), not MR. Profits scale with move size, not oscillation count.*

| Asset | Role | Rationale |
|---|---|---|
| **GC=F (Gold)** | 🟢 Active deploy | Strongest persistent trend: H=0.73 at 3Y, HL=8,274d at 5Y. Aggregate grid score 15.2 (correctly near-zero — pure trending asset). Zone-entry approach matches its structural character. |
| **SI=F (Silver)** | 🟢 Active deploy | Similar persistent trend profile: H=0.69 at 3Y, HL=289d at 3Y/848d at 5Y. High 1Y volatility (4.38% daily σ) useful for large zone moves. Complements Gold with higher vol multiplier. |
| **BTC-USD** | 🟢 Active deploy | Consistently trending: H=0.59–0.63 across all horizons, HL 53d→766d. Higher aggregate grid score than Gold/Silver (29.5) but still disqualified from MR grid on blowout grounds. Zone-entry approach suits its character. |

### Observed Only

| Asset | Rationale |
|---|---|
| **ETH-USD** | Regime flip: H=0.642 (1Y, trending) → H=0.497 (3Y, MR) → H=0.500 (5Y, MR). Highest aggregate grid score (62.4) but least consistent across horizons — the score flatters an asset whose regime is unresolved. Kept under observation; eligible for Lane A reconsideration if the MR regime (H < 0.52) is confirmed at 1Y in the next review cycle. |

---

## 6. Decision Summary

| Asset | Deployment Lane | Primary Signal | Key Risk |
|---|---|---|---|
| CL=F | Lane A — MR Grid (Primary) | Stable Hurst 0.45–0.53, HL 44–54d | Commodity supply shock (regime break) |
| BZ=F | Lane A — MR Grid (Parallel) | Stable Hurst 0.46–0.52, HL 48–57d | Geopolitical spread to WTI |
| XRP-USD | Lane A — MR Grid (Learning) | MR at 1Y; HL stable only short-term | Long-term HL blowout; not primary |
| GC=F | Lane B — Zone Entry | H=0.73 persistent trend | Low volatility (1.47% daily σ) |
| SI=F | Lane B — Zone Entry | H=0.69 persistent trend, high vol | Volatile zone moves — sizing discipline |
| BTC-USD | Lane B — Zone Entry | H=0.59–0.63 consistent trend | Crypto-specific regime events |
| ETH-USD | Observed Only | Regime flip — unresolved character | Cannot rely on either lane assumption |

---

## 7. Data & Implementation References

| Artifact | Location |
|---|---|
| Raw metric matrix | `reports/metric_matrix.csv` |
| Per-horizon composite scores | `reports/candidate_scores.csv` |
| Multi-horizon aggregate rankings | `reports/aggregate_rankings.csv` |
| Scoring implementation | `src/scoring.py` |
| Scoring weight guide | `docs/scoring-weights-guide.md` |
| Statistical method decisions | Issue #3 (research branch `research/survey-stat-methods`) |
| Regime evaluation detail | Issue #6 (resolution comment) |

---

## 8. Next Steps (Post-Map)

The following are outside the scope of this map and suitable for a new planning effort:

- **Lane A: Grid Sizing & Banding** — Determine grid count, range width, and band spacing for CL=F and BZ=F based on their historical daily σ and half-life profiles.
- **Lane B: Zone-Concentration Entry Parameters** — Identify key support/resistance zones for GC=F, SI=F, and BTC-USD; define position sizing and trailing exit rules.
- **ETH Review Trigger** — Define the quantitative threshold (e.g., 1Y Hurst < 0.52 for two consecutive monthly reviews) that triggers ETH-USD promotion to Lane A.
