# Asset Selection & Strategy Allocation Report
**Quantitative Asset Observation & Mean-Reversion Grid Screening**
*Generated: 2026-09-26*

---

## Executive Summary

Candidate assets have been quantitatively evaluated across volatility, mean-reversion persistence, half-life decay, and regime stability across 1Y, 3Y, and 5Y horizons:

| Strategy Tier | Assets | Operational Objective |
|---|---|---|
| **Lane A (Mean-Reversion Grid)** | XRP-USD, CL=F, BZ=F | Laddered limit orders in range; profits from oscillatory swings |
| **Lane B (Zone-Concentration Entry)** | BTC-USD, GC=F, SI=F | Single/few entries at support zones; captures directional trend |
| **Observed Only** | ETH-USD | Disqualified from automated execution pending regime stabilization |

---

## 1. Scoring Methodology

### Two-Pillar Multiplicative Model
```
CompositeGridScore = VolPotential × Q_MR

VolPotential = min((daily_σ / 0.03) × 100, 150)
Q_MR         = 0.50 × HurstScore + 0.35 × HalfLifeScore + 0.15 × ADFScore
```

- **HurstScore**: Linear decay between $H \le 0.40 \rightarrow 1.0$ (strong MR) and $H \ge 0.70 \rightarrow 0.0$ (strong trend).
- **HalfLifeScore**: $\exp(-HL / 90d)$ exponential decay factor.
- **ADFScore**: $1 - \min(p, 1.0)$ unit root rejection confidence.
- **Multi-Horizon Aggregation**: 30% × 1Y + 50% × 3Y + 20% × 5Y

---

## 2. Volatility & Mean-Reversion Metric Matrix

| Asset | Lookback | Daily σ | Annual σ | Hurst (H) | Half-Life | ADF p-value |
|---|---|---|---|---|---|---|
| BTC-USD | 1Y | 2.38% | 45.5% | 0.611 | 53.2d | 0.309 |
| BTC-USD | 3Y | 2.47% | 47.2% | 0.634 | 102.7d | 0.025 |
| BTC-USD | 5Y | 2.70% | 51.6% | 0.593 | 766.0d | 0.814 |
| ETH-USD | 1Y | 3.38% | 64.5% | 0.642 | 53.7d | 0.301 |
| ETH-USD | 3Y | 3.48% | 66.5% | 0.497 | 73.0d | 0.117 |
| ETH-USD | 5Y | 3.61% | 68.9% | 0.500 | 132.4d | 0.208 |
| XRP-USD | 1Y | 3.53% | 67.4% | 0.499 | 50.2d | 0.217 |
| XRP-USD | 3Y | 3.98% | 76.0% | 0.558 | 273.5d | 0.589 |
| XRP-USD | 5Y | 4.17% | 79.6% | 0.548 | 423.8d | 0.720 |
| GC=F | 1Y | 1.84% | 29.1% | 0.502 | 19.7d | 0.142 |
| GC=F | 3Y | 1.36% | 21.7% | 0.732 | 379.3d | 0.711 |
| GC=F | 5Y | 1.20% | 19.1% | 0.666 | 8274.3d | 0.965 |
| SI=F | 1Y | 4.38% | 69.6% | 0.605 | 21.0d | 0.149 |
| SI=F | 3Y | 2.94% | 46.7% | 0.693 | 289.1d | 0.781 |
| SI=F | 5Y | 2.55% | 40.4% | 0.601 | 848.0d | 0.932 |
| CL=F | 1Y | 3.51% | 55.8% | 0.531 | 53.8d | 0.605 |
| CL=F | 3Y | 2.61% | 41.4% | 0.488 | 44.9d | 0.147 |
| CL=F | 5Y | 2.63% | 41.7% | 0.453 | 54.1d | 0.123 |
| BZ=F | 1Y | 3.47% | 55.0% | 0.522 | 53.2d | 0.533 |
| BZ=F | 3Y | 2.51% | 39.8% | 0.493 | 48.5d | 0.255 |
| BZ=F | 5Y | 2.50% | 39.8% | 0.457 | 56.8d | 0.164 |

---

## 3. Composite Grid Scores (Per Horizon)

### 1Y Horizon Rankings

| Rank | Asset | Daily σ | Hurst | Half-Life | Q_MR | Grid Score |
|---|---|---|---|---|---|---|
| 1 | SI=F | 4.38% | 0.605 | 21.0d | 0.564 | 82.3 |
| 2 | XRP-USD | 3.53% | 0.499 | 50.2d | 0.653 | 76.8 |
| 3 | BZ=F | 3.47% | 0.522 | 53.2d | 0.560 | 64.7 |
| 4 | CL=F | 3.51% | 0.531 | 53.8d | 0.533 | 62.5 |
| 5 | GC=F | 1.84% | 0.502 | 19.7d | 0.739 | 45.2 |
| 6 | ETH-USD | 3.38% | 0.642 | 53.7d | 0.394 | 44.3 |
| 7 | BTC-USD | 2.38% | 0.611 | 53.2d | 0.446 | 35.4 |

### 3Y Horizon Rankings

| Rank | Asset | Daily σ | Hurst | Half-Life | Q_MR | Grid Score |
|---|---|---|---|---|---|---|
| 1 | ETH-USD | 3.48% | 0.497 | 73.0d | 0.626 | 72.6 |
| 2 | CL=F | 2.61% | 0.488 | 44.9d | 0.693 | 60.3 |
| 3 | BZ=F | 2.51% | 0.493 | 48.5d | 0.662 | 55.3 |
| 4 | XRP-USD | 3.98% | 0.558 | 273.5d | 0.316 | 41.9 |
| 5 | BTC-USD | 2.47% | 0.634 | 102.7d | 0.368 | 30.3 |
| 6 | SI=F | 2.94% | 0.693 | 289.1d | 0.058 | 5.7 |
| 7 | GC=F | 1.36% | 0.732 | 379.3d | 0.048 | 2.2 |

### 5Y Horizon Rankings

| Rank | Asset | Daily σ | Hurst | Half-Life | Q_MR | Grid Score |
|---|---|---|---|---|---|---|
| 1 | CL=F | 2.63% | 0.453 | 54.1d | 0.735 | 64.3 |
| 2 | ETH-USD | 3.61% | 0.500 | 132.4d | 0.533 | 64.0 |
| 3 | BZ=F | 2.50% | 0.457 | 56.8d | 0.717 | 59.9 |
| 4 | XRP-USD | 4.17% | 0.548 | 423.8d | 0.298 | 41.4 |
| 5 | BTC-USD | 2.70% | 0.593 | 766.0d | 0.206 | 18.5 |
| 6 | SI=F | 2.55% | 0.601 | 848.0d | 0.175 | 14.8 |
| 7 | GC=F | 1.20% | 0.666 | 8274.3d | 0.062 | 2.5 |

---

## 4. Multi-Horizon Aggregate Rankings

| Rank | Asset | Aggregate Grid Score | Additive Score | Aggregate Q_MR | Mean Daily σ | Mean Hurst | Median Half-Life |
|---|---|---|---|---|---|---|---|
| 1 | ETH-USD | **62.4** | 72.3 | 0.538 | 3.49% | 0.546 | 73.0d |
| 2 | CL=F | **61.7** | 75.6 | 0.654 | 2.92% | 0.491 | 53.8d |
| 3 | BZ=F | **59.0** | 73.9 | 0.642 | 2.83% | 0.491 | 53.2d |
| 4 | XRP-USD | **52.3** | 64.8 | 0.413 | 3.89% | 0.535 | 273.5d |
| 5 | SI=F | **30.5** | 52.4 | 0.233 | 3.29% | 0.633 | 289.1d |
| 6 | BTC-USD | **29.5** | 54.7 | 0.359 | 2.52% | 0.613 | 102.7d |
| 7 | GC=F | **15.2** | 35.1 | 0.258 | 1.47% | 0.634 | 379.3d |

---

## 5. Deployment Lane Allocation & Regime Scorecard

| Asset | Target Lane | Role | Regime Stability | Allocation Verdict | Primary Risk Factor |
|---|---|---|---|---|---|
| **BTC-USD** | Lane B (Zone-Concentration Entry) | Active Deploy | Consistent | Stable persistent trend | High trend-blowout risk for MR grids; suitable for directional zone entry |
| **ETH-USD** | Observed Only | Observed Only | Regime Flip | Regime flip across horizons | Regime flip; unreliable grid assumptions |
| **XRP-USD** | Lane A (Mean-Reversion Grid) | Learning / Short-Cycle | Consistent | MR short-cycle; long-term half-life blowout | Half-life expansion over long horizons |
| **GC=F** | Lane B (Zone-Concentration Entry) | Active Deploy | Consistent | Stable persistent trend | High trend-blowout risk for MR grids; suitable for directional zone entry |
| **SI=F** | Lane B (Zone-Concentration Entry) | Active Deploy | Consistent | Stable persistent trend | High trend-blowout risk for MR grids; suitable for directional zone entry |
| **CL=F** | Lane A (Mean-Reversion Grid) | Primary Deploy | Consistent | Stable MR, primary deploy | Commodity/macro supply shock |
| **BZ=F** | Lane A (Mean-Reversion Grid) | Parallel Deploy | Consistent | Stable MR, parallel commodity deploy | Geopolitical spread to WTI |

---

## 6. Operational Recommendations & Guardrails

1. **Primary Deployment (Lane A):** Focus automated grid execution on assets with verified multi-horizon mean-reversion (`CL=F`, `BZ=F`).
2. **Learning / Short-Cycle Deployment:** Assets with short-horizon mean-reversion but long-horizon half-life expansion (`XRP-USD`) must use strict dynamic cycle timeouts and tightened inventory caps.
3. **Directional Zone Deployment (Lane B):** High-volatility trending assets (`GC=F`, `SI=F`, `BTC-USD`) should be traded with directional pullback entries rather than continuous bi-directional grids.
4. **Observed Quarantine:** Assets with regime flips across horizons (`ETH-USD`) must remain observed-only until 1Y and 3Y memory metrics re-align.
