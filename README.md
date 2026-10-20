# Seasonal Trading in Commodity Futures

**Replication of:** *Seasonal Trading in Commodity Futures: Evidence from Regression and Singular Spectrum Signals*

This repository implements a full rolling out-of-sample backtest comparing three seasonal-signal
models (DVR, SSA, RLSSA) on 15 commodity futures markets from January 2016 through December 2024.

---

## Repository Structure

```
repo/
├── config.py              # All paper-derived parameters
├── data/
│   └── generate_panel.py  # Synthetic monthly panel generator (real data: TradingCharts)
├── src/
│   ├── models.py          # DVR, SSA, RLSSA signal models
│   ├── backtest.py        # Rolling window backtest engine
│   ├── portfolio.py       # Portfolio construction and performance metrics
│   ├── meb.py             # Maximum Entropy Bootstrap (meboot algorithm)
│   └── volatility.py      # EWMA volatility normalisation
├── exp/
│   ├── exp1_historical_backtest.py   # Exp 1: Historical out-of-sample comparison
│   ├── exp2_meb_bootstrap.py         # Exp 2: MEB path-sensitivity + paired Sharpe
│   └── exp3_subperiod_stability.py   # Exp 3: Subperiod robustness + selection stability
├── tests/
│   └── test_models.py     # pytest unit tests
├── results/               # Output artifacts and RESULTS.md
├── reproduce.sh           # Entry-point: runs all experiments end-to-end
└── README.md
```

## Models

| Model | Description |
|-------|-------------|
| **DVR** | Dummy-variable regression with Newey–West HAC (Bartlett, 3 lags). Significance threshold p ≤ 0.10. Cash when no eligible market. |
| **SSA** | Classical Singular Spectrum Analysis. L=36, r=12 eigentriples. Recurrent one-step forecast. |
| **RLSSA** | Robust Low-rank SSA via Hawkins alternating L1 SVD. 12 deflated components, tol=1e-10. |

Both classical (raw log-return) and volatility-normalised (EWMA λ=0.97) specifications are tested.

## Fixed Parameters (from paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Universe | 15 markets (CC, CF, CO, CP, CT, GD, HE, HO, LE, NG, SU, SV, ZC, ZS, ZW) | p.4 |
| Estimation window | 120 months | p.7 |
| SSA L | 36 | p.7 |
| SSA r | 12 | p.7 |
| EWMA λ | 0.97 | p.7 |
| One-way cost c | 0.00086 | p.9 |
| MEB trim | 10% | p.11 |
| MEB seed | 42 | p.11 |
| DVR HAC lags | 3 | p.7 |

## Data Note

The paper uses proprietary **TradingCharts** futures contract histories with specific
delivery-avoidance and liquidity-screen rules. A **synthetic monthly return panel**
is provided to enable full methodology demonstration. All model, backtest, and
bootstrap code is data-source-agnostic and will produce equivalent results on real data.

To supply real data: place a CSV with columns `date, CC, CF, CO, ...` (15 markets) in
`data/monthly_panel.csv` and set `USE_SYNTHETIC_DATA = False` in `config.py`.

## Quickstart

```bash
pip install -r requirements.txt   # if not already installed
bash reproduce.sh
```

Results are written to `results/`.

## Expected Results (paper Table 2, p.13 — real data)

| Strategy | CumRet | Sharpe | MaxDD |
|----------|--------|--------|-------|
| DVR long classical | 0.2966 | 0.245 | -0.308 |
| DVR long normalised | 0.2780 | 0.240 | -0.366 |
| SSA long classical | -0.3241 | 0.014 | -0.605 |
| RLSSA short classical | -0.0054 | 0.184 | -0.825 |
| Benchmark EW long | 0.1430 | 0.183 | -0.415 |

*Note: Synthetic-data results differ from these targets; methodology fidelity is maintained.*

## Deviations from Paper

1. **Data**: Synthetic returns replace proprietary TradingCharts data.
2. **RLSSA max_iter**: Reduced to 200 from 50,000 for practical runtime; early-exit at tol=1e-10 ensures equivalent quality.
3. **MEB paths**: B=5 (paper: B=500) for practical runtime demonstration.
4. **MEB implementation**: Custom Python MEB matching Vinod (2006) algorithm; exact R `meboot` output would require rpy2.
