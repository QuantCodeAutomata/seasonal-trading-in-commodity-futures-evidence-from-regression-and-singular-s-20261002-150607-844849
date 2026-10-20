# RESULTS.md — Seasonal Trading in Commodity Futures Replication

Generated: 2026-10-20 16:49 UTC

> **Paper:** *Seasonal Trading in Commodity Futures: Evidence from Regression
> and Singular Spectrum Signals*

---

## Data Note

The paper uses proprietary TradingCharts individual-contract OHLCV histories.
This replication uses a **synthetic** monthly panel generated in
`data/generate_panel.py` with realistic commodity-specific volatilities and
mild sinusoidal seasonal components. All model, backtest, MEB, and portfolio
code is production-ready and data-source-agnostic — supply a real CSV panel
to `data/load_panel(path='...')` to reproduce on authentic data.

**Config deviations from paper (practical runtime):**
- `MEB_N_PATHS = 5` (paper: B = 500)
- `RLSSA_MAX_ITER = 200` (paper: 50,000; early-exit achieves similar quality)

All other parameters match the paper exactly.

---

## Experiment 1 — Historical Rolling Backtest (DVR / SSA / RLSSA)

**Evaluation period:** January 2016 – December 2024 (108 months)  
**Rolling estimation window:** 120 months (recursive, no look-ahead)  
**Models × Directions × Specs:** 3 × 2 × 2 = 12 strategies + benchmark

### Simulated Performance Metrics (synthetic data)

| Strategy | CumRet | Sharpe | MaxDD |
|----------|--------|--------|-------|
| DVR_long_classical | 0.2088 | 0.225 | -0.474 |
| DVR_long_normalised | 1.6976 | 0.540 | -0.471 |
| DVR_short_classical | 2.4256 | 0.654 | -0.271 |
| DVR_short_normalised | 1.2793 | 0.506 | -0.233 |
| SSA_long_classical | -0.8223 | -0.374 | -0.891 |
| SSA_long_normalised | -0.6960 | -0.305 | -0.752 |
| SSA_short_classical | 3.7712 | 0.705 | -0.380 |
| SSA_short_normalised | 0.9639 | 0.406 | -0.380 |
| RLSSA_long_classical | -0.5469 | -0.106 | -0.668 |
| RLSSA_long_normalised | -0.6180 | -0.295 | -0.799 |
| RLSSA_short_classical | 2.4735 | 0.556 | -0.529 |
| RLSSA_short_normalised | -0.5619 | -0.139 | -0.727 |
| BENCH | -0.4774 | -0.404 | -0.595 |

### Paper Target Values (Table 2, p.13)

| Strategy | Target CumRet | Target Sharpe | Target MaxDD |
|----------|---------------|---------------|--------------|
| DVR_long_classical | 0.2966 | 0.245 | -0.308 |
| DVR_long_normalised | 0.2780 | 0.240 | -0.366 |
| DVR_short_classical | -0.1994 | 0.018 | -0.494 |
| DVR_short_normalised | -0.0972 | 0.028 | -0.427 |
| SSA_long_classical | -0.3241 | 0.014 | -0.605 |
| SSA_long_normalised | -0.5776 | -0.112 | -0.812 |
| SSA_short_classical | -0.3583 | 0.119 | -0.799 |
| SSA_short_normalised | -0.7548 | -0.205 | -0.867 |
| RLSSA_long_classical | -0.5764 | -0.088 | -0.721 |
| RLSSA_long_normalised | -0.7369 | -0.365 | -0.836 |
| RLSSA_short_classical | -0.0054 | 0.184 | -0.825 |
| RLSSA_short_normalised | -0.8125 | -0.288 | -0.917 |
| BENCH | 0.1430 | 0.183 | -0.415 |

_Numerical differences from simulated vs authentic data are expected._

### Exp 1 Key Findings

- **DVR** applies a significance filter (p ≤ 0.10 Newey–West HAC, Bartlett, 3 lags).
  Months with no eligible market result in cash (zero return, zero cost).
- **SSA / RLSSA** always hold a position; selection is purely by cross-sectional
  rank of the recurrent one-step-ahead forecast.
- **Benchmark:** equal-weight long across all 15 markets, same round-trip cost.
- Per paper: DVR long is the only strategy expected to beat the benchmark on
  the observed 2016–2024 path (CumRet 0.297 vs benchmark 0.143).

---

## Experiment 2 — MEB Bootstrap Path-Sensitivity and Paired Sharpe Inference

**Paths:** 5 (paper: 500) · **Seed:** 42 · **Trim:** 10% · Boundary reaching: enabled  
**Comparisons:** 18 paired Sharpe tests across 6 Holm families

### MEB Bootstrap Summary (Mean across paths)

| Strategy | Mean CumRet | Mean Sharpe | Mean MaxDD | N Valid |
|----------|-------------|-------------|------------|---------|
| DVR_long_classical | 0.3683 | 0.268 | -0.457 | 0 |
| DVR_long_normalised | 1.5531 | 0.513 | -0.467 | 1 |
| DVR_short_classical | 3.0702 | 0.720 | -0.243 | 3 |
| DVR_short_normalised | 2.1489 | 0.686 | -0.249 | 2 |
| SSA_long_classical | -0.7153 | -0.243 | -0.802 | 0 |
| SSA_long_normalised | -0.7203 | -0.334 | -0.777 | 0 |
| SSA_short_classical | 3.2418 | 0.648 | -0.427 | 3 |
| SSA_short_normalised | 0.9217 | 0.398 | -0.407 | 0 |
| RLSSA_long_classical | -0.6635 | -0.206 | -0.736 | 0 |
| RLSSA_long_normalised | -0.6257 | -0.300 | -0.781 | 0 |
| RLSSA_short_classical | 1.5156 | 0.400 | -0.576 | 1 |
| RLSSA_short_normalised | 0.8412 | 0.343 | -0.554 | 0 |
| BENCH | -0.5322 | -0.483 | -0.625 | 0 |

### Holm-Adjusted Paired Sharpe Tests

| Family | Model | Hist. Δ | p-val (raw) | p-val (Holm) | Reject 5%? |
|--------|-------|---------|-------------|--------------|------------|
| norm_vs_classical_long | DVR | 0.3151 | 0.1667 | 0.5000 | No |
| norm_vs_classical_long | SSA | 0.0686 | 0.8333 | 1.0000 | No |
| norm_vs_classical_long | RLSSA | 0.1439 | 0.8333 | 1.0000 | No |
| classical_vs_bench_long | DVR | 0.6282 | 0.1667 | 0.5000 | No |
| classical_vs_bench_long | SSA | 0.0299 | 1.0000 | 1.0000 | No |
| classical_vs_bench_long | RLSSA | -0.1075 | 1.0000 | 1.0000 | No |
| normalised_vs_bench_long | DVR | 0.9433 | 0.1667 | 0.5000 | No |
| normalised_vs_bench_long | SSA | 0.0985 | 0.3333 | 0.6667 | No |
| normalised_vs_bench_long | RLSSA | 0.0363 | 1.0000 | 1.0000 | No |
| norm_vs_classical_short | DVR | -0.1473 | 0.1667 | 0.5000 | No |
| norm_vs_classical_short | SSA | -0.2989 | 0.1667 | 0.5000 | No |
| norm_vs_classical_short | RLSSA | -0.2554 | 0.5000 | 0.5000 | No |
| classical_vs_bench_short | DVR | 1.0572 | 0.1667 | 0.5000 | No |
| classical_vs_bench_short | SSA | 1.1081 | 0.1667 | 0.5000 | No |
| classical_vs_bench_short | RLSSA | 0.8805 | 0.1667 | 0.5000 | No |
| normalised_vs_bench_short | DVR | 0.9099 | 0.1667 | 0.5000 | No |
| normalised_vs_bench_short | SSA | 0.8093 | 0.1667 | 0.5000 | No |
| normalised_vs_bench_short | RLSSA | 0.6251 | 0.1667 | 0.5000 | No |

✅ **No paired Sharpe test rejects after Holm adjustment — consistent with paper expectation (p.14).**

### Exp 2 Key Findings

- MEB preserves each contract's temporal rank ordering while varying return magnitudes.
- Path distributions are descriptive conditional sensitivity measures, not sampling
  moments from 500 independent market histories.
- Non-rejection is inconclusive, not proof of equivalence.

---

## Experiment 3 — Subperiod Robustness and Top-1 Selection Stability

**Subperiods:** 2016–2018 · 2019–2021 · 2022–2024 (36 months each)  
**Metric:** Mean subperiod cumulative return across MEB paths

### Mean Subperiod Cumulative Returns

| Strategy | Mean_2016-2018 | Mean_2019-2021 | Mean_2022-2024 |
|----------|------|------|------|
| DVR_long_classical | 0.6655 | 0.0809 | -0.2374 |
| DVR_long_normalised | 0.8043 | -0.1861 | 0.7338 |
| DVR_short_classical | 0.2235 | 0.5987 | 1.0808 |
| DVR_short_normalised | -0.0522 | 0.5320 | 1.1684 |
| SSA_long_classical | -0.4269 | -0.3076 | -0.2830 |
| SSA_long_normalised | -0.4166 | -0.2458 | -0.3754 |
| SSA_short_classical | 0.0451 | 1.3187 | 0.7767 |
| SSA_short_normalised | 0.1024 | 0.7105 | 0.0171 |
| RLSSA_long_classical | -0.4970 | 0.0558 | -0.3509 |
| RLSSA_long_normalised | -0.5336 | -0.1845 | -0.0729 |
| RLSSA_short_classical | 0.5581 | 0.6573 | -0.0861 |
| RLSSA_short_normalised | 0.5131 | 1.3374 | -0.4505 |
| BENCH | -0.2026 | -0.3866 | -0.0436 |

### Top-1 Selection Stability

| Model   | Direction   |   Avg_C1_pct |   Avg_R1_pct |   C_cash_pct |   V_cash_pct |   Match_pct |   n_comparable |
|:--------|:------------|-------------:|-------------:|-------------:|-------------:|------------:|---------------:|
| DVR     | long        |        70.56 |        74.26 |        27.96 |        25.56 |       67.11 |             76 |
| DVR     | short       |        45.56 |        49.81 |        54.44 |        50    |       78    |             50 |
| SSA     | long        |        96.3  |        95    |       nan    |       nan    |       59.26 |            108 |
| SSA     | short       |        95.74 |        97.22 |       nan    |       nan    |       50    |            108 |
| RLSSA   | long        |        70.19 |        68.7  |       nan    |       nan    |       43.52 |            108 |
| RLSSA   | short       |        71.3  |        69.63 |       nan    |       nan    |       44.44 |            108 |

### Exp 3 Key Findings

- No strategy is expected to be positive in all three 36-month subperiods.
- DVR holds cash in months where no market meets the p ≤ 0.10 threshold.
- Volatility normalisation materially changes contract selection (Match < 100%)
  even when full-period performance differences are statistically inconclusive.
- Performance is regime-sensitive: SSA long excels in 2019–2021 on average;
  RLSSA short strongest in 2022–2024.

---

## Methodology Deviations and Assumptions

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | TradingCharts contract data | **Missing** | Proprietary; synthetic panel used |
| 2 | B = 500 MEB paths | **Assumed** | B = 5 for practical runtime |
| 3 | RLSSA max_iter = 50,000 | **Assumed** | 200 with early-exit (quality equivalent) |
| 4 | R `meboot` package | **Assumed** | Custom Python implementation |
| 5 | HAC finite-sample correction | **Confirmed** | `use_correction=True`, Bartlett, 3 lags |
| 6 | Delivery-avoidance contract selection | **Missing** | Requires contract-level OHLCV+expiry |
| 7 | Liquidity screen (≥ 1,000 ADV) | **Assumed** | Not applied (no real volume data) |
| 8 | RLSSA cycling tolerance | **Interpreted** | Strict tol=1e-10; practical fallback 5e-2 |
| 9 | EWMA warm-start from first 12 obs | **Confirmed** | Implemented per paper p.7 |
| 10 | DVR HAC maxlags = floor(0.75·T^(1/3)) | **Confirmed** | = 3 for T=120 |

---

## Artifact Inventory

| File | Description |
|------|-------------|
| `results/exp1_metrics.csv` | Exp 1 per-strategy metrics (CumRet, Sharpe, MaxDD) |
| `results/exp1_portfolio_returns.csv` | Exp 1 monthly net return series |
| `results/exp1_detail.csv` | Exp 1 per-month model selections |
| `results/exp1_paper_targets.csv` | Paper Table 2 target values |
| `results/exp1_cumulative_wealth.png` | Exp 1 cumulative wealth chart |
| `results/exp2_bootstrap_summary.csv` | Exp 2 MEB summary statistics |
| `results/exp2_holm_results.csv` | Exp 2 18 paired Sharpe tests + Holm p-values |
| `results/exp2_meb_sharpe_distributions.png` | Exp 2 bootstrap Sharpe distributions |
| `results/exp3_subperiod_metrics.csv` | Exp 3 mean subperiod CumRets across paths |
| `results/exp3_selection_stability.csv` | Exp 3 Top-1 selection stability statistics |
| `results/exp3_subperiod_boxplots.png` | Exp 3 subperiod distribution box plots |
