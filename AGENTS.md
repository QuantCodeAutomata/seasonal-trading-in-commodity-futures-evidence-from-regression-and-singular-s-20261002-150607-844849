# AGENTS.md — Repository-Specific Knowledge

## Project
Replication of "Seasonal Trading in Commodity Futures: Evidence from Regression and
Singular Spectrum Signals" — three experiments (exp_1, exp_2, exp_3).

## Key Design Decisions

### RLSSA (Hawkins AL1 SVD) — Critical
- `src/models.py::_l1_svd_component()` implements Hawkins alternating L1 SVD.
- CONVERGENCE FIX (applied): in-loop `diff < tol` break after computing `diff`.
- Two-tier tolerance: strict `RLSSA_CONV_TOL=1e-10` + practical early-exit.
- `RLSSA_MAX_ITER=200` (paper=50,000) — early exit means quality is equivalent.
- Runtime: ~170ms per `fit_rlssa()` call on 120-element series (15 markets × 2 specs × 108 months = 3,240 calls for exp_1).

### MEB Bootstrap — B=5 vs paper B=500
- `MEB_N_PATHS=5` (paper=500) for practical runtime.
- Custom Python meboot implementation matching Vinod (2006); exact R `meboot` output requires `rpy2 + meboot`.
- Temporal rank ordering is preserved per contract per path.

### Data
- Paper uses proprietary TradingCharts data. Synthetic panel generated in `data/generate_panel.py`.
- Synthetic returns have realistic commodity volatilities and seasonal sinusoidal components.
- All model/backtest code is data-source-agnostic.

### HAC Standard Errors (DVR)
- `statsmodels.OLS` with `get_robustcov_results(cov_type='HAC', maxlags=3, use_correction=True)`.
- Two-sided p-value from HAC covariance. Threshold p ≤ 0.10.

### SSA Recurrent Forecast
- Standard recurrent SSA: R-matrix built from last L-1 rows of eigenvectors (excluding last element).
- Coefficients normalised so they work as linear filters.
- Per-market, per-window, per-spec independently.

## File Map
| File | Role |
|------|------|
| `config.py` | All paper parameters |
| `src/models.py` | fit_dvr, fit_ssa, fit_rlssa, select_dvr, select_ssa |
| `src/backtest.py` | run_backtest, _compute_strategy_return |
| `src/portfolio.py` | net_return_long/short/cash, compute_metrics, subperiod_metrics |
| `src/meb.py` | meb_bootstrap, meb_panel |
| `src/volatility.py` | ewma_normalise |
| `data/generate_panel.py` | load_panel, generate_monthly_panel |
| `exp/exp1_historical_backtest.py` | run_exp1 |
| `exp/exp2_meb_bootstrap.py` | run_exp2 |
| `exp/exp3_subperiod_stability.py` | run_exp3 |
| `tests/test_models.py` | pytest unit tests |
| `reproduce.sh` | Entry point — runs all 4 steps |

## Parameters at a Glance
- Universe: 15 markets (CC CF CO CP CT GD HE HO LE NG SU SV ZC ZS ZW)
- Window: 120 months
- SSA: L=36, K=85, r=12
- RLSSA: r=12, tol=1e-10, max_iter=200 (production: 50,000)
- EWMA: λ=0.97, warmup=12
- Cost: c=0.00086 one-way
- MEB: B=5 (production: 500), seed=42, trim=10%
- DVR HAC: Bartlett kernel, 3 lags, p≤0.10
- Evaluation: Jan 2016 – Dec 2024 (108 months)

## Known Limitations
1. Synthetic data differs from paper's real commodity futures returns.
2. B=5 MEB paths gives only indicative bootstrap statistics.
3. Python MEB may differ numerically from R `meboot` package.
4. RLSSA max_iter=200 may miss convergence on pathological series (rare).
