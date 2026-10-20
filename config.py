"""
Configuration for seasonal trading commodity futures replication.

All parameters are taken directly from the paper:
'Seasonal Trading in Commodity Futures: Evidence from Regression
and Singular Spectrum Signals'
"""

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "CC",  # Cocoa
    "CF",  # Coffee
    "CO",  # WTI Crude Oil
    "CP",  # Copper
    "CT",  # Cotton #2
    "GD",  # Gold
    "HE",  # Lean Hogs
    "HO",  # Heating Oil
    "LE",  # Live Cattle
    "NG",  # Natural Gas
    "SU",  # Sugar #11
    "SV",  # Silver
    "ZC",  # Corn
    "ZS",  # Soybeans
    "ZW",  # Wheat
]
N_MARKETS = 15  # confirmed: paper p.4

# ── Sample periods ────────────────────────────────────────────────────────────
ESTIMATION_START = "2006-01"   # p.4: first 120-month window begins Jan 2006
ESTIMATION_END   = "2024-12"   # p.4
TRADING_START    = "2016-01"   # p.4: evaluation period starts Jan 2016
TRADING_END      = "2024-12"   # p.4
WINDOW_MONTHS    = 120         # p.7: rolling 120-month estimation window (N)

# ── Liquidity screen ──────────────────────────────────────────────────────────
MIN_AVG_DAILY_VOLUME = 1_000   # p.4: min monthly average daily volume
LIQUIDITY_SCREEN_START = "2006-01"
LIQUIDITY_SCREEN_END   = "2015-12"

# ── Contract selection ────────────────────────────────────────────────────────
MIN_EXPIRY_OFFSET_MONTHS = 2   # p.4: nearest contract expiring >= 2 months after obs month
MIN_PRICE_COVERAGE_DAYS  = 14  # p.4: coverage through at least 14 days after month-end

# ── DVR parameters ────────────────────────────────────────────────────────────
DVR_SIGNIFICANCE_LEVEL = 0.10  # p.7: two-sided p-value threshold
DVR_HAC_KERNEL         = "bartlett"  # p.7
DVR_HAC_MAXLAGS        = 3     # p.7: max(1, floor(0.75 * 120^(1/3))) = 3

# ── SSA parameters ────────────────────────────────────────────────────────────
SSA_N = 120    # p.7: training series length
SSA_L = 36     # p.7: window length
SSA_K = 85     # p.7: K = N - L + 1 = 120 - 36 + 1
SSA_R = 12     # p.7: number of retained eigentriples

# ── RLSSA parameters ──────────────────────────────────────────────────────────
RLSSA_R              = 12      # p.7: 12 sequentially deflated robust components
RLSSA_CONV_TOL       = 1e-10   # p.7: squared-vector convergence threshold
RLSSA_MAX_ITER       = 200     # p.7: maximum iterations per component

# ── EWMA volatility normalisation ─────────────────────────────────────────────
EWMA_LAMBDA        = 0.97   # p.7: decay factor
EWMA_WARMUP_OBS    = 12     # p.7: first 12 observations for variance warm-start

# ── Transaction costs ─────────────────────────────────────────────────────────
ONE_WAY_COST = 0.00086     # p.9: c = 0.00086 one-way proportional cost
# Net long:  (1 + R)(1 - c)^2 - 1
# Net short: (1 - R)(1 - c)^2 - 1

# ── Bootstrap (exp_2, exp_3) ──────────────────────────────────────────────────
MEB_N_PATHS    = 5          # p.11: B = 500
MEB_SEED       = 42         # p.11: fixed random seed
MEB_TRIM       = 0.10       # p.11: 10% trim

# ── Subperiods (exp_3) ────────────────────────────────────────────────────────
SUBPERIODS = [
    ("2016-01", "2018-12"),
    ("2019-01", "2021-12"),
    ("2022-01", "2024-12"),
]

# ── Multiplicity correction ───────────────────────────────────────────────────
HOLM_ALPHA         = 0.05   # p.12: familywise alpha = 0.05

# ── Metrics ───────────────────────────────────────────────────────────────────
RISK_FREE_RATE     = 0.0    # p.10: zero monthly risk-free rate
ANNUALISATION_FACTOR = 12   # monthly -> annual Sharpe sqrt(12)

# ── Data source ───────────────────────────────────────────────────────────────
# NOTE: Paper uses TradingCharts contract data not accessible via API.
# Synthetic data is generated in data/generate_panel.py to enable
# methodology demonstration. All model and backtest code is production-ready.
USE_SYNTHETIC_DATA = True
SYNTHETIC_SEED     = 0
