"""
Synthetic monthly commodity futures return panel generator.

NOTE: The paper uses proprietary TradingCharts futures contract data with
specific delivery-avoidance and liquidity-screen rules that require access
to individual contract histories (OHLCV + expiry dates). This synthetic
generator produces plausible commodity futures monthly simple returns
for methodology demonstration.

Return characteristics are modelled as:
  - Monthly drift ~0% (commodities have near-zero long-run real drift)
  - Monthly volatility 4–9% (varying by commodity)
  - Mild seasonal component added via sinusoidal terms
  - Cross-commodity correlation ~0.2–0.4

All model code (DVR, SSA, RLSSA, portfolio, MEB) is independent of the
data source and will work identically on real data if supplied.
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    UNIVERSE, ESTIMATION_START, ESTIMATION_END, SYNTHETIC_SEED
)


# Commodity-specific volatility proxies (annualised, rough empirical values)
_VOL_MAP = {
    "CC": 0.28,  # Cocoa
    "CF": 0.30,  # Coffee
    "CO": 0.35,  # WTI Crude Oil
    "CP": 0.22,  # Copper
    "CT": 0.25,  # Cotton
    "GD": 0.15,  # Gold
    "HE": 0.30,  # Lean Hogs
    "HO": 0.35,  # Heating Oil
    "LE": 0.14,  # Live Cattle
    "NG": 0.55,  # Natural Gas
    "SU": 0.30,  # Sugar
    "SV": 0.28,  # Silver
    "ZC": 0.25,  # Corn
    "ZS": 0.22,  # Soybeans
    "ZW": 0.28,  # Wheat
}

# Seasonal amplitude (fraction of monthly vol) and phase (peak month 1-12)
_SEASONAL_MAP = {
    "CC": (0.3, 10),   # Cocoa peaks Oct (harvest)
    "CF": (0.25, 5),   # Coffee peaks May
    "CO": (0.2, 1),    # Crude peaks Jan (heating demand)
    "CP": (0.2, 3),    # Copper peaks Mar (construction season)
    "CT": (0.25, 8),   # Cotton peaks Aug (US harvest)
    "GD": (0.15, 9),   # Gold peaks Sep
    "HE": (0.3, 6),    # Lean Hogs peaks Jun (summer grilling)
    "HO": (0.25, 1),   # Heating Oil peaks Jan
    "LE": (0.2, 4),    # Live Cattle peaks Apr
    "NG": (0.35, 1),   # Natural Gas peaks Jan
    "SU": (0.3, 10),   # Sugar peaks Oct
    "SV": (0.15, 9),   # Silver peaks Sep
    "ZC": (0.3, 7),    # Corn peaks Jul (growing season)
    "ZS": (0.25, 7),   # Soybeans peaks Jul
    "ZW": (0.25, 5),   # Wheat peaks May (US harvest approach)
}


def generate_monthly_panel(
    markets: list = UNIVERSE,
    start: str = ESTIMATION_START,
    end: str = ESTIMATION_END,
    seed: int = SYNTHETIC_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic monthly simple returns for commodity futures.

    Parameters
    ----------
    markets : list of str
        Market ticker symbols.
    start : str
        First month in 'YYYY-MM' format.
    end : str
        Last month in 'YYYY-MM' format.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Monthly simple returns with DatetimeIndex (month-end) and market
        tickers as columns. Missing values are NaN.
    """
    rng = np.random.default_rng(seed)
    dates = pd.period_range(start=start, end=end, freq="M").to_timestamp("M")
    T = len(dates)
    n = len(markets)

    # Build correlated innovations
    corr = np.full((n, n), 0.25)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((T, n)) @ L.T  # shape (T, n)

    returns = np.empty((T, n))
    for j, mkt in enumerate(markets):
        ann_vol = _VOL_MAP.get(mkt, 0.25)
        monthly_vol = ann_vol / np.sqrt(12)
        amp, peak_month = _SEASONAL_MAP.get(mkt, (0.2, 6))

        # Monthly drift ~ 0 for commodities
        mu = 0.0

        # Add seasonal component: sinusoidal with market-specific phase
        month_nums = np.array([d.month for d in dates])
        season = amp * monthly_vol * np.sin(
            2 * np.pi * (month_nums - peak_month) / 12
        )

        returns[:, j] = mu + season + monthly_vol * z[:, j]

    df = pd.DataFrame(returns, index=dates, columns=markets)
    return df


def load_panel(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load or generate the monthly return panel.

    Parameters
    ----------
    path : str or None
        Path to a CSV file with a 'date' column and market-ticker columns.
        If None, the synthetic panel is generated.

    Returns
    -------
    pd.DataFrame
        Monthly simple returns indexed by month-end timestamps.
    """
    if path is not None and os.path.exists(path):
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    return generate_monthly_panel()


if __name__ == "__main__":
    panel = generate_monthly_panel()
    print(f"Panel shape: {panel.shape}")
    print(panel.head())
    print(f"\nDate range: {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"\nDescriptive stats (monthly returns):")
    print(panel.describe().round(4))
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
