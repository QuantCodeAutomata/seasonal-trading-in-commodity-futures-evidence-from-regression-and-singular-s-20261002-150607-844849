"""
Portfolio construction and performance metrics.

Methodology (paper pp. 9-10):
  - Top-1 monthly position, unit notional, reset each month.
  - One-way transaction cost c = 0.00086 applied twice per month.
  - Long  net return: (1 + R)(1 - c)^2 - 1
  - Short net return: (1 - R)(1 - c)^2 - 1   [linear futures exposure]
  - Benchmark: equal-weight long across all 15 markets, same cost rule.
  - DVR cash months: net return = 0, no cost.
  - Metrics (paper p.10):
      CumRet = prod(1 + R_t) - 1
      Sharpe = sqrt(12) * mean(R_t) / std(R_t)   [zero risk-free rate]
      MaxDD from wealth starting at 1 before first return.
"""

import numpy as np
import pandas as pd
from typing import Sequence, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ONE_WAY_COST, ANNUALISATION_FACTOR


# ── Return calculation ────────────────────────────────────────────────────────

def net_return_long(simple_return: float, c: float = ONE_WAY_COST) -> float:
    """Net long return after round-trip cost. Paper p.9."""
    return (1 + simple_return) * (1 - c) ** 2 - 1


def net_return_short(simple_return: float, c: float = ONE_WAY_COST) -> float:
    """Net short return (linear futures exposure). Paper p.9."""
    return (1 - simple_return) * (1 - c) ** 2 - 1


def net_return_cash() -> float:
    """DVR cash month: zero return, zero cost. Paper p.7."""
    return 0.0


def benchmark_net_return(
    simple_returns: Sequence[float],
    c: float = ONE_WAY_COST,
) -> float:
    """
    Equal-weight long benchmark net return for one month. Paper p.9.

    Gross return = arithmetic mean of market simple returns.
    Net return = (1 + gross)(1 - c)^2 - 1.
    """
    gross = float(np.nanmean(simple_returns))
    return (1 + gross) * (1 - c) ** 2 - 1


# ── Performance metrics ────────────────────────────────────────────────────────

def cumulative_return(net_returns: np.ndarray) -> float:
    """
    Compounded cumulative return. Paper p.10.
    CumRet = prod(1 + R_t) - 1
    """
    return float(np.prod(1 + net_returns) - 1)


def annualised_sharpe(net_returns: np.ndarray) -> float:
    """
    Annualised Sharpe ratio. Paper p.10.
    Sharpe = sqrt(12) * mean(R_t) / std(R_t)   [zero risk-free rate]
    Uses sample standard deviation (ddof=1).
    """
    if len(net_returns) < 2:
        return np.nan
    std = float(np.std(net_returns, ddof=1))
    if std < 1e-12:
        return np.nan
    return float(np.sqrt(ANNUALISATION_FACTOR) * np.mean(net_returns) / std)


def maximum_drawdown(net_returns: np.ndarray) -> float:
    """
    Maximum drawdown from wealth index starting at 1. Paper p.10.
    wealth_t  = prod_{u<=t}(1 + R_u),   with wealth_0 = 1 (before first return)
    peak_t    = max_{u<=t}(wealth_u)
    drawdown_t = wealth_t / peak_t - 1
    MaxDD = min_t(drawdown_t)
    """
    # Prepend 1.0 as starting wealth
    wealth = np.concatenate([[1.0], np.cumprod(1 + net_returns)])
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1
    return float(np.min(drawdowns))


def compute_metrics(net_returns: np.ndarray) -> dict:
    """
    Compute all three paper metrics: CumRet, Sharpe, MaxDD.

    Parameters
    ----------
    net_returns : np.ndarray
        Monthly net portfolio returns.

    Returns
    -------
    dict with keys 'CumRet', 'Sharpe', 'MaxDD'.
    """
    return {
        "CumRet": cumulative_return(net_returns),
        "Sharpe": annualised_sharpe(net_returns),
        "MaxDD":  maximum_drawdown(net_returns),
    }


def subperiod_metrics(net_returns: pd.Series, start: str, end: str) -> dict:
    """
    Compute metrics over a subperiod slice. Paper p.10 (exp_3).

    Subperiod wealth index is reset to 1 at the start of each subperiod.

    Parameters
    ----------
    net_returns : pd.Series
        Monthly net returns indexed by DatetimeIndex (month-end).
    start, end : str
        'YYYY-MM' strings.

    Returns
    -------
    dict with keys 'CumRet', 'Sharpe', 'MaxDD'.
    """
    mask = (net_returns.index >= pd.Period(start, "M").to_timestamp("M")) & \
           (net_returns.index <= pd.Period(end, "M").to_timestamp("M"))
    sub = net_returns[mask].values
    if len(sub) == 0:
        return {"CumRet": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    return compute_metrics(sub)
