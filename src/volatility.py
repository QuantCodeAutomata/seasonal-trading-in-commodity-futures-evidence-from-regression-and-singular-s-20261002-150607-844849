"""
EWMA volatility normalisation as described in the paper (p.7).

Methodology (confirmed, paper p.7):
  lambda = 0.97
  Warm-start: sigma_1^2 = mean(r_1^2, ..., r_12^2)
  Update from tau=2: sigma_tau^2 = 0.97*sigma_{tau-1}^2 + 0.03*r_{tau-1}^2
  Standardised: z_tau = (r_tau - expanding_mean_tau) / sigma_tau

The transformation is applied only within the current training window.
"""

import numpy as np
from typing import Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EWMA_LAMBDA, EWMA_WARMUP_OBS


def ewma_normalise(
    series: np.ndarray,
    lam: float = EWMA_LAMBDA,
    warmup: int = EWMA_WARMUP_OBS,
) -> np.ndarray:
    """
    Apply EWMA volatility normalisation to a training window.

    Parameters
    ----------
    series : np.ndarray, shape (T,)
        Raw monthly log returns for the current training window.
    lam : float
        EWMA decay parameter (lambda = 0.97).
    warmup : int
        Number of observations used to warm-start the variance estimate.

    Returns
    -------
    np.ndarray, shape (T,)
        Volatility-standardised returns with expanding-mean subtracted.

    Notes
    -----
    Paper p.7: sigma_1^2 is initialised from mean squared return of the
    first `warmup` observations. From tau=2 onward:
        sigma_tau^2 = lam * sigma_{tau-1}^2 + (1-lam) * r_{tau-1}^2
    Expanding mean uses all window observations through tau.
    """
    T = len(series)
    assert T >= warmup, f"Series length {T} < warmup {warmup}"

    sigma2 = np.empty(T)
    # Warm-start variance from first `warmup` squared returns
    sigma2[0] = np.mean(series[:warmup] ** 2)

    # Propagate EWMA variance using lagged raw return
    for tau in range(1, T):
        sigma2[tau] = lam * sigma2[tau - 1] + (1 - lam) * series[tau - 1] ** 2

    sigma = np.sqrt(np.maximum(sigma2, 1e-12))  # guard against zero variance

    # Expanding mean: mean of series[0..tau] for each tau
    cumsum = np.cumsum(series)
    expanding_mean = cumsum / np.arange(1, T + 1)

    standardised = (series - expanding_mean) / sigma
    return standardised


def compute_ewma_vol(
    series: np.ndarray,
    lam: float = EWMA_LAMBDA,
    warmup: int = EWMA_WARMUP_OBS,
) -> np.ndarray:
    """
    Return the EWMA conditional volatility series (sigma_tau) for a window.

    Parameters
    ----------
    series : np.ndarray, shape (T,)
        Raw log returns.
    lam : float
        EWMA decay parameter.
    warmup : int
        Warm-start observations.

    Returns
    -------
    np.ndarray, shape (T,)
        Conditional standard deviations.
    """
    T = len(series)
    sigma2 = np.empty(T)
    sigma2[0] = np.mean(series[:warmup] ** 2)
    for tau in range(1, T):
        sigma2[tau] = lam * sigma2[tau - 1] + (1 - lam) * series[tau - 1] ** 2
    return np.sqrt(np.maximum(sigma2, 1e-12))
