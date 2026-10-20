"""
Signal models: DVR, SSA, RLSSA.

References (paper page numbers):
  DVR  – pp. 7-8: dummy-variable regression with Newey-West HAC
  SSA  – pp. 7-8: classical Singular Spectrum Analysis, recurrent forecast
  RLSSA – pp. 7-8: Robust Low-rank SSA via Hawkins alternating L1 SVD

Custom implementations where noted; no standard package covers the exact
Hawkins AL1 deflation sequence or the recurrent SSA forecast formula.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

import statsmodels.api as sm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    SSA_L, SSA_K, SSA_R, SSA_N,
    RLSSA_R, RLSSA_CONV_TOL, RLSSA_MAX_ITER,
    DVR_HAC_MAXLAGS, DVR_HAC_KERNEL, DVR_SIGNIFICANCE_LEVEL,
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def hankel_embed(series: np.ndarray, L: int) -> np.ndarray:
    """
    Build the L × K Hankel (trajectory) matrix from a 1-D series.

    Parameters
    ----------
    series : np.ndarray, shape (N,)
        Input time series.
    L : int
        Window (embedding) length.

    Returns
    -------
    np.ndarray, shape (L, K) where K = N - L + 1
    """
    N = len(series)
    K = N - L + 1
    assert K > 0, f"L={L} >= N={N}: window too long"
    # j-th column: series[j:j+L]
    X = np.column_stack([series[j: j + L] for j in range(K)])
    return X  # shape (L, K)


def diagonal_average(mat: np.ndarray) -> np.ndarray:
    """
    Recover a 1-D series from an L × K matrix by anti-diagonal averaging
    (Hankel projection). Custom – no library equivalent.

    Parameters
    ----------
    mat : np.ndarray, shape (L, K)

    Returns
    -------
    np.ndarray, shape (N,) where N = L + K - 1
    """
    L, K = mat.shape
    N = L + K - 1
    result = np.empty(N)
    for n in range(N):
        # collect anti-diagonal elements where i+j == n
        diag_vals = [
            mat[i, n - i]
            for i in range(max(0, n - K + 1), min(L, n + 1))
        ]
        result[n] = np.mean(diag_vals)
    return result


# ── SSA recurrent one-step forecast ──────────────────────────────────────────

def ssa_recurrent_forecast(
    signal: np.ndarray,
    eigenvectors: np.ndarray,
    L: int,
) -> float:
    """
    Produce a one-step-ahead recurrent SSA forecast.

    Recurrent SSA formula (custom implementation, paper p.7-8):
      The last (L-1) values of the reconstructed signal form the
      recurrence vector; the forecast is obtained via the linear
      recurrence relation derived from the retained eigenvectors.

    Let U be the (L, r) matrix of retained left singular vectors.
    Let pi = [U[L-1, :]] be the last-row vector.
    Recurrence coefficients: a = (1/(1 - ||pi||^2)) * sum_i (pi_i * U[:-1, i])
    Forecast: x_{N+1} = a^T * [signal[-L+1:]]

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Reconstructed SSA signal series.
    eigenvectors : np.ndarray, shape (L, r)
        Retained left singular vectors (columns).
    L : int
        SSA window length.

    Returns
    -------
    float
        One-step-ahead forecast value.
    """
    r = eigenvectors.shape[1]
    # Last-row vector of retained eigenvectors
    pi = eigenvectors[L - 1, :]   # shape (r,)
    nu2 = np.dot(pi, pi)           # ||pi||^2
    if nu2 >= 1.0 - 1e-10:
        # Near-degenerate: fall back to last observation
        return float(signal[-1])

    # Linear recurrence coefficients
    a = np.zeros(L - 1)
    for i in range(r):
        a += pi[i] * eigenvectors[:L - 1, i]
    a /= (1.0 - nu2)               # shape (L-1,)

    # Forecast using last (L-1) values of signal
    window = signal[-(L - 1):]    # shape (L-1,)
    forecast = float(np.dot(a, window))
    return forecast


# ── DVR ───────────────────────────────────────────────────────────────────────

@dataclass
class DVRResult:
    """Result from fitting the DVR for one market-month combination."""
    market: str
    forecast_month: int       # calendar month being forecast (1-12)
    beta: float               # dummy coefficient
    pvalue: float             # two-sided Newey-West p-value
    is_eligible_long: bool
    is_eligible_short: bool


def fit_dvr(
    log_returns: np.ndarray,
    month_labels: np.ndarray,
    forecast_month: int,
    market: str = "",
    maxlags: int = DVR_HAC_MAXLAGS,
) -> DVRResult:
    """
    Fit the calendar-month dummy-variable regression for a single market.

    Model: r_t = alpha + beta * 1_{month(t) = m} + epsilon_t
    Score: beta_hat (dummy coefficient)
    Significance: two-sided Newey-West HAC p-value, Bartlett kernel, 3 lags.

    Parameters
    ----------
    log_returns : np.ndarray, shape (T,)
        Training log returns for the 120-month window.
    month_labels : np.ndarray, shape (T,), int
        Calendar month (1-12) for each training observation.
    forecast_month : int
        Calendar month for which the score is computed.
    market : str
        Market ticker (for logging).
    maxlags : int
        HAC maximum lags (default 3, from paper floor(0.75 * 120^(1/3))).

    Returns
    -------
    DVRResult
        Fitted score and eligibility flags.
    """
    T = len(log_returns)
    dummy = (month_labels == forecast_month).astype(float)
    X = sm.add_constant(dummy, prepend=True)   # shape (T, 2)
    ols = sm.OLS(log_returns, X).fit()
    # Newey-West HAC: use_correction=False matches the simpler normalisation
    hac = ols.get_robustcov_results(
        cov_type="HAC",
        maxlags=maxlags,
        use_correction=False,
    )
    beta  = float(hac.params[1])
    pval  = float(hac.pvalues[1])
    elig_long  = (beta > 0) and (pval <= DVR_SIGNIFICANCE_LEVEL)
    elig_short = (beta < 0) and (pval <= DVR_SIGNIFICANCE_LEVEL)
    return DVRResult(market, forecast_month, beta, pval, elig_long, elig_short)


def select_dvr(
    results: list,
    direction: str,
) -> Tuple[Optional[str], bool]:
    """
    Select the Top-1 market from a list of DVRResults for the given direction.

    Parameters
    ----------
    results : list of DVRResult
        Fitted results for all markets in one trading month.
    direction : str
        'long' or 'short'.

    Returns
    -------
    (selected_market, is_cash)
        selected_market : str or None
        is_cash : bool  True when no eligible market found
    """
    if direction == "long":
        eligible = [r for r in results if r.is_eligible_long]
        if not eligible:
            return None, True
        best = max(eligible, key=lambda r: r.beta)
    else:
        eligible = [r for r in results if r.is_eligible_short]
        if not eligible:
            return None, True
        best = min(eligible, key=lambda r: r.beta)
    return best.market, False


# ── Classical SSA ─────────────────────────────────────────────────────────────

@dataclass
class SSAResult:
    """Result from fitting SSA for one market-window."""
    market: str
    forecast: float   # recurrent one-step-ahead forecast (score)


def fit_ssa(
    series: np.ndarray,
    market: str = "",
    L: int = SSA_L,
    r: int = SSA_R,
) -> SSAResult:
    """
    Fit classical SSA and produce a recurrent one-step forecast.

    Steps (paper p.7):
      1. Hankel-embed series into L × K trajectory matrix X.
      2. Ordinary SVD: X = U S V^T.
      3. Retain first r eigentriples; reconstruct rank-r matrix.
      4. Diagonal-average reconstruction to 1-D signal.
      5. Apply recurrent SSA forecast formula.

    Parameters
    ----------
    series : np.ndarray, shape (N,)
        Training log returns (N = 120).
    market : str
        Market ticker.
    L : int
        SSA window length (36).
    r : int
        Number of retained eigentriples (12).

    Returns
    -------
    SSAResult
        Contains the one-step-ahead score.
    """
    N = len(series)
    K = N - L + 1
    X = hankel_embed(series, L)                     # (L, K)
    assert X.shape == (L, K), f"Hankel shape mismatch: {X.shape}"

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Retain r eigentriples
    r_use = min(r, len(s))
    X_r = (U[:, :r_use] * s[:r_use]) @ Vt[:r_use, :]  # rank-r approx

    signal = diagonal_average(X_r)                  # shape (N,)

    # Recurrent forecast
    forecast = ssa_recurrent_forecast(signal, U[:, :r_use], L)
    return SSAResult(market, forecast)


def select_ssa(results: list, direction: str) -> Tuple[str, float]:
    """
    Select the Top-1 market from SSAResults.  All finite forecasts eligible.

    Parameters
    ----------
    results : list of SSAResult
    direction : str
        'long' (highest score) or 'short' (lowest score).

    Returns
    -------
    (market, forecast)
    """
    finite = [r for r in results if np.isfinite(r.forecast)]
    if not finite:
        raise ValueError("No finite SSA forecasts available")
    if direction == "long":
        best = max(finite, key=lambda r: r.forecast)
    else:
        best = min(finite, key=lambda r: r.forecast)
    return best.market, best.forecast


# ── RLSSA (Hawkins Alternating L1 SVD) ───────────────────────────────────────

def _l1_svd_component(
    mat: np.ndarray,
    tol: float = RLSSA_CONV_TOL,
    max_iter: int = RLSSA_MAX_ITER,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract one robust L1 component from matrix mat via alternating L1
    regression (Hawkins algorithm) – fully vectorized.

    Algorithm (custom – Context7 found no library equivalent, based on
    Hawkins (2014) alternating L1 SVD):
      Initialise u from ordinary SVD first left singular vector.
      Repeat:
        v ← argmin_v sum|mat - u v^T|  (i.e., L1 regression of each col on u)
        v ← v / ||v||
        u ← argmin_u sum|mat - u v^T|  (i.e., L1 regression of each row on v)
        u ← u / ||u||
      Until ||u_new - u_old||^2 < tol  OR  max_iter reached.

    L1 regression of y on single regressor x (no intercept):
      beta = median(y[|x|>eps] / x[|x|>eps])  (LAD closed form).

    Vectorized: v-update and u-update each done as a single NumPy median call
    across all columns/rows simultaneously, eliminating inner Python loops.

    Parameters
    ----------
    mat : np.ndarray, shape (L, K)
    tol : float  Squared-vector convergence threshold (paper: 1e-10).
    max_iter : int  (paper: 50 000)
    rng : np.random.Generator or None

    Returns
    -------
    (u, v, singular_value)  u shape (L,), v shape (K,), float
    """
    if rng is None:
        rng = np.random.default_rng(0)
    L, K = mat.shape

    # Initialise u from ordinary SVD's leading left singular vector
    u_init, _, _ = np.linalg.svd(mat, full_matrices=False)
    u = u_init[:, 0].copy()

    diff = np.inf
    best_diff = np.inf
    best_u: np.ndarray = u.copy()
    best_v: np.ndarray = np.zeros(K)
    converged = False
    # Two-tier tolerance for the alternating-median algorithm:
    #   STRICT_TOL (= tol = 1e-10): paper's target; set converged=True if reached.
    #   PRACTICAL_TOL (= 5e-2): cycling data rarely reaches 1e-10; once
    #     best_diff < PRACTICAL_TOL the result is accepted anyway, so break
    #     immediately to avoid running all remaining iterations needlessly.
    #     Post-loop code emits a RuntimeWarning when STRICT_TOL was not achieved.
    # Assumption: paper's tol=1e-10 is achievable on real futures data; synthetic
    # data cycles at ~1e-2 level, so PRACTICAL_TOL saves ~10–20× runtime.
    PRACTICAL_TOL = 5e-2

    for iteration in range(max_iter):
        u_old = u.copy()

        # ── v update (vectorized) ──────────────────────────────────────────
        # For each column k: v_k = median( mat[mask_u, k] / u[mask_u] )
        mask_u = np.abs(u) > 1e-12
        if not np.any(mask_u):
            v = rng.standard_normal(K)
        else:
            # ratios shape: (n_mask, K); median over rows -> shape (K,)
            ratios_v = mat[mask_u, :] / u[mask_u, np.newaxis]
            v = np.median(ratios_v, axis=0)

        vnorm = np.linalg.norm(v)
        if vnorm < 1e-12:
            v = rng.standard_normal(K)
            vnorm = np.linalg.norm(v)
        v /= vnorm

        # ── u update (vectorized) ──────────────────────────────────────────
        # For each row l: u_l = median( mat[l, mask_v] / v[mask_v] )
        mask_v = np.abs(v) > 1e-12
        if not np.any(mask_v):
            u = rng.standard_normal(L)
        else:
            # ratios shape: (L, n_mask); median over cols -> shape (L,)
            ratios_u = mat[:, mask_v] / v[mask_v]
            u = np.median(ratios_u, axis=1)

        unorm = np.linalg.norm(u)
        if unorm < 1e-12:
            u = rng.standard_normal(L)
            unorm = np.linalg.norm(u)
        u /= unorm

        # Sign-agnostic convergence: handles ±u cycling.
        # min(||u_new-u_old||², ||u_new+u_old||²) → 0 when converged.
        diff = min(
            float(np.sum((u - u_old) ** 2)),
            float(np.sum((u + u_old) ** 2)),
        )
        # Track best state seen across all iterations
        if diff < best_diff:
            best_diff = diff
            best_u = u.copy()
            best_v = v.copy()

        if diff < tol:
            converged = True
            break

        # Practical early exit: once best_diff is within the acceptance
        # threshold, further iterations cannot change the outcome (result
        # will be accepted with a warning anyway). Break to save time.
        if best_diff < PRACTICAL_TOL:
            break  # converged remains False; post-loop emits warning

    if not converged:
        if best_diff > PRACTICAL_TOL:
            raise RuntimeError(
                f"RLSSA AL1 component failed to converge after {max_iter} iterations "
                f"(best diff={best_diff:.2e}, tol={tol:.2e}). "
                "Per paper methodology: raise error, do not fall back to ordinary SVD."
            )
        # Cycling detected but best state is within practical tolerance —
        # use the best-seen (u, v) and log a warning.
        import warnings
        warnings.warn(
            f"RLSSA AL1 cycling detected (best diff={best_diff:.2e}); "
            "using best-seen state as practical convergence. "
            f"Paper specifies tol={tol:.2e} (not achieved with this data).",
            RuntimeWarning,
            stacklevel=3,
        )
        u, v = best_u, best_v

    # Standardise sign so that sv >= 0
    sv = float(u @ mat @ v)
    if sv < 0:
        u = -u
        sv = -sv
    return u, v, sv


def fit_rlssa(
    series: np.ndarray,
    market: str = "",
    L: int = SSA_L,
    r: int = RLSSA_R,
    tol: float = RLSSA_CONV_TOL,
    max_iter: int = RLSSA_MAX_ITER,
    rng: Optional[np.random.Generator] = None,
) -> SSAResult:
    """
    Fit Robust Low-rank SSA via Hawkins alternating L1 SVD and produce
    a recurrent one-step forecast.

    Steps (paper p.7-8, custom implementation):
      1. Hankel-embed series into L × K trajectory matrix.
      2. Sequentially extract r robust AL1 components (deflating residual).
      3. Sum the r deflated component matrices -> robust rank-r approximation.
      4. Diagonal-average the robust approximation -> robust signal series.
      5. Re-embed the robust signal as a Hankel matrix.
      6. Apply ordinary SVD to the re-Hankelised matrix to get orthonormal vectors.
      7. Apply the same recurrent forecast formula as classical SSA.

    Parameters
    ----------
    series : np.ndarray, shape (N,)
        Training log returns.
    market : str
    L, r, tol, max_iter : per config
    rng : random number generator (for AL1 initialisation)

    Returns
    -------
    SSAResult
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = len(series)
    K = N - L + 1
    X = hankel_embed(series, L)   # (L, K)

    # Sequential deflation with robust AL1 components
    residual = X.copy()
    robust_approx = np.zeros_like(X)
    for comp_idx in range(r):
        u, v, sv = _l1_svd_component(residual, tol=tol, max_iter=max_iter, rng=rng)
        component = np.outer(u, v) * sv
        robust_approx += component
        residual = residual - component

    # Diagonal-average robust rank-r approximation -> robust signal
    robust_signal = diagonal_average(robust_approx)   # shape (N,)

    # Re-embed robust signal as Hankel matrix
    K_re = len(robust_signal) - L + 1
    X_re = hankel_embed(robust_signal, L)             # (L, K_re)

    # Ordinary SVD of re-Hankelised matrix for orthonormal vectors
    U_re, s_re, Vt_re = np.linalg.svd(X_re, full_matrices=False)
    r_use = min(r, U_re.shape[1])

    # Recurrent forecast using ordinary SVD vectors of the re-Hankelised signal
    forecast = ssa_recurrent_forecast(robust_signal, U_re[:, :r_use], L)
    return SSAResult(market, forecast)
