"""
Core rolling-window backtest engine.

Implements the recursive out-of-sample backtest for DVR, SSA, and RLSSA
strategies on a monthly return panel.

Methodology (paper pp. 7-10, confirmed):
  - Rolling 120-month estimation window for each trading month.
  - Two specifications: classical (raw log returns) and vol-normalised.
  - Three models: DVR, SSA, RLSSA.
  - Two directions: long, short.
  - Top-1 selection per model-direction.
  - Equal-weight long benchmark.
  - Transaction costs c = 0.00086 one-way.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Dict, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    UNIVERSE, WINDOW_MONTHS, TRADING_START, TRADING_END,
    ONE_WAY_COST,
)
from src.volatility import ewma_normalise
from src.models import (
    fit_dvr, select_dvr, DVRResult,
    fit_ssa, select_ssa, SSAResult,
    fit_rlssa,
)
from src.portfolio import (
    net_return_long, net_return_short, net_return_cash,
    benchmark_net_return, compute_metrics,
)


# ── Rolling backtest runner ────────────────────────────────────────────────────

def run_backtest(
    simple_returns: pd.DataFrame,
    models: List[str] = ("DVR", "SSA", "RLSSA"),
    directions: List[str] = ("long", "short"),
    specs: List[str] = ("classical", "normalised"),
    window: int = WINDOW_MONTHS,
    verbose: bool = False,
    rlssa_rng=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full rolling out-of-sample backtest.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Monthly simple returns, DatetimeIndex (month-end), columns = markets.
        Must span at least ESTIMATION_START through TRADING_END.
    models : list of str
        Subset of ['DVR', 'SSA', 'RLSSA'].
    directions : list of str
        Subset of ['long', 'short'].
    specs : list of str
        Subset of ['classical', 'normalised'].
    window : int
        Rolling estimation window in months (120).
    verbose : bool
        Print progress.
    rlssa_rng : np.random.Generator or None
        RNG for RLSSA AL1 initialisation (ensures determinism).

    Returns
    -------
    (portfolio_returns, detail_records)
    portfolio_returns : pd.DataFrame
        Columns = strategy labels + 'BENCH', index = trading dates.
    detail_records : pd.DataFrame
        Row per (trading_date, model, direction, spec, market) with
        score, selection, realised return, net return, cash flag, etc.
    """
    if rlssa_rng is None:
        rlssa_rng = np.random.default_rng(0)

    markets = list(simple_returns.columns)
    trading_dates = pd.date_range(
        start=pd.Period(TRADING_START, "M").to_timestamp("M"),
        end=pd.Period(TRADING_END, "M").to_timestamp("M"),
        freq="ME",
    )
    # Filter to dates actually in the panel
    trading_dates = trading_dates[trading_dates.isin(simple_returns.index)]

    # Log-return panel for estimation
    log_returns = np.log1p(simple_returns)

    strategy_labels = [
        f"{model}_{direction}_{spec}"
        for model in models
        for direction in directions
        for spec in specs
    ]
    port_ret = pd.DataFrame(index=trading_dates, columns=strategy_labels + ["BENCH"], dtype=float)
    records = []

    for i, trade_date in enumerate(trading_dates):
        if verbose and i % 12 == 0:
            print(f"  {trade_date.strftime('%Y-%m')}...")

        # Locate the 120-month training window ending before this trading month
        trade_pos = simple_returns.index.get_loc(trade_date)
        win_start = trade_pos - window
        if win_start < 0:
            continue  # not enough history

        train_simple = simple_returns.iloc[win_start: trade_pos]  # (120, n)
        train_log    = log_returns.iloc[win_start: trade_pos]      # (120, n)
        month_labels = np.array([d.month for d in train_simple.index])
        forecast_month = trade_date.month

        # Realised simple returns for the trading month
        real_simple = simple_returns.loc[trade_date]

        # ── Benchmark ─────────────────────────────────────────────────────
        bench_nr = benchmark_net_return(real_simple.values)
        port_ret.loc[trade_date, "BENCH"] = bench_nr

        # ── Each model-direction-spec combination ─────────────────────────
        for model in models:
            for direction in directions:
                for spec in specs:
                    label = f"{model}_{direction}_{spec}"
                    try:
                        selected, net_ret, extra = _compute_strategy_return(
                            model=model,
                            direction=direction,
                            spec=spec,
                            train_log=train_log.values,
                            train_simple=train_simple.values,
                            month_labels=month_labels,
                            forecast_month=forecast_month,
                            markets=markets,
                            real_simple=real_simple.values,
                            rlssa_rng=rlssa_rng,
                        )
                    except Exception as exc:
                        warnings.warn(
                            f"[{trade_date.strftime('%Y-%m')} {label}] {exc}", RuntimeWarning
                        )
                        selected, net_ret, extra = None, np.nan, {}

                    port_ret.loc[trade_date, label] = net_ret
                    rec = {
                        "date":       trade_date,
                        "model":      model,
                        "direction":  direction,
                        "spec":       spec,
                        "selected":   selected,
                        "net_return": net_ret,
                        "real_simple": float(real_simple[selected]) if selected and selected in real_simple.index else np.nan,
                    }
                    rec.update(extra)
                    records.append(rec)

    detail = pd.DataFrame(records)
    return port_ret.astype(float), detail


def _compute_strategy_return(
    model: str,
    direction: str,
    spec: str,
    train_log: np.ndarray,
    train_simple: np.ndarray,
    month_labels: np.ndarray,
    forecast_month: int,
    markets: list,
    real_simple: np.ndarray,
    rlssa_rng,
) -> Tuple[Optional[str], float, dict]:
    """
    Compute the net return for one model-direction-spec in one trading month.

    Returns
    -------
    (selected_market, net_return, extra_info_dict)
    """
    extra: Dict = {"is_cash": False}

    # Apply vol normalisation if needed (per-market inside the training window)
    if spec == "normalised":
        est_series = np.apply_along_axis(ewma_normalise, 0, train_log)
    else:
        est_series = train_log  # (T, n_markets)

    n = len(markets)

    if model == "DVR":
        results = []
        for j, mkt in enumerate(markets):
            res = fit_dvr(
                log_returns=est_series[:, j],
                month_labels=month_labels,
                forecast_month=forecast_month,
                market=mkt,
            )
            results.append(res)
        selected, is_cash = select_dvr(results, direction)
        extra["dvr_betas"]  = {r.market: r.beta   for r in results}
        extra["dvr_pvals"]  = {r.market: r.pvalue  for r in results}
        extra["is_cash"]    = is_cash

        if is_cash:
            return None, net_return_cash(), extra

        j_sel = markets.index(selected)
        r_sel = float(real_simple[j_sel])
        nr = net_return_long(r_sel) if direction == "long" else net_return_short(r_sel)
        return selected, nr, extra

    elif model == "SSA":
        ssa_results = []
        for j, mkt in enumerate(markets):
            try:
                res = fit_ssa(est_series[:, j], market=mkt)
                ssa_results.append(res)
            except Exception:
                pass  # skip non-finite
        if not ssa_results:
            return None, np.nan, extra

        selected, score = select_ssa(ssa_results, direction)
        extra["ssa_score"] = score
        j_sel = markets.index(selected)
        r_sel = float(real_simple[j_sel])
        nr = net_return_long(r_sel) if direction == "long" else net_return_short(r_sel)
        return selected, nr, extra

    elif model == "RLSSA":
        rlssa_results = []
        for j, mkt in enumerate(markets):
            try:
                res = fit_rlssa(est_series[:, j], market=mkt, rng=rlssa_rng)
                rlssa_results.append(res)
            except RuntimeError as exc:
                # Paper: raise error, do not substitute; skip this market
                warnings.warn(str(exc), RuntimeWarning)
            except Exception:
                pass
        if not rlssa_results:
            return None, np.nan, extra

        selected, score = select_ssa(rlssa_results, direction)
        extra["rlssa_score"] = score
        j_sel = markets.index(selected)
        r_sel = float(real_simple[j_sel])
        nr = net_return_long(r_sel) if direction == "long" else net_return_short(r_sel)
        return selected, nr, extra

    else:
        raise ValueError(f"Unknown model: {model}")
