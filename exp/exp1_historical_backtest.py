"""
Experiment 1: Historical rolling out-of-sample comparison of DVR, SSA,
and RLSSA commodity-futures strategies.

Reproduces paper Table 2 (pp.13-14):
  - 12 model portfolios: DVR/SSA/RLSSA × long/short × classical/normalised
  - Equal-weight long benchmark
  - Evaluation: Jan 2016 – Dec 2024 (108 months)

Expected results (paper Table 2, p.13):
  DVR_long_classical:    CumRet=0.2966, Sharpe=0.245, MaxDD=-0.308
  DVR_long_normalised:   CumRet=0.2780, Sharpe=0.240, MaxDD=-0.366
  DVR_short_classical:   CumRet=-0.1994, Sharpe=0.018, MaxDD=-0.494
  DVR_short_normalised:  CumRet=-0.0972, Sharpe=0.028, MaxDD=-0.427
  SSA_long_classical:    CumRet=-0.3241, Sharpe=0.014, MaxDD=-0.605
  SSA_long_normalised:   CumRet=-0.5776, Sharpe=-0.112, MaxDD=-0.812
  SSA_short_classical:   CumRet=-0.3583, Sharpe=0.119, MaxDD=-0.799
  SSA_short_normalised:  CumRet=-0.7548, Sharpe=-0.205, MaxDD=-0.867
  RLSSA_long_classical:  CumRet=-0.5764, Sharpe=-0.088, MaxDD=-0.721
  RLSSA_long_normalised: CumRet=-0.7369, Sharpe=-0.365, MaxDD=-0.836
  RLSSA_short_classical: CumRet=-0.0054, Sharpe=0.184, MaxDD=-0.825
  RLSSA_short_normalised:CumRet=-0.8125, Sharpe=-0.288, MaxDD=-0.917
  BENCH:                 CumRet=0.1430, Sharpe=0.183, MaxDD=-0.415

NOTE: These are targets from the paper's proprietary data. Synthetic-data
results will differ; methodology implementation is faithful.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import json
from pathlib import Path

from config import UNIVERSE, TRADING_START, TRADING_END, WINDOW_MONTHS
from data.generate_panel import load_panel
from src.backtest import run_backtest
from src.portfolio import compute_metrics

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp1(panel: pd.DataFrame = None, verbose: bool = True) -> dict:
    """
    Run Experiment 1: full historical backtest over Jan 2016 – Dec 2024.

    Parameters
    ----------
    panel : pd.DataFrame or None
        Monthly simple return panel. If None, loads (or generates) the panel.
    verbose : bool

    Returns
    -------
    dict with keys 'metrics_table', 'portfolio_returns', 'detail'
    """
    if panel is None:
        panel = load_panel()

    if verbose:
        print(f"Panel: {panel.shape[0]} months × {panel.shape[1]} markets")
        print(f"Date range: {panel.index[0].strftime('%Y-%m')} to {panel.index[-1].strftime('%Y-%m')}")
        print("Running backtest (this may take several minutes for RLSSA)...")

    rlssa_rng = np.random.default_rng(0)

    portfolio_returns, detail = run_backtest(
        simple_returns=panel,
        models=["DVR", "SSA", "RLSSA"],
        directions=["long", "short"],
        specs=["classical", "normalised"],
        window=WINDOW_MONTHS,
        verbose=verbose,
        rlssa_rng=rlssa_rng,
    )

    # ── Compute metrics for all strategies ────────────────────────────────
    metrics_rows = []
    for col in portfolio_returns.columns:
        ret = portfolio_returns[col].dropna().values
        if len(ret) == 0:
            continue
        m = compute_metrics(ret)
        m["Strategy"] = col
        metrics_rows.append(m)

    metrics_table = pd.DataFrame(metrics_rows).set_index("Strategy")

    # ── Paper targets for comparison ──────────────────────────────────────
    paper_targets = {
        "DVR_long_classical":     {"CumRet": 0.2966, "Sharpe": 0.245,  "MaxDD": -0.308},
        "DVR_long_normalised":    {"CumRet": 0.2780, "Sharpe": 0.240,  "MaxDD": -0.366},
        "DVR_short_classical":    {"CumRet":-0.1994, "Sharpe": 0.018,  "MaxDD": -0.494},
        "DVR_short_normalised":   {"CumRet":-0.0972, "Sharpe": 0.028,  "MaxDD": -0.427},
        "SSA_long_classical":     {"CumRet":-0.3241, "Sharpe": 0.014,  "MaxDD": -0.605},
        "SSA_long_normalised":    {"CumRet":-0.5776, "Sharpe":-0.112,  "MaxDD": -0.812},
        "SSA_short_classical":    {"CumRet":-0.3583, "Sharpe": 0.119,  "MaxDD": -0.799},
        "SSA_short_normalised":   {"CumRet":-0.7548, "Sharpe":-0.205,  "MaxDD": -0.867},
        "RLSSA_long_classical":   {"CumRet":-0.5764, "Sharpe":-0.088,  "MaxDD": -0.721},
        "RLSSA_long_normalised":  {"CumRet":-0.7369, "Sharpe":-0.365,  "MaxDD": -0.836},
        "RLSSA_short_classical":  {"CumRet":-0.0054, "Sharpe": 0.184,  "MaxDD": -0.825},
        "RLSSA_short_normalised": {"CumRet":-0.8125, "Sharpe":-0.288,  "MaxDD": -0.917},
        "BENCH":                  {"CumRet": 0.1430, "Sharpe": 0.183,  "MaxDD": -0.415},
    }
    paper_df = pd.DataFrame(paper_targets).T
    paper_df.index.name = "Strategy"

    # ── Print summary ──────────────────────────────────────────────────────
    if verbose:
        print("\n═══ Exp 1: Performance Metrics (Synthetic Data) ═══")
        print(metrics_table.round(4).to_string())
        print("\n─── Paper Targets (Table 2, p.13) ───")
        print(paper_df.round(4).to_string())

    # ── Save results ──────────────────────────────────────────────────────
    portfolio_returns.to_csv(RESULTS_DIR / "exp1_portfolio_returns.csv")
    metrics_table.round(4).to_csv(RESULTS_DIR / "exp1_metrics.csv")
    paper_df.round(4).to_csv(RESULTS_DIR / "exp1_paper_targets.csv")
    detail.to_csv(RESULTS_DIR / "exp1_detail.csv", index=False)

    # ── Cumulative wealth plot ─────────────────────────────────────────────
    _plot_cumulative_wealth(portfolio_returns, RESULTS_DIR / "exp1_cumulative_wealth.png")

    return {
        "metrics_table":     metrics_table,
        "paper_targets":     paper_df,
        "portfolio_returns": portfolio_returns,
        "detail":            detail,
    }


def _plot_cumulative_wealth(port_ret: pd.DataFrame, save_path: Path) -> None:
    """Plot cumulative wealth paths for all strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    model_groups = {
        "DVR":   ["DVR_long_classical", "DVR_long_normalised",
                  "DVR_short_classical", "DVR_short_normalised"],
        "SSA":   ["SSA_long_classical", "SSA_long_normalised",
                  "SSA_short_classical", "SSA_short_normalised"],
        "RLSSA": ["RLSSA_long_classical", "RLSSA_long_normalised",
                  "RLSSA_short_classical", "RLSSA_short_normalised"],
    }
    bench = port_ret["BENCH"].dropna()
    bench_wealth = (1 + bench).cumprod()

    colors = {"long_classical": "steelblue", "long_normalised": "dodgerblue",
              "short_classical": "firebrick", "short_normalised": "salmon"}

    ax_flat = axes.flatten()
    ax_idx = 0
    for model, cols in model_groups.items():
        ax = ax_flat[ax_idx]
        ax_idx += 1
        for col in cols:
            if col not in port_ret.columns:
                continue
            s = port_ret[col].dropna()
            w = (1 + s).cumprod()
            key = "_".join(col.split("_")[1:])
            ax.plot(w.index, w.values, label=key, color=colors.get(key, "gray"), linewidth=1.5)
        ax.plot(bench_wealth.index, bench_wealth.values, "k--", label="Benchmark", linewidth=1.0)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"{model} Strategy")
        ax.set_ylabel("Wealth (start = 1)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 4th panel: all long strategies
    ax = ax_flat[3]
    for model in ["DVR", "SSA", "RLSSA"]:
        col = f"{model}_long_classical"
        if col in port_ret.columns:
            s = port_ret[col].dropna()
            w = (1 + s).cumprod()
            ax.plot(w.index, w.values, label=f"{model} Long (classical)", linewidth=1.5)
    ax.plot(bench_wealth.index, bench_wealth.values, "k--", label="Benchmark", linewidth=1.0)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("All Models: Long Classical")
    ax.set_ylabel("Wealth (start = 1)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Exp 1: Cumulative Wealth — DVR, SSA, RLSSA Strategies (Synthetic Data)\n"
        "Jan 2016 – Dec 2024",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    results = run_exp1(verbose=True)
    print("\nExp 1 complete. Results saved to results/")
