"""
Experiment 3: Subperiod robustness and Top-1 contract-selection stability
across MEB paths.

Methodology (paper pp.15-16):
  - Three non-overlapping 36-month subperiods: 2016-18, 2019-21, 2022-24.
  - For each MEB path, compute subperiod CumRet, Sharpe, MaxDD.
  - Average across paths.
  - Tabulate leading-ticker frequency (C1/R1), cash frequency, match rate.

Expected targets (paper p.15):
  DVR long classical subperiod means: 0.0816, -0.1489, 0.0966
  SSA long classical subperiod means: -0.5548, 1.0959, 0.1906
  Benchmark subperiod means: -0.1355, 0.1922, 0.1251

NOTE: Synthetic data will not reproduce paper values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List
from collections import Counter

from config import (
    MEB_N_PATHS, MEB_SEED, MEB_TRIM, SUBPERIODS, WINDOW_MONTHS,
)
from data.generate_panel import load_panel
from src.meb import meb_panel
from src.backtest import run_backtest
from src.portfolio import subperiod_metrics

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp3(
    panel: pd.DataFrame = None,
    n_paths: int = MEB_N_PATHS,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment 3: subperiod robustness and ticker-selection stability.

    Parameters
    ----------
    panel : pd.DataFrame or None
    n_paths : int
    verbose : bool

    Returns
    -------
    dict with 'subperiod_summary', 'selection_stability'
    """
    if panel is None:
        panel = load_panel()

    if verbose:
        print(f"Generating {n_paths} MEB panels for Exp 3...")

    # ── MEB panels ─────────────────────────────────────────────────────────
    synthetic_panels = meb_panel(panel, B=n_paths, trim=MEB_TRIM, seed=MEB_SEED)

    # ── Storage ────────────────────────────────────────────────────────────
    strategies = [
        f"{m}_{d}_{s}"
        for m in ["DVR", "SSA", "RLSSA"]
        for d in ["long", "short"]
        for s in ["classical", "normalised"]
    ] + ["BENCH"]

    # For subperiod metrics: strategy -> subperiod -> list of values
    sub_data: Dict[str, Dict[str, list]] = {
        strat: {sp: [] for sp in ["2016-2018", "2019-2021", "2022-2024"]}
        for strat in strategies
    }

    # For ticker selection: (model, direction, spec) -> month -> list of tickers
    selection_records: List[dict] = []

    for b, syn_panel in enumerate(synthetic_panels):
        if verbose and b % 50 == 0:
            print(f"  Exp3 path {b+1}/{n_paths}...")
        try:
            port_ret, detail = run_backtest(
                simple_returns=syn_panel,
                models=["DVR", "SSA", "RLSSA"],
                directions=["long", "short"],
                specs=["classical", "normalised"],
                window=WINDOW_MONTHS,
                verbose=False,
                rlssa_rng=np.random.default_rng(b),
            )
        except Exception as exc:
            warnings.warn(f"Exp3 path {b} failed: {exc}", RuntimeWarning)
            continue

        # ── Subperiod metrics ──────────────────────────────────────────────
        for strat in strategies:
            if strat not in port_ret.columns:
                continue
            sr = port_ret[strat].dropna()
            for sp_start, sp_end in SUBPERIODS:
                sp_key = f"{sp_start[:4]}-{sp_end[:4]}"
                m = subperiod_metrics(sr, sp_start, sp_end)
                sub_data[strat][sp_key].append(m)

        # ── Ticker selections ──────────────────────────────────────────────
        for _, row in detail.iterrows():
            selection_records.append({
                "path":      b,
                "date":      row["date"],
                "model":     row["model"],
                "direction": row["direction"],
                "spec":      row["spec"],
                "selected":  row["selected"] if pd.notna(row.get("selected")) else "CASH",
                "is_cash":   bool(row.get("is_cash", row["selected"] is None)),
            })

    # ── Subperiod summary ──────────────────────────────────────────────────
    sub_rows = []
    for strat in strategies:
        for sp_key in ["2016-2018", "2019-2021", "2022-2024"]:
            data = sub_data[strat][sp_key]
            if not data:
                continue
            cum_rets = np.array([d["CumRet"] for d in data if np.isfinite(d["CumRet"])])
            sharpes  = np.array([d["Sharpe"]  for d in data if np.isfinite(d["Sharpe"])])
            maxdds   = np.array([d["MaxDD"]   for d in data if np.isfinite(d["MaxDD"])])
            sub_rows.append({
                "Strategy":      strat,
                "Subperiod":     sp_key,
                "CumRet_mean":   np.nanmean(cum_rets) if len(cum_rets) else np.nan,
                "CumRet_median": np.nanmedian(cum_rets) if len(cum_rets) else np.nan,
                "Sharpe_mean":   np.nanmean(sharpes) if len(sharpes) else np.nan,
                "MaxDD_mean":    np.nanmean(maxdds) if len(maxdds) else np.nan,
                "N_paths":       len(data),
            })

    sub_df = pd.DataFrame(sub_rows)

    # ── Selection stability ────────────────────────────────────────────────
    selection_df = pd.DataFrame(selection_records)
    stability_rows = _compute_selection_stability(selection_df, n_paths=n_paths)
    stability_df = pd.DataFrame(stability_rows)

    # ── Paper targets for subperiod (Table 5, p.15) ────────────────────────
    paper_subperiod_targets = {
        "DVR_long_classical":   [0.0816, -0.1489, 0.0966],
        "SSA_long_classical":   [-0.5548, 1.0959, 0.1906],
        "RLSSA_long_classical": [-0.4348, 0.1938, -0.2864],
        "BENCH":                [-0.1355, 0.1922, 0.1251],
    }

    # ── Print summary ──────────────────────────────────────────────────────
    if verbose:
        print("\n═══ Exp 3: Subperiod Robustness ═══")
        for strat in ["DVR_long_classical", "SSA_long_classical", "RLSSA_long_classical", "BENCH"]:
            strat_sub = sub_df[sub_df["Strategy"] == strat][["Subperiod", "CumRet_mean"]].set_index("Subperiod")
            print(f"\n  {strat}:")
            print(f"    Synthetic mean CumRet: {strat_sub['CumRet_mean'].values.round(4).tolist()}")
            if strat in paper_subperiod_targets:
                print(f"    Paper target:          {paper_subperiod_targets[strat]}")

        print("\n─── Selection Stability ───")
        print(stability_df[["model", "direction", "spec",
                             "avg_top1_share", "avg_cash_share", "match_pct", "n_comparable"]
                            ].round(4).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    sub_df.round(4).to_csv(RESULTS_DIR / "exp3_subperiod_metrics.csv", index=False)
    stability_df.round(4).to_csv(RESULTS_DIR / "exp3_selection_stability.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_subperiod(sub_df, RESULTS_DIR / "exp3_subperiod_boxplots.png")

    return {
        "subperiod_summary":    sub_df,
        "selection_stability":  stability_df,
        "paper_targets":        paper_subperiod_targets,
    }


def _compute_selection_stability(sel_df: pd.DataFrame, n_paths: int) -> list:
    """
    Compute Avg. C1/R1, cash frequency, and classical-vs-normalised match rate.

    Returns list of row dicts.
    """
    rows = []
    for model in ["DVR", "SSA", "RLSSA"]:
        for direction in ["long", "short"]:
            # Classical spec
            cl = sel_df[(sel_df["model"] == model) &
                        (sel_df["direction"] == direction) &
                        (sel_df["spec"] == "classical")]
            # Normalised spec
            vo = sel_df[(sel_df["model"] == model) &
                        (sel_df["direction"] == direction) &
                        (sel_df["spec"] == "normalised")]

            # For each spec: for each month, find leading traded ticker
            def monthly_leader_share(spec_df):
                """Returns mean leading-ticker share and mean cash share."""
                if spec_df.empty:
                    return np.nan, np.nan, {}
                all_months = spec_df["date"].unique()
                top1_shares = []
                cash_shares = []
                leading_by_month = {}
                for month in sorted(all_months):
                    month_data = spec_df[spec_df["date"] == month]
                    total = len(month_data)
                    cash_ct = (month_data["selected"] == "CASH").sum()
                    cash_share = cash_ct / total if total > 0 else 0
                    cash_shares.append(cash_share)
                    # Leading traded ticker
                    traded = month_data[month_data["selected"] != "CASH"]["selected"]
                    if len(traded) == 0:
                        top1_shares.append(0.0)
                        leading_by_month[month] = None
                    else:
                        cnt = traded.value_counts()
                        leader = cnt.index[0]
                        leader_share = cnt.iloc[0] / total
                        top1_shares.append(leader_share)
                        leading_by_month[month] = leader
                return np.mean(top1_shares), np.mean(cash_shares), leading_by_month

            avg_c1, avg_cash_c, cl_leader = monthly_leader_share(cl)
            avg_r1, avg_cash_v, vo_leader = monthly_leader_share(vo)

            # Match rate: months where both have a traded leader and they agree
            common_months = [m for m in cl_leader if m in vo_leader
                             and cl_leader.get(m) is not None
                             and vo_leader.get(m) is not None]
            n_comparable = len(common_months)
            if n_comparable > 0:
                matches = sum(1 for m in common_months if cl_leader[m] == vo_leader[m])
                match_pct = 100 * matches / n_comparable
            else:
                match_pct = np.nan

            rows.append({
                "model":           model,
                "direction":       direction,
                "spec":            "classical",
                "avg_top1_share":  avg_c1,
                "avg_cash_share":  avg_cash_c,
                "match_pct":       match_pct,
                "n_comparable":    n_comparable,
            })
            rows.append({
                "model":           model,
                "direction":       direction,
                "spec":            "normalised",
                "avg_top1_share":  avg_r1,
                "avg_cash_share":  avg_cash_v,
                "match_pct":       match_pct,
                "n_comparable":    n_comparable,
            })

    return rows


def _plot_subperiod(sub_df: pd.DataFrame, save_path: Path) -> None:
    """Bar chart of subperiod mean CumRet for selected strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_strats = [
        "DVR_long_classical", "SSA_long_classical", "RLSSA_long_classical",
        "DVR_short_classical", "SSA_short_classical", "RLSSA_short_classical",
        "BENCH",
    ]
    subperiods = ["2016-2018", "2019-2021", "2022-2024"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(subperiods))
    width = 0.1
    offsets = np.linspace(-0.3, 0.3, len(plot_strats))

    for i, strat in enumerate(plot_strats):
        vals = []
        for sp in subperiods:
            row = sub_df[(sub_df["Strategy"] == strat) & (sub_df["Subperiod"] == sp)]
            vals.append(row["CumRet_mean"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + offsets[i], vals, width, label=strat.replace("_", "\n"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(subperiods)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Cumulative Return")
    ax.set_title("Exp 3: Subperiod Mean Cumulative Return (MEB Paths, Synthetic Data)")
    ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-paths", type=int, default=MEB_N_PATHS)
    args = p.parse_args()

    results = run_exp3(n_paths=args.n_paths, verbose=True)
    print("\nExp 3 complete. Results saved to results/")
