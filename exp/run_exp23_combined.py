"""
Combined Exp 2 + Exp 3 runner.

Generates MEB panels once and runs the full backtest once per path,
collecting all data needed by both experiments simultaneously.

Exp 2: MEB path-sensitivity summary + paired Sharpe tests with Holm correction
Exp 3: Subperiod robustness + Top-1 ticker-selection stability

Outputs to results/:
  exp2_bootstrap_summary.csv
  exp2_holm_results.csv
  exp2_meb_sharpe_distributions.png
  exp3_subperiod_metrics.csv
  exp3_selection_stability.csv
  exp3_subperiod_boxplots.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests

from config import (
    MEB_N_PATHS, MEB_SEED, MEB_TRIM,
    SUBPERIODS, TRADING_START, TRADING_END,
)
from data.generate_panel import load_panel
from src.meb import meb_panel
from src.backtest import run_backtest
from src.portfolio import compute_metrics, annualised_sharpe, subperiod_metrics

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STRATEGIES = [
    f"{m}_{d}_{s}"
    for m in ["DVR", "SSA", "RLSSA"]
    for d in ["long", "short"]
    for s in ["classical", "normalised"]
] + ["BENCH"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_metrics(returns: pd.Series) -> dict:
    """Return CumRet/Sharpe/MaxDD; NaN if fewer than 2 observations."""
    valid = returns.dropna()
    if len(valid) < 2:
        return {"CumRet": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    m = compute_metrics(valid)
    return {"CumRet": m["CumRet"], "Sharpe": m["Sharpe"], "MaxDD": m["MaxDD"]}


def _safe_sharpe(returns: pd.Series) -> float:
    valid = returns.dropna()
    if len(valid) < 2:
        return np.nan
    return float(annualised_sharpe(valid))


def _subperiod_returns(
    port_ret: pd.DataFrame, strat: str, sp_labels: list
) -> dict:
    """Extract subperiod return series for a strategy column.
    sp_labels is a list of (label, start_YYYY-MM, end_YYYY-MM) 3-tuples.
    """
    col = "BENCH" if strat == "BENCH" else strat
    if col not in port_ret.columns:
        return {label: pd.Series(dtype=float) for label, _, _ in sp_labels}
    result = {}
    for label, start, end in sp_labels:
        t_start = pd.Period(start, "M").to_timestamp("M")
        t_end   = pd.Period(end,   "M").to_timestamp("M")
        mask = (port_ret.index >= t_start) & (port_ret.index <= t_end)
        result[label] = port_ret.loc[mask, col].dropna()
    return result


# ── Paired Sharpe test helpers ────────────────────────────────────────────────

def _comparison_families() -> list:
    """
    18 comparisons = 3 types × 2 directions × 3 models.
    Returns list of dicts with family, strat_a, strat_b labels.
    """
    comparisons = []
    for direction in ["long", "short"]:
        for model in ["DVR", "SSA", "RLSSA"]:
            classical_strat  = f"{model}_{direction}_classical"
            normalised_strat = f"{model}_{direction}_normalised"
            bench            = "BENCH"
            comparisons += [
                {
                    "family":  f"norm_vs_classical_{direction}",
                    "model":   model,
                    "direction": direction,
                    "comparison": "normalised_minus_classical",
                    "strat_a": normalised_strat,
                    "strat_b": classical_strat,
                },
                {
                    "family":  f"classical_vs_bench_{direction}",
                    "model":   model,
                    "direction": direction,
                    "comparison": "classical_minus_bench",
                    "strat_a": classical_strat,
                    "strat_b": bench,
                },
                {
                    "family":  f"normalised_vs_bench_{direction}",
                    "model":   model,
                    "direction": direction,
                    "comparison": "normalised_minus_bench",
                    "strat_a": normalised_strat,
                    "strat_b": bench,
                },
            ]
    return comparisons


def _paired_sharpe_pvalues(
    comp_families: list,
    boot_sharpes: Dict[str, list],
    hist_sharpes: Dict[str, float],
    B: int,
) -> pd.DataFrame:
    """
    Compute add-one two-sided non-studentized p-values (paper p.12).
    d_obs = hist_sharpe_a - hist_sharpe_b
    p = (1 + #{|d_b* - d_obs| >= |d_obs|}) / (B + 1)
    """
    rows = []
    for comp in comp_families:
        a, b_strat = comp["strat_a"], comp["strat_b"]
        sha = np.asarray(boot_sharpes.get(a, []), dtype=float)
        shb = np.asarray(boot_sharpes.get(b_strat, []), dtype=float)
        n = min(len(sha), len(shb))
        d_obs = hist_sharpes.get(a, np.nan) - hist_sharpes.get(b_strat, np.nan)

        if n < 2 or not np.isfinite(d_obs):
            rows.append({**comp, "hist_d": d_obs, "pval_raw": np.nan, "N_pairs": n})
        else:
            diff_boot = sha[:n] - shb[:n]
            centred   = diff_boot - d_obs
            n_extreme = int(np.sum(np.abs(centred) >= np.abs(d_obs)))
            pval      = (1 + n_extreme) / (B + 1)
            rows.append({**comp, "hist_d": d_obs, "pval_raw": pval, "N_pairs": n})

    return pd.DataFrame(rows)


def _apply_holm(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm correction within each family column."""
    parts = []
    for fam in df["family"].unique():
        sub = df[df["family"] == fam].copy()
        pvals = sub["pval_raw"].values
        valid = np.isfinite(pvals)
        if valid.any():
            _, padj, _, _ = multipletests(
                np.where(valid, pvals, 1.0),
                alpha=0.05, method="holm",
            )
            sub["pval_holm"] = np.where(valid, padj, np.nan)
        else:
            sub["pval_holm"] = np.nan
        sub["reject_05"] = sub["pval_holm"] <= 0.05
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# ── Ticker-stability helpers ──────────────────────────────────────────────────

def _update_selection_storage(
    sel_store: Dict[str, Dict[str, List]],
    detail_records: list,
) -> None:
    """Add one path's ticker selections to the running per-month counters."""
    for rec in detail_records:
        key    = f"{rec['model']}_{rec['direction']}_{rec['spec']}"
        month  = str(rec["date"])[:7]
        ticker = rec.get("selected", None)  # None or str

        if key not in sel_store:
            sel_store[key] = {}
        if month not in sel_store[key]:
            sel_store[key][month] = []
        sel_store[key][month].append(ticker if ticker else "__CASH__")


def _selection_stability(sel_store: Dict[str, Dict[str, List]]) -> pd.DataFrame:
    """
    For each strategy and month, find the leading ticker (most frequent non-cash).
    Compute Avg C1 / Avg R1, cash freq, and classical-vs-normalised match rate.
    """
    rows = []
    all_months = sorted(set(
        m for strat_months in sel_store.values() for m in strat_months
    ))

    for model in ["DVR", "SSA", "RLSSA"]:
        for direction in ["long", "short"]:
            c_key = f"{model}_{direction}_classical"
            n_key = f"{model}_{direction}_normalised"

            c_data = sel_store.get(c_key, {})
            n_data = sel_store.get(n_key, {})

            c_lead_shares, n_lead_shares = [], []
            c_cash_shares, n_cash_shares = [], []
            match_months, comparable = 0, 0

            for month in all_months:
                c_picks = c_data.get(month, [])
                n_picks = n_data.get(month, [])
                B_local = len(c_picks) if c_picks else len(n_picks)
                if B_local == 0:
                    continue

                # --- classical ---
                c_noncash = [t for t in c_picks if t != "__CASH__"]
                c_cash_sh = 1.0 - len(c_noncash) / len(c_picks) if c_picks else np.nan
                c_cash_shares.append(c_cash_sh)

                c_lead, c_lead_sh = None, 0.0
                if c_noncash:
                    ctr = Counter(c_noncash)
                    c_lead, cnt = ctr.most_common(1)[0]
                    c_lead_sh = cnt / len(c_picks)
                c_lead_shares.append(c_lead_sh)

                # --- normalised ---
                n_noncash = [t for t in n_picks if t != "__CASH__"]
                n_cash_sh = 1.0 - len(n_noncash) / len(n_picks) if n_picks else np.nan
                n_cash_shares.append(n_cash_sh)

                n_lead, n_lead_sh = None, 0.0
                if n_noncash:
                    ctr = Counter(n_noncash)
                    n_lead, cnt = ctr.most_common(1)[0]
                    n_lead_sh = cnt / len(n_picks)
                n_lead_shares.append(n_lead_sh)

                # --- match ---
                if c_lead and n_lead:
                    comparable += 1
                    if c_lead == n_lead:
                        match_months += 1

            avg_c1 = 100 * np.mean(c_lead_shares) if c_lead_shares else np.nan
            avg_r1 = 100 * np.mean(n_lead_shares) if n_lead_shares else np.nan
            c_cash = 100 * np.mean(c_cash_shares) if c_cash_shares else np.nan
            v_cash = 100 * np.mean(n_cash_shares) if n_cash_shares else np.nan
            match_pct = 100 * match_months / comparable if comparable else np.nan

            rows.append({
                "Model":     model,
                "Direction": direction,
                "Avg_C1_pct":   round(avg_c1, 2),
                "Avg_R1_pct":   round(avg_r1, 2),
                "C_cash_pct":   round(c_cash, 2) if model == "DVR" else np.nan,
                "V_cash_pct":   round(v_cash, 2) if model == "DVR" else np.nan,
                "Match_pct":    round(match_pct, 2),
                "n_comparable": comparable,
            })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_sharpe_dists(all_metrics: Dict[str, list], path: Path) -> None:
    strats = [
        "DVR_long_classical", "SSA_long_classical", "RLSSA_long_classical",
        "DVR_short_classical", "SSA_short_classical", "RLSSA_short_classical",
        "BENCH",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, strat in enumerate(strats):
        ax = axes[i]
        data = [d["Sharpe"] for d in all_metrics.get(strat, []) if np.isfinite(d["Sharpe"])]
        if data:
            ax.hist(data, bins=max(3, len(data) // 2), color="steelblue",
                    edgecolor="white", alpha=0.8)
            ax.axvline(np.mean(data), color="red", lw=1.5,
                       label=f"Mean={np.mean(data):.3f}")
            ax.axvline(0, color="black", lw=0.8, ls="--")
            ax.legend(fontsize=7)
        ax.set_title(strat.replace("_", "\n"), fontsize=8)
        ax.set_xlabel("Sharpe")
    if len(strats) < len(axes):
        for ax in axes[len(strats):]:
            ax.set_visible(False)
    fig.suptitle(f"Exp 2: MEB Sharpe Distribution ({len(list(all_metrics.values())[0])} Paths)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_subperiod_boxes(sub_data: Dict[str, Dict[str, list]], path: Path) -> None:
    models  = ["DVR", "SSA", "RLSSA"]
    periods = ["2016-2018", "2019-2021", "2022-2024"]
    fig, axes = plt.subplots(len(models), len(periods), figsize=(14, 10), sharey=False)
    for mi, model in enumerate(models):
        for pi, period in enumerate(periods):
            ax = axes[mi, pi]
            data = [sub_data.get(f"{model}_long_classical", {}).get(period, [])]
            ax.boxplot([d for d in data if d], tick_labels=[f"{model}\nlong_cl"])
            ax.set_title(f"{model} | {period}", fontsize=8)
            ax.axhline(0, color="black", lw=0.8, ls="--")
    plt.suptitle("Exp 3: Subperiod CumRet Distribution (MEB paths)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_exp23(panel: pd.DataFrame = None, n_paths: int = MEB_N_PATHS,
              verbose: bool = True) -> dict:
    """
    Single-pass combined runner for Experiments 2 and 3.
    Generates MEB panels once; runs run_backtest once per path.
    Saves all output CSV/PNG to results/.
    """
    if panel is None:
        panel = load_panel()

    print(f"Generating {n_paths} MEB panels (seed={MEB_SEED}, trim={MEB_TRIM})...")
    synthetic_panels = meb_panel(panel, B=n_paths, trim=MEB_TRIM, seed=MEB_SEED)

    # SUBPERIODS from config is list of (start, end) 2-tuples; create labels
    sp_labels = [
        (f"{s[:4]}-{e[:4]}", s, e)   # ("2016-2018", "2016-01", "2018-12")
        for s, e in SUBPERIODS
    ]

    # ── Storage initialisation ────────────────────────────────────────────
    # Exp2: list of metric dicts per strategy
    all_metrics:  Dict[str, list] = {s: [] for s in STRATEGIES}
    boot_sharpes: Dict[str, list] = {s: [] for s in STRATEGIES}

    # Exp3: subperiod CumRet per (strategy, subperiod)
    sub_cumrets: Dict[str, Dict[str, list]] = {
        strat: {label: [] for label, _, _ in sp_labels}
        for strat in STRATEGIES
    }

    # Exp3: ticker selections per (strategy, month)
    sel_store: Dict[str, Dict[str, list]] = {}

    # Historical path (observed) Sharpe for paired tests
    print("Running historical backtest for observed-path Sharpe values...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist_port, hist_detail = run_backtest(panel)

    hist_sharpes: Dict[str, float] = {}
    for strat in STRATEGIES:
        col = "BENCH" if strat == "BENCH" else strat
        if col in hist_port.columns:
            hist_sharpes[strat] = _safe_sharpe(hist_port[col])
        else:
            hist_sharpes[strat] = np.nan

    # ── Main MEB loop ─────────────────────────────────────────────────────
    for b, syn_panel in enumerate(synthetic_panels):
        if verbose:
            print(f"  Path {b + 1}/{n_paths}...", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            port_ret, detail_records = run_backtest(syn_panel)

        # ── Exp 2: full-period metrics ─────────────────────────────────
        for strat in STRATEGIES:
            col = "BENCH" if strat == "BENCH" else strat
            if col in port_ret.columns:
                m = _safe_metrics(port_ret[col])
                all_metrics[strat].append(m)
                boot_sharpes[strat].append(m["Sharpe"])
            else:
                all_metrics[strat].append(
                    {"CumRet": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
                )
                boot_sharpes[strat].append(np.nan)

        # ── Exp 3: subperiod metrics ───────────────────────────────────
        for strat in STRATEGIES:
            sp_rets = _subperiod_returns(port_ret, strat, sp_labels)
            for label, ret_series in sp_rets.items():
                if len(ret_series) >= 2:
                    m = compute_metrics(ret_series)
                    sub_cumrets[strat][label].append(m["CumRet"])
                else:
                    sub_cumrets[strat][label].append(np.nan)

        # ── Exp 3: ticker selections (detail_records is a DataFrame) ──
        if len(detail_records) > 0:
            for _, row in detail_records.iterrows():
                model_name = str(row.get("model", ""))
                if model_name == "BENCH" or not model_name:
                    continue
                direction_ = str(row.get("direction", ""))
                spec_      = str(row.get("spec", ""))
                key        = f"{model_name}_{direction_}_{spec_}"
                month      = str(row["date"])[:7]
                tick       = row.get("selected") or "__CASH__"
                sel_store.setdefault(key, {}).setdefault(month, []).append(tick)

    # ── Exp 2: build summary table ────────────────────────────────────────
    print("\nBuilding Exp 2 summary statistics...")
    summary_rows = []
    for strat in STRATEGIES:
        vals = all_metrics[strat]
        valid_cr = [v["CumRet"] for v in vals if np.isfinite(v["CumRet"])]
        valid_sh = [v["Sharpe"]  for v in vals if np.isfinite(v["Sharpe"])]
        valid_dd = [v["MaxDD"]   for v in vals if np.isfinite(v["MaxDD"])]
        summary_rows.append({
            "Strategy":    strat,
            "Mean_CumRet": np.mean(valid_cr) if valid_cr else np.nan,
            "Med_CumRet":  np.median(valid_cr) if valid_cr else np.nan,
            "SD_CumRet":   np.std(valid_cr)   if valid_cr else np.nan,
            "Mean_Sharpe": np.mean(valid_sh) if valid_sh else np.nan,
            "Med_Sharpe":  np.median(valid_sh) if valid_sh else np.nan,
            "SD_Sharpe":   np.std(valid_sh)   if valid_sh else np.nan,
            "Mean_MaxDD":  np.mean(valid_dd) if valid_dd else np.nan,
            "Med_MaxDD":   np.median(valid_dd) if valid_dd else np.nan,
            "SD_MaxDD":    np.std(valid_dd)   if valid_dd else np.nan,
            "N_paths":     len(valid_cr),
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Strategy")
    summary_df.round(4).to_csv(RESULTS_DIR / "exp2_bootstrap_summary.csv")
    print(f"  Saved: {RESULTS_DIR}/exp2_bootstrap_summary.csv")

    # ── Exp 2: paired Sharpe tests ────────────────────────────────────────
    comp_fams  = _comparison_families()
    paired_df  = _paired_sharpe_pvalues(comp_fams, boot_sharpes, hist_sharpes, n_paths)
    holm_df    = _apply_holm(paired_df)
    holm_df.to_csv(RESULTS_DIR / "exp2_holm_results.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}/exp2_holm_results.csv")
    print(f"\n  Any Holm-rejections at 5%? {holm_df['reject_05'].any()}")

    # ── Exp 2: plots ──────────────────────────────────────────────────────
    _plot_sharpe_dists(all_metrics, RESULTS_DIR / "exp2_meb_sharpe_distributions.png")

    # ── Exp 3: subperiod summary ──────────────────────────────────────────
    print("\nBuilding Exp 3 subperiod summary...")
    sp_rows = []
    for strat in STRATEGIES:
        row = {"Strategy": strat}
        for label, _, _ in sp_labels:
            vals = [v for v in sub_cumrets[strat][label] if np.isfinite(v)]
            row[f"Mean_{label}"]    = round(np.mean(vals), 4)   if vals else np.nan
            row[f"Median_{label}"]  = round(np.median(vals), 4) if vals else np.nan
            row[f"SD_{label}"]      = round(np.std(vals), 4)    if vals else np.nan
        sp_rows.append(row)

    sp_df = pd.DataFrame(sp_rows).set_index("Strategy")
    sp_df.to_csv(RESULTS_DIR / "exp3_subperiod_metrics.csv")
    print(f"  Saved: {RESULTS_DIR}/exp3_subperiod_metrics.csv")

    # ── Exp 3: selection stability ────────────────────────────────────────
    print("Building Exp 3 selection-stability table...")
    stab_df = _selection_stability(sel_store)
    stab_df.to_csv(RESULTS_DIR / "exp3_selection_stability.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}/exp3_selection_stability.csv")

    # ── Exp 3: plots ──────────────────────────────────────────────────────
    _plot_subperiod_boxes(sub_cumrets, RESULTS_DIR / "exp3_subperiod_boxplots.png")

    # ── Print summaries ───────────────────────────────────────────────────
    print("\n═══ Exp 2: Bootstrap Summary (top rows) ═══")
    print(summary_df[["Mean_CumRet", "Mean_Sharpe", "Mean_MaxDD", "N_paths"]].to_string())

    print("\n═══ Exp 2: Holm-adjusted Sharpe Tests ═══")
    cols = ["family", "model", "comparison", "hist_d", "pval_raw", "pval_holm", "reject_05"]
    print(holm_df[cols].to_string(index=False))

    print("\n═══ Exp 3: Subperiod Mean CumRets ═══")
    sp_cols = [c for c in sp_df.columns if "Mean" in c]
    print(sp_df[sp_cols].to_string())

    print("\n═══ Exp 3: Selection Stability ═══")
    print(stab_df.to_string(index=False))

    return {
        "exp2_summary":    summary_df,
        "exp2_holm":       holm_df,
        "exp3_subperiod":  sp_df,
        "exp3_stability":  stab_df,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-paths", type=int, default=MEB_N_PATHS)
    args = p.parse_args()
    results = run_exp23(n_paths=args.n_paths, verbose=True)
    print("\nCombined Exp 2 + 3 complete. Results saved to results/")
