"""
Experiment 2: Maximum Entropy Bootstrap path-sensitivity assessment and
paired Sharpe inference.

Methodology (paper pp. 11-12):
  - Generate B=500 MEB panels (seed=42, 10% trim).
  - Re-estimate all models recursively on each synthetic panel.
  - Report mean/median/SD of CumRet, Sharpe, MaxDD across 500 paths.
  - Conduct paired Sharpe tests (add-one, two-sided, non-studentized).
  - Apply Holm's correction within 6 families of 3 tests each.

Expected target (paper, p.13-14):
  Equal-weight benchmark: mean CumRet=0.1681, Sharpe=0.191, MaxDD=-0.414
  Classical SSA long: mean CumRet=0.1296, Sharpe=0.131
  None of 18 paired Sharpe tests rejects at familywise 5%.

NOTE: With synthetic data these exact values will not be reproduced; the
methodology is faithfully implemented.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict

from config import (
    MEB_N_PATHS, MEB_SEED, MEB_TRIM, WINDOW_MONTHS,
    TRADING_START, TRADING_END,
)
from data.generate_panel import load_panel
from src.meb import meb_panel
from src.backtest import run_backtest
from src.portfolio import compute_metrics, annualised_sharpe

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp2(
    panel: pd.DataFrame = None,
    n_paths: int = MEB_N_PATHS,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment 2: MEB bootstrap path-sensitivity and paired Sharpe tests.

    Parameters
    ----------
    panel : pd.DataFrame or None
    n_paths : int  (B = 500 in the paper; reduce for speed in testing)
    verbose : bool

    Returns
    -------
    dict with 'summary_stats', 'paired_tests', 'holm_results'
    """
    if panel is None:
        panel = load_panel()

    if verbose:
        print(f"Generating {n_paths} MEB panels (seed={MEB_SEED})...")

    # ── Step 1: Generate MEB panels ────────────────────────────────────────
    synthetic_panels = meb_panel(panel, B=n_paths, trim=MEB_TRIM, seed=MEB_SEED)

    # ── Step 2: Run backtest on each synthetic panel ───────────────────────
    all_metrics: Dict[str, list] = {}   # strategy -> list of metric dicts

    strategies = [
        f"{m}_{d}_{s}"
        for m in ["DVR", "SSA", "RLSSA"]
        for d in ["long", "short"]
        for s in ["classical", "normalised"]
    ] + ["BENCH"]

    for strat in strategies:
        all_metrics[strat] = []

    all_sharpes: Dict[str, list] = {s: [] for s in strategies}

    for b, syn_panel in enumerate(synthetic_panels):
        if verbose and b % 50 == 0:
            print(f"  MEB path {b+1}/{n_paths}...")
        try:
            port_ret, _ = run_backtest(
                simple_returns=syn_panel,
                models=["DVR", "SSA", "RLSSA"],
                directions=["long", "short"],
                specs=["classical", "normalised"],
                window=WINDOW_MONTHS,
                verbose=False,
                rlssa_rng=np.random.default_rng(b),
            )
        except Exception as exc:
            warnings.warn(f"Path {b} failed: {exc}", RuntimeWarning)
            continue

        for strat in strategies:
            if strat not in port_ret.columns:
                continue
            ret = port_ret[strat].dropna().values
            if len(ret) < 2:
                continue
            m = compute_metrics(ret)
            all_metrics[strat].append(m)
            all_sharpes[strat].append(m["Sharpe"])

    # ── Step 3: Summary statistics across paths ────────────────────────────
    summary_rows = []
    for strat in strategies:
        data = all_metrics[strat]
        if not data:
            continue
        cum_rets = np.array([d["CumRet"] for d in data])
        sharpes  = np.array([d["Sharpe"]  for d in data])
        maxdds   = np.array([d["MaxDD"]   for d in data])
        summary_rows.append({
            "Strategy":           strat,
            "CumRet_mean":        np.nanmean(cum_rets),
            "CumRet_median":      np.nanmedian(cum_rets),
            "CumRet_sd":          np.nanstd(cum_rets, ddof=1),
            "Sharpe_mean":        np.nanmean(sharpes),
            "Sharpe_median":      np.nanmedian(sharpes),
            "Sharpe_sd":          np.nanstd(sharpes, ddof=1),
            "MaxDD_mean":         np.nanmean(maxdds),
            "MaxDD_median":       np.nanmedian(maxdds),
            "MaxDD_sd":           np.nanstd(maxdds, ddof=1),
            "N_valid_paths":      len(data),
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Strategy")

    # ── Step 4: Historical observed-path Sharpe for paired tests ──────────
    # Run historical backtest to get observed Sharpe differences
    hist_port_ret, _ = run_backtest(
        simple_returns=panel,
        models=["DVR", "SSA", "RLSSA"],
        directions=["long", "short"],
        specs=["classical", "normalised"],
        window=WINDOW_MONTHS,
        verbose=False,
        rlssa_rng=np.random.default_rng(0),
    )
    hist_sharpes = {}
    for strat in strategies:
        if strat in hist_port_ret.columns:
            ret = hist_port_ret[strat].dropna().values
            hist_sharpes[strat] = annualised_sharpe(ret) if len(ret) >= 2 else np.nan
        else:
            hist_sharpes[strat] = np.nan

    # ── Step 5: Paired Sharpe tests ────────────────────────────────────────
    # 18 comparisons = 3 comparison_types × 2 directions × 3 models
    comparison_families = _define_paired_comparisons()
    paired_results = _run_paired_sharpe_tests(
        comparison_families=comparison_families,
        all_sharpes=all_sharpes,
        hist_sharpes=hist_sharpes,
        B=n_paths,
    )

    # ── Step 6: Holm correction ────────────────────────────────────────────
    holm_results = _apply_holm_correction(paired_results)

    # ── Print summaries ────────────────────────────────────────────────────
    if verbose:
        print("\n═══ Exp 2: MEB Bootstrap Summary Statistics ═══")
        cols_to_show = ["CumRet_mean", "CumRet_median", "Sharpe_mean", "MaxDD_mean", "N_valid_paths"]
        print(summary_df[cols_to_show].round(4).to_string())
        print("\n─── Paired Sharpe Tests (Holm adjusted) ───")
        print(holm_results[["family", "comparison", "hist_d", "pval_raw", "pval_holm", "reject"]].round(4).to_string())

    # ── Save results ──────────────────────────────────────────────────────
    summary_df.round(4).to_csv(RESULTS_DIR / "exp2_bootstrap_summary.csv")
    holm_results.round(4).to_csv(RESULTS_DIR / "exp2_holm_results.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_meb_distributions(all_metrics, RESULTS_DIR / "exp2_meb_sharpe_distributions.png")

    return {
        "summary_df":    summary_df,
        "paired_results": paired_results,
        "holm_results":  holm_results,
        "hist_sharpes":  hist_sharpes,
    }


def _define_paired_comparisons() -> list:
    """
    Define the 18 paired Sharpe comparisons across 6 families.

    Returns list of dicts: {family, comparison, strat_a, strat_b, direction, comp_type}
    """
    comparisons = []
    for direction in ["long", "short"]:
        for model in ["DVR", "SSA", "RLSSA"]:
            cl = f"{model}_{direction}_classical"
            vo = f"{model}_{direction}_normalised"
            bn = "BENCH"
            # Type 1: normalised vs classical
            comparisons.append({
                "family":      f"norm_vs_classical_{direction}",
                "comparison":  f"{model}_{direction}",
                "comp_type":   "normalised_vs_classical",
                "strat_a":     vo,
                "strat_b":     cl,
                "direction":   direction,
                "model":       model,
            })
            # Type 2: classical vs benchmark
            comparisons.append({
                "family":      f"classical_vs_bench_{direction}",
                "comparison":  f"{model}_{direction}",
                "comp_type":   "classical_vs_bench",
                "strat_a":     cl,
                "strat_b":     bn,
                "direction":   direction,
                "model":       model,
            })
            # Type 3: normalised vs benchmark
            comparisons.append({
                "family":      f"normalised_vs_bench_{direction}",
                "comparison":  f"{model}_{direction}",
                "comp_type":   "normalised_vs_bench",
                "strat_a":     vo,
                "strat_b":     bn,
                "direction":   direction,
                "model":       model,
            })
    return comparisons


def _run_paired_sharpe_tests(
    comparison_families: list,
    all_sharpes: dict,
    hist_sharpes: dict,
    B: int,
) -> pd.DataFrame:
    """
    Compute add-one, two-sided, non-studentized paired bootstrap p-values.

    Formula (paper p.12):
      d = hist_sharpe_a - hist_sharpe_b
      d_b* = bootstrap_sharpe_a[b] - bootstrap_sharpe_b[b]
      Null distribution: d_b* - d  (centred bootstrap differences)
      p-value = [1 + count(|d_b* - d| >= |d|)] / (B + 1)
    """
    rows = []
    for comp in comparison_families:
        strat_a, strat_b = comp["strat_a"], comp["strat_b"]
        sha = np.array(all_sharpes.get(strat_a, []))
        shb = np.array(all_sharpes.get(strat_b, []))
        n_valid = min(len(sha), len(shb))

        if n_valid < 2:
            rows.append({**comp, "hist_d": np.nan, "pval_raw": np.nan, "N_pairs": n_valid})
            continue

        sha = sha[:n_valid]
        shb = shb[:n_valid]

        d_obs = hist_sharpes.get(strat_a, np.nan) - hist_sharpes.get(strat_b, np.nan)
        db    = sha - shb                          # bootstrap differences d_b*
        centred = db - d_obs                       # centred under null

        if not np.isfinite(d_obs):
            pval = np.nan
        else:
            n_extreme = np.sum(np.abs(centred) >= np.abs(d_obs))
            pval = (1 + n_extreme) / (B + 1)

        rows.append({
            **comp,
            "hist_d":   d_obs,
            "pval_raw": pval,
            "N_pairs":  n_valid,
        })

    return pd.DataFrame(rows)


def _apply_holm_correction(paired_results: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm's sequential correction within each of the 6 families."""
    from statsmodels.stats.multitest import multipletests

    all_rows = []
    for family in paired_results["family"].unique():
        fam = paired_results[paired_results["family"] == family].copy()
        pvals = fam["pval_raw"].values
        valid = np.isfinite(pvals)
        if not np.any(valid):
            fam["pval_holm"] = np.nan
            fam["reject"]    = False
        else:
            _, padj, _, _ = multipletests(
                np.where(valid, pvals, 1.0),
                alpha=0.05,
                method="holm",
            )
            fam["pval_holm"] = np.where(valid, padj, np.nan)
            fam["reject"]    = fam["pval_holm"] <= 0.05
        all_rows.append(fam)

    return pd.concat(all_rows, ignore_index=True)


def _plot_meb_distributions(all_metrics: dict, save_path: Path) -> None:
    """Plot distribution of Sharpe ratios across 500 MEB paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies_to_plot = [
        "DVR_long_classical", "SSA_long_classical", "RLSSA_long_classical",
        "DVR_short_classical", "SSA_short_classical", "RLSSA_short_classical",
        "BENCH",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, strat in enumerate(strategies_to_plot):
        ax = axes[i]
        data = all_metrics.get(strat, [])
        if not data:
            ax.set_title(strat)
            continue
        sharpes = [d["Sharpe"] for d in data if np.isfinite(d["Sharpe"])]
        ax.hist(sharpes, bins=30, edgecolor="white", color="steelblue", alpha=0.8)
        ax.axvline(np.mean(sharpes), color="red", linewidth=1.5, label=f"Mean={np.mean(sharpes):.3f}")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(strat.replace("_", "\n"), fontsize=8)
        ax.set_xlabel("Sharpe")
        ax.legend(fontsize=6)

    if len(strategies_to_plot) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle("Exp 2: MEB Sharpe Distribution (500 Paths)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    # Use smaller n_paths for interactive testing; full run uses 500
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-paths", type=int, default=MEB_N_PATHS)
    args = p.parse_args()

    results = run_exp2(n_paths=args.n_paths, verbose=True)
    print("\nExp 2 complete. Results saved to results/")
