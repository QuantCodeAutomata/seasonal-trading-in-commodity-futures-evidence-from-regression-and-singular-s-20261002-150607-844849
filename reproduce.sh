#!/usr/bin/env bash
# reproduce.sh — Run all three experiments end-to-end.
#
# Usage:
#   bash reproduce.sh
#
# Outputs written to results/:
#   exp1_metrics.csv, exp1_portfolio_returns.csv
#   exp2_bootstrap_summary.csv, exp2_holm_results.csv
#   exp3_subperiod_metrics.csv, exp3_selection_stability.csv
#   RESULTS.md
#
# NOTE: Paper uses B=500 MEB paths and RLSSA_MAX_ITER=50,000.
#       Default config uses B=5 and max_iter=200 for practical runtime.
#       Override: MEB_PATHS=500 bash reproduce.sh
#       To change RLSSA iterations edit RLSSA_MAX_ITER in config.py.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${MEB_PATHS:=5}"   # default B=5; override with MEB_PATHS=500 bash reproduce.sh

echo "========================================================"
echo "  Seasonal Trading in Commodity Futures — Replication"
echo "  MEB paths: ${MEB_PATHS} (paper: 500)"
echo "========================================================"

# ── 1. Unit tests ─────────────────────────────────────────────────────────────
echo ""
echo "[Step 1/5] Running unit tests..."
python -m pytest tests/ -v --tb=short
echo "Tests complete."

# ── 2. Experiment 1: Historical backtest ──────────────────────────────────────
echo ""
echo "[Step 2/5] Running Experiment 1: Historical rolling backtest..."
python exp/exp1_historical_backtest.py
echo "Exp 1 complete → results/exp1_metrics.csv"

# ── 3+4. Experiments 2 & 3: MEB bootstrap (combined — generates panels once) ──
echo ""
echo "[Step 3/5] Running Experiments 2 & 3 (combined, B=${MEB_PATHS})..."
# run_exp23_combined.py generates one shared set of MEB panels and produces:
#   exp2_bootstrap_summary.csv, exp2_holm_results.csv, exp2_meb_sharpe_distributions.png
#   exp3_subperiod_metrics.csv, exp3_selection_stability.csv, exp3_subperiod_boxplots.png
python exp/run_exp23_combined.py
echo "Exp 2 & 3 complete → results/exp2_*.csv  results/exp3_*.csv"

# Skip separate step 4 (covered above); keep step numbering consistent.
echo ""
echo "[Step 4/5] (Experiments 2 & 3 completed in combined runner above)"

# ── 5. Generate RESULTS.md summary ────────────────────────────────────────────
echo ""
echo "[Step 5/5] Generating results/RESULTS.md..."
python scripts/generate_results_md.py
echo "RESULTS.md written."

echo ""
echo "========================================================"
echo "  All experiments complete.  See results/RESULTS.md"
echo "========================================================"
