#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT=/data/projects/IPDCA_QCQP
CACHE=${PYCUTEST_CACHE:-$PROJECT_ROOT/pycutest_cache}
OUT=$PROJECT_ROOT/results/v0214_kktstop/A4X12
LOG=$PROJECT_ROOT/logs/v0214_kktstop/A4X12.log
mkdir -p "$OUT" "$(dirname "$LOG")"

python "$ROOT/python/run_cutest_qcqp.py" A4X12 \
  --binary "$ROOT/build/lazyipdca_qcqp_cli" \
  --reformulation auto \
  --cache "$CACHE" \
  --work-dir "$OUT" \
  --keep-model \
  --inner-backend native \
  --linear-backend native \
  --linear-metis 1 \
  --time-limit 900 \
  --max-outer 1000 \
  --max-inner 1000 \
  --verbose 1 \
  --feas-tol 1e-7 \
  --opt-tol 1e-6 \
  --dual-inf-tol 1 \
  --compl-inf-tol 1e-4 \
  --s-max 100 \
  --inner-tol 1e-7 \
  --inner-mu-fallback-tol 1e-6 \
  --restoration-iters 600 \
  --restoration-restarts 16 \
  --restoration-restart-iters 400 \
  --restoration-minimax-iters 80 \
  --restoration-minimax-prox 1e-6 \
  --constraint-model auto \
  --max-constraint-rank 24 \
  --max-lagrangian-rank 64 \
  --working-set-drop-slack 2e-4 \
  --working-set-drop-mu 1e-9 \
  --line-search-armijo 1e-4 \
  --soc-max-ratio 0.5 \
  --soc-trigger-ratio 1000 \
  --no-terminal-kkt \
  --prox 1e-6 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "===== v0.2.14-KKTSTOP A4X12 ====="
grep -E \
'KKT-STOP-CHECK|status|message|objective|feasibility|model stationarity|KKT-stop|outer / inner|line-search backtracks|SOC attempt|elapsed' \
"$LOG" | tail -100

if [ "$rc" -ne 0 ]; then echo "[WARN] runner returned rc=$rc"; fi
