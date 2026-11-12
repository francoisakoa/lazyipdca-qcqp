#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT=/data/projects/IPDCA_QCQP
CACHE=${PYCUTEST_CACHE:-$PROJECT_ROOT/pycutest_cache}
OUT=$PROJECT_ROOT/results/v0214_kktstop_r10_live_feasibility_progress/A4X12
LOG=$PROJECT_ROOT/logs/v0214_kktstop_r10_live_feasibility_progress/A4X12.log
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
  --soc-max-ratio 0.5 \
  --soc-trigger-ratio 1000 \
  --no-terminal-kkt \
  --prox 1e-6 \
  --trust-region \
  --tr-initial-radius 10 \
  --tr-min-radius 1e-10 \
  --tr-max-radius 10 \
  --tr-eta-accept 0.10 \
  --tr-eta-shrink 0.25 \
  --tr-eta-expand 0.75 \
  --tr-shrink 0.25 \
  --tr-expand 2.0 \
  --tr-boundary-fraction 0.80 \
  --tr-boundary-rel-tol 1e-4 \
  --tr-boundary-abs-tol 1e-9 \
  --tr-max-rejects 18 \
  --tr-feas-floor 1e-12 \
  --tr-feas-growth 2.0 \
  --tr-feas-reduction 0.10 \
  --tr-penalty-min 10 \
  --tr-penalty-safety 1.25 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "===== v0.2.14-KKTSTOP-r10 LIVE-FEASIBILITY-PROGRESS A4X12 ====="
grep -E \
'TR-SQP|TR-REJECT|^\[TR\]|KKT-STOP-CHECK|status|message|objective|feasibility|model stationarity|KKT-stop|outer / inner|trust-region|line-search backtracks|Armijo rejections|SOC attempt|elapsed' \
"$LOG" | tail -220

if [ "$rc" -ne 0 ]; then echo "[WARN] runner returned rc=$rc"; fi
