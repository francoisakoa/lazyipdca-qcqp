#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
BACKEND="${BACKEND:-ldl}"
MAX_OUTER="${MAX_OUTER:-1000}"
MAX_INNER="${MAX_INNER:-1000}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
AUTO_SQP_MAX_N="${AUTO_SQP_MAX_N:-64}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG="v0412_failfix_hs116_core1_${BACKEND}_tol1e6_${STAMP}"
RESULTS="${RESULTS:-/data/projects/IPDCA_QCQP/results/$TAG}"
LOGS="${LOGS:-/data/projects/IPDCA_QCQP/logs/$TAG}"

mkdir -p "$CACHE" "$RESULTS" "$LOGS"
export PYCUTEST_CACHE="$CACHE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1

printf "%-10s %-4s %-20s %-12s %-14s %-14s %-14s %-8s %-8s %-10s %-12s\n" \
  PROBLEM RC STATUS ROUTE OBJECTIVE FEASIBILITY DUAL KKT OUTER INNER TIME_s

for P in HS116 CORE1; do
  WORK="$RESULTS/$P"
  LOG="$LOGS/$P.log"
  mkdir -p "$WORK"
  python -u "$ROOT/python/run_cutest_qcqp.py" "$P" \
    --binary "$BIN" --work-dir "$WORK" --cache "$CACHE" --keep-model \
    --reformulation auto \
    --algorithm auto --auto-sqp-max-n "$AUTO_SQP_MAX_N" \
    --inner-backend lazyqp --linear-backend "$BACKEND" --linear-metis 1 \
    --kkt-adaptive-reg --kkt-persistent-symbolic-cache \
    --lazyqp-kkt-form full --lazyqp-low-rank-form auto \
    --inner-full-dual-warm-start --lazyqp-warm-guard 2.0 \
    --max-outer "$MAX_OUTER" --max-inner "$MAX_INNER" --time-limit "$TIME_LIMIT" \
    --feas-tol 1e-6 --opt-tol 1e-6 --dual-inf-tol 1e-6 --compl-inf-tol 1e-6 \
    --inner-tol 1e-8 --inner-mu-fallback-tol 1e-7 --s-max 100 \
    --restoration-iters 600 --restoration-restarts 16 --restoration-restart-iters 400 \
    --restoration-minimax-iters 80 --restoration-minimax-prox 1e-6 \
    --prox 1e-6 --dca-extra-prox 0 --dca-affine-full-limit 4096 \
    --activation 1e-4 --constraint-model auto \
    --working-set-drop-slack 2e-4 --working-set-drop-mu 1e-9 \
    --max-objective-rank 64 --max-constraint-rank 24 --max-lagrangian-rank 64 \
    --terminal-kkt-trigger 1e-2 --terminal-kkt-iters 16 \
    --terminal-kkt-mu-abs 1e-9 --terminal-kkt-mu-rel 1e-8 --terminal-kkt-stability 3 \
    --lanczos-steps 40 --lanczos-starts 2 --exact-eigen-threshold 256 \
    --verbose 1 >"$LOG" 2>&1
  rc=$?
  JSON="$WORK/$P.json"
  python - "$P" "$rc" "$JSON" <<'PY'
import json, sys
from pathlib import Path
p, rc, path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if not path.is_file():
    print(f"{p:<10} {rc:<4} {'NO_JSON':<20} {'-':<12} {'-':>14} {'-':>14} {'-':>14} {'FAIL':>8} {'-':>8} {'-':>10} {'-':>12}")
else:
    d=json.loads(path.read_text())
    f=lambda v: '-' if v is None else (f'{v:.6e}' if isinstance(v,float) else str(v))
    print(f"{p:<10} {rc:<4} {str(d.get('status')):<20} {str(d.get('effective_outer_algorithm')):<12} "
          f"{f(d.get('cutest_objective',d.get('objective'))):>14} "
          f"{f(d.get('cutest_feasibility',d.get('feasibility'))):>14} "
          f"{f(d.get('dual_infeasibility_inf')):>14} "
          f"{('PASS' if d.get('ipopt_desired_convergence') else 'FAIL'):>8} "
          f"{f(d.get('outer_iterations')):>8} {f(d.get('inner_iterations')):>10} "
          f"{f(d.get('elapsed_seconds')):>12}")
PY
done

echo "RESULTS=$RESULTS"
echo "LOGS=$LOGS"
