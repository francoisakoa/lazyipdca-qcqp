#!/usr/bin/env bash
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE="${BASE:-/data/projects/IPDCA_QCQP}"
BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
CACHE="${PYCUTEST_CACHE:-$BASE/pycutest_cache}"
BACKEND="${BACKEND:-ldl}"
MAX_OUTER="${MAX_OUTER:-1000}"
MAX_INNER="${MAX_INNER:-1000}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
VERBOSE="${VERBOSE:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG="v0413_core1_certificate_refinement_${BACKEND}_tol1e6_${STAMP}"
WORK="${WORK:-$BASE/results/$TAG/CORE1}"
LOGDIR="${LOGDIR:-$BASE/logs/$TAG}"
LOG="$LOGDIR/CORE1.log"

cd "$ROOT"
unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export PYCUTEST_CACHE="$CACHE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
mkdir -p "$WORK" "$LOGDIR" "$CACHE"

BACKENDS="$($BIN --linear-backends)"
if [[ ",$BACKENDS," != *",$BACKEND,"* ]]; then
  echo "[FAIL] requested backend '$BACKEND' is not compiled: $BACKENDS" >&2
  return 2 2>/dev/null || true
fi

python -u "$ROOT/python/run_cutest_qcqp.py" CORE1 \
  --binary "$BIN" \
  --work-dir "$WORK" \
  --cache "$CACHE" \
  --keep-model \
  --reformulation auto \
  --algorithm auto \
  --auto-sqp-max-n 64 \
  --inner-backend lazyqp \
  --linear-backend "$BACKEND" \
  --linear-metis 1 \
  --kkt-adaptive-reg \
  --kkt-persistent-symbolic-cache \
  --lazyqp-kkt-form full \
  --lazyqp-low-rank-form auto \
  --lazyqp-condense-threshold 256 \
  --inner-full-dual-warm-start \
  --lazyqp-warm-guard 2.0 \
  --max-outer "$MAX_OUTER" \
  --max-inner "$MAX_INNER" \
  --time-limit "$TIME_LIMIT" \
  --feas-tol 1e-6 \
  --opt-tol 1e-6 \
  --dual-inf-tol 1e-6 \
  --compl-inf-tol 1e-6 \
  --inner-tol 1e-8 \
  --inner-mu-fallback-tol 1e-7 \
  --s-max 100 \
  --restoration-iters 600 \
  --restoration-restarts 16 \
  --restoration-restart-iters 400 \
  --restoration-minimax-iters 80 \
  --restoration-minimax-prox 1e-6 \
  --prox 1e-6 \
  --dca-extra-prox 0 \
  --activation 1e-4 \
  --constraint-model auto \
  --working-set-drop-slack 2e-4 \
  --working-set-drop-mu 1e-9 \
  --max-objective-rank 64 \
  --max-constraint-rank 24 \
  --max-lagrangian-rank 64 \
  --lanczos-steps 40 \
  --lanczos-starts 2 \
  --exact-eigen-threshold 256 \
  --verbose "$VERBOSE" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "===== CORE1 v0.4.13 CERTIFICATE REFINEMENT ====="
grep -E \
'^\[1/2\]|^\[ALGORITHM-ROUTE\]|^\[PHASE-I\]|^\[QCQP-SQP\]|^status|^message|^objective|^feasibility|^KKT-stop dual|^KKT-stop complement|^KKT-stop scaled|^KKT-stop fit|^KKT-stop terminal rescue|^KKT-stop desired|^outer / inner|^KKT factorizations|^elapsed' \
"$LOG" | tail -260

echo "[RUNNER] rc=$rc log=$LOG json=$WORK/CORE1.json"
