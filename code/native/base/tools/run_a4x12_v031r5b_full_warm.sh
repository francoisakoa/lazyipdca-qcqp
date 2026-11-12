#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/data/projects/IPDCA_QCQP/lazyipdca_qcqp_sqp_v0.2.31_r5b_guarded_ipm_init_source}"
BACKEND="${BACKEND:-native}"
CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
TAG="v031r5b_fullwarm_${BACKEND}"
OUT="/data/projects/IPDCA_QCQP/results/${TAG}/A4X12"
LOG="/data/projects/IPDCA_QCQP/logs/${TAG}/A4X12.log"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p "$OUT" "$(dirname "$LOG")"; cd "$ROOT"
python -u python/run_cutest_qcqp.py A4X12 --binary "$ROOT/build/lazyipdca_qcqp_cli" --work-dir "$OUT" --cache "$CACHE" \
 --inner-backend native --linear-backend "$BACKEND" --linear-metis 1 \
 --kkt-primal-reg 1e-12 --kkt-dual-reg 1e-12 --kkt-adaptive-reg --rho-x-floor 1e-12 --rho-x-growth 10 --rho-x-decay 0.5 --rho-x-max 1e-2 --rho-x-clean 3 \
 --delta-dual-base 1e-8 --delta-dual-floor 1e-12 --delta-dual-max 1e-4 --delta-dual-mode inverse --kkt-berr-tol 1e-9 --kkt-berr-reject 1e-7 --kkt-refine 3 \
 --max-outer 1000 --max-inner 1000 --time-limit 900 --feas-tol 1e-7 --opt-tol 1e-6 \
 --inner-full-dual-warm-start --inner-bound-push 1e-2 --inner-bound-frac 1e-2 --inner-bound-push-ref 1e-6 --inner-bound-mult-init 1.0 \
 --terminal-kkt-trigger 1e-4 --no-tangential --terminal-kkt 2>&1 | tee "$LOG"
