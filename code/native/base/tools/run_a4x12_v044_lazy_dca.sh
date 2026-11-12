#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INNER="${INNER:-lazyqp}"
BACKEND="${BACKEND:-native}"
CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
OUT="/data/projects/IPDCA_QCQP/results/v044_lazy_dca_${INNER}_${BACKEND}/A4X12"
LOGDIR="/data/projects/IPDCA_QCQP/logs/v044_lazy_dca_${INNER}_${BACKEND}"
mkdir -p "$OUT" "$LOGDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYCUTEST_CACHE="$CACHE"

printf '[RUNNER] version='; "$ROOT/build/lazyipdca_qcqp_cli" --version
printf '[RUNNER] algorithm=lazy-dca inner=%s linear=%s\n' "$INNER" "$BACKEND"
printf '[RUNNER] work=%s\n' "$OUT"

python -u "$ROOT/python/run_cutest_qcqp.py" A4X12 \
  --binary "$ROOT/build/lazyipdca_qcqp_cli" \
  --work-dir "$OUT" --cache "$CACHE" \
  --algorithm lazy-dca \
  --inner-backend "$INNER" --linear-backend "$BACKEND" --linear-metis 1 \
  --max-outer "${MAX_OUTER:-1000}" --max-inner "${MAX_INNER:-1000}" \
  --time-limit "${TIME_LIMIT:-900}" \
  --feas-tol "${FEAS_TOL:-1e-6}" \
  --opt-tol "${OPT_TOL:-1e-5}" \
  --dual-inf-tol "${DUAL_TOL:-1e-5}" \
  --compl-inf-tol "${COMP_TOL:-1e-5}" \
  --inner-tol "${INNER_TOL:-1e-7}" \
  --inner-mu-fallback-tol "${INNER_MU_FALLBACK_TOL:-1e-6}" \
  --lazyqp-mu0 "${LAZYQP_MU0:-1e-4}" \
  --lazyqp-mu-r "${LAZYQP_MU_R:-0.1}" \
  --lazyqp-mu-theta "${LAZYQP_MU_THETA:-0.1}" \
  --lazyqp-mu-inner-max "${LAZYQP_MU_INNER_MAX:-0}" \
  --no-inner-warm-start \
  --dca-extra-prox "${DCA_EXTRA_PROX:-0}" \
  --activation "${ACTIVATION:-1e-4}" \
  --working-set-drop-slack "${WS_DROP_SLACK:-2e-4}" \
  --working-set-drop-mu "${WS_DROP_MU:-1e-9}" \
  --max-objective-rank "${MAX_OBJECTIVE_RANK:-64}" \
  --max-constraint-rank "${MAX_CONSTRAINT_RANK:-24}" \
  --lanczos-steps "${LANCZOS_STEPS:-40}" \
  --lanczos-starts "${LANCZOS_STARTS:-2}" \
  --exact-eigen-threshold "${EXACT_EIGEN_THRESHOLD:-256}" \
  --verbose 1 2>&1 | tee "$LOGDIR/A4X12.log"
