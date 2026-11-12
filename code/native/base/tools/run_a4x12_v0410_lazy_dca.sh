#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INNER="${INNER:-lazyqp}"
BACKEND="${BACKEND:-native}"
CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
PERSISTENT_SYMBOLIC_CACHE="${PERSISTENT_SYMBOLIC_CACHE:-1}"
KKT_FORM="${LAZYQP_KKT_FORM:-auto}"
LOW_RANK_FORM="${LAZYQP_LOW_RANK_FORM:-auto}"
CONDENSE_THRESHOLD="${LAZYQP_CONDENSE_THRESHOLD:-256}"
CACHE_TAG="persist"
if [[ "$PERSISTENT_SYMBOLIC_CACHE" != "1" ]]; then CACHE_TAG="nocache"; fi
TAG="v0410_lazy_dca_${INNER}_${BACKEND}_${KKT_FORM}_${LOW_RANK_FORM}_${CACHE_TAG}_tol1e6"
OUT="/data/projects/IPDCA_QCQP/results/${TAG}/A4X12"
LOGDIR="/data/projects/IPDCA_QCQP/logs/${TAG}"
LOG="$LOGDIR/A4X12.log"
mkdir -p "$OUT" "$LOGDIR"

unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYCUTEST_CACHE="$CACHE"

printf '[RUNNER] version='; "$ROOT/build/lazyipdca_qcqp_cli" --version
printf '[RUNNER] algorithm=lazy-dca inner=%s linear=%s\n' "$INNER" "$BACKEND"
CACHE_FLAG="--kkt-persistent-symbolic-cache"
if [[ "$PERSISTENT_SYMBOLIC_CACHE" != "1" ]]; then
  CACHE_FLAG="--no-kkt-persistent-symbolic-cache"
fi
printf '[RUNNER] continuation=guarded-full-dual kkt=%s lowrank=%s condense-threshold=%s persistent-symbolic=%s tol=1e-6\n' \
  "$KKT_FORM" "$LOW_RANK_FORM" "$CONDENSE_THRESHOLD" "$PERSISTENT_SYMBOLIC_CACHE"
printf '[RUNNER] work=%s\n' "$OUT"
printf '[RUNNER] log=%s\n' "$LOG"

python -u "$ROOT/python/run_cutest_qcqp.py" A4X12 \
  --binary "$ROOT/build/lazyipdca_qcqp_cli" \
  --work-dir "$OUT" --cache "$CACHE" \
  --algorithm lazy-dca \
  --inner-backend "$INNER" --linear-backend "$BACKEND" --linear-metis 1 \
  --kkt-adaptive-reg \
  "$CACHE_FLAG" \
  --max-outer "${MAX_OUTER:-1000}" --max-inner "${MAX_INNER:-1000}" \
  --time-limit "${TIME_LIMIT:-900}" \
  --feas-tol "${FEAS_TOL:-1e-6}" \
  --opt-tol "${OPT_TOL:-1e-6}" \
  --dual-inf-tol "${DUAL_TOL:-1e-6}" \
  --compl-inf-tol "${COMP_TOL:-1e-6}" \
  --inner-tol "${INNER_TOL:-1e-8}" \
  --inner-mu-fallback-tol "${INNER_MU_FALLBACK_TOL:-1e-7}" \
  --lazyqp-mu0 "${LAZYQP_MU0:-1e-4}" \
  --lazyqp-mu-r "${LAZYQP_MU_R:-0.1}" \
  --lazyqp-mu-theta "${LAZYQP_MU_THETA:-0.1}" \
  --lazyqp-mu-inner-max "${LAZYQP_MU_INNER_MAX:-0}" \
  --lazyqp-kkt-form "$KKT_FORM" \
  --lazyqp-low-rank-form "$LOW_RANK_FORM" \
  --lazyqp-condense-threshold "$CONDENSE_THRESHOLD" \
  --lazyqp-reduced-kkt-min-ineq "${LAZYQP_REDUCED_KKT_MIN_INEQ:-32}" \
  --lazyqp-warm-guard "${LAZYQP_WARM_GUARD:-2.0}" \
  --inner-full-dual-warm-start \
  --dca-extra-prox "${DCA_EXTRA_PROX:-0}" \
  --dca-affine-full-limit "${DCA_AFFINE_FULL_LIMIT:-4096}" \
  --activation "${ACTIVATION:-1e-4}" \
  --working-set-drop-slack "${WS_DROP_SLACK:-2e-4}" \
  --working-set-drop-mu "${WS_DROP_MU:-1e-9}" \
  --max-objective-rank "${MAX_OBJECTIVE_RANK:-64}" \
  --max-constraint-rank "${MAX_CONSTRAINT_RANK:-24}" \
  --max-lagrangian-rank "${MAX_LAGRANGIAN_RANK:-64}" \
  --lanczos-steps "${LANCZOS_STEPS:-40}" \
  --lanczos-starts "${LANCZOS_STARTS:-2}" \
  --exact-eigen-threshold "${EXACT_EIGEN_THRESHOLD:-256}" \
  --verbose 1 2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
printf '[RUNNER] solver_rc=%s\n' "$rc"
