#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="${ROOT:-/data/projects/IPDCA_QCQP/lazyipdca_qcqp_v0.4.10_lazyqp_verified_assembly_source}"
CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
LOW_RANK_FORM="${LAZYQP_LOW_RANK_FORM:-auto}"
CONDENSE_THRESHOLD="${LAZYQP_CONDENSE_THRESHOLD:-256}"
TAG="v0410_core2_lazyqp_ldl_full_${LOW_RANK_FORM}_persist_tol1e6"
OUT="${OUT:-/data/projects/IPDCA_QCQP/results/${TAG}/CORE2}"
LOGDIR="${LOGDIR:-/data/projects/IPDCA_QCQP/logs/${TAG}}"
LOG="$LOGDIR/CORE2.log"

mkdir -p "$OUT" "$LOGDIR"

unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS

export PYCUTEST_CACHE="$CACHE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

printf '[RUNNER] version='; "$ROOT/build/lazyipdca_qcqp_cli" --version
printf '[RUNNER] problem=CORE2 reformulation=core-xabs algorithm=lazy-dca inner=lazyqp linear=ldl metis=1\n'
printf '[RUNNER] kkt=full lowrank=%s condense-threshold=%s persistent-symbolic=1 warm=guarded-full-dual tol=1e-6\n' \
  "$LOW_RANK_FORM" "$CONDENSE_THRESHOLD"
printf '[RUNNER] work=%s\n' "$OUT"
printf '[RUNNER] log=%s\n' "$LOG"

python -u "$ROOT/python/run_cutest_qcqp.py" CORE2 \
  --binary "$ROOT/build/lazyipdca_qcqp_cli" \
  --work-dir "$OUT" \
  --cache "$CACHE" \
  --keep-model \
  --reformulation core-xabs \
  --algorithm lazy-dca \
  --inner-backend lazyqp \
  --linear-backend ldl \
  --linear-metis 1 \
  --kkt-adaptive-reg \
  --kkt-persistent-symbolic-cache \
  --lazyqp-kkt-form full \
  --lazyqp-low-rank-form "$LOW_RANK_FORM" \
  --lazyqp-condense-threshold "$CONDENSE_THRESHOLD" \
  --inner-full-dual-warm-start \
  --lazyqp-warm-guard "${LAZYQP_WARM_GUARD:-2.0}" \
  --max-outer "${MAX_OUTER:-1000}" \
  --max-inner "${MAX_INNER:-1000}" \
  --time-limit "${TIME_LIMIT:-1800}" \
  --feas-tol "${FEAS_TOL:-1e-6}" \
  --opt-tol "${OPT_TOL:-1e-6}" \
  --dual-inf-tol "${DUAL_TOL:-1e-6}" \
  --compl-inf-tol "${COMP_TOL:-1e-6}" \
  --inner-tol "${INNER_TOL:-1e-8}" \
  --inner-mu-fallback-tol "${INNER_MU_FALLBACK_TOL:-1e-7}" \
  --s-max "${S_MAX:-100}" \
  --restoration-iters "${RESTORATION_ITERS:-600}" \
  --restoration-restarts "${RESTORATION_RESTARTS:-16}" \
  --restoration-restart-iters "${RESTORATION_RESTART_ITERS:-400}" \
  --prox "${PROX:-1e-6}" \
  --dca-extra-prox "${DCA_EXTRA_PROX:-0}" \
  --dca-affine-full-limit "${DCA_AFFINE_FULL_LIMIT:-4096}" \
  --activation "${ACTIVATION:-1e-4}" \
  --working-set-drop-slack "${WS_DROP_SLACK:-2e-4}" \
  --working-set-drop-mu "${WS_DROP_MU:-1e-9}" \
  --max-objective-rank "${MAX_OBJECTIVE_RANK:-64}" \
  --max-constraint-rank "${MAX_CONSTRAINT_RANK:-24}" \
  --max-lagrangian-rank "${MAX_LAGRANGIAN_RANK:-64}" \
  --constraint-model auto \
  --lanczos-steps "${LANCZOS_STEPS:-40}" \
  --lanczos-starts "${LANCZOS_STARTS:-2}" \
  --exact-eigen-threshold "${EXACT_EIGEN_THRESHOLD:-256}" \
  --verbose 1 2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
printf '[RUNNER] solver_rc=%s\n' "$rc"

printf '\n===== CORE2 v0.4.10 LDL+METIS TOL 1e-6 =====\n'
grep -E \
'^\[RUNNER\]|^\[1/2\]|^\[PHASE-I\]|^\[LAZY-DCA\]|^\[LAZYQP-WARM\]|^\[LAZYQP-KKT\]|^linear solver|^persistent symbolic|^problem|^n / eq / ineq|^status|^message|^objective|^feasibility|^model stationarity|^KKT-stop dual|^KKT-stop complement|^KKT-stop scaled|^KKT-stop desired|^KKT-stop tol/dual/cv/comp|^restoration abs links|^restoration eff\. dim|^restoration theta|^outer / inner|^KKT factorizations|^KKT symbolic|^inner warm|^LazyQP reduced|^LazyQP max KKT|^elapsed|^\[RESULT\]' \
  "$LOG" | tail -420 || true

printf '\n[FILES] model=%s/CORE2.qcq\n' "$OUT"
printf '[FILES] json=%s/CORE2.json\n' "$OUT"
printf '[FILES] log=%s\n' "$LOG"
