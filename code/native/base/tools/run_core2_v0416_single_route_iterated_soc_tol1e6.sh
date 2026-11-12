#!/usr/bin/env bash
set -u -o pipefail

main() {
  ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
  CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
  BASE="${BASE:-/data/projects/IPDCA_QCQP}"
  BACKEND="${BACKEND:-ldl}"
  MAX_OUTER="${MAX_OUTER:-1000}"
  MAX_INNER="${MAX_INNER:-1000}"
  TIME_LIMIT="${TIME_LIMIT:-1800}"
  VERBOSE="${VERBOSE:-1}"
  SOC_ITERS="${SOC_ITERS:-12}"
  SOC_MAX_RATIO="${SOC_MAX_RATIO:-2.0}"
  SOC_TRIGGER="${SOC_TRIGGER:-1e6}"
  WARM_FAILURE_LIMIT="${WARM_FAILURE_LIMIT:-3}"
  WARM_COOLDOWN="${WARM_COOLDOWN:-32}"

  if [[ ! -x "$BIN" ]]; then
    echo "[FAIL] binary not found or not executable: $BIN" >&2
    return 2
  fi
  BACKENDS=$($BIN --linear-backends)
  if [[ ",$BACKENDS," != *",$BACKEND,"* ]]; then
    echo "[FAIL] requested backend '$BACKEND' is not compiled: $BACKENDS" >&2
    return 2
  fi

  unset LD_PRELOAD
  unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
  export PYCUTEST_CACHE="$CACHE"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1

  STAMP=$(date +%Y%m%d_%H%M%S)
  TAG="v0416_core2_single_route_iterated_soc_${BACKEND}_tol1e6_${STAMP}"
  WORK="${WORK:-$BASE/results/$TAG/CORE2}"
  LOGDIR="${LOGDIR:-$BASE/logs/$TAG}"
  LOG="$LOGDIR/CORE2.log"
  mkdir -p "$WORK" "$LOGDIR" "$CACHE"

  cat <<BANNER
======================================================================
LazyIPDCA-QCQP v0.4.16 — CORE2 targeted test
version          : $($BIN --version)
compiled linear  : $BACKENDS
outer route      : lazy-dca only
quadratic eq.    : classical SQP linearization
SOC              : one-shot + guarded iterated normal correction
SOC iters/ratio  : $SOC_ITERS / $SOC_MAX_RATIO
SOC trigger      : $SOC_TRIGGER x current feasibility scale
warm guard       : $WARM_FAILURE_LIMIT failures -> $WARM_COOLDOWN cold outer steps
backend          : $BACKEND + METIS
final tolerances : opt/dual/feas/comp=1e-6; inner=1e-8
limits           : outer=$MAX_OUTER inner=$MAX_INNER time=${TIME_LIMIT}s
work             : $WORK
log              : $LOG
======================================================================
BANNER

  python -u "$ROOT/python/run_cutest_qcqp.py" CORE2 \
    --binary "$BIN" \
    --work-dir "$WORK" \
    --cache "$CACHE" \
    --keep-model \
    --reformulation core-xabs \
    --algorithm lazy-dca \
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
    --dca-affine-full-limit 4096 \
    --dca-corrector \
    --dca-corrector-start 2 \
    --dca-corrector-frequency 1 \
    --dca-iterated-soc \
    --dca-iterated-soc-iters "$SOC_ITERS" \
    --dca-iterated-soc-max-ratio "$SOC_MAX_RATIO" \
    --dca-iterated-soc-trigger "$SOC_TRIGGER" \
    --dca-warm-failure-limit "$WARM_FAILURE_LIMIT" \
    --dca-warm-cooldown "$WARM_COOLDOWN" \
    --dca-inner-retries 3 \
    --dca-inner-retry-prox 1e-6 \
    --dca-equality-funnel 0 \
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
    --verbose "$VERBOSE" 2>&1 \
  | tee "$LOG" \
  | grep --line-buffered -E \
      '^\[RUNNER\]|^\[1/2\]|^\[ALGORITHM-ROUTE\]|^\[PHASE-I\]|^\[LAZY-DCA-ITERATED-SOC\]|^\[LAZY-DCA-SOC\]|^\[LAZY-DCA-WARM-GUARD\]|^\[LAZY-DCA-INNER-RETRY\]|^\[LAZY-DCA-CORRECTOR-RETRY\]|^\[LAZY-DCA-CORRECTOR\]|^\[LAZY-DCA\] k=|^status|^message|^objective|^feasibility|^KKT-stop dual|^KKT-stop complement|^KKT-stop scaled|^KKT-stop desired|^Lazy-DCA last equality alpha/funnel|^Lazy-DCA corrector|^Lazy-DCA iterated SOC|^Lazy-DCA warm suppress|^Lazy-DCA inner retry|^outer / inner|^KKT factorizations|^KKT symbolic|^elapsed|^\[RESULT\]'

  rc=${PIPESTATUS[0]}
  JSON="$WORK/CORE2.json"
  echo
  python - "$rc" "$JSON" <<'PY'
import json, sys
rc=int(sys.argv[1]); path=sys.argv[2]
try:
    d=json.load(open(path))
except Exception as exc:
    print(f"[DONE] CORE2 rc={rc} JSON=NO error={exc}")
else:
    print(
        f"[DONE] CORE2 rc={rc} status={d.get('status')}"
        f" route={d.get('effective_outer_algorithm')}"
        f" obj={d.get('cutest_objective', d.get('objective'))}"
        f" feas={d.get('cutest_feasibility', d.get('feasibility'))}"
        f" dual={d.get('dual_infeasibility_inf')}"
        f" comp={d.get('complementarity_inf')}"
        f" nlp={d.get('scaled_nlp_error')}"
        f" alpha={d.get('lazy_dca_last_equality_step_alpha')}"
        f" isoc={d.get('lazy_dca_iterated_soc_acceptances')}/{d.get('lazy_dca_iterated_soc_attempts')}"
        f" corrector={d.get('lazy_dca_corrector_acceptances')}/{d.get('lazy_dca_corrector_attempts')}"
        f" corr_cold={d.get('lazy_dca_corrector_cold_retries')}"
        f" warm_guard={d.get('lazy_dca_warm_start_suppressions')}/{d.get('lazy_dca_warm_start_skips')}"
        f" outer={d.get('outer_iterations')} inner={d.get('inner_iterations')}"
        f" time={d.get('elapsed_seconds')}"
    )
    if d.get('effective_outer_algorithm') != 'lazy-dca':
        print('[CONTRACT-FAIL] route is not lazy-dca')
    else:
        print('[CONTRACT-PASS] single Lazy-DCA route preserved')
PY
  echo "LOG=$LOG"
  echo "JSON=$JSON"
  return "$rc"
}

main "$@"
