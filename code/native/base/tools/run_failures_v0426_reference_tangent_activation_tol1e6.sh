#!/usr/bin/env bash
set -u -o pipefail

main() {
  ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
  BASE="${BASE:-/data/projects/IPDCA_QCQP}"
  CACHE="${PYCUTEST_CACHE:-$BASE/pycutest_cache}"
  BACKEND="${BACKEND:-ldl}"
  QP_BACKEND="${QP_BACKEND:-$BACKEND}"
  MAX_OUTER="${MAX_OUTER:-1000}"
  MAX_INNER="${MAX_INNER:-1000}"
  TIME_LIMIT="${TIME_LIMIT:-1800}"
  LIVE="${LIVE:-1}"
  DCA_STEP_TARGET_ABS="${DCA_STEP_TARGET_ABS:-1.0}"
  DCA_STEP_TARGET_REL="${DCA_STEP_TARGET_REL:-1e-2}"
  DCA_STEP_TARGET_MAX="${DCA_STEP_TARGET_MAX:-10.0}"
  DCA_STEP_CONTROL_RETRIES="${DCA_STEP_CONTROL_RETRIES:-8}"
  RUN_ID="${RUN_ID:-v0426_reference_tangent_activation_${BACKEND}_tol1e6_$(date +%Y%m%d_%H%M%S)}"
  RESULTS="${RESULTS:-$BASE/results/$RUN_ID}"
  LOGS="${LOGS:-$BASE/logs/$RUN_ID}"
  read -r -a CASES <<< "${PROBLEMS:-CATENARY CATENA CORE2}"

  if [[ ! -x "$BIN" ]]; then
    echo "[FAIL] binary missing: $BIN" >&2
    return 2
  fi
  BACKENDS=$($BIN --linear-backends)
  if [[ ",$BACKENDS," != *",$BACKEND,"* || ",$BACKENDS," != *",$QP_BACKEND,"* ]]; then
    echo "[FAIL] backend mismatch: compiled=$BACKENDS requested=$BACKEND qp=$QP_BACKEND" >&2
    return 2
  fi
  mkdir -p "$RESULTS" "$LOGS" "$CACHE"
  unset LD_PRELOAD
  unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
  export PYCUTEST_CACHE="$CACHE"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1

  cat <<BANNER
======================================================================
LazyIPDCA-QCQP v0.4.26 — tangent-activation targeted validation
version         : $($BIN --version)
problems        : ${CASES[*]}
route           : lazy-dca only
backend         : $BACKEND + METIS; LazyQP=$QP_BACKEND
live output     : $LIVE (1=terminal + log, 0=log only)
final tolerances: opt/dual/feas/comp = 1e-6
inner target    : absolute 1e-8; mixed relative acceptance capped at 1e-7
Phase I         : anchored harmonic start + objective tangent activation
pathology guard : one-anchor flat quadratic components receive a graph-distance ramp
first majorant  : stationarity-based proximal predictor + solved-step guard
multiplier path : original-Jacobian sparse row-space refit after accepted steps
results         : $RESULTS
logs            : $LOGS
======================================================================
BANNER

  for P in "${CASES[@]}"; do
    CASE_DIR="$RESULTS/$P"
    LOG="$LOGS/$P.log"
    JSON="$CASE_DIR/$P.json"
    mkdir -p "$CASE_DIR"
    if [[ -s "$JSON" ]]; then cp -f "$JSON" "$JSON.before_v0426" 2>/dev/null || true; fi

    echo
    echo "===== $P ====="
    echo "log     : $LOG"

    CMD=(python -u "$ROOT/python/run_cutest_qcqp.py" "$P"
      --binary "$BIN" --work-dir "$CASE_DIR" --cache "$CACHE" --keep-model
      --reformulation auto --algorithm lazy-dca --inner-backend lazyqp
      --linear-backend "$BACKEND" --linear-metis 1
      --qp-option C --qp-amrs --qp-init 2 --qp-backend "$QP_BACKEND"
      --qp-rho-trial-every 2 --qp-max-atoms 500 --qp-seed-coverage 0.25
      --qp-seed-stall-tol 0 --qp-seed-patience 999 --qp-atom-events
      --qp-atom-event-every 50 --qp-bound-push 0 --qp-bp-push-kappa 1e-4
      --qp-bp-margin-tol 1e-4 --qp-escapes 3 --qp-mehrotra --qp-sigma-max 1e14
      --qp-model-solved-tol 1e-12 --qp-dca-stop --qp-dca-stall-tol-rel 1e-10
      --qp-dca-stall-iters 5 --qp-dca-monotone-tol-abs 1e-6 --qp-bp-adaptive
      --kkt-adaptive-reg --kkt-persistent-symbolic-cache
      --lazyqp-kkt-form full --lazyqp-low-rank-form auto
      --lazyqp-condense-threshold 256 --inner-full-dual-warm-start
      --lazyqp-warm-guard 2.0 --max-outer "$MAX_OUTER" --max-inner "$MAX_INNER"
      --time-limit "$TIME_LIMIT" --feas-tol 1e-6 --opt-tol 1e-6
      --dual-inf-tol 1e-6 --compl-inf-tol 1e-6 --inner-tol 1e-8
      --inner-mu-fallback-tol 1e-7 --s-max 100 --restoration-iters 600
      --restoration-restarts 16 --restoration-restart-iters 400
      --restoration-tangent-activation
      --restoration-minimax-iters 80 --restoration-minimax-prox 1e-6
      --prox 1e-6 --dca-extra-prox 0 --dca-step-control
      --dca-step-target-abs "$DCA_STEP_TARGET_ABS"
      --dca-step-target-rel "$DCA_STEP_TARGET_REL"
      --dca-step-target-max "$DCA_STEP_TARGET_MAX"
      --dca-step-control-retries "$DCA_STEP_CONTROL_RETRIES"
      --dca-affine-full-limit 4096
      --dca-corrector --dca-corrector-start 2 --dca-corrector-frequency 1
      --dca-iterated-soc --dca-iterated-soc-iters 12
      --dca-iterated-soc-max-ratio 2.0 --dca-iterated-soc-trigger 1e6
      --dca-warm-failure-limit 3 --dca-warm-cooldown 32 --dca-inner-retries 3
      --dca-inner-retry-prox 1e-6 --dca-equality-funnel 0
      --dca-equality-funnel-auto --dca-equality-funnel-ceiling 1e-2
      --dca-equality-retraction --dca-equality-retraction-max-ratio 2.0
      --activation 1e-4 --constraint-model auto --working-set-drop-slack 2e-4
      --working-set-drop-mu 1e-9 --max-objective-rank 64
      --max-constraint-rank 24 --max-lagrangian-rank 64 --lanczos-steps 40
      --lanczos-starts 2 --exact-eigen-threshold 256 --verbose 1)

    if [[ "$LIVE" == "1" ]]; then
      "${CMD[@]}" 2>&1 | tee "$LOG"
      rc=${PIPESTATUS[0]}
    else
      "${CMD[@]}" >"$LOG" 2>&1
      rc=$?
      echo "monitor : tail -F '$LOG'"
    fi

    if [[ -s "$JSON" ]]; then
      python - "$JSON" "$rc" <<'PY'
import json,sys
D=json.load(open(sys.argv[1])); rc=sys.argv[2]
def v(k,alt=None):
    x=D.get(k)
    if x is None and alt: x=D.get(alt)
    return x
print("[DONE]", "rc="+rc, "status="+str(v('status')),
      "obj="+str(v('cutest_objective','objective')),
      "feas="+str(v('cutest_feasibility','feasibility')),
      "dual="+str(v('dual_infeasibility_inf')),
      "nlp="+str(v('scaled_nlp_error')),
      "outer="+str(v('outer_iterations')), "inner="+str(v('inner_iterations')),
      "tangent="+str(v('restoration_tangent_activation_acceptances'))+'/'+str(v('restoration_tangent_activation_attempts')),
      "step_predict="+str(v('lazy_dca_step_control_predictions')),
      "step_retry="+str(v('lazy_dca_step_control_retries')),
      "mrefit="+str(v('multiplier_refit_successes'))+'/'+str(v('multiplier_refit_attempts')),
      "mstat="+str(v('last_multiplier_refit_stationarity_inf')),
      "time="+str(v('elapsed_seconds')))
PY
    else
      echo "[DONE] rc=$rc JSON=NO"
    fi
  done

  echo "RESULTS=$RESULTS"
  echo "LOGS=$LOGS"
  echo "RUN_ID=$RUN_ID"
}
main "$@"
