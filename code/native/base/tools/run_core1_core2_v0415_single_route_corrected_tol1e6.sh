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
  PROBLEMS=(CORE1 CORE2)

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
  TAG="v0415_core12_single_route_corrected_${BACKEND}_tol1e6_${STAMP}"
  RESULTS="${RESULTS:-$BASE/results/$TAG}"
  LOGS="${LOGS:-$BASE/logs/$TAG}"
  mkdir -p "$RESULTS" "$LOGS" "$CACHE"

  echo "======================================================================"
  echo "LazyIPDCA-QCQP v0.4.15 — CORE1/CORE2 single-route diagnostic"
  echo "version        : $($BIN --version)"
  echo "backends       : $BACKENDS"
  echo "route          : lazy-dca only"
  echo "equalities     : classical SQP linearization"
  echo "corrector      : enabled"
  echo "inner retries  : cold + model proximal, first prox=1e-6"
  echo "backend        : $BACKEND + METIS"
  echo "tolerances     : opt/dual/feas/comp=1e-6; inner=1e-8"
  echo "results        : $RESULTS"
  echo "logs           : $LOGS"
  echo "======================================================================"

  printf "problem\trc\n" > "$RESULTS/return_codes.tsv"
  for P in "${PROBLEMS[@]}"; do
    WORK="$RESULTS/$P"
    LOG="$LOGS/$P.log"
    mkdir -p "$WORK"
    echo
    echo "===== $P ====="

    python -u "$ROOT/python/run_cutest_qcqp.py" "$P" \
      --binary "$BIN" \
      --work-dir "$WORK" \
      --cache "$CACHE" \
      --keep-model \
      --reformulation auto \
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
        '^\[1/2\]|^\[ALGORITHM-ROUTE\]|^\[PHASE-I\]|^\[LAZY-DCA-INNER-RETRY\]|^\[LAZY-DCA-CORRECTOR\]|^\[LAZY-DCA-KKT-CORRECTOR\]|^\[LAZY-DCA\] k=|^status|^message|^objective|^feasibility|^KKT-stop dual|^KKT-stop complement|^KKT-stop scaled|^KKT-stop desired|^Lazy-DCA corrector|^Lazy-DCA inner retry|^Lazy-DCA max model proximal|^outer / inner|^KKT factorizations|^KKT symbolic|^elapsed|^\[RESULT\]'

    rc=${PIPESTATUS[0]}
    printf "%s\t%s\n" "$P" "$rc" >> "$RESULTS/return_codes.tsv"
    JSON="$WORK/$P.json"
    python - "$P" "$rc" "$JSON" <<'PY'
import json, sys
p, rc, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    d=json.load(open(path))
except Exception as e:
    print(f"[DONE] problem={p} rc={rc} JSON=NO error={e}")
else:
    print(
        f"[DONE] problem={p} rc={rc} status={d.get('status')}"
        f" route={d.get('effective_outer_algorithm')}"
        f" obj={d.get('cutest_objective', d.get('objective'))}"
        f" feas={d.get('cutest_feasibility', d.get('feasibility'))}"
        f" dual={d.get('dual_infeasibility_inf')}"
        f" comp={d.get('complementarity_inf')}"
        f" nlp={d.get('scaled_nlp_error')}"
        f" corrector={d.get('lazy_dca_corrector_acceptances')}/{d.get('lazy_dca_corrector_attempts')}"
        f" retries={d.get('lazy_dca_inner_retries')}"
        f" cold={d.get('lazy_dca_cold_retries')}"
        f" prox={d.get('lazy_dca_proximal_retries')}"
        f" maxprox={d.get('lazy_dca_max_model_proximal')}"
        f" outer={d.get('outer_iterations')} inner={d.get('inner_iterations')}"
        f" time={d.get('elapsed_seconds')}"
    )
PY
  done

  echo
  echo "===== CORE1/CORE2 TERMINÉ ====="
  echo "RESULTS=$RESULTS"
  echo "LOGS=$LOGS"
}

main "$@"
