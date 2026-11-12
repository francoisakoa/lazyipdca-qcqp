#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${IPDCA_QCQP_BASE:-/data/projects/IPDCA_QCQP}"
BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
CACHE="${PYCUTEST_CACHE:-$BASE/pycutest_cache}"
RUN_ID="${RUN_ID:-V0428R3P3_CATENA_VERIFIED_SCALAR_SAFEGUARD_$(date +%Y%m%d_%H%M%S)}"
OUT="$BASE/results/$RUN_ID/CATENA"
LOG="$BASE/logs/$RUN_ID/CATENA.log"
JSON="$OUT/CATENA.json"
MODEL="$OUT/CATENA.qcq"

unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export PYCUTEST_CACHE="$CACHE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

if [[ ! -x "$BIN" ]]; then
  echo "[FAIL] solver binary missing or not executable: $BIN" >&2
  echo "Build the source tree before running this script." >&2
  exit 2
fi
SOLVER_VERSION="$($BIN --version 2>/dev/null || true)"
if [[ "$SOLVER_VERSION" != *"0.4.28r3p3-verified-scalar-safeguard"* ]]; then
  echo "[FAIL] unexpected solver build: ${SOLVER_VERSION:-<no version output>}" >&2
  echo "Expected 0.4.28r3p3-verified-scalar-safeguard from: $ROOT" >&2
  exit 2
fi
if ! python -c 'import pycutest' >/dev/null 2>&1; then
  echo "[FAIL] pycutest is not importable in the active Python environment." >&2
  echo "Activate the CUTEst-enabled pf_gpu environment before running this script." >&2
  exit 2
fi
if ! "$BIN" --linear-backends 2>/dev/null | tr ',' '\n' | sed 's/[[:space:]]//g' | grep -qx 'ldl'; then
  echo "[FAIL] the required SuiteSparse LDL backend is not compiled in $BIN." >&2
  "$BIN" --linear-backends >&2 || true
  exit 2
fi

mkdir -p "$OUT" "$(dirname "$LOG")" "$CACHE"
rm -f "$JSON" "$MODEL"

printf '%s\n' '======================================================================'
printf 'Real CUTEst problem : CATENA\n'
printf 'solver              : %s\n' "$SOLVER_VERSION"
printf 'source              : %s\n' "$ROOT"
printf 'output              : %s\n' "$OUT"
printf 'log                 : %s\n' "$LOG"
printf 'PyCUTEst cache       : %s\n' "$CACHE"
printf 'equality route       : v0.4.28 condensed retraction + rank-8 model + verified scalar curvature safeguard\n'
printf 'fast affine Y reuse  : %s\n' "${FAST_AFFINE_REUSE:-1}"
printf 'fast quadratic LS    : %s\n' "${FAST_QUADRATIC_LS:-1}"
printf 'sparse equalities    : %s\n' "${SPARSE_EQUALITIES:-1}"
printf 'cached normal reg    : %s\n' "${CACHED_REG:-1}"
printf 'parallel equalities  : %s (OMP_NUM_THREADS=%s)\n' "${PARALLEL_EQ:-0}" "$OMP_NUM_THREADS"
printf '%s\n' '======================================================================'

CMD=(
  python -u "$ROOT/python/run_cutest_qcqp.py" CATENA
  --binary "$BIN"
  --work-dir "$OUT"
  --cache "$CACHE"
  --keep-model
  --reformulation auto
  --algorithm lazy-dca
  --inner-backend lazyqp
  --linear-backend ldl
  --qp-backend ldl
  --linear-metis 1
  --qp-option C
  --qp-amrs
  --qp-init 2
  --qp-rho-trial-every 2
  --qp-max-atoms 500
  --qp-seed-coverage 0.25
  --qp-seed-stall-tol 0
  --qp-seed-patience 999
  --qp-atom-events
  --qp-atom-event-every 50
  --qp-bound-push 0
  --qp-bp-push-kappa 1e-4
  --qp-bp-margin-tol 1e-4
  --qp-escapes 3
  --qp-mehrotra
  --qp-sigma-max 1e14
  --qp-model-solved-tol 1e-12
  --qp-dca-stop
  --qp-dca-stall-tol-rel 1e-10
  --qp-dca-stall-iters 5
  --qp-dca-monotone-tol-abs 1e-6
  --qp-bp-adaptive
  --kkt-adaptive-reg
  --kkt-persistent-symbolic-cache
  --lazyqp-kkt-form full
  --lazyqp-low-rank-form bordered
  --lazyqp-condense-threshold 256
  --inner-full-dual-warm-start
  --lazyqp-warm-guard 2
  --max-outer "${MAX_OUTER:-1000}"
  --max-inner "${MAX_INNER:-1000}"
  --time-limit "${TIME_LIMIT:-1800}"
  --feas-tol 1e-6
  --opt-tol 1e-6
  --dual-inf-tol 1e-6
  --compl-inf-tol 1e-6
  --inner-tol 1e-8
  --inner-mu-fallback-tol 1e-7
  --s-max 100
  --restoration-iters 600
  --restoration-restarts 16
  --restoration-restart-iters 400
  --restoration-tangent-activation
  --restoration-minimax-iters 80
  --restoration-minimax-prox 1e-6
  --prox 1e-6
  --dca-extra-prox 0
  --dca-step-control
  --dca-step-target-abs 1
  --dca-step-target-rel 1e-2
  --dca-step-target-max 10
  --dca-step-control-retries 8
  --dca-affine-full-limit 4096
  --no-dca-corrector
  --no-dca-iterated-soc
  --no-dca-equality-retraction
  --no-soc
  --dca-warm-failure-limit 3
  --dca-warm-cooldown 32
  --dca-inner-retries 3
  --dca-inner-retry-prox 1e-6
  --dca-equality-funnel 0
  --dca-equality-funnel-auto
  --dca-equality-funnel-ceiling 1e-2
  --dca-equality-model condensed-retraction
  --dca-condensed-curvature
  --dca-condensed-newton-iters 12
  --dca-condensed-krylov-iters 32
  --dca-condensed-krylov-rtol 1e-6
  --dca-condensed-continuation 32
  --dca-condensed-tol 1e-9
  --dca-condensed-regularization 1e-12
  --dca-condensed-reg-growth 10
  --dca-condensed-reg-max 1e-4
  --dca-condensed-max-ratio 2
  --dca-condensed-lanczos-steps 24
  --dca-condensed-lanczos-starts 2
  --dca-condensed-max-rank 8
  --dca-condensed-scalar-safeguard-updates "${SCALAR_SAFEGUARD_UPDATES:-8}"
  --activation 1e-4
  --constraint-model auto
  --working-set-drop-slack 2e-4
  --working-set-drop-mu 1e-9
  --max-objective-rank 64
  --max-constraint-rank 24
  --max-lagrangian-rank 64
  --lanczos-steps 40
  --lanczos-starts 2
  --exact-eigen-threshold 256
  --verbose 1
 )

if [[ "${FAST_AFFINE_REUSE:-1}" == "1" ]]; then
  CMD+=(--dca-condensed-affine-normal-reuse)
else
  CMD+=(--no-dca-condensed-affine-normal-reuse)
fi
if [[ "${FAST_QUADRATIC_LS:-1}" == "1" ]]; then
  CMD+=(--dca-condensed-quadratic-linesearch-reuse)
else
  CMD+=(--no-dca-condensed-quadratic-linesearch-reuse)
fi
if [[ "${SPARSE_EQUALITIES:-1}" == "1" ]]; then
  CMD+=(--dca-condensed-sparse-equalities)
else
  CMD+=(--no-dca-condensed-sparse-equalities)
fi
if [[ "${CACHED_REG:-1}" == "1" ]]; then
  CMD+=(--dca-condensed-cached-regularization)
else
  CMD+=(--no-dca-condensed-cached-regularization)
fi
if [[ "${PARALLEL_EQ:-0}" == "1" ]]; then
  CMD+=(--dca-condensed-parallel-equalities)
else
  CMD+=(--no-dca-condensed-parallel-equalities)
fi
CMD+=(--dca-condensed-parallel-min-equalities "${PARALLEL_MIN_EQ:-128}")

set +e
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

python - "$JSON" "$RC" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
process_rc = int(sys.argv[2])
errors = []

if process_rc != 0:
    errors.append(f"runner/solver return code is {process_rc}, expected 0")
if not path.is_file():
    errors.append(f"result JSON is missing: {path}")
    data = {}
else:
    data = json.loads(path.read_text())

def require_equal(key, expected):
    value = data.get(key)
    if value != expected:
        errors.append(f"{key}={value!r}, expected {expected!r}")

def require_positive(key):
    value = data.get(key)
    try:
        ok = float(value) > 0.0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        errors.append(f"{key}={value!r}, expected a positive value")

def require_le(key, limit):
    value = data.get(key)
    try:
        numeric = float(value)
        ok = math.isfinite(numeric) and numeric <= limit
    except (TypeError, ValueError):
        ok = False
    if not ok:
        errors.append(f"{key}={value!r}, expected <= {limit:.1e}")

def require_int(key, expected):
    value = data.get(key)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric != expected:
        errors.append(f"{key}={value!r}, expected {expected}")

def require_ge(key, lower):
    value = data.get(key)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is None or numeric < lower:
        errors.append(f"{key}={value!r}, expected >= {lower}")

require_equal("problem", "CATENA")
require_equal("status", "solved")
require_equal("effective_outer_algorithm", "lazy-dca")
require_equal("lazy_dca_equality_model", "condensed-retraction")
require_equal("quadratic_validation_passed", True)
require_equal("ipopt_desired_convergence", True)
require_equal("soc_factorizations", 0)
require_equal("lazy_dca_corrector_attempts", 0)
require_positive("lazy_dca_condensed_factorizations")
require_positive("lazy_dca_condensed_tangent_curvature_calls")
require_positive("lazy_dca_condensed_retraction_acceptances")
require_positive("lazy_dca_condensed_accepted_steps")
require_le("cutest_feasibility", 1e-6)
require_le("dual_infeasibility_inf", 1e-6)
require_le("complementarity_inf", 1e-6)
require_le("scaled_nlp_error", 1e-6)
require_le("lazy_dca_condensed_last_residual_inf", 1e-8)
# This is the regression that distinguishes r3p3 from both failed runs:
# rank 8 must remain small, while the scalar safeguard must be allowed to
# update more than once when an independent Lanczos check exposes another
# negative tangent direction.
require_ge("lazy_dca_condensed_scalar_safeguard_updates_max", 2)

if errors:
    print("\n[FAIL] CATENA condensed-retraction validation")
    for error in errors:
        print("  -", error)
    if data:
        print("  status/message:", data.get("status"), "/", data.get("message"))
    raise SystemExit(1)

print("\n[PASS] CATENA solved with rank-8 corrections and the r3p3 verified scalar safeguard")
print("  objective              =", data.get("cutest_objective"))
print("  CUTEst feasibility     =", data.get("cutest_feasibility"))
print("  dual infeasibility     =", data.get("dual_infeasibility_inf"))
print("  complementarity        =", data.get("complementarity_inf"))
print("  scaled NLP error       =", data.get("scaled_nlp_error"))
print("  accepted manifold steps=", data.get("lazy_dca_condensed_accepted_steps"))
print("  terminal KKT accepts    =", data.get("terminal_kkt_accepted_steps"))
print("  scalar safeguard max upd =", data.get("lazy_dca_condensed_scalar_safeguard_updates_max"))
print("  affine-normal reuses    =", data.get("lazy_dca_condensed_affine_normal_map_reuses"))
print("  quadratic LS reuse/check=", data.get("lazy_dca_condensed_quadratic_linesearch_reuses"), "/", data.get("lazy_dca_condensed_quadratic_linesearch_exact_checks"))
print("  quadratic LS early/skip =", data.get("lazy_dca_condensed_quadratic_linesearch_early_rejections"), "/", data.get("lazy_dca_condensed_quadratic_linesearch_rows_skipped"))
print("  cached reg att/hit      =", data.get("lazy_dca_condensed_cached_regularization_attempts"), "/", data.get("lazy_dca_condensed_cached_regularization_hits"))
print("  sparse equality evals   =", data.get("lazy_dca_condensed_sparse_equality_evaluations"))
print("  parallel equality batch =", data.get("lazy_dca_condensed_parallel_equality_batches"))
print("  normal refinement steps =", data.get("lazy_dca_condensed_normal_refinement_steps"))
print("  refine solves 0/1/2/3+  =", data.get("lazy_dca_condensed_normal_refinement_zero_solves"), "/", data.get("lazy_dca_condensed_normal_refinement_one_solves"), "/", data.get("lazy_dca_condensed_normal_refinement_two_solves"), "/", data.get("lazy_dca_condensed_normal_refinement_three_plus_solves"))
print("  initial berr max/mean   =", data.get("lazy_dca_condensed_max_normal_initial_backward_error"), "/", data.get("lazy_dca_condensed_mean_normal_initial_backward_error"))
print("  final berr max/mean     =", data.get("lazy_dca_condensed_max_normal_backward_error"), "/", data.get("lazy_dca_condensed_mean_normal_final_backward_error"))
print("  GMRES workspace growths =", data.get("lazy_dca_condensed_gmres_workspace_growths"))
print("  zero-rhs solve skips     =", data.get("lazy_dca_condensed_zero_rhs_solve_skips"))
print("  Phase-I profile seconds =", data.get("restoration_harmonic_seconds"), data.get("restoration_tangent_activation_seconds"), data.get("restoration_legacy_gate_seconds"), data.get("restoration_exact_seconds"), data.get("restoration_total_seconds"))
print("  normal fact/solve       =", data.get("lazy_dca_condensed_factorizations"), "/", data.get("lazy_dca_condensed_normal_solves"))
print("  result JSON            =", path)
PY

printf 'MODEL=%s\n' "$MODEL"
printf 'JSON=%s\n' "$JSON"
printf 'LOG=%s\n' "$LOG"
