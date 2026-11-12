#!/usr/bin/env bash
set -u -o pipefail

main() {
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
BASE="${BASE:-/data/projects/IPDCA_QCQP}"

MAX_OUTER="${MAX_OUTER:-1000}"
MAX_INNER="${MAX_INNER:-1000}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
LOW_RANK_FORM="${LOW_RANK_FORM:-auto}"
BACKEND="${BACKEND:-ldl}"
VERBOSE="${VERBOSE:-1}"

QP_OPTION="${QP_OPTION:-C}"
QP_AMRS="${QP_AMRS:-1}"
QP_INIT="${QP_INIT:-2}"
QP_BACKEND="${QP_BACKEND:-$BACKEND}"
QP_RHO_TRIAL_EVERY="${QP_RHO_TRIAL_EVERY:-2}"
QP_MAX_ATOMS="${QP_MAX_ATOMS:-500}"
QP_SEED_COVERAGE="${QP_SEED_COVERAGE:-0.25}"
QP_SEED_STALL_TOL="${QP_SEED_STALL_TOL:-0}"
QP_SEED_PATIENCE="${QP_SEED_PATIENCE:-999}"
QP_ATOM_EVENTS="${QP_ATOM_EVENTS:-1}"
QP_ATOM_EVENT_EVERY="${QP_ATOM_EVENT_EVERY:-50}"
QP_BOUND_PUSH="${QP_BOUND_PUSH:-0}"
QP_BP_PUSH_KAPPA="${QP_BP_PUSH_KAPPA:-1e-4}"
QP_BP_MARGIN_TOL="${QP_BP_MARGIN_TOL:-1e-4}"
QP_ESCAPES="${QP_ESCAPES:-3}"
QP_MEHROTRA="${QP_MEHROTRA:-1}"
QP_SIGMA_MAX="${QP_SIGMA_MAX:-1e14}"
QP_MODEL_SOLVED_TOL="${QP_MODEL_SOLVED_TOL:-1e-12}"
QP_DCA_STOP="${QP_DCA_STOP:-1}"
QP_DCA_STALL_TOL_REL="${QP_DCA_STALL_TOL_REL:-1e-10}"
QP_DCA_STALL_ITERS="${QP_DCA_STALL_ITERS:-5}"
QP_DCA_MONOTONE_TOL_ABS="${QP_DCA_MONOTONE_TOL_ABS:-1e-6}"
QP_BP_ADAPTIVE="${QP_BP_ADAPTIVE:-1}"

PROBLEMS=(BT1 GOTTFR HIMMELBD DISCS HS116 A4X12 CORE1 CORE2)

if [[ ! -x "$BIN" ]]; then
  echo "[FAIL] binary not found or not executable: $BIN" >&2
  echo "Compile the source tree before running this campaign." >&2
  return 2
fi

BACKENDS=$($BIN --linear-backends)
if [[ ",$BACKENDS," != *",$BACKEND,"* ]]; then
  echo "[FAIL] requested backend '$BACKEND' is not compiled: $BACKENDS" >&2
  return 2
fi
if [[ ",$BACKENDS," != *",$QP_BACKEND,"* ]]; then
  echo "[FAIL] requested LazyQP backend '$QP_BACKEND' is not compiled: $BACKENDS" >&2
  return 2
fi

unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export PYCUTEST_CACHE="$CACHE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="v0419_cutest8_all_constraint_defect_soc_qp_controls_${BACKEND}_tol1e6_${STAMP}"
RESULTS="${RESULTS:-$BASE/results/$TAG}"
LOGS="${LOGS:-$BASE/logs/$TAG}"
mkdir -p "$RESULTS" "$LOGS" "$CACHE"
printf "%s\n" "${PROBLEMS[@]}" > "$RESULTS/manifest.txt"
printf "problem\trc\n" > "$RESULTS/return_codes.tsv"

QP_BOOL_ARGS=()
[[ "$QP_AMRS" == "1" ]] && QP_BOOL_ARGS+=(--qp-amrs) || QP_BOOL_ARGS+=(--no-qp-amrs)
[[ "$QP_ATOM_EVENTS" == "1" ]] && QP_BOOL_ARGS+=(--qp-atom-events) || QP_BOOL_ARGS+=(--no-qp-atom-events)
[[ "$QP_MEHROTRA" == "1" ]] && QP_BOOL_ARGS+=(--qp-mehrotra) || QP_BOOL_ARGS+=(--no-qp-mehrotra)
[[ "$QP_DCA_STOP" == "1" ]] && QP_BOOL_ARGS+=(--qp-dca-stop) || QP_BOOL_ARGS+=(--no-qp-dca-stop)
[[ "$QP_BP_ADAPTIVE" == "1" ]] && QP_BOOL_ARGS+=(--qp-bp-adaptive) || QP_BOOL_ARGS+=(--no-qp-bp-adaptive)

cat <<BANNER
======================================================================
LazyIPDCA-QCQP v0.4.19 — 8 QCQP, scaled all-constraint defect SOC + qp_ controls
version        : $($BIN --version)
compiled linear: $BACKENDS
outer algorithm: lazy-dca only; no structure-dependent routing
equalities     : classical SQP linearization h(xk)+grad h(xk)^T d=0
inequalities   : lazy convex DC majorants; guarded local corrector after the DCA step
SOC equality   : h(z)+Jh p = h(x)+Jh(z-x)
SOC inequality : g(z)+Jg p <= max(0,g(x)+Jg(z-x))
backend        : $BACKEND + METIS
qp_ profile     : option=$QP_OPTION AMRS=$QP_AMRS init=$QP_INIT backend=$QP_BACKEND
qp_ parameters  : max_atoms=$QP_MAX_ATOMS seed=$QP_SEED_COVERAGE bound_push=$QP_BOUND_PUSH Mehrotra=$QP_MEHROTRA
qp_ DC controls : exposed/recorded; inactive for already-convex LazyQP subproblems
KKT policy     : full, persistent symbolic cache
validation     : mandatory multi-scale quadratic reconstruction
reformulation  : auto (CORE1/CORE2 -> exact core-xabs QCQP lift)
tolerances     : opt/dual/feas/comp = 1e-6; inner = 1e-8
limits         : outer=$MAX_OUTER inner=$MAX_INNER time=${TIME_LIMIT}s/problem
results        : $RESULTS
logs           : $LOGS
======================================================================
BANNER

for i in "${!PROBLEMS[@]}"; do
  P="${PROBLEMS[$i]}"
  CASE_DIR="$RESULTS/$P"
  LOG="$LOGS/$P.log"
  mkdir -p "$CASE_DIR"

  echo
  echo "######################################################################"
  echo "[$((i+1))/${#PROBLEMS[@]}] $P"
  echo "log    : $LOG"
  echo "result : $CASE_DIR/$P.json"
  echo "######################################################################"

  python -u "$ROOT/python/run_cutest_qcqp.py" "$P" \
    --binary "$BIN" \
    --work-dir "$CASE_DIR" \
    --cache "$CACHE" \
    --keep-model \
    --reformulation auto \
    --algorithm lazy-dca \
    --inner-backend lazyqp \
    --linear-backend "$BACKEND" \
    --qp-option "$QP_OPTION" \
    --qp-init "$QP_INIT" \
    --qp-backend "$QP_BACKEND" \
    --qp-rho-trial-every "$QP_RHO_TRIAL_EVERY" \
    --qp-max-atoms "$QP_MAX_ATOMS" \
    --qp-seed-coverage "$QP_SEED_COVERAGE" \
    --qp-seed-stall-tol "$QP_SEED_STALL_TOL" \
    --qp-seed-patience "$QP_SEED_PATIENCE" \
    --qp-atom-event-every "$QP_ATOM_EVENT_EVERY" \
    --qp-bound-push "$QP_BOUND_PUSH" \
    --qp-bp-push-kappa "$QP_BP_PUSH_KAPPA" \
    --qp-bp-margin-tol "$QP_BP_MARGIN_TOL" \
    --qp-escapes "$QP_ESCAPES" \
    --qp-sigma-max "$QP_SIGMA_MAX" \
    --qp-model-solved-tol "$QP_MODEL_SOLVED_TOL" \
    --qp-dca-stall-tol-rel "$QP_DCA_STALL_TOL_REL" \
    --qp-dca-stall-iters "$QP_DCA_STALL_ITERS" \
    --qp-dca-monotone-tol-abs "$QP_DCA_MONOTONE_TOL_ABS" \
    --linear-metis 1 \
    --kkt-adaptive-reg \
    --kkt-persistent-symbolic-cache \
    --lazyqp-kkt-form full \
    --lazyqp-low-rank-form "$LOW_RANK_FORM" \
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
    --dca-iterated-soc-iters 12 \
    --dca-iterated-soc-max-ratio 2.0 \
    --dca-iterated-soc-trigger 1e6 \
    --dca-warm-failure-limit 3 \
    --dca-warm-cooldown 32 \
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
    --verbose "$VERBOSE" \
    "${QP_BOOL_ARGS[@]}" 2>&1 \
  | tee "$LOG" \
  | grep --line-buffered -E \
      '^\[RUNNER\]|^\[EXPORT-REJECTED\]|^\[1/2\]|^\[ALGORITHM-ROUTE\]|^\[PHASE-I\]|^\[LAZY-DCA-ITERATED-SOC\]|^\[LAZY-DCA-SOC\]|^\[LAZY-DCA-WARM-GUARD\]|^\[LAZY-DCA-CORRECTOR-RETRY\]|^\[LAZY-DCA-CORRECTOR\]|^\[LAZY-DCA-KKT-CORRECTOR\]|^\[LAZY-DCA-INNER-RETRY\]|^\[LAZY-DCA\] k=|^status|^message|^objective|^feasibility|^KKT-stop dual|^KKT-stop complement|^KKT-stop scaled|^KKT-stop fit|^KKT-stop terminal rescue|^KKT-stop desired|^Lazy-DCA linearized equality steps|^Lazy-DCA equality f/h steps|^Lazy-DCA last equality alpha/funnel|^Lazy-DCA corrector|^Lazy-DCA inner retry|^outer / inner|^KKT factorizations|^KKT symbolic|^elapsed|^\[RESULT\]'

  rc=${PIPESTATUS[0]}
  printf "%s\t%s\n" "$P" "$rc" >> "$RESULTS/return_codes.tsv"

  JSON="$CASE_DIR/$P.json"
  if [[ -f "$JSON" ]]; then
    python - "$P" "$rc" "$JSON" <<'PY'
import json, sys
problem, rc, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = json.load(open(path))
print(
    "[DONE]"
    f" problem={problem} rc={rc} status={data.get('status')}"
    f" route={data.get('effective_outer_algorithm')}"
    f" obj={data.get('cutest_objective', data.get('objective'))}"
    f" feas={data.get('cutest_feasibility', data.get('feasibility'))}"
    f" dual={data.get('dual_infeasibility_inf')}"
    f" comp={data.get('complementarity_inf')}"
    f" nlp={data.get('scaled_nlp_error')}"
    f" eqsteps={data.get('lazy_dca_linearized_equality_steps')}"
    f" eqalpha={data.get('lazy_dca_last_equality_step_alpha')}"
    f" corr={data.get('lazy_dca_corrector_acceptances')}/{data.get('lazy_dca_corrector_attempts')}"
    f" isoc={data.get('lazy_dca_iterated_soc_acceptances')}/{data.get('lazy_dca_iterated_soc_attempts')}"
    f" warmguard={data.get('lazy_dca_warm_start_suppressions')}/{data.get('lazy_dca_warm_start_skips')}"
    f" socinner={data.get('soc_inner_iterations')} socfact={data.get('soc_factorizations')}"
    f" retries={data.get('lazy_dca_inner_retries')}"
    f" outer={data.get('outer_iterations')} inner={data.get('inner_iterations')}"
    f" time={data.get('elapsed_seconds')}"
)
PY
  else
    echo "[DONE] problem=$P rc=$rc JSON=NO"
  fi
done

python - "$RESULTS" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
problems = [x.strip() for x in (root / "manifest.txt").read_text().splitlines() if x.strip()]
rcs = {}
for line in (root / "return_codes.tsv").read_text().splitlines()[1:]:
    fields = line.split("\t")
    if len(fields) == 2:
        rcs[fields[0]] = fields[1]

rows = []
for p in problems:
    path = root / p / f"{p}.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    rows.append({
        "problem": p,
        "rc": rcs.get(p),
        "status": data.get("status", "missing"),
        "route": data.get("effective_outer_algorithm"),
        "reformulation": data.get("reformulation"),
        "quadratic_gate": data.get("quadratic_validation_passed"),
        "objective": data.get("cutest_objective", data.get("objective")),
        "feasibility": data.get("cutest_feasibility", data.get("feasibility")),
        "dual": data.get("dual_infeasibility_inf"),
        "complementarity": data.get("complementarity_inf"),
        "scaled_nlp": data.get("scaled_nlp_error"),
        "kkt_pass": data.get("ipopt_desired_convergence"),
        "eq_steps": data.get("lazy_dca_linearized_equality_steps"),
        "eq_alpha": data.get("lazy_dca_last_equality_step_alpha"),
        "corrector_accept": data.get("lazy_dca_corrector_acceptances"),
        "corrector_attempt": data.get("lazy_dca_corrector_attempts"),
        "inner_retries": data.get("lazy_dca_inner_retries"),
        "prox_retries": data.get("lazy_dca_proximal_retries"),
        "soc_inner": data.get("soc_inner_iterations"),
        "soc_fact": data.get("soc_factorizations"),
        "outer": data.get("outer_iterations"),
        "inner": data.get("inner_iterations"),
        "kkt_fact": data.get("kkt_factorizations"),
        "symbolic": data.get("kkt_symbolic_analyses"),
        "fit_reg": data.get("termination_multiplier_regularization"),
        "fit_tries": data.get("termination_multiplier_refinement_attempts"),
        "terminal_rescue": data.get("termination_certificate_rescued"),
        "elapsed_s": data.get("elapsed_seconds"),
        "message": data.get("message"),
    })

summary = root / "summary.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

def display(v, width=13, boolean_mode="pass"):
    if v is None:
        return "-".rjust(width)
    if isinstance(v, bool):
        if boolean_mode == "yes":
            return ("YES" if v else "NO").rjust(width)
        return ("PASS" if v else "FAIL").rjust(width)
    if isinstance(v, float):
        return f"{v:.6e}".rjust(width)
    return str(v).rjust(width)

columns = [
    ("PROBLEM", "problem", 9, "text"),
    ("RC", "rc", 3, "text"),
    ("STATUS", "status", 21, "text"),
    ("ROUTE", "route", 10, "text"),
    ("REFORM", "reformulation", 24, "text"),
    ("OBJECTIVE", "objective", 13, "float"),
    ("FEASIBILITY", "feasibility", 13, "float"),
    ("DUAL", "dual", 13, "float"),
    ("COMP", "complementarity", 13, "float"),
    ("SCALED NLP", "scaled_nlp", 13, "float"),
    ("KKT", "kkt_pass", 5, "pass"),
    ("EQSTEP", "eq_steps", 6, "text"),
    ("EQALPHA", "eq_alpha", 11, "float"),
    ("CORR", "corrector_accept", 6, "text"),
    ("CATT", "corrector_attempt", 6, "text"),
    ("RETRY", "inner_retries", 6, "text"),
    ("PRETRY", "prox_retries", 6, "text"),
    ("SOCIN", "soc_inner", 7, "text"),
    ("SOCFAC", "soc_fact", 7, "text"),
    ("OUTER", "outer", 6, "text"),
    ("INNER", "inner", 8, "text"),
    ("KKTFACT", "kkt_fact", 8, "text"),
    ("SYMB", "symbolic", 5, "text"),
    ("FITREG", "fit_reg", 11, "float"),
    ("TRIES", "fit_tries", 5, "text"),
    ("RESCUE", "terminal_rescue", 6, "yes"),
    ("TIME(s)", "elapsed_s", 11, "float"),
]

header = " ".join(name.rjust(width) for name, _, width, _ in columns)
sep = "=" * len(header)
print()
print(sep)
print(header)
print(sep)
for row in rows:
    fields = []
    for _, key, width, mode in columns:
        if mode == "yes":
            fields.append(display(row[key], width, "yes"))
        elif mode == "pass":
            fields.append(display(row[key], width, "pass"))
        else:
            fields.append(display(row[key], width))
    print(" ".join(fields))
print(sep)
print(f"[SUMMARY] {summary}")
print(f"[RESULTS] {root}")

bad_route = [r["problem"] for r in rows if r["route"] not in (None, "lazy-dca")]
if bad_route:
    print(f"[CONTRACT-FAIL] non-Lazy-DCA routes found: {bad_route}")
else:
    print("[CONTRACT-PASS] every executed optimization used route=lazy-dca")
PY

echo
echo "===== CAMPAGNE TERMINÉE ====="
echo "RESULTS=$RESULTS"
echo "LOGS=$LOGS"
echo "SUMMARY=$RESULTS/summary.csv"
}

main "$@"
