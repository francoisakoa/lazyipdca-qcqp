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

unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export PYCUTEST_CACHE="$CACHE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="v0414_cutest8_single_route_lazy_dca_${BACKEND}_tol1e6_${STAMP}"
RESULTS="${RESULTS:-$BASE/results/$TAG}"
LOGS="${LOGS:-$BASE/logs/$TAG}"
mkdir -p "$RESULTS" "$LOGS" "$CACHE"
printf "%s\n" "${PROBLEMS[@]}" > "$RESULTS/manifest.txt"
printf "problem\trc\n" > "$RESULTS/return_codes.tsv"

cat <<BANNER
======================================================================
LazyIPDCA-QCQP v0.4.14 — 8 QCQP, single Lazy-DCA route
version        : $($BIN --version)
compiled linear: $BACKENDS
outer algorithm: lazy-dca only; no structure-dependent routing
equalities     : classical SQP linearization h(xk)+grad h(xk)^T d=0
inequalities   : lazy convex DC majorants
backend        : $BACKEND + METIS
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
      '^\[RUNNER\]|^\[EXPORT-REJECTED\]|^\[1/2\]|^\[ALGORITHM-ROUTE\]|^\[PHASE-I\]|^\[LAZY-DCA-SOC\]|^\[LAZY-DCA\] k=|^status|^message|^objective|^feasibility|^KKT-stop dual|^KKT-stop complement|^KKT-stop scaled|^KKT-stop fit|^KKT-stop terminal rescue|^KKT-stop desired|^Lazy-DCA linearized equality steps|^Lazy-DCA last equality alpha|^outer / inner|^KKT factorizations|^KKT symbolic|^elapsed|^\[RESULT\]'

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
