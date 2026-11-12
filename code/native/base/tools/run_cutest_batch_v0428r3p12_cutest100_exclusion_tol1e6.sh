#!/usr/bin/env bash
set -u -o pipefail

main() {
  ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  BASE="${BASE:-/data/projects/IPDCA_QCQP}"
  BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
  CACHE="${PYCUTEST_CACHE:-$BASE/pycutest_cache}"
  BACKEND="${BACKEND:-ldl}"
  EQUALITY_MODEL="${DCA_EQUALITY_MODEL:-linearized}"
  MAX_OUTER="${MAX_OUTER:-1000}"
  MAX_INNER="${MAX_INNER:-1000}"
  TIME_LIMIT="${TIME_LIMIT:-1800}"
  LIVE_EVERY="${LIVE_EVERY:-1}"
  PHASE1_THREADS="${PHASE1_THREADS:-1}"
  MANIFEST_FILE=""
  PROBLEMS_OPTION_SET=0

  # Exactly 20 QCQPs by default, including HS116, HS1108 and CORE2.  Override
  # either through --problems or the legacy PROBLEMS environment variable.
  PROBLEMS="${PROBLEMS:-A4X12 ALJAZZAF BT1 BT2 BT5 BYRDSPHR CORE1 CORE2 DISC2 DISCS HIMMELBD HS106 HS108 HS1108 HS113 HS116 MARATOS MIFFLIN1 S316-322 ZECEVIC4}"

  usage() {
    cat <<'EOF'
Usage: bash tools/run_cutest_batch_v0428r3p12_cutest100_exclusion_tol1e6.sh [options]

Options:
  --linear-backend NAME       native | ldl | qdldl | mumps (default: ldl)
  --dca-equality-model NAME   linearized | condensed-retraction | elastic-split
  --problems "P1 P2 ..."      override the default 20-problem list
  --manifest FILE             read one CUTEst problem name per nonblank line
  --live-every N              print one [ITER] line every N accepted iterations
  --phase1-threads N          persistent OpenMP team for Phase-I row dots (default: 1)
  --help                      display this help

Examples:
  bash tools/run_cutest_batch_v0428r3p12_cutest100_exclusion_tol1e6.sh \
    --linear-backend ldl --dca-equality-model condensed-retraction \
    --problems "CATENA CATENARY"

  bash tools/run_cutest_batch_v0428r3p12_cutest100_exclusion_tol1e6.sh \
    --linear-backend native --dca-equality-model linearized
EOF
  }
  need_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
      echo "[FAIL] missing value after $1" >&2
      return 2
    fi
  }
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --linear-backend)
        need_value "$@" || return 2
        BACKEND="$2"; shift 2 ;;
      --dca-equality-model)
        need_value "$@" || return 2
        EQUALITY_MODEL="$2"; shift 2 ;;
      --problems)
        need_value "$@" || return 2
        PROBLEMS="$2"; PROBLEMS_OPTION_SET=1; shift 2 ;;
      --manifest)
        need_value "$@" || return 2
        MANIFEST_FILE="$2"; shift 2 ;;
      --live-every)
        need_value "$@" || return 2
        LIVE_EVERY="$2"; shift 2 ;;
      --phase1-threads)
        need_value "$@" || return 2
        PHASE1_THREADS="$2"; shift 2 ;;
      --help|-h)
        usage; return 0 ;;
      *)
        echo "[FAIL] unknown option: $1" >&2
        usage >&2
        return 2 ;;
    esac
  done

  case "$BACKEND" in
    native|ldl|qdldl|mumps) ;;
    *) echo "[FAIL] invalid --linear-backend: $BACKEND" >&2; return 2 ;;
  esac
  case "$EQUALITY_MODEL" in
    linearized)
      EQ_CODE="LIN"
      DCA_GLOBAL_ARGS=(--dca-corrector --dca-iterated-soc
                       --dca-equality-retraction)
      EQUALITY_ARGS=(--dca-equality-model linearized) ;;
    condensed-retraction)
      EQ_CODE="COND"
      # Selecting the equality model must not disable safeguards that also
      # serve inequality-only problems.  The solver itself suppresses the
      # legacy corrector/SOC paths when condensed_equality_mode is effective.
      DCA_GLOBAL_ARGS=(--dca-corrector --dca-iterated-soc
                       --dca-equality-retraction)
      # CATENA's validated condensed policy keeps the low-rank border small.
      # A rank-64 border is ill-conditioned on its first equality QP; the
      # verified scalar safeguard supplies any remaining tangent curvature
      # without adding border columns.
      EQUALITY_ARGS=(--dca-equality-model condensed-retraction
                     --dca-condensed-curvature
                     --lazyqp-low-rank-form bordered
                     --dca-condensed-max-rank 8
                     --dca-condensed-scalar-safeguard-updates 8
                     --dca-condensed-newton-iters 12
                     --dca-condensed-krylov-iters 32
                     --dca-condensed-krylov-rtol 1e-6
                     --dca-condensed-continuation 32
                     # This is an internal retraction acceptance threshold,
                     # not the final feasibility tolerance.  ALJAZZAF's
                     # scale makes 1e-9 indistinguishable from LDL roundoff;
                     # 1e-8 remains 100x tighter than the 1e-6 certificate.
                     --dca-condensed-tol 1e-8) ;;
    elastic-split)
      EQ_CODE="ELAS"
      DCA_GLOBAL_ARGS=(--no-dca-corrector --no-dca-iterated-soc
                       --no-dca-equality-retraction --no-soc)
      EQUALITY_ARGS=(--dca-equality-model elastic-split
                     --no-dca-equality-elastic-post-retraction) ;;
    *)
      echo "[FAIL] invalid --dca-equality-model: $EQUALITY_MODEL" >&2
      return 2 ;;
  esac
  if [[ ! "$LIVE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
    echo "[FAIL] --live-every must be a positive integer" >&2
    return 2
  fi
  if [[ ! "$PHASE1_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[FAIL] --phase1-threads must be a positive integer" >&2
    return 2
  fi
  if [[ "$PHASE1_THREADS" -gt 1 ]]; then
    PHASE1_ARGS=(--restoration-exact-parallel-row-dots)
  else
    PHASE1_ARGS=(--no-restoration-exact-parallel-row-dots)
  fi
  if [[ -n "$MANIFEST_FILE" && "$PROBLEMS_OPTION_SET" -eq 1 ]]; then
    echo "[FAIL] --manifest and --problems are mutually exclusive" >&2
    return 2
  fi
  if [[ -n "$MANIFEST_FILE" && ! -f "$MANIFEST_FILE" ]]; then
    echo "[FAIL] manifest not found: $MANIFEST_FILE" >&2
    return 2
  fi

  RUN_ID="${RUN_ID:-V0428R3P12_${EQ_CODE}_${BACKEND}_$(date +%Y%m%d_%H%M%S)}"
  RESULTS="${RESULTS:-$BASE/results/$RUN_ID}"
  LOGS="${LOGS:-$BASE/logs/$RUN_ID}"
  if [[ -n "$MANIFEST_FILE" ]]; then
    mapfile -t CASES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$MANIFEST_FILE")
    PROBLEMS="${CASES[*]}"
  else
    read -r -a CASES <<< "$PROBLEMS"
  fi
  if [[ ${#CASES[@]} -eq 0 ]]; then
    echo "[FAIL] empty PROBLEMS list" >&2
    return 2
  fi
  if [[ ! -x "$BIN" ]]; then
    echo "[FAIL] binary missing or not executable: $BIN" >&2
    return 2
  fi
  VERSION=$($BIN --version)
  if [[ "$VERSION" != *"0.4.28r3p12-cutest100-drcavty2-excluded"* ]]; then
    echo "[FAIL] wrong binary: $VERSION" >&2
    echo "Expected 0.4.28r3p12-cutest100-drcavty2-excluded; refusing a stale executable." >&2
    return 2
  fi
  BACKENDS=$($BIN --linear-backends)
  if [[ ",$BACKENDS," != *",$BACKEND,"* ]]; then
    echo "[FAIL] backend '$BACKEND' is not compiled; available: $BACKENDS" >&2
    return 2
  fi

  mkdir -p "$RESULTS" "$LOGS" "$CACHE"
  printf '%s\n' "${CASES[@]}" > "$RESULTS/manifest.txt"
  printf 'problem\trc\n' > "$RESULTS/return_codes.tsv"
  unset LD_PRELOAD
  unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
  export PYCUTEST_CACHE="$CACHE"
  export OMP_NUM_THREADS="$PHASE1_THREADS" OMP_DYNAMIC=FALSE
  export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1

  cat <<EOF
======================================================================
LazyIPDCA-QCQP r3p12 — r3p11 solver + reproducible CUTEst100 exclusion
version        : $VERSION
problems       : ${#CASES[@]} ($PROBLEMS)
linear backend : $BACKEND + METIS
equality model : $EQUALITY_MODEL ($EQ_CODE; quadratic equalities only)
global guards  : ${DCA_GLOBAL_ARGS[*]}
model options  : ${EQUALITY_ARGS[*]}
KKT rescue     : original bordered KKT, flexible GMRES, max 128
Phase-I OpenMP : $PHASE1_THREADS thread(s), one persistent team per PCG solve
display        : flushed [ITER] row every $LIVE_EVERY accepted iteration(s)
tolerances     : opt/dual/feas/comp=1e-6; inner=1e-8
limits         : outer=$MAX_OUTER inner=$MAX_INNER time=${TIME_LIMIT}s/problem
results        : $RESULTS
logs           : $LOGS
======================================================================
EOF

  for idx in "${!CASES[@]}"; do
    P="${CASES[$idx]}"
    CASE_DIR="$RESULTS/$P"
    LOG="$LOGS/$P.log"
    mkdir -p "$CASE_DIR"
    echo
    echo "[$((idx+1))/${#CASES[@]}] $P"
    echo "  log: $LOG"

    python -u "$ROOT/python/run_cutest_qcqp.py" "$P" \
      --binary "$BIN" --work-dir "$CASE_DIR" --cache "$CACHE" --keep-model \
      --reformulation auto --algorithm lazy-dca --inner-backend lazyqp \
      --linear-backend "$BACKEND" --qp-backend "$BACKEND" --linear-metis 1 \
      --qp-option C --qp-amrs --qp-init 2 --qp-mehrotra --qp-bp-adaptive \
      --kkt-adaptive-reg --kkt-persistent-symbolic-cache \
      --equality-qp-original-kkt-gmres \
      --equality-qp-original-kkt-gmres-max-iterations 128 \
      --lazyqp-kkt-form full --lazyqp-low-rank-form auto \
      --inner-full-dual-warm-start --lazyqp-warm-guard 2 \
      --max-outer "$MAX_OUTER" --max-inner "$MAX_INNER" --time-limit "$TIME_LIMIT" \
      --feas-tol 1e-6 --opt-tol 1e-6 --dual-inf-tol 1e-6 \
      --compl-inf-tol 1e-6 --inner-tol 1e-8 --inner-mu-fallback-tol 1e-7 \
      --restoration-iters 600 --restoration-restarts 16 \
      --restoration-restart-iters 400 --dca-inner-retries 3 \
      --restoration-exact-workspace-reuse \
      --restoration-exact-parallel-min-rows 128 "${PHASE1_ARGS[@]}" \
      --dca-inner-retry-prox 1e-6 --dca-step-control \
      --dca-equality-funnel-auto "${DCA_GLOBAL_ARGS[@]}" "${EQUALITY_ARGS[@]}" \
      --live-iterations --live-iterations-every "$LIVE_EVERY" \
      --live-header-every 20 --verbose 0 2>&1 \
      | tee "$LOG" \
      | grep --line-buffered -E \
          '^\[ITER\]|^\[1/2\]|^\[2/2\]|^\[FAIL\]|^\[EXPORT-REJECTED\]|^Traceback|^  File |^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):|^status|^message|^objective|^feasibility|^  equality|^  inequality|^KKT-stop dual|^KKT-stop desired|^outer / inner|^elapsed|^\[RESULT\]'
    rc=${PIPESTATUS[0]}
    printf '%s\t%s\n' "$P" "$rc" >> "$RESULTS/return_codes.tsv"

    JSON="$CASE_DIR/$P.json"
    if [[ -s "$JSON" ]]; then
      python - "$P" "$rc" "$JSON" <<'PY'
import json, sys
p, rc, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(path))
print("[DONE]", p, f"rc={rc}", f"status={d.get('status')}",
      f"obj={d.get('cutest_objective', d.get('objective'))}",
      f"eq={d.get('equality_violation')}",
      f"ineq={d.get('inequality_violation')}",
      f"dual={d.get('dual_infeasibility_inf')}",
      f"outer/inner={d.get('outer_iterations')}/{d.get('inner_iterations')}",
      f"time={d.get('elapsed_seconds')}")
PY
    else
      echo "[FAIL-DETAIL] $P produced no JSON; last 80 log lines follow:"
      tail -n 80 "$LOG"
      echo "[FAIL-DETAIL-END] $P"
      python - "$P" "$rc" "$JSON" "$LOG" <<'PY'
import json, sys
from pathlib import Path

problem, rc, json_path, log_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
tail = log_path.read_text(errors='replace').splitlines()[-80:] if log_path.is_file() else []
payload = {
    'problem': problem,
    'status': 'runner_failure',
    'message': f'child process exited with rc={rc} before producing its result JSON',
    'runner_return_code': rc,
    'log_tail': tail,
}
json_path.write_text(json.dumps(payload, indent=2) + '\n')
PY
      echo "[DONE] $P rc=$rc status=runner_failure"
    fi
  done

  python - "$RESULTS" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
problems = [x.strip() for x in (root/'manifest.txt').read_text().splitlines() if x.strip()]
rcs = {}
for line in (root/'return_codes.tsv').read_text().splitlines()[1:]:
    p, rc = line.split('\t', 1)
    rcs[p] = int(rc)
rows = []
for p in problems:
    path = root/p/f'{p}.json'
    d = json.loads(path.read_text()) if path.is_file() else {}
    rows.append({
        'problem': p, 'rc': rcs.get(p), 'status': d.get('status', 'missing'),
        'objective': d.get('cutest_objective', d.get('objective')),
        'feasibility': d.get('cutest_feasibility', d.get('feasibility')),
        'equality': d.get('equality_violation'),
        'inequality': d.get('inequality_violation'),
        'bounds': d.get('bound_violation'),
        'dual': d.get('dual_infeasibility_inf'),
        'nlp': d.get('scaled_nlp_error'),
        'outer': d.get('outer_iterations'), 'inner': d.get('inner_iterations'),
        'seconds': d.get('elapsed_seconds')})
with (root/'summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
solved = sum(r['status'] == 'solved' for r in rows)
lines = [f"SUMMARY solved={solved}/{len(rows)} failed={len(rows)-solved}"]
for r in rows:
    lines.append("{problem:12s} rc={rc!s:>2s} status={status:22s} obj={objective!s:>18s} "
                 "eq={equality!s:>12s} ineq={inequality!s:>12s} dual={dual!s:>12s} "
                 "outer/inner={outer!s}/{inner!s} time={seconds!s}".format(**r))
text = '\n'.join(lines) + '\n'
(root/'summary.txt').write_text(text)
print('\n' + text, end='')
print('SUMMARY_CSV=' + str(root/'summary.csv'))
print('SUMMARY_TXT=' + str(root/'summary.txt'))
PY
  echo "RESULTS=$RESULTS"
  echo "LOGS=$LOGS"
}

main "$@"
