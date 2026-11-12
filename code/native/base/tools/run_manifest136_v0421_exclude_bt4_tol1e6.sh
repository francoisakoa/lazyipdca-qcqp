#!/usr/bin/env bash
set -u -o pipefail

main() {
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
BASE="${BASE:-/data/projects/IPDCA_QCQP}"
CACHE="${PYCUTEST_CACHE:-$BASE/pycutest_cache}"
MANIFEST="${MANIFEST:-$ROOT/manifests/manifest_136_user.txt}"
BACKEND="${BACKEND:-ldl}"
QP_BACKEND="${QP_BACKEND:-$BACKEND}"
MAX_OUTER="${MAX_OUTER:-1000}"
MAX_INNER="${MAX_INNER:-1000}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
LOW_RANK_FORM="${LOW_RANK_FORM:-auto}"
RESUME="${RESUME:-1}"
RERUN_FAILED="${RERUN_FAILED:-1}"
RUN_ID="${RUN_ID:-v0421_manifest136_exclude_BT4_lazy_dca_${BACKEND}_tol1e6_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${RESULTS:-$BASE/results/$RUN_ID}"
LOGS="${LOGS:-$BASE/logs/$RUN_ID}"

if [[ ! -x "$BIN" ]]; then
  echo "[FAIL] binary missing: $BIN" >&2
  return 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[FAIL] manifest missing: $MANIFEST" >&2
  return 2
fi

BACKENDS=$($BIN --linear-backends)
if [[ ",$BACKENDS," != *",$BACKEND,"* || ",$BACKENDS," != *",$QP_BACKEND,"* ]]; then
  echo "[FAIL] backend mismatch: compiled=$BACKENDS requested=$BACKEND qp=$QP_BACKEND" >&2
  return 2
fi

mapfile -t ALL < <(grep -v '^[[:space:]]*#' "$MANIFEST" | sed '/^[[:space:]]*$/d')
if [[ ${#ALL[@]} -ne 136 ]]; then
  echo "[FAIL] expected 136 manifest entries, got ${#ALL[@]}" >&2
  return 2
fi

mkdir -p "$RESULTS" "$LOGS" "$CACHE"
printf "%s\n" "${ALL[@]}" > "$RESULTS/manifest_136.txt"
printf "%s\n" "${ALL[@]}" | grep -vx 'BT4' > "$RESULTS/manifest_run_135.txt"
printf "problem\trc\tstatus\n" > "$RESULTS/return_codes.tsv"

unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export PYCUTEST_CACHE="$CACHE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1

cat <<EOF
======================================================================
LazyIPDCA-QCQP v0.4.21 — manifest 136 / execute 135
version        : $($BIN --version)
compiled linear: $BACKENDS
manifest       : $MANIFEST
policy         : BT4 excluded; every other manifest entry executed
exact lifts    : BT2; EIGENA2/EIGENB2/EIGENC2; CORE1/CORE2
outer route    : lazy-dca only
backend        : $BACKEND + METIS
LazyQP backend : $QP_BACKEND; option C + AMRS profile
KKT            : full + persistent symbolic cache
tolerances     : opt/dual/feas/comp=1e-6; inner=1e-8
limits         : outer=$MAX_OUTER inner=$MAX_INNER time=${TIME_LIMIT}s/problem
resume         : $RESUME (rerun_failed=$RERUN_FAILED)
results        : $RESULTS
logs           : $LOGS
======================================================================
EOF

# Record the sole benchmark exclusion explicitly.
mkdir -p "$RESULTS/BT4"
cat > "$RESULTS/BT4/BT4.json" <<'JSON'
{
  "problem": "BT4",
  "status": "excluded_by_policy",
  "message": "BT4 is the only instance excluded from the requested 136-instance campaign",
  "effective_outer_algorithm": null,
  "reformulation": "none",
  "ipopt_desired_convergence": false,
  "outer_iterations": 0,
  "inner_iterations": 0,
  "elapsed_seconds": 0.0
}
JSON
printf "BT4\t0\texcluded_by_policy\n" >> "$RESULTS/return_codes.tsv"

QP_BOOL_ARGS=(--qp-amrs --qp-atom-events --qp-mehrotra --qp-dca-stop --qp-bp-adaptive)

nrun=0
for idx in "${!ALL[@]}"; do
  P="${ALL[$idx]}"
  [[ "$P" == "BT4" ]] && { echo "[$((idx+1))/136] BT4 — EXCLUDED BY POLICY"; continue; }
  nrun=$((nrun+1))
  CASE_DIR="$RESULTS/$P"; LOG="$LOGS/$P.log"; JSON="$CASE_DIR/$P.json"
  mkdir -p "$CASE_DIR"

  if [[ "$RESUME" == "1" && -s "$JSON" ]]; then
    STATUS=$(python - "$JSON" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get('status','unknown'))
except Exception: print('invalid_json')
PY
)
    if [[ "$STATUS" == "solved" || "$STATUS" == "excluded_by_policy" || "$RERUN_FAILED" != "1" ]]; then
      echo "[$((idx+1))/136 | run $nrun/135] $P — SKIP existing status=$STATUS"
      printf "%s\t0\t%s\n" "$P" "$STATUS" >> "$RESULTS/return_codes.tsv"
      continue
    fi
    echo "[$((idx+1))/136 | run $nrun/135] $P — RERUN existing failure status=$STATUS"
    cp -f "$JSON" "$JSON.before_v0421_rerun" 2>/dev/null || true
  fi

  echo
  echo "[$((idx+1))/136 | run $nrun/135] $P"
  echo "  log     : $LOG"
  echo "  monitor : tail -f '$LOG'"

  python -u "$ROOT/python/run_cutest_qcqp.py" "$P" \
    --binary "$BIN" --work-dir "$CASE_DIR" --cache "$CACHE" --keep-model \
    --reformulation auto --algorithm lazy-dca --inner-backend lazyqp \
    --linear-backend "$BACKEND" --linear-metis 1 \
    --qp-option C --qp-init 2 --qp-backend "$QP_BACKEND" \
    --qp-rho-trial-every 2 --qp-max-atoms 500 \
    --qp-seed-coverage 0.25 --qp-seed-stall-tol 0 --qp-seed-patience 999 \
    --qp-atom-event-every 50 --qp-bound-push 0 --qp-bp-push-kappa 1e-4 \
    --qp-bp-margin-tol 1e-4 --qp-escapes 3 --qp-sigma-max 1e14 \
    --qp-model-solved-tol 1e-12 --qp-dca-stall-tol-rel 1e-10 \
    --qp-dca-stall-iters 5 --qp-dca-monotone-tol-abs 1e-6 \
    "${QP_BOOL_ARGS[@]}" \
    --kkt-adaptive-reg --kkt-persistent-symbolic-cache \
    --lazyqp-kkt-form full --lazyqp-low-rank-form "$LOW_RANK_FORM" \
    --lazyqp-condense-threshold 256 --inner-full-dual-warm-start --lazyqp-warm-guard 2.0 \
    --max-outer "$MAX_OUTER" --max-inner "$MAX_INNER" --time-limit "$TIME_LIMIT" \
    --feas-tol 1e-6 --opt-tol 1e-6 --dual-inf-tol 1e-6 --compl-inf-tol 1e-6 \
    --inner-tol 1e-8 --inner-mu-fallback-tol 1e-7 --s-max 100 \
    --restoration-iters 600 --restoration-restarts 16 --restoration-restart-iters 400 \
    --restoration-minimax-iters 80 --restoration-minimax-prox 1e-6 \
    --prox 1e-6 --dca-extra-prox 0 --dca-affine-full-limit 4096 \
    --dca-corrector --dca-corrector-start 2 --dca-corrector-frequency 1 \
    --dca-iterated-soc --dca-iterated-soc-iters 12 --dca-iterated-soc-max-ratio 2.0 \
    --dca-iterated-soc-trigger 1e6 --dca-warm-failure-limit 3 --dca-warm-cooldown 32 \
    --dca-inner-retries 3 --dca-inner-retry-prox 1e-6 --dca-equality-funnel 0 \
    --activation 1e-4 --constraint-model auto --working-set-drop-slack 2e-4 \
    --working-set-drop-mu 1e-9 --max-objective-rank 64 --max-constraint-rank 24 \
    --max-lagrangian-rank 64 --lanczos-steps 40 --lanczos-starts 2 \
    --exact-eigen-threshold 256 --verbose 1 > "$LOG" 2>&1
  rc=$?

  if [[ -s "$JSON" ]]; then
    read -r STATUS OBJ FEAS DUAL NLP OUTER INNER SEC REFORM <<<"$(python - "$JSON" <<'PY'
import json,sys
D=json.load(open(sys.argv[1]))
def v(k,alt=None):
    x=D.get(k)
    if x is None and alt: x=D.get(alt)
    return '-' if x is None else str(x)
print(v('status'),v('cutest_objective','objective'),v('cutest_feasibility','feasibility'),
      v('dual_infeasibility_inf'),v('scaled_nlp_error'),v('outer_iterations'),
      v('inner_iterations'),v('elapsed_seconds'),v('reformulation'))
PY
)"
  else
    STATUS="no_json"; OBJ=-; FEAS=-; DUAL=-; NLP=-; OUTER=-; INNER=-; SEC=-; REFORM=-
  fi
  printf "%s\t%s\t%s\n" "$P" "$rc" "$STATUS" >> "$RESULTS/return_codes.tsv"
  echo "  [DONE] rc=$rc status=$STATUS reform=$REFORM obj=$OBJ feas=$FEAS dual=$DUAL nlp=$NLP outer=$OUTER inner=$INNER time=$SEC"
  grep -E '^status|^message|^objective|^feasibility|^KKT-stop dual|^KKT-stop complement|^KKT-stop scaled|^KKT-stop desired|^outer / inner|^elapsed' "$LOG" | tail -12 || true
done

python - "$RESULTS" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=[x.strip() for x in (root/'manifest_136.txt').read_text().splitlines() if x.strip()]
rows=[]
for p in manifest:
    f=root/p/f'{p}.json'; d=json.loads(f.read_text()) if f.is_file() else {}
    rows.append({
      'problem':p,'status':d.get('status','missing'),'route':d.get('effective_outer_algorithm'),
      'reformulation':d.get('reformulation'),'objective':d.get('cutest_objective',d.get('objective')),
      'feasibility':d.get('cutest_feasibility',d.get('feasibility')),'dual':d.get('dual_infeasibility_inf'),
      'complementarity':d.get('complementarity_inf'),'scaled_nlp':d.get('scaled_nlp_error'),
      'kkt_pass':d.get('ipopt_desired_convergence'),'outer':d.get('outer_iterations'),'inner':d.get('inner_iterations'),
      'kkt_fact':d.get('kkt_factorizations'),'symbolic':d.get('kkt_symbolic_analyses'),'elapsed_s':d.get('elapsed_seconds'),
      'message':d.get('message')})
with (root/'summary.csv').open('w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
counts={}
for r in rows: counts[r['status']]=counts.get(r['status'],0)+1
print('\n===== SUMMARY 136 / 135 EXECUTED =====')
print('statuses:', ' '.join(f'{k}={v}' for k,v in sorted(counts.items())))
print('summary :',root/'summary.csv')
# Contract: BT4 is the only policy exclusion; all other entries must at least have a result JSON.
missing=[r['problem'] for r in rows if r['problem']!='BT4' and r['status']=='missing']
excluded=[r['problem'] for r in rows if r['problem']!='BT4' and str(r['status']).startswith('excluded')]
notqcqp=[r['problem'] for r in rows if r['problem']!='BT4' and r['status']=='not_qcqp']
if missing or excluded or notqcqp:
    print('[CONTRACT-FAIL] missing=',missing,'unexpected_excluded=',excluded,'unexpected_not_qcqp=',notqcqp)
else:
    print('[CONTRACT-PASS] BT4 is the only excluded instance; all other 135 were launched.')
PY

echo "RESULTS=$RESULTS"
echo "LOGS=$LOGS"
echo "SUMMARY=$RESULTS/summary.csv"
echo "RUN_ID=$RUN_ID"
}

main "$@"
