#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/data/projects/IPDCA_QCQP
BIN="$ROOT/build/lazyipdca_qcqp_cli"
CACHE="${PYCUTEST_CACHE:-$BASE/pycutest_cache}"
RUN_ID="${RUN_ID:-V0427_ELASTIC_TARGET}"
EQUALITY_MODEL="${EQUALITY_MODEL:-elastic-split}"
PROBLEMS_STR="${PROBLEMS:-CATENARY CATENA CORE2}"
LIVE="${LIVE:-1}"
BACKEND="${BACKEND:-ldl}"
MAX_OUTER="${MAX_OUTER:-1000}"
MAX_INNER="${MAX_INNER:-1000}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
ELASTIC_PENALTY="${ELASTIC_PENALTY:-10}"
ELASTIC_QUADRATIC="${ELASTIC_QUADRATIC:-1e-8}"
ELASTIC_CAP_FACTOR="${ELASTIC_CAP_FACTOR:-1}"
mkdir -p "$BASE/results/$RUN_ID" "$BASE/logs/$RUN_ID" "$CACHE"
read -r -a PROBLEMS_ARR <<< "$PROBLEMS_STR"
printf '======================================================================\n'
printf 'LazyIPDCA-QCQP v0.4.27 — equality model validation\n'
printf 'version        : %s\n' "$($BIN --version 2>/dev/null)"
printf 'problems       : %s\n' "$PROBLEMS_STR"
printf 'equality model : %s\n' "$EQUALITY_MODEL"
printf 'backend        : %s\n' "$BACKEND"
printf 'live output    : %s\n' "$LIVE"
printf '======================================================================\n'
for P in "${PROBLEMS_ARR[@]}"; do
  OUT="$BASE/results/$RUN_ID/$P"
  LOG="$BASE/logs/$RUN_ID/$P.log"
  mkdir -p "$OUT"
  printf '\n===== %s =====\n' "$P"
  printf 'log : %s\n' "$LOG"
  CMD=(python -u "$ROOT/python/run_cutest_qcqp.py" "$P"
    --binary "$BIN" --work-dir "$OUT" --cache "$CACHE" --keep-model
    --reformulation auto --algorithm lazy-dca --inner-backend lazyqp
    --linear-backend "$BACKEND" --linear-metis 1
    --kkt-adaptive-reg --kkt-persistent-symbolic-cache
    --lazyqp-kkt-form full --lazyqp-low-rank-form auto --lazyqp-condense-threshold 256
    --max-outer "$MAX_OUTER" --max-inner "$MAX_INNER" --time-limit "$TIME_LIMIT"
    --feas-tol 1e-6 --opt-tol 1e-6 --dual-inf-tol 1e-6 --compl-inf-tol 1e-6
    --inner-tol 1e-8 --inner-mu-fallback-tol 1e-7
    --dca-equality-model "$EQUALITY_MODEL"
    --dca-equality-elastic-penalty "$ELASTIC_PENALTY"
    --dca-equality-elastic-quadratic "$ELASTIC_QUADRATIC"
    --dca-equality-elastic-cap-factor "$ELASTIC_CAP_FACTOR"
    --no-dca-equality-elastic-post-retraction
    --no-restoration-tangent-activation
    --verbose 1)
  if [[ "$LIVE" == "1" ]]; then
    "${CMD[@]}" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
  else
    "${CMD[@]}" >"$LOG" 2>&1
    RC=$?
  fi
  printf '[DONE] %s rc=%s\n' "$P" "$RC"
done
printf 'RESULTS=%s\n' "$BASE/results/$RUN_ID"
printf 'LOGS=%s\n' "$BASE/logs/$RUN_ID"
