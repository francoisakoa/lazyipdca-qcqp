#!/usr/bin/env bash
set -u -o pipefail

main() {
  ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  BASE="${BASE:-/data/projects/IPDCA_QCQP}"
  BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
  CACHE="${PYCUTEST_CACHE:-$BASE/pycutest_cache}"
  SOURCE_MANIFEST="${SOURCE_MANIFEST:-$ROOT/manifests/manifest_136_user.txt}"
  BACKEND="${BACKEND:-ldl}"
  EQUALITY_MODEL="${DCA_EQUALITY_MODEL:-condensed-retraction}"
  TARGET_COUNT="${TARGET_COUNT:-100}"
  N_LIMIT="${N_LIMIT:-10000}"
  ME_LIMIT="${ME_LIMIT:-5000}"
  MI_LIMIT="${MI_LIMIT:-5000}"
  LIVE_EVERY="${LIVE_EVERY:-1}"
  PHASE1_THREADS="${PHASE1_THREADS:-1}"

  usage() {
    cat <<'EOF'
Usage: bash tools/run_cutest100_v0428r3p12_drcavty2_excluded_tol1e6.sh [options]

Select exactly 100 CUTEst QCQPs, then run them sequentially. An instance is
excluded when n >= 10000, m_e >= 5000, or m_I >= 5000. BT4 and DRCAVTY2 are
also excluded by name. The limits and target are configurable.

Options:
  --linear-backend NAME       native | ldl | qdldl | mumps (default: ldl)
  --dca-equality-model NAME   linearized | condensed-retraction | elastic-split
  --source-manifest FILE      source candidates (default: manifest_136_user.txt)
  --target-count N            exact number selected (default: 100)
  --n-limit N                 exclude n >= N (default: 10000)
  --me-limit N                exclude m_e >= N (default: 5000)
  --mi-limit N                exclude m_I >= N (default: 5000)
  --live-every N              print one live row every N accepted iterations
  --phase1-threads N          persistent OpenMP team for Phase I (default: 1)
  --help                      display this help
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
      --source-manifest)
        need_value "$@" || return 2
        SOURCE_MANIFEST="$2"; shift 2 ;;
      --target-count)
        need_value "$@" || return 2
        TARGET_COUNT="$2"; shift 2 ;;
      --n-limit)
        need_value "$@" || return 2
        N_LIMIT="$2"; shift 2 ;;
      --me-limit)
        need_value "$@" || return 2
        ME_LIMIT="$2"; shift 2 ;;
      --mi-limit)
        need_value "$@" || return 2
        MI_LIMIT="$2"; shift 2 ;;
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
    linearized) EQ_CODE="LIN" ;;
    condensed-retraction) EQ_CODE="COND" ;;
    elastic-split) EQ_CODE="ELAS" ;;
    *) echo "[FAIL] invalid --dca-equality-model: $EQUALITY_MODEL" >&2; return 2 ;;
  esac
  for pair in \
    "target-count:$TARGET_COUNT" \
    "n-limit:$N_LIMIT" \
    "me-limit:$ME_LIMIT" \
    "mi-limit:$MI_LIMIT" \
    "live-every:$LIVE_EVERY"; do
    label=${pair%%:*}
    value=${pair#*:}
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "[FAIL] --$label must be a positive integer" >&2
      return 2
    fi
  done
  if [[ ! "$PHASE1_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[FAIL] --phase1-threads must be a positive integer" >&2
    return 2
  fi
  if [[ ! -f "$SOURCE_MANIFEST" ]]; then
    echo "[FAIL] source manifest not found: $SOURCE_MANIFEST" >&2
    return 2
  fi

  RUN_ID="${RUN_ID:-R3P12_CUTEST${TARGET_COUNT}_${EQ_CODE}_${BACKEND}_$(date +%Y%m%d_%H%M%S)}"
  RESULTS="${RESULTS:-$BASE/results/$RUN_ID}"
  LOGS="${LOGS:-$BASE/logs/$RUN_ID}"
  SELECTED="$RESULTS/selected_${TARGET_COUNT}.txt"
  AUDIT="$RESULTS/selection_audit.csv"
  SELECTION_JSON="$RESULTS/selection_summary.json"
  SELECTION_LOG="$LOGS/selection.log"
  mkdir -p "$RESULTS" "$LOGS" "$CACHE"
  export PYCUTEST_CACHE="$CACHE"

  cat <<EOF
======================================================================
LazyIPDCA-QCQP r3p12 — CUTEst size-filtered campaign
source          : $SOURCE_MANIFEST
target          : exactly $TARGET_COUNT instances
eligibility     : n<$N_LIMIT and m_e<$ME_LIMIT and m_I<$MI_LIMIT
name exclusions : BT4, DRCAVTY2
linear backend  : $BACKEND
equality model  : $EQUALITY_MODEL
Phase-I threads : $PHASE1_THREADS
results         : $RESULTS
logs            : $LOGS
======================================================================
EOF

  python -u "$ROOT/python/select_cutest_qcqp_manifest.py" \
    --manifest "$SOURCE_MANIFEST" \
    --selected "$SELECTED" \
    --audit "$AUDIT" \
    --summary-json "$SELECTION_JSON" \
    --cache "$CACHE" \
    --target "$TARGET_COUNT" \
    --n-limit "$N_LIMIT" \
    --me-limit "$ME_LIMIT" \
    --mi-limit "$MI_LIMIT" \
    --exclude BT4 \
    --exclude DRCAVTY2 2>&1 | tee "$SELECTION_LOG"
  selection_rc=${PIPESTATUS[0]}
  if [[ "$selection_rc" -ne 0 ]]; then
    echo "[FAIL] CUTEst selection failed with rc=$selection_rc; optimization was not started." >&2
    return "$selection_rc"
  fi

  selected_count=$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$SELECTED")
  if [[ "$selected_count" -ne "$TARGET_COUNT" ]]; then
    echo "[FAIL] selection contract violated: got $selected_count, expected $TARGET_COUNT" >&2
    return 2
  fi

  BASE="$BASE" \
  BIN="$BIN" \
  RUN_ID="$RUN_ID" \
  RESULTS="$RESULTS" \
  LOGS="$LOGS" \
  PYCUTEST_CACHE="$CACHE" \
  PHASE1_THREADS="$PHASE1_THREADS" \
  bash "$ROOT/tools/run_cutest_batch_v0428r3p12_cutest100_exclusion_tol1e6.sh" \
    --linear-backend "$BACKEND" \
    --dca-equality-model "$EQUALITY_MODEL" \
    --manifest "$SELECTED" \
    --live-every "$LIVE_EVERY" \
    --phase1-threads "$PHASE1_THREADS"
  batch_rc=$?
  echo "SELECTED_MANIFEST=$SELECTED"
  echo "SELECTION_AUDIT=$AUDIT"
  echo "SELECTION_JSON=$SELECTION_JSON"
  return "$batch_rc"
}

main "$@"
