#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BINARY=""
QCQ=""
OUTPUT=""
THREADS=8
LINEAR_BACKEND=ldl
TIME_LIMIT=1800

while (($#)); do
  case "$1" in
    --binary) BINARY=$2; shift 2 ;;
    --qcq) QCQ=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --linear-backend) LINEAR_BACKEND=$2; shift 2 ;;
    --time-limit) TIME_LIMIT=$2; shift 2 ;;
    -h|--help)
      echo "usage: $0 --binary BIN --qcq MODEL.qcq [--output DIR] [--threads N] [--linear-backend NAME] [--time-limit SEC]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$BINARY" ]] || { echo "binary not executable: $BINARY" >&2; exit 2; }
[[ -s "$QCQ" ]] || { echo "QCQP not found: $QCQ" >&2; exit 2; }
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "--threads must be positive" >&2; exit 2; }

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$PWD/R3P27A2_SOCP_LIN_COND_$(date -u +%Y%m%d_%H%M%S)_${RANDOM}"
fi
if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "output directory is not empty: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT"

export OMP_NUM_THREADS="$THREADS"
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

VERSION=$($BINARY --version)
EXPECTED=v0.4.28r3p28-lazy-newton-api-4313
[[ "$VERSION" == *"$EXPECTED"* ]] || {
  echo "wrong binary identity: $VERSION" >&2
  exit 2
}

MODEL_SHA=$(sha256sum "$QCQ" | awk '{print $1}')
printf 'CAMPAIGN=%s\nMODEL_SHA256=%s\n%s\n' "$OUTPUT" "$MODEL_SHA" "$VERSION"

COMMON=(
  --problem "$QCQ"
  --algorithm lazy-dca
  --inner-backend socp
  --linear-backend "$LINEAR_BACKEND"
  --qp-backend "$LINEAR_BACKEND"
  --lazyqp-model-support-fastpath
  --lazyqp-flow-lift-elimination
  --kkt-adaptive-reg
  --kkt-persistent-symbolic-cache
  --feas-tol 1e-6
  --opt-tol 1e-6
  --dual-inf-tol 1e-6
  --compl-inf-tol 1e-6
  --inner-tol 1e-8
  --max-outer 1000
  --max-inner 1000
  --time-limit "$TIME_LIMIT"
  --no-dca-equality-retraction
  --quiet
)

for MODE in lin cond; do
  if [[ "$MODE" == lin ]]; then
    EQUALITY_MODEL=linearized
  else
    EQUALITY_MODEL=condensed-retraction
  fi
  "$BINARY" "${COMMON[@]}" \
    --dca-equality-model "$EQUALITY_MODEL" \
    --json "$OUTPUT/$MODE.json" \
    >"$OUTPUT/$MODE.log" 2>&1
  [[ "$(sha256sum "$QCQ" | awk '{print $1}')" == "$MODEL_SHA" ]] || {
    echo "model changed during $MODE" >&2
    exit 1
  }
done

python "$ROOT/python/verify_socp_lin_cond_r3p27a2.py" \
  --lin "$OUTPUT/lin.json" \
  --cond "$OUTPUT/cond.json" \
  --model-sha256 "$MODEL_SHA" \
  --output "$OUTPUT/socp_lin_cond_summary.json"

printf 'SUMMARY=%s\nR3P27A2_SOCP_LIN_COND=PASS\n' \
  "$OUTPUT/socp_lin_cond_summary.json"
