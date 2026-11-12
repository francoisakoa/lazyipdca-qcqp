#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/lazyipdca_qcqp_cli"
MODEL="$ROOT/tests/toy.qcq"

AVAILABLE="$($BIN --linear-backends)"
echo "[CHECK] compiled linear backends: $AVAILABLE"
[ "$AVAILABLE" = "native,ldl,qdldl,mumps" ]

for B in native ldl qdldl mumps; do
  echo "===== backend=$B ====="
  "$BIN" --problem "$MODEL" --linear-backend "$B" --linear-metis 1 --quiet \
    | grep -E 'linear solver|status|objective|feasibility|model stationarity|outer / inner|KKT factorizations'
done
