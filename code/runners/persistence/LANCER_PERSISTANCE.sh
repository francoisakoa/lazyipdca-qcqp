#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PERSISTANCE_OUT="${PERSISTANCE_OUT:-/extra/LAZYQCQP_PERSISTANCE_RESULTS/ETUDE_PERSISTANCE_20261110_R1}"
python3 "$KIT_DIR/VERIFIER_KIT.py"
mkdir -p -- "$PERSISTANCE_OUT"
python3 -u "$KIT_DIR/ETUDE_PERSISTANCE.py" --out "$PERSISTANCE_OUT" "$@" \
  2>&1 | tee -a "$PERSISTANCE_OUT/ETUDE.log"
