#!/usr/bin/env bash
set -u -o pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "[INFO] r3p10 CUTEst100 runner redirected to the corrected r3p11 runner." >&2
exec bash "$ROOT/tools/run_cutest100_v0428r3p11_phase1_persistent_openmp_tol1e6.sh" "$@"
