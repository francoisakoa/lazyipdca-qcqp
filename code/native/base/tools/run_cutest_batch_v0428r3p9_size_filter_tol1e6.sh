#!/usr/bin/env bash
set -u -o pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "[INFO] r3p9 runner name redirected to the corrected r3p10 runner." >&2
exec bash "$ROOT/tools/run_cutest_batch_v0428r3p10_original_kkt_gmres_tol1e6.sh" "$@"
