#!/usr/bin/env bash
# Compatibility entry point retained for commands copied from the r3p6 notes.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "[INFO] r3p6 runner name redirected to the corrected r3p10 runner." >&2
exec bash "$ROOT/tools/run_cutest_batch_v0428r3p10_original_kkt_gmres_tol1e6.sh" "$@"
