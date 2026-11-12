#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# This wrapper deliberately does not invoke CMake: BIN points to an already
# compiled r3p21 executable (or defaults to this source tree's build output).
export BIN="${BIN:-$ROOT/build/lazyipdca_qcqp_cli}"
exec python3 -u "$ROOT/python/run_pglib_qcqp_r3p21.py" "$@"
