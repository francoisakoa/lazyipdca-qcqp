#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 -u "$ROOT/python/run_pglib_qcqp_r3p23.py" "$@"
