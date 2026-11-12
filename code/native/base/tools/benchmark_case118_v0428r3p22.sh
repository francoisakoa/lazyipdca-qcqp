#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# No configure or build step is allowed here.  Select an existing executable
# with BIN, or let the Python driver use ROOT/build/lazyipdca_qcqp_cli.
exec python3 -u "$ROOT/python/benchmark_case118_r3p22.py" "$@"
