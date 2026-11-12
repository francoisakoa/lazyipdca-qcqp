#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# This wrapper consumes an already exported .qcq file.  Consequently the
# caller can pass the identical file (and SHA-256) to both Ipopt and
# LazyIPDCA; no model reconstruction occurs on the Ipopt path.
exec python3 -u "$ROOT/python/solve_qcq_ipopt_casadi_r3p23.py" "$@"
