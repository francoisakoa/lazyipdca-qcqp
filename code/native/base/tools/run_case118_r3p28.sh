#!/usr/bin/env bash
# r3p28 shell runner: same-QCQP comparison Ipopt / r3p27 sparse control / lazy-newton.
# Every option of python/benchmark_case118_r3p28.py (including all --ln-* controls
# of the C++ binary) is accepted verbatim; run with --help for the list.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 -u "$ROOT/python/benchmark_case118_r3p28.py" "$@"
