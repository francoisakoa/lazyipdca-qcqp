#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python -u "$ROOT/python/benchmark_case118_r3p27a1.py" "$@"
