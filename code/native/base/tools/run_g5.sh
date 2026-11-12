#!/usr/bin/env bash
set -Eeuo pipefail
SRC=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" "$SRC/python/benchmark_filter_dca_g5.py" "$@"
