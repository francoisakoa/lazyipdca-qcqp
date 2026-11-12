#!/usr/bin/env bash
set -Eeo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python "$SCRIPT_DIR/../python/benchmark_g1.py" "$@"
