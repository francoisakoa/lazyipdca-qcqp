#!/usr/bin/env bash
set -u
INNER="lazyqp" BACKEND="${BACKEND:-native}" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a4x12_v044_lazy_dca.sh"
