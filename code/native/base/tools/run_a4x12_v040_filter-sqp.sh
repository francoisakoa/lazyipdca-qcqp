#!/usr/bin/env bash
set -u
INNER="filter-sqp" BACKEND="${BACKEND:-native}" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a4x12_v040_lazy_dca.sh"
