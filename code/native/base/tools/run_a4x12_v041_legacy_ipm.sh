#!/usr/bin/env bash
set -u
INNER="legacy-ipm" BACKEND="${BACKEND:-native}" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a4x12_v041_lazy_dca.sh"
