#!/usr/bin/env bash
set -u
INNER=lazyqp \
BACKEND=ldl \
LAZYQP_KKT_FORM=full \
PERSISTENT_SYMBOLIC_CACHE="${PERSISTENT_SYMBOLIC_CACHE:-1}" \
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a4x12_v049_lazy_dca.sh"
