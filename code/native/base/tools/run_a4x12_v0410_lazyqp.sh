#!/usr/bin/env bash
set -u
INNER=lazyqp \
BACKEND="${BACKEND:-native}" \
LAZYQP_KKT_FORM="${LAZYQP_KKT_FORM:-auto}" \
LAZYQP_LOW_RANK_FORM="${LAZYQP_LOW_RANK_FORM:-auto}" \
PERSISTENT_SYMBOLIC_CACHE="${PERSISTENT_SYMBOLIC_CACHE:-1}" \
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a4x12_v0410_lazy_dca.sh"
