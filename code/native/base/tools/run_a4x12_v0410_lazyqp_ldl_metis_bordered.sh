#!/usr/bin/env bash
set -u
LAZYQP_LOW_RANK_FORM=bordered \
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a4x12_v0410_lazyqp_ldl_metis.sh"
