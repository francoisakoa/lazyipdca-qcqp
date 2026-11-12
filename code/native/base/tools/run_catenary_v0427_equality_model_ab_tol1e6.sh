#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for MODEL in linearized elastic-split; do
  RUN_ID="${RUN_ID_PREFIX:-V0427_CATENARY_AB}_${MODEL//-/_}" \
  PROBLEMS=CATENARY \
  EQUALITY_MODEL="$MODEL" \
  LIVE="${LIVE:-1}" \
  BACKEND="${BACKEND:-ldl}" \
  MAX_OUTER="${MAX_OUTER:-1000}" \
  MAX_INNER="${MAX_INNER:-1000}" \
  TIME_LIMIT="${TIME_LIMIT:-1800}" \
  bash "$ROOT/tools/run_failures_v0427_optional_elastic_equalities_tol1e6.sh"
done
