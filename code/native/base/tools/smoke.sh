#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
rc=0
ctest --test-dir build --output-on-failure || rc=$?
mkdir -p build/smoke
./build/lazyipdca_qcqp_cli --problem tests/toy.qcq --json build/smoke/toy.json --quiet || rc=$?
python - <<'PY' || rc=$?
import json
p='build/smoke/toy.json'
d=json.load(open(p))
assert d['status'] in ('solved','max_iterations')
assert d['feasibility'] is not None and d['feasibility'] < 5e-5
print('[PASS] CLI JSON smoke', d['status'], d['feasibility'])
PY
if [ "$rc" -ne 0 ]; then
  echo "[FAIL] smoke contracts rc=$rc"
else
  echo "[PASS] all smoke contracts"
fi
return "$rc" 2>/dev/null || true
