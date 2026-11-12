#!/usr/bin/env bash
set -u -o pipefail

main() {
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE="${PYCUTEST_CACHE:-/data/projects/IPDCA_QCQP/pycutest_cache}"
OUT="${OUT:-/data/projects/IPDCA_QCQP/results/v0411_non_qcqp_gate_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT" "$CACHE"
export PYCUTEST_CACHE="$CACHE"
rc_all=0
for P in BT2 BT4 EIGENA2; do
  echo "===== $P: expected not_qcqp ====="
  python -u "$ROOT/python/run_cutest_qcqp.py" "$P" \
    --binary "$ROOT/build/lazyipdca_qcqp_cli" \
    --work-dir "$OUT/$P" --cache "$CACHE" --keep-model \
    --reformulation none --algorithm auto --quiet
  rc=$?
  python - "$OUT/$P/$P.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps({k:d.get(k) for k in ("problem","status","message")}, indent=2))
raise SystemExit(0 if d.get("status")=="not_qcqp" else 1)
PY
  ok=$?
  [[ $rc -eq 2 && $ok -eq 0 ]] || rc_all=1
done
if [[ $rc_all -eq 0 ]]; then
  echo "[PASS] BT2, BT4 and EIGENA2 rejected before optimization"
else
  echo "[FAIL] at least one non-QCQP was not rejected" >&2
fi
return "$rc_all"
}

main "$@"
