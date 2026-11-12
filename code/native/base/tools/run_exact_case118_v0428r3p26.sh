#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=/data/projects/IPDCA_QCQP
CASE=
BIN=
RESULTS_ROOT=
REPETITIONS=5
TARGET_SECONDS=0.5
THREADS=8

usage() {
  cat <<'EOF'
Usage: run_exact_case118_v0428r3p26.sh [options]
  --base DIR            project base (default: /data/projects/IPDCA_QCQP)
  --case FILE           PGLib case118 .m file (default: BASE/pglib-opf/...)
  --binary FILE         r3p26 CLI (default: ROOT/build_pf_gpu/...)
  --results-root DIR    output root (default: BASE/results)
  --repetitions N       LazyIPDCA runs (default: 5)
  --target-seconds SEC  separately reported median target (default: 0.5)
  --threads N           OpenMP Phase-I threads (default: 8)
EOF
}

die() {
  printf 'R3P26_PROTOCOL=FAIL: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --base) BASE=${2:?missing value for --base}; shift 2 ;;
    --case) CASE=${2:?missing value for --case}; shift 2 ;;
    --binary) BIN=${2:?missing value for --binary}; shift 2 ;;
    --results-root) RESULTS_ROOT=${2:?missing value for --results-root}; shift 2 ;;
    --repetitions) REPETITIONS=${2:?missing value for --repetitions}; shift 2 ;;
    --target-seconds) TARGET_SECONDS=${2:?missing value for --target-seconds}; shift 2 ;;
    --threads) THREADS=${2:?missing value for --threads}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

CASE=${CASE:-"$BASE/pglib-opf/pglib_opf_case118_ieee.m"}
BIN=${BIN:-"$ROOT/build_pf_gpu/lazyipdca_qcqp_cli"}
RESULTS_ROOT=${RESULTS_ROOT:-"$BASE/results"}

[[ $REPETITIONS =~ ^[1-9][0-9]*$ ]] || die "--repetitions must be positive"
[[ $THREADS =~ ^[1-9][0-9]*$ ]] || die "--threads must be positive"
python - "$TARGET_SECONDS" <<'PY' || die "--target-seconds must be finite and positive"
import math
import sys
value = float(sys.argv[1])
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY

[[ -f "$CASE" ]] || die "case file is missing: $CASE"
[[ -x "$BIN" ]] || die "solver binary is missing or not executable: $BIN"
for file in \
  "$ROOT/python/export_pglib_qcqp_r3p23.py" \
  "$ROOT/python/solve_qcq_ipopt_casadi_r3p23.py" \
  "$ROOT/python/verify_r3p26_candidate.py"; do
  [[ -f "$file" ]] || die "required runner is missing: $file"
done

VERSION_TEXT=$("$BIN" --version 2>&1) || die "cannot query solver identity"
BACKENDS_TEXT=$("$BIN" --linear-backends 2>&1) || die "cannot query backends"
grep -Fq '0.4.28r3p26-single-loop-corrected' <<<"$VERSION_TEXT" \
  || die "wrong solver version"
grep -Fq 'API 4309' <<<"$VERSION_TEXT" || die "wrong solver API"
grep -Fq 'v0.4.28r3p26-single-loop-corrected-api-4309' <<<"$VERSION_TEXT" \
  || die "wrong build identifier"
grep -Eq '(^|,)ldl(,|$)' <<<"$BACKENDS_TEXT" \
  || die "historical LDL backend is unavailable"

unset LD_PRELOAD
unset ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export OMP_NUM_THREADS=$THREADS
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p "$RESULTS_ROOT"
STAMP=$(date +%Y%m%d_%H%M%S_%N)
RUN="$RESULTS_ROOT/R3P26_EXACT_CASE118_${STAMP}_$$"
QCQ="$RUN/pglib_opf_case118_ieee.shared.qcq"
EXPORT_JSON="$RUN/export.json"
IPOPT_JSON="$RUN/ipopt.result.json"
IPOPT_LOG="$RUN/ipopt.solve.log"
LAZY_RUN="$RUN/lazyipdca"
mkdir "$RUN"

printf 'CAMPAIGN=%s\n%s\nlinear_backends=%s\n' \
  "$RUN" "$VERSION_TEXT" "$BACKENDS_TEXT"

if ! python -u "$ROOT/python/export_pglib_qcqp_r3p23.py" "$CASE" \
  --output "$QCQ" \
  --json "$EXPORT_JSON" \
  --flow-limit apparent \
  --power-flow-start \
  --no-power-flow-voltage-continuation \
  >"$RUN/export.log" 2>&1; then
  tail -n 40 "$RUN/export.log" >&2 || true
  die "case118 export failed"
fi
[[ -s "$QCQ" && -s "$EXPORT_JSON" ]] \
  || die "export did not create both model and manifest"
MODEL_SHA=$(sha256sum "$QCQ" | awk '{print $1}')
CASE_SHA=$(sha256sum "$CASE" | awk '{print $1}')
python - "$EXPORT_JSON" "$QCQ" "$CASE" "$MODEL_SHA" "$CASE_SHA" <<'PY' \
  || die "export manifest, dimensions, profile, or hashes are inconsistent"
import json
from pathlib import Path
import sys

manifest_path, model_arg, case_arg, model_sha, case_sha = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
assert manifest["schema"] == "lazyipdca-pglib-qcqp-export-v1"
assert manifest["export_revision"] == "r3p23-same-qcqp"
assert Path(manifest["model"]).resolve() == Path(model_arg).resolve()
assert Path(manifest["source_case"]).resolve() == Path(case_arg).resolve()
assert manifest["model_sha256"] == model_sha
assert manifest["source_case_sha256"] == case_sha
assert (manifest["n"], manifest["equalities"], manifest["inequalities"]) \
    == (1088, 980, 979)
assert manifest["flow_limit"] == "apparent"
assert manifest["power_flow_start"] is True
assert manifest["power_flow_voltage_continuation"] is False
PY
printf '%s  %s\n' "$MODEL_SHA" "$QCQ" >"$RUN/qcq.sha256"
chmod a-w "$QCQ"
printf 'EXPORT_AND_FREEZE=PASS SHA256=%s dimensions=1088/980/979\n' "$MODEL_SHA"

# Ipopt is deliberately the first optimizer process in this protocol.
IPOPT_BEFORE=$(sha256sum "$QCQ" | awk '{print $1}')
[[ "$IPOPT_BEFORE" == "$MODEL_SHA" ]] || die "model changed before Ipopt"
if ! python -u "$ROOT/python/solve_qcq_ipopt_casadi_r3p23.py" "$QCQ" \
  --json "$IPOPT_JSON" \
  --tol 1e-6 \
  --constr-viol-tol 1e-6 \
  --dual-inf-tol 1e-6 \
  --compl-inf-tol 1e-6 \
  --max-iter 3000 \
  --time-limit 1800 \
  --linear-solver mumps \
  --print-level 5 \
  >"$IPOPT_LOG" 2>&1; then
  tail -n 80 "$IPOPT_LOG" >&2 || true
  die "direct Ipopt failed"
fi
IPOPT_AFTER=$(sha256sum "$QCQ" | awk '{print $1}')
printf 'ipopt\t0\t%s\t%s\n' "$IPOPT_BEFORE" "$IPOPT_AFTER" \
  >"$RUN/ipopt_model_hashes.tsv"
[[ "$IPOPT_AFTER" == "$MODEL_SHA" && -s "$IPOPT_JSON" ]] \
  || die "Ipopt result missing or model changed"
python - "$IPOPT_JSON" "$MODEL_SHA" <<'PY' \
  || die "Ipopt original-QCQP certificate failed"
import json
import math
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["schema"] == "lazyipdca-qcqp-ipopt-direct-result-v1"
assert result["model_sha256"] == sys.argv[2]
assert result["success"] is True and result["kkt_passed"] is True
d = result["dimensions"]
assert (d["variables"], d["equalities"], d["inequalities"]) == (1088, 980, 979)
for key in ("equality_inf", "inequality_inf", "bound_inf", "dual_inf",
            "complementarity_inf", "dual_sign_inf"):
    value = float(result["residuals"][key])
    assert math.isfinite(value) and 0.0 <= value <= 1.0e-6, (key, value)
print("IPOPT_EXACT_REFERENCE=PASS", f"iterations={result['iterations']}",
      f"solver_s={result['solver_seconds']}", f"objective={result['objective']}")
PY

COMMON_ARGS=(
  --algorithm lazy-dca
  --inner-backend lazyqp
  --linear-backend ldl
  --qp-backend ldl
  --linear-metis 1
  --qp-option C
  --qp-amrs
  --qp-init 2
  --qp-mehrotra
  --qp-bp-adaptive
  --kkt-adaptive-reg
  --kkt-persistent-symbolic-cache
  --no-equality-qp-original-kkt-gmres
  --lazyqp-kkt-form reduced
  --no-lazyqp-numeric-kkt-reuse
  --inner-full-dual-warm-start
  --lazyqp-warm-guard 2
  --dca-affine-box-bounded-lazy
  --feas-tol 1e-6
  --opt-tol 1e-6
  --dual-inf-tol 1e-6
  --compl-inf-tol 1e-6
  --kkt-berr-tol 1e-9
  --kkt-berr-reject 1e-7
  --kkt-refine 3
  --inner-tol 1e-8
  --inner-mu-fallback-tol 1e-7
  --restoration-iters 600
  --restoration-restarts 16
  --restoration-restart-iters 400
  --restoration-exact-workspace-reuse
  --restoration-exact-sparse-row-dots
  --restoration-exact-parallel-row-dots
  --restoration-exact-parallel-min-rows 128
  --dca-inner-retries 3
  --dca-inner-retry-prox 1e-6
  --dca-step-control
  --no-soc
  --working-set-drop-patience 8
  --activation 5e-4
  --dca-equality-funnel-auto
  --no-dca-corrector
  --no-dca-iterated-soc
  --no-dca-equality-retraction
  --dca-equality-model condensed-retraction
  --dca-condensed-curvature
  --lazyqp-low-rank-form bordered
  --dca-condensed-max-rank 8
  --dca-condensed-scalar-safeguard-updates 8
  --dca-condensed-newton-iters 12
  --dca-condensed-krylov-iters 32
  --dca-condensed-krylov-rtol 1e-6
  --dca-condensed-continuation 32
  --dca-condensed-tol 1e-8
  --dca-condensed-adaptive-normal-refinement
  --dca-condensed-normal-intermediate-berr-tol 1e-6
  --dca-condensed-normal-intermediate-berr-reject 1e-4
  --dca-condensed-normal-strict-stationarity-factor 1e3
  --dca-condensed-alpha-predictor
  --dca-condensed-alpha-growth 2
  --dca-condensed-alpha-full-step-frequency 8
  --dca-condensed-flow-lift-schur
  --no-dca-condensed-post-retraction-ws-expansion
  --no-dca-condensed-modeled-side-filter
  --dca-condensed-block-exact-curvature
  --no-dca-condensed-projected-multi-ritz-inertia
  --lazyqp-flow-lift-elimination
  --no-dca-condensed-outer-extrapolation
  --no-terminal-kkt
  --max-outer 1000
  --max-inner 1
  --time-limit 1800
  --no-live-iterations
)

CANDIDATE_ARGS=(
  --dca-single-loop-primal-dual
  --dca-single-loop-inner-steps 1
  --dca-single-loop-forcing-eta 0.9
  --dca-single-loop-baseline-prox 0.05
  --dca-single-loop-multiplier-transport
  --no-dca-single-loop-direct-start
  --no-lazyqp-inexact-forcing
  --lazyqp-zero-dual-continuation
  --lazyqp-zero-dual-mu-cap 1e-4
)

mkdir "$LAZY_RUN"
: >"$LAZY_RUN/model_hashes.tsv"
for ((INDEX=1; INDEX<=REPETITIONS; ++INDEX)); do
  RESULT_JSON="$LAZY_RUN/run_${INDEX}.json"
  RESULT_LOG="$LAZY_RUN/run_${INDEX}.log"
  BEFORE_SHA=$(sha256sum "$QCQ" | awk '{print $1}')
  [[ "$BEFORE_SHA" == "$MODEL_SHA" ]] \
    || die "model changed before LazyIPDCA run $INDEX"
  if "$BIN" --problem "$QCQ" --json "$RESULT_JSON" \
    "${COMMON_ARGS[@]}" "${CANDIDATE_ARGS[@]}" \
    >"$RESULT_LOG" 2>&1; then
    RUN_RC=0
  else
    RUN_RC=$?
  fi
  AFTER_SHA=$(sha256sum "$QCQ" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\n' \
    "$INDEX" "$RUN_RC" "$BEFORE_SHA" "$AFTER_SHA" \
    >>"$LAZY_RUN/model_hashes.tsv"
  if ((RUN_RC != 0)); then
    tail -n 100 "$RESULT_LOG" >&2 || true
    die "LazyIPDCA run $INDEX returned $RUN_RC"
  fi
  [[ "$AFTER_SHA" == "$MODEL_SHA" && -s "$RESULT_JSON" ]] \
    || die "LazyIPDCA run $INDEX result missing or model changed"
  python - "$RESULT_JSON" "$INDEX" <<'PY' \
    || die "malformed LazyIPDCA result JSON"
import json
from pathlib import Path
import sys
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"LAZY_RUN={sys.argv[2]}", f"status={r.get('status')}",
      f"solver_s={r.get('elapsed_seconds')}",
      f"outer/steps/fact={r.get('outer_iterations')}/"
      f"{r.get('dca_single_loop_inner_steps')}/{r.get('kkt_factorizations')}")
PY
done

if python -u "$ROOT/python/verify_r3p26_candidate.py" \
  --qcq "$QCQ" \
  --ipopt-json "$IPOPT_JSON" \
  --run-dir "$LAZY_RUN" \
  --expected-runs "$REPETITIONS" \
  --tolerance 1e-6 \
  --target-seconds "$TARGET_SECONDS" \
  --require-zero-dual \
  --output "$RUN/benchmark_summary.json" \
  >"$RUN/verification.log" 2>&1; then
  cat "$RUN/verification.log"
else
  cat "$RUN/verification.log" >&2 || true
  die "KKT/identity/single-loop contract verification failed"
fi

printf '%s\n' "$RUN" >"$RUN/r3p26-last.pending"
mv -f "$RUN/r3p26-last.pending" "$RESULTS_ROOT/R3P26_EXACT_CASE118_LAST"
printf 'R3P26_PROTOCOL=PASS\nCAMPAIGN=%s\n' "$RUN"
