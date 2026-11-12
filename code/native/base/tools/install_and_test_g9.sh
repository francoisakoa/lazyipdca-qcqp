#!/usr/bin/env bash
# Run with bash; the interactive shell remains open on failure.
set -Eeo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lazydca_mumps}"
set -u
unset LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

BASE=${BASE:-/data/projects/IPDCA_QCQP}
TOP=lazyipdca_qcqp_v0.4.28r3p28g9_krylov_source
ARCHIVE="$BASE/${TOP}_only.tar.gz"
G9_DISPLAY_LEVEL=${DISPLAY_LEVEL:-2}
WARMUPS=${WARMUPS:-0}
REPETITIONS=${REPETITIONS:-1}
TIME_LIMIT=${TIME_LIMIT:-180}
CASES=(case300 case1354 case2869)

gzip -t "$ARCHIVE"
INSTALL=$(mktemp -d "$BASE/r3p28g9_XXXXXXXX")
tar --no-same-owner -xzf "$ARCHIVE" -C "$INSTALL"
SRC="$INSTALL/$TOP"
for FILE in tools/build_g9.sh tools/run_g9.sh python/benchmark_filter_dca_g9.py src/dca_krylov.hpp; do
  test -s "$SRC/$FILE"
done
(cd "$SRC" && sha256sum --quiet -c RELEASE_MANIFEST_SHA256_G9.txt)
BUILD_DIR="$SRC/build_pf_gpu" BUILD_JOBS="${BUILD_JOBS:-4}" TEST_JOBS="${TEST_JOBS:-2}" \
  bash "$SRC/tools/build_g9.sh"
mkdir -p "$BASE/results"
printf '%s\n' "$SRC" > "$BASE/results/R3P28G9_SOURCE_LAST"
printf 'INSTALL=PASS\nSOURCE=%s\n' "$SRC"

BINARY="$SRC/build_pf_gpu/lazyipdca_qcqp_cli"
BASELINE_FLAGS=()
if [[ -s "$BASE/results/R3P28G7_SOURCE_LAST" ]]; then
  OLD=$(cat "$BASE/results/R3P28G7_SOURCE_LAST")
  test -x "$OLD/build_pf_gpu/lazyipdca_qcqp_cli"
  BASELINE_FLAGS=(--baseline-binary "$OLD/build_pf_gpu/lazyipdca_qcqp_cli")
else
  printf 'G7 absent : comparaison G9 avec/sans Krylov et Ipopt.\n'
fi
FAILED=0
for PROBLEM in "${CASES[@]}"; do
  printf '\nComparaison de %s\n' "$PROBLEM"
  if bash "$SRC/tools/run_g9.sh" \
      --base "$BASE" --frozen "$PROBLEM" --binary "$BINARY" \
      "${BASELINE_FLAGS[@]}" --research-profile krylov-local --ablation \
      --krylov-max 24 --krylov-cooldown 2 \
      --threads 1 --display-level "$G9_DISPLAY_LEVEL" \
      --warmups "$WARMUPS" --repetitions "$REPETITIONS" \
      --max-iter 300 --time-limit "$TIME_LIMIT" --target-seconds 4 --require-target
  then
    printf '%s : certificats valides et cible G9 atteinte\n' "$PROBLEM"
  else
    RC=$?
    printf '%s : rc=%s — consulter CORRECTNESS et TARGET\n' "$PROBLEM" "$RC"
    FAILED=1
  fi
done
if [[ "$FAILED" == 0 ]]; then
  printf 'CAMPAIGN=PASS — précision et cible vérifiées\n'
else
  printf 'CAMPAIGN=FAIL — échec numérique ou cible manquée ; terminal préservé\n'
fi
test "$FAILED" -eq 0
