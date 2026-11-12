# Benchmark data

The compact observations in `../analyses/` reproduce the reported tables and
figures without running the solvers. The main benchmark contains 65,248
observations for 4,078 cases and 16 configurations. The separate ASU and
persistence records are in `../analyses/campaigns_R18/`.

`BENCHMARK_CASES.json.gz` identifies all 4,078 cases, their dimensions, the 28
reported dataset labels, original input hashes and raw-start bindings. The
regime counts are I=1,037, M=797, E=1,704 and Q=540. Each case has one input hash
and one start hash across the 16 recorded configurations. Selection follows
the archived inventory: 3,103 historical cases minus 50 Dey cases, plus 1,025
extensions after excluding two declared duplicates. Input paths are archived
locators; they are not coefficient arrays.

## Included frozen inputs

`bundled_frozen/` contains nine original benchmark QCQPs with embedded raw
starts (6,395,435 bytes in total). Their SHA256 hashes match the main benchmark
records exactly. They comprise CORE2, six boundary-control cases and two
minimal-surface cases. `BUNDLED_INPUTS.json` lists them. The native source tests
also contain small functional and regression fixtures, which are separate
from the full benchmark.

## Collecting the full frozen corpus

The original installation kept the complete frozen corpus on the author's
TRX machine. The supplied kits reference that installation; they do not embed
all 4,078 polynomial coefficient files. The complete benchmark can be rerun
with the supplied launchers after these frozen inputs and the numerical
libraries are available. Compact residual records are retained for analysis;
they are not a replacement for coefficient files and endpoint vectors when
recomputing KKT residuals.

Run the optional collector from the repository root on the machine holding
the frozen corpus:

```bash
python3 data/collect_frozen_inputs.py \
  --source-root /data/projects/IPDCA_QCQP/LAZYQCQP_MEGA_STABILISEE_20261107_R2 \
  --out /extra/LAZYQCQP_PUBLIC_FROZEN_R24
```

Repeat `--source-root` if another preserved installation contains part of the
corpus. The collector reads declared input locators and `PREPARATION.json` or
`FROZEN_INVENTORY.json` at those roots, verifies every input against the
recorded SHA256 and copies matching inputs to `frozen/<case>/input.qcq`.
It preserves embedded starts, runs no solver and does not copy endpoint
files. `COLLECTION.json` lists every collected or missing case; an incomplete
collection exits with code 2. Existing files with different contents are
never overwritten. Use repeated `--case CASE_ID` options for a subset.

For 120 constant-modulus beamforming cases, historical inventory hashes refer
to an earlier serialization. The primary `input_sha256` in this manifest is
the actual main-campaign hash; `source_metadata_input_sha256` preserves the
older value. The collector accepts only the main-campaign hash. The existing
preparation records supply the corresponding `canonical_inputs` locators.

No input is regenerated or replaced by a numerically similar model.
