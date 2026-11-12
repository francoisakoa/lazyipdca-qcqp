# Reproduce the reported analyses

The scripts use the frozen records in `analyses/`. They recompute the descriptive
statistics, tables and performance profiles without launching an optimizer or
using a network connection.

Requirements: Python 3.12, NumPy and Matplotlib. From the repository root:

```bash
python3 -m pip install -r analyses/requirements.txt
python3 reproduction/run_reproduction.py --out reproduced
```

The driver reproduces the following material:

| Material | Frozen inputs | Outputs |
| --- | --- | --- |
| Main benchmark and paired comparisons | 65,248 classified observations, 16 configurations | Coverage, objective and time tables; 630 paired summaries |
| Published reference values | QPLIB and MINLPLib snapshots and conversion metadata | Reference-value comparisons |
| Controlled refinement | 14,672 classified observations and the study definition | Paired refinement tables |
| ASU bound removal | 6,036 classified observations and recorded controls | ASU counts, common cohorts and objective comparisons |
| Persistent construction | 144 timed observations and 24 qualification pair controls | 24 paired reuse/rebuild comparisons |
| Performance profiles | The main benchmark observations | PDF and PNG profiles for regimes I/M and E/Q |
| Controlled geometry example | Archived trajectories and analytic formulas | PDF and PNG geometry illustration |

Generated TeX tables are compared byte for byte with their archived outputs.
The paired summaries and the refinement, ASU and persistence summaries are
compared with their archived JSON values. The result is written to
`reproduced/GITHUB_REPRODUCTION_VALIDATION.json`.

These commands recompute statistics from recorded certificates. They do not
re-evaluate KKT residuals from endpoint vectors or rerun the numerical campaigns.
The frozen records retain the supplied classifications and input/start hashes.
The four benchmark regimes are I (quadratic inequalities, no quadratic
equalities), M (both types), E (quadratic equalities, no quadratic inequalities)
and Q (affine constraints only).

The standalone ASU/persistence analysis is also available:

```bash
python3 analyses/campaigns_R18/recompute_campaigns.py --out reproduced/asu_persistence
```

The archive names in `analyses/campaigns_R18/PROVENANCE.json` identify the
original campaign exports from which the compact records were extracted.
The source and scope of each analysis are described in the corresponding
`analyses/` subdirectory.
