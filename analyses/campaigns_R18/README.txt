R18: controlled ASU bound-removal and persistent-construction ablations
=====================================================================

Run with Python 3 (standard library only):

  python3 recompute_campaigns.py --out derived

The script checks the compact records and reproduces the statistics and complete
pair tables in derived/. It does not launch a solver. Records preserve the
supplied original-QCQP scalar audits, frozen input/start hashes and selected raw
artifact hashes. Full endpoint vectors are absent from the supplied exports;
this package reconciles the recorded certificates and does not claim a fresh
evaluation or recertification of those vectors. The archive names and SHA-256
digests in PROVENANCE.json bind these extracts to the supplied campaign exports.
The receipts/ directory contains independently checked controls and expected
statistics used for reconciliation. Full process logs and native sources are
kept in the original exports/kits rather than duplicated here.

ASU
---
ASU_OBSERVATIONS.jsonl.gz contains 6036 observations: 503 cases x R3/SS/HH x
keep/remove x MA27/MUMPS. The intervention toggles only the existing, proof-based
removal of bounds implied by quadratic equalities within one common source
profile. Input, raw start and numerical budgets are matched. This experiment
does not assert that this source reproduces every historical MUMPS executable.
The main 4078-case benchmark counts are not augmented by these observations.

4285 observations have strict recorded original-KKT certificates; 1729 are
documented nonstrict endpoints; 22 remain indeterminate because the controller
was interrupted and required resource/audit proofs were not exported. The last
category is retained separately, not converted into a numerical failure.

The objective statistic is

  g = (f_keep - f_remove) / max(1, abs(f_keep), abs(f_remove)).

Positive g favors removal. This normalized gap is not in general an ordinary
relative percentage when objectives are below one. Objective comparisons use
jointly strict observations. The six-cell common cohorts contain 305 cases
(MA27) and 304 (MUMPS). ASU ran under concurrent load; these observations do not
support an isolated timing or speed claim. Qualification comprises 144 extra
executions; the 144 pilot observations are already promoted into the 6036 full
campaign records and must not be counted a second time (6180 total executions).

Persistence
-----------
PERSISTENCE_OBSERVATIONS.jsonl.gz contains 144 finalized timed observations:
12 selected cases x two backends x reuse/rebuild x three repetitions. Each pair
changes only repeated construction of the same deterministic R3 correction:
reuse constructs once per model instance; rebuild reconstructs at every model
update, including updates at an unchanged iterate. Ordered correction/state traces, original
audits, numerical endpoints and numerical work are controlled by the preceding
72-run qualification (unpatched reference, patched reuse and patched rebuild).
PERSISTENCE_QUALIFICATION.json retains those 24 pair controls.

The timed process is fresh, single-threaded, sequential and pinned to CPU 0.
Each run follows a ten-second CPU-quiet window. Foreign process CPU and aggregate
CPU counters on every online CPU are sampled once per second (threshold 0.05
CPU-second per second, with 0.03-second tick allowance). Time is Popen through
wait4, including loading/startup/shutdown and excluding the independent audit
and quiet waits. The quiet test does not exclude idle processes, I/O waits or
all forms of memory contention. Qualification tracing is absent from timed runs;
the summary sidecar is present in both modes.

There are 216 nominal observations (72 qualification + 144 timing), requiring
220 actual attempts. Four contaminated timed attempts, all already numerically
strict, were retained and repeated under the fixed environment-validity rule.
They were not retried to turn numerical failures into successes. Raw final tail
aggregate flags alone are not the final quiet decision: wait4 corrects CPU work
of the just-reaped solver, which is absent from /proc. All 220 final quiet
decisions were independently reconciled (see receipts/PERSISTENCE_INTEGRITY.json).

All 144 timed observations are strict and all 24 pairs have identical numerical
core hashes, objectives, residuals and numerical work. Qualification preparations
total 25 for reuse and 4638 for rebuild. The measured effect concerns construction
cost, not different correction geometry, different stationary objectives or
learned correction-memory benefits. The 12 selected cases cover regimes I/M;
six are PGLib and one QPLIB 3312 has a zero correction. Median case/backend ratios
are reported rather than total-time aggregation, which is dominated by multicast.
These results do not represent all 4078 benchmark cases or regimes E/Q.
