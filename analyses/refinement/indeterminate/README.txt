DIAGNOSIS OF THE 189 UNQUALIFIED OBSERVATIONS

Run, without invoking any optimizer:
python3 analyze_indeterminate.py --archive /path/to/RAFFINEMENT_TERMINE_20261110T121503_184817Z.zip --out /tmp/refinement_indeterminate

The supplied report additionally verifies four native source files against
the experiment_source_sha256 map in the archived build receipt. To repeat
this optional source check, pass --source-root /path/to/native/experiment_source.

No classification or failed completeness verdict is modified. The 143
original-audit strict endpoints inside the 152 trace-bookkeeping refusals
are reported separately; they are not admitted as controlled-experiment
certificates. The report diagnoses observer/schema failures and external
execution/audit incompleteness. It does not establish an out-of-memory
cause for SIGKILL or complete a new original-KKT evaluation.

The original archive is the canonical input. Initial diagnostic JSON files
are working notes; INDETERMINATE_DIAGNOSIS.json, INDETERMINATE_DETAILS.csv,
and this script are the reproducible final diagnosis.

The four authenticated native headers used in the supplied diagnosis are
provided under source_evidence/ with SOURCE_EVIDENCE_SHA256.json.
They are evidence for inspection, not a full standalone solver build.
For the optional source check, use --source-root source_evidence.
