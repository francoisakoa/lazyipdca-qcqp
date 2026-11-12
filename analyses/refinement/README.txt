CONTROLLED REFINEMENT — ARCHIVED CLASSIFICATIONS — 10 NOVEMBER 2026

No optimizer is run. The 14,672 archived classifications are not changed:
12,970 strict certificates, 1,513 nonstrict results and 189 indeterminate
observations. A strict observation must also carry the admitted intervention
and process-budget proof and finite original KKT residuals at most 1e-6.

F means the common initial hybrid inequality correction is retained;
A means adaptive subsequent inequality refinement. F is not the historical
complete R3 baseline. GR3 and GHS change globalization while the safeguards,
exact objective, representation, starts and nominal budgets stay common.

REPRODUCE
python3 analyze_refinement_pairs.py \
  --compact REFINEMENT_OBSERVATIONS_8_CONFIGS.jsonl.gz \
  --study STUDY.json --out reproduced

--study is optional and only supplies the sealed-contract provenance hash;
the four registered engineering thresholds are checked against the code.
The study evidence must be copied to the named relative path, or its exact
location supplied. The resulting summary/CSV/TeX files are deterministic.

MAIN INTERPRETATION
The registered confirmatory comparison is A-GHS minus F-GHS pooled over
I+M separately for each backend. It is not the sum of all eight cells.
Known discordances are 5/5 for MA27 and 5/4 for MUMPS. Treating technical
outcomes as unqualified gives observed certificate differences 0 and +1.
Keeping them numerically unknown gives conservative net bounds [-23,23]
and [-23,25]. The exact registered rule reports
not_identified_due_to_technical_or_missing_outcomes for both backends.
There is no robust positive claim and no proof of failure for every possible
completion of the unresolved outcomes.

registered_ANALYSER_RAFFINEMENT.py is the exact bound original source from
the uploaded evidence. REGISTERED_ANALYSIS_SOURCE.json gives its archive
member and SHA256. REGISTERED_PRIMARY_ORIGINAL_IMPLEMENTATION.json contains
its direct primary() outputs. They equal the portable report exactly.

SECONDARY DESCRIPTIVE RESULTS
The eight regime/backend/globalization comparisons retain planned
denominators. Common-certificate timing ratios use process elapsed A/F,
with both times greater than one second; no startup subtraction. Separate
all-positive-time and objective-equivalent cohorts are also reported.
Objectives are original values including their constant. Gaps use
(f_A-f_F)/max(1,abs(f_A),abs(f_F)). Lower/equivalent/higher uses 0.1% and 1%.

Iteration differences on ASU use every common strict certificate pair;
timing ratios on ASU use their explicitly smaller both-over-one-second
cohort. The 38–64 ASU timing pairs are not the 373–375 iteration pairs.
ASU is the pre-existing beam_structural collection of 503 cases, not a
new outcome-selected subgroup. Unknown refinement counters remain unknown.

The measurements are single executions under concurrent machine load.
Backends are blocks, not independent replications or random samples.
These descriptive summaries do not estimate a universal speed advantage.

ADDITIONAL PRINTED TABLES
python3 render_compact_tables.py --summary reproduced/REFINEMENT_PAIR_SUMMARY.json --out reproduced
Compare refinement_coverage_times.tex and refinement_asu_compact.tex with expected/.

Full archived-report/hash audit requires the separately supplied ZIP:
python3 archive_integrity/audit.py --help
python3 indeterminate/analyze_indeterminate.py --help
No native executable or new QCQP residual evaluation is required by the descriptive analysis.
