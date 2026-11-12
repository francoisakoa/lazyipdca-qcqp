Independent original-QCQP endpoint qualification
==============================================

AUDIT_ORIGINAL.py never invokes an optimizer and never uses Ipopt's own
success flag or reported residuals to decide the mathematical certificate.
It reads all original objective/constraint coefficients and all physical
primal and multiplier coordinates. Finite original bounds remain unchanged;
either infinity sign means that the corresponding bound side is absent.
Internally relaxed Ipopt bounds are not substituted for the original bounds,
and an endpoint is not projected before evaluation.

Command:
  python3 AUDIT_ORIGINAL.py --input problem.qcq --endpoint endpoint.json \
    --out AUDIT.json --resources solver_resources.json --case-json case.json

The original threshold is 1e-6 for primal, dual and complementarity residuals.
The dual residual includes stationarity, multiplier sign and absent-bound
multiplier conditions. Objective quality and second-order optimality are
separate: a first-order certificate is not a proof of a local/global minimum.

Precision
---------
All binary64 input/endpoint values are interpreted as the exact frozen
numbers. Outward-rounded interval products and fsum enclosures provide a
fast certificate where the threshold is unambiguous. A two-ulp fsum guard
also covers documented extended-precision double rounding. Ambiguous or
overflowing binary64 calculations are repeated with directed Decimal150
interval arithmetic. A point whose interval still straddles 1e-6 is
indeterminate rather than rounded into a success or numerical failure.
The reported scalar residuals are conservative interval upper bounds;
the exact decimal strings of both enclosure endpoints remain in the audit.

Memory
------
Inputs up to 32 MiB use the sparse model reader. Larger inputs are evaluated
as a coefficient stream in both V1 dense-row and V2 sparse-row formats.
No full list of model coefficients or model-gradient terms is retained.
Gradient sums use compact nonoverlapping floating expansions, with storage
proportional to variable count, plus endpoint arrays. A Decimal fallback
uses another streaming pass. Selection depends only on input size/format,
never on the instance or dataset name. Very large audits can still consume
substantial CPU time; the controller records an audit-time limit separately
and retains an interrupted assessment as indeterminate.

Observation status
------------------
original_KKT_endpoint_strict describes only the mathematical endpoint.
strict_qcqp and classification also account for external censoring.
Resources with censored/timed_out/interrupted flags or a negative signal
return code keep strict_qcqp=null even if a stored point can be audited.
Missing/nonfinite points are indeterminate. A nonzero ordinary solver return
code alone does not invalidate a completed independent point audit.
Internal maximum-iteration/time returns with a finite endpoint are assessed
as completed solver observations unless the resource contract declares an
external censor or budget violation. The scheduler owns resource/budget
classification and the audit process's separate timeout.

Qualification
-------------
  python3 TEST_AUDIT.py

AUDIT_VALIDATION.json records the mathematical checks, including an active
bound, original-bound relaxation violation, cancellation at the exact decimal
threshold, indefinite first-order stationarity, and censoring semantics.
Its streaming-memory fixture processes 160,000 nonzero coefficients without
retaining them; traced peak allocation is recorded. The archived validation
also recomputes the twelve supplied boundary-control points: eight strict,
four completed nonstrict, with no optimizer invocation.
