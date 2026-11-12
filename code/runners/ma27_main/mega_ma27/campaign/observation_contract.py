"""Tri-state qualification of independently audited original-QCQP observations."""
from __future__ import annotations
import copy
import math

TOLERANCE = 1e-6


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def normalize_result(result, *, integrity_verified=None):
    """Return a copy; an incomplete/invalid observation never becomes a failure.

    ``False`` is reserved for a completed, within-budget, integrity-checked
    original audit whose actual three finite residuals violate tolerance.
    Native statuses, including ``time_limit``, cannot establish a certificate.
    """
    result = copy.deepcopy(result)
    if integrity_verified is None:
        integrity_verified = result.get('integrity_verified') is True
    result['integrity_verified'] = integrity_verified is True
    reasons = []
    work = result.get('work') or {}
    audit_work = result.get('audit_work') or {}
    audit = result.get('audit')
    route = result.get('route_policy_check') or {}
    if integrity_verified is not True:
        reasons.append('input_start_binary_runtime_integrity_unverified')
    if work.get('interrupted') is True:
        reasons.append('solver_interrupted')
    if work.get('external_timeout') is True:
        reasons.append('solver_externally_censored')
    if type(work.get('returncode')) is int and work['returncode'] < 0:
        reasons.append('solver_signal_termination')
    if (type(work.get('returncode')) is not int or work['returncode'] < 0
            or work.get('interrupted') is not False or work.get('external_timeout') is not False):
        reasons.append('solver_completion_unverified')
    if not isinstance(audit, dict) or audit.get('audit_available') is not True:
        reasons.append('independent_original_audit_unavailable')
    if audit_work.get('interrupted') is True or audit_work.get('external_timeout') is True:
        reasons.append('independent_audit_interrupted_or_censored')
    if audit_work and audit_work.get('returncode') != 0:
        reasons.append('independent_audit_process_failed')
    if (type(audit_work.get('returncode')) is not int or audit_work['returncode'] != 0
            or audit_work.get('interrupted') is not False or audit_work.get('external_timeout') is not False):
        reasons.append('independent_audit_completion_unverified')
    if route.get('passed') is not True:
        reasons.append('native_route_or_Ipopt_backend_proof_invalid')
    audit = audit if isinstance(audit, dict) else {}
    timing = audit.get('budget_and_timing') or {}
    if audit.get('native_route_integrity_pass') is False:
        reasons.append('native_optimizer_route_not_pure_lazy_AL_off')
    expected_origin = 'ipopt' if result.get('method') in ('ipopt_qcqp','ipopt_mumps') else 'lazy'
    if audit.get('solution_origin', result.get('solution_origin')) != expected_origin:
        reasons.append('endpoint_origin_unverified')
    if timing.get('budget_integrity_pass') is not True:
        reasons.append('budget_contract_unverified_or_exceeded')
    metrics = [audit.get(k) for k in ('primal', 'dual', 'complementarity')]
    metrics_valid = all(finite_nonnegative(v) for v in metrics)
    if not metrics_valid:
        reasons.append('original_residuals_absent_or_nonfinite')
    mathematical = audit.get('mathematical_strict_qcqp')
    if type(mathematical) is not bool:
        reasons.append('original_audit_indeterminate')
    elif metrics_valid and mathematical != all(v <= TOLERANCE for v in metrics):
        reasons.append('original_audit_residual_decision_disagrees')
    if audit.get('precision_agreement') is False:
        reasons.append('original_precision_audits_disagree')
    # A documented clock-limit stop without a strict endpoint is a censored
    # observation, even if its final timer rounded below the nominal limit.
    # A genuine full strict certificate already obtained within all budgets
    # remains determined solely by the audited residuals and integrity checks.
    if mathematical is False and timing.get('budget_exhausted') is True:
        reasons.append('native_time_budget_exhausted_before_strict_certificate')
    if reasons:
        strict = None
        native_status = str(audit.get('native_status', result.get('native_status', ''))).casefold()
        if 'solver_externally_censored' in reasons or 'solver_interrupted' in reasons or 'solver_signal_termination' in reasons:
            classification = 'censored_indeterminate'
        elif timing.get('budget_exhausted') is True:
            classification = 'budget_exhausted_indeterminate'
        elif native_status in ('time_limit', 'maximum_cputime_exceeded', 'maximum_walltime_exceeded'):
            classification = 'terminal_time_limit_indeterminate'
        else:
            classification = 'indeterminate_observation'
    else:
        strict = mathematical
        classification = 'strict_original_qcqp' if strict else 'audited_original_qcqp_nonstrict'
    result.update(strict_qcqp=strict, strict_original=strict,
                  observation_classification=classification,
                  observation_indeterminate_reasons=list(dict.fromkeys(reasons)),
                  numerical_failure_classified=strict is False,
                  endpoint_status_determines_certificate=False)
    return result
