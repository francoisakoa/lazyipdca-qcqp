#!/usr/bin/env python3
"""Describe archived unqualified observations without changing admission.

No solver or original-KKT evaluator is invoked. Native and parser diagnostics
are reported separately; a successful original audit is not reclassified as a
qualified controlled-experiment observation.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import zipfile


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def safe_id(value):
    import re
    changed = re.sub(r'[^A-Za-z0-9_.-]', '_', value)
    return changed if changed == value and len(value) < 180 else changed[:150]+'_'+sha_bytes(value.encode())[:16]


def analyze(archive, out, source_root=None):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        export = json.loads(z.read('EXPORT.json'))
        rows, classes, groups = [], Counter(), defaultdict(set)
        for line in z.open('PRODUCTION/RESULTS.jsonl'):
            row = json.loads(line)
            classes[row['classification']] += 1
            groups[row['case_id']].add(row['classification'])
            if row['classification'] == 'indeterminate':
                rows.append(row)
        assert dict(classes) == export['classifications']
        assert len(rows) == 189
        details = []
        for row in rows:
            folder = 'PRODUCTION/tasks/'+safe_id(row['case_id']+'__'+row['method'])
            def item(name):
                key = folder+'/'+name
                return json.loads(z.read(key)) if key in names else {}
            side = item('endpoint.json.factorial.json')
            endpoint = item('endpoint.json')
            resource = item('solver_resources.json')
            audit_resource = item('audit_resources.json')
            trace_key = folder+'/accepted_trace.jsonl'
            trace = [json.loads(line) for line in z.open(trace_key)] if trace_key in names else []
            accepted = [x for x in trace if x.get('event') == 'accepted']
            initialized = [x for x in trace if x.get('event') == 'metric_initialized']
            prefix = []
            for event in trace:
                if event.get('event') == 'metric_initialized':
                    break
                if event.get('event') == 'accepted':
                    prefix.append(event)
            frozen_metric_differences = [x['accepted_index'] for x in accepted
                if x.get('phase_metric_index', 0) > 0 and row['method'].startswith('FROZEN')
                and x.get('current_inequality_model_sha256') != x.get('initial_inequality_model_sha256')]
            prephase_valid = all(x.get('phase_metric_index') == 0
                and x.get('initial_inequality_model_sha256') == ''
                and x.get('current_inequality_model_sha256') == ''
                and all(x.get(k) == 0 for k in ['refinement_attempt_events',
                        'accepted_refinements', 'refinement_update_events', 'refinement_rollback_events'])
                for x in prefix)
            details.append(dict(case_id=row['case_id'], dataset=row['dataset'], regime=row['regime'],
                cell=row['method'], backend=row['method'].rsplit('_', 1)[1],
                status=row.get('status'), proof_error=row.get('experiment_proof_error'),
                original_audit_class=(row.get('original_audit') or {}).get('classification'),
                original_audit_strict=(row.get('original_audit') or {}).get('strict_qcqp'),
                trace_complete=export['files'].get(trace_key, {}).get('complete', False),
                trace_success=side.get('trace_success'), native_trace_error=side.get('error'),
                native_frozen_metric_verified=side.get('frozen_metric_invariant_verified'),
                metric_initialization_events=len(initialized), committed_events=len(accepted),
                leading_pre_metric_commits=len(prefix), pre_metric_zero_fingerprints_and_counters=prephase_valid,
                pre_metric_zero_refinement_counters=all(all(x.get(k) == 0 for k in [
                    'refinement_attempt_events','accepted_refinements','refinement_update_events',
                    'refinement_rollback_events']) for x in prefix),
                pre_metric_empty_initial_fingerprints=all(x.get('initial_inequality_model_sha256') == '' for x in prefix),
                pre_metric_nonempty_current_fingerprint_commits=sum(bool(x.get('current_inequality_model_sha256')) for x in prefix),
                pre_metric_state_hashes=[x.get('state_sha256') for x in prefix],
                initialized_phase_metrics=[x.get('initial_inequality_model_sha256') for x in initialized],
                frozen_hash_differences_at_recorded_commits=frozen_metric_differences,
                recorded_phase_indices=dict(Counter(str(x.get('phase_metric_index')) for x in accepted)),
                sidecar_accepted_events=side.get('accepted_events'),
                sidecar_endpoint_accepted_steps=side.get('endpoint_accepted_steps'),
                endpoint_flat_accepted_counter_present='lazy_newton_accepted_steps' in endpoint,
                endpoint_flat_accepted_counter=endpoint.get('lazy_newton_accepted_steps'),
                solver_returncode=resource.get('returncode'), solver_censored=resource.get('censored'),
                solver_timed_out=resource.get('timed_out'), solver_seconds=resource.get('elapsed_seconds'),
                audit_returncode=audit_resource.get('returncode'), audit_timed_out=audit_resource.get('timed_out'),
                audit_seconds=audit_resource.get('elapsed_seconds')))

        phase = [r for r in details if r['proof_error'] == 'Accepted metric phase is unavailable']
        early = [r for r in details if r['proof_error'] == 'Pas accepté avant une initialisation absente']
        absent = [r for r in details if r['status'] == 'endpoint_unavailable']
        audits = [r for r in details if r['status'] == 'original_audit_unavailable']
        assert len(phase) == 152 and len(early) == 8 and len(absent) == 27 and len(audits) == 2
        prefix_anomalies = [dict(case_id=r['case_id'], cell=r['cell'],
                                trace_complete=r['trace_complete'],
                                prefix_commits=r['leading_pre_metric_commits'],
                                zero_fingerprints_and_counters=r['pre_metric_zero_fingerprints_and_counters'])
                            for r in phase if not (r['trace_complete'] and r['leading_pre_metric_commits'] > 0
                                 and r['pre_metric_zero_fingerprints_and_counters'])]
        assert not any(r['frozen_hash_differences_at_recorded_commits'] for r in phase)
        assert all(r['committed_events'] == r['sidecar_accepted_events'] == r['sidecar_endpoint_accepted_steps']
                   for r in phase+early)
        assert all(not r['endpoint_flat_accepted_counter_present'] and r['committed_events'] == 0
                   and r['trace_success'] is True for r in early)
        prefix_pairs = []
        for cid in sorted({r['case_id'] for r in phase}):
            for backend in ['ma27', 'mumps']:
                for globalization in ['GR3', 'GHS']:
                    pair = [next(r for r in phase if r['case_id'] == cid and
                                r['cell'] == mode+'_'+globalization+'_'+backend)
                            for mode in ['FROZEN', 'REFINED']]
                    prefix_pairs.append(dict(case_id=cid, backend=backend, globalization=globalization,
                        identical_committed_prefix=pair[0]['pre_metric_state_hashes'] == pair[1]['pre_metric_state_hashes'],
                        identical_initial_metric=pair[0]['initialized_phase_metrics'][0] == pair[1]['initialized_phase_metrics'][0]))
        native_errors = Counter((r['trace_success'], r['native_trace_error']) for r in phase)
        diagnostic_groups = [
            dict(reason='pre_model_metric_phase_refusal', count=152,
                 original_strict=sum(r['original_audit_class'] == 'strict' for r in phase),
                 original_nonstrict=sum(r['original_audit_class'] == 'nonstrict' for r in phase)),
            dict(reason='pre_model_accepted_counter_schema_absence', count=8, original_strict=0, original_nonstrict=8),
            dict(reason='missing_endpoint_supervisor_7200s_timeout', count=sum(r['solver_timed_out'] is True for r in absent)),
            dict(reason='missing_endpoint_process_SIGKILL_no_cause_proof', count=sum(r['solver_returncode'] == -9 for r in absent)),
            dict(reason='independent_audit_600s_timeout', count=2),
        ]
        assert [x['count'] for x in diagnostic_groups] == [152, 8, 9, 18, 2]
        source_bindings = []
        if source_root:
            build = json.loads(z.read('REPRISE/build/BUILD_EXPERIMENT.json'))
            for rel in ['src/factorial_trace.hpp', 'src/dca_convexification_study_models.hpp',
                        'src/dca_inequality_majorant_models.hpp', 'src/dca_inequality_majorant_engine.hpp']:
                path = source_root/rel
                digest = sha_bytes(path.read_bytes())
                assert digest == build['experiment_source_sha256'][rel]
                source_bindings.append(dict(path=rel, sha256=digest, matches_campaign_build=True))
        report = dict(schema='refinement-indeterminate-diagnosis-v1', input_archive_sha256=sha_bytes(archive.read_bytes()),
            archived_classifications=dict(classes), indeterminate_observations=189,
            affected_case_ids=sorted({r['case_id'] for r in details}), affected_cases=25,
            all_eight_cells_complete_proof_cases=sum('indeterminate' not in c for c in groups.values()),
            by_dataset=dict(Counter(r['dataset'] for r in details)),
            by_regime=dict(Counter(r['regime'] for r in details)),
            by_cell=dict(Counter(r['cell'] for r in details)),
            groups=diagnostic_groups,
            native_errors_in_phase_refusals=[dict(trace_success=s, error=e, count=n) for (s,e),n in native_errors.items()],
            leading_phase_prefix_anomalies=prefix_anomalies,
            phase_prefix_pairs=prefix_pairs,
            original_strict_but_unqualified_by_cell=dict(Counter(r['cell'] for r in details if r['original_audit_class'] == 'strict')),
            source_bindings=source_bindings,
            source_explanation=[
                'Native affine startup sets convexification_study=off while retaining the shared factorial observer and emits accepted phase-index zero states before hybrid initialization.',
                'The phase-aware parser requires every primary accepted state to have a positive already-initialized metric phase, so it rejects the validly recorded leading phase-zero prefixes before later checks.',
                'DcaInequalityModels destructor calls study_summary; study_summary calls study_factorial_final_check before its study_enabled guard. In the historical affine phase, curvature_ready can be true although no hybrid study metrics were built. The observer can then persist an unconstructed-metric or empty-initial-fingerprint error into the later sidecar.',
                'In 16 CBLIB/pooled observations the leading phase-zero prefix contains a nonempty current fingerprint after such a diagnostic check, although its initial fingerprint remains empty; all prefix refinement counters are zero. These stale observer fields are retained as further evidence incompleteness, not ignored.',
                'Complete traces record no frozen metric-hash difference at positive-phase committed states in these 152 observations. This explains an instrumentation failure but does not by itself satisfy the preregistered complete-proof admission rules.',
                'The eight zero-step early endpoints omit the flat accepted-step field required by the controller, although trace and sidecar both record zero accepted events. Their original audits are nonstrict.',
                'SIGKILL identifies a termination signal; the archive does not contain kernel or cgroup evidence proving an out-of-memory kill.'
            ],
            admission_decision='All original 189 indeterminate classifications and the failed final completeness check are preserved; no retrospective admission or numerical reclassification is made.',
            qualified_scope='Existing complete-proof observations and paired complete-proof cohorts can be described as such. The full planned-corpus success counts include these unqualified outcomes as noncertificates; they are not an unconditional all-run causal validation.',
            optimizer_invocations=0, fresh_original_KKT_evaluations=0,
            detailed_observations=details)
        (out/'INDETERMINATE_DIAGNOSIS.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        with (out/'INDETERMINATE_DETAILS.csv').open('w', newline='') as f:
            fields=['case_id','dataset','regime','cell','status','proof_error','original_audit_class',
                    'trace_success','native_trace_error','leading_pre_metric_commits','solver_returncode',
                    'solver_timed_out','solver_seconds','audit_timed_out','audit_seconds']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(details)
        print(json.dumps({k:report[k] for k in ['archived_classifications','affected_cases','all_eight_cells_complete_proof_cases','groups']},indent=2))
        return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-root',type=Path)
    a=p.parse_args();analyze(a.archive,a.out,a.source_root)
