#!/usr/bin/env python3
"""Freeze the four historical nontrivial R3/Ipopt pairs before repetitions.

This is read-only with respect to the input observations and never invokes a
solver. Selection uses historical classifications and times, never future times.
"""
from pathlib import Path
from collections import Counter
import argparse
import gzip
import hashlib
import json
import math

BACKENDS = ('ma27', 'mumps')
REGIMES = ('I', 'M')
EXPECTED = {'ma27_I': 30, 'ma27_M': 89, 'mumps_I': 81, 'mumps_M': 75}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def selected(row):
    seconds = row.get('seconds_process')
    if row.get('classification') != 'strict' or row.get('strict') is not True:
        return False
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 1:
        return False
    for key in ('primal', 'dual', 'complementarity'):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e-6:
            raise ValueError('Historical strict row has invalid residual: ' + row['case_id'])
    return True


def build(compact):
    with gzip.open(compact, 'rt') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    index = {(r['case_id'], r['config']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate historical observation identity.')
    cohorts, pairs, jobs, config_rows = [], [], {}, {}
    for backend in BACKENDS:
        native, reference = 'R3_' + backend, 'ipopt_' + backend + '_monotone_aligned'
        for regime in REGIMES:
            cohort_id = backend + '_' + regime
            cohort_pair_ids = []
            natives = sorted((r for r in rows if r['config'] == native and r['regime'] == regime),
                             key=lambda r: r['case_id'])
            for n in natives:
                r = index[n['case_id'], reference]
                if not (selected(n) and selected(r)):
                    continue
                if n['input_sha256'] != r['input_sha256'] or n['start_sha256'] != r['start_sha256']:
                    raise ValueError('Historical pair data/start mismatch: ' + n['case_id'])
                pair_id = 'pair_' + digest([n['case_id'], backend])[:24]
                member_jobs = []
                for role, row in [('native', n), ('reference', r)]:
                    job_id = 'job_' + digest([row['case_id'], row['config']])[:24]
                    record = dict(job_id=job_id, case_id=row['case_id'], config=row['config'],
                                  role=role, backend=backend, regime=regime,
                                  dataset=row['dataset'], family=row.get('family'),
                                  input_sha256=row['input_sha256'],
                                  start_sha256=row['start_sha256'],
                                  binary_sha256=row['binary_sha256'],
                                  profile_sha256=row.get('profile_sha256'),
                                  historical_seconds_process=row['seconds_process'],
                                  historical_seconds_process_scope=row['seconds_process_scope'],
                                  original_objective=row['objective'],
                                  original_residuals={k: row[k] for k in ('primal', 'dual', 'complementarity')},
                                  original_observation_sha256=digest(row),
                                  original_bindings={k: row.get(k) for k in (
                                      'execution_binding_sha256', 'result_sha256',
                                      'task_contract_sha256', 'proof_level')},
                                  start_binding=row.get('start_binding'))
                    if job_id in jobs and jobs[job_id] != record:
                        raise ValueError('Job identity collision: ' + job_id)
                    jobs[job_id] = record
                    member_jobs.append(job_id)
                    config_value = dict(binary_sha256=row['binary_sha256'],
                                        profile_sha256=row.get('profile_sha256'),
                                        effective_options=row.get('effective_options'))
                    prior = config_rows.setdefault(row['config'], config_value)
                    if prior != config_value:
                        raise ValueError('Historical options/binary vary inside selected configuration: ' + row['config'])
                pairs.append(dict(pair_id=pair_id, cohort_id=cohort_id,
                                  case_id=n['case_id'], backend=backend, regime=regime,
                                  native_job_id=member_jobs[0], reference_job_id=member_jobs[1]))
                cohort_pair_ids.append(pair_id)
            if len(cohort_pair_ids) != EXPECTED[cohort_id]:
                raise ValueError(f'Frozen cohort {cohort_id}: {len(cohort_pair_ids)} != {EXPECTED[cohort_id]}')
            cohorts.append(dict(cohort_id=cohort_id, backend=backend, regime=regime,
                                native_config=native, reference_config=reference,
                                pair_ids=cohort_pair_ids, pairs=len(cohort_pair_ids)))
    cases = sorted({p['case_id'] for p in pairs})
    assert len(pairs) == 275 and len(cases) == 179 and len(jobs) == 550
    assert len({p['pair_id'] for p in pairs}) == len(pairs)
    assert len({(j['case_id'], j['config']) for j in jobs.values()}) == len(jobs)
    historical_sum = sum(j['historical_seconds_process'] for j in jobs.values())
    result = dict(schema='targeted-quiet-timing-cohort-v1',
                  source_name=Path(compact).name, source_sha256=sha(compact),
                  selection=dict(native='R3', reference='Ipopt monotone certificate-matched',
                                 same_backend=True, regimes=list(REGIMES),
                                 both_historical_classifications='strict',
                                 both_historical_process_times_strictly_greater_than_seconds=1,
                                 future_results_used=False,
                                 input_and_raw_start_hashes_equal=True),
                  fixed_protocol=dict(repetitions=3, solver_processes_at_once=1,
                                      numerical_threads=1, pinned_cpu=0,
                                      fresh_process_for_every_job=True,
                                      warm_start=False, solver_seconds_limit=7200,
                                      max_outer_iterations=5000,
                                      independent_original_KKT_tolerance=1e-6,
                                      no_best_repeat_selection=True,
                                      no_future_cohort_reselection=True),
                  counts=dict(pairs=275, distinct_cases=179,
                              unique_case_configuration_jobs=550,
                              planned_solver_processes=1650,
                              cohorts=EXPECTED),
                  original_duration_inventory=dict(unique_jobs_seconds=historical_sum,
                                                   three_repetitions_historical_hours=3*historical_sum/3600,
                                                   scope='Descriptive historical concurrent process envelopes; not a quiet-run forecast.'),
                  configs=config_rows, cohorts=cohorts, pairs=pairs,
                  jobs=sorted(jobs.values(), key=lambda j: (j['case_id'], j['config'])))
    result['cohort_content_sha256'] = digest(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compact', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = build(args.compact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.out.exists() and args.out.read_text() != text:
        raise ValueError('Existing frozen cohort differs; do not overwrite it silently.')
    args.out.write_text(text)
    print(json.dumps(dict(counts=result['counts'], source_sha256=result['source_sha256'],
                         cohort_content_sha256=result['cohort_content_sha256'],
                         original_duration_inventory=result['original_duration_inventory']), indent=2))


if __name__ == '__main__':
    main()
