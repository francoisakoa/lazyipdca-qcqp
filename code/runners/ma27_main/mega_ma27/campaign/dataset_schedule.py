"""Frozen dataset identities and audited terminal summaries.

Grouping uses source metadata and identifiers only. It never consults solver
outcomes. Dataset boundaries concern scheduling; numerical task contracts and
original-coordinate audits remain authoritative.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import random

REGIMES = ('I', 'M', 'E', 'Q', 'A')
DIMENSIONS = ('n', 'quadratic_inequalities', 'quadratic_equalities',
              'linear_inequalities', 'linear_equalities')
SCHEMA = 'continuous-datasets-no-dey-v1'
PRIORITIES=('CUTEst','multicast_beamforming','PGLIB')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def excluded_dey(case):
    """Explicit user removal of Dey only; other pooling families remain."""
    ident=str(case.get('id','')).casefold()
    return ident.startswith('dey__') or any(str(case.get(k,'')).casefold() in ('dey','pooling_dey') for k in ('dataset_id','family','reporting_family'))


def dataset_identity(case):
    """A library or a documented model lineage, never the broad applications bin."""
    ident = case['id']
    prefix = ident.split('__', 1)[0]
    family = str(case.get('family', ''))
    reporting = str(case.get('reporting_family', ''))
    source = case.get('source_kind', 'historical')
    if source == 'extension':
        return dict(id=family, label=family, basis='frozen-extension-model-family')
    if prefix.casefold() == 'cutest':
        return dict(id='CUTEst', label='CUTEst', basis='frozen-library-identifier')
    if prefix.casefold() == 'pglib':
        return dict(id='PGLIB', label='PGLib OPF', basis='frozen-library-identifier')
    # Camshape and multicast already have a source lineage shared with the
    # prospective extension. They remain one dataset across source cohorts.
    if reporting in ('COPS_camshape', 'multicast_beamforming'):
        return dict(id=reporting, label=reporting, basis='frozen-documented-model-lineage')
    if prefix.casefold() == 'gas' and ident.casefold().startswith('gas__gaslib134__'):
        return dict(id='GasLib134', label='GasLib134', basis='frozen-network-identifier')
    if prefix.startswith('LJ') and re.fullmatch(r'LJ\d+_seed\d+', prefix):
        return dict(id='lennard_jones', label='Lennard Jones', basis='frozen-model-family')
    labels = {
        'beam': ('beam_structural', 'Beam structural'),
        'beam_safety': ('beam_safety', 'Beam safety'),
        'robot': ('robot_arm', 'Robot arm'),
        'dey': ('pooling_dey', 'Pooling Dey'),
        'luedtke': ('pooling_luedtke', 'Pooling Luedtke'),
        'pooling': ('pooling_MINLPLIB', 'Pooling MINLPLib'),
        'pooling_pair': ('pooling_reduced', 'Reduced pooling'),
        'water': ('water_networks', 'Water networks'),
        'pgo2d': ('pose_graph_2d', 'Pose graph 2D'),
        'gas': ('GasLib_other', 'GasLib other networks'),
        'gaspower': ('gas_power', 'Gas power'),
        'gastransnlp': ('gas_transport', 'Gas transport'),
        'minlplib': ('MINLPLIB', 'MINLPLib'),
    }
    if prefix.casefold() in labels:
        key, label = labels[prefix.casefold()]
        return dict(id=key, label=label, basis='frozen-source-identifier-prefix')
    if family and family.casefold() not in ('applications', 'application', 'legacy_applications'):
        return dict(id=family, label=family, basis='frozen-source-family')
    if not prefix or prefix.casefold() in ('applications', 'application'):
        raise ValueError('Independent dataset lineage is missing: ' + ident)
    return dict(id='source_' + prefix, label=prefix, basis='frozen-source-identifier-prefix')


RESEARCH_METHODS=('F_R3','F_X1','F_X2','F_H0')

def case_methods(case,methods):
    return [method for method in methods if method not in RESEARCH_METHODS or case.get('regime')in('I','M')]

def task_pairs(cases,methods):
    """Published8order in principal run; balanced12/8order in crossed run."""
    methods=list(methods);active=all(m in methods for m in RESEARCH_METHODS)
    rng=random.Random(20261108);orders={};counts={}
    if active:
        for n,order in ((12,list(methods)),(8,[m for m in methods if m not in RESEARCH_METHODS])):
            rng.shuffle(order);orders[n]=order;counts[n]=0
    for case in cases:
        eligible=case_methods(case,methods)
        if active:
            n=len(eligible);offset=counts[n]%n;counts[n]+=1
            eligible=orders[n][offset:]+orders[n][:offset]
        for method in eligible:yield case,method


def prepare_datasets(cases, methods):
    """Copy cases, freeze assignments and publish full membership identities."""
    groups = {}
    seen = set()
    cases=[case for case in cases if not excluded_dey(case)]
    for original in cases:
        case = dict(original)
        if case['id'] in seen:
            raise ValueError('Duplicate dataset case: ' + case['id'])
        seen.add(case['id'])
        identity = dataset_identity(case)
        if not identity['id'] or case.get('regime') not in REGIMES:
            raise ValueError('Unknown dataset or structural regime: ' + case['id'])
        case.update(dataset_id=identity['id'], dataset_label=identity['label'],
                    dataset_classification_basis=identity['basis'])
        groups.setdefault(identity['id'], []).append(case)
    datasets, ordered = [], []
    keys = sorted(groups, key=lambda key: (PRIORITIES.index(key) if key in PRIORITIES else len(PRIORITIES), key.casefold(), key))
    for number, key in enumerate(keys, 1):
        members = sorted(groups[key], key=lambda case: (REGIMES.index(case['regime']), case['id']))
        ordered.extend(members)
        identities = [dict(id=case['id'], input_sha256=case['input_sha256'],
                           start_sha256=case.get('start_sha256'),
                           embedded_start_sha256=case.get('embedded_start_sha256'),
                           start_file_sha256=case.get('start_file_sha256'),
                           external_start_required=case.get('external_start_required', False),
                           regime=case['regime'], dimensions=case['dimensions']) for case in members]
        counts = {regime: sum(case['regime'] == regime for case in members) for regime in REGIMES}
        datasets.append(dict(id=key, label=members[0]['dataset_label'], position=number,
                             N=len(members), total_cells=sum(len(case_methods(c,methods))for c in members),
                             case_ids=[case['id'] for case in members],
                             membership_sha256=digest(identities), by_regime=counts,
                             classification_basis=sorted({case['dataset_classification_basis'] for case in members}),
                             source_families=sorted({str(case['family']) for case in members}),
                             source_cohorts=sorted({str(case.get('source_kind', 'historical')) for case in members})))
    return ordered, datasets


def verify_dataset_plan(plan):
    if any(excluded_dey(case) for case in plan['cases']) or any(excluded_dey(task['case']) for task in plan['tasks']):
        raise ValueError('Dey is excluded from both case and task denominators')
    ordered, datasets = prepare_datasets(plan['cases'], plan['methods'])
    if plan.get('dataset_schedule_schema') != SCHEMA or plan.get('datasets') != datasets:
        raise ValueError('Frozen dataset membership or order differs from the declared source classification')
    expected = [(case['id'], method) for case,method in task_pairs(ordered,plan['methods'])]
    actual = [(task['case']['id'], task['method']) for task in plan['tasks']]
    if expected != actual:
        raise ValueError('Tasks differ from the declared continuous dataset/treatment order')
    for task in plan['tasks']:
        case = task['case']
        identity = dataset_identity(case)
        if case.get('dataset_id') != identity['id']:
            raise ValueError('Task dataset identity differs: ' + task['id'])
    return True


def empty_counts(N):
    return dict(planned=N, done=0, pending=N, strict=0, failed=0,
                lazy=0, native_al=0, ipopt=0, unknown=0, indeterminate=0)


def count_result(item, result):
    item['done'] += 1
    item['pending'] -= 1
    if result.get('strict_qcqp') is None:
        item['indeterminate'] += 1
        return
    if result.get('strict_qcqp') is False:
        item['failed'] += 1
        return
    item['strict'] += 1
    origin = {'al': 'native_al', 'AL_natif': 'native_al', 'native-al': 'native_al'}.get(
        result.get('solution_origin'), result.get('solution_origin'))
    item[origin if origin in ('lazy', 'native_al', 'ipopt') else 'unknown'] += 1


def dataset_summaries(plan, results):
    """Fixed denominators; absent/interrupted cells are pending, not failures."""
    lookup = {case['id']: case for case in plan['cases']}
    groups = {}
    for declared in plan['datasets']:
        item = dict(declared)
        item['by_method'] = {method: empty_counts(sum(method in case_methods(lookup[cid],plan['methods'])for cid in item['case_ids'])) for method in plan['methods']}
        item['regimes'] = {regime: dict(N=N, methods={method: empty_counts(N if method not in RESEARCH_METHODS or regime in('I','M')else 0) for method in plan['methods']})
                          for regime, N in declared['by_regime'].items() if N}
        item['done'] = item['strict_qcqp'] = 0
        item['state'] = 'pending'
        groups[item['id']] = item
    seen = set()
    for result in results:
        key = (result['case_id'], result['method'])
        if key in seen or key[0] not in lookup or key[1] not in plan['methods']:
            raise ValueError('Duplicate or unknown terminal result in dataset report: ' + str(key))
        seen.add(key)
        case = lookup[key[0]]
        if key[1]not in case_methods(case,plan['methods']):raise ValueError('Result outside the declared per-case method matrix')
        group = groups[case['dataset_id']]
        count_result(group['by_method'][key[1]], result)
        count_result(group['regimes'][case['regime']]['methods'][key[1]], result)
        group['done'] += 1
        group['strict_qcqp'] += result.get('strict_qcqp') is True
    for item in groups.values():
        item['state'] = 'completed' if item['done'] == item['total_cells'] else 'incomplete' if item['done'] else 'pending'
    return groups


def csv_text(rows, columns=None):
    output = io.StringIO(newline='')
    columns = columns or (list(rows[0]) if rows else [])
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def report_text(item, methods):
    lines = ['DATASET %d %s — %s' % (item['position'], item['label'], item['state']),
             'Instances=%d terminés=%d/%d KKT_QCQP=%d' % (item['N'], item['done'], item['total_cells'], item['strict_qcqp'])]
    for method in methods:
        c = item['by_method'][method]
        lines.append('%-12s terminés=%d/%d KKT_QCQP=%d Lazy=%d AL_natif=%d Ipopt=%d non_stricts=%d indéterminés=%d en_attente=%d' %
                     (method, c['done'], c['planned'], c['strict'], c['lazy'], c['native_al'], c['ipopt'], c['failed'], c['indeterminate'], c['pending']))
    for regime in REGIMES:
        if regime not in item['regimes']:
            continue
        group = item['regimes'][regime]
        lines.append('REGIME %s N=%d %s' % (regime, group['N'], ' '.join(
            '%s=%d/%d' % (method, group['methods'][method]['strict'], group['N']) for method in methods)))
    return '\n'.join(lines) + '\n'


def write_dataset_report(out, plan, dataset_id, results, write, timestamp):
    """Called only after all solver/audit futures of this dataset have settled."""
    members = {case['id'] for case in plan['cases'] if case['dataset_id'] == dataset_id}
    results = [result for result in results if result['case_id'] in members]
    item = dataset_summaries(plan, results)[dataset_id]
    if item['state'] != 'completed':
        raise ValueError('Cannot finalize a dataset with pending or interrupted cells: ' + dataset_id)
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset_id)
    directory = Path(out) / 'datasets' / ('%03d_%s' % (item['position'], safe_name))
    directory.mkdir(parents=True, exist_ok=True)
    item.update(schema='audited-dataset-summary-v1', created=timestamp,
                task_contracts_sha256=digest(sorted((task['id'], task['contract_sha256']) for task in plan['tasks'] if task['case']['dataset_id'] == dataset_id)),
                terminal_results_sha256=digest(sorted(results, key=lambda result: (result['case_id'], result['method']))),
                all_terminal_results_retained=True, original_common_evaluator_unchanged=True,
                al_mode=plan.get('al_mode', 'auto'), tolerance=plan['tolerance'])
    summary_file = directory / 'SUMMARY.json'
    if summary_file.exists():
        previous = json.loads(summary_file.read_text())
        for field in ('membership_sha256', 'task_contracts_sha256', 'terminal_results_sha256'):
            if previous.get(field) != item[field]:
                raise ValueError('A completed dataset report disagrees with its immutable evidence: ' + str(summary_file))
        return previous, directory, False
    rows = []
    for regime, group in [('ALL', dict(N=item['N'], methods=item['by_method']))] + list(item['regimes'].items()):
        for method in plan['methods']:
            rows.append(dict(dataset=dataset_id, regime=regime, N=group['N'], method=method, **group['methods'][method]))
    (directory / 'SUMMARY.csv').write_text(csv_text(rows), encoding='utf-8')
    (directory / 'SUMMARY.txt').write_text(report_text(item, plan['methods']), encoding='utf-8')
    cases = [case for case in plan['cases'] if case['dataset_id'] == dataset_id]
    instances = [dict(id=case['id'], dataset=dataset_id, family=case['family'], regime=case['regime'],
                      **{key: case['dimensions'][key] for key in DIMENSIONS},
                      input_sha256=case['input_sha256'], start_sha256=case.get('start_sha256'),
                      start_file_sha256=case.get('start_file_sha256')) for case in cases]
    (directory / 'INSTANCES.csv').write_text(csv_text(instances), encoding='utf-8')
    case_lookup = {case['id']: case for case in cases}
    endpoints = []
    for result in sorted(results, key=lambda result: (result['case_id'], result['method'])):
        case = case_lookup[result['case_id']]
        audit = result.get('audit', {})
        endpoints.append(dict(id=result['case_id'], dataset=dataset_id, method=result['method'],
                              regime=case['regime'], status=result.get('status'),
                              strict_qcqp=result.get('strict_qcqp'), solution_origin=result.get('solution_origin'),
                              objective=audit.get('objective'), primal=audit.get('primal'),
                              dual=audit.get('dual'), complementarity=audit.get('complementarity'),
                              solver_process_seconds=result.get('work', {}).get('seconds'),
                              audit_process_seconds=result.get('audit_work', {}).get('seconds'),
                              contract_sha256=result['contract_sha256']))
    (directory / 'RESULTS.csv').write_text(csv_text(endpoints), encoding='utf-8')
    # Publish the JSON commit marker last; a controller interrupted while
    # writing descriptive CSVs can safely regenerate them on resume.
    write(summary_file, item)
    return item, directory, True
