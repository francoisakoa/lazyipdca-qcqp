#!/usr/bin/env python3
"""Audit archived refinement bindings and reported residuals; run no solver."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import re
import zipfile


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_id(value):
    changed = re.sub(r'[^A-Za-z0-9_.-]', '_', value)
    return changed if changed == value and len(value) < 180 else changed[:150]+'_'+hashlib.sha256(value.encode()).hexdigest()[:16]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(archive, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    issues = []
    def check(condition, code, context=None):
        if not condition:
            issues.append(dict(code=code, context=context))
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        check(len(names) == len(z.infolist()), 'duplicate_archive_member')
        load = lambda name: json.loads(z.read(name))
        export = load('EXPORT.json')
        status = load('PRODUCTION/STATUS.json')
        contract = load('PRODUCTION/RUN_CONTRACT.json')
        methods = set(contract['methods'])
        methods_count = Counter()
        classes, by_regime, by_method = Counter(), defaultdict(Counter), defaultdict(Counter)
        records, originals = {}, {}
        case_inputs, case_starts, case_methods = defaultdict(set), defaultdict(set), defaultdict(set)
        guards, initial = defaultdict(set), defaultdict(set)
        initial_counts=Counter()
        missing_expected = Counter()
        strict_checked = nonstrict_checked = 0
        positive_proof_rejections = 0
        early_indeterminate_installation_binding_absent = []
        with z.open('PRODUCTION/RESULTS.jsonl') as stream:
            for index, line in enumerate(stream, 1):
                row = json.loads(line)
                key = row['case_id'], row['method']
                context = {'case_id':key[0], 'method':key[1]}
                check(key not in originals, 'duplicate_scalar_identity', context)
                folder = 'PRODUCTION/tasks/'+safe_id(key[0]+'__'+key[1])+'/'
                originals[key] = canonical(row)
                methods_count[key[1]] += 1
                case_methods[key[0]].add(key[1])
                case_inputs[key[0]].add(row['input_sha256'])
                case_starts[key[0]].add(canonical(row['start_binding']))
                classification = row['classification']
                classes[classification] += 1
                by_regime[row['regime']][classification] += 1
                by_method[key[1]][classification] += 1
                check(key[1] in methods, 'undeclared_cell', context)
                if classification=='indeterminate' and row.get('installation_sha256') is None:
                    early_indeterminate_installation_binding_absent.append(context)
                else:
                    check(row.get('installation_sha256') == contract['installation_sha256'], 'installation_binding', context)
                check(row.get('study_contract_sha256') == contract['study_contract_sha256'], 'study_binding', context)
                check(row.get('preregistration_sha256') == contract['registration_sha256'], 'registration_binding', context)
                cell = contract['cells'][key[1]]
                for name in ['objective_exact','inequality_policy','globalization_policy','refinement_policy','common_guards','common_representation']:
                    check(row.get(name) == cell[name], 'cell_metadata_'+name, context)
                solver = row.get('solver_resources')
                audit_res = row.get('audit_resources')
                original = row.get('original_audit')
                proof = row.get('provenance') or {}
                guard = row.get('guard_config_sha256')
                if guard:
                    guards[row['backend']].add(guard)
                state = row.get('initial_model_state')
                if state in ('reached','not_reached'):
                    initial[key[0], row['backend'], row['globalization_policy']].add((state,row.get('initial_inequality_model_sha256')))
                    initial_counts[key[0], row['backend'], row['globalization_policy']]+=1
                if classification in ('strict','nonstrict'):
                    check(row.get('budget_verified') is True, 'qualified_budget_not_verified', context)
                    check(row.get('intervention_verified') is True, 'qualified_intervention_not_verified', context)
                    check(proof.get('intervention_verified') is True, 'qualified_provenance_missing', context)
                    check(proof.get('objective_exact') is True, 'objective_not_exact', context)
                    if row['refinement_policy']=='frozen':
                        check(all(row.get(k)==0 for k in ['refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events']), 'frozen_refinement_counter_nonzero', context)
                        check(row.get('frozen_metric_invariant_verified') is True, 'frozen_invariance_unproved', context)
                    check(isinstance(original,dict), 'qualified_audit_absent', context)
                    check(original.get('classification') == classification, 'qualified_audit_classification', context)
                    for name in ['primal','dual','complementarity','objective']:
                        check(original.get(name)==row.get(name), 'archived_audit_scalar_'+name, context)
                    for name in ['primal','dual','complementarity']:
                        check(original.get('residual_upper',{}).get(name)==row.get(name), 'archived_outward_residual_upper_'+name, context)
                    check(original.get('input_sha256') == row['input_sha256'], 'audit_input_binding', context)
                    values = [row.get(k) for k in ['primal','dual','complementarity']]
                    finite = all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in values)
                    check(finite, 'qualified_residual_nonfinite', context)
                    if classification=='strict':
                        strict_checked += 1
                        check(finite and max(values)<=1e-6, 'strict_residual_above_tolerance', context)
                        check(original.get('strict_qcqp') is True and original.get('original_KKT_endpoint_strict') is True, 'strict_audit_flag', context)
                    else:
                        nonstrict_checked += 1
                        check(finite and max(values)>1e-6, 'nonstrict_residual_not_above_tolerance', context)
                    check(isinstance(solver,dict) and type(solver.get('returncode'))is int and solver['returncode']>=0 and solver.get('censored')is False and solver.get('timed_out')is False and solver.get('interrupted')is False and 0<=solver.get('elapsed_seconds',math.inf)<=7200, 'qualified_solver_resources', context)
                elif original and original.get('classification') == 'strict':
                    positive_proof_rejections += 1
                if isinstance(solver,dict):
                    cmd = solver.get('command') or []
                    check(hashlib.sha256(json.dumps(cmd).encode()).hexdigest()==solver.get('command_sha256'), 'solver_command_digest', context)
                    check(solver.get('threads') == 1 and solver.get('cpu') in list(range(22)), 'solver_cpu_or_threads', context)
                    check(cmd[:3] == ['taskset','-c',str(solver.get('cpu'))], 'solver_taskset_binding', context)
                records[folder] = dict(key=key,classification=classification,
                    result=canonical(row), audit=canonical(original) if original is not None else None,
                    solver=canonical(solver) if solver is not None else None,
                    audit_resources=canonical(audit_res) if audit_res is not None else None,
                    command=canonical(solver['command'][3:]) if isinstance(solver,dict) else None,
                    input_sha256=row['input_sha256'],start_sha256=(row.get('start_binding')or{}).get('start_sha256'),
                    sidecar_sha256=proof.get('sidecar_sha256'),trace_sha256=proof.get('trace_sha256'),
                    binary_sha256=row.get('binary_sha256'),native_binary_sha256=row.get('native_binary_sha256'),
                    endpoint_sha256=original.get('endpoint_sha256') if isinstance(original,dict) else None,
                    fields={k:row.get(k) for k in ['guard_config_sha256','initial_inequality_model_sha256','initial_hybrid_models_sha256','initial_model_events','refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events','frozen_metric_invariant_verified','actual_linear_backend','requested_linear_backend','primary_newton_backend_used','primary_backend_factor_attempts','native_pre_model_return','native_solver_completed','initial_model_state','initialization_calls','guard_policy_version']})
        check(len(originals)==14672 and contract['tasks']==14672, 'total_task_count')
        check(len(case_methods)==1834 and contract['cases']==1834, 'case_count')
        check(all(v==methods for v in case_methods.values()), 'eight_cells_each_case')
        check(all(len(v)==1 for v in case_inputs.values()), 'common_frozen_inputs')
        check(all(len(v)==1 for v in case_starts.values()), 'common_raw_start')
        check(all(len(v)==1 for v in guards.values()), 'fixed_common_guards')
        check(all(len(v)==1 for v in initial.values()), 'initial_metric_pair_identity')
        check(status['state']=='completed' and status['done']==14672 and status['active_slots']==0, 'completed_status')
        check(dict(classes)==export['classifications'], 'export_classification_counts')
        for k in ['strict','nonstrict','indeterminate']:
            check(classes[k]==status[k], 'status_classification_'+k)
        for method,count in by_method.items():
            for k in ['strict','nonstrict','indeterminate']:
                check(count[k]==status['by_method'][method][k], 'method_status_'+k, method)
        for regime,count in by_regime.items():
            for k in ['strict','nonstrict','indeterminate']:
                check(count[k]==status['by_regime'][regime][k], 'regime_status_'+k, regime)
        manifest=export['files']
        expected_members={'EXPORT.json','DIAGNOSTIC/OBSERVATIONS_INDETERMINEES.csv'}
        for name,item in manifest.items():
            expected_members.update([name] if item['complete'] else [name+'.head',name+'.tail'])
        check(expected_members==names, 'unmanifested_or_missing_archive_members', {'extra':sorted(names-expected_members),'missing':sorted(expected_members-names)})
        generated_csv=z.read('DIAGNOSTIC/OBSERVATIONS_INDETERMINEES.csv')
        csv_rows=list(csv.DictReader(io.StringIO(generated_csv.decode())))
        expected_indeterminate={tuple(record['key'])for record in records.values()if record['classification']=='indeterminate'}
        check(len(csv_rows)==189 and {(row['case_id'],row['method'])for row in csv_rows}==expected_indeterminate, 'generated_indeterminate_csv_identity')
        checked_files=checked_bytes=partial=0
        expected_json={'RESULT.json':'result','AUDIT.json':'audit','solver_resources.json':'solver','audit_resources.json':'audit_resources','COMMAND.json':'command'}
        for index,(name,item) in enumerate(manifest.items(),1):
            if item['complete']:
                data=z.read(name);digest=hashlib.sha256(data).hexdigest()
                check(len(data)==item['bytes'] and digest==item['sha256'], 'manifest_hash_or_size', name)
                checked_bytes += len(data);checked_files += 1
                if name.startswith('PRODUCTION/tasks/'):
                    folder,leaf=name.rsplit('/',1);record=records.get(folder+'/')
                    check(record is not None,'unplanned_task_folder',folder)
                    if record:
                        if leaf in expected_json:
                            value=json.loads(data);expected=record[expected_json[leaf]]
                            check(expected is None or canonical(value)==expected, 'aggregate_per_file_'+leaf, record['key'])
                        elif leaf=='endpoint.json.factorial.json' and record['classification']in('strict','nonstrict'):
                            check(digest==record['sidecar_sha256'], 'sidecar_digest_binding', record['key'])
                            side=json.loads(data)
                            for field,value in record['fields'].items():
                                if field in side:
                                    check(side[field]==value,'sidecar_field_'+field,record['key'])
                        elif leaf=='case.json':
                            case=json.loads(data)
                            check(case.get('input_sha256')==record['input_sha256'],'case_input_binding',record['key'])
                            check(case.get('start_sha256')==record['start_sha256'],'case_start_binding',record['key'])
                            check(case.get('budget_seconds')==7200 and case.get('max_iterations')==5000,'case_budget',record['key'])
                        elif leaf=='endpoint.json' and record['endpoint_sha256']:
                            check(digest==record['endpoint_sha256'],'indeterminate_endpoint_audit_digest',record['key'])
                        elif leaf=='accepted_trace.jsonl' and record['trace_sha256']:
                            check(digest==record['trace_sha256'],'indeterminate_trace_digest',record['key'])
            else:
                partial+=1
                for suffix,key in [('.head','head'),('.tail','tail')]:
                    data=z.read(name+suffix)
                    check(len(data)==item[key+'_bytes'] and hashlib.sha256(data).hexdigest()==item[key+'_sha256'],'partial_excerpt_hash_or_size',name+suffix)
                    checked_files+=1;checked_bytes+=len(data)
            if index%20000==0:
                print(f'Checked manifest {index}/{len(manifest)}',flush=True)
        # Verify retained metadata/source bindings using the export's path map.
        external=export['external_path_map']
        unresolved=[];verified_refs=set()
        def reference(item,context):
            if not isinstance(item,dict)or not isinstance(item.get('path'),str)or not isinstance(item.get('sha256'),str):
                return
            path=item['path'];member=external.get(path)
            if member is None:
                unresolved.append(dict(path=path,sha256=item['sha256'],context=context));return
            check(manifest[member]['sha256']==item['sha256'],'referenced_evidence_digest',{'path':path,'context':context})
            verified_refs.add(path)
        rec=load('REPRISE/RECOVERY.json');launch=load('REPRISE/PRODUCTION_LAUNCH.json')
        for key in ['original_gate','preparation','build','study','registration','qualifier','driver']:
            reference(rec.get(key),'recovery.'+key)
        for path,digest in rec['effective_tools'].items():reference(dict(path=path,sha256=digest),'effective_tool')
        reference(launch['installation'],'production_installation')
        installation=load(external[launch['installation']['path']])
        check(launch['installation']['sha256']==contract['installation_sha256'],'run_installation_receipt')
        for key in ['study_contract','registration','build_manifest','case_matrix']:
            if key+'_sha256'in installation:
                check(installation[key+'_sha256']==contract[key+'_sha256'],'installation_contract_'+key)
        for key,item in installation['registration_evidence'].items():reference(item,'registration_evidence.'+key)
        for record in records.values():
            if record['classification']in('strict','nonstrict'):
                backend='ma27'if record['key'][1].endswith('_ma27')else'mumps'
                binary=installation['build']['backends'][backend]['experiment_binary_sha256']
                check(record['binary_sha256']==binary and record['native_binary_sha256']==binary,'qualified_native_binary_binding',record['key'])
        sealed=load(external[installation['registration']])
        check(sealed['registration_status']=='sealed_before_production','sealed_registration_status')
        check(datetime.datetime.fromisoformat(sealed['sealed_utc'])<datetime.datetime.fromisoformat(status['started_utc']),'seal_precedes_production')
        check(sealed.get('production_outcomes_read')is False,'seal_did_not_read_production_outcomes')
        check(sealed['evidence']==installation['registration_evidence'],'sealed_evidence_exact')
        gate_item=installation['qualification_gate'];reference(gate_item,'qualification_gate')
        gate=load(external[gate_item['path']])
        for field in ['passed','production_authorized','reference_reproduction_passed','study_intervention_verified','effective_tools_unchanged']:
            check(gate.get(field)is True,'qualification_gate_'+field)
        check(gate.get('completed_native_runs')==748,'qualification_runs')
        check(gate.get('original_failed_gate')==rec['original_gate'],'original_refusal_preserved')
        original_gate=load(external[rec['original_gate']['path']])
        check(original_gate['passed']is False,'original_gate_remains_false')
        binding_coverage=Counter()
        # These references are only checked when exact archived source bytes are present.
        for path,digest in installation['source_snapshot'].items():
            if path in external:reference(dict(path=path,sha256=digest),'source_snapshot');binding_coverage['archived_source_snapshot_refs']+=1
            else:binding_coverage['source_snapshot_hash_only_refs']+=1
        input_case_map={c['case_id']:c for c in installation['cases']}
        check(len(input_case_map)==1834,'installation_cases')
        for cid,case in input_case_map.items():
            check(case_inputs[cid]=={case['input_sha256']},'installation_input_match',cid)
            check(case_starts[cid]=={canonical(case['start_binding'])},'installation_start_match',cid)
        intervention=load('PRODUCTION/INTERVENTION_CHECK.json')
        check(intervention['passed']is False and intervention['intervention_integrity_verified']is True and intervention['failures']==[],'final_intervention_verdict')
        check(intervention['verified_native_interventions']==14483 and intervention['incomplete_native_interventions']==189,'final_intervention_counts')
        qualified_files_missing=[]
        for item in export.get('missing_files',[]):
            member=item['member']
            if member.startswith('PRODUCTION/tasks/'):
                folder=member.rsplit('/',1)[0]+'/'
                record=records.get(folder)
                classification=record['classification']if record else'unknown'
                missing_expected[classification]+=1
                if classification in ('strict','nonstrict'):qualified_files_missing.append(member)
        check(not qualified_files_missing,'qualified_evidence_missing',qualified_files_missing[:20])
    result=dict(schema='refinement-archive-integrity-audit-v1',passed=not issues,
                archive_path=str(Path(archive).resolve()),archive_sha256=sha_file(archive),
                archived_members=len(names),manifest_original_file_entries=len(manifest),
                verified_complete_or_excerpt_members=checked_files,verified_uncompressed_bytes=checked_bytes,
                explicitly_excerpted_source_files=partial,classes=dict(classes),
                strict_archived_certificates_checked=strict_checked,nonstrict_archived_rejections_checked=nonstrict_checked,
                indeterminate_with_strict_archived_original_audit=positive_proof_rejections,
                early_indeterminate_installation_binding_absent=early_indeterminate_installation_binding_absent,
                matrix_cases=len(case_methods),methods=sorted(methods),
                common_declared_input_sha256_and_raw_start_bindings=True,
                common_guard_hashes={k:sorted(v)for k,v in guards.items()},
                initial_metric_pair_groups_checked=len(initial),
                initial_metric_complete_pairs_checked=sum(v==2 for v in initial_counts.values()),
                initial_metric_single_available_records=sum(v==1 for v in initial_counts.values()),
                archive_crc_members_verified=len(names),
                binding_references_verified=len(verified_refs),metadata_binding_coverage=dict(binding_coverage),
                archive_manifest_missing_files=len(export.get('missing_files',[])),
                missing_files_by_classification=dict(missing_expected),
                export_reported_disk_inconsistencies=export.get('disk_inconsistencies'),
                qualification_passed=gate.get('passed'),qualification_native_runs=gate.get('completed_native_runs'),
                registration_sealed_utc=sealed['sealed_utc'],production_started_utc=status['started_utc'],
                original_failed_gate_retained=original_gate.get('passed')is False,
                final_production_intervention_check=intervention,
                generated_manifest_and_indeterminate_csv_crc_verified=True,
                issues=issues,
                limitations=['No raw original QCQP inputs or general solver endpoint vectors were supplied; this validates archived audits and bindings, not fresh original KKT residuals.',
                             'Full accepted traces for qualified runs are represented by archived proof/sidecar hashes and summaries, not replayed here.',
                             'Full hashes of excerpted solver logs cannot be recomputed from excerpts; only head/tail members are verified.',
                             'Native binaries/libraries, raw source trees and most original qualification task artifacts remain hash references, not byte-verified local files.',
                             'Production completed does not imply the final intervention check passed; its189 incomplete observations remain indeterminate.'],
                optimizer_invocations=0,fresh_endpoint_audits=0,classification_changes=0)
    (out/'INTEGRITY_REPORT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ['issues','limitations','common_guard_hashes']},indent=2),flush=True)
    print('Issue_count',len(issues),flush=True)
    if issues:print(json.dumps(issues[:10],indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.archive,a.out)
