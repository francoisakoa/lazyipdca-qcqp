"""Exact audited pilot observations may enter the full plan without another solve.

COMMAND, resource, endpoint and original-audit bytes always remain unchanged.
New plan metadata is explicit and bound to the preserved source RESULT record.
"""
from pathlib import Path
import copy
import json
import math
import os
import shutil
import BOUNDS as study

base=study.base


def artifact(path):return dict(path=str(Path(path).resolve()),sha256=base.sha(path))


def scientific_protocol(config):
    return {k:config[k]for k in ['root','build_manifest_sha256','build','python','hsl_library_sha256',
      'contract_sha256','profiles','profile_factory_sha256','controller_sha256','auditor_sha256',
      'reader_sha256','source_snapshot','budgets','intervention']}


def canonical_command(command):
    result=list(command)
    for flag in ['--json','--dca-rank-presolve-report']:
        if result.count(flag)!=1:raise ValueError('Output option absent/duplicate: '+flag)
        result[result.index(flag)+1]='{'+flag+'}'
    return result


def validate_existing(campaign,case,method,result):
    folder=campaign.folder(case,method)
    if not isinstance(result,dict)or result.get('case_id')!=case['case_id']or result.get('method')!=method:
        raise ValueError('Cached case/cell identity differs')
    if result.get('input_sha256')!=case['input_sha256']or result.get('start_binding')!=case['start_binding']:
        raise ValueError('Cached frozen input/start differs')
    if result.get('installation_sha256')!=campaign.config['_sha256']or result.get('binary_sha256')!=campaign.binary_identity(method,study.profiles.CELLS[method]['backend'])[1]:
        raise ValueError('Cached installation/native binary differs')
    for key,value in campaign.observation_metadata(case,method).items():
        if result.get(key)!=value:raise ValueError('Cached observation metadata differs: '+key)
    reuse=folder/'REUSE_RECEIPT.json'
    if reuse.is_file():
        receipt=base.read(reuse);original=base.read(folder/'REUSED_SOURCE_RESULT.json')
        for path,wanted in receipt['copied_files'].items():
            if base.sha(folder/path)!=wanted:raise ValueError('Copied pilot observation changed: '+str(folder/path))
        allowed={'run_id','installation_sha256','reused_solver_observation','reused_native_source','reuse_created_utc'}
        for key,value in original.items():
            if key not in allowed and result.get(key)!=value:raise ValueError('Reused observation content changed: '+key)
        if set(result)-set(original)-allowed:raise ValueError('Undeclared metadata added to cached pilot')
        if receipt['full_installation']['sha256']!=campaign.config['_sha256']:
            raise ValueError('Pilot reuse belongs to another full installation')
        if receipt['scientific_protocol_sha256']!=study.matrix_hash(scientific_protocol(campaign.config)):
            raise ValueError('Pilot/full scientific envelope changed')
        if result.get('reused_native_source')!=receipt['source_result']:raise ValueError('Reused source binding differs')
    if result['classification']=='indeterminate':return
    if result.get('installation_sha256')!=campaign.config['_sha256']:
        raise ValueError('Cached result installation mismatch')
    endpoint=base.read(folder/'endpoint.json');audit=base.read(folder/'AUDIT.json')
    resources=base.read(folder/'solver_resources.json');command=base.read(folder/'COMMAND.json')
    if audit!=result.get('original_audit')or resources!=result.get('solver_resources'):
        raise ValueError('Cached independent audit/resources differ')
    for key in ['classification','strict_qcqp','primal','dual','complementarity','objective']:
        if result.get(key)!=audit.get(key):raise ValueError('Cached certified numerical field differs: '+key)
    if audit.get('input_sha256')!=case['input_sha256']or audit.get('endpoint_sha256')!=base.sha(folder/'endpoint.json'):
        raise ValueError('Cached independent audit input/endpoint binding differs')
    cpu=resources.get('cpu')
    if type(cpu)is not int or cpu not in campaign.cpus:raise ValueError('Cached pilot CPU outside full envelope')
    actual=['taskset','-c',str(cpu),*command]
    import hashlib
    if resources.get('command')!=actual or resources.get('command_sha256')!=hashlib.sha256(json.dumps(actual).encode()).hexdigest():
        raise ValueError('Cached process command binding differs')
    if resources.get('threads')!=1 or resources.get('censored')is not False or resources.get('returncode',-1)<0:
        raise ValueError('Cached process budget is incomplete')
    seconds=resources.get('elapsed_seconds')
    if type(seconds)not in(int,float)or not math.isfinite(seconds)or not 0<=seconds<=7200:
        raise ValueError('Cached process elapsed time is absent/over budget')
    expected=campaign.make_command(case,method,study.profiles.CELLS[method]['backend'],folder/'endpoint.json',folder)
    if canonical_command(command)!=canonical_command(expected):raise ValueError('Cached native command differs beyond output paths')
    if base.sha(command[0])!=campaign.binary_identity(method,study.profiles.CELLS[method]['backend'])[1]:
        raise ValueError('Cached native executable changed')
    proof=campaign._check_endpoint(endpoint,method,study.profiles.CELLS[method]['backend'],folder)
    if proof.get('budget_verified')is not True or result.get('budget_verified')is not True:
        raise ValueError('Cached native budget not verified')
    for key in ['intervention_verified','rank_presolve_report_sha256','quadratic_equality_rule_enabled',
                'equality_implied_bound_sides','implied_bound_sides','actual_linear_backend','removed_sides']:
        if result.get(key)!=proof.get(key):raise ValueError('Cached intervention differs: '+key)


def pilot_gate(installation,run):
    config=study.installation(Path(installation));run=Path(run).resolve(strict=True)
    contract=base.read(run/'RUN_CONTRACT.json');status=base.read(run/'STATUS.json')
    if status.get('state')!='completed'or status.get('done')!=config['tasks']or status.get('active_slots')!=0:
        raise ValueError('The complete pilot must finish before full promotion')
    if contract['installation_sha256']!=config['_sha256']or config.get('pilot_cases')not in(12,24):
        raise ValueError('Pilot installation mismatch')
    cpus=tuple(contract['cpus']);campaign=study.BoundCampaign(config,run,cpus)
    records=[];positive={k:0 for k in [(p,b)for p in study.profiles.POLICIES for b in study.profiles.BACKENDS]}
    for case,method,backend in campaign.planned:
        result=base.read(campaign.folder(case,method)/'RESULT.json')
        if not result or result.get('classification')not in('strict','nonstrict'):
            raise ValueError('Pilot proof incomplete: '+case['case_id']+'/'+method)
        validate_existing(campaign,case,method,result);records.append(result)
        cell=study.profiles.CELLS[method]
        if cell['bounds']=='remove'and result['equality_implied_bound_sides']>0:
            positive[cell['policy'],backend]+=1
    if any(n==0 for n in positive.values()):raise ValueError('No actual removal observed for a pilot policy/backend')
    receipt=dict(schema='bound-removal-pilot-gate-v2',passed=True,installation=artifact(installation),
      run_contract=artifact(run/'RUN_CONTRACT.json'),observations=len(records),
      actual_removal_cases={p+'_'+b:n for(p,b),n in positive.items()},
      no_numerical_success_requirement=True,optimizer_invocations=0,
      result_sha256={str(campaign.folder(c,m)/'RESULT.json'):base.sha(campaign.folder(c,m)/'RESULT.json')for c,m,b in campaign.planned})
    base.atomic(run/'PILOT_GATE.json',receipt)
    return config,receipt,records


def promote(pilot_installation,pilot_run,full_installation,output,qualification):
    full=study.installation(Path(full_installation));pilot,gate,records=pilot_gate(pilot_installation,pilot_run)
    from QUALIFIER_BOUNDS import admission
    full['_qualification']=admission(qualification,full)
    if full.get('pilot_cases')is not None or len(full['cases'])!=503:raise ValueError('Full503-case installation required')
    if scientific_protocol(pilot)!=scientific_protocol(full):raise ValueError('Pilot/full implementation or envelope differs')
    by_id={c['case_id']:c for c in full['cases']}
    if any(by_id.get(c['case_id'])!=c for c in pilot['cases']):raise ValueError('Pilot cases/start/input differ from full matrix')
    output=Path(output).resolve();pilot_run=Path(pilot_run).resolve();old=base.read(output/'REUSE_PROMOTION.json')
    if old:
        if old['full_installation']!=artifact(full_installation)or old['pilot_gate']!=artifact(pilot_run/'PILOT_GATE.json'):
            raise ValueError('Existing pilot promotion differs')
        campaign=study.BoundCampaign(full,output,tuple(old['cpus']))
        for row in records:
            current=base.read(campaign.folder(by_id[row['case_id']],row['method'])/'RESULT.json')
            validate_existing(campaign,by_id[row['case_id']],row['method'],current)
        return old
    if output.exists():raise ValueError('Full result directory already exists without an admitted pilot promotion')
    final_output=output;stage=output.with_name(output.name+'.pilot_promotion_in_progress')
    marker=dict(full_installation=artifact(full_installation),pilot_gate=artifact(pilot_run/'PILOT_GATE.json'))
    if stage.exists():
        if base.read(stage/'PROMOTION_STAGE.json')!=marker:raise ValueError('Unrecognized pilot promotion staging folder')
        shutil.rmtree(stage) # Only a verified, solver-free copy transaction is rebuilt.
    stage.mkdir(parents=True);base.atomic(stage/'PROMOTION_STAGE.json',marker);output=stage
    cpus=tuple(base.read(pilot_run/'RUN_CONTRACT.json')['cpus'])
    contract=study.run_contract(full,output,cpus);base.atomic(output/'RUN_CONTRACT.json',contract)
    campaign=study.BoundCampaign(full,output,cpus);rows=[]
    for row in records:
        case=by_id[row['case_id']];method=row['method'];source=pilot_run/'tasks'/base.safe_id(case['case_id']+'__'+method)
        folder=campaign.folder(case,method);folder.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,folder)
        # Preserve the original record before declaring its placement in the
        # full matrix. No endpoint, command or process receipt is rewritten.
        shutil.copy2(source/'RESULT.json',folder/'REUSED_SOURCE_RESULT.json')
        copied={str(p.relative_to(folder)):base.sha(p)for p in folder.rglob('*')if p.is_file()and p.name!='RESULT.json'}
        receipt=dict(schema='bound-removal-exact-pilot-reuse-v2',source_result=artifact(source/'RESULT.json'),
          source_installation=artifact(pilot_installation),full_installation=artifact(full_installation),
          pilot_gate=artifact(pilot_run/'PILOT_GATE.json'),copied_files=copied,
          scientific_protocol_sha256=study.matrix_hash(scientific_protocol(full)),
          new_native_processes=0,warm_start=False,actual_commands_and_resources_preserved=True)
        base.atomic(folder/'REUSE_RECEIPT.json',receipt)
        adapted=copy.deepcopy(row);adapted.update(run_id=campaign.run_id,installation_sha256=full['_sha256'],
          reused_solver_observation=True,reused_native_source=receipt['source_result'],reuse_created_utc=base.utc())
        base.atomic(folder/'RESULT.json',adapted);validate_existing(campaign,case,method,adapted);rows.append(adapted)
    with(output/'RESULTS.jsonl').open('x')as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    receipt=dict(schema='bound-removal-pilot-promotion-v2',passed=True,full_installation=artifact(full_installation),
      pilot_gate=artifact(pilot_run/'PILOT_GATE.json'),cpus=list(cpus),reused_observations=len(rows),
      full_planned=full['tasks'],remaining_native_processes=full['tasks']-len(rows),new_native_processes=0)
    base.atomic(output/'REUSE_PROMOTION.json',receipt)
    os.rename(output,final_output)
    return receipt
