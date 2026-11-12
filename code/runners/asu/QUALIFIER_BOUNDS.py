#!/usr/bin/env python3
"""Compare unpatched and profile-default research builds on the frozen pilot.

No numerical endpoint, work difference or censoring is admitted by tolerance.
Only observational clock fields are excluded from the exactbinary64 comparison.
"""
from pathlib import Path
import argparse
import collections
import json
import hashlib
import math
import BOUNDS as study
from NATIVE_PROOF import endpoint_core,rank_core,validate_side_report,validate_case_report

base=study.base;profiles=study.profiles;HERE=Path(__file__).resolve().parent
CELLS={f'{role}_{p}_{b}':dict(role=role,policy=p,backend=b)
       for role in ('REFERENCE','PROFILE')for p in profiles.POLICIES for b in profiles.BACKENDS}


def artifact(path):return dict(path=str(Path(path).resolve()),sha256=base.sha(path))


class Qualification(base.Campaign):
    def __init__(self,config,out,cpus):
        super().__init__(config,out,config['cases'],cpus=cpus)
        self.methods=tuple(CELLS)
        self.planned=[(case,label,c['backend'])for case in config['cases']for label,c in CELLS.items()]
    def indeterminate(self,case,method,reason,resources=None,audit=None):
        result=super().indeterminate(case,method,reason,resources,audit)
        result.update(installation_sha256=self.config['_sha256'],binary_sha256=self.binary_identity(method,CELLS[method]['backend'])[1])
        return result
    def binary_identity(self,method,backend):
        role=CELLS[method]['role'];key='reference_binary'if role=='REFERENCE'else'binary'
        return self.config['build'][key],self.config['build'][key+'_sha256']
    def runtime_environment(self,method,backend):return self.config['build']['runtime_environment']
    def observation_metadata(self,case,method):return dict(run_id=self.run_id,**CELLS[method],qualification_control=True)
    def make_command(self,case,method,backend,endpoint,folder):
        c=CELLS[method]
        return profiles.profile_command(c['policy'],backend,self.binary_identity(method,backend)[0],
          base.input_path(self.config,case),endpoint,self.config['hsl_library'],reference=c['role']=='REFERENCE')
    def check_endpoint(self,ep,method,backend,folder):
        c=CELLS[method];rank=base.read(folder/'endpoint.json.rank_presolve.json')
        validate_case_report(rank,base.read(folder/'case.json'))
        expected_removal=c['policy']=='HH'
        removal=validate_side_report(rank,keep=not expected_removal)
        if c['role']=='PROFILE':
            expected=dict(research_equality_implied_bounds='profile',quadratic_equality_rule_enabled=expected_removal,
                          radial_inequality_master_enabled=True,affine_equality_rule_enabled=c['policy']=='R3')
            if any(rank.get(k)!=v for k,v in expected.items()):raise ValueError('Profile-default native gate differs')
        if ep.get('linear_solver_backend')!='inequality-majorants/'+backend:raise ValueError('Actual qualification backend differs')
        if ep.get('lazy_newton_pivoted_fallbacks')!=0:raise ValueError('Primary factorization fallback in qualification')
        al=ep.get('native_al_fallback',{})
        if not isinstance(al,dict)or any(al.get(k)is True for k in ('enabled','attempted','accepted')):
            raise ValueError('AL route in native qualification')
        for key,limit in [('outer_iterations',5000),('restoration_iterations',600)]:
            value=ep.get(key)
            if type(value)is not int or not 0<=value<=limit:raise ValueError('Native qualification counter absent/over budget: '+key)
        seconds=ep.get('elapsed_seconds')
        if type(seconds)not in(int,float)or not math.isfinite(seconds)or not 0<=seconds<=7200:
            raise ValueError('Native qualification clock absent/over budget')
        return dict(budget_verified=True,intervention_verified=True,removed_sides=removal,
                    rank_presolve_report_sha256=base.sha(folder/'endpoint.json.rank_presolve.json'))
    def decorate_result(self,result,case,method,resources,proof):
        super().decorate_result(result,case,method,resources,proof)
        seconds=resources.get('elapsed_seconds')
        budget=result.get('budget_verified')is True and proof.get('budget_verified')is True and type(seconds)in(int,float)and math.isfinite(seconds)and 0<=seconds<=7200
        result.update(proof);result['budget_verified']=budget
        if not budget or proof.get('intervention_verified')is not True:
            result.update(classification='indeterminate',strict_qcqp=None)
        return result


def validate_control(campaign,case,method,result):
    folder=campaign.folder(case,method);backend=CELLS[method]['backend']
    for filename in ['RESULT.json','endpoint.json','AUDIT.json','solver_resources.json','COMMAND.json','endpoint.json.rank_presolve.json']:
        if not(folder/filename).is_file():raise ValueError('Qualification artifact absent: '+filename)
    if not result or result.get('classification')not in('strict','nonstrict')or result.get('budget_verified')is not True or result.get('intervention_verified')is not True:
        raise ValueError('Incomplete qualification result')
    if result.get('installation_sha256')!=campaign.config['_sha256']or result.get('binary_sha256')!=campaign.binary_identity(method,backend)[1]:
        raise ValueError('Qualification result installation/binary differs')
    if result.get('input_sha256')!=case['input_sha256']or result.get('start_binding')!=case['start_binding']:
        raise ValueError('Qualification frozen input/start differs')
    for key,value in campaign.observation_metadata(case,method).items():
        if result.get(key)!=value:raise ValueError('Qualification metadata differs: '+key)
    command=base.read(folder/'COMMAND.json');resources=base.read(folder/'solver_resources.json')
    if command!=campaign.make_command(case,method,backend,folder/'endpoint.json',folder):raise ValueError('Qualification native command differs')
    if base.sha(command[0])!=campaign.binary_identity(method,backend)[1]:raise ValueError('Qualification executable changed')
    cpu=resources.get('cpu');seconds=resources.get('elapsed_seconds')
    if type(cpu)is not int or cpu not in campaign.cpus or resources.get('threads')!=1 or resources.get('censored')is not False or resources.get('returncode',-1)<0:
        raise ValueError('Qualification process envelope differs')
    if type(seconds)not in(int,float)or not math.isfinite(seconds)or not 0<=seconds<=7200:raise ValueError('Qualification process exceeds time budget')
    actual=['taskset','-c',str(cpu),*command]
    if resources.get('command')!=actual or resources.get('command_sha256')!=hashlib.sha256(json.dumps(actual).encode()).hexdigest():raise ValueError('Qualification command resource binding differs')
    audit=base.read(folder/'AUDIT.json')
    if resources!=result.get('solver_resources')or audit!=result.get('original_audit'):raise ValueError('Qualification saved evidence differs')
    if audit.get('input_sha256')!=case['input_sha256']or audit.get('endpoint_sha256')!=base.sha(folder/'endpoint.json'):raise ValueError('Qualification audit input/endpoint binding differs')
    for key in ['classification','strict_qcqp','primal','dual','complementarity','objective']:
        if result.get(key)!=audit.get(key):raise ValueError('Qualification certified field differs: '+key)
    proof=campaign.check_endpoint(base.read(folder/'endpoint.json'),method,backend,folder)
    for key,value in proof.items():
        if result.get(key)!=value:raise ValueError('Qualification native proof differs: '+key)


def gate(config,run):
    run=Path(run);pairs=[];passed=True
    contract=base.read(run/'RUN_CONTRACT.json',{})
    campaign=Qualification(config,run,tuple(contract.get('cpus',[])))
    required_audit=['classification','strict_qcqp','primal','dual','complementarity','objective']
    for case in config['cases']:
        for policy in profiles.POLICIES:
            for backend in profiles.BACKENDS:
                paths={role:run/'tasks'/base.safe_id(case['case_id']+'__'+f'{role}_{policy}_{backend}')for role in ['REFERENCE','PROFILE']}
                results={role:base.read(path/'RESULT.json')for role,path in paths.items()}
                record=dict(case_id=case['case_id'],policy=policy,backend=backend,passed=False)
                try:
                    for role,result in results.items():
                        validate_control(campaign,case,f'{role}_{policy}_{backend}',result)
                    endpoints={role:base.read(path/'endpoint.json')for role,path in paths.items()}
                    ranks={role:base.read(path/'endpoint.json.rank_presolve.json')for role,path in paths.items()}
                    if endpoint_core(endpoints['REFERENCE'])!=endpoint_core(endpoints['PROFILE']):raise ValueError('Exact endpoint/work core differs')
                    if rank_core(ranks['REFERENCE'])!=rank_core(ranks['PROFILE']):raise ValueError('Exact presolve content differs')
                    audits={role:{k:base.read(path/'AUDIT.json').get(k)for k in required_audit}for role,path in paths.items()}
                    if endpoint_core(audits['REFERENCE'])!=endpoint_core(audits['PROFILE']):raise ValueError('Exact original numerical audit differs')
                    record.update(passed=True,endpoint_work_core_exact=True,presolve_core_exact=True,original_audit_exact=True)
                except (ValueError,KeyError,TypeError,OSError)as error:
                    record['failure']=str(error);passed=False
                record['artifacts']={role:{name:artifact(path/name)for name in ['RESULT.json','endpoint.json','AUDIT.json','solver_resources.json','COMMAND.json','endpoint.json.rank_presolve.json']if(path/name).is_file()}for role,path in paths.items()}
                pairs.append(record)
    status=base.read(run/'STATUS.json',{})
    passed=passed and status.get('state')=='completed'and status.get('done')==len(config['cases'])*12 and status.get('active_slots',0)==0
    receipt=dict(schema='bound-removal-profile-qualification-v2',passed=passed,
      observations=status.get('done',0),expected_observations=len(config['cases'])*12,
      pairs=pairs,installation=artifact(config['_path']),run_contract=artifact(run/'RUN_CONTRACT.json'),
      qualifier=artifact(__file__),clock_fields_excluded='Only dictionary keys ending _seconds, recursively; all numerical vectors, status and work counters compared exactlybinary64.',
      historical_mumps_equivalence_claimed=False,homogeneous_common_reference=True,
      qualified_source_switch='profile preserves the common unpatched implementation',
      full_ablation_authorized=passed)
    base.atomic(run/'QUALIFICATION.json',receipt)
    return receipt


def run(installation,output,cpus=tuple(range(22))):
    config=study.installation(Path(installation));output=Path(output).resolve()
    if config.get('pilot_cases')not in(12,24):raise ValueError('Qualification uses the declared pilot cases')
    if not set(cpus).issubset(__import__('os').sched_getaffinity(0)):raise ValueError('Requested CPUs unavailable')
    output.mkdir(parents=True,exist_ok=True)
    with study.locks(tuple(cpus)):
        old=base.read(output/'RUN_CONTRACT.json')
        contract=dict(schema='bound-removal-profile-qualification-run-v2',run_id=old['run_id']if old else output.name,
          installation_sha256=config['_sha256'],cases=len(config['cases']),tasks=len(config['cases'])*12,
          cells=CELLS,cpus=list(cpus),jobs=len(cpus),source_snapshot=config['source_snapshot'])
        if old and old!=contract:raise ValueError('Qualification output contract changed')
        base.atomic(output/'RUN_CONTRACT.json',contract)
        status=base.read(output/'STATUS.json',{})
        if status.get('state')!='completed':Qualification(config,output,tuple(cpus)).execute(bool(old),10.)
        receipt=gate(config,output)
    if receipt['passed']is not True:raise ValueError('Profile qualification failed; inspect '+str(output/'QUALIFICATION.json'))
    print('QUALIFICATION PROFIL VALIDÉE : '+str(output/'QUALIFICATION.json'),flush=True)
    return receipt


def admission(path,config):
    """Validate immutable control evidence before starting any ablation task."""
    from REUSE_BOUNDS import scientific_protocol
    path=Path(path).resolve(strict=True);receipt=base.read(path)
    if receipt.get('schema')!='bound-removal-profile-qualification-v2'or receipt.get('passed')is not True or receipt.get('full_ablation_authorized')is not True:
        raise ValueError('Qualification of the default source switch is required')
    for key in ['installation','run_contract','qualifier']:
        item=receipt[key];study.verify_artifact(item['path'],item['sha256'])
    if receipt['qualifier']['sha256']!=base.sha(__file__):raise ValueError('Qualification controller changed')
    reference=study.installation(Path(receipt['installation']['path']))
    if scientific_protocol(reference)!=scientific_protocol(config):raise ValueError('Qualification implementation/envelope differs')
    if receipt['expected_observations']!=len(reference['cases'])*12 or receipt['observations']!=receipt['expected_observations']:
        raise ValueError('Qualification observation count incomplete')
    if len(receipt['pairs'])!=len(reference['cases'])*6:raise ValueError('Qualification pairs incomplete')
    expected={(c['case_id'],p,b)for c in reference['cases']for p in profiles.POLICIES for b in profiles.BACKENDS}
    actual={(r['case_id'],r['policy'],r['backend'])for r in receipt['pairs']}
    if actual!=expected:raise ValueError('Qualification case/policy/backend matrix differs')
    for pair in receipt['pairs']:
        if pair.get('passed')is not True:raise ValueError('Unqualified reference/profile pair')
        if set(pair['artifacts'])!={'REFERENCE','PROFILE'}:raise ValueError('Qualification role artifacts incomplete')
        for evidence in pair['artifacts'].values():
            if set(evidence)!={'RESULT.json','endpoint.json','AUDIT.json','solver_resources.json','COMMAND.json','endpoint.json.rank_presolve.json'}:
                raise ValueError('Qualification artifact inventory incomplete')
            for item in evidence.values():study.verify_artifact(item['path'],item['sha256'])
    return artifact(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--installation',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.installation,a.out)
