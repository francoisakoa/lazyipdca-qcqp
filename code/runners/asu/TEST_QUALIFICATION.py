#!/usr/bin/env python3
"""Reject missing/changed qualification evidence using synthetic files only."""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import BOUNDS as study
import QUALIFIER_BOUNDS as q

base=study.base


def test(out):
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True);binary=out/'not_executable';binary.write_text('Synthetic identity; never executed\n')
    installation=out/'INSTALLATION.json';base.atomic(installation,dict(synthetic=True))
    case=next(c for c in study.selected_cases(12)if c['dimensions']['n']==64)
    config=dict(root=str(out),cases=[case],_path=str(installation),_sha256=base.sha(installation),hsl_library='synthetic',
      build=dict(binary=str(binary),binary_sha256=base.sha(binary),reference_binary=str(binary),reference_binary_sha256=base.sha(binary)))
    run=out/'controls';run.mkdir();base.atomic(run/'RUN_CONTRACT.json',dict(run_id='synthetic-controls',cpus=[0]))
    campaign=q.Qualification(config,run,(0,));d=case['dimensions']
    for method,c in q.CELLS.items():
        folder=campaign.folder(case,method);folder.mkdir(parents=True)
        base.atomic(folder/'case.json',case)
        removed=[[0,-1,0],[0,1,0]]if c['policy']=='HH'else[]
        rank=dict(schema='exact-rank-presolve-v1',original_variables=d['n'],reduced_variables=d['n'],
          original_equalities=d['equalities_total'],original_inequalities=d['inequalities_total'],reduced_inequalities=d['inequalities_total'],
          equality_implied_bounds=removed,implied_bounds=[],implied_bound_sides=len(removed),equality_implied_bound_sides=len(removed))
        if c['role']=='PROFILE':rank.update(research_equality_implied_bounds='profile',quadratic_equality_rule_enabled=c['policy']=='HH',
            radial_inequality_master_enabled=True,affine_equality_rule_enabled=c['policy']=='R3')
        endpoint=dict(x=[0.]*d['n'],status='synthetic',outer_iterations=1,restoration_iterations=0,
          elapsed_seconds=.2 if c['role']=='REFERENCE'else.3,linear_solver_backend='inequality-majorants/'+c['backend'],
          lazy_newton_pivoted_fallbacks=0,native_al_fallback=dict(enabled=False,attempted=False,accepted=False))
        base.atomic(folder/'endpoint.json',endpoint);base.atomic(folder/'endpoint.json.rank_presolve.json',rank)
        command=campaign.make_command(case,method,c['backend'],folder/'endpoint.json',folder);actual=['taskset','-c','0',*command]
        resources=dict(cpu=0,threads=1,censored=False,returncode=0,elapsed_seconds=.4,command=actual,command_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest())
        audit=dict(classification='nonstrict',strict_qcqp=False,primal=1.,dual=1.,complementarity=1.,objective=1.,
          input_sha256=case['input_sha256'],endpoint_sha256=base.sha(folder/'endpoint.json'))
        for name,value in [('COMMAND.json',command),('solver_resources.json',resources),('AUDIT.json',audit)]:base.atomic(folder/name,value)
        row=dict(case_id=case['case_id'],method=method,input_sha256=case['input_sha256'],start_binding=case['start_binding'],
          installation_sha256=config['_sha256'],binary_sha256=base.sha(binary),original_audit=audit,solver_resources=resources)
        row.update(audit);row.update(campaign.observation_metadata(case,method));row.update(campaign.check_endpoint(endpoint,method,c['backend'],folder))
        base.atomic(folder/'RESULT.json',row)
    base.atomic(run/'STATUS.json',dict(state='completed',done=12,active_slots=0))
    assert q.gate(config,run)['passed']is True;checks=1
    folder=campaign.folder(case,'PROFILE_R3_ma27');saved={name:(folder/name).read_bytes()for name in ['COMMAND.json','solver_resources.json','RESULT.json','endpoint.json','AUDIT.json']}
    (folder/'COMMAND.json').unlink();assert q.gate(config,run)['passed']is False;checks+=1
    for name,data in saved.items():(folder/name).write_bytes(data)
    resources=base.read(folder/'solver_resources.json');resources['threads']=2;base.atomic(folder/'solver_resources.json',resources)
    row=base.read(folder/'RESULT.json');row['solver_resources']=resources;base.atomic(folder/'RESULT.json',row)
    assert q.gate(config,run)['passed']is False;checks+=1
    for name,data in saved.items():(folder/name).write_bytes(data)
    endpoint=base.read(folder/'endpoint.json');endpoint['outer_iterations']=2;base.atomic(folder/'endpoint.json',endpoint)
    audit=base.read(folder/'AUDIT.json');audit['endpoint_sha256']=base.sha(folder/'endpoint.json');base.atomic(folder/'AUDIT.json',audit)
    row=base.read(folder/'RESULT.json');row['original_audit']=audit;base.atomic(folder/'RESULT.json',row)
    assert q.gate(config,run)['passed']is False;checks+=1
    report=dict(schema='asu-synthetic-qualification-tests-v2',passed=True,checks=checks,optimizer_invocations=0,
      native_runtime_qualified=False,scope='Synthetic pair passes clock-only differences; missing command, thread drift, and work-counter drift are rejected.')
    base.atomic(out/'VALIDATION_QUALIFICATION.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(test(args.out.resolve()),indent=2))
