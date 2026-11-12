#!/usr/bin/env python3
"""Synthetic file-receipt tests: no native solver or numerical result is used."""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import BOUNDS as study
import QUALIFIER_BOUNDS as qualification
import REUSE_BOUNDS as reuse
import STUDY_BOUNDS as controller

base=study.base


def test(out):
    if out.exists():raise ValueError('A fresh synthetic-test output folder is required')
    out.mkdir(parents=True);root=out/'source';root.mkdir()
    binary=root/'not_an_executable';binary.write_text('synthetic file identity; never executed\n')
    source=out/'pilot';source.mkdir();fullrun=out/'full'
    full_cases=study.cases();pilot_cases=study.selected_cases(12)
    configs={};common=dict(root=str(root),build_manifest_sha256='synthetic',build=dict(binary=str(binary),binary_sha256=base.sha(binary)),
      python=sys.executable,hsl_library='synthetic_hsl',hsl_library_sha256='synthetic',contract_sha256='synthetic',
      profiles=study.profiles.templates(),profile_factory_sha256='synthetic',controller_sha256='synthetic',
      auditor_sha256='synthetic',reader_sha256='synthetic',source_snapshot={},
      budgets=dict(seconds=7200,outer=5000,restoration_shared=600,tolerance=1e-6),intervention='quadratic-equality-implied-bound rule only')
    for name,cases,pilot in [('pilot_installation',pilot_cases,12),('full_installation',full_cases,None)]:
        path=out/(name+'.json');base.atomic(path,dict(synthetic=True,case_ids=[c['case_id']for c in cases]))
        configs[str(path.resolve())]=dict(copy.deepcopy(common),cases=cases,tasks=len(cases)*12,pilot_cases=pilot,
          case_matrix_sha256=study.matrix_hash(cases),_path=str(path.resolve()),_sha256=base.sha(path))
    pilot_installation=out/'pilot_installation.json';full_installation=out/'full_installation.json'
    qpath=out/'synthetic_qualification.json';base.atomic(qpath,dict(synthetic=True));qartifact=reuse.artifact(qpath)
    original_installation=study.installation;original_admission=qualification.admission
    study.installation=lambda path:copy.deepcopy(configs[str(Path(path).resolve())])
    qualification.admission=lambda path,config:qartifact
    checks=0
    try:
        pilot=study.installation(pilot_installation);pilot['_qualification']=qartifact
        base.atomic(source/'RUN_CONTRACT.json',study.run_contract(pilot,source,(0,)))
        campaign=study.BoundCampaign(pilot,source,(0,));rows=[]
        for case,method,backend in campaign.planned:
            folder=campaign.folder(case,method);folder.mkdir(parents=True)
            base.atomic(folder/'case.json',case);d=case['dimensions'];cell=study.profiles.CELLS[method]
            removed=[[0,-1,0],[0,1,0]]if cell['bounds']=='remove'else[]
            rank=dict(schema='exact-rank-presolve-v1',original_variables=d['n'],reduced_variables=d['n'],
              original_equalities=d['equalities_total'],original_inequalities=d['inequalities_total'],reduced_inequalities=d['inequalities_total'],
              equality_implied_bounds=removed,implied_bounds=[],implied_bound_sides=len(removed),equality_implied_bound_sides=len(removed),
              research_equality_implied_bounds=cell['bounds'],quadratic_equality_rule_enabled=cell['bounds']=='remove',
              affine_equality_rule_enabled=cell['policy']=='R3',radial_inequality_master_enabled=True)
            endpoint=dict(status='synthetic',x=[0.]*d['n'],outer_iterations=1,restoration_iterations=0,elapsed_seconds=.2,
              linear_solver_backend='inequality-majorants/'+backend,lazy_newton_pivoted_fallbacks=0,native_al_fallback=dict(enabled=False,attempted=False,accepted=False))
            base.atomic(folder/'endpoint.json',endpoint);base.atomic(folder/'endpoint.json.rank_presolve.json',rank)
            audit=dict(classification='nonstrict',strict_qcqp=False,input_sha256=case['input_sha256'],endpoint_sha256=base.sha(folder/'endpoint.json'),objective=1.,primal=1.,dual=1.,complementarity=1.)
            command=campaign.make_command(case,method,backend,folder/'endpoint.json',folder);actual=['taskset','-c','0',*command]
            resources=dict(cpu=0,threads=1,censored=False,returncode=0,elapsed_seconds=.3,command=actual,command_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest())
            for filename,value in [('COMMAND.json',command),('AUDIT.json',audit),('solver_resources.json',resources)]:base.atomic(folder/filename,value)
            proof=campaign._check_endpoint(endpoint,method,backend,folder)
            row=dict(case_id=case['case_id'],method=method,input_sha256=case['input_sha256'],start_binding=case['start_binding'],
              classification='nonstrict',strict_qcqp=False,installation_sha256=pilot['_sha256'],binary_sha256=base.sha(binary),
              original_audit=audit,solver_resources=resources,budget_verified=True,**campaign.observation_metadata(case,method))
            row.update({k:v for k,v in proof.items()if k!='budget_verified'});base.atomic(folder/'RESULT.json',row);rows.append(row)
            row.update({k:audit[k]for k in ['primal','dual','complementarity','objective']});base.atomic(folder/'RESULT.json',row)
        base.atomic(source/'STATUS.json',dict(state='completed',done=144,active_slots=0))
        reuse.pilot_gate(pilot_installation,source)
        stage=fullrun.with_name(fullrun.name+'.pilot_promotion_in_progress');stage.mkdir()
        base.atomic(stage/'PROMOTION_STAGE.json',dict(full_installation=reuse.artifact(full_installation),pilot_gate=reuse.artifact(source/'PILOT_GATE.json')))
        (stage/'partial_copy').write_text('Interrupted solver-free copy')
        result=reuse.promote(pilot_installation,source,full_installation,fullrun,qpath)
        assert result['reused_observations']==144 and result['remaining_native_processes']==5892;checks+=1
        assert not stage.exists()and not(fullrun/'partial_copy').exists();checks+=1
        full=study.installation(full_installation);full['_qualification']=qartifact
        fullcampaign=study.BoundCampaign(full,fullrun,(0,));fullcampaign.prepare(True)
        assert len(fullcampaign.results)==144 and fullcampaign.jobs.qsize()==5892;checks+=1
        before=base.sha(fullrun/'RESULTS.jsonl')
        repeated=reuse.promote(pilot_installation,source,full_installation,fullrun,qpath)
        assert repeated==result and base.sha(fullrun/'RESULTS.jsonl')==before;checks+=1
        first_case,first_method,first_backend=campaign.planned[0]
        first_folder=campaign.folder(first_case,first_method);original_row=base.read(first_folder/'RESULT.json')
        minimal=campaign.indeterminate(first_case,first_method,'synthetic_endpoint_unavailable')
        base.atomic(first_folder/'RESULT.json',minimal)
        resumed_pilot=study.BoundCampaign(pilot,source,(0,));resumed_pilot.prepare(True)
        assert len(resumed_pilot.results)==144 and resumed_pilot.jobs.qsize()==0;checks+=1
        base.atomic(first_folder/'RESULT.json',original_row)
        # A modified copied endpoint must stop reuse, without scheduling another solve.
        case,method,backend=campaign.planned[0];folder=fullcampaign.folder(case,method)
        saved=(folder/'endpoint.json').read_bytes();base.atomic(folder/'endpoint.json',dict(mutated=True))
        try:reuse.validate_existing(fullcampaign,case,method,base.read(folder/'RESULT.json'))
        except ValueError:checks+=1
        else:raise AssertionError('Changed copied endpoint was admitted')
        (folder/'endpoint.json').write_bytes(saved)
        # A same-command fixture with an over-budget process is rejected.
        row=copy.deepcopy(rows[0]);sourcefolder=campaign.folder(case,method);old_resources=base.read(sourcefolder/'solver_resources.json')
        altered=copy.deepcopy(old_resources);altered['elapsed_seconds']=7200.1;base.atomic(sourcefolder/'solver_resources.json',altered);row['solver_resources']=altered
        try:reuse.validate_existing(campaign,case,method,row)
        except ValueError:checks+=1
        else:raise AssertionError('Over-budget cached process was admitted')
        base.atomic(sourcefolder/'solver_resources.json',old_resources)
        # Complete state is read again with no optimizer call, including both preparations.
        calls=[]
        original_prepare=study.prepare
        def fake_prepare(args):
            calls.append(args.pilot_cases);args.out.mkdir();target=args.out/'INSTALLATION.json'
            base.atomic(target,dict(synthetic=True));configs[str(target.resolve())]=dict(copy.deepcopy(common),root=str(root),build_manifest=str(binary),
              pilot_cases=args.pilot_cases,cases=pilot_cases if args.pilot_cases else full_cases,
              _path=str(target.resolve()),_sha256=base.sha(target))
        study.prepare=fake_prepare
        try:
            for pilot_count in (12,None):
                preparation=out/('prep_'+str(pilot_count))
                a=controller.prepare_once(root,binary,preparation,[],pilot_count)
                b=controller.prepare_once(root,binary,preparation,[],pilot_count)
                assert a==b
            assert calls==[12,None];checks+=1
            interrupted=out/'prep_interrupted';interrupted.mkdir();base.atomic(interrupted/'FROZEN_RESOLUTION.json',dict(synthetic=True,incomplete=True))
            controller.prepare_once(root,binary,interrupted,[],12)
            assert(interrupted/'INSTALLATION.json').is_file()and(out/'prep_interrupted_incomplete_001/FROZEN_RESOLUTION.json').is_file();checks+=1
        finally:study.prepare=original_prepare
    finally:study.installation=original_installation;qualification.admission=original_admission
    report=dict(schema='asu-synthetic-orchestration-tests-v2',passed=True,checks=checks,
      pilot_observations_reused=144,full_pending_without_redo=5892,optimizer_invocations=0,
      local_native_numerical_runs=0,native_runtime_qualified=False,
      scope='Synthetic receipts only; exact promotion/resume, idempotent preparation, copied-artifact mutation and budget rejection.')
    base.atomic(out/'VALIDATION_ORCHESTRATION.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(test(args.out.resolve()),indent=2))
