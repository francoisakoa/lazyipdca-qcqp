#!/usr/bin/env python3
"""One real native executor/auditor integration; no MA27 or campaign qualification."""
import argparse,hashlib,json,os,re,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import MEGA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE.parent)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--method',choices=['R3','ipopt_mumps'],default='R3')
    args=parser.parse_args();root=args.root.resolve();sha=MEGA.sha
    cid='cops_minsurf_epi_nx0010_ny0010'
    source_input=root/'test_data/protected81'/cid/'input.qcq'
    binary=root/'install/bin/lazyipdca_qcqp_cli'
    if not binary.exists():binary=root/'build/lazyipdca_qcqp_cli'
    native_binary=binary.resolve(strict=True)
    binary=native_binary if args.method=='R3' else (root/'ipopt/build/ipopt_qcqp_cli').resolve(strict=True)
    out=(args.out or root/'validation'/('native_executor_'+str(time.time_ns()))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    runtime=out/'runtime';campaign=runtime/'campaign';campaign.mkdir(parents=True)
    for name in ('resource_worker.py','audit_worker.py','observation_contract.py','ipopt_worker.py'):
        shutil.copy2(HERE/name,campaign/name)
    frozen=runtime/'data/input.qcq';frozen.parent.mkdir();shutil.copy2(source_input,frozen)
    before=sha(frozen);binary_before=sha(binary)
    auditor=root/'campaign/common/original_auditor.py';precise=root/'validation_tools/precise_original_auditor.py'
    configuration=dict(component_test_only=True,campaign_installation_success_not_claimed=True,
        binary=str(native_binary),binary_sha256=sha(native_binary),python=sys.executable,
        native_auditor=str(auditor),native_auditor_sha256=sha(auditor),
        precise_auditor=str(precise),precise_auditor_sha256=sha(precise),
        native_al_fallback_policy='off',environment={})
    if args.method=='ipopt_mumps':
        build=MEGA.read(root/'ipopt/build/BUILD_IPOPT.json')
        qualification=root/'validation/ipopt_mumps/real_qualification/QUALIFICATION.json'
        if not qualification.is_file():raise ValueError('Real MUMPS qualification required before component test')
        configuration.update(ipopt_binary=str(binary),ipopt_binary_sha256=sha(binary),
            ipopt_version=re.search(r'ipopt=(\d+\.\d+\.\d+)',build['version']).group(1),
            mumps_qualification=str(qualification),mumps_qualification_sha256=sha(qualification),
            ipopt_mumps_linear_solver='mumps',ipopt_mumps_pivot_order=5,
            ipopt_command=['{python}',str(campaign/'ipopt_worker.py'),'--case-json','{case_json}',
            '--input','{input}','--endpoint','{endpoint}','--seconds','{seconds}','--max-outer','{max_outer}'])
        configuration['environment']['LD_LIBRARY_PATH']=str(Path(build['ipopt_library']).parent)
        sys.path.insert(0,str(root))
    MEGA.write(runtime/'INSTALLATION.json',configuration)
    case=dict(id=cid,family='COPS_minimal_surface',reporting_family='COPS_minimal_surface',
        regime='I',source_kind='extension',input=str(frozen),input_sha256=before,
        external_start_required=False,dimensions=dict(n=386,quadratic_inequalities=242,
        quadratic_equalities=0,linear_equalities=48,linear_inequalities=0),dataset_id='COPS_minimal_surface')
    profiles=MEGA.read(HERE/'profiles.json')
    plan=dict(cases=[case],methods=[args.method],profiles=profiles,installation=configuration,
        seconds=7200,max_outer=5000,shared_restoration=600,tolerance=1e-6,threads=1,al_mode='off',
        runtime_sha256={str(runtime/'INSTALLATION.json'):sha(runtime/'INSTALLATION.json')},source_sha256={},
        datasets=[dict(id='COPS_minimal_surface')])
    plan['execution_binding_sha256']=MEGA.execution_binding(plan)
    task=dict(id=cid+'__'+args.method,case=case,method=args.method,contract_sha256=MEGA.task_digest(case,args.method,plan['execution_binding_sha256']))
    guard=MEGA.RuntimeGuard({**plan['runtime_sha256'],str(binary):binary_before,str(auditor):sha(auditor),str(precise):sha(precise),
        **{str(campaign/n):sha(campaign/n) for n in ('resource_worker.py','audit_worker.py','observation_contract.py','ipopt_worker.py')}})
    MEGA.executor.HERE=campaign
    MEGA.executor.STOP.clear()
    cpu=min(os.sched_getaffinity(0));result=MEGA.guarded_run_task(out,plan,task,cpu,guard)
    if result is None:raise AssertionError('Native component interrupted')
    audit=result.get('audit',{})
    passed=result.get('strict_qcqp') is True and result.get('integrity_verified') is True and audit.get('precision_agreement') is True
    passed=passed and before==sha(frozen)==sha(source_input) and binary_before==sha(binary)
    passed=passed and result.get('route_policy_check',{}).get('passed') is True
    answer=dict(pass_=passed,scope='one real '+args.method+' profile/resource-worker/endpoint/scalar+interval150-audit/normalization/prepost-hash component',
        case_id=cid,binary_sha256=binary_before,input_sha256=before,cpu=cpu,
        strict_original=result.get('strict_qcqp'),classification=result.get('observation_classification'),
        primal=audit.get('primal'),dual=audit.get('dual'),complementarity=audit.get('complementarity'),
        precision_agreement=audit.get('precision_agreement'),budget=audit.get('budget_and_timing'),
        work=result.get('work'),audit_work=result.get('audit_work'),integrity_verified=result.get('integrity_verified'),
        method=args.method,native_profile='R3' if args.method=='R3' else None,native_route_pass=result.get('route_policy_check',{}).get('passed'),
        MA27_qualification_claimed=False,full_campaign_launched=False,Dey_run=False,
        frozen_input_and_raw_start_unchanged=True,external_endpoint_start=False)
    answer['pass']=answer.pop('pass_');MEGA.write(out/'COMPONENT_TEST.json',answer)
    print(json.dumps(answer,indent=2))
    if not passed:raise AssertionError('Native executor integration did not qualify original KKT; see '+str(out))


if __name__=='__main__':main()
