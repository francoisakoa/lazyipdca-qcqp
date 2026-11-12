#!/usr/bin/env python3
"""Behavioral checks of failure finalization, deadlines and continuous feeding."""
from pathlib import Path
import hashlib,json,os,signal,sys,tempfile,threading,time
import CAMPAGNE as c

FIXTURE='''LAZYIPDCA_QCQP_V2
name scheduler_fixture
n 1
x0 1
lb 1
ub -inf
objective
quadratic
name objective
constant 0
linear_nnz 1
0 1
hessian_nnz 0
end_quadratic
equalities 0
inequalities 0
end_problem
'''

def case(root,name,dataset='CUTEst'):
    p=root/'data/frozen'/name/'input.qcq';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(FIXTURE)
    raw=next(x for x in p.open('rb') if x.startswith(b'x0 '))
    return dict(case_id=name,dataset=dataset,regime='I',family='fixture',input_sha256=c.sha(p),
       start_binding=dict(external_start_required=False,raw_x0_line_sha256=hashlib.sha256(raw).hexdigest()))

def run():
    checks=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);binary=root/'fake_ipopt.py'
        options=dict(c.EXPECTED_DEFAULTS,linear_solver='mumps',mumps_pivot_order=7,max_iter=5000,max_cpu_time=7200.,max_wall_time=7200.)
        payload=dict(solver='Ipopt',status='ipopt_failure',ipopt_application_status='MAXITER_EXCEEDED',ipopt_application_status_code=-1,
             adapter_version='ipopt-library-defaults-original-qcqp-v1',effective_options=options,
             x=[1.],equality_multipliers=[],inequality_multipliers=[],lower_bound_multipliers=[1.],upper_bound_multipliers=[0.],iterations=5000)
        binary.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\n'
            'p=Path(sys.argv[sys.argv.index("--problem")+1]);out=Path(sys.argv[sys.argv.index("--json")+1])\n'
            'if p.parent.name!="missing":out.write_text('+repr(json.dumps(payload))+')\nraise SystemExit(1)\n')
        binary.chmod(0o755)
        config=dict(root=str(root),default_binary=str(binary),default_binary_sha256=c.sha(binary),
            hsl_library=str(binary),hsl_library_sha256=c.sha(binary),python=sys.executable,_sha256='fixture')
        cpu=min(os.sched_getaffinity(0))
        valid=case(root,'valid');missing=case(root,'missing')
        out=root/'run';out.mkdir();campaign=c.Campaign(config,out,[valid,missing],cpus=(cpu,))
        campaign.task(cpu,valid,'ipopt_mumps_default','mumps')
        result=c.read(campaign.folder(valid,'ipopt_mumps_default')/'RESULT.json')
        assert result['solver_resources']['returncode']==1 and result['classification']=='strict' and result['strict_qcqp'] is True,result
        assert result['ipopt_own_success'] is False
        checks.append(dict(test='completed_rc1_endpoint_is_independently_audited_and_finalized',passed=True))
        campaign.task(cpu,missing,'ipopt_mumps_default','mumps')
        result=c.read(campaign.folder(missing,'ipopt_mumps_default')/'RESULT.json')
        assert result['classification']=='indeterminate' and result['strict_qcqp'] is None
        checks.append(dict(test='missing_endpoint_is_indeterminate_not_numerical_failure',passed=True))
        # Recovery of a completed nonzero solve must not invoke the optimizer again.
        f=campaign.folder(valid,'ipopt_mumps_default');(f/'RESULT.json').unlink();campaign.results.clear()
        before=c.sha(f/'solver.log');campaign.task(cpu,valid,'ipopt_mumps_default','mumps',recover=True)
        result=c.read(f/'RESULT.json')
        assert result['recovered_without_solver'] is True and result['classification']=='strict' and c.sha(f/'solver.log')==before
        checks.append(dict(test='resume_recovers_finished_rc1_result_by_audit_without_second_solve',passed=True))
        # A process ignoring TERM must be killed after the bounded grace period.
        sleeping=root/'sleep.py';sleeping.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(20)\n')
        resources=c.run_process([sys.executable,str(sleeping)],root,root/'timeout.log',.15,threading.Event(),cpu)
        assert resources['timed_out'] and resources['censored'] and resources['returncode']<0 and resources['elapsed_seconds']<3,resources
        assert not c.owned_live(resources['pid'],resources['pid_start_ticks'])
        checks.append(dict(test='wall_deadline_terminates_only_owned_process_group',passed=True,seconds=resources['elapsed_seconds']))

        a=case(root,'A','CUTEst');b=case(root,'B','CUTEst');beam=case(root,'C','multicast_beamforming');pglib=case(root,'D','PGLIB')
        events=[];event_lock=threading.Lock();queued=c.plan_cases(dict(cases=[pglib,beam,b,a]),require_full=False)
        class Feeding(c.Campaign):
            def task(self,cpu,case,method,backend,recover=False):
                self.phase(cpu,case,method,'solver')
                with event_lock:events.append(('start',case['case_id'],method,time.monotonic()))
                time.sleep(.35 if case['case_id']=='A' and backend=='ma27' else .01)
                with event_lock:events.append(('finish',case['case_id'],method,time.monotonic()))
                self.finish(case,method,dict(case_id=case['case_id'],dataset=case['dataset'],regime='I',method=method,classification='strict',strict_qcqp=True,input_sha256=case['input_sha256']))
                self.set_active(cpu,None)
        qout=root/'feeding';qout.mkdir();feeding=Feeding(config,qout,queued,cpus=(0,1));feeding.execute(interval=.02)
        long_finish=next(x[3] for x in events if x[:3]==('finish','A','ipopt_ma27_default'))
        beam_start=min(x[3] for x in events if x[:2]==('start','C'))
        assert beam_start<long_finish and len(feeding.results)==8
        checks.append(dict(test='free_slot_crosses_dataset_boundary_before_previous_dataset_finishes',passed=True))
        c.verify_effective_defaults(payload,'mumps')
        bad=dict(payload,effective_options=dict(options,mu_strategy='adaptive'))
        try:c.verify_effective_defaults(bad,'mumps')
        except ValueError:pass
        else:raise AssertionError('Nondefault adaptive setting accepted')
        checks.append(dict(test='effective_defaults_gate_rejects_adaptive_override',passed=True))
    report=dict(schema='continuous-default-ipopt-scheduler-behavior-tests-v1',passed=True,optimizer_invocations=0,tests=checks)
    (Path(__file__).parent/'SCHEDULER_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
