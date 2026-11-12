#!/usr/bin/env python3
"""Execute the actual continuous loop with22 simulated numeric/audit workers."""
import json,sys,tempfile,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import MEGA
from dataset_schedule import prepare_datasets,dataset_summaries,excluded_dey,verify_dataset_plan


def main():
    MEGA.executor.STOP.clear();lock=threading.RLock();events=[];active=set();peak=0
    barrier=threading.Barrier(22);began=time.monotonic();started_workers=0
    cases=[]
    for dataset,count in [('CUTEst',22),('multicast_beamforming',8),('PGLIB',8),('pooling_luedtke',3)]:
        for i in range(count):
            prefix={'CUTEst':'CUTEst','PGLIB':'PGLIB','pooling_luedtke':'luedtke'}.get(dataset,'multicast')
            cases.append(dict(id=prefix+'__fake'+str(i),family=dataset,reporting_family=dataset,
                source_kind='extension' if dataset=='multicast_beamforming' else 'historical',
                regime='I',input_sha256='0'*64,dimensions=dict(n=2,quadratic_inequalities=1,
                quadratic_equalities=0,linear_equalities=0,linear_inequalities=0)))
    methods=list(MEGA.METHODS)
    cases,datasets=prepare_datasets(cases,methods)
    plan=dict(cases=cases,datasets=datasets,methods=methods,jobs=22,cpus=list(range(22)),
        tasks=[dict(id=c['id']+'__'+method,case=c,method=method,contract_sha256='fake') for c in cases for method in methods])
    def record(event,**fields):
        with lock:events.append(dict(event=event,seconds=time.monotonic()-began,**fields))
    def worker(out,plan,task,cpu,guard):
        nonlocal peak,started_workers
        with lock:
            assert cpu not in active,'CPU assigned twice concurrently'
            active.add(cpu);peak=max(peak,len(active))
            started_workers+=1;initial=started_workers<=22
        record('start',id=task['id'],dataset=task['case']['dataset_id'],cpu=cpu,method=task['method'])
        if initial:barrier.wait(timeout=10)
        time.sleep(.7 if task['id']=='CUTEst__fake0__R3' else .005)
        uncertain=task['id']=='PGLIB__fake0__ipopt_mumps'
        receipt=str(Path('tasks')/task['id']/'PREPOST_INTEGRITY.json')
        result=dict(case_id=task['case']['id'],method=task['method'],strict_qcqp=None if uncertain else False,
            integrity_verified=True,integrity_receipt=receipt,evidence_sha256={receipt:'fake'},
            observation_classification='indeterminate_observation' if uncertain else 'audited_original_qcqp_nonstrict')
        with MEGA.executor.RESULT_LOCK:MEGA.executor.RUN_RESULTS[str(out)][task['id']]=result
        with lock:active.remove(cpu)
        record('end',id=task['id'],dataset=task['case']['dataset_id'],cpu=cpu)
        return result
    def report(dataset,records):
        record('report-start',dataset=dataset)
        time.sleep(.15)
        record('report-end',dataset=dataset)
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)
        status=MEGA.continuous_loop(out,plan,None,worker=worker,restore=lambda *a:None,report=report)
        assert status['state']=='completed' and status['done']==328 and status['total']==328
        assert status['nonstrict']==327 and status['indeterminate']==1
        assert status['by_dataset']['PGLIB']['by_method']['ipopt_mumps']['indeterminate']==1
        assert status['by_dataset']['PGLIB']['by_method']['ipopt_mumps']['failed']==7
        assert not active and peak==22
        slow=next(e['seconds'] for e in events if e['event']=='end' and e['id']=='CUTEst__fake0__R3')
        starts={d:min(e['seconds'] for e in events if e['event']=='start' and e['dataset']==d) for d in ['multicast_beamforming','PGLIB','pooling_luedtke']}
        assert all(v<slow for v in starts.values()),'Dataset barrier detected'
        assert {e['cpu'] for e in events if e['event']=='start'}==set(range(22))
        assert {e['method'] for e in events if e['event']=='start'}==set(MEGA.METHODS)
        bad=dict(plan,cases=plan['cases']+[dict(plan['cases'][0],id='dey__randstd11',family='pooling_dey')])
        try:verify_dataset_plan(bad)
        except ValueError:pass
        else:raise AssertionError('Dey in denominator was not refused')
    answer=dict(test='actual MEGA.continuous_loop, fake numeric and audit workers only',pass_=True,
        workers=22,cpus=list(range(22)),methods=methods,peak=peak,completed=328,no_dataset_barrier=True,
        next_dataset_start_seconds=starts,last_slow_CUTEst_end_seconds=slow,
        uncertain_audit_counted_indeterminate=True,Dey_denominator_injection_refused=True,
        solvers_launched=0,remote_run=False,numerical_qualification=False,events=events)
    answer['pass']=answer.pop('pass_')
    receipt=Path(__file__).resolve().parents[1]/'validation/campaign/CONTINUOUS_22_TEST.json'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(answer,indent=2)+'\n')
    print(json.dumps({k:v for k,v in answer.items() if k!='events'},indent=2))


if __name__=='__main__':main()
