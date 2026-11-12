#!/usr/bin/env python3
"""Synthetic controller tests. No QCQP optimizer is invoked."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import contextlib, io, json, os, sys, unittest
import CHRONO_CALME as C

class ChronoTests(unittest.TestCase):
    def test_preparation_resumes_existing_strictly_empty_output(self):
        with TemporaryDirectory() as t:
            out=Path(t)/'preparation'
            C.empty_preparation_output(out);out.mkdir()
            # This is the exact state after interruption between mkdir and installation.
            self.assertEqual(C.empty_preparation_output(out),out)
            C.write(out/'INSTALLATION.json',{'completed':True})
            with self.assertRaises(ValueError):C.empty_preparation_output(out)
            self.assertEqual(C.read(out/'INSTALLATION.json'),{'completed':True})
            (out/'INSTALLATION.json').unlink();(out/'INSTALLATION.json.tmp.123').write_text('partial')
            with self.assertRaises(ValueError):C.empty_preparation_output(out)
            self.assertEqual((out/'INSTALLATION.json.tmp.123').read_text(),'partial')

    def test_exact_frozen_cohorts_and_no_duplicate_schedule(self):
        cohort=C.load_cohort();schedule=C.schedule(cohort)
        self.assertEqual(cohort['counts']['cohorts'],{'ma27_I':30,'ma27_M':89,'mumps_I':81,'mumps_M':75})
        self.assertEqual(len(schedule),1650);self.assertEqual(len({s['nominal_id'] for s in schedule}),1650)
        for job in cohort['jobs']:
            self.assertEqual({s['repetition'] for s in schedule if s['job_id']==job['job_id']},{1,2,3})
        self.assertEqual(schedule,C.schedule(cohort))
        # Every native/reference pair reverses its order in repetitions 1 and 2.
        for pair in cohort['pairs']:
            a,b=pair['native_job_id'],pair['reference_job_id']
            positions={r:{s['job_id']:i for i,s in enumerate(schedule) if s['repetition']==r} for r in (1,2)}
            self.assertNotEqual(positions[1][a]<positions[1][b],positions[2][a]<positions[2][b])

    def test_systemwide_process_and_short_lived_cpu_detection(self):
        old=dict(time=0.,busy=10.,processes={(42,1):dict(cpu=0.,group=42,comm='solver')})
        now=dict(time=1.,busy=11.5,processes={(42,1):dict(cpu=1.,group=42,comm='solver'),
            (2223,9):dict(cpu=.5,group=2223,comm='foreign_on_cpu23')})
        row=C.quiet_interval(old,now,(42,))
        self.assertTrue(row['process_contamination']);self.assertTrue(row['aggregate_contamination'])
        self.assertEqual(row['foreign_processes'][0]['pid'],2223)
        # A short-lived foreign process is no longer in /proc; aggregate still catches it.
        del now['processes'][(2223,9)]
        row=C.quiet_interval(old,now,(42,))
        self.assertFalse(row['process_contamination']);self.assertTrue(row['aggregate_contamination'])
        # A burst must not disappear into a 100-second whole-run average.
        row['solver_observed']=True
        self.assertTrue(C.contaminated_measurement([row],.5,100.,0.,.5))
        # A vanished solver's CPU is corrected by wait4 in the tail, not called foreign.
        row['solver_observed']=False
        self.assertFalse(C.contaminated_measurement([row],0.,100.,0.,.5))
        self.assertTrue(C.contaminated_measurement([row],.4,100.,.4,.5))

    def test_frozen_start_and_input_provenance(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'in.qcq';p.write_bytes(b'header\nx0 1 2\n')
            job=dict(case_id='test',input_sha256=C.sha(p),start_sha256='semantic',
                start_binding=dict(start_sha256='semantic',raw_x0_line_sha256=C.hashlib.sha256(b'x0 1 2\n').hexdigest()))
            C.verify_start(p,job);p.write_bytes(b'header\nx0 2 1\n')
            with self.assertRaises(ValueError):C.verify_start(p,job)

    def test_native_commands_preserve_historical_wrapper_options(self):
        ctx=C.read(C.HERE/'evidence/CONTEXTE_HISTORIQUE.json')
        d=dict(R3=ctx['R3'],hsl_library='/pinned/hsl',inputs={'case':{'input':'/frozen/input','external_start':None}},
            configs={'R3_ma27':{'binary':'/old/ma27','backend':'ma27','native_al_fallback_policy':'off'},
                     'R3_mumps':{'binary':'/old/mumps','backend':'mumps','native_al_fallback_policy':'off'}})
        for backend in ('ma27','mumps'):
            job=dict(role='native',config='R3_'+backend,case_id='case',backend=backend)
            cmd=C.command_for(d,job,Path('/output/attempt'))
            for flag,value in (('--linear-backend',backend),('--linear-metis','1'),('--native-al-fallback','off'),
                ('--standard-r10-policy','auto'),('--max-outer','5000'),('--time-limit','7200'),('--dca-rank-implied-bounds','1')):
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertNotIn('mumps_metis',cmd)
            if backend=='ma27':self.assertIn('--dca-rank-presolve-report',cmd);self.assertIn('--hsl-library',cmd)
        with patch.dict(os.environ,{'LAZY_HIDDEN_TUNING':'1','IPDCA_HIDDEN_TUNING':'1','OMP_NUM_THREADS':'99'}):
            env=C.environment({});self.assertNotIn('LAZY_HIDDEN_TUNING',env);self.assertNotIn('IPDCA_HIDDEN_TUNING',env)
            self.assertEqual(env['OMP_NUM_THREADS'],'1')

    def test_process_scope_affinity_and_budget(self):
        with TemporaryDirectory() as t:
            p=Path(t);cpu=min(os.sched_getaffinity(0))
            code='import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'
            r=C.run_process([sys.executable,'-c',code],p,C.environment({}),[False],timeout=5,cpu=cpu)
            self.assertEqual(r['returncode'],0);self.assertTrue(C.budget_verified(r))
            self.assertEqual(json.loads((p/'process.log').read_text()),[cpu])
            self.assertGreater(r['elapsed_seconds'],0);self.assertEqual(r['threads'],1)
            bad=dict(r,elapsed_seconds=7200.001);self.assertFalse(C.budget_verified(bad))
            bad=dict(r,user_cpu_seconds=7200.1);self.assertFalse(C.budget_verified(bad))
            q=p/'timeout';q.mkdir()
            r=C.run_process([sys.executable,'-c','import time;time.sleep(5)'],q,C.environment({}),[False],timeout=.1,cpu=cpu)
            self.assertTrue(r['censored']);self.assertTrue(r['timed_out']);self.assertFalse(C.budget_verified(r))

    def fixture_attempt(self,folder,task,install_sha,quiet=True,classification='nonstrict'):
        folder.mkdir(parents=True,exist_ok=True)
        resources=dict(elapsed_seconds=2.,interrupted=False,censored=False,returncode=0)
        C.write(folder/'solver_resources.json',resources)
        r=dict(nominal_id=task['nominal_id'],job_id=task['job_id'],case_id='case',config='R3_ma27',
            repetition=task['repetition'],installation_sha256=install_sha,quiet_valid=quiet,finalized=True,
            classification=classification,solver_resources=resources,seconds_process=2.,
            raw_artifacts={'solver_resources.json':C.artifact(folder/'solver_resources.json')})
        r['attempt_content_sha256']=C.digest(r);C.write(folder/'ATTEMPT.json',r)
        return r

    def test_crash_after_clean_numeric_failure_commit_is_recovered_without_rerun(self):
        with TemporaryDirectory() as t:
            out=Path(t);C.write(out/'INSTALLATION.json',{'synthetic':True});ish=C.sha(out/'INSTALLATION.json')
            task=dict(nominal_id='job_rep1',job_id='job',repetition=1)
            old=out/'runs/job_rep1/attempt_0001'
            self.fixture_attempt(old,task,ish,classification='nonstrict')
            d={'schedule':[task]};cohort={'jobs':[{'job_id':'job'}]}
            with patch.object(C,'load_installation',return_value=d),patch.object(C,'load_cohort',return_value=cohort),\
                 patch.object(C,'attempt_result',side_effect=AssertionError('A completed numerical result must not be rerun')):
                self.assertEqual(C.run(out),1);self.assertEqual(C.run(out),1)
            result=C.read(out/'runs/job_rep1/RESULT.json')
            self.assertEqual(result['classification'],'nonstrict')
            self.assertEqual(len(list((out/'runs/job_rep1').glob('attempt_*'))),1)
            self.assertTrue(C.result_complete(out/'runs/job_rep1/RESULT.json',ish,'job_rep1'))

    def test_contamination_retried_but_retained_and_not_best_selected(self):
        with TemporaryDirectory() as t:
            out=Path(t);C.write(out/'INSTALLATION.json',{'synthetic':True});ish=C.sha(out/'INSTALLATION.json')
            task=dict(nominal_id='job_rep1',job_id='job',repetition=1)
            self.fixture_attempt(out/'runs/job_rep1/attempt_0001',task,ish,quiet=False,classification='strict')
            called=[]
            def attempt(d,j,task,folder,stop):
                called.append(folder)
                C.write(folder/'solver_resources.json',dict(elapsed_seconds=3.,interrupted=False,censored=False,returncode=0))
                return dict(nominal_id=task['nominal_id'],job_id='job',case_id='case',config='R3_ma27',repetition=1,
                    quiet_valid=True,solver_resources=C.read(folder/'solver_resources.json'),classification='nonstrict',seconds_process=3.)
            with patch.object(C,'load_installation',return_value={'schedule':[task]}),patch.object(C,'load_cohort',return_value={'jobs':[{'job_id':'job'}]}),\
                 patch.object(C,'attempt_result',side_effect=attempt),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.run(out),1);self.assertEqual(C.run(out),1)
            self.assertEqual(len(called),1)
            self.assertEqual(C.read(out/'runs/job_rep1/RESULT.json')['classification'],'nonstrict')
            self.assertTrue((out/'runs/job_rep1/attempt_0001/ATTEMPT.json').exists())
            self.assertEqual(len(list((out/'runs/job_rep1').glob('attempt_*'))),2)

    def test_artifact_mutation_refuses_recovery(self):
        with TemporaryDirectory() as t:
            folder=Path(t)/'attempt';task={'nominal_id':'nominal','job_id':'job','repetition':1}
            self.fixture_attempt(folder,task,'install')
            (folder/'solver_resources.json').write_text('{}')
            with self.assertRaises(ValueError):C.commit_attempt(folder,Path(t)/'RESULT.json','install','nominal')

    def test_live_previous_owned_process_blocks_duplicate_launch(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'runs/job_rep1/attempt_0001/OWNED_PROCESS.json'
            r=dict(pid=os.getpid(),start_ticks=C.process_start_ticks(os.getpid()),finished=False)
            C.write(p,r)
            with self.assertRaises(ValueError):C.ensure_no_owned_live(t)
            C.write(p,dict(r,start_ticks=r['start_ticks']+1));C.ensure_no_owned_live(t)
            C.write(p,dict(r,finished=True));C.ensure_no_owned_live(t)

    def test_ratio_quartiles_are_casewise_not_reciprocal_summary(self):
        self.assertEqual(C.quartiles([.5,1.,2.,4.]),(.875,2.5))
        self.assertEqual(C.quartiles([2.]),(2.,2.))
        self.assertEqual(C.quartiles([]),(None,None))
        self.assertNotEqual(C.statistics.median([1/v for v in [1,2,10,20]]),1/C.statistics.median([1,2,10,20]))

    def test_summary_keeps_all_attempt_costs_and_three_rep_median(self):
        with TemporaryDirectory() as t:
            out=Path(t);cohort=C.load_cohort();tasks=C.schedule(cohort)
            d=dict(schedule=tasks);d['installation_content_sha256']=C.digest(d);C.write(out/'INSTALLATION.json',d)
            ish=C.sha(out/'INSTALLATION.json');pair=cohort['pairs'][0]
            for task in tasks:
                if task['job_id'] not in (pair['native_job_id'],pair['reference_job_id']):continue
                folder=out/'runs'/task['nominal_id']/'attempt_0001'
                r=self.fixture_attempt(folder,task,ish,classification='strict')
                r['config']='R3_ma27' if task['job_id']==pair['native_job_id'] else 'ipopt_ma27_monotone_aligned'
                r['seconds_process']=(task['repetition'] if task['job_id']==pair['native_job_id'] else 2*task['repetition'])
                r['solver_resources']['elapsed_seconds']=r['seconds_process'];C.write(folder/'solver_resources.json',r['solver_resources'])
                r['raw_artifacts']={'solver_resources.json':C.artifact(folder/'solver_resources.json')}
                r.pop('attempt_content_sha256');r['attempt_content_sha256']=C.digest(r);C.write(folder/'ATTEMPT.json',r)
                C.commit_attempt(folder,folder.parent/'RESULT.json',ish,task['nominal_id'])
            # Historical cohort is retained even though only one pair has new data.
            with contextlib.redirect_stdout(io.StringIO()):report=C.summarize(out)
            self.assertFalse(report['complete']);self.assertEqual(report['nominal_repetitions_completed'],6)
            p=next(p for p in report['pairs'] if p['pair_id']==pair['pair_id'])
            self.assertEqual(p['native_median_seconds'],2);self.assertEqual(p['reference_median_seconds'],4)
            self.assertEqual(p['ratio_R3_over_reference'],.5);self.assertEqual(p['speedup_reference_over_R3'],2)
            self.assertEqual(report['all_attempt_solver_seconds'],18)
            self.assertEqual(sum(x['frozen_pairs'] for x in report['cohorts']),275)
            self.assertTrue((out/'OBSERVATIONS_CHRONO.csv').exists())

if __name__=='__main__':unittest.main(verbosity=2)
