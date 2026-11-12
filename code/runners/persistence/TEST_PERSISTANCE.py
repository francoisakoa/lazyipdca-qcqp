#!/usr/bin/env python3
"""Adversarial synthetic controller checks; no QCQP optimizer is invoked."""
from pathlib import Path
import contextlib
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import PERSISTANCE as p


def resources(command=None):
    command=command or ['synthetic']
    return dict(command=command,command_sha256=p.quiet.digest(command),elapsed_seconds=.2,
        user_cpu_seconds=.01,system_cpu_seconds=.01,returncode=0,censored=False,timed_out=False,interrupted=False,
        quiet_observation=dict(contaminated=False))


def endpoint():
    return dict(status='solved',x=[-0.,1.],outer_iterations=3,restoration_iterations=0,elapsed_seconds=.1,
        dca_filter_models=3,dca_filter_curvature_reuses=2,dca_filter_majorant_curvature_reuses=2,
        linear_solver_backend='inequality-majorants/ma27',lazy_newton_factor_attempts=3,
        dca_filter_pivoted_backend=False,native_al_fallback=dict(policy='off',attempted=False,accepted=False,enabled=False))


def report_rows(mode,anchors=3):
    preparations=1 if mode=='reuse' else anchors
    return [dict(schema='research-curvature-cache-summary-v1',mode=mode,model_instance=1,
        preparations=preparations,reanchors=anchors,quadratic_row_builds=5*preparations,corrected_rows=2,
        construction_seconds=.01*preparations,rebuild_reset_seconds=.001 if mode=='rebuild' else 0.,
        trace_records=anchors*3,reuse_hits=anchors-1 if mode=='reuse' else 0,complete=True),
        dict(schema='research-curvature-cache-session-v1',mode=mode,model_instances=1,solver_returned=True)]


def trace_rows(mode,anchors=3):
    out=[]
    for anchor in range(1,anchors+1):
        for event in ('reanchor','controller','accepted'):
            out.append(dict(schema='research-curvature-cache-trace-v1',event=event,mode=mode,model_instance=1,
                reanchor=anchor,outer=anchor-1,accepted_steps=anchor-1,origin='none' if event!='accepted' else 'main-accepted',
                correction_identity='a'*64,state_identity=p.quiet.digest([anchor,event])))
    return out


def write_lines(path,rows):
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))


def fixture_context():
    context=p.quiet.read(p.HERE/'support_chronos/evidence/CONTEXTE_HISTORIQUE.json')
    pair=copy.deepcopy(p.load_cohort()['pairs'][0]);pair['backend']='ma27'
    d=dict(R3=context['R3'],binary='/synthetic/research',reference_binary='/synthetic/reference',
        binary_sha256='1'*64,reference_binary_sha256='2'*64,hsl_library='/synthetic/hsl.so',
        inputs={pair['case_id']:dict(input='/synthetic/input.qcq',external_start=None)},runtime_environment={})
    return d,pair


def qualification_fixture(root):
    d,pair=fixture_context();group={}
    for mode in p.CONTROL_MODES:
        folder=root/mode;folder.mkdir()
        ep=endpoint()
        if mode=='rebuild':ep['dca_filter_curvature_reuses']=ep['dca_filter_majorant_curvature_reuses']=0
        p.quiet.write(folder/'endpoint.json',ep)
        command=p.command_for(d,pair,mode,folder,True);res=resources(command)
        p.quiet.write(folder/'solver_resources.json',res)
        p.quiet.write(folder/'COMMAND.json',dict(command=command,numerical_runtime_environment=p.quiet.THREAD_ENV))
        audit=dict(classification='strict',strict_qcqp=True,primal=0.,dual=1e-7,complementarity=1e-8,objective=2.,
            audit_available=True,original_KKT_endpoint_strict=True,input_sha256=pair['input_sha256'],endpoint_sha256=p.quiet.sha(folder/'endpoint.json'))
        p.quiet.write(folder/'AUDIT.json',audit)
        if mode!='reference':
            write_lines(folder/'curvature_report.jsonl',report_rows(mode));write_lines(folder/'curvature_trace.jsonl',trace_rows(mode))
        group[mode]=dict(classification='strict',budget_verified=True,errors=[],solver_resources=res,
            residuals={k:audit[k] for k in ('primal','dual','complementarity')},
            artifacts={'endpoint.json':p.quiet.artifact(folder/'endpoint.json')})
    return d,pair,group


def synthetic_attempt(task,classification='nonstrict',contaminated=False):
    res=resources();res['quiet_observation']['contaminated']=contaminated
    return dict(schema='curvature-persistence-attempt-v1',nominal_id=task['nominal_id'],job_id=task['job_id'],
        pair_id=task['pair_id'],mode=task['mode'],phase=task['phase'],repetition=task['repetition'],
        quiet_valid=not contaminated,solver_resources=res,classification=classification,seconds_process=.2)


class PersistenceTests(unittest.TestCase):
    def test_fixed_cohort_schedule_balanced_unique(self):
        cohort=p.load_cohort();controls,timed=p.schedules(cohort)
        self.assertEqual(len(controls),72);self.assertEqual(len(timed),144)
        self.assertEqual(len({t['nominal_id'] for t in controls+timed}),216)
        for pair in cohort['pairs']:
            rows=[r for r in timed if r['pair_id']==pair['pair_id']]
            self.assertEqual({(r['mode'],r['repetition']) for r in rows},{(m,n) for m in p.MODES for n in range(1,4)})
            self.assertEqual([r['mode'] for r in rows if r['repetition']==1],list(reversed([r['mode'] for r in rows if r['repetition']==2])))

    def test_same_binary_and_command_intervention_only(self):
        d,pair=fixture_context();folder=Path('/synthetic/run')
        a=p.command_for(d,pair,'reuse',folder,False);b=p.command_for(d,pair,'rebuild',folder,False)
        self.assertEqual([(x,y) for x,y in zip(a,b) if x!=y],[('reuse','rebuild')])
        self.assertEqual(a[0],b[0]);self.assertNotIn('--research-curvature-trace',a)
        self.assertIn('--research-curvature-report',a)
        ref=p.command_for(d,pair,'reference',folder,True)
        self.assertNotIn('--research-curvature-cache',ref);self.assertEqual(ref[0],d['reference_binary'])
        pair['backend']='mumps';m=p.command_for(d,pair,'reuse',folder,False)
        self.assertEqual(m[m.index('--linear-backend')+1],'mumps');self.assertEqual(m[m.index('--linear-metis')+1],'1')

    def test_exact_core_preserves_signed_zero_and_numerical_work(self):
        a=endpoint();b=copy.deepcopy(a);b['elapsed_seconds']=3.
        self.assertEqual(p.exact_core(a),p.exact_core(b))
        b['x'][0]=0.;self.assertNotEqual(p.exact_core(a),p.exact_core(b))
        b=copy.deepcopy(a);b['dca_filter_curvature_reuses']=0
        self.assertNotEqual(p.exact_core(a),p.exact_core(b));self.assertEqual(p.exact_core(a,True),p.exact_core(b,True))
        b['lazy_newton_factor_attempts']=4;self.assertNotEqual(p.exact_core(a,True),p.exact_core(b,True))
        self.assertNotEqual(p.exact_core({'counter':1}),p.exact_core({'counter':True}))
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'a.json';second=Path(tmp)/'b.json'
            first.write_text('{"x":[-0],"elapsed_seconds":1}')
            second.write_text('{"x":[0],"elapsed_seconds":2}')
            self.assertNotEqual(p.endpoint_core(first),p.endpoint_core(second))

    def test_reports_reject_missing_closure_wrong_mode_missing_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.jsonl';write_lines(path,report_rows('rebuild'))
            self.assertEqual(p.cache_report(path,'rebuild',True)['totals']['preparations'],3)
            for mutate in ('closure','mode','complete','preparations','rows','trace_in_timing'):
                rows=report_rows('rebuild')
                if mutate=='closure':rows.pop()
                elif mutate=='mode':rows[-1]['mode']='reuse'
                elif mutate=='complete':rows[0]['complete']=False
                elif mutate=='preparations':rows[0]['preparations']=1
                elif mutate=='rows':rows[0]['corrected_rows']=16
                write_lines(path,rows)
                with self.assertRaises(ValueError):p.cache_report(path,'rebuild',mutate!='trace_in_timing')

    def test_zero_models_requires_explicit_completion_and_native_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.jsonl';path.write_text('')
            with self.assertRaises(ValueError):p.cache_report(path,'reuse',False,0)
            write_lines(path,[dict(schema='research-curvature-cache-session-v1',mode='reuse',model_instances=0,solver_returned=True)])
            self.assertEqual(p.cache_report(path,'reuse',False,0)['model_instances'],0)
            with self.assertRaises(ValueError):p.cache_report(path,'reuse',False,1)

    def test_qualification_exact_trace_and_counter_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            d,pair,group=qualification_fixture(Path(tmp))
            result=p.compare_qualification(group,d,pair)
            self.assertTrue(result['passed']);self.assertEqual(result['preparations_avoided'],2)
            self.assertTrue(result['mechanism_informative'])

    def test_qualification_refuses_work_trace_origin_or_model_mutation(self):
        for mutate in ('work','origin','outer','mode','instance','command','audit_binding'):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);d,pair,group=qualification_fixture(root)
                if mutate=='work':
                    ep=p.quiet.read(root/'rebuild/endpoint.json');ep['lazy_newton_factor_attempts']=4;p.quiet.write(root/'rebuild/endpoint.json',ep)
                elif mutate=='command':
                    c=p.quiet.read(root/'rebuild/COMMAND.json');c['command']+=['--other'];p.quiet.write(root/'rebuild/COMMAND.json',c)
                elif mutate=='audit_binding':
                    a=p.quiet.read(root/'rebuild/AUDIT.json');a['input_sha256']='f'*64;p.quiet.write(root/'rebuild/AUDIT.json',a)
                else:
                    rows=trace_rows('rebuild')
                    if mutate=='origin':rows[-1]['origin']='different-direction'
                    elif mutate=='outer':rows[-1]['outer']=99
                    elif mutate=='mode':rows[-1]['mode']='reuse'
                    else:rows[-1]['model_instance']=2
                    write_lines(root/'rebuild/curvature_trace.jsonl',rows)
                with self.assertRaises(ValueError):p.compare_qualification(group,d,pair)

    def test_resume_recovers_finalized_clean_numeric_failure_no_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p.quiet.write(out/'INSTALLATION.json',{})
            task=dict(pair_id='pair',mode='reuse',job_id='pair_reuse',nominal_id='pair_reuse_rep1',repetition=1,phase='timing')
            d=dict(schedule=[task],pairs=[dict(pair_id='pair')]);calls=[]
            def attempt(*args):calls.append(1);return synthetic_attempt(task)
            original_commit=p.quiet.commit_attempt
            with patch.object(p,'attempt_result',side_effect=attempt),patch.object(p.quiet,'commit_attempt',side_effect=RuntimeError('synthetic crash after ATTEMPT')):
                with self.assertRaises(RuntimeError):p.phase_run(d,out,'timing')
            with patch.object(p,'attempt_result',side_effect=AssertionError('numeric solve repeated')):
                self.assertEqual(p.phase_run(d,out,'timing'),1);self.assertEqual(p.phase_run(d,out,'timing'),1)
            self.assertEqual(len(calls),1)
            r=p.quiet.read(out/'runs'/task['nominal_id']/'RESULT.json');self.assertEqual(r['classification'],'nonstrict')

    def test_contaminated_attempt_preserved_same_repetition_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p.quiet.write(out/'INSTALLATION.json',{})
            task=dict(pair_id='pair',mode='reuse',job_id='pair_reuse',nominal_id='pair_reuse_rep1',repetition=1,phase='timing')
            d=dict(schedule=[task],pairs=[dict(pair_id='pair')]);responses=[synthetic_attempt(task,contaminated=True),synthetic_attempt(task)]
            with patch.object(p,'attempt_result',side_effect=responses):self.assertEqual(p.phase_run(d,out,'timing'),1)
            nominal=out/'runs'/task['nominal_id'];self.assertTrue((nominal/'attempt_0001/ATTEMPT.json').is_file())
            self.assertTrue((nominal/'attempt_0002/ATTEMPT.json').is_file());self.assertEqual(len(list(nominal.glob('attempt_*'))),2)

    def test_completed_malformed_endpoint_is_finalized_indeterminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);d,pair=fixture_context()
            task=dict(pair_id=pair['pair_id'],mode='reference',job_id='job',nominal_id='nominal',phase='timing',repetition=1)
            def process(command,where,*args,**kwargs):
                if '--case-json' not in command:(folder/'endpoint.json').write_text('{malformed')
                else:p.quiet.write(folder/'AUDIT.json',dict(audit_available=False,classification='indeterminate',input_sha256=pair['input_sha256'],endpoint_sha256=p.quiet.sha(folder/'endpoint.json')))
                return resources(command)
            with patch.object(p.quiet,'verify_start'),patch.object(p.quiet,'artifact',return_value={}),patch.object(p.quiet.QuietMonitor,'wait_quiet'),patch.object(p.quiet,'run_process',side_effect=process):
                r=p.attempt_result(d,pair,task,folder,[False])
            self.assertEqual(r['classification'],'indeterminate');self.assertIsNone(r['numerical_core_sha256']);self.assertTrue(r['quiet_valid'])

    def test_bad_or_misbound_audit_never_becomes_certificate(self):
        for kind in ('malformed','other_input'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);d,pair=fixture_context()
                task=dict(pair_id=pair['pair_id'],mode='reference',job_id='job',nominal_id='nominal',phase='timing',repetition=1)
                def process(command,where,*args,**kwargs):
                    if '--case-json' not in command:p.quiet.write(folder/'endpoint.json',endpoint())
                    elif kind=='malformed':(folder/'AUDIT.json').write_text('{broken')
                    else:p.quiet.write(folder/'AUDIT.json',dict(classification='strict',strict_qcqp=True,primal=0.,dual=0.,complementarity=0.,input_sha256='other',endpoint_sha256=p.quiet.sha(folder/'endpoint.json')))
                    return resources(command)
                with patch.object(p.quiet,'verify_start'),patch.object(p.quiet,'artifact',return_value={}),patch.object(p.quiet.QuietMonitor,'wait_quiet'),patch.object(p.quiet,'run_process',side_effect=process):
                    r=p.attempt_result(d,pair,task,folder,[False])
                self.assertEqual(r['classification'],'indeterminate');self.assertTrue(r['quiet_valid'])
                self.assertTrue(any('original_audit_contract' in x for x in r['errors']))

    def test_qualification_receipt_mutation_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q=dict(schema='curvature-persistence-qualification-v1',passed=False)
            q['qualification_content_sha256']=p.quiet.digest(q);q['passed']=True
            p.quiet.write(root/'QUALIFICATION.json',q)
            with self.assertRaisesRegex(ValueError,'modifié'):p.verify_qualification({},root)

    def test_summary_requires_qualification_for_existing_timed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p.quiet.write(root/'INSTALLATION.json',{})
            task=dict(nominal_id='synthetic',job_id='synthetic')
            folder=root/'runs/synthetic';folder.mkdir(parents=True)
            result=dict(installation_sha256=p.quiet.sha(root/'INSTALLATION.json'),nominal_id='synthetic',job_id='synthetic',quiet_valid=True,finalized=True,artifacts={})
            result['result_content_sha256']=p.quiet.digest(result);p.quiet.write(folder/'RESULT.json',result)
            with patch.object(p,'load_installation',return_value=dict(schedule=[task])):
                with self.assertRaisesRegex(ValueError,'sans qualification'):p.summarize(root)

    def test_summary_retains_strict_but_numerically_different_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);cohort=p.load_cohort();controls,schedule=p.schedules(cohort)
            d=dict(pairs=cohort['pairs'],schedule=schedule);p.quiet.write(out/'INSTALLATION.json',{})
            installation_sha=p.quiet.sha(out/'INSTALLATION.json');bad=cohort['pairs'][0]['pair_id'];inactive=cohort['pairs'][1]['pair_id']
            p.quiet.write(out/'QUALIFICATION.json',{})
            qualification=dict(passed=True,pairs=[dict(pair_id=r['pair_id'],cache_exercised=r['pair_id']!=inactive,nonzero_correction_observed=True) for r in cohort['pairs']])
            for task in schedule:
                folder=out/'runs'/task['nominal_id'];folder.mkdir(parents=True)
                ep=endpoint()
                if task['pair_id']==bad and task['mode']=='rebuild':ep['x'][1]=2.
                p.quiet.write(folder/'endpoint.json',ep)
                r=dict(nominal_id=task['nominal_id'],job_id=task['job_id'],pair_id=task['pair_id'],case_id='synthetic',
                    backend='ma27',regime='I',mode=task['mode'],repetition=task['repetition'],classification='strict',
                    failure_cause='none',quiet_valid=True,finalized=True,installation_sha256=installation_sha,
                    seconds_process=1. if task['mode']=='reuse' else 2.,numerical_core_sha256=p.quiet.digest(p.endpoint_core(folder/'endpoint.json',True)),
                    artifacts={'endpoint.json':p.quiet.artifact(folder/'endpoint.json')})
                r['result_content_sha256']=p.quiet.digest(r);p.quiet.write(folder/'RESULT.json',r)
            with patch.object(p,'load_installation',return_value=d),patch.object(p,'verify_qualification',return_value=qualification),contextlib.redirect_stdout(io.StringIO()):report=p.summarize(out)
            self.assertTrue(report['complete']);self.assertEqual(report['timed_repetitions_completed'],144)
            row=next(r for r in report['pairs'] if r['pair_id']==bad)
            self.assertTrue(row['both_strict_three']);self.assertFalse(row['pure_cache_cost_eligible']);self.assertIsNone(row['ratio_reuse_over_rebuild'])
            self.assertEqual(sum(r['pure_cache_cost_eligible'] for r in report['pairs']),22)
            inactive_row=next(r for r in report['pairs'] if r['pair_id']==inactive)
            self.assertFalse(inactive_row['cache_exercised']);self.assertFalse(inactive_row['pure_cache_cost_eligible'])
            self.assertIsNone(inactive_row['ratio_reuse_over_rebuild']);self.assertEqual(inactive_row['descriptive_ratio_reuse_over_rebuild'],.5)


if __name__=='__main__':unittest.main(verbosity=2)
