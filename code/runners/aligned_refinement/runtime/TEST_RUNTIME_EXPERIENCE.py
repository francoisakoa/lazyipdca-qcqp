#!/usr/bin/env python3
"""Synthetic protocol tests. No native optimization or campaign is launched."""
from pathlib import Path
import copy,hashlib,json,struct,tempfile,threading,unittest
from unittest.mock import patch
import PROFILES_EXPERIENCE as P
import PREPARER_EXPERIENCE as PRE
import CONTROLEUR_EXPERIENCE as C
B=C.base

def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v)+'\n')
def fixture(folder,label='FROZEN_GR3_ma27'):
    c=P.CELLS[label];digest='a'*64;guard='b'*64
    expected=dict(objective_exact=True,inequality_policy='hybrid',globalization_policy=c['globalization_policy'],refinement_policy=c['refinement_policy'],common_guards='r3',common_representation='original_qcqp',guard_config_sha256=guard)
    rows=[dict(expected,event='header',schema='lazy-factorial-accepted-trace-v1',state_encoding='sha256-length-prefixed-binary64-le-x-s-z-lambda',state_coordinates='native-phase-internal'),dict(expected,event='metric_initialized',phase_metric_index=1,phase_n=1,phase_mi=1,phase_me=0,initial_inequality_model_sha256=digest,initialization='common_certified_hybrid'),dict(expected,event='accepted',accepted_index=1,state_sha256='c'*64,outer_iteration=1,phase_n=1,phase_mi=1,phase_me=0,alpha=1.,mu=.1,phase_metric_index=1,initial_inequality_model_sha256=digest,current_inequality_model_sha256=digest,refinement_attempt_events=0,accepted_refinements=0,refinement_update_events=0,refinement_rollback_events=0,objective_E0_zero=True,direction_origin='majorant_primary',postprocess='none',final_origin='majorant_primary')]
    trace=folder/'accepted_trace.jsonl';trace.write_text(''.join(json.dumps(r)+'\n'for r in rows))
    seq=hashlib.sha256();seq.update(struct.pack('<Q',64));seq.update(digest.encode())
    actual='inequality-majorants/'+c['backend']
    side=dict(expected,schema='lazy-factorial-endpoint-v1',native_solver_completed=True,initial_model_state='reached',initialization_calls=1,native_pre_model_return=False,primary_newton_backend_used=True,primary_backend_factor_attempts=1,guard_policy_version='common_r3_guards_v1',trace_success=True,legacy_objective_validation=False,budget_verified=True,outer_iterations=1,restoration_iterations=0,seconds=.02,requested_linear_backend='ma27'if c['backend']=='ma27'else'mumps_metis',actual_linear_backend=actual,refinement_attempt_events=0,accepted_refinements=0,refinement_update_events=0,refinement_rollback_events=0,initial_model_events=1,frozen_metric_invariant_verified=True,initial_inequality_model_sha256=digest,initial_hybrid_models_sha256=seq.hexdigest(),accepted_events=1,endpoint_accepted_steps=1)
    write(folder/'endpoint.json.factorial.json',side);ep=dict(linear_solver_backend=actual,lazy_newton_accepted_steps=1)
    return ep,side,rows

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_profile_intervention_only_declared_two_factors(self):
        commands={m:P.make_command(m,'bin','input','endpoint','trace','HSL')for m in P.CELLS if m.endswith('_ma27')}
        for x in commands.values():
            self.assertEqual(x[x.index('--factorial-inequalities')+1],'hybrid');self.assertEqual(x[x.index('--factorial-common-guards')+1],'r3')
            self.assertEqual(x[x.index('--accepted-checkpoint')+1],'0');self.assertEqual(x[x.index('--restoration-iters')+1],'600');self.assertEqual(x[x.index('--native-al-fallback')+1],'off')
        a=commands['FROZEN_GR3_ma27'];b=commands['REFINED_GR3_ma27']
        different=[i for i,(u,v)in enumerate(zip(a,b))if u!=v];self.assertEqual(different,[a.index('--factorial-refinement')+1])
    def test_backend_is_explicit_and_ma27_requires_library(self):
        with self.assertRaisesRegex(ValueError,'HSL'):P.make_command('FROZEN_GR3_ma27','b','i','e','t')
        x=P.make_command('FROZEN_GR3_mumps','b','i','e','t');self.assertNotIn('--hsl-library',x);self.assertEqual(x[x.index('--linear-backend')+1],'mumps')
    def test_valid_frozen_trace_and_sequence_reconcile(self):
        ep,side,rows=fixture(self.root);r=C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27');self.assertTrue(r['intervention_verified']);self.assertEqual(r['accepted_direction_origins'],{'majorant_primary':1})
    def test_frozen_metric_any_attempt_refused(self):
        ep,side,_=fixture(self.root);side['refinement_attempt_events']=1;write(self.root/'endpoint.json.factorial.json',side)
        with self.assertRaises(ValueError):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')
    def test_backend_dense_fallback_not_called_ma27(self):
        ep,side,_=fixture(self.root);ep['linear_solver_backend']='native_dense_lapack';side['actual_linear_backend']='native_dense_lapack';write(self.root/'endpoint.json.factorial.json',side)
        with self.assertRaisesRegex(ValueError,'Backend'):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')
    def test_trace_guard_change_refused(self):
        ep,side,rows=fixture(self.root);rows[-1]['guard_config_sha256']='c'*64;(self.root/'accepted_trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        with self.assertRaises(ValueError):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')
    def test_initial_model_sequence_hash_refused(self):
        ep,side,_=fixture(self.root);side['initial_hybrid_models_sha256']='d'*64;write(self.root/'endpoint.json.factorial.json',side)
        with self.assertRaises(ValueError):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')
    def test_original_objective_not_exact_refused(self):
        ep,side,rows=fixture(self.root);rows[-1]['objective_E0_zero']=False;(self.root/'accepted_trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        with self.assertRaises(ValueError):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')
    def test_owned_monitor_does_not_show_stale_active_work(self):
        s=dict(state='running',controller_pid=10,controller_start_ticks='old',active=[dict(pid=11,pid_start_ticks='old',phase='solver')],active_slots=1,active_solver=1,active_audit=0)
        with patch.object(B,'owned_live',return_value=False):r=B.observed_status(s)
        self.assertEqual(r['state'],'controller_dead_stale_state');self.assertEqual(r['active_slots'],0);self.assertEqual(s['active_slots'],1)
    def test_continuous_queue_has_no_dataset_barrier(self):
        # Two workers: a blocked first task must not prevent later-dataset
        # work on the other CPU. Events make this independent of sleep timing.
        cases=[dict(case_id='a',regime='I',dataset='CUTEst'),dict(case_id='b',regime='M',dataset='multicast_beamforming')]
        cfg=dict(registration_sha256='r',study_contract_sha256='s')
        c=C.ExperimentCampaign(cfg,self.root,cases,cpus=(0,1));start=threading.Event();release=threading.Event();later=threading.Event()
        def task(cpu,case,method,backend,recover):
            if case['case_id']=='a'and not start.is_set():start.set();release.wait(2)
            if case['case_id']=='b':later.set();release.set()
        c.task=task
        for case,m,b in c.planned:c.jobs.put((case,m,b,False))
        a=threading.Thread(target=c.worker,args=(0,));b=threading.Thread(target=c.worker,args=(1,));a.start();self.assertTrue(start.wait(1));b.start();self.assertTrue(later.wait(1));release.set();a.join(2);b.join(2);self.assertFalse(a.is_alive()or b.is_alive())
    def test_recovery_without_resource_proof_never_restarts_solver(self):
        inp=self.root/'input.qcq';inp.write_bytes(b'fixture\nx0 0\n');case=dict(case_id='test',regime='I',dataset='CUTEst',input_sha256=B.sha(inp),start_binding=dict(start_sha256='x',raw_x0_line_sha256=hashlib.sha256(b'x0 0\n').hexdigest(),external_start_required=False))
        cfg=dict(_sha256='install',root=str(self.root),frozen_bindings={'test':{'input':str(inp)}},registration_sha256='r',study_contract_sha256='s')
        c=C.ExperimentCampaign(cfg,self.root,[case],cpus=(0,));m='FROZEN_GR3_ma27';f=c.folder(case,m);f.mkdir(parents=True);write(f/'case.json',dict(run_id=c.run_id,installation_sha256='install'));write(f/'solver_resources.json',dict(returncode=None))
        with patch.object(B,'run_process',side_effect=AssertionError('must not restart optimizer')):c.task(0,case,m,'ma27',True)
        result=B.read(f/'RESULT.json');self.assertIsNone(result['strict_qcqp']);self.assertEqual(result['status'],'controller_interrupted_resources_unavailable')
    def test_new_output_cannot_overlap_existing_inputs(self):
        parent=self.root/'installed';parent.mkdir()
        with self.assertRaisesRegex(ValueError,'séparée'):PRE.fresh(parent/'new',[parent])
    def test_treatment_order_is_balanced_not_seeded_starts(self):
        cases=[dict(case_id=str(i),regime='I',dataset='CUTEst',raw_start=[i])for i in range(16)]
        jobs,order=C.make_plan(cases);self.assertEqual(len(jobs),128);self.assertEqual(cases[0]['raw_start'],[0])
        from collections import Counter
        first=Counter(x['treatments'][0]for x in order['rotations']);self.assertEqual(set(first.values()),{2})
    def test_promotion_requires_completed_gate_with_exact_binding(self):
        path=self.root/'prep.json';write(path,dict(schema='lazy-refinement-preparation-v1',production_authorized=False,source_snapshot={}))
        # Missing proof objects fails closed before any production file exists.
        with self.assertRaises((KeyError,ValueError)):PRE.promote(path,self.root/'absent_gate.json')
        self.assertFalse((self.root/'EXPERIMENT_INSTALLATION.json').exists())

    def test_frozen_actual_committed_metric_change_refused(self):
        ep,side,rows=fixture(self.root);rows[-1]['current_inequality_model_sha256']='d'*64;(self.root/'accepted_trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        with self.assertRaises(ValueError):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')
    def test_committed_state_hash_is_required(self):
        ep,side,rows=fixture(self.root);rows[-1].pop('state_sha256');(self.root/'accepted_trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        with self.assertRaises(ValueError):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')
    def test_persisted_intervention_violation_blocks_resume(self):
        write(self.root/'INTERVENTION_VIOLATION.json',dict(passed=False,failures=[dict(reason='initial_metric_mismatch')]))
        c=C.ExperimentCampaign(dict(registration_sha256='r',study_contract_sha256='s'),self.root,[],cpus=(0,))
        self.assertTrue(c.stop.is_set())
        with self.assertRaisesRegex(ValueError,'persistée'):c.prepare(True)
        self.assertFalse((self.root/'INTERVENTION_CHECK.json').exists())
    def test_exact_pre_model_return_is_explicit_not_observed_backend(self):
        ep,side,rows=fixture(self.root);rows=rows[:1];(self.root/'accepted_trace.jsonl').write_text(json.dumps(rows[0])+'\n')
        ep.update(lazy_newton_accepted_steps=0,linear_solver_backend='native_dense_lapack')
        side.update(initial_model_events=0,initialization_calls=0,initial_model_state='not_reached',initial_inequality_model_sha256='',initial_hybrid_models_sha256=hashlib.sha256(b'').hexdigest(),accepted_events=0,endpoint_accepted_steps=0,native_pre_model_return=True,primary_newton_backend_used=False,primary_backend_factor_attempts=0,actual_linear_backend='native_dense_lapack')
        write(self.root/'endpoint.json.factorial.json',side);r=C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27');self.assertTrue(r['native_pre_model_return']);self.assertFalse(r['primary_newton_backend_used'])
        side['native_solver_completed']=False;write(self.root/'endpoint.json.factorial.json',side)
        with self.assertRaises(ValueError):C.check_native_proof(ep,self.root,'FROZEN_GR3_ma27')

    def test_concurrent_finishes_preserve_common_guard_violation(self):
        cfg=dict(registration_sha256='r',study_contract_sha256='s');case=dict(case_id='test',regime='I',dataset='CUTEst')
        c=C.ExperimentCampaign(cfg,self.root,[case],cpus=(0,1));barrier=threading.Barrier(2)
        methods=['FROZEN_GR3_ma27','REFINED_GR3_ma27']
        def finish(index):
            m=methods[index];c.folder(case,m).mkdir(parents=True,exist_ok=True);barrier.wait()
            c.finish(case,m,dict(case_id='test',method=m,regime='I',dataset='CUTEst',classification='nonstrict',guard_config_sha256=('a'if index else'b')*64,globalization_policy='r3',initial_inequality_model_sha256='c'*64))
        threads=[threading.Thread(target=finish,args=(i,))for i in range(2)]
        for t in threads:t.start()
        for t in threads:t.join(2)
        self.assertTrue(c.stop.is_set());self.assertTrue((self.root/'INTERVENTION_VIOLATION.json').is_file());self.assertEqual(len(c.results),2)

    def test_same_globalization_model_reach_mismatch_is_persisted(self):
        cfg=dict(registration_sha256='r',study_contract_sha256='s');case=dict(case_id='test',regime='I',dataset='CUTEst')
        c=C.ExperimentCampaign(cfg,self.root,[case],cpus=(0,))
        for method,state,digest in [('FROZEN_GR3_ma27','not_reached',''),('REFINED_GR3_ma27','reached','a'*64)]:
            c.folder(case,method).mkdir(parents=True,exist_ok=True)
            c.finish(case,method,dict(case_id='test',method=method,regime='I',dataset='CUTEst',classification='nonstrict',guard_config_sha256='b'*64,globalization_policy='r3',initial_model_state=state,initial_inequality_model_sha256=digest))
        self.assertTrue(c.stop.is_set());self.assertEqual(B.read(self.root/'INTERVENTION_VIOLATION.json')['failures'][0]['reason'],'initial_metric_mismatch')
        with self.assertRaisesRegex(ValueError,'persistée'):c.prepare(True)
    def test_different_globalization_may_reach_different_initial_models(self):
        cfg=dict(registration_sha256='r',study_contract_sha256='s');case=dict(case_id='test',regime='I',dataset='CUTEst')
        c=C.ExperimentCampaign(cfg,self.root,[case],cpus=(0,))
        for method,glob,state,digest in [('FROZEN_GR3_ma27','r3','not_reached',''),('FROZEN_GHS_ma27','hs','reached','a'*64)]:
            c.folder(case,method).mkdir(parents=True,exist_ok=True)
            c.finish(case,method,dict(case_id='test',method=method,regime='I',dataset='CUTEst',classification='nonstrict',guard_config_sha256='b'*64,globalization_policy=glob,initial_model_state=state,initial_inequality_model_sha256=digest))
        self.assertFalse(c.stop.is_set());self.assertFalse((self.root/'INTERVENTION_VIOLATION.json').exists())

    def test_final_intervention_receipt_requires_all_native_proofs(self):
        rows=[dict(intervention_verified=True,budget_verified=True),dict(intervention_verified=False,budget_verified=True)]
        r=C.intervention_summary(rows,[],2)
        self.assertFalse(r['passed']);self.assertTrue(r['intervention_integrity_verified']);self.assertEqual(r['incomplete_native_interventions'],1)
        rows[1]['intervention_verified']=True;self.assertTrue(C.intervention_summary(rows,[],2)['passed'])
        self.assertFalse(C.intervention_summary(rows,[dict(reason='guard_mismatch')],2)['passed'])

    def test_production_ablation_pool_is_cpu23_only(self):
        c=C.ExperimentCampaign(dict(registration_sha256='r',study_contract_sha256='s'),self.root,[])
        self.assertEqual(c.cpus,(23,));self.assertEqual(C.CPUS,(23,));self.assertEqual(C.JOBS,1)

    def test_native_budget_never_overrides_process_budget(self):
        case=dict(case_id='test',dataset='CUTEst',regime='I',start_binding=dict(start_sha256='x'))
        cfg=dict(registration_sha256='r',study_contract_sha256='s');c=C.ExperimentCampaign(cfg,self.root,[case],cpus=(0,));m='FROZEN_GR3_ma27'
        result=dict(classification='strict',strict_qcqp=True)
        c.decorate_result(result,case,m,dict(elapsed_seconds=7201,returncode=0,censored=False),dict(budget_verified=True))
        self.assertFalse(result['budget_verified']);self.assertIsNone(result['strict_qcqp']);self.assertEqual(result['classification'],'indeterminate')

if __name__=='__main__':unittest.main(verbosity=2)
