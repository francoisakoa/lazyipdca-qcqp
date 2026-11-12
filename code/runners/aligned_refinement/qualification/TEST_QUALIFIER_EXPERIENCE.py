#!/usr/bin/env python3
"""Adversarial controller/provenance tests, never a native campaign gate."""
from pathlib import Path
import collections,copy,hashlib,importlib.util,json,struct,subprocess,sys,tempfile,unittest
import QUALIFIER_EXPERIENCE as q

class QualificationTests(unittest.TestCase):
    def trace_fixture(self,root,refinement='frozen',globalization='r3',legacy=False,model=True):
        root=Path(root);root.mkdir(parents=True,exist_ok=True);policies=dict(inequality_policy='hybrid',globalization_policy=globalization,objective_exact=True)
        if not legacy:policies.update(refinement_policy=refinement,common_guards='r3',guard_config_sha256='c'*64,common_representation='original_qcqp')
        h=dict(schema='lazy-factorial-accepted-trace-v1',event='header',state_encoding='sha256-length-prefixed-binary64-le-x-s-z-lambda',state_coordinates='native-phase-internal',**policies)
        m=dict(event='metric_initialized',initialization='common_certified_hybrid',phase_metric_index=1,phase_n=2,phase_mi=1,phase_me=0,initial_inequality_model_sha256='a'*64,**policies)
        a=dict(event='accepted',accepted_index=1,state_sha256='1'*64,outer_iteration=2,phase_n=2,phase_mi=1,phase_me=0,alpha=.5,mu=.1,objective_E0_zero=True,direction_origin='majorant_primary',postprocess='none',final_origin='majorant_primary',**policies)
        if not legacy:a.update(phase_metric_index=1,initial_inequality_model_sha256='a'*64,current_inequality_model_sha256='a'*64,refinement_attempt_events=0,accepted_refinements=0,refinement_update_events=0,refinement_rollback_events=0)
        state=dict(event='post_multiplier_update',after_accepted_index=1,state_sha256='2'*64,outer_iteration=2,phase_n=2,phase_mi=1,phase_me=0,mu=.1,**policies)
        rows=[h]+([m]if model and not legacy else[])+([a,state]if model else[])
        side=dict(schema='lazy-factorial-endpoint-v1',trace_success=True,accepted_events=1 if model else 0,endpoint_accepted_steps=1 if model else 0,budget_verified=True,outer_iterations=2,restoration_iterations=0,seconds=.1,legacy_objective_validation=False,**policies)
        if not legacy:
            sequence=hashlib.sha256()
            if model:sequence.update(struct.pack('<Q',64));sequence.update(('a'*64).encode())
            side.update(initial_model_events=1 if model else 0,initial_inequality_model_sha256='a'*64 if model else'',initial_hybrid_models_sha256=sequence.hexdigest(),refinement_attempt_events=0,accepted_refinements=0,refinement_update_events=0,refinement_rollback_events=0,frozen_metric_invariant_verified=True,requested_linear_backend='ma27',actual_linear_backend='inequality-majorants/ma27',native_solver_completed=True,initial_model_state='reached'if model else'not_reached',initialization_calls=1 if model else 0,native_pre_model_return=not model,primary_newton_backend_used=model,primary_backend_factor_attempts=1 if model else 0)
        tr=root/'trace.jsonl';ep=root/'endpoint.json';ep.write_text('{}');self.save(tr,ep,rows,side)
        return tr,ep,rows,side
    def save(self,tr,ep,rows,side):
        tr.write_text(''.join(json.dumps(r)+'\n'for r in rows));Path(str(ep)+'.factorial.json').write_text(json.dumps(side))
    def test_binary64_state_parity_preserves_negative_zero_and_ulp(self):
        self.assertFalse(q.endpoint_parity({'x':[-0.0]},{'x':[0.0]})['passed'])
        self.assertFalse(q.endpoint_parity({'x':[1.]},{'x':[1.0000000000000002]})['passed'])
    def test_clocks_excluded_but_numerical_work_not_excluded(self):
        a=dict(x=[1.],elapsed_seconds=1.,factorizations=5,solves=8,outer_iterations=4);b=copy.deepcopy(a);b['elapsed_seconds']=9.
        self.assertTrue(q.endpoint_parity(a,b)['passed']);b['factorizations']=6;self.assertFalse(q.endpoint_parity(a,b)['passed'])
    def test_installed_ma27_reference_is_mandatory_and_fully_compared(self):
        endpoint=dict(x=[1.],elapsed_seconds=1.,factorizations=5,solves=8,outer_iterations=4)
        audit=dict(original_KKT_endpoint_strict=True,intervals={'primal':[0.,0.]})
        data={r:copy.deepcopy(endpoint)for r in q.legacy_roles('ma27')};audits={r:copy.deepcopy(audit)for r in data}
        tr=dict(state_sequence_sha256='a'*64,accepted_events=1,states=[{'state_sha256':'b'*64}]);observed=dict(observer=tr,experiment=copy.deepcopy(tr))
        self.assertTrue(all(c['passed']for c in q.legacy_comparisons(data,observed,audits,'ma27').values()))
        data['installed_reference']['factorizations']=6
        self.assertFalse(q.legacy_comparisons(data,observed,audits,'ma27')['installed_reference']['passed'])
        data['installed_reference']=copy.deepcopy(endpoint);audits['installed_reference']['intervals']['primal'][1]=1e-8
        self.assertFalse(q.legacy_comparisons(data,observed,audits,'ma27')['original_certification']['passed'])
        data.pop('installed_reference')
        with self.assertRaisesRegex(ValueError,'installed/fresh'):q.legacy_comparisons(data,observed,audits,'ma27')
    def test_seven_hundred_fidelity_runs_plus_forty_eight_controls(self):
        self.assertEqual(q.legacy_roles('mumps'),('reference','observer','experiment'))
        self.assertEqual(sum(len(q.legacy_roles(b))*50*2 for b in q.BACKENDS),q.LEGACY_RUNS)
        self.assertEqual(q.LEGACY_RUNS,700);self.assertEqual(q.PLANNED_NATIVE_RUNS,748);self.assertEqual(q.CPUS,(23,))
    def test_static_contract_requires_the_installed_reference_executions(self):
        c=dict(qualification=dict(cpus=[23],jobs=1),production=dict(cpus=[23],jobs=1),parity_gate=dict(legacy_runs=700,legacy_runs_by_backend=dict(ma27=400,mumps=300),synthetic_controls=6,synthetic_runs=48,total_runs=748,installed_ma27_reference_runs=100,installed_ma27_reference_required=True))
        self.assertTrue(q.verify_qualification_spec(c));c['parity_gate']['installed_ma27_reference_required']=False
        with self.assertRaisesRegex(ValueError,'748 required'):q.verify_qualification_spec(c)
    def test_iteration_limit_is_complete_not_wall_censored(self):
        e=dict(status='max_outer_iterations',elapsed_seconds=40.,outer_iterations=5000,restoration_iterations=600)
        self.assertEqual(q.censor_reasons(dict(returncode=1,censored=False,elapsed_seconds=40.),e),[])
        e['status']='time_limit';self.assertTrue(q.censor_reasons(dict(returncode=1),e))
    def test_signal_missing_return_and_budget_overrun_rejected(self):
        e=dict(status='solved',elapsed_seconds=1.,outer_iterations=3,restoration_iterations=0)
        self.assertIn('process_signal',q.censor_reasons(dict(returncode=-9),e));self.assertIn('returncode_unavailable',q.censor_reasons({},e))
        for k,v in [('elapsed_seconds',7200.001),('outer_iterations',5001),('restoration_iterations',601)]:
            z=copy.deepcopy(e);z[k]=v;self.assertTrue(q.censor_reasons(dict(returncode=0),z))
    def test_process_budget_is_independent_of_endpoint_claim(self):
        e=dict(status='solved',elapsed_seconds=7199.,outer_iterations=3,restoration_iterations=0)
        self.assertEqual(q.censor_reasons(dict(returncode=0,elapsed_seconds=7199.5),e),[])
        for seconds in(7200.001,float('inf'),-1,None):self.assertIn('invalid_process_elapsed_seconds',q.censor_reasons(dict(returncode=0,elapsed_seconds=seconds),e))
    def test_fixed_command_budget_and_raw_start_are_enforced(self):
        cmd=['binary','--linear-backend','ma27','--time-limit','7200','--max-outer','5000','--restoration-iters','600','--shared-restoration-budget','1','--native-al-fallback','off']
        self.assertTrue(q.check_command(cmd,'ma27'))
        changed=list(cmd);changed[changed.index('--max-outer')+1]='50'
        with self.assertRaisesRegex(ValueError,'declared budget'):q.check_command(changed,'ma27')
        with self.assertRaisesRegex(ValueError,'External endpoint'):q.check_command(cmd+['--dca-endpoint-start','endpoint.json'],'ma27')
        with self.assertRaisesRegex(ValueError,'declared budget'):q.check_command(cmd+['--time-limit','60'],'ma27')
    def test_actual_profile_factories_satisfy_qualification_bindings(self):
        profiles=q.load_module('actual_qualification_factory',q.KIT/'runtime/PROFILES_EXPERIENCE.py')
        for b in q.BACKENDS:
            for method in('R3','HH'):
                c=profiles.make_legacy_command(method,b,'binary','raw_input','endpoint',trace='trace',hsl_library='real_runtime_required_at_execution')
                self.assertTrue(q.check_command(c,b,'raw_input','endpoint'))
            for cell in q.CELLS:
                c=profiles.make_command(cell+'_'+b,'binary','raw_input','endpoint','trace',hsl_library='real_runtime_required_at_execution')
                self.assertTrue(q.check_command(c,b,'raw_input','endpoint'))
        with self.assertRaisesRegex(ValueError,'frozen input'):q.check_command(c,b,'different_input','endpoint')
    def test_legacy_trace_actual_commits_and_multiplier_states(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d,legacy=True);a=q.read_trace(t,e,True);self.assertEqual(a['accepted_events'],1)
            rows[-1]['state_sha256']='3'*64;self.save(t,e,rows,side);self.assertFalse(q.trace_parity(a,q.read_trace(t,e,True))['passed'])
            self.save(t,e,rows[:-1],side);self.assertFalse(q.trace_parity(a,q.read_trace(t,e,True))['passed'])
    def test_frozen_trace_is_witnessed_by_actual_metric_hash(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d);self.assertTrue(q.read_trace(t,e,True,'FROZEN_GR3')['model_reached'])
            rows[2]['current_inequality_model_sha256']='b'*64;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'Frozen correction changed'):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_frozen_refinement_attempt_is_rejected_even_without_commit(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d);side['refinement_attempt_events']=1;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'Frozen correction changed'):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_guard_change_in_metric_header_or_commit_is_rejected(self):
        for index in(1,2,3):
            with tempfile.TemporaryDirectory()as d:
                t,e,rows,side=self.trace_fixture(d);rows[index]['guard_config_sha256']='d'*64;self.save(t,e,rows,side)
                with self.assertRaises(ValueError):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_original_representation_and_exact_objective_required(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d);rows[2]['objective_E0_zero']=False;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'Objective correction'):q.read_trace(t,e,True,'FROZEN_GR3')
            t,e,rows,side=self.trace_fixture(d);rows[1]['common_representation']='signed_square';self.save(t,e,rows,side)
            with self.assertRaises(ValueError):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_initial_metric_sequence_digest_is_bound(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d);side['initial_hybrid_models_sha256']='f'*64;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'sequence digest'):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_no_model_route_is_recorded_not_invented_as_metric(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d,model=False);a=q.read_trace(t,e,True,'FROZEN_GR3');self.assertFalse(a['model_reached']);self.assertEqual(a['accepted_events'],0)
    def test_missing_sidecar_budget_and_accepted_mismatch_rejected(self):
        for k,v in [('budget_verified',False),('restoration_iterations',601),('seconds',7200.01),('accepted_events',2)]:
            with tempfile.TemporaryDirectory()as d:
                t,e,rows,side=self.trace_fixture(d);side[k]=v;self.save(t,e,rows,side)
                with self.assertRaises(ValueError):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_direction_origin_is_final_committed_origin(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d);rows[2]['postprocess']='soc_replacement';self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'Final origin'):q.read_trace(t,e,True,'FROZEN_GR3')
            rows[2]['final_origin']='soc_replacement';self.save(t,e,rows,side);self.assertEqual(q.read_trace(t,e,True,'FROZEN_GR3')['direction_origins'],{'soc_replacement':1})
    def test_metric_phase_dimensions_must_match_commit(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d);rows[2]['phase_n']=3;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'Metric dimensions'):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_primary_accepted_update_cannot_precede_model_initialization(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d);rows[2]['phase_metric_index']=0;rows[2]['initial_inequality_model_sha256']='';rows[2]['current_inequality_model_sha256']='';self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'metric phase'):q.read_trace(t,e,True,'FROZEN_GR3')
    def test_adaptive_work_counters_are_monotone_and_reconciled(self):
        with tempfile.TemporaryDirectory()as d:
            t,e,rows,side=self.trace_fixture(d,'adaptive');rows[2]['refinement_attempt_events']=2;side['refinement_attempt_events']=2
            second=copy.deepcopy(rows[2]);second['accepted_index']=2;second['outer_iteration']=3;second['refinement_attempt_events']=1;rows.append(second);side['accepted_events']=side['endpoint_accepted_steps']=2;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'decreasing'):q.read_trace(t,e,True,'REFINED_GR3')
            t,e,rows,side=self.trace_fixture(d,'adaptive');rows[2]['refinement_attempt_events']=2;side['refinement_attempt_events']=1;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'less than'):q.read_trace(t,e,True,'REFINED_GR3')
            t,e,rows,side=self.trace_fixture(d,'adaptive');rows[2]['refinement_update_events']=1;self.save(t,e,rows,side)
            with self.assertRaisesRegex(ValueError,'Committed metric-update'):q.read_trace(t,e,True,'REFINED_GR3')
    def cases(self):
        return [dict(case_id=f'{g}_{i}',dataset='d',regime=g,input_sha256='a'*64,start_binding={'start_sha256':'b'*64})for g,n in [('I',1037),('M',797)]for i in range(n)]
    def test_contract_sample_is_immutable_and_outcome_blind(self):
        cases=self.cases();ids=[f'{g}_{i}'for g in'IM'for i in range(25)];c=dict(parity_gate=dict(sample=list(ids),selection_uses_outcomes=False))
        got=q.sample_cases(c,cases);self.assertEqual([x['case_id']for x in got],ids)
        c['parity_gate']['sample'][-1]=ids[0]
        with self.assertRaisesRegex(ValueError,'unique'):q.sample_cases(c,cases)
        c['parity_gate']['sample']=ids;c['parity_gate']['selection_uses_outcomes']=True
        with self.assertRaisesRegex(ValueError,'outcome blind'):q.sample_cases(c,cases)
    def test_sample_binding_cannot_change_input_or_start(self):
        cases=self.cases();ids=[f'{g}_{i}'for g in'IM'for i in range(25)];bs=[dict(case_id=i,input_sha256='a'*64,start_sha256='b'*64)for i in ids]
        c=dict(parity_gate=dict(sample=ids,sample_bindings=bs,selection_uses_outcomes=False));bs[0]['input_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'different input'):q.sample_cases(c,cases)
    def test_resolved_input_used_without_nominal_fallback(self):
        with tempfile.TemporaryDirectory()as d:
            r=Path(d);p=r/'alternate'/'original.data';p.parent.mkdir();p.write_bytes(b'LAZYIPDCA_QCQP_V2\nx0 -0 1.0\n')
            raw=hashlib.sha256(b'x0 -0 1.0\n').hexdigest();case=dict(case_id='c',input_sha256=q.sha(p),start_binding=dict(start_sha256='b'*64,raw_x0_line_sha256=raw))
            prep={'frozen_bindings':{'c':dict(input=str(p),input_sha256=q.sha(p),start_sha256='b'*64,raw_x0_line_sha256=raw)}}
            self.assertEqual(q.bound_input(case,prep),p)
            (r/'data/frozen/c').mkdir(parents=True);(r/'data/frozen/c/input.qcq').write_bytes(p.read_bytes())
            with self.assertRaisesRegex(ValueError,'resolved frozen input'):q.bound_input(case,{'frozen_bindings':{}})
            prep['frozen_bindings']['c']['start_sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'start differs'):q.bound_input(case,prep)
    def intervention_records(self):
        records=[]
        with tempfile.TemporaryDirectory()as d:
            for i in range(6):
                for b in q.BACKENDS:
                    for c in q.CELLS:
                        refinement,g=q.policy_tuple(c);t,e,rows,side=self.trace_fixture(Path(d)/f'{i}_{b}_{c}',refinement,g)
                        if refinement=='adaptive':side['refinement_attempt_events']=1;rows[2]['refinement_attempt_events']=1;self.save(t,e,rows,side)
                        obs=q.read_trace(t,e,True,c);obs.pop('states');obs.pop('initial_events');records.append(dict(case_id=f'c{i}',backend=b,cell=c,passed=True,trace=obs))
        return records
    def test_intervention_does_not_require_positive_adaptive_benefit(self):
        rr=self.intervention_records();check=q.intervention_comparison(rr);self.assertTrue(check['passed']);self.assertTrue(all(x['accepted_refinements']==0 for x in check['realized_adaptation']))
        rr[0]['trace']['sidecar']['initial_inequality_model_sha256']='e'*64;self.assertFalse(q.intervention_comparison(rr)['passed'])
    def test_incomplete_or_nonexecuted_treatment_cannot_qualify(self):
        rr=self.intervention_records();self.assertFalse(q.intervention_comparison(rr[:-1])['passed'])
        for r in rr:
            if r['cell'].startswith('REFINED'):r['trace']['sidecar']['refinement_attempt_events']=0
        self.assertFalse(q.intervention_comparison(rr)['passed'])
    def test_synthetic_controls_are_parseable_original_QCQP(self):
        sys.path.insert(0,str(q.KIT/'runtime_base'));reader=q.load_module('fixture_original_reader',q.KIT/'runtime_base/frozen_qcqp_reader.py')
        with tempfile.TemporaryDirectory()as d:
            fs=q.fixtures(Path(d)/'controls');self.assertEqual(len(fs),6)
            for f in fs:
                m=reader.read_model(f['input']);self.assertEqual(len(m['x0']),len(m['lb']));self.assertEqual(q.sha(f['input']),f['input_sha256'])
    def test_cli_never_claims_pass_on_missing_preparation(self):
        with tempfile.TemporaryDirectory()as d:
            out=Path(d)/'new';r=subprocess.run([sys.executable,str(Path(q.__file__)),'--preparation',str(Path(d)/'absent.json'),'--out',str(out)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);gate=json.loads((out/'GATE50.json').read_text());self.assertFalse(gate['passed']);self.assertFalse(gate['production_authorized']);self.assertEqual(gate['completed_native_runs'],0)

if __name__=='__main__':unittest.main(verbosity=2)
