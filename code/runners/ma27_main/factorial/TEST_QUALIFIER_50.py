#!/usr/bin/env python3
"""Qualification-controller checks; these never claim corpus qualification."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys,tempfile,unittest,types
import QUALIFIER_50 as gate
import PROFILES

class GateTests(unittest.TestCase):
    def fixture(self,root,exact=True):
        policies=dict(inequality_policy='cache' if exact else 'hybrid',globalization_policy='r3' if exact else 'hs',objective_exact=exact)
        header=dict(schema='lazy-factorial-accepted-trace-v1',event='header',state_encoding='sha256-length-prefixed-binary64-le-x-s-z-lambda',state_coordinates='native-phase-internal',**policies)
        accepted=dict(event='accepted',accepted_index=1,state_sha256='1'*64,outer_iteration=1,phase_n=2,phase_mi=1,phase_me=0,alpha=0.5,mu=0.1,objective_E0_zero=exact,**policies)
        update=dict(event='post_multiplier_update',after_accepted_index=1,state_sha256='2'*64,outer_iteration=1,phase_n=2,phase_mi=1,phase_me=0,mu=0.1,**policies)
        side=dict(schema='lazy-factorial-endpoint-v1',trace_success=True,accepted_events=1,endpoint_accepted_steps=1,budget_verified=True,outer_iterations=1,restoration_iterations=0,seconds=1.0,legacy_objective_validation=not exact,**policies)
        trace=root/'trace.jsonl';endpoint=root/'endpoint.json';root.mkdir(parents=True,exist_ok=True)
        trace.write_text(''.join(json.dumps(r)+'\n' for r in (header,accepted,update)))
        endpoint.write_text('{}');Path(str(endpoint)+'.factorial.json').write_text(json.dumps(side))
        return trace,endpoint,[header,accepted,update],side
    def write_fixture(self,t,e,rows,side):
        t.write_text(''.join(json.dumps(r)+'\n' for r in rows));Path(str(e)+'.factorial.json').write_text(json.dumps(side))
    def test_clock_exclusion_preserves_numerical_work(self):
        a={'x':[1.0,-0.0],'outer_iterations':4,'linear_solves':8,'elapsed_seconds':1.0}
        b=copy.deepcopy(a);b['elapsed_seconds']=5.0
        self.assertTrue(gate.endpoint_parity(a,b)['passed'])
        b['linear_solves']=9;self.assertFalse(gate.endpoint_parity(a,b)['passed'])
        self.assertFalse(gate.endpoint_parity({'factorial_factorizations':3},{'factorial_factorizations':4})['passed'])
    def test_binary64_coordinate_parity(self):
        self.assertFalse(gate.endpoint_parity({'x':[-0.0]},{'x':[0.0]})['passed'])
        self.assertFalse(gate.endpoint_parity({'x':[1.0]},{'x':[1.0000000000000002]})['passed'])
    def test_only_declared_observation_counter_is_separated(self):
        a={'outer_iterations':5,'free_epigraph_checkpoint_lifts':9};b={'outer_iterations':5,'free_epigraph_checkpoint_lifts':0}
        self.assertFalse(gate.endpoint_parity(a,b)['passed'])
        r=gate.endpoint_parity(a,b,True);self.assertTrue(r['passed'])
        self.assertEqual(r['observation_work_deltas']['free_epigraph_checkpoint_lifts']['research_minus_reference'],-9)
        b['outer_iterations']=6;self.assertFalse(gate.endpoint_parity(a,b,True)['passed'])
    def test_checkpoint_metadata_is_reported_only_on_legacy_diagonal(self):
        a={'standard_auto_checkpoint_requested':True,'standard_auto_checkpoint_effective':True,
           'native_al_fallback':{'primary_standard_auto_checkpoint_requested':True,'primary_standard_auto_checkpoint_effective':True,'active':False},'factorizations':12}
        b=copy.deepcopy(a)
        for k in ('standard_auto_checkpoint_requested','standard_auto_checkpoint_effective'):b[k]=False
        for k in ('primary_standard_auto_checkpoint_requested','primary_standard_auto_checkpoint_effective'):b['native_al_fallback'][k]=False
        self.assertFalse(gate.endpoint_parity(a,b)['passed'])
        r=gate.endpoint_parity(a,b,True);self.assertTrue(r['passed']);self.assertEqual(len(r['observation_configuration_values']),4)
        self.assertTrue(all(v=={'reference':True,'research':False}for v in r['observation_configuration_values'].values()))
        b['native_al_fallback']['active']=True;self.assertFalse(gate.endpoint_parity(a,b,True)['passed'])
    def test_trace_is_actual_and_complete(self):
        with tempfile.TemporaryDirectory() as d:
            t,e,rows,side=self.fixture(Path(d))
            a=gate.trace(t,e,True);self.assertEqual(a['accepted_events'],1)
            self.assertEqual([r['event']for r in a['states']],['accepted','post_multiplier_update'])
            rows[-1]['state_sha256']='3'*64;self.write_fixture(t,e,rows,side)
            self.assertFalse(gate.trace_parity(a,gate.trace(t,e,True))['passed'])
    def test_missing_multiplier_state_rejects_parity(self):
        with tempfile.TemporaryDirectory() as d:
            t,e,rows,side=self.fixture(Path(d));a=gate.trace(t,e,True)
            self.write_fixture(t,e,rows[:-1],side)
            self.assertFalse(gate.trace_parity(a,gate.trace(t,e,True))['passed'])
    def test_missing_sidecar_budget_and_out_of_range_actual_budget(self):
        with tempfile.TemporaryDirectory() as d:
            t,e,rows,side=self.fixture(Path(d))
            for key,value in [('budget_verified',False),('restoration_iterations',601),('outer_iterations',5001),('seconds',7200.001)]:
                bad=copy.deepcopy(side);bad[key]=value;self.write_fixture(t,e,rows,bad)
                with self.assertRaises(ValueError):gate.trace(t,e,True)
    def test_wrong_exact_objective_and_inconsistent_policy_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            t,e,rows,side=self.fixture(Path(d));rows[1]['objective_E0_zero']=False;self.write_fixture(t,e,rows,side)
            with self.assertRaises(ValueError):gate.trace(t,e,True)
            t,e,rows,side=self.fixture(Path(d));rows[-1]['globalization_policy']='hs';self.write_fixture(t,e,rows,side)
            with self.assertRaises(ValueError):gate.trace(t,e,True)
    def test_legacy_hh_distinct_from_exact_h0(self):
        with tempfile.TemporaryDirectory() as d:
            t,e,rows,side=self.fixture(Path(d),False)
            self.assertFalse(gate.trace(t,e,False)['sidecar']['objective_exact'])
            with self.assertRaises(ValueError):gate.trace(t,e,True)
        h0=PROFILES.make_command(None,'H0','binary','input','endpoint','trace')
        hh=PROFILES.make_command(None,'H0','binary','input','endpoint','trace',legacy=True)
        self.assertNotIn('--factorial-validation-legacy-objective',h0)
        self.assertIn('--factorial-validation-legacy-objective',hh)
        self.assertEqual(h0[h0.index('--accepted-checkpoint')+1],'0')
    def test_all_four_factory_commands_share_exact_objective_protocol(self):
        for cell,factors in gate.CELLS.items():
            cmd=PROFILES.make_command(None,cell,'binary','input','endpoint','trace')
            self.assertNotIn('--factorial-validation-legacy-objective',cmd)
            self.assertEqual(cmd[cmd.index('--factorial-inequalities')+1],factors[0])
            self.assertEqual(cmd[cmd.index('--factorial-globalization')+1],factors[1])
            self.assertEqual(cmd[cmd.index('--accepted-checkpoint')+1],'0')
            self.assertEqual(cmd[cmd.index('--time-limit')+1],'7200')
            self.assertEqual(cmd[cmd.index('--max-outer')+1],'5000')
    def test_censors_do_not_become_diagonal_successes(self):
        self.assertTrue(gate.censored({'censored':True},{'status':'solved'}))
        self.assertTrue(gate.censored({'returncode':1},{'status':'max_outer_iterations'}))
        self.assertFalse(gate.censored({'returncode':1},{'status':'line_search_failure'}))
    def test_missing_passive_observer_proof_rejects(self):
        with self.assertRaisesRegex(ValueError,'source/observer'):gate.verify_build({},None,{})
    def test_full_coupling_receipt_checks_every_frozen_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binary=root/'binary';binary.write_bytes(b'test fixture, no optimizer')
            registration=root/'registration.json';registration.write_text('{}')
            proof_path=root/'COUPLING_GATE.json';rows_path=root/'rows.jsonl'
            args=types.SimpleNamespace(binary=binary,registration=registration,coupling_proof=proof_path)
            bindings=[dict(case_id='fixture_'+str(i),input_sha256='a'*64,start_sha256='b'*64)for i in range(1834)]
            reg={'inputs':{'case_bindings':bindings,'contract_file_sha256':'c'*64}}
            build={'research_source_sha256':{'fixture.cpp':'d'*64},'native_backend':'ma27','hsl_library_sha256':'f'*64}
            rows=[dict(**b,dataset='fixture',regime='I'if i<1037 else'M',signed_square_changed=False,passed=True,unresolved_structural_phase=False,inspection_sha256='e'*64)for i,b in enumerate(bindings)]
            proof=dict(schema='factorial-corpus-coupling-gate-v1',native_backend='ma27',hsl_library_sha256='f'*64,passed=True,coupling_separation_verified=True,cases=1834,inspected_cases=1834,changed_cases=0,unresolved_cases=0,missing_cases=0,regimes={'I':1037,'M':797},factor_binary_sha256=gate.sha(binary),registration_sha256=gate.sha(registration),contract_sha256='c'*64,research_source_sha256=build['research_source_sha256'],rows_file='rows.jsonl')
            def save():
                rows_path.write_text(''.join(json.dumps(r)+'\n'for r in rows));proof['rows_sha256']=gate.sha(rows_path);proof_path.write_text(json.dumps(proof))
            installed=root/'input_installation.json'
            installed.write_text(json.dumps({'frozen_bindings':{b['case_id']:{'input':str(root/'aliases'/(b['case_id']+'.data'))}for b in bindings}}))
            args.input_installation=installed
            proof['input_installation_sha256']=gate.sha(installed)
            for row in rows:
                row['input_path']=str(root/'aliases'/(row['case_id']+'.data'))
                row['input_installation_sha256']=gate.sha(installed)
                row['native_backend']='ma27';row['hsl_library_sha256']='f'*64
            save();self.assertTrue(gate.verify_coupling(proof,args,reg,build)['coupling_separation_verified'])
            actual_path=rows[-1]['input_path'];rows[-1]['input_path']=str(root/'wrong_input.qcq');save()
            with self.assertRaisesRegex(ValueError,'resolved input path differs'):gate.verify_coupling(proof,args,reg,build)
            rows[-1]['input_path']=actual_path

            rows[-1]['signed_square_changed']=True;save()
            with self.assertRaisesRegex(ValueError,'Unresolved or changed'):gate.verify_coupling(proof,args,reg,build)
            rows[-1]['signed_square_changed']=False;rows[-1]['start_sha256']='f'*64;save()
            with self.assertRaisesRegex(ValueError,'input/start'):gate.verify_coupling(proof,args,reg,build)
            rows[-1]['start_sha256']='b'*64;rows[-1]['case_id']=rows[0]['case_id'];save()
            with self.assertRaisesRegex(ValueError,'exact 1834 unique'):gate.verify_coupling(proof,args,reg,build)
    def test_indefiniteness_uses_exact_aggregate_coefficients(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.qcq';p.write_text('objective\nhessian_nnz 4\n0 0 1\n1 1 1\n0 1 2\n0 1 -2\n')
            self.assertFalse(gate.objective_indefinite(p))
            p.write_text('objective\nhessian_nnz 3\n0 0 1\n1 1 1\n0 1 2\n')
            self.assertTrue(gate.objective_indefinite(p))
    def test_input_binding_uses_declared_path_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'different layouts';folder.mkdir();actual=folder/'exact_original.data'
            actual.write_bytes(b'LAZYIPDCA_QCQP_V2\nname fixture\nn 2\nx0 0.25 -0\n')
            start=[.25,-0.0]
            case={'case_id':'generic_bound_case','input_sha256':gate.sha(actual),
                'start_sha256':hashlib.sha256(json.dumps(start,separators=(',',':')).encode()).hexdigest()}
            config={'root':str(root),'frozen_bindings':{case['case_id']:{'input':str(actual),'input_sha256':case['input_sha256']}}}
            self.assertEqual(gate.input_binding(root,case,config),actual)
            # Even an identical canonical copy cannot hide a missing binding.
            canonical=root/'data/frozen'/case['case_id'];canonical.mkdir(parents=True)
            (canonical/'input.qcq').write_bytes(actual.read_bytes())
            with self.assertRaisesRegex(ValueError,'Explicit frozen input path absent'):
                gate.input_binding(root,case,{'root':str(root),'frozen_bindings':{}})
            actual.write_bytes(actual.read_bytes().replace(b'0.25',b'0.5'))
            with self.assertRaisesRegex(ValueError,'Frozen coefficients differ'):
                gate.input_binding(root,case,config)

    def test_main_missing_inputs_writes_blocked_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out'
            cmd=[sys.executable,str(Path(gate.__file__)),'--root',str(root),'--binary',str(root/'none'),'--baseline-trace-binary',str(root/'none2'),'--registration',str(Path(gate.__file__).parent/'PRE_ENREGISTREMENT.json'),'--coupling-proof',str(root/'coupling.json'),'--out',str(out),'--cpus','999']
            p=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(p.returncode,2)
            result=json.loads((out/'GATE.json').read_text());self.assertFalse(result['passed']);self.assertFalse(result['production_authorized'])
            self.assertEqual((out/'GATE.sha256').read_text().split()[0],hashlib.sha256((out/'GATE.json').read_bytes()).hexdigest())

    def test_all_ten_native_commands_bind_ma27_and_the_exact_hsl_path(self):
        hsl='/explicit/licensed/libcoinhsl.so'
        commands=[PROFILES.make_command(None,c,'binary','input','endpoint','trace',hsl_library=hsl)for c in gate.CELLS]
        commands += [PROFILES.make_published_command(m,'binary','input','endpoint',hsl_library=hsl)for m in ('R3','HH','HS','SH','SS','SS_legacy')]
        for command in commands:
            self.assertEqual(command[command.index('--linear-backend')+1],'ma27')
            self.assertEqual(command[command.index('--hsl-library')+1],hsl)
            self.assertEqual(command[command.index('--dca-curvature-backend')+1],'same')
            self.assertIn('--no-dca-sparse-fallback',command)

    def test_historical_mumps_qualification_never_authorizes_ma27(self):
        historical={'schema':'native-ma27-six-profiles-qualification-v1','passed':True,'six_profiles_qualified':True,'native_backend':'mumps'}
        with self.assertRaisesRegex(ValueError,'MA27 qualification'):
            gate.verify_native_ma27_qualification(historical,Path('unused'),Path('unused'),'a'*64)

    def test_source_admission_cannot_change_models_or_claim_hidden_mumps(self):
        reference={'x.cpp':'a'*64};adapted={'x.cpp':'b'*64}
        receipt={'schema':'lazy-ma27-backend-support-adaptation-v1','primary_backend':'ma27','adapter_sha256':'c'*64,'original_tree_unchanged':True,'hidden_mumps_substitution':True}
        with self.assertRaisesRegex(ValueError,'routing'):
            gate.verify_admission(receipt,reference,adapted,'c'*64)
        receipt.update(hidden_mumps_substitution=False,models_inputs_starts_changed=True,original_certificate_tests_changed=False,pivot_decisions_changed=False)
        with self.assertRaisesRegex(ValueError,'protected numerical'):
            gate.verify_admission(receipt,reference,adapted,'c'*64)

    def test_missing_licensed_hsl_produces_blocked_receipt_without_solver_calls(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);binary=root/'not_executable';binary.write_text('test fixture; no optimizer')
            out=root/'qualification'
            command=[sys.executable,str(Path(gate.__file__).parent/'QUALIFIER_MA27.py'),'--binary',str(binary),'--hsl-library',str(root/'missing.so'),'--auditor',str(root/'missing_auditor.py'),'--out',str(out)]
            p=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(p.returncode,2)
            receipt=json.loads((out/'QUALIFICATION_MA27.json').read_text())
            self.assertFalse(receipt['passed']);self.assertFalse(receipt['six_profiles_qualified'])
            self.assertEqual(receipt['native_executions'],0)

if __name__=='__main__':unittest.main(verbosity=2)
