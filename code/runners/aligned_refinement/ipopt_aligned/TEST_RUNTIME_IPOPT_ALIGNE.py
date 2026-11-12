#!/usr/bin/env python3
"""Adversarial synthetic runtime checks; no optimization is invoked."""
from pathlib import Path
import contextlib,copy,hashlib,io,json,os,tempfile,threading,unittest
from unittest.mock import patch
import IPOPT_ALIGNE_CONTRACT as K
import PREPARER_IPOPT_ALIGNE as P
import CONTROLEUR_IPOPT_ALIGNE as C
B=C.base

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

def endpoint(label='ipopt_ma27_monotone_aligned',hsl='/frozen/hsl.so'):
    c=K.CELLS[label]
    return dict(adapter_version='ipopt-aligned-original-qcqp-v1',reference_policy='aligned_original_kkt_tolerances_v1',
        aligned_contract_verified=True,option_file_disabled=True,solver='Ipopt',mode='direct-frozen-qcqp',available=True,
        ipopt_version='3.14.11',linear_solver=c['backend'],mu_strategy=c['strategy'],optimization_wall_limit_seconds=7200,
        max_iterations=5000,requested_mumps_ordering=0,effective_options=dict(K.ALIGNED_OPTIONS,mu_strategy=c['strategy'],
        linear_solver=c['backend'],nlp_scaling_method='gradient-based',hessian_approximation='exact',fixed_variable_treatment='make_parameter',hsllib=hsl,mumps_pivot_order=7,nlp_lower_bound_inf=-1e19,nlp_upper_bound_inf=1e19),input_representation_overrides=[],
        elapsed_seconds=.02,iterations=3,ipopt_application_status_code=0,x=[1.],equality_multipliers=[],inequality_multipliers=[],lower_bound_multipliers=[0.],upper_bound_multipliers=[0.])

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def config(self):return dict(_sha256='installation',binary='native',binary_sha256='binary',hsl_library='/frozen/hsl.so',ipopt_version='3.14.11',input_representation_expectations={'test':dict(input_representation_overrides=[],effective_bound_infinity=dict(nlp_lower_bound_inf=-1e19,nlp_upper_bound_inf=1e19))})
    def case(self,cid='test',dataset='CUTEst',regime='I'):return dict(case_id=cid,dataset=dataset,regime=regime,input_sha256='input',start_binding=dict(start_sha256='start'))
    def test_four_cells_and_resources_are_explicit(self):
        self.assertEqual(len(K.CELLS),4);self.assertEqual(K.TASKS,4078*4);self.assertEqual(K.CPUS,tuple(range(22)))
        for label,c in K.CELLS.items():
            cmd=K.make_command(self.config(),self.case(),label,'endpoint','rawinput')
            self.assertEqual(cmd[cmd.index('--mu-strategy')+1],c['strategy']);self.assertEqual(cmd[cmd.index('--linear-solver')+1],c['backend'])
            self.assertEqual(cmd[cmd.index('--max-iter')+1],'5000');self.assertEqual(cmd[cmd.index('--time-limit')+1],'7200')
            self.assertEqual('--hsl-library'in cmd,c['backend']=='ma27')
    def test_all_four_effective_option_sets_pass(self):
        for label in K.CELLS:self.assertTrue(K.verify_endpoint(endpoint(label),label,'3.14.11','/frozen/hsl.so')['aligned_options_verified'])
    def test_each_changed_effective_tolerance_is_refused(self):
        for key in K.ALIGNED_OPTIONS:
            e=endpoint();e['effective_options'][key]='corrupted'
            with self.subTest(key=key),self.assertRaises(ValueError):K.verify_endpoint(e,'ipopt_ma27_monotone_aligned','3.14.11','/frozen/hsl.so')
    def test_adaptive_cannot_be_substituted_for_monotone(self):
        e=endpoint();e['effective_options']['mu_strategy']='adaptive'
        with self.assertRaises(ValueError):K.verify_endpoint(e,'ipopt_ma27_monotone_aligned','3.14.11','/frozen/hsl.so')
    def test_wrong_hsl_and_defaults_are_refused(self):
        for key,value in [('hsllib','/other/hsl.so'),('nlp_scaling_method','none'),('hessian_approximation','limited-memory'),('fixed_variable_treatment','relax_bounds')]:
            e=endpoint();e['effective_options'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):K.verify_endpoint(e,'ipopt_ma27_monotone_aligned','3.14.11','/frozen/hsl.so')
    def test_native_limits_and_nonfinite_vectors_are_refused(self):
        for key,value in [('iterations',5001),('elapsed_seconds',7200.01),('x',[float('nan')]),('available',False),('option_file_disabled',False)]:
            e=endpoint();e[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):K.verify_endpoint(e,'ipopt_ma27_monotone_aligned','3.14.11','/frozen/hsl.so')
    def test_process_cpu_and_wall_and_native_budgets_are_conjunctive(self):
        resources=dict(elapsed_seconds=1.,user_cpu_seconds=.8,system_cpu_seconds=.1,returncode=0,censored=False)
        self.assertTrue(K.process_budget_verified(resources))
        for change in [dict(elapsed_seconds=7201),dict(user_cpu_seconds=7200,system_cpu_seconds=.1),dict(censored=True),dict(returncode=-9),dict(user_cpu_seconds=float('nan'))]:
            with self.subTest(change=change):self.assertFalse(K.process_budget_verified(dict(resources,**change)))
        c=C.IpoptCampaign(self.config(),self.root,[self.case()],cpus=(0,));result=dict(classification='strict',strict_qcqp=True)
        c.decorate_result(result,self.case(),'ipopt_ma27_monotone_aligned',dict(resources,elapsed_seconds=7201),dict(budget_verified=True))
        self.assertFalse(result['budget_verified']);self.assertIsNone(result['strict_qcqp'])
    def test_backend_banner_is_required_beyond_effective_options(self):
        c=C.IpoptCampaign(self.config(),self.root,[self.case()],cpus=(0,));e=endpoint();write(self.root/'endpoint.json',e)
        write(self.root/'case.json',dict(case_id='test'))
        (self.root/'solver.log').write_text('running with linear solver mumps\n')
        with self.assertRaisesRegex(ValueError,'Backend'):c.check_endpoint(e,'ipopt_ma27_monotone_aligned','ma27',self.root)
        (self.root/'solver.log').write_text('Ipopt version 3.14.11, running with linear solver ma27.\n')
        self.assertTrue(c.check_endpoint(e,'ipopt_ma27_monotone_aligned','ma27',self.root)['backend_banner_verified'])
        e=endpoint('ipopt_mumps_monotone_aligned');write(self.root/'endpoint.json',e)
        (self.root/'solver.log').write_text('This is Ipopt version 3.14.11, running with linear solver MUMPS 5.4.1.\n')
        self.assertTrue(c.check_endpoint(e,'ipopt_mumps_monotone_aligned','mumps',self.root)['backend_banner_verified'])
    def test_balanced_order_never_changes_starts(self):
        cases=[self.case(str(i))for i in range(8)];original=copy.deepcopy(cases);jobs,order=C.make_plan(cases)
        self.assertEqual(cases,original);self.assertEqual(len(jobs),32)
        from collections import Counter
        self.assertEqual(set(Counter(x['treatments'][0]for x in order['rotations']).values()),{2})
    def test_queue_is_continuous_across_datasets(self):
        cases=[self.case('a'),self.case('b','multicast_beamforming','M')];c=C.IpoptCampaign(self.config(),self.root,cases,cpus=(0,1))
        first=threading.Event();release=threading.Event();later=threading.Event()
        def task(cpu,case,method,backend,recover):
            if case['case_id']=='a'and not first.is_set():first.set();release.wait(2)
            if case['case_id']=='b':later.set();release.set()
        c.task=task
        for job in c.planned:c.jobs.put((*job,False))
        a=threading.Thread(target=c.worker,args=(0,));b=threading.Thread(target=c.worker,args=(1,));a.start();self.assertTrue(first.wait(1));b.start();self.assertTrue(later.wait(1));release.set();a.join(2);b.join(2);self.assertFalse(a.is_alive()or b.is_alive())
    def test_stale_saved_pid_is_not_active_work(self):
        saved=dict(state='running',controller_pid=1,controller_start_ticks='stale',active=[dict(pid=2,pid_start_ticks='stale',phase='solver')],active_slots=1,active_solver=1,active_audit=0)
        with patch.object(B,'owned_live',return_value=False):s=B.observed_status(saved)
        self.assertEqual(s['state'],'controller_dead_stale_state');self.assertEqual(s['active_slots'],0)
    def test_partial_recovery_does_not_restart_solver(self):
        inp=self.root/'input.qcq';inp.write_bytes(b'fixture\nx0 0\n');case=self.case();case.update(input_sha256=B.sha(inp));case['start_binding'].update(raw_x0_line_sha256=hashlib.sha256(b'x0 0\n').hexdigest(),external_start_required=False)
        cfg=dict(self.config(),root=str(self.root),frozen_bindings={'test':{'input':str(inp)}})
        c=C.IpoptCampaign(cfg,self.root,[case],cpus=(0,));m='ipopt_ma27_monotone_aligned';f=c.folder(case,m);f.mkdir(parents=True);write(f/'case.json',dict(run_id=c.run_id,installation_sha256='installation'));write(f/'solver_resources.json',dict(returncode=None))
        with patch.object(B,'run_process',side_effect=AssertionError('must not restart optimizer')):c.task(0,case,m,'ma27',True)
        result=B.read(f/'RESULT.json');self.assertEqual(result['status'],'controller_interrupted_resources_unavailable');self.assertIsNone(result['strict_qcqp'])
        self.assertEqual(result['binary_sha256'],'binary');self.assertTrue((self.root/'OBSERVATIONS.jsonl').is_file())
    def test_local_build_cannot_authorize_production(self):
        build=self.root/'BUILD.json';write(build,dict(schema='isolated-ipopt-aligned-options-build-v1',original_installation_unchanged=True,production_dependencies_verified=False))
        with self.assertRaisesRegex(ValueError,'Dépendances'):P.load_build(build)
    def test_fresh_output_cannot_be_created_inside_inputs(self):
        p=self.root/'data';p.mkdir()
        with self.assertRaisesRegex(ValueError,'séparée'):P.fresh(p/'results',[p])
    def test_observation_export_repairs_missing_append(self):
        c=C.IpoptCampaign(self.config(),self.root,[],cpus=(0,));row=dict(case_id='x',method='ipopt_ma27_monotone_aligned',classification='indeterminate');c.results[('x',row['method'])]=row
        c.export_observations();self.assertEqual([json.loads(x)for x in(self.root/'OBSERVATIONS.jsonl').read_text().splitlines()],[row])
    def test_compact_status_has_four_regimes_and_four_methods(self):
        c=C.IpoptCampaign(self.config(),self.root,[self.case()],cpus=(0,));s=c.snapshot();stream=io.StringIO()
        with contextlib.redirect_stdout(stream):C.show(s)
        self.assertEqual(set(s['by_regime_method']),set(K.REGIMES));self.assertLessEqual(len(stream.getvalue().splitlines()),7);self.assertNotIn('DATASET 1',stream.getvalue())
    def test_full_corpus_preparation_and_reload_preserve_resolved_bindings(self):
        # Real 4,078-ID contract, synthetic read-only resolver: no QCQP is
        # generated and no optimizer or new KKT evaluation is invoked.
        source=self.root/'source';run=self.root/'run';kit=self.root/'kit';builddir=self.root/'build'
        for p in(source,run,kit/'evidence',builddir):p.mkdir(parents=True)
        write(source/'INSTALLATION.json',{})
        write(run/'PLAN.json',dict(identity='source plan fixture'))
        evidence=B.read(P.KIT/'evidence/CURRENT_PLAN_COMPACT.json');evidence['source_plan_sha256']=B.sha(run/'PLAN.json');write(kit/'evidence/CURRENT_PLAN_COMPACT.json',evidence)
        corpus=P.BASE/'CONTRAT_4078.json.gz';cases=B.plan_cases(B.contract(corpus))
        write(builddir/'BUILD.json',dict(identity='build fixture'));write(builddir/'QUALIFICATION.json',dict(identity='qualification fixture'))
        resolution=self.root/'prior.json';write(resolution,dict(all_inputs_and_raw_starts_verified=True,frozen_roots=[]))
        build=dict(binary='native',binary_sha256='binary',ipopt_library='ipopt',ipopt_library_sha256='ipopt-sha',hsl_library='hsl',hsl_library_sha256='hsl-sha',ipopt_version='3.14.11',runtime_environment={})
        bindings={c['case_id']:dict(input='/canonical_inputs/'+c['case_id']+'/input.qcq',input_sha256=c['input_sha256'],raw_x0_line_sha256=c['start_binding']['raw_x0_line_sha256'],start_sha256=c['start_binding']['start_sha256'])for c in cases}
        def resolve(root,selected,output,**kwargs):
            self.assertEqual(selected,cases)
            rows=[dict(case_id=c['case_id'],status='verified',raw_x0_verified=True,selected_input=bindings[c['case_id']]['input'],expected_sha256=c['input_sha256'],raw_x0_line_sha256=c['start_binding']['raw_x0_line_sha256'])for c in cases]
            report=dict(all_inputs_and_raw_starts_verified=True,rows=rows,frozen_roots=[]);write(output,report);return bindings,report
        with patch.object(P,'KIT',kit),patch.object(P,'load_build',return_value=build),patch.object(P,'check_qualification',return_value={'passed':True}),patch.object(P.FROZEN_INPUTS,'resolve_all',side_effect=resolve),patch.object(K,'representation_for_input',return_value=self.config()['input_representation_expectations']['test']):
            installed=P.prepare(source,run,builddir/'BUILD.json',self.root/'prepared',contract_path=corpus,resolution_source=resolution,progress=lambda _:None)
            d=P.load_installation(installed)
        self.assertEqual(len(d['cases']),4078);self.assertEqual(d['tasks'],16312);self.assertEqual(d['frozen_bindings'],bindings)
        self.assertTrue(d['independent_of_refinement_qualification']);self.assertFalse(d['input_files_modified'])
        self.assertEqual(sum(c['case_id'].startswith(('multicast_rayleigh_lifted__','multicast_constant_modulus__'))for c in d['cases']),120)
    def test_large_finite_bounds_have_exact_declared_representation(self):
        path=self.root/'header.qcq';path.write_text('LAZYIPDCA_QCQP_V2\nname bounds\nn 2\nx0 0 0\nlb -inf -1.2e20\nub inf 1e20\n')
        representation=K.representation_for_input(path)
        self.assertEqual(representation['effective_bound_infinity'],dict(nlp_lower_bound_inf=-2.4e20,nlp_upper_bound_inf=2.4e20))
        e=endpoint();e['effective_options'].update(representation['effective_bound_infinity']);e['input_representation_overrides']=representation['input_representation_overrides']
        p=K.verify_endpoint(e,'ipopt_ma27_monotone_aligned','3.14.11','/frozen/hsl.so',representation)
        self.assertEqual(p['input_representation_overrides'],representation['input_representation_overrides'])
        e['input_representation_overrides']=[]
        with self.assertRaisesRegex(ValueError,'représentation'):K.verify_endpoint(e,'ipopt_ma27_monotone_aligned','3.14.11','/frozen/hsl.so',representation)
    def test_default_cutoffs_cannot_silently_replace_large_finite_bounds(self):
        path=self.root/'header.qcq';path.write_text('LAZYIPDCA_QCQP_V1\nname bounds\nn 1\nx0 0\nlb -1e20\nub 1e20\n')
        representation=K.representation_for_input(path);e=endpoint();e['input_representation_overrides']=representation['input_representation_overrides']
        with self.assertRaisesRegex(ValueError,'infini'):K.verify_endpoint(e,'ipopt_ma27_monotone_aligned','3.14.11','/frozen/hsl.so',representation)

if __name__=='__main__':unittest.main(verbosity=2)
