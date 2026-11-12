#!/usr/bin/env python3
"""Synthetic statistical/provenance tests; no native optimizer or real audit."""
import copy,hashlib,importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).with_name('ANALYSER_RAFFINEMENT.py');spec=importlib.util.spec_from_file_location('a',P);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def cases_with(discordances):
 cases=[];idx={};j=0
 for group,wins,losses in discordances:
  for outcome,num in [('win',wins),('loss',losses)]:
   for _ in range(num):
    c=dict(case_id=f'c{j}',regime='I'if j%2 else'M',dataset=group,family=group);j+=1;cases.append(c)
    for b in a.BACKENDS:
     for cell in a.CELLS:
      strict=cell=='REFINED_GHS'if outcome=='win'else cell=='FROZEN_GHS'
      idx[c['case_id'],b,cell]=dict(strict=strict,objective=1.,seconds_process=2.,accepted_direction_origins={},accepted_primary_updates=0)
 return cases,idx

def fixture_record(cell='FROZEN_GR3',backend='ma27'):
 f=a.FACTORS[cell]
 return dict(case_id='one',cell=cell,backend=backend,run_id='same',preregistration_sha256='p',binary_sha256='b'*64,input_sha256='i',start_sha256='x',objective_exact=True,budget_verified=True,trace_success=True,backend_verified=True,native_route_verified=True,original_audit_verified=True,strict_original=True,primal=0.,dual=1e-7,complementarity=1e-7,
  factorial_proof=dict(correction_evolution=f['evolution'],globalization=f['globalization'],guard_policy='common_r3_guards_v1',correction_initialization='shared_hybrid_initialization_v1',initial_model_state='reached',initialization_calls=1,initial_inequality_model_sha256='c'*64,invariant_options_sha256='e'*64,guard_config_sha256='f'*64,accepted_refinements=0,refinement_attempt_events=0,refinement_update_events=0,refinement_rollback_events=0,frozen_metric_invariant_verified=True))
def pre():return dict(backends=list(a.BACKENDS),production=dict(single_binary_for_both_backends=True),inputs=dict(case_bindings=[dict(case_id='one',regime='I',dataset='d',family='f',input_sha256='i',start_sha256='x')]),invariants=dict(guard_policy='common_r3_guards_v1',correction_initialization='shared_hybrid_initialization_v1'))

class Tests(unittest.TestCase):
 def test_positive_rule_pairwise_and_exact_half_concentration(self):
  cs,idx=cases_with([('A',15,5),('B',8,3),('C',7,2)])
  p=a.primary(cs,idx,'ma27');self.assertEqual(p['net_gain_bounds'],[20,20]);self.assertEqual(p['known_wins'],30);self.assertEqual(p['known_losses'],10);self.assertEqual(p['decision'],'positive_under_preregistered_rule')
 def test_one_collection_dominates_even_when_total_gain_large(self):
  cs,idx=cases_with([('A',21,0),('B',9,0)])
  p=a.primary(cs,idx,'ma27');self.assertFalse(p['conditions']['no_collection_above_half_net_gain_robust']);self.assertEqual(p['decision'],'criterion_not_met')
 def test_discordance_rule_is_not_marginal_ratio(self):
  cs,idx=cases_with([('A',30,20),('B',30,20)])
  p=a.primary(cs,idx,'ma27');self.assertEqual(p['net_gain_bounds'],[20,20]);self.assertFalse(p['conditions']['paired_discordances_at_least_two_to_one_robust'])
 def test_family_macro_can_reverse_a_positive_raw_gain(self):
  cs,idx=cases_with([('large',100,0),('small',0,5),('small2',0,5)])
  p=a.primary(cs,idx,'ma27');self.assertGreater(p['net_gain_bounds'][0],0);self.assertLess(p['family_macro_delta_bounds'][1],0);self.assertEqual(p['decision'],'criterion_not_met')
 def test_technical_pair_can_leave_primary_unidentified(self):
  cs,idx=cases_with([('A',10,0),('B',10,0)])
  idx[cs[-1]['case_id'],'ma27','REFINED_GHS']['strict']=None
  p=a.primary(cs,idx,'ma27');self.assertEqual(p['net_gain_bounds'],[19,20]);self.assertEqual(p['decision'],'not_identified_due_to_technical_or_missing_outcomes')
 def test_backends_are_averaged_not_added(self):
  cs,idx=cases_with([('A',15,5),('B',8,3),('C',7,2)])
  out,_=a.summarize(cs,idx);v=out['balanced_backend_effect_summaries']['refinement_under_hs_globalization'];self.assertEqual(v['lower_mean_certificate_equivalents'],20);self.assertEqual(out['overall_decision'],'positive_on_both_backends_under_preregistered_rule')
 def test_one_backend_cannot_substitute_for_other(self):
  cs,idx=cases_with([('A',15,5),('B',8,3),('C',7,2)])
  for c in cs:idx[c['case_id'],'mumps','REFINED_GHS']['strict']=idx[c['case_id'],'mumps','FROZEN_GHS']['strict']
  out,_=a.summarize(cs,idx);self.assertEqual(out['overall_decision'],'general_two_backend_claim_not_established')
 def test_secondary_does_not_use_nonstrict_or_subsecond_pairs(self):
  cs,idx=cases_with([('A',1,0)])
  for cell in a.CELLS:idx['c0','ma27',cell]=dict(strict=True,objective=1.002 if cell=='REFINED_GHS' else 1.,seconds_process=.9 if cell=='REFINED_GHS' else 2.)
  rows=a.secondary(cs,idx,'ma27');r=next(x for x in rows if x['left']=='FROZEN_GHS'and x['right']=='REFINED_GHS'and x['threshold']==1e-3);self.assertEqual(r['left_lower'],1);self.assertEqual(r['both_process_times_gt_1s_N'],0)
 def test_final_SOC_origin_is_not_pure_majorant(self):
  cs,idx=cases_with([('A',1,0)]);idx['c0','ma27','REFINED_GHS'].update(accepted_primary_updates=3,accepted_direction_origins=dict(majorant_primary=1,soc_descendant=2))
  r=next(x for x in a.origins(cs,idx,'ma27')if x['cell']=='REFINED_GHS');self.assertAlmostEqual(r['pure_majorant_fraction_on_identified_updates'],1/3)
 def test_valid_proof_and_strict_original_alias(self):
  idx,issues,runs=a.validate([fixture_record()],pre(),'p');self.assertFalse(issues);self.assertTrue(idx['one','ma27','FROZEN_GR3']['strict']);self.assertEqual(runs,['same'])
 def test_bad_budget_remains_unknown_not_a_numeric_failure(self):
  r=fixture_record();r['budget_verified']=False;idx,issues,_=a.validate([r],pre(),'p');self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)
 def test_duplicate_rejected(self):
  r=fixture_record()
  with self.assertRaisesRegex(ValueError,'Duplicate'):a.validate([r,r],pre(),'p')
 def test_initialization_mismatch_rejected(self):
  r=fixture_record();s=fixture_record('REFINED_GR3');s['factorial_proof']['initial_inequality_model_sha256']='d'*64
  with self.assertRaisesRegex(ValueError,'Initial corrections'):a.validate([r,s],pre(),'p')
 def test_guard_mismatch_invalidates_cell(self):
  r=fixture_record();r['factorial_proof']['guard_policy']='legacy_hs';idx,issues,_=a.validate([r],pre(),'p');self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)
 def test_frozen_may_not_refine(self):
  r=fixture_record();r['factorial_proof']['accepted_refinements']=1;idx,issues,_=a.validate([r],pre(),'p');self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)
 def test_binary_mismatch_rejected_across_backend_blocks(self):
  r=fixture_record();s=fixture_record(backend='mumps');s['binary_sha256']='d'*64
  with self.assertRaisesRegex(ValueError,'same preregistered binary'):a.validate([r,s],pre(),'p')
 def test_different_experiments_rejected(self):
  r=fixture_record();s=fixture_record('REFINED_GR3');s['run_id']='other'
  with self.assertRaisesRegex(ValueError,'different experiments'):a.validate([r,s],pre(),'p')
 def test_residual_mismatch_invalidates_claimed_certificate(self):
  r=fixture_record();r['dual']=2e-6;idx,issues,_=a.validate([r],pre(),'p');self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)

 def test_native_pre_model_solution_needs_no_fake_matrix_digest(self):
  r=fixture_record();r['factorial_proof'].update(initial_model_state='not_reached',initialization_calls=0,initial_inequality_model_sha256=None)
  idx,issues,_=a.validate([r],pre(),'p');self.assertFalse(issues);self.assertTrue(idx['one','ma27','FROZEN_GR3']['strict'])
 def test_reached_and_not_reached_cells_cannot_be_merged(self):
  r=fixture_record();s=fixture_record('REFINED_GR3');s['factorial_proof'].update(initial_model_state='not_reached',initialization_calls=0,initial_inequality_model_sha256=None)
  with self.assertRaisesRegex(ValueError,'Initial corrections'):a.validate([r,s],pre(),'p')
 def test_unknown_direction_origin_is_bounded_not_claimed_identified(self):
  cs,idx=cases_with([('A',1,0)]);idx['c0','ma27','REFINED_GHS'].update(accepted_primary_updates=3,accepted_direction_origins=dict(majorant_primary=1,unknown=2))
  r=next(x for x in a.origins(cs,idx,'ma27')if x['cell']=='REFINED_GHS');self.assertEqual(r['pure_majorant_fraction_bounds'],[1/3,1]);self.assertFalse(r['full_scope_fraction_identified'])
 def test_production_study_hash_mismatch_does_not_become_numerical_failure(self):
  r=fixture_record();r['study_contract_sha256']='wrong';idx,issues,_=a.validate([r],pre(),'p',study_sha='expected');self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)
 def test_binary_must_match_registered_build_not_just_other_rows(self):
  r=fixture_record();idx,issues,_=a.validate([r],pre(),'p',expected_binaries=dict(ma27='e'*64));self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)

 def test_normalize_actual_controller_suffix_and_provenance(self):
  r=fixture_record();p=r.pop('factorial_proof');r['cell']='FROZEN_GR3_ma27';r['native_backend']='ma27'
  r['provenance']=dict(intervention_verified=True,refinement_policy='frozen',globalization_policy='r3',common_guards='r3',inequality_policy='hybrid',initial_model_events=1,initial_inequality_model_sha256='c'*64,accepted_refinements=0,refinement_attempt_events=0,refinement_update_events=0,refinement_rollback_events=0,frozen_metric_invariant_verified=True,accepted_events=0,requested_linear_backend='ma27',actual_linear_backend='inequality-majorants/ma27',guard_config_sha256='f'*64)
  cmd=['taskset','-c','0','/binary','--problem','/data/a','--json','/out/a','--linear-backend','ma27','--factorial-refinement','frozen','--factorial-globalization','r3','--threads','1']
  r['solver_resources']=dict(command=cmd,command_sha256=hashlib.sha256(json.dumps(cmd).encode()).hexdigest());q=a.canonical_observation(r)
  self.assertEqual(q['cell'],'FROZEN_GR3');self.assertEqual(q['backend'],'ma27');self.assertEqual(len(q['factorial_proof']['invariant_options_sha256']),64)
  idx,issues,_=a.validate([r],pre(),'p');self.assertFalse(issues);self.assertTrue(idx['one','ma27','FROZEN_GR3']['strict'])
 def test_globalization_may_exit_before_M0_in_only_one_globalization(self):
  rr=[fixture_record(c)for c in a.CELLS]
  for r in rr:
   if r['cell'].endswith('GR3'):r['factorial_proof'].update(initial_model_state='not_reached',initialization_calls=0,initial_inequality_model_sha256=None)
  idx,issues,_=a.validate(rr,pre(),'p');self.assertFalse(issues);self.assertEqual(len(idx),4)
 def test_effective_command_unregistered_numerical_change_rejected(self):
  rr=[]
  for cell,inner in [('FROZEN_GR3','100'),('REFINED_GR3','101')]:
   r=fixture_record(cell);r['factorial_proof']['invariant_options_sha256']=a.invariant_command_digest(dict(solver_resources=dict(command=['/binary','--problem','/a','--json','/b','--inner-iters',inner])));rr.append(r)
  with self.assertRaisesRegex(ValueError,'settings differ'):a.validate(rr,pre(),'p')

 def test_native_common_guard_digest_mismatch_is_fatal(self):
  r=fixture_record();s=fixture_record('REFINED_GR3');s['factorial_proof']['guard_config_sha256']='d'*64
  with self.assertRaisesRegex(ValueError,'safeguard settings differ'):a.validate([r,s],pre(),'p')
 def test_frozen_attempt_without_accepted_update_is_invalid(self):
  r=fixture_record();r['factorial_proof']['refinement_attempt_events']=1;idx,issues,_=a.validate([r],pre(),'p');self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)

 def test_wrong_actual_backend_is_not_a_certified_backend_run(self):
  r=fixture_record();r['backend_verified']=False;idx,issues,_=a.validate([r],pre(),'p');self.assertIsNone(idx['one','ma27','FROZEN_GR3']['strict']);self.assertTrue(issues)

 def test_one_definitively_negative_backend_precludes_general_claim(self):
  cs,idx=cases_with([('A',15,5),('B',8,3),('C',7,2)])
  for c in cs:idx[c['case_id'],'ma27','REFINED_GHS']['strict']=idx[c['case_id'],'ma27','FROZEN_GHS']['strict']
  idx[cs[0]['case_id'],'mumps','REFINED_GHS']['strict']=None
  out,_=a.summarize(cs,idx);self.assertEqual(out['overall_decision'],'general_two_backend_claim_not_established')

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),analysis_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),optimizer_invocations=0,scope='Synthetic paired-rule and byte-provenance tests only; no native production qualification.')
 Path(__file__).with_name('TEST_ANALYSE_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful()else 1)
