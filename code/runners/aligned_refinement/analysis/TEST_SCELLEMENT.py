#!/usr/bin/env python3
"""Synthetic file-binding tests only; never a real qualification receipt."""
from pathlib import Path
import importlib.util,json,tempfile,unittest,hashlib
M=Path(__file__).with_name('SCELLER_ENREGISTREMENT.py');sp=importlib.util.spec_from_file_location('seal',M);s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
def write(p,x):p.write_text(json.dumps(x)+'\n')
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name)
  self.study=self.r/'STUDY_CONTRACT.json';write(self.study,dict(synthetic_fixture=True))
  self.binary=self.r/'not_an_executable';self.binary.write_bytes(b'fixture only');self.installed=self.r/'installed_reference';self.installed.write_bytes(b'installed fixture only')
  self.build=self.r/'BUILD.json';b=dict(homogeneous_compilation=True,original_installation_unchanged=True,experiment_source_sha256={'fixture.cpp':'a'*64},backends={k:dict(experiment_binary=str(self.binary),experiment_binary_sha256=s.sha(self.binary))for k in('ma27','mumps')});write(self.build,b)
  self.prep=self.r/'PREPARATION.json';write(self.prep,dict(study_contract_sha256=s.sha(self.study),build_manifest_sha256=s.sha(self.build)))
  self.audit=self.r/'SOURCE_SEPARATION_AUDIT.json';write(self.audit,dict(passed=True,source_tree_sha256=s.digest(b['experiment_source_sha256']),synthetic_fixture=True))
  self.schema=self.r/'DIRECTION_TRACE_SCHEMA.json';write(self.schema,dict(synthetic_fixture=True))
  self.gate=self.r/'GATE50.json';guards=dict(ma27='b'*64,mumps='b'*64)
  write(self.gate,dict(passed=True,production_authorized=True,reference_reproduction_passed=True,study_intervention_verified=True,bound_files_unchanged=True,completed_native_runs=748,cases=50,installed_ma27_reference_passed=True,installed_ma27_reference_runs=100,installed_ma27_reference_binary=str(self.installed),installed_ma27_reference_binary_sha256=s.sha(self.installed),study_contract_sha256=s.sha(self.study),preparation_sha256=s.sha(self.prep),build_manifest_sha256=s.sha(self.build),registration_sha256=s.sha(self.study),intervention_checks=dict(guard_config_sha256_by_backend=guards,common_guard_vector_sha256=s.digest(guards)),synthetic_fixture=True))
  self.out=self.r/'PRODUCTION_REGISTRATION.json'
 def tearDown(self):self.tmp.cleanup()
 def call(self):return s.seal_registration(self.study,self.prep,self.build,self.gate,self.out)
 def test_complete_bindings_can_seal_a_synthetic_fixture(self):
  self.call();r=json.loads(self.out.read_text());self.assertEqual(r['registration_status'],'sealed_before_production');self.assertEqual(r['qualification_registration_sha256'],s.sha(self.study));self.assertEqual(r['bindings']['qualification50_receipt_sha256'],s.sha(self.gate))
 def test_resume_reuses_exact_registration_without_changing_bytes(self):
  self.call();before=s.sha(self.out);self.call();self.assertEqual(s.sha(self.out),before)
 def test_failed_real_gate_flag_blocks_seal(self):
  x=json.loads(self.gate.read_text());x['passed']=False;write(self.gate,x)
  with self.assertRaisesRegex(ValueError,'qualification'):self.call()
  self.assertFalse(self.out.exists())
 def test_short_execution_count_blocks_seal(self):
  x=json.loads(self.gate.read_text());x['completed_native_runs']=747;write(self.gate,x)
  with self.assertRaisesRegex(ValueError,'748'):self.call()
 def test_changed_source_audit_does_not_qualify_build(self):
  x=json.loads(self.audit.read_text());x['source_tree_sha256']='d'*64;write(self.audit,x)
  with self.assertRaisesRegex(ValueError,'source separation'):self.call()
 def test_changed_binary_after_gate_blocks_seal(self):
  self.binary.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'binary identity'):self.call()
 def test_changed_static_study_blocks_seal(self):
  with self.study.open('a')as f:f.write(' ')
  with self.assertRaisesRegex(ValueError,'static study'):self.call()
 def test_conflicting_existing_registration_is_preserved(self):
  self.call();x=json.loads(self.out.read_text());x['study_contract_sha256']='d'*64;write(self.out,x);before=s.sha(self.out)
  with self.assertRaisesRegex(ValueError,'must not be overwritten'):self.call()
  self.assertEqual(s.sha(self.out),before)
 def test_installed_MA27_reference_cannot_be_omitted(self):
  x=json.loads(self.gate.read_text());x['installed_ma27_reference_passed']=False;write(self.gate,x)
  with self.assertRaisesRegex(ValueError,'installed MA27'):self.call()
  self.assertFalse(self.out.exists())
 def test_installed_MA27_bytes_cannot_change_after_qualification(self):
  self.installed.write_bytes(b'changed installed binary')
  with self.assertRaisesRegex(ValueError,'Installed MA27 reference changed'):self.call()
  self.assertFalse(self.out.exists())
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));Path(__file__).with_name('TEST_SCELLEMENT_VALIDATION.json').write_text(json.dumps(dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),sealer_sha256=hashlib.sha256(M.read_bytes()).hexdigest(),scope='Synthetic file/provenance fixtures only; no real native qualification or production authorization',optimizer_invocations=0),indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful()else 1)
