#!/usr/bin/env python3
from pathlib import Path
import hashlib,importlib.util,json,tempfile,unittest
from unittest.mock import patch
import FROZEN_INPUTS as r
FIXTURE=b'LAZYIPDCA_QCQP_V2\nname frozen\nn 1\nx0 0.25\nlb -inf\nub inf\nobjective\nquadratic\nname f\nconstant 0\nlinear_nnz 0\nhessian_nnz 1\n0 0 2\nend_quadratic\nequalities 0\ninequalities 0\nend_problem\n'
def case(cid='public_label',data=FIXTURE):
 line=next(v for v in data.splitlines(keepends=True)if v.startswith(b'x0 '))
 return dict(case_id=cid,dataset='example',regime='I',input_sha256=hashlib.sha256(data).hexdigest(),
  start_binding=dict(external_start_required=False,raw_x0_line_sha256=hashlib.sha256(line).hexdigest(),start_sha256='declared'))
def put(root,rel,data=FIXTURE):
 p=root/'data/frozen'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
class Tests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.report=self.root/'report.json'
 def tearDown(self):self.temp.cleanup()
 def run_plan(self,cases):return r.resolve_all(self.root,cases,self.report)
 def test_noncanonical_name_is_selected_by_bytes_and_raw_start(self):
  p=put(self.root,'private_storage/arbitrary_model.qcq');old=p.read_bytes();bindings,report=self.run_plan([case()])
  self.assertEqual(bindings['public_label']['input'],str(p.resolve()));self.assertEqual(report['relocated'],1)
  self.assertTrue(report['all_inputs_and_raw_starts_verified']);self.assertEqual(p.read_bytes(),old)
 def test_previous_canonical_layer_is_used_instead_of_raw_serialization(self):
  raw=FIXTURE.replace(b'LAZYIPDCA_QCQP_V2',b'LAZYIPDCA_QCQP_V1').replace(b'linear_nnz 0',b'linear 0')
  old=put(self.root,'public_label/input.qcq',raw)
  canonical=self.root/'canonical_inputs/public_label/input.qcq';canonical.parent.mkdir(parents=True);canonical.write_bytes(FIXTURE)
  (self.root/'PREPARATION.json').write_text(json.dumps({'cases':[dict(id='public_label',input=str(canonical),input_sha256=case()['input_sha256'])]}))
  bindings,report=self.run_plan([case()]);self.assertEqual(bindings['public_label']['input'],str(canonical))
  self.assertEqual(report['declared_paths'],1);self.assertEqual(old.read_bytes(),raw)
 def test_declared_previous_path_outside_storage_roots_is_hash_verified(self):
  external=self.root/'previous_explicit_input';external.write_bytes(FIXTURE)
  (self.root/'PREPARATION.json').write_text(json.dumps({'cases':[dict(id='public_label',input=str(external),input_sha256=case()['input_sha256'])]}))
  bindings,report=self.run_plan([case()]);self.assertEqual(bindings['public_label']['input'],str(external))
 def test_previous_preparation_cannot_override_public_hash(self):
  (self.root/'PREPARATION.json').write_text(json.dumps({'cases':[dict(id='public_label',input='different',input_sha256='0'*64)]}))
  with self.assertRaisesRegex(ValueError,'autre empreinte'):self.run_plan([case()])
 def test_wrong_nominal_model_does_not_replace_published_hash(self):
  wrong=put(self.root,'public_label/input.qcq',FIXTURE.replace(b'x0 0.25',b'x0 0.5'))
  actual=put(self.root,'other/physical_name.qcq');before=wrong.read_bytes()
  bindings,report=self.run_plan([case()]);row=report['rows'][0]
  self.assertEqual(row['nominal_state'],'different');self.assertEqual(row['nominal_observed_sha256'],r.sha(wrong))
  self.assertEqual(bindings['public_label']['input'],str(actual.resolve()));self.assertEqual(wrong.read_bytes(),before)
 def test_absent_and_different_are_reported_separately(self):
  put(self.root,'different/input.qcq',FIXTURE.replace(b'x0 0.25',b'x0 0.5'))
  with self.assertRaises(r.FrozenInputsError):self.run_plan([case('absent'),case('different')])
  report=json.loads(self.report.read_text());self.assertEqual(report['missing_or_invalid'],2)
  self.assertEqual([x['nominal_state']for x in report['rows']],['absent','different'])
  self.assertEqual(report['verified'],0)
 def test_wrong_raw_start_binding_is_never_accepted(self):
  put(self.root,'model.qcq');c=case();c['start_binding']['raw_x0_line_sha256']='0'*64
  with self.assertRaises(r.FrozenInputsError):self.run_plan([c])
  self.assertEqual(json.loads(self.report.read_text())['rows'][0]['status'],'start_or_input_invalid')
 def test_line_ending_change_is_not_normalized(self):
  put(self.root,'public_label/input.qcq',FIXTURE.replace(b'\n',b'\r\n'))
  with self.assertRaises(r.FrozenInputsError):self.run_plan([case()])
  self.assertFalse(json.loads(self.report.read_text())['all_inputs_and_raw_starts_verified'])
 def test_equal_content_does_not_merge_case_identifiers(self):
  put(self.root,'storage.qcq');bindings,report=self.run_plan([case('one'),case('two')])
  self.assertEqual(set(bindings),{'one','two'});self.assertEqual(report['verified'],2)
 def test_directory_symlink_and_cycle_terminate_without_rewriting_data(self):
  physical=self.root/'physical';physical.mkdir();p=physical/'model.qcq';p.write_bytes(FIXTURE)
  (self.root/'data').mkdir();(self.root/'data/frozen').symlink_to(physical,target_is_directory=True)
  (physical/'cycle').symlink_to(physical,target_is_directory=True)
  bindings,report=self.run_plan([case()]);self.assertEqual(report['hashed_files'],1)
  self.assertEqual(bindings['public_label']['input'],str(p.resolve()));self.assertEqual(p.read_bytes(),FIXTURE)
 def test_existing_nominal_path_is_preferred(self):
  p=put(self.root,'public_label/input.qcq');put(self.root,'aaa_duplicate/model.qcq')
  bindings,report=self.run_plan([case()]);self.assertEqual(bindings['public_label']['input'],str(p.resolve()))
  self.assertEqual(report['relocated'],0)
 def test_mutation_during_raw_start_check_blocks_resolution(self):
  p=put(self.root,'storage.qcq');original=r.verify_raw_start
  def changing(path,c):
   result=original(path,c);p.write_bytes(FIXTURE.replace(b'x0 0.25',b'x0 0.5'));return result
  with patch.object(r,'verify_raw_start',changing),self.assertRaises(r.FrozenInputsError):self.run_plan([case()])
  self.assertEqual(json.loads(self.report.read_text())['verified'],0)
 def test_extra_frozen_root_is_explicit_and_hash_verified(self):
  other=self.root/'second_frozen';other.mkdir();p=other/'model.qcq';p.write_bytes(FIXTURE)
  bindings,report=r.resolve_all(self.root,[case()],self.report,[other]);self.assertEqual(bindings['public_label']['input'],str(p))
  self.assertEqual(report['verified'],1)
 def test_compressed_bytes_are_not_silently_reinterpreted(self):
  import gzip
  put(self.root,'storage/input.qcq.gz',gzip.compress(FIXTURE))
  with self.assertRaises(r.FrozenInputsError):self.run_plan([case()])
 def test_complete_missing_frozen_root_writes_diagnostic(self):
  with self.assertRaises(r.FrozenInputsError):self.run_plan([case()])
  report=json.loads(self.report.read_text());self.assertEqual(report['verified'],0);self.assertTrue(report['scan_issues'])
if __name__=='__main__':unittest.main(verbosity=2)
