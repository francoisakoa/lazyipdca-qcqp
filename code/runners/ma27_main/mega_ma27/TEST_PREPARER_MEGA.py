#!/usr/bin/env python3
"""Staging/wiring tests only. Every native/HSL qualification is synthetic here."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import builtins,copy,gzip,hashlib,json,sys,tempfile,unittest
import PREPARER_MEGA as p
HERE=Path(__file__).resolve().parent

def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
class StageTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='staging MA27 ');self.root=Path(self.temp.name);self.old=self.root/'old';self.old.mkdir()
  self.oldcfg={'native_cutest_qualification':'OLD_MUMPS_132_MUST_NOT_BE_REUSED','native_compilation_reused_byte_exact':True}
  put(self.old/'INSTALLATION.json',self.oldcfg)
  contract=json.load(gzip.open(p.DEFAULT/'CONTRAT_4078.json.gz','rt'))
  self.cases=[dict(id=c['case_id'],input=str(self.old/'canonical_inputs'/c['case_id']/'input.qcq'),input_sha256=c['input_sha256'],regime=c['regime'])for c in contract['cases']]
  self.prepared=dict(cases=self.cases,counts={'cases':4078},beamforming_file={'scope':'unchanged previous reader'})
  put(self.old/'PREPARATION.json',self.prepared)
  src=self.root/'adapted_reference_source';src.mkdir();(src/'source.cpp').write_text('synthetic source wiring fixture')
  binary=self.root/'synthetic_reference';binary.write_text('NOT A NATIVE EXECUTABLE')
  q=put(self.root/'native_qualification.json',dict(passed=True,six_profiles_qualified=True,binary_sha256=p.common.sha(binary),hsl_library_sha256='HSL_FIXTURE',synthetic_only=True))
  deps=put(self.root/'deps.json',{'all_loaded_library_sha256':{}});hsl=put(self.root/'HSL.json',{'dependency_sha256':{}})
  buildfile=put(self.root/'BUILD.json',{'synthetic_only':True})
  self.build=dict(ma27_reference_source=str(src),ma27_reference_binary=str(binary),native_ma27_qualification=str(q),native_dependency_proof=str(deps),hsl_runtime_proof=str(hsl),runtime_environment={},
   _proof=str(buildfile),ma27_reference_source_sha256={'source.cpp':p.common.sha(src/'source.cpp')},baseline_source_sha256={},ma27_admissions={'baseline':{'synthetic_only':True}})
  df=put(self.root/'DEFAULT.json',{'synthetic_only':True})
  self.defaults=dict(hsl_library=str(self.root/'realHSL_not_loaded_in_wiring_test'),hsl_library_sha256='HSL_FIXTURE',
   ipopt_library=str(self.root/'ipopt_not_loaded'),ipopt_library_sha256='IPOPT_FIXTURE',default_binary=str(self.root/'ipopt_defaults_not_run'),default_binary_sha256='DEFAULT_BINARY_FIXTURE',
   _path=str(df),build=str(buildfile),build_sha256=p.common.sha(buildfile),qualification=str(q),qualification_sha256=p.common.sha(q),
   frozen_resolution=str(self.root/'resolution_not_read'),frozen_resolution_sha256='RESOLUTION_FIXTURE',
   frozen_bindings={c['id']:dict(input=c['input'],input_sha256=c['input_sha256'])for c in self.cases})
  self.gate=[];self.bind=[];self.original_import=builtins.__import__;self.path=list(sys.path)
 def tearDown(self):sys.path[:]=self.path;self.temp.cleanup()
 def import_hook(self,name,*args,**kwargs):
  if name=='MEGA':return SimpleNamespace()
  if name=='prepare_inventory':return SimpleNamespace(bind_preparation=lambda root,cfg,inv:self.bind.append((root,cfg,inv)))
  return self.original_import(name,*args,**kwargs)
 def stage(self):
  fake=SimpleNamespace(qualify=lambda root,**kw:self.gate.append((root,kw)))
  with patch.object(p,'load',return_value=fake),patch('builtins.__import__',side_effect=self.import_hook):
   return p.stage_previous_device(self.root/'device',self.old,self.oldcfg,self.prepared,self.defaults,self.build,False,None)
 def test_same_previous_inputs_and_six_commands_with_new_backend_and_no_old_qualification(self):
  oldbytes=(self.old/'PREPARATION.json').read_bytes();cfg=self.stage()
  self.assertEqual((self.old/'PREPARATION.json').read_bytes(),oldbytes)
  self.assertEqual([c['input']for c in self.bind[0][2]['cases']],[c['input']for c in self.cases])
  profiles=json.loads((self.root/'device/campaign/profiles.json').read_text())
  published=json.loads((HERE/'campaign/profiles.json').read_text())
  self.assertEqual(profiles['profiles'],published['profiles']);self.assertEqual(profiles['backend'],'ma27')
  self.assertNotIn('native_cutest_qualification',cfg);self.assertFalse(cfg['native_compilation_reused_byte_exact'])
  self.assertEqual(cfg['ipopt_methods'],{'ipopt_qcqp':'ma27','ipopt_mumps':'mumps'})
  self.assertEqual(cfg['ipopt_mumps_ordering_policy'],'library-default');self.assertEqual(len(self.gate),1);self.assertEqual(len(self.bind),1)
 def test_wrong_declared_canonical_path_blocks_before_audit_or_binding(self):
  self.defaults['frozen_bindings'][self.cases[0]['id']]['input']=str(self.old/'data/frozen/wrong_serialization/input.qcq')
  with self.assertRaisesRegex(ValueError,'Chemin gelé'):self.stage()
  self.assertFalse(self.gate);self.assertFalse(self.bind)
 def test_native_qualification_other_HSL_blocks_before_any_plan(self):
  q=Path(self.build['native_ma27_qualification']);value=json.loads(q.read_text());value['hsl_library_sha256']='OTHER_HSL';put(q,value)
  with self.assertRaisesRegex(ValueError,'autre binaire ou HSL'):self.stage()
  self.assertFalse(self.gate);self.assertFalse(self.bind)
if __name__=='__main__':unittest.main(verbosity=2)
