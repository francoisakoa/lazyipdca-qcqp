#!/usr/bin/env python3
"""Seal production only after source and real native qualification gates pass.

The static study contract precedes qualification. The final registration binds
that contract and the completed gate; the gate never binds this later seal.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,tempfile

def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def equivalent_registration(left,right):
 left=dict(left);right=dict(right);left.pop('sealed_utc',None);right.pop('sealed_utc',None);return left==right
def atom(path,value):
 path=Path(path)
 if path.exists():
  if equivalent_registration(read(path),value):return
  raise ValueError('Existing production registration differs and must not be overwritten: '+str(path))
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.tmp.',dir=path.parent)
 try:
  with os.fdopen(fd,'w')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  try:os.link(tmp,path)
  except FileExistsError:
   if not equivalent_registration(read(path),value):raise ValueError('Concurrent production registration differs')
 finally:os.unlink(tmp)
def seal_registration(study_contract,preparation,build,gate,out,source_audit=None,direction_schema=None):
 paths={k:Path(v).expanduser().resolve(strict=True)for k,v in dict(study_contract=study_contract,preparation=preparation,build=build,gate=gate).items()}
 study=read(paths['study_contract']);prep=read(paths['preparation']);b=read(paths['build']);g=read(paths['gate']);study_sha=sha(paths['study_contract'])
 if g.get('study_contract_sha256')!=study_sha or prep.get('study_contract_sha256')!=study_sha:raise ValueError('Qualification and preparation do not bind the static study')
 if g.get('preparation_sha256',g.get('preparation_manifest_sha256'))!=sha(paths['preparation']):raise ValueError('Gate does not bind the exact preparation')
 if g.get('build_manifest_sha256')!=sha(paths['build'])or prep.get('build_manifest_sha256')!=sha(paths['build']):raise ValueError('Build identity differs')
 for key in('passed','production_authorized','reference_reproduction_passed','study_intervention_verified','bound_files_unchanged'):
  if g.get(key)is not True:raise ValueError('Native qualification is absent, incomplete or failed: '+key)
 if g.get('completed_native_runs')!=748 or g.get('cases')!=50:raise ValueError('748 complete native qualification executions required')
 if g.get('installed_ma27_reference_passed')is not True or g.get('installed_ma27_reference_runs')!=100:raise ValueError('One hundred comparisons with the real installed MA27 reference are required')
 installed=Path(g.get('installed_ma27_reference_binary','')).expanduser().resolve(strict=True)
 if not installed.is_file()or sha(installed)!=g.get('installed_ma27_reference_binary_sha256'):raise ValueError('Installed MA27 reference changed after qualification')
 if b.get('homogeneous_compilation')is not True or b.get('original_installation_unchanged')is not True:raise ValueError('Build admission unavailable')
 source_map=b.get('experiment_source_sha256')
 if not isinstance(source_map,dict)or not source_map:raise ValueError('Experiment source map absent')
 source_tree=digest(source_map)
 audit_path=Path(source_audit or paths['study_contract'].parent/'SOURCE_SEPARATION_AUDIT.json').resolve(strict=True);audit=read(audit_path)
 if audit.get('passed')is not True or audit.get('source_tree_sha256')!=source_tree:raise ValueError('Passed source separation audit is not bound to the compiled source tree')
 schema_path=Path(direction_schema or paths['study_contract'].parent/'DIRECTION_TRACE_SCHEMA.json').resolve(strict=True)
 binaries={}
 for backend in('ma27','mumps'):
  cfg=b.get('backends',{}).get(backend,{})
  path=Path(cfg.get('experiment_binary','')).expanduser().resolve(strict=True)
  if not path.is_file()or cfg.get('experiment_binary_sha256')!=sha(path):raise ValueError('Experiment binary identity differs: '+backend)
  binaries[backend]=sha(path)
 if len(set(binaries.values()))!=1:raise ValueError('Both backend blocks must use the same homogeneous experiment binary')
 intervention=g.get('intervention_checks')or{};guard_map=intervention.get('guard_config_sha256_by_backend')
 if not isinstance(guard_map,dict)or set(guard_map)!={'ma27','mumps'}or any(not isinstance(x,str)or len(x)!=64 for x in guard_map.values()):raise ValueError('Qualified common guard vector is absent')
 if intervention.get('common_guard_vector_sha256')!=digest(guard_map):raise ValueError('Common guard vector digest differs')
 result=dict(schema='adaptive-refinement-production-registration-v2',registration_status='sealed_before_production',sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  study_contract_file=os.path.relpath(paths['study_contract'],Path(out).resolve().parent),study_contract_sha256=study_sha,
  bindings=dict(source_tree_sha256=source_tree,binary_sha256_by_backend=binaries,source_separation_audit_sha256=sha(audit_path),qualification50_receipt_sha256=sha(paths['gate']),direction_trace_schema_sha256=sha(schema_path),common_guard_vector_sha256=digest(guard_map)),
  evidence={k:dict(path=str(v),sha256=sha(v))for k,v in dict(**paths,source_audit=audit_path,direction_schema=schema_path).items()},
  qualified_guard_config_sha256_by_backend=guard_map,qualification_registration_sha256=g.get('registration_sha256'),qualification_registration_role='Static study or draft protocol only; production observations bind the SHA256 of this completed seal',
  installed_ma27_reference_binary=str(installed),installed_ma27_reference_binary_sha256=sha(installed),qualification_completed_runs=748,production_tasks=14672,production_outcomes_read=False)
 atom(out,result);return Path(out).resolve()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in('study-contract','preparation','build','gate','out'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--source-audit',type=Path);p.add_argument('--direction-schema',type=Path);a=p.parse_args()
 path=seal_registration(a.study_contract,a.preparation,a.build,a.gate,a.out,a.source_audit,a.direction_schema);print(json.dumps(dict(registration=str(path),registration_sha256=sha(path),sealed_before_production=True),indent=2))
if __name__=='__main__':main()
