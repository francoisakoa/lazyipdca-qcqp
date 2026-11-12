#!/usr/bin/env python3
"""Verify all frozen source profiles and the independent original-KKT auditor."""
from pathlib import Path
import argparse,json,subprocess,sys,tempfile
from prepare_profile import PROFILES,prepare
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 result={'schema':'lazyqcqp-code-check-v1','profiles':{},'optimizer_invocations':0}
 with tempfile.TemporaryDirectory(prefix='lazyqcqp_sources_')as tmp:
  for name in PROFILES:
   prepare(name,Path(tmp)/name);result['profiles'][name]={'verified_files':PROFILES[name]['file_count'],'byte_exact':True}
 run=subprocess.run([sys.executable,str(HERE/'audit/TEST_AUDIT.py')],capture_output=True,text=True)
 result['auditor_tests']={'returncode':run.returncode,'output':run.stdout.strip()};assert run.returncode==0,run.stdout+run.stderr
 for kind in ['convex','nonconvex']:
  out=a.out/('AUDIT_'+kind+'.json')
  run=subprocess.run([sys.executable,str(HERE/'audit/AUDIT_ORIGINAL.py'),'--input',str(HERE/'audit'/(kind+'.qcq')),'--endpoint',str(HERE/'audit'/(kind+'_reference')/'endpoint.json'),'--out',str(out)],capture_output=True,text=True)
  d=json.loads(out.read_text());assert run.returncode==0 and d['strict_qcqp'] is True,(run.stdout,run.stderr,d)
  result[kind+'_endpoint_audit']={'strict_qcqp':True,'objective':d['objective'],'input_sha256':d['input_sha256'],'endpoint_sha256':d['endpoint_sha256']}
 result['passed']=True
 (a.out/'CODE_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
