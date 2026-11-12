#!/usr/bin/env python3
"""Restore runner source directories; this prepares files without launching solvers."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil
from prepare_profile import prepare
HERE=Path(__file__).resolve().parent
INSERTIONS={'ma27_main':{'factorial/source':'factorial_observer','factorial/qualified_source':'mumps'},'aligned_refinement':{'native/reference_source':'ma27','native/experiment_source':'refinement'},'asu':{'native/reference_source':'ma27'},'persistence':{'native/reference_source':'ma27'}}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner',choices=INSERTIONS,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit('Output directory must be empty')
 shutil.copytree(HERE/'runners'/a.runner,out,dirs_exist_ok=True)
 for rel,profile in INSERTIONS[a.runner].items():prepare(profile,out/rel)
 compressed=json.loads((HERE/'provenance/COMPRESSED_RUNNER_FILES.json').read_text())
 for name,receipt in compressed.items():
  if name.startswith(a.runner+'/'):
   rel=name[len(a.runner)+1:];target=out/rel;packed=out/(rel+'.gz');data=gzip.decompress(packed.read_bytes())
   if hashlib.sha256(data).hexdigest()!=receipt['sha256']:raise RuntimeError('Compressed runner file mismatch')
   target.write_bytes(data);packed.unlink()
 print(json.dumps({'runner':a.runner,'path':str(out),'optimizer_invocations':0,'external_inputs_required':True}))
