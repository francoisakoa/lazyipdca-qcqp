#!/usr/bin/env python3
"""Reconstruct a frozen source profile and verify every file against its manifest."""
from pathlib import Path
import argparse,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
PROFILES=json.loads((HERE/'native/PROFILES.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(root):return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
def prepare(profile,out):
 out=Path(out).resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty: '+str(out))
 chain=[];current=profile
 while current:
  chain.append(current);current=PROFILES[current]['parent']
 shutil.copytree(HERE/'native/base',out,dirs_exist_ok=True)
 for name in reversed(chain):
  overlay=HERE/'native/overlays'/name
  if overlay.is_dir():shutil.copytree(overlay,out,dirs_exist_ok=True)
  for rel in PROFILES[name]['removed_files']:(out/rel).unlink()
 got=snapshot(out);expected=PROFILES[profile]['sha256']
 if got!=expected:raise RuntimeError('Source manifest mismatch: '+profile)
 print(json.dumps({'profile':profile,'source':str(out),'verified_files':len(got),'byte_exact':True}))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.profile,a.out)
