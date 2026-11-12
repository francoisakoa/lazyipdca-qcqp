#!/usr/bin/env python3
"""Pin the user's actual HSL runtime. This module supplies no HSL library."""
from pathlib import Path
import ctypes,hashlib,json,os,re,subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin_hsl(path,expected_sha256=None):
    path=Path(path).expanduser().resolve(strict=True)
    actual=sha(path)
    if expected_sha256 is not None and actual!=expected_sha256:raise ValueError('HSL library differs from its recorded SHA256')
    lib=ctypes.CDLL(str(path),mode=os.RTLD_NOW|os.RTLD_LOCAL)
    for name in ('mock_ma27_test_double','fake_ma27_test_double'):
        if hasattr(lib,name):raise ValueError('A MA27 test double cannot qualify or run the native campaign')
    symbols={}
    for name in ('ma27id','ma27ad','ma27bd','ma27cd'):
        found=next((s for s in (name+'_',name,name.upper(),name.upper()+'_')if hasattr(lib,s)),None)
        if found is None:raise ValueError('The supplied real HSL runtime lacks '+name)
        symbols[name]=found
    p=subprocess.run(['ldd',str(path)],capture_output=True,text=True)
    if p.returncode or'not found'in p.stdout:raise ValueError('HSL runtime dependency resolution failed: '+p.stdout+p.stderr)
    deps={m.group(2):sha(m.group(2))for m in re.finditer(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(',p.stdout,re.M)}
    return dict(schema='native-ma27-hsl-runtime-pin-v1',native_backend='ma27',hsl_library=str(path),hsl_library_sha256=actual,required_symbols=symbols,test_double=False,ldd=p.stdout,dependency_sha256=deps,numerically_qualified=False,scope='Actual library loading and symbol/dependency pin only; native qualification is separate')
def verify_pin(proof):
    if proof.get('schema')!='native-ma27-hsl-runtime-pin-v1'or proof.get('native_backend')!='ma27'or proof.get('test_double')is not False:raise ValueError('Actual MA27 HSL runtime pin absent')
    now=pin_hsl(proof['hsl_library'],proof['hsl_library_sha256'])
    if now['required_symbols']!=proof['required_symbols']or now['dependency_sha256']!=proof['dependency_sha256']:raise ValueError('MA27 loaded runtime or dependencies changed')
    return now
