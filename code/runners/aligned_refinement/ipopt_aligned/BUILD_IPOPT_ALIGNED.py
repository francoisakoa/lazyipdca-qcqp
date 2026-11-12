#!/usr/bin/env python3
"""Build an isolated aligned-options Ipopt comparator against the campaign library."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,time

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def library_from_binary(binary):
    p=subprocess.run(['ldd',str(binary)],capture_output=True,text=True)
    if p.returncode:raise RuntimeError('Cannot inspect linked campaign comparator: '+p.stderr)
    for line in p.stdout.splitlines():
        if re.search(r'\blibipopt\.so',line):
            match=re.search(r'=>\s+(/\S+)',line)
            if match:return Path(match.group(1)).resolve()
    raise RuntimeError('No dynamic Ipopt library found in the existing campaign binary')

def locate(a):
    installation=a.root/'INSTALLATION.json'
    baseline_binary=None
    if installation.is_file():
        cfg=json.loads(installation.read_text())
        if cfg.get('ipopt_binary'):
            baseline_binary=Path(cfg['ipopt_binary'])
            if not baseline_binary.is_absolute():baseline_binary=a.root/baseline_binary
            if not baseline_binary.is_file():raise RuntimeError('Recorded comparator absent: '+str(baseline_binary))
            if cfg.get('ipopt_binary_sha256') and sha(baseline_binary)!=cfg['ipopt_binary_sha256']:
                raise RuntimeError('Existing campaign comparator digest differs from INSTALLATION')
    if a.ipopt_library:
        library=a.ipopt_library.resolve()
        if baseline_binary and library_from_binary(baseline_binary)!=library:
            raise RuntimeError('Explicit library differs from existing campaign library')
    elif baseline_binary:
        library=library_from_binary(baseline_binary)
    else:
        raise RuntimeError('No qualified existing comparator; give --ipopt-library and --ipopt-include explicitly for a local qualification only')
    if not library.is_file():raise RuntimeError('Ipopt library absent: '+str(library))
    candidates=[]
    if a.ipopt_include:candidates.append(a.ipopt_include)
    if a.ipopt_prefix:candidates += [a.ipopt_prefix/'include/coin-or',a.ipopt_prefix/'include/coin',a.ipopt_prefix/'include']
    for parent in [library.parent,library.parent.parent,a.root]:
        candidates += [parent/'include/coin-or',parent/'include/coin',parent/'include',parent/'ipopt/build/compat/coin-or']
    candidates += [Path('/usr/include/coin-or'),Path('/usr/include/coin')]
    headers=next((p.resolve()for p in candidates if (p/'IpIpoptApplication.hpp').is_file()),None)
    if headers is None:raise RuntimeError('Headers absent; supply --ipopt-include directory containing IpIpoptApplication.hpp')
    version=None
    for config in [headers/'IpoptConfig.h',headers/'IpoptConfig.hpp']:
        if config.is_file():
            text=config.read_text()
            m=re.search(r'#define\s+IPOPT_VERSION\s+"([^"]+)"',text)
            if m:version=m.group(1);break
    if version!=a.expected_version:
        raise RuntimeError('Headers version '+str(version)+' differs from required '+a.expected_version)
    symbols=subprocess.run(['nm','-D',str(library)],capture_output=True,text=True,check=True).stdout
    names=[s for s in symbols.splitlines()if 'OptionsList' in s and 'SetStringValue' in s]
    if any('NSt7__cxx11' in s for s in names):abi='1'
    elif any('ERKSs' in s for s in names):abi='0'
    else:raise RuntimeError('Cannot determine Ipopt C++ string ABI from exported OptionsList symbols')
    return library,headers,version,abi,baseline_binary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ipopt-library',type=Path)
    p.add_argument('--ipopt-include',type=Path)
    p.add_argument('--ipopt-prefix',type=Path)
    p.add_argument('--expected-version',default='3.14.11')
    p.add_argument('--hsl-library',type=Path)
    p.add_argument('--dependency-receipt',type=Path,default=HERE/'DEPENDENCIES_TRX.json')
    p.add_argument('--local-qualification-only',action='store_true')
    p.add_argument('--cxx',default=os.environ.get('CXX','g++'))
    a=p.parse_args();a.root=a.root.resolve();out=a.out.resolve()
    if out.exists():p.error('Output must be fresh; no existing installation or binary is overwritten')
    if a.expected_version!='3.14.11':p.error('This contract requires Ipopt 3.14.11')
    deps=json.loads(a.dependency_receipt.read_text())
    if not a.ipopt_library and not (a.root/'INSTALLATION.json').is_file():
        a.ipopt_library=Path(deps['ipopt_library'])
    if not a.ipopt_include and Path(deps['ipopt_headers']).is_dir():a.ipopt_include=Path(deps['ipopt_headers'])
    library,headers,version,abi,baseline=locate(a)
    hsl=(a.hsl_library or Path(deps['hsl_library'])).resolve()
    if not a.local_qualification_only:
        if sha(library)!=deps['ipopt_library_sha256']:raise RuntimeError('Ipopt dependency differs from audited TRX receipt')
        if not hsl.is_file() or sha(hsl)!=deps['hsl_library_sha256']:raise RuntimeError('HSL dependency absent or differs from audited TRX receipt')
    elif not hsl.is_file():hsl=None
    baseline_sha=sha(baseline) if baseline else None
    installation_sha=sha(a.root/'INSTALLATION.json') if (a.root/'INSTALLATION.json').is_file() else None
    source=HERE/'adapter'
    def overlaps(left,right):return left==right or left in right.parents or right in left.parents
    for protected in [a.root,source,library.parent,headers]+([hsl.parent]if hsl else []):
        if overlaps(out,protected.resolve()):raise RuntimeError('Fresh build output overlaps protected path: '+str(protected))
    library_sha_before=sha(library);hsl_sha_before=sha(hsl)if hsl else None
    header_hashes={str(q.relative_to(headers)):sha(q)for q in sorted(headers.rglob('*'))if q.is_file()}
    out.mkdir(parents=True)
    compat=out/'compat/coin-or';compat.mkdir(parents=True)
    for h in headers.glob('*'):
        if h.is_file():(compat/h.name).symlink_to(h)
    source=HERE/'adapter' if (HERE/'adapter').is_dir() else HERE/'source'
    contract=json.loads((HERE/'PATCH_CONTRACT.json').read_text())
    for name,value in contract['source_sha256'].items():
        if sha(source/name)!=value:raise RuntimeError('Distributed default source differs: '+name)
    output=out/'ipopt_aligned_cli'
    command=[a.cxx,'-O2','-std=c++17','-DHAVE_CSTDDEF=1',
             '-D_GLIBCXX_USE_CXX11_ABI='+abi,'-DLAZYIPDCA_HAVE_IPOPT=1',
             '-I'+str(source/'include'),'-I'+str(source/'src'),'-I'+str(out/'compat'),
             *[str(source/'src'/n)for n in ('direct_ipopt.cpp','ipopt_qcqp_main.cpp','io.cpp','io_compact.cpp')],
             str(library),'-Wl,-rpath,'+str(library.parent),'-o',str(output)]
    started=time.monotonic()
    with(out/'compile.log').open('w')as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Aligned comparator compilation failed; see '+str(out/'compile.log'))
    env=dict(os.environ)
    runtime_environment={'LD_LIBRARY_PATH':str(library.parent)+((':'+str(hsl.parent))if hsl else '')}
    runtime_environment.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    env.update(runtime_environment)
    env.pop('LD_PRELOAD',None)
    linked=subprocess.run(['ldd',str(output)],env=env,capture_output=True,text=True,check=True)
    loaded={}
    for line in linked.stdout.splitlines():
        match=re.search(r'^\s*(\S+)\s+=>\s+(/\S+)',line)
        if match:
            path=Path(match.group(2)).resolve()
            if path.is_file():loaded[match.group(1)]={'path':str(path),'sha256':sha(path)}
    actual_ipopt=[v for key,v in loaded.items()if key.startswith('libipopt.so')]
    if len(actual_ipopt)!=1 or actual_ipopt[0]['sha256']!=library_sha_before:
        raise RuntimeError('Built binary resolves a different Ipopt library under the declared environment')
    hsl_loaded={}
    if hsl:
        hsl_ldd=subprocess.run(['ldd',str(hsl)],env=env,capture_output=True,text=True,check=True)
        for line in hsl_ldd.stdout.splitlines():
            match=re.search(r'^\s*(\S+)\s+=>\s+(/\S+)',line)
            if match and Path(match.group(2)).is_file():hsl_loaded[match.group(1)]={'path':str(Path(match.group(2)).resolve()),'sha256':sha(match.group(2))}
        if 'not found' in hsl_ldd.stdout:raise RuntimeError('Unresolved HSL transitive dependency')
    if 'not found' in linked.stdout:raise RuntimeError('Unresolved built comparator dependency')
    probe=subprocess.run([str(output),'--version'],env=env,capture_output=True,text=True)
    if probe.returncode or version not in probe.stdout:
        raise RuntimeError('Built comparator version check failed: '+probe.stdout+probe.stderr)
    report={'schema':'isolated-ipopt-aligned-options-build-v1','binary':str(output),
            'binary_sha256':sha(output),'ipopt_library':str(library),'ipopt_library_sha256':sha(library),
            'ipopt_headers':str(headers),'ipopt_version':version,'string_ABI':int(abi),
            'hsl_library':str(hsl)if hsl else None,'hsl_library_sha256':sha(hsl)if hsl else None,
            'runtime_environment':runtime_environment,'unset_environment':['LD_PRELOAD'],
            'production_dependencies_verified':not a.local_qualification_only,
            'dependency_receipt':str(a.dependency_receipt.resolve()),'dependency_receipt_sha256':sha(a.dependency_receipt),
            'baseline_binary':str(baseline)if baseline else None,
            'baseline_binary_sha256':sha(baseline)if baseline else None,
            'source_directory':str(source.resolve()),'source_sha256':contract['source_sha256'],
            'linked_libraries':loaded,'hsl_linked_libraries':hsl_loaded,'ldd_output':linked.stdout,
            'aligned_source_sha256':hashlib.sha256(json.dumps(contract['source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'patch_contract_sha256':sha(HERE/'PATCH_CONTRACT.json'),
            'compile_command':command,'compile_seconds':time.monotonic()-started,
            'original_installation_modified':False,'original_installation_unchanged':True,
            'effective_runtime_proof':'Set/Get-verified effective_options in every endpoint',
            'aligned_contract':contract['aligned_contract'],
            'backends':['ma27','mumps'],'strategies':['adaptive','monotone'],
            'version_output':probe.stdout.strip()}
    if sha(library)!=library_sha_before or (hsl and sha(hsl)!=hsl_sha_before):raise RuntimeError('Dependency library changed during isolated build')
    for name,value in header_hashes.items():
        if sha(headers/name)!=value:raise RuntimeError('Ipopt header changed during isolated build: '+name)
    for name,value in contract['source_sha256'].items():
        if sha(source/name)!=value:raise RuntimeError('Adapter source changed during isolated build: '+name)
    for values in (loaded,hsl_loaded):
        for name,item in values.items():
            if sha(item['path'])!=item['sha256']:raise RuntimeError('Linked library changed during build: '+name)
    report['ipopt_headers_sha256']=header_hashes
    report['dependencies_rehashed_after_build']=True
    report['library_version_proof']='Ipopt headers 3.14.11 and full dynamic-library SHA256 bound to audited dependency receipt; local builds are inadmissible for production'
    if baseline and sha(baseline)!=baseline_sha:raise RuntimeError('Existing comparator changed during isolated build')
    if installation_sha and sha(a.root/'INSTALLATION.json')!=installation_sha:raise RuntimeError('Existing INSTALLATION changed during isolated build')
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'binary':str(output),'ipopt_version':version,'binary_sha256':report['binary_sha256'],
                      'ipopt_library_sha256':report['ipopt_library_sha256']},indent=2))

if __name__=='__main__':main()
