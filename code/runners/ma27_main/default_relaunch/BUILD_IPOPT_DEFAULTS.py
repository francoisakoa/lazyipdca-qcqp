#!/usr/bin/env python3
"""Build an isolated default-only Ipopt comparator against the campaign library."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,time

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
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
    p.add_argument('--cxx',default=os.environ.get('CXX','g++'))
    a=p.parse_args();a.root=a.root.resolve();out=a.out.resolve()
    if out.exists():p.error('Output must be fresh; no existing installation or binary is overwritten')
    library,headers,version,abi,baseline=locate(a)
    out.mkdir(parents=True)
    compat=out/'compat/coin-or';compat.mkdir(parents=True)
    for h in headers.glob('*'):
        if h.is_file():(compat/h.name).symlink_to(h)
    source=HERE/'adapter' if (HERE/'adapter').is_dir() else HERE/'source'
    contract=json.loads((HERE/'PATCH_CONTRACT.json').read_text())
    for name,value in contract['patched_source_sha256'].items():
        if sha(source/name)!=value:raise RuntimeError('Distributed default source differs: '+name)
    output=out/'ipopt_defaults_cli'
    command=[a.cxx,'-O2','-std=c++17','-DHAVE_CSTDDEF=1',
             '-D_GLIBCXX_USE_CXX11_ABI='+abi,'-DLAZYIPDCA_HAVE_IPOPT=1',
             '-I'+str(source/'include'),'-I'+str(source/'src'),'-I'+str(out/'compat'),
             *[str(source/'src'/n)for n in ('direct_ipopt.cpp','ipopt_qcqp_main.cpp','io.cpp','io_compact.cpp')],
             str(library),'-Wl,-rpath,'+str(library.parent),'-o',str(output)]
    started=time.monotonic()
    with(out/'compile.log').open('w')as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Default comparator compilation failed; see '+str(out/'compile.log'))
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(library.parent)+(':'+env['LD_LIBRARY_PATH']if env.get('LD_LIBRARY_PATH')else'')
    probe=subprocess.run([str(output),'--version'],env=env,capture_output=True,text=True)
    if probe.returncode or version not in probe.stdout:
        raise RuntimeError('Built comparator version check failed: '+probe.stdout+probe.stderr)
    report={'schema':'isolated-ipopt-library-defaults-build-v1','binary':str(output),
            'binary_sha256':sha(output),'ipopt_library':str(library),'ipopt_library_sha256':sha(library),
            'ipopt_headers':str(headers),'ipopt_version':version,'string_ABI':int(abi),
            'baseline_binary':str(baseline)if baseline else None,
            'baseline_binary_sha256':sha(baseline)if baseline else None,
            'source_sha256':contract['patched_source_sha256'],
            'default_source_sha256':hashlib.sha256(json.dumps(contract['patched_source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'patch_contract_sha256':sha(HERE/'PATCH_CONTRACT.json'),
            'compile_command':command,'compile_seconds':time.monotonic()-started,
            'original_installation_modified':False,'numerical_default_runtime_proof':'effective_options in every endpoint',
            'version_output':probe.stdout.strip()}
    (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'binary':str(output),'ipopt_version':version,'binary_sha256':report['binary_sha256'],
                      'ipopt_library_sha256':report['ipopt_library_sha256']},indent=2))

if __name__=='__main__':main()
