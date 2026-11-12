#!/usr/bin/env python3
"""Collect experiment metadata without starting solvers or altering sources."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import zipfile


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def collect(runs,output,include_endpoints=False):
    runs={name:Path(path).expanduser().resolve(strict=True)for name,path in runs.items()}
    if not runs:raise ValueError('Au moins un dossier de résultats est requis.')
    if any(not name or '/'in name or name in ('.','..')for name in runs):raise ValueError('Nom de collecte invalide.')
    output=Path(output).expanduser().resolve()
    if output.exists():raise ValueError('Archive existante ; choisir un nouveau nom.')
    if not output.parent.is_dir():raise ValueError('Dossier de sortie absent.')
    if any(output.is_relative_to(p)for p in runs.values()):raise ValueError('Écrire l’archive hors des résultats sources.')
    manifest={};states={};excluded=[]
    forbidden={'__pycache__','research_source','reference_source','source_evidence','inputs','native'}
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True)as z:
        for name,root in runs.items():
            if not root.is_dir():raise ValueError('Un dossier de résultats est requis : '+str(root))
            states[name]={}
            for statefile in [root/'ETUDE_STATUS.json',root/'STATUS.json']:
                if statefile.is_file():
                    try:states[name][statefile.name]=json.loads(statefile.read_text())
                    except ValueError:states[name][statefile.name]={'status_parse_error':True}
            count=0
            for p in sorted(root.rglob('*')):
                if not p.is_file()or p.is_symlink():continue
                rel=p.relative_to(root)
                if any(part in forbidden for part in rel.parts):continue
                member=name+'/'+str(rel)
                if p.name=='endpoint.json'and not include_endpoints:
                    excluded.append(dict(path=member,reason='full_endpoint_vector_omitted',bytes=p.stat().st_size));continue
                if p.suffix in {'.json','.jsonl','.csv','.gz','.txt','.tex','.sha256'}:
                    before=p.stat();digest=sha(p);z.write(p,member)
                    after=p.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                        raise ValueError('Fichier modifié pendant la collecte : '+str(p))
                    manifest[member]=dict(bytes=before.st_size,sha256=digest,complete=True)
                elif p.suffix=='.log':
                    size=p.stat().st_size
                    with p.open('rb')as stream:
                        stream.seek(max(0,size-131072));data=stream.read()
                    z.writestr(member+'.tail',data)
                    manifest[member+'.tail']=dict(original_bytes=size,excerpt_sha256=hashlib.sha256(data).hexdigest(),complete=size==len(data),tail_bytes=len(data))
                else:continue
                count+=1
                if count%2000==0:print(f'{name} : {count} pièces collectées',flush=True)
        receipt=dict(schema='bounds-and-quiet-timing-export-v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     source_directories={k:str(v)for k,v in runs.items()},states=states,files=manifest,
                     excluded_endpoint_vectors=excluded,optimizer_invocations=0,source_files_modified=False,
                     scope='Archived scalar records, original audits, contracts and monitor evidence; not a fresh KKT audit.')
        z.writestr('EXPORT.json',json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print('ARCHIVE À TRANSMETTRE : '+str(output),flush=True)
    print('TAILLE : %.2f Mio'%(output.stat().st_size/1048576),flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,metavar='NOM=DOSSIER')
    p.add_argument('--out',type=Path)
    p.add_argument('--include-endpoints',action='store_true')
    a=p.parse_args();runs={}
    for item in a.run:
        name,sep,path=item.partition('=')
        if not sep or name in runs:raise ValueError('Chaque collecte doit avoir un nom unique : NOM=DOSSIER')
        runs[name]=path
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=a.out or Path.home()/'Téléchargements'/('BORNES_CHRONOS_RESULTATS_'+stamp+'.zip')
    collect(runs,out,a.include_endpoints)


if __name__=='__main__':main()
