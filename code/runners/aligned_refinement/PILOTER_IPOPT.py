#!/usr/bin/env python3
"""Independent four-configuration Ipopt experiment on the 4,078 frozen QCQPs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,threading,time,uuid,zipfile

KIT=Path(__file__).resolve().parent
LAST=KIT/'LAST_IPOPT_RUN.json'
CPUS=set(range(22))
TOTAL=16312
MUTABLE={'LAST_IPOPT_RUN.json','LAST_RUN.json','ipopt_launcher.lock','launcher.lock'}

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with tmp.open('w')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default

def proc(pid):
 try:
  fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=int(pid),state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),session=int(fields[3]),pid_start_ticks=int(fields[19]))
 except (OSError,ValueError,IndexError):return None

def ticks(pid):
 p=proc(pid);return p['pid_start_ticks']if p else None

def live(pid,start):
 p=proc(pid)if type(pid)is int else None
 return p is not None and type(start)is int and p['pid_start_ticks']==start and p['state']!='Z'

def verify_kit():
 manifest=read(KIT/'KIT_SHA256.json')
 if not isinstance(manifest,dict)or not manifest:raise ValueError('Manifeste du kit absent')
 for name,h in manifest.items():
  rel=Path(name)
  if name in MUTABLE or rel.name.endswith('.lock')or '__pycache__'in rel.parts:continue
  if rel.is_absolute()or '..'in rel.parts:raise ValueError('Chemin de manifeste invalide : '+name)
  p=KIT/rel
  if not p.is_file()or sha(p)!=h:raise ValueError('Fichier du kit modifié ou absent : '+name)
 return sha(KIT/'KIT_SHA256.json')

def roots():
 source=read(KIT/'evidence/DEFAULT_INSTALLATION_SOURCE.json')
 plan=read(KIT/'evidence/CURRENT_PLAN_COMPACT.json')['plan_header']
 return Path(source['root']),Path(plan['output_directory'])

def pipeline_lock(path=None):
 path=Path(path)if path is not None else Path('/tmp')/('lazyqcqp_ipopt_'+str(os.getuid())+'.lock')
 f=path.open('a+')
 try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:f.close();raise ValueError('Un dispositif Ipopt réserve déjà les cœurs 0 à 21')
 return f

class OwnedProcesses:
 """Track only observed descendants with immutable Linux PID start times."""
 def __init__(self):self.records={}
 def track(self,pid):
  p=proc(pid)
  if p:self.records[pid]=p
 def discover(self):
  known={pid for pid,p in self.records.items()if live(pid,p['pid_start_ticks'])}
  candidates=[]
  for path in Path('/proc').iterdir():
   if path.name.isdigit():
    p=proc(int(path.name))
    if p and p['state']!='Z':candidates.append(p)
  changed=True
  while changed:
   changed=False
   for p in candidates:
    if p['pid']not in known and p['ppid']in known:
     self.records[p['pid']]=p;known.add(p['pid']);changed=True
 def remaining(self):
  return [p for p in self.records.values()if live(p['pid'],p['pid_start_ticks'])]
 def signal(self,sig):
  seen=set()
  for record in self.remaining():
   p=proc(record['pid'])
   if not p or p['pid_start_ticks']!=record['pid_start_ticks']:continue
   pg=p['pgid']
   try:
    if pg>1 and pg!=os.getpgrp()and pg not in seen:os.killpg(pg,sig);seen.add(pg)
    elif pg==os.getpgrp():os.kill(p['pid'],sig)
   except ProcessLookupError:pass

class Pipeline:
 def __init__(self,out,stop=None,heartbeat=10,terminate_deadline=30):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.stop=stop or threading.Event();self.heartbeat=heartbeat;self.terminate_deadline=terminate_deadline
  self.owned=OwnedProcesses();self.state=dict(state='preparing',phase='initialisation',timestamp=utc(),started_utc=utc(),pid=os.getpid(),pid_start_ticks=ticks(os.getpid()),cpu_affinity=sorted(CPUS),jobs=22,out=str(self.out),active_child=None,owned_processes=[])
 def save(self,**values):
  self.state.update(values,timestamp=utc(),owned_processes=self.owned.remaining());atomic(self.out/'IPOPT_PIPELINE_STATUS.json',self.state)
 def phase(self,name,command):
  if self.stop.is_set():raise InterruptedError('Arrêt demandé')
  print(name+' — démarrage',flush=True);started=time.monotonic();child=subprocess.Popen(command,start_new_session=True);self.owned.track(child.pid)
  self.save(state='running',phase=name,active_child=dict(pid=child.pid,pid_start_ticks=ticks(child.pid),command=command))
  last=started;stopping=None
  while child.poll()is None:
   self.owned.discover();now=time.monotonic()
   if self.stop.is_set():
    if stopping is None:stopping=now;self.owned.signal(signal.SIGTERM);print('ARRÊT — fermeture des seuls processus possédés',flush=True)
    elif now-stopping>=self.terminate_deadline:self.owned.signal(signal.SIGKILL)
   if now-last>=self.heartbeat:
    print(name+' — en cours, '+str(int(now-started))+' s',flush=True);self.save(phase_elapsed_seconds=now-started);last=now
   time.sleep(.1 if self.stop.is_set()else min(.5,self.heartbeat))
  code=child.wait();self.save(active_child=None,phase_elapsed_seconds=time.monotonic()-started)
  if self.stop.is_set():raise InterruptedError('Arrêt demandé')
  if code:raise RuntimeError(name+' : code '+str(code)+' ; consulter les lignes précédentes du journal')
  if self.owned.remaining():raise RuntimeError('La phase a laissé des processus descendants actifs')
  print(name+' — terminé',flush=True)
 def cleanup(self):
  self.owned.discover();remaining=self.owned.remaining()
  if not remaining:return
  self.owned.signal(signal.SIGTERM);began=time.monotonic()
  while self.owned.remaining()and time.monotonic()-began<self.terminate_deadline:
   self.owned.discover();time.sleep(.1)
  if self.owned.remaining():self.owned.signal(signal.SIGKILL)
  until=time.monotonic()+3
  while self.owned.remaining()and time.monotonic()<until:time.sleep(.1)
  self.save(owned_processes_remaining=len(self.owned.remaining()))

def bound_installation(out):
 out=Path(out);state=read(out/'IPOPT_PIPELINE_STATUS.json',{})
 paths=[out/'build/BUILD.json',out/'qualification/QUALIFICATION.json',out/'preparation/PREPARATION.json',out/'preparation/INSTALLATION_IPOPT_ALIGNE.json']
 bindings=state.get('qualified_bindings',{})
 if len(bindings)!=4:raise ValueError('Aucune installation Ipopt intégralement qualifiée à reprendre')
 for p in paths:
  if not p.is_file()or bindings.get(str(p))!=sha(p):raise ValueError('Installation qualifiée absente ou modifiée : '+str(p))
 if read(paths[1],{}).get('passed')is not True:raise ValueError('Les quatre contrôles Ipopt ne sont pas qualifiés')
 return paths[-1]

def launch(a,resume=False):
 with (KIT/'ipopt_launcher.lock').open('a')as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  item=read(LAST,{})
  if live(item.get('pid'),item.get('pid_start_ticks')):
   print('DISPOSITIF IPOPT DÉJÀ ACTIF — PID',item['pid'],flush=True);print('JOURNAL :',item['log'],flush=True);return
  manifest=verify_kit()
  if resume:
   if not item:raise ValueError('Aucun lancement Ipopt enregistré')
   out=Path(item['out']);root=Path(item['root']);run=Path(item['run']);bound_installation(out)
  else:
   if item and not a.nouveau and a.out is None:
    print('Un lancement Ipopt est déjà enregistré : '+item['out'],flush=True);print('Utiliser REPRENDRE_IPOPT_ALIGNE.sh ou LANCER_IPOPT_ALIGNE.sh --nouveau.',flush=True);return
   dr,ds=roots();root=(a.root or dr).expanduser().resolve();run=(a.run or ds).expanduser().resolve()
   out=(a.out or Path('/extra/LAZYQCQP_IPOPT_ALIGNE_RESULTS')/('IPOPT_4CONFIG_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8])).expanduser().resolve()
   if out.exists():raise ValueError('Le dossier de sortie doit être neuf : '+str(out))
   for value in(KIT,root,run):
    value=value.resolve()
    if out==value or value in out.parents or out in value.parents:raise ValueError('La sortie doit être isolée du kit et des campagnes précédentes')
   out.mkdir(parents=True)
  command=[sys.executable,'-u',str(Path(__file__).resolve()),'executer','--root',str(root),'--run',str(run),'--out',str(out)]
  if resume:command.append('--reprise-production')
  for name in('ipopt_library','ipopt_include','ipopt_prefix','hsl_library'):
   if getattr(a,name,None)is not None:command+=['--'+name.replace('_','-'),str(getattr(a,name))]
  log=out/'IPOPT_ALIGNE.log'
  with log.open('ab',buffering=0)as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  item=dict(schema='ipopt-aligned-launcher-v1',out=str(out),root=str(root),run=str(run),log=str(log),pid=child.pid,pid_start_ticks=ticks(child.pid),started_utc=utc(),kit_sha256=manifest,new_production_endpoints=True)
  atomic(LAST,item);atomic(out/'LAUNCHER_IPOPT.json',item)
  print('IPOPT — PRÉPARATION EN ARRIÈRE-PLAN : PID '+str(child.pid),flush=True);print('JOURNAL : '+str(log),flush=True);print('Suivi : bash "'+str(KIT/'SUIVRE_IPOPT_ALIGNE.sh')+'"',flush=True)

def analyze(pipeline):
 out=pipeline.out;campaign=out/'campaign';analysis=out/'DEPOUILLEMENT'
 if analysis.exists():analysis=out/('DEPOUILLEMENT_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8])
 pipeline.save(analysis=str(analysis));command=[sys.executable,'-u',str(KIT/'ipopt_aligned/ANALYSER_IPOPT_ALIGNE.py'),'--plan',str(campaign/'PLAN.json'),'--observations',str(campaign/'OBSERVATIONS.jsonl'),'--out',str(analysis)]
 reference=KIT/'evidence/COMPACT_14_CONFIGS.jsonl.gz'
 if reference.is_file():command+=['--lazy-reference',str(reference)]
 pipeline.phase('DÉPOUILLEMENT DES QUATRE CONFIGURATIONS',command)
 return export(pipeline)

def export(pipeline):
 out=pipeline.out;destination=out/('IPOPT_ALIGNE_A_TRANSMETTRE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8]+'.zip')
 pipeline.phase('EXPORT COMPLET DES OBSERVATIONS ET PREUVES', [sys.executable,'-u',str(KIT/'EXPORTER_RESULTATS.py'),'--kind','ipopt','--run',str(out),'--out',str(destination)])
 if not destination.is_file():raise ValueError('Archive de dépouillement absente')
 with zipfile.ZipFile(destination)as archive:
  receipt=json.loads(archive.read('DEPOUILLEMENT/EXPORT.json'))
 if receipt.get('complete')is not True or receipt.get('done')!=TOTAL or receipt.get('total')!=TOTAL:raise ValueError('Export incomplet des 16 312 observations contemporaines')
 atomic(out/'EXPORT_IPOPT.json',dict(archive=str(destination),archive_sha256=sha(destination),created_utc=utc(),source='Newly computed four-configuration Ipopt campaign; no prior Ipopt endpoint imported'))
 print('ARCHIVE À TRANSMETTRE : '+str(destination),flush=True);return destination

def execute(a):
 pipeline=Pipeline(a.out);stop=pipeline.stop
 for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
 lock=None
 try:
  lock=pipeline_lock();verify_kit()
  if a.reprise_production:
   bound_installation(pipeline.out);pipeline.state['qualified_bindings']=read(pipeline.out/'IPOPT_PIPELINE_STATUS.json')['qualified_bindings']
  if not CPUS<=set(os.sched_getaffinity(0)):raise ValueError('Les cœurs 0 à 21 ne sont pas disponibles')
  os.sched_setaffinity(0,CPUS);pipeline.save()
  print('IPOPT — 4 078 QCQP × MA27/MUMPS × monotone/adaptive = 16 312 nouveaux endpoints',flush=True)
  print('EXÉCUTION — 22 créneaux continus sur les cœurs 0 à 21 ; ablation indépendante sur le cœur 23',flush=True)
  py=[sys.executable,'-u'];out=pipeline.out;build=out/'build/BUILD.json';qualification=out/'qualification/QUALIFICATION.json';preparation=out/'preparation/PREPARATION.json';installation=out/'preparation/INSTALLATION_IPOPT_ALIGNE.json'
  if a.reprise_production:
   bound_installation(out)
  else:
   command=py+[str(KIT/'ipopt_aligned/BUILD_IPOPT_ALIGNED.py'),'--root',str(a.root),'--out',str(out/'build')]
   for name in('ipopt_library','ipopt_include','ipopt_prefix','hsl_library'):
    if getattr(a,name,None)is not None:command+=['--'+name.replace('_','-'),str(getattr(a,name))]
   pipeline.phase('COMPILATION DU COMPARATEUR IPOPT ISOLÉ',command)
   pipeline.phase('QUALIFICATION NATIVE DES QUATRE CONFIGURATIONS',py+[str(KIT/'ipopt_aligned/QUALIFIER_IPOPT_ALIGNE.py'),'--build',str(build),'--out',str(out/'qualification')])
   if read(qualification,{}).get('passed')is not True:raise ValueError('Les quatre contrôles réels Ipopt n’ont pas tous réussi')
   pipeline.phase('VÉRIFICATION DES 4 078 ENTRÉES ET DÉPARTS',py+[str(KIT/'ipopt_aligned/PREPARER_IPOPT_ALIGNE.py'),'--root',str(a.root),'--run',str(a.run),'--build',str(build),'--qualification',str(qualification),'--out',str(out/'preparation')])
   bindings={str(p):sha(p)for p in(build,qualification,preparation,installation)};pipeline.save(qualified_bindings=bindings)
  action='reprendre'if a.reprise_production and(out/'campaign/RUN_CONTRACT.json').exists()else'lancer'
  pipeline.phase('RÉSOLUTIONS IPOPT — QUATRE CONFIGURATIONS',py+[str(KIT/'ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py'),action,'--installation',str(installation),'--out',str(out/'campaign'),'--interval','10'])
  final=read(out/'campaign/STATUS.json',{})
  if final.get('state')!='completed'or final.get('done')!=TOTAL:
   if final.get('state')=='stopped':raise InterruptedError('Campagne arrêtée ; la reprise conserve uniquement ses propres résultats')
   raise ValueError('Les 16 312 tâches ne sont pas toutes finalisées ; le dépouillement final est bloqué')
  archive=analyze(pipeline);pipeline.save(state='completed',phase='terminée',finished_utc=utc(),archive=str(archive))
  print('IPOPT TERMINÉ — '+str(out/'DEPOUILLEMENT'),flush=True);return 0
 except InterruptedError as e:pipeline.save(state='stopped',error=str(e),finished_utc=utc());print('IPOPT ARRÊTÉ — '+str(e),flush=True);return 130
 except BaseException as e:pipeline.save(state='failed',error=type(e).__name__+': '+str(e),finished_utc=utc());print('IPOPT BLOQUÉ — '+pipeline.state['error'],flush=True);return 2
 finally:
  pipeline.cleanup()
  if lock:lock.close()

def follow(a):
 item=read(LAST)
 if not item:raise ValueError('Aucun lancement Ipopt enregistré')
 out=Path(item['out']);module=None
 while True:
  state=read(out/'IPOPT_PIPELINE_STATUS.json',{});alive=live(item.get('pid'),item.get('pid_start_ticks'))
  if state.get('phase')=='RÉSOLUTIONS IPOPT — QUATRE CONFIGURATIONS'and(out/'campaign/STATUS.json').is_file():
   subprocess.run([sys.executable,str(KIT/'ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py'),'suivre','--out',str(out/'campaign'),'--once'],check=True)
  else:print('['+utc()+'] '+state.get('phase','Démarrage du pilote Ipopt')+' — '+state.get('state','en attente')+(' ; processus présent'if alive else' ; processus terminé'),flush=True)
  if state.get('state')=='failed'or(not alive and state.get('state')not in('completed','stopped')):
   print('MOTIF : '+state.get('error','Le pilote ne tourne plus ; consulter le journal'),flush=True)
   log=Path(item['log']);lines=log.read_text(errors='replace').splitlines()if log.is_file()else[];print('\n'.join(lines[-24:]),flush=True)
  if a.once or not alive or state.get('state')in('completed','stopped','failed'):
   print('JOURNAL : '+item['log'],flush=True)
   if state.get('archive'):print('ARCHIVE : '+state['archive'],flush=True)
   return
  time.sleep(max(1,a.interval))

def stop(a):
 item=read(LAST,{})
 if live(item.get('pid'),item.get('pid_start_ticks')):
  os.kill(item['pid'],signal.SIGTERM);print('ARRÊT IPOPT demandé au pilote identifié ; attente de fermeture des processus possédés.',flush=True)
  deadline=time.monotonic()+40
  while live(item['pid'],item['pid_start_ticks'])and time.monotonic()<deadline:time.sleep(.5)
  if live(item['pid'],item['pid_start_ticks']):raise RuntimeError('Le pilote reste actif ; consulter le journal avant toute relance')
 else:
  out=Path(item.get('out','/nonexistent'))
  if(out/'campaign/CONTROLLER.json').is_file():subprocess.run([sys.executable,str(KIT/'ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py'),'arret','--out',str(out/'campaign')],check=True)
  else:print('Aucun pilote Ipopt identifié actif.',flush=True)

def depouiller(a):
 item=read(LAST)
 if not item:raise ValueError('Aucun lancement Ipopt enregistré')
 out=Path(item['out']);bound_installation(out);status=read(out/'campaign/STATUS.json',{})
 if status.get('state')!='completed'or status.get('done')!=TOTAL:raise ValueError('La campagne des 16 312 tâches doit être terminée avant dépouillement final')
 # Analysis is read-only for solver endpoints and requires its own fresh output.
 if live(item.get('pid'),item.get('pid_start_ticks')):raise ValueError('Le pilote Ipopt est encore actif ; son dépouillement est automatique')
 prior=read(out/'IPOPT_PIPELINE_STATUS.json',{});pipeline=Pipeline(out);pipeline.state.update(prior)
 archive=analyze(pipeline);pipeline.save(state='completed',phase='terminée',archive=str(archive),finished_utc=utc())

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
 for name in('demarrer','reprendre','executer'):
  p=sub.add_parser(name);p.add_argument('--out',type=Path);p.add_argument('--root',type=Path);p.add_argument('--run',type=Path);p.add_argument('--nouveau',action='store_true');p.add_argument('--reprise-production',action='store_true')
  for flag in('ipopt-library','ipopt-include','ipopt-prefix','hsl-library'):p.add_argument('--'+flag,type=Path)
 p=sub.add_parser('suivre');p.add_argument('--interval',type=float,default=10);p.add_argument('--once',action='store_true');sub.add_parser('arret');sub.add_parser('depouiller')
 a=parser.parse_args(argv)
 if a.action=='demarrer':launch(a)
 elif a.action=='reprendre':launch(a,True)
 elif a.action=='executer':
  if not a.out or not a.root or not a.run:raise ValueError('--out/--root/--run sont requis pour le pilote interne')
  return execute(a)
 elif a.action=='suivre':follow(a)
 elif a.action=='depouiller':depouiller(a)
 else:stop(a)
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except(OSError,ValueError,RuntimeError,subprocess.CalledProcessError)as e:print('IPOPT REFUSÉ — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
