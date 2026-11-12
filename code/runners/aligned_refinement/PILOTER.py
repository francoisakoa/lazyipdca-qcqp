#!/usr/bin/env python3
"""Build, qualify, seal and launch the controlled native refinement experiment."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,threading,time,uuid
KIT=Path(__file__).resolve().parent;LAST=KIT/'LAST_RUN.json';CPUS={23}

def utc():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default

def ticks(pid):
 try:
  parts=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
  return int(parts[19])if parts[0]!='Z'else None
 except (OSError,ValueError,IndexError):return None

def live(pid,start):return type(pid)is int and type(start)is int and ticks(pid)==start

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def roots():
 plan=read(KIT/'evidence/CURRENT_PLAN_COMPACT.json')['plan_header'];frozen=read(KIT/'evidence/DEFAULT_INSTALLATION_SOURCE.json')
 return Path(frozen['root']),Path(plan['output_directory'])

def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify_kit():
 manifest=read(KIT/'KIT_SHA256.json')
 if not isinstance(manifest,dict)or not manifest:raise ValueError('Manifeste de kit absent')
 for name,h in manifest.items():
  p=KIT/name
  if not p.is_file()or sha(p)!=h:raise ValueError('Fichier du kit modifié ou absent : '+name)
 return sha(KIT/'KIT_SHA256.json')

def demarrer(a,resume=False):
 with (KIT/'launcher.lock').open('a')as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  previous=read(LAST,{})
  if live(previous.get('pid'),previous.get('pid_start_ticks')):
   print('DISPOSITIF DÉJÀ ACTIF — PID',previous['pid']);print('JOURNAL :',previous['log']);return
  if resume:
   if not previous:raise ValueError('Aucune expérience enregistrée à reprendre')
   out=Path(previous['out']);root=Path(previous['root']);run=Path(previous['run'])
   if not(out/'preparation/EXPERIMENT_INSTALLATION.json').is_file():raise ValueError('La préparation n’a pas atteint la qualification. Consulter le journal ; un nouveau dossier est requis après correction.')
  else:
   if previous and not a.nouveau and a.out is None:
    print('Une expérience est déjà enregistrée :',previous['out']);print('Utiliser REPRENDRE.sh pour la reprendre, ou LANCER.sh --nouveau pour un nouveau dossier.');return
   default_root,default_run=roots();root=(a.root or default_root).expanduser().resolve();run=(a.run or default_run).expanduser().resolve()
   out=(a.out or Path('/extra/LAZYQCQP_RAFFINEMENT_RESULTS')/('ABLATION_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')+uuid.uuid4().hex[:8])).expanduser().resolve()
   if out.exists():raise ValueError('Le dossier de sortie doit être neuf : '+str(out))
   for value in(KIT,root,run):
    value=value.resolve()
    if out==value or value in out.parents or out in value.parents:raise ValueError('La sortie doit être séparée du kit et des sources')
   out.mkdir(parents=True)
  verify_kit();log=out/'EXPERIENCE.log';cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'executer','--out',str(out),'--root',str(root),'--run',str(run),'--jobs-build',str(a.jobs_build)]
  if resume:cmd.append('--reprise-production')
  for option in('boost_include','hsl_library','mumps_reference_binary'):
   v=getattr(a,option,None)
   if v is not None:cmd+=['--'+option.replace('_','-'),str(v)]
  env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
  with log.open('ab',buffering=0)as stream:p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
  receipt=dict(out=str(out),root=str(root),run=str(run),log=str(log),pid=p.pid,pid_start_ticks=ticks(p.pid),started_utc=utc(),kit_sha256=sha(KIT/'KIT_SHA256.json'))
  atomic(LAST,receipt);atomic(out/'LAUNCHER.json',receipt)
  print('EXPÉRIENCE PRÉPARÉE EN ARRIÈRE-PLAN — PID',p.pid,flush=True);print('JOURNAL :',log,flush=True);print('Suivi : bash "'+str(KIT/'SUIVRE.sh')+'"',flush=True)

def executer(a):
 pool_lock=(Path('/tmp')/('lazyqcqp_refinement_cpu23_'+str(os.getuid())+'.lock')).open('a')
 try:fcntl.flock(pool_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise ValueError('Une autre expérience de raffinement utilise déjà le cœur23.')
 out=a.out.resolve();state=dict(state='preparing',phase='initialisation',started_utc=utc(),timestamp=utc(),pid=os.getpid(),pid_start_ticks=ticks(os.getpid()),out=str(out),active_child=None)
 stop=threading.Event();child=None
 owned=load('shared_process_ownership',KIT/'PILOTER_IPOPT.py').OwnedProcesses()
 def signal_stop(*_):stop.set()
 for s in(signal.SIGTERM,signal.SIGINT):signal.signal(s,signal_stop)
 def save(**values):state.update(values,timestamp=utc());atomic(out/'PIPELINE_STATUS.json',state)
 def phase(name,command):
  nonlocal child
  if stop.is_set():raise InterruptedError('Arrêt demandé')
  print(name+' — démarrage',flush=True);started=time.monotonic();child=subprocess.Popen(command,start_new_session=True);child_ticks=ticks(child.pid);owned.track(child.pid);save(state='running',phase=name,active_child=dict(pid=child.pid,pid_start_ticks=child_ticks,command=command))
  last=started;stopping=None
  while child.poll()is None:
   owned.discover();now=time.monotonic()
   if stop.is_set():
    if stopping is None:
     stopping=now
     owned.signal(signal.SIGTERM)
     print('ARRÊT — attente de la fermeture des processus possédés',flush=True)
    elif now-stopping>=30:owned.signal(signal.SIGKILL)
   if now-last>=10:print(name+' — en cours, '+str(int(now-started))+' s',flush=True);save(phase_elapsed_seconds=now-started);last=now
   stop.wait(1) if not stop.is_set()else time.sleep(.2)
  code=child.wait();child=None;save(active_child=None,phase_elapsed_seconds=time.monotonic()-started)
  if stop.is_set():raise InterruptedError('Arrêt demandé')
  if code:raise RuntimeError(name+' : code '+str(code)+' ; consulter les messages détaillés précédents du journal')
  if owned.remaining():raise RuntimeError('La phase a laissé des descendants actifs')
  print(name+' — terminé',flush=True)
 try:
  verify_kit()
  if not CPUS<=set(os.sched_getaffinity(0)):raise ValueError('Le cœur23 n’est pas disponible')
  os.sched_setaffinity(0,CPUS)
  print('PROTOCOLE — 1 834 cas I/M × 4 cellules × 2 backends = 14 672 tâches ; cœur23, une tâche à la fois',flush=True)
  print('CORRECTIONS — même initialisation hybride ; gelées ou raffinées ; objectif exact ; garde-fous communs',flush=True)
  print('SOURCE — 823 fichiers de référence liés à la dernière campagne ; aucun résultat antérieur utilisé en production',flush=True)
  py=[sys.executable,'-u'];build=out/'build/BUILD_EXPERIMENT.json';prep=out/'preparation/PREPARATION.json';inst=out/'preparation/EXPERIMENT_INSTALLATION.json';gate=out/'qualification/GATE50.json'
  if not a.reprise_production:
   cmd=py+[str(KIT/'BUILD_EXPERIMENT.py'),'--out',str(out/'build'),'--jobs',str(a.jobs_build)]
   for option in('boost_include','hsl_library','mumps_reference_binary'):
    v=getattr(a,option,None)
    if v is not None:cmd+=['--'+option.replace('_','-'),str(v)]
   phase('COMPILATION NATIVE HOMOGÈNE',cmd)
   phase('VÉRIFICATION DES ENTRÉES ET DÉPARTS',py+[str(KIT/'runtime/PREPARER_EXPERIENCE.py'),'--root',str(a.root),'--run',str(a.run),'--build',str(build),'--out',str(out/'preparation'),'--study-contract',str(KIT/'STUDY_CONTRACT.json'),'--registration',str(KIT/'PRE_ENREGISTREMENT_RAFFINEMENT.json')])
   phase('QUALIFICATION — 50 CAS ET SIX CONTRÔLES',py+[str(KIT/'qualification/QUALIFIER_EXPERIENCE.py'),'--preparation',str(prep),'--build',str(build),'--study-contract',str(KIT/'STUDY_CONTRACT.json'),'--registration',str(KIT/'PRE_ENREGISTREMENT_RAFFINEMENT.json'),'--out',str(out/'qualification'),'--jobs','1','--cpus','23'])
   if read(gate,{}).get('passed')is not True:raise ValueError('La qualification native n’est pas complète et positive')
   phase('SCELLEMENT ET ADMISSION',py+[str(KIT/'runtime/PREPARER_EXPERIENCE.py'),'--promouvoir',str(prep),'--qualification-gate',str(gate),'--study-contract',str(KIT/'STUDY_CONTRACT.json')])
  save(state='running',phase='RÉSOLUTIONS FACTORIELLES',qualified_installation=str(inst),campaign=str(out/'campaign'))
  phase('RÉSOLUTIONS FACTORIELLES',py+[str(KIT/'runtime/CONTROLEUR_EXPERIENCE.py'),'reprendre'if a.reprise_production and(out/'campaign/RUN_CONTRACT.json').exists()else'lancer','--installation',str(inst),'--out',str(out/'campaign'),'--interval','10'])
  final=read(out/'campaign/STATUS.json',{})
  if final.get('state')!='completed'or final.get('done')!=14672:
   if final.get('state')=='stopped':raise InterruptedError('Résolutions arrêtées ; résultats conservés')
   raise ValueError('Les 14 672 tâches ne sont pas toutes finalisées ; aucun bilan confirmatoire final')
  registration=read(inst)['registration']
  phase('DÉPOUILLEMENT PRÉENREGISTRÉ',py+[str(KIT/'analysis/ANALYSER_RAFFINEMENT.py'),'--preregistration',registration,'--observations',str(out/'campaign/RESULTS.jsonl'),'--out',str(out/'analysis')])
  save(state='completed',phase='terminée',finished_utc=utc());print('EXPÉRIENCE TERMINÉE — '+str(out/'analysis'),flush=True)
 except InterruptedError as e:save(state='stopped',error=str(e),finished_utc=utc());print('EXPÉRIENCE ARRÊTÉE — résultats et preuves conservés',flush=True);return 130
 except BaseException as e:
  save(state='failed',error=type(e).__name__+': '+str(e),finished_utc=utc());print('EXPÉRIENCE BLOQUÉE — '+state['error'],flush=True);return 2
 finally:
  owned.discover()
  if owned.remaining():
   owned.signal(signal.SIGTERM);deadline=time.monotonic()+30
   while owned.remaining()and time.monotonic()<deadline:time.sleep(.1)
   if owned.remaining():
    owned.signal(signal.SIGKILL);deadline=time.monotonic()+3
    while owned.remaining()and time.monotonic()<deadline:time.sleep(.1)
   save(owned_processes_remaining=len(owned.remaining()))
  pool_lock.close()
 return 0

def suivre(a):
 item=read(LAST)
 if not item:raise ValueError('Aucun lancement enregistré')
 out=Path(item['out']);previous_error=None
 while True:
  s=read(out/'PIPELINE_STATUS.json',{});alive=live(item.get('pid'),item.get('pid_start_ticks'))
  if s.get('phase')=='RÉSOLUTIONS FACTORIELLES'and(out/'campaign/STATUS.json').is_file():
   sys.path.insert(0,str(KIT/'runtime'));m=load('controlled_follow',KIT/'runtime/CONTROLEUR_EXPERIENCE.py');snapshot=m.base.observed_status(read(out/'campaign/STATUS.json'));m.show(snapshot)
  elif 'QUALIFICATION'in s.get('phase','')and(out/'qualification/STATUS.json').is_file():
   q=read(out/'qualification/STATUS.json');print('QUALIFICATION — '+str(q.get('state'))+' ; résolutions '+str(q.get('native_completed',0))+'/748 ; créneaux '+str(len(q.get('active',{})))+'/1',flush=True)
  else:print('['+utc()+'] '+s.get('phase','démarrage')+' — '+s.get('state','en attente')+(' ; processus présent'if alive else' ; processus terminé'),flush=True)
  if s.get('state')=='failed'or(not alive and s.get('state')not in('completed','stopped')):
   error=s.get('error','État ancien : le pilote ne tourne plus ; contrôler le journal')
   if error!=previous_error:
    print('MOTIF : '+error,flush=True);log=Path(item['log']);lines=log.read_text(errors='replace').splitlines()if log.exists()else[];print('\n'.join(lines[-20:]),flush=True);previous_error=error
  if a.once or not alive or s.get('state')in('failed','completed','stopped'):print('JOURNAL : '+item['log'],flush=True);return
  time.sleep(max(1,a.interval))

def arreter(a):
 item=read(LAST,{})
 if live(item.get('pid'),item.get('pid_start_ticks')):os.kill(item['pid'],signal.SIGTERM);print('Arrêt demandé au pilote identifié ; aucune suppression de résultats.',flush=True)
 else:
  out=Path(item.get('out','/nonexistent'))
  if(out/'campaign/CONTROLLER.json').is_file():subprocess.run([sys.executable,str(KIT/'runtime/CONTROLEUR_EXPERIENCE.py'),'arret','--out',str(out/'campaign')],check=True)
  else:print('Aucun pilote identifié actif.',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 for name in('demarrer','reprendre','executer'):
  q=sub.add_parser(name);q.add_argument('--out',type=Path);q.add_argument('--root',type=Path);q.add_argument('--run',type=Path);q.add_argument('--jobs-build',type=int,default=1);q.add_argument('--nouveau',action='store_true');q.add_argument('--reprise-production',action='store_true')
  for flag in('boost-include','hsl-library','mumps-reference-binary'):q.add_argument('--'+flag,type=Path)
 q=sub.add_parser('suivre');q.add_argument('--interval',type=float,default=10);q.add_argument('--once',action='store_true');sub.add_parser('arret')
 a=p.parse_args()
 if getattr(a,'jobs_build',1)!=1:raise ValueError('Le raffinement utilise un seul cœur23, y compris pour la compilation')
 if a.action=='demarrer':demarrer(a)
 elif a.action=='reprendre':demarrer(a,True)
 elif a.action=='executer':
  if not a.out or not a.root or not a.run:raise ValueError('--out/--root/--run requis pour le pilote interne')
  raise SystemExit(executer(a))
 elif a.action=='suivre':suivre(a)
 else:arreter(a)
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,RuntimeError)as e:print('DISPOSITIF REFUSÉ — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
