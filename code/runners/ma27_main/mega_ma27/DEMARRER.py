#!/usr/bin/env python3
"""Prepare and launch one previous-device mega campaign in the background."""
from pathlib import Path
from datetime import datetime,timezone
from contextlib import contextmanager
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,uuid
HERE=Path(__file__).resolve().parent
STATE=HERE/'LAUNCH_STATE.json'
DEFAULT_ROOT=Path('/data/projects/IPDCA_QCQP/LAZYQCQP_MEGA_STABILISEE_20261107_R2')
NATIVE_METHODS=('R3','HH','HS','SH','SS','SS_legacy')

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default

def identity(pid):
    try:
        fields=Path('/proc/'+str(int(pid))+'/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z'else fields[19]
    except (OSError,ValueError,IndexError,TypeError):return None

def owned_live(pid,started):return started is not None and identity(pid)==str(started)

def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('w')as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

@contextmanager
def lock(name,nonblocking=False):
    with (HERE/name).open('a')as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError:raise ValueError('Une préparation de ce dispositif est déjà active')
        yield

def state_write(value):
    with lock('state.lock'):atomic(STATE,value)

def log_tail(path,start=0,lines=12):
    if path is None:return []
    try:
        with Path(path).open('rb')as stream:
            stream.seek(max(start,Path(path).stat().st_size-16384))
            return [line for line in stream.read().decode(errors='replace').splitlines()if line.strip()][-lines:]
    except OSError:return []

class PhaseFailure(RuntimeError):
    def __init__(self,phase,code,log,tail):
        self.phase=phase;self.returncode=code;self.log=log;self.tail=tail
        markers=('REFUSÉ','REFUSE','BLOQUÉ','Error:','Exception:','ERROR')
        relevant=[line for line in tail if any(marker in line for marker in markers)]
        self.diagnostic=(relevant[-1]if relevant else(tail[-1]if tail else'Consulter le journal de cette préparation'))[-4000:]
        super().__init__(phase+' : code '+str(code)+' — '+self.diagnostic)

def validate_ready(path,out,oldroot,with_factorial):
    ready=read(path)
    if not isinstance(ready,dict)or ready.get('passed')is not True:raise ValueError('Préparation sans reçu READY validé')
    device=Path(ready.get('root','')).resolve()
    if device!=(out/'device').resolve()or not device.is_dir():raise ValueError('Le reçu READY ne désigne pas le nouveau dispositif isolé')
    if device==oldroot or device in oldroot.parents or oldroot in device.parents:raise ValueError('Le dispositif précédent doit rester distinct de la nouvelle installation')
    expected=39960 if with_factorial else 32624
    if ready.get('tasks')!=expected or ready.get('with_factorial')is not with_factorial:raise ValueError('Périmètre READY différent des '+str(expected)+' tâches demandées')
    if ready.get('native_backend')!='ma27'or tuple(ready.get('native_methods',[]))!=NATIVE_METHODS:raise ValueError('Les six profils Lazy sous MA27 sont requis')
    if set(ready.get('ipopt_backends',[]))!={'ma27','mumps'}or ready.get('ipopt_protocol')!='library-defaults':raise ValueError('Les deux références Ipopt par défaut sont requises')
    if ready.get('source_unchanged')is not True or Path(ready.get('source_root','')).resolve()!=oldroot:raise ValueError('Identité de la source précédente non confirmée')
    python=ready.get('python')
    if not isinstance(python,str)or not Path(python).is_file():raise ValueError('Interpréteur du dispositif préparé absent')
    controller=device/'campaign/MEGA.py'
    if not controller.is_file():raise ValueError('Contrôleur du dispositif méga précédent absent : '+str(controller))
    return ready,device,controller,python

def campaign_receipts(device,run):
    actual=read(run/'CONTROLLER.json',{})or{}
    owner=read(run/'CONTROLLER_OWNER.json',{})or{}
    status=read(run/'STATUS.json',{})or{}
    valid=bool(actual.get('token')and actual.get('token')==owner.get('token')and actual.get('pid')==owner.get('pid'))
    started=status.get('controller_start_ticks') if status.get('controller_pid')==actual.get('pid')else identity(actual.get('pid'))
    return dict(controller_pid=actual.get('pid')if valid else None,controller_start_ticks=started if valid else None,
                controller_token=actual.get('token')if valid else None,controller_live=valid and owned_live(actual.get('pid'),started))

def worker(args):
    child=None;child_started=None;current_phase='initialisation';launch_attempted=False;device=None;controller=None
    log=Path(args.log).resolve()if args.log else None
    oldroot=args.root.resolve();out=args.out.resolve();results=args.results.resolve()
    common=dict(source_root=str(oldroot),preparation=str(out),results=str(results),with_factorial=args.with_factorial,
                preparation_log=str(log)if log else None,launch_token=args.launch_token,
                worker_pid=os.getpid(),worker_start_ticks=identity(os.getpid()))
    def terminate(*_):
        if child is not None and child.poll()is None and owned_live(child.pid,child_started):
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise KeyboardInterrupt('Préparation interrompue')
    old_signals={s:signal.getsignal(s)for s in(signal.SIGINT,signal.SIGTERM)}
    for s in old_signals:signal.signal(s,terminate)
    def run(phase,command):
        nonlocal child,current_phase,child_started
        current_phase=phase
        offset=log.stat().st_size if log and log.is_file()else 0
        state_write(dict(common,state='preparing',phase=phase,timestamp=utc()))
        print(phase,flush=True)
        child=subprocess.Popen(list(map(str,command)),start_new_session=True)
        child_started=identity(child.pid)
        code=child.wait();child=None
        if code:raise PhaseFailure(phase,code,log,log_tail(log,offset))
    try:
        if not oldroot.is_dir():raise ValueError('Installation précédente absente : '+str(oldroot))
        if out.exists()or results.exists():raise ValueError('Préparation et résultats doivent utiliser deux nouveaux répertoires')
        if out==oldroot or oldroot in out.parents or results==oldroot or oldroot in results.parents:raise ValueError('Les nouveaux répertoires doivent rester hors de l’installation précédente')
        out.parent.mkdir(parents=True,exist_ok=True);results.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(HERE/'PREPARER_MEGA.py'),'--root',str(oldroot),'--out',str(out)]
        if args.with_factorial:command+=['--with-factorial']
        run('PRÉPARATION DE LA MÉGA CAMPAGNE — SIX LAZY MA27 ET DEUX IPOPT PAR DÉFAUT',command)
        ready,device,controller,python=validate_ready(out/'READY.json',out,oldroot,args.with_factorial)
        launch_attempted=True
        run('LANCEMENT DE LA FILE UNIQUE — 22 CŒURS (0 À 21)',
            [python,'-u',str(controller),'--root',str(device),'lancer','--cpus','0-21','--jobs','22','--out',str(results)])
        pointer=device/'LAST_RUN.txt'
        if not pointer.is_file()or Path(pointer.read_text().strip()).resolve()!=results:raise ValueError('Le contrôleur ne confirme pas le nouveau run demandé')
        if not(results/'PLAN.json').is_file():raise ValueError('Le nouveau plan méga est absent')
        (HERE/'LAST_DEVICE.txt').write_text(str(device)+'\n');(HERE/'LAST_RUN.txt').write_text(str(results)+'\n')
        (HERE/'LAST_CONTROLLER.txt').write_text(str(controller)+'\n')
        state_write(dict(common,state='launched',phase='campaign',device=str(device),run=str(results),
                         controller_script=str(controller),python=python,tasks=ready['tasks'],ready=str(out/'READY.json'),
                         ready_sha256=sha(out/'READY.json'),timestamp=utc(),**campaign_receipts(device,results)))
        print('MÉGA LANCÉE — '+str(ready['tasks'])+' tâches ; 22 créneaux continus',flush=True)
        print('SORTIE : '+str(results),flush=True)
    except BaseException as error:
        # A launcher error after controller acknowledgement must not pretend
        # that an already-owned campaign is absent. Its monitor remains usable.
        receipt=campaign_receipts(device,results)if device is not None and launch_attempted else{}
        if receipt.get('controller_live')is True and controller is not None:
            (HERE/'LAST_DEVICE.txt').write_text(str(device)+'\n');(HERE/'LAST_RUN.txt').write_text(str(results)+'\n')
            (HERE/'LAST_CONTROLLER.txt').write_text(str(controller)+'\n')
            state_write(dict(common,state='launched',phase='campaign',device=str(device),run=str(results),
                             controller_script=str(controller),python=sys.executable,launch_warning=type(error).__name__+': '+str(error),timestamp=utc(),**receipt))
            print('CONTRÔLEUR IDENTIFIÉ TOUJOURS ACTIF — '+str(error),flush=True);return 2
        state_write(dict(common,state='blocked',phase=current_phase,benchmark_launched=False if not launch_attempted else None,
                         child_returncode=getattr(error,'returncode',None),diagnostic=getattr(error,'diagnostic',str(error)),
                         log_tail=getattr(error,'tail',log_tail(log)),error=type(error).__name__+': '+str(error),timestamp=utc()))
        print('PRÉPARATION MÉGA BLOQUÉE — '+str(error),flush=True);return 2
    finally:
        for s,handler in old_signals.items():signal.signal(s,handler)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--with-factorial',action='store_true')
    p.add_argument('--out',type=Path,help='Nouveau répertoire de préparation')
    p.add_argument('--results',type=Path,help='Nouveau répertoire de résultats')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--log',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--launch-token',help=argparse.SUPPRESS)
    a=p.parse_args();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    a.out=(a.out or Path('/extra/LAZYQCQP_MEGA_MA27_INSTALL')/stamp).resolve()
    a.results=(a.results or Path('/extra/LAZYQCQP_MEGA_MA27_RESULTS')/('CUTEST_BEAM_PGLIB_'+stamp)).resolve()
    if a.worker:
        with lock('preparation.lock',nonblocking=True):return worker(a)
    with lock('launch.lock',nonblocking=True):
        current=read(STATE,{})or{}
        if current.get('state')in('starting','preparing')and owned_live(current.get('worker_pid'),current.get('worker_start_ticks')):
            raise ValueError('Préparation déjà active : PID '+str(current['worker_pid']))
        if current.get('state')=='launched':
            status=read(Path(current.get('run',''))/'STATUS.json',{})or{}
            if status.get('state')in('initializing','running','stopping')and owned_live(status.get('controller_pid'),status.get('controller_start_ticks')):
                raise ValueError('Une méga campagne de ce kit est déjà active : '+str(current['run']))
        if a.out.exists()or a.results.exists():raise ValueError('Nouveaux répertoires de préparation et résultats requis')
        log=HERE/('PREPARATION_'+stamp+'.log');token=uuid.uuid4().hex
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--root',str(a.root.resolve()),
                 '--out',str(a.out),'--results',str(a.results),'--log',str(log),'--launch-token',token]
        if a.with_factorial:command+=['--with-factorial']
        with log.open('x')as stream,lock('state.lock'):
            initial=dict(state='starting',phase='initialisation',timestamp=utc(),preparation_log=str(log),
                         source_root=str(a.root.resolve()),preparation=str(a.out),results=str(a.results),with_factorial=a.with_factorial,launch_token=token)
            atomic(STATE,initial);(HERE/'LAST_PREPARATION_LOG.txt').write_text(str(log)+'\n')
            try:
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                initial.update(worker_pid=proc.pid,worker_start_ticks=identity(proc.pid));atomic(STATE,initial)
            except BaseException as error:
                initial.update(state='blocked',benchmark_launched=False,error=type(error).__name__+': '+str(error));atomic(STATE,initial);raise
        print('PRÉPARATION MÉGA EN ARRIÈRE-PLAN : PID',proc.pid,flush=True)
        print('JOURNAL :',log,flush=True)
        print('Suivi : bash "'+str(HERE.parent/'SUIVRE.sh')+'"',flush=True)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:print('DÉMARRAGE MÉGA REFUSÉ — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
