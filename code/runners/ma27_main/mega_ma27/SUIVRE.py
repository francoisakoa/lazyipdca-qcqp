#!/usr/bin/env python3
"""Watch this preparation, then the previous mega device's compact regime monitor."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,sys,time
import DEMARRER as launch
HERE=Path(__file__).resolve().parent

def selected_log(state):
    value=state.get('preparation_log')
    if not value:return None
    path=Path(value).resolve()
    return path if path.parent==HERE.resolve()and path.name.startswith('PREPARATION_')and path.suffix=='.log'else None

def blocked(state,log):
    print('PRÉPARATION MÉGA BLOQUÉE — '+state.get('error','Consulter le journal'),flush=True)
    if log:print('JOURNAL : '+str(log),flush=True)
    lines=state.get('log_tail')or launch.log_tail(log)
    if lines:
        print('DERNIÈRES LIGNES UTILES :',flush=True)
        for line in lines[-12:]:print(line[-4000:],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=float,default=10)
    p.add_argument('--json',action='store_true');a=p.parse_args()
    if a.interval<=0 or a.interval>60:raise ValueError('Intervalle attendu entre 0 et 60 secondes')
    while True:
        state=launch.read(HERE/'LAUNCH_STATE.json',{})or{}
        if state.get('state')=='launched':
            device=Path(state['device']).resolve();controller=device/'campaign/MEGA.py'
            if Path(state.get('controller_script','')).resolve()!=controller or not controller.is_file():raise ValueError('Contrôleur suivi différent du dispositif lancé')
            if state.get('launch_warning'):print('AVIS DU LANCEUR : '+state['launch_warning'],flush=True)
            run=Path(state['run']).resolve()
            # Delegate only after the controller has written its first status.
            # Otherwise a supervisor dying between its owner receipt and that
            # first status could leave the historical follow loop waiting.
            status=launch.read(run/'STATUS.json')
            if status is None:
                live=launch.owned_live(state.get('controller_pid'),state.get('controller_start_ticks'))
                print('EN ATTENTE DU PREMIER STATUT MÉGA — '+str(run),flush=True)
                print('JOURNAL CONTRÔLEUR : '+str(run/'controller.log'),flush=True)
                if not live:
                    print('CONTRÔLEUR NON CONFIRMÉ — aucun premier statut et aucun processus identifié actif.',flush=True)
                    for line in launch.log_tail(run/'controller.log'):print(line[-4000:],flush=True)
                    return 2
                if a.once:return 0
                time.sleep(a.interval);continue
            cmd=[state.get('python',sys.executable),'-u',str(controller),'--root',str(device),'suivre','--out',state['run'],'--interval',str(a.interval)]
            if a.once:cmd+=['--once']
            if a.json:cmd+=['--json']
            return subprocess.call(cmd)
        log=selected_log(state)
        if state.get('state')=='blocked':blocked(state,log);return 2
        if not state:raise ValueError('Aucune préparation méga démarrée dans ce kit')
        print('PRÉPARATION MÉGA — '+state.get('phase','initialisation'),flush=True)
        if log:
            for line in launch.log_tail(log,lines=3):print(line[-500:],flush=True)
        if state.get('state')in('starting','preparing')and not launch.owned_live(state.get('worker_pid'),state.get('worker_start_ticks')):
            print('PRÉPARATION INTERROMPUE — contrôleur identifié absent ; aucun lancement supposé.',flush=True)
            if log:print('JOURNAL : '+str(log),flush=True)
            return 2
        if a.once:return 0
        time.sleep(a.interval)
if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(0)
    except Exception as error:print('SUIVI MÉGA REFUSÉ — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
