#!/usr/bin/env bash
set -euo pipefail
LAZYQCQP_KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python3 - "$LAZYQCQP_KIT/mega_ma27" <<'STOP'
from pathlib import Path
import json,os,signal,subprocess,sys
folder=Path(sys.argv[1]);sys.path.insert(0,str(folder));import DEMARRER as launch
state=launch.read(folder/'LAUNCH_STATE.json',{})or{}
if state.get('state')in('starting','preparing')and launch.owned_live(state.get('worker_pid'),state.get('worker_start_ticks')):
    os.kill(state['worker_pid'],signal.SIGTERM);print('Arrêt demandé à la préparation identifiée.')
elif state.get('device')and state.get('run'):
    device=Path(state['device']);python=state.get('python',sys.executable)
    raise SystemExit(subprocess.call([python,str(device/'campaign/MEGA.py'),'--root',str(device),'arreter','--out',state['run']]))
else:print('Aucune préparation ou campagne identifiée dans ce kit.')
STOP
