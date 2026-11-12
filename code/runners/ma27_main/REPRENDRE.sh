#!/usr/bin/env bash
set -euo pipefail
LAZYQCQP_KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
python3 - "$LAZYQCQP_KIT/mega_ma27" <<'RESUME'
from pathlib import Path
import json,subprocess,sys
folder=Path(sys.argv[1]);state=json.loads((folder/'LAUNCH_STATE.json').read_text());device=Path(state['device']);run=Path(state['run']);python=state['python']
subprocess.run([python,str(device/'campaign/MEGA.py'),'--root',str(device),'reprendre','--out',str(run)],check=True)
subprocess.run([python,str(device/'campaign/MEGA.py'),'--root',str(device),'suivre','--out',str(run),'--interval','10'],check=True)
RESUME
