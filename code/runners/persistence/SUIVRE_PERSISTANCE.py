#!/usr/bin/env python3
from pathlib import Path
import argparse
import datetime
import json
import time


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def show(out):
    print('\n['+datetime.datetime.now(datetime.timezone.utc).isoformat()+']',flush=True)
    state=read(out/'ETUDE_STATUS.json')
    print('PERSISTANCE : '+str(state.get('stage','aucun démarrage')),flush=True)
    if state.get('reason'):print('MOTIF : '+str(state['reason']),flush=True)
    for name in ('qualification/STATUS.json','STATUS.json'):
        detail=read(out/'experience'/name)
        if detail:print(name+' : '+json.dumps(detail,ensure_ascii=False),flush=True)
    qualification=read(out/'experience/QUALIFICATION.json')
    if qualification:print('Qualification validée : '+str(qualification.get('passed')),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/extra/LAZYQCQP_PERSISTANCE_RESULTS/ETUDE_PERSISTANCE_20261110_R1'));p.add_argument('--once',action='store_true');a=p.parse_args()
    try:
        while True:
            show(a.out)
            if a.once:break
            time.sleep(10)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
