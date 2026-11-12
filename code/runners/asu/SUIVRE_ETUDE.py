#!/usr/bin/env python3
"""Read-only monitoring of all ASU study stages."""
from pathlib import Path
import argparse,time
import BOUNDS as study


def show(out):
    value=study.base.read(out/'ETUDE_STATUS.json',{})
    print('ÉTUDE ASU — '+value.get('stage','en attente')+' — '+value.get('timestamp',''),flush=True)
    if value.get('reason'):print('MOTIF : '+value['reason'],flush=True)
    stage=value.get('blocked_stage')if value.get('stage')=='blocked'else value.get('stage')
    folder={'qualification profil':'qualification','pilote ablation':'pilote','campagne complète':'campagne'}.get(stage)
    if folder:
        state=study.base.read(out/folder/'STATUS.json')
        if state:study.show(study.base.observed_status(state))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        state=show(a.out)
        if a.once or state.get('stage')in('completed','blocked'):break
        time.sleep(10.)
