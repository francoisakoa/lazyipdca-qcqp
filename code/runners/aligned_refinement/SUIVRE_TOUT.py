#!/usr/bin/env python3
"""Two independent CPU allocations, one compact ten-second display."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys,time
KIT=Path(__file__).resolve().parent

def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None

def alive(pid,ticks):
    try:
        parts=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return type(pid)is int and type(ticks)is int and parts[0]!='Z'and int(parts[19])==ticks
    except (OSError,ValueError,IndexError):return False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interval',type=float,default=10);p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        print('\n['+datetime.datetime.now(datetime.timezone.utc).isoformat()+']',flush=True);running=False
        for last,pilot,label in [('LAST_IPOPT_RUN.json','PILOTER_IPOPT.py','IPOPT — cœurs0–21'),('LAST_RUN.json','PILOTER.py','RAFFINEMENT — cœur23')]:
            item=read(KIT/last);print(label,flush=True)
            if item is None:print('Aucun lancement enregistré.',flush=True);continue
            subprocess.run([sys.executable,'-u',str(KIT/pilot),'suivre','--once'],check=False)
            running=running or alive(item.get('pid'),item.get('pid_start_ticks'))
        if a.once or not running:return
        time.sleep(max(1,a.interval))
if __name__=='__main__':main()
