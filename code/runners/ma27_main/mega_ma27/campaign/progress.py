#!/usr/bin/env python3
"""Fast, read-only campaign progress; no solver, audit or integrity scan."""
from __future__ import annotations
import json
from pathlib import Path
import time

MAX_STATUS_BYTES=16*1024*1024
FINAL_STATES={'completed','stopped','failed','controller_dead_stale_state'}
METHODS=('R3','HH','HS','SH','SS','SS_legacy','ipopt_qcqp','ipopt_mumps')
REGIMES=('I','M','E','Q','A')
LABELS={'ipopt_qcqp':'Ipopt + MA27','ipopt_mumps':'Ipopt + MUMPS'}
SHORT_LABELS={'ipopt_qcqp':'Ipopt+MA27','ipopt_mumps':'Ipopt+MUMPS'}
STATES={'running':'en cours','initializing':'préparation','completed':'terminée',
        'stopping':'arrêt en cours','stopped':'arrêtée','failed':'interrompue sur erreur'}


def read_status(out):
    try:
        with (Path(out)/'STATUS.json').open('rb') as stream:
            data=stream.read(MAX_STATUS_BYTES+1)
    except FileNotFoundError:
        return None
    if len(data)>MAX_STATUS_BYTES:
        raise ValueError('STATUS.json dépasse la taille de lecture autorisée')
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('STATUS.json doit contenir un objet')
    return value


def number(value):
    return str(value) if type(value) is int else '?'


def table_row(label,done,total,strict,nonstrict,unknown):
    pending=total-done if type(total) is int and type(done) is int else None
    return '%-18s %14s %11s %11s %12s %11s' % (label,
        number(done)+'/'+number(total),number(strict),number(nonstrict),number(unknown),number(pending))


def declared_methods(mapping):
    return list(METHODS)+[m for m in('F_R3','F_X1','F_X2','F_H0')if m in mapping]+[m for m in mapping if m not in METHODS and m not in('F_R3','F_X1','F_X2','F_H0')]


def method_table(methods):
    lines=['Méthode                 Terminées  KKT strict  Non strict  Indéterminé  En attente']
    for method in declared_methods(methods):
        c=methods.get(method) or {}
        lines.append(table_row(LABELS.get(method,method),c.get('done',0),c.get('planned'),
            c.get('strict',0),c.get('nonstrict',c.get('failed',0)),c.get('indeterminate',0)))
    return lines


def regime_matrix(regimes,denominators=None):
    """Every method and all five regimes, including zero/pending denominators."""
    names=declared_methods({m:None for group in regimes.values()for m in(group.get('methods')or{})})
    lines=['Régimes — chaque case : strict/non strict/indéterminé ; terminées/prévues ; en attente = prévues − terminées',
           '%-13s'%('Régime (N)')+' '.join('%-20s'%SHORT_LABELS.get(m,m) for m in names)]
    for regime in REGIMES:
        group=regimes.get(regime) or {};n=group.get('N',(denominators or {}).get(regime,0))
        methods=group.get('methods') or {};cells=[]
        for method in names:
            c=methods.get(method) or {};planned=c.get('planned',n)
            cells.append('%s/%s/%s ; %s/%s' % (number(c.get('strict',0)),
                number(c.get('nonstrict',c.get('failed',0))),number(c.get('indeterminate',0)),
                number(c.get('done',0)),number(planned)))
        lines.append('%-13s'%('%s (%s)'%(regime,number(n)))+' '.join('%-20s'%cell for cell in cells))
    return lines


def format_status(status):
    pending=(status['total']-status['done']
             if type(status.get('total')) is int and type(status.get('done')) is int else None)
    state=status.get('state');lines=[
        '\n[%s] Campagne %s — terminées %s/%s | KKT strict %s | non strict %s | indéterminées %s | en attente %s' % (
            status.get('timestamp') or 'heure indisponible',STATES.get(state,state or 'en attente'),
            number(status.get('done')),number(status.get('total')),number(status.get('strict_qcqp')),
            number(status.get('nonstrict')),number(status.get('indeterminate')),number(pending)),
        'Créneaux occupés %s/%s | résolutions %s | audits %s' % (
            number(status.get('active_slots')),number(status.get('jobs',22)),
            number(status.get('active_solver')),number(status.get('active_audit')))]
    if status.get('phase'):lines.append(str(status['phase']))
    by_method=status.get('by_method') or {}
    if by_method:
        lines.append('\nGLOBAL — toutes les instances')
        lines.extend(method_table(by_method))
    by_regime=status.get('by_regime') or {}
    if by_regime:
        lines.append('')
        lines.extend(regime_matrix(by_regime))
    datasets=status.get('by_dataset') or {};running=status.get('running_datasets') or []
    if datasets:
        lines.append('\nPAR DATASET — « en attente » : aucun résultat final, aucune défaillance numérique classée')
        for name,d in sorted(datasets.items(),key=lambda item:item[1].get('position',0)):
            label=d.get('label',name)
            if label!=name:label=name+' ('+label+')'
            active=' — actif' if name in running else ''
            pending=d['total_cells']-d['done'] if type(d.get('total_cells')) is int and type(d.get('done')) is int else None
            lines.append('\nDATASET %s — %s%s | instances %s | terminées %s/%s | KKT strict %s | en attente %s' % (
                number(d.get('position')),label,active,number(d.get('N')),number(d.get('done')),
                number(d.get('total_cells')),number(d.get('strict_qcqp')),number(pending)))
            if not d.get('done') and name not in running:
                denominators=d.get('by_regime') or {}
                lines.append('En attente pour les méthodes prévues : strict/non strict/indéterminé = 0/0/0 ; terminées/prévues par méthode : '+
                    ' | '.join('%s : 0/%s'%(r,number(denominators.get(r,0))) for r in REGIMES))
                continue
            lines.extend(method_table(d.get('by_method') or {}))
            lines.extend(regime_matrix(d.get('regimes') or {},d.get('by_regime')))
    lines.append('Datasets terminés : %s/%s' % (number(status.get('datasets_completed')),
                                              number(status.get('datasets_total'))))
    if status.get('error'):lines.append('Erreur : '+str(status['error']))
    return '\n'.join(lines)


def _emit(text):
    print(text,flush=True)


def follow_status(out,*,once=False,interval=10,json_output=False,emit=None,loader=None,detail=False):
    if interval<=0:raise ValueError('Positive monitor interval required')
    emit=emit or _emit;loader=loader or read_status;out=Path(out)
    # Emit before the first filesystem access, including when the run is new.
    emit(json.dumps(dict(event='monitor_started',run=str(out)),ensure_ascii=False) if json_output
         else 'SUIVI ACTIF — '+str(out))
    while True:
        try:
            status=loader(out)
        except (OSError,ValueError,UnicodeError) as error:
            emit(json.dumps(dict(event='status_unavailable',error=str(error)),ensure_ascii=False)
                 if json_output else 'Statut momentanément indisponible : '+str(error))
            status=None
        else:
            if status is None:
                emit(json.dumps(dict(event='waiting_for_status',run=str(out)),ensure_ascii=False)
                     if json_output else 'EN ATTENTE DU PREMIER STATUT — préparation en cours ; journal : '+str(out/'controller.log'))
            else:
                status=observed_controller_state(status)
                emit(json.dumps(status,ensure_ascii=False) if json_output else format_status(status)if detail else compact_status(status))
        if once or (status or {}).get('state') in FINAL_STATES:return status
        time.sleep(min(interval,60))


def compact_status(status):
    """One regime grid and active datasets, without28repeated full reports."""
    methods=list((status.get('by_method')or{}))or list(METHODS)
    # JSON status files are key-sorted; keep the declared scientific order.
    preferred=list(METHODS)+['F_R3','F_X1','F_X2','F_H0']
    methods=[m for m in preferred if m in methods]+[m for m in methods if m not in preferred]
    lines=['[%s] %s | terminées %s/%s | KKT %s | non strict %s | indéterminé %s | créneaux %s/%s'%(status.get('timestamp','?'),STATES.get(status.get('state'),status.get('state','?')),number(status.get('done')),number(status.get('total')),number(status.get('strict_qcqp')),number(status.get('nonstrict')),number(status.get('indeterminate')),number(status.get('active_slots')),number(status.get('jobs',22))),
      'Chaque case : strict/non strict/indéterminé ; terminées/prévues',
      '%-10s'%('Régime N')+' '.join('%-19s'%SHORT_LABELS.get(m,m)for m in methods)]
    for regime in('I','M','E','Q'):
        group=(status.get('by_regime')or{}).get(regime)or{}
        if not group:continue
        cells=[]
        for method in methods:
            c=(group.get('methods')or{}).get(method)or{};n=c.get('planned',group.get('N',0))
            cells.append('%s/%s/%s ; %s/%s'%(number(c.get('strict',0)),number(c.get('nonstrict',c.get('failed',0))),number(c.get('indeterminate',0)),number(c.get('done',0)),number(n)))
        lines.append('%-10s'%('%s %s'%(regime,number(group.get('N'))))+' '.join('%-19s'%c for c in cells))
    datasets=status.get('by_dataset')or{};running=status.get('running_datasets')or[]
    if running:lines.append('Datasets actifs : '+' | '.join('%s %s/%s'%(name,number(datasets.get(name,{}).get('done')),number(datasets.get(name,{}).get('total_cells')))for name in running))
    lines.append('Datasets terminés : %s/%s'%(number(status.get('datasets_completed')),number(status.get('datasets_total'))))
    if status.get('error'):lines.append('Erreur : '+str(status['error']))
    return '\n'.join(lines)


def observed_controller_state(status):
    """A dead supervisor is an observed state, never24h of fictitious slots."""
    if status.get('state')not in('running','initializing','stopping'):return status
    pid=status.get('controller_pid');expected=status.get('controller_start_ticks')
    if pid is None or expected is None:return status
    try:
        fields=Path('/proc/%d/stat'%int(pid)).read_text().rsplit(')',1)[1].split()
        live=fields[0]!='Z'and fields[19]==str(expected)
    except (FileNotFoundError,ProcessLookupError):live=False
    except (PermissionError,ValueError,IndexError):return status
    if live:return status
    value=dict(status,state='controller_dead_stale_state',active_slots=0,active_solver=0,active_audit=0,active={},running_datasets=[],
               error='Le contrôleur identifié est absent ; cet ancien état ne représente plus des tâches actives.')
    return value
