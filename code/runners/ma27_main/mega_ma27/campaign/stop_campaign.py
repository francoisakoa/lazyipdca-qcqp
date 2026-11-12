#!/usr/bin/env python3
"""Cooperative stop and process verification for one existing MEGA run.

Usage:
    python stop_campaign.py --root /path/to/R6 --run /path/to/run
    python stop_campaign.py --root /path/to/R6 --run /path/to/run --check-only

No process is signalled and no result is deleted.  A surviving worker makes the
check fail, even when STATUS.json already says ``stopped``.  The original
controller's LAUNCH.lock is held while requesting and checking its stop.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid


class StopRefused(RuntimeError):
    """The exact reference run could not be proved inactive."""


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _read_object(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StopRefused("Lecture impossible : %s (%s)" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise StopRefused("Objet JSON attendu : " + str(path))
    return value


def _validated_paths(root, run):
    root = Path(root).expanduser().resolve(strict=True)
    run = Path(run).expanduser().resolve(strict=True)
    if not root.is_dir() or not run.is_dir() or root == run:
        raise StopRefused("Les dossiers source et résultats doivent être distincts.")
    plan = _read_object(run / "PLAN.json")
    for field, expected in (("root", root), ("output_directory", run)):
        value = plan.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise StopRefused("Identité absolue absente du plan : " + field)
        if Path(value).resolve() != expected:
            raise StopRefused("Le plan ne désigne pas le dossier demandé : " + field)
    return root, run


def _proc_info(pid):
    """Read one stable /proc identity; a reused or disappearing PID is ignored."""
    directory = Path("/proc") / str(pid)
    try:
        before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if before[0] == "Z":
            return None
        args = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if before[19] != after[19] or after[0] == "Z":
            return None
        return dict(pid=int(pid), ppid=int(after[1]), start_ticks=after[19],
                    args=[os.fsdecode(arg) for arg in args if arg])
    except PermissionError as exc:
        try:
            same_user = directory.stat().st_uid == os.getuid()
        except OSError:
            return None
        if same_user:
            raise StopRefused("Un processus du même utilisateur est illisible dans /proc : " + str(pid)) from exc
        return None
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def _process_table():
    if not Path("/proc/self/stat").is_file():
        raise StopRefused("Le contrôle d'arrêt exige Linux et /proc.")
    table = {}
    for item in Path("/proc").iterdir():
        if item.name.isdecimal():
            info = _proc_info(int(item.name))
            if info is not None:
                table[info["pid"]] = info
    return table


def _ancestors(table):
    excluded = set()
    # A container can expose host /proc while getpid() uses a nested namespace.
    # Use the PID in this procfs mount, not the process's namespace-local PID.
    try:
        pid = int(Path("/proc/self/stat").read_text().split("(", 1)[0].strip())
    except (OSError, ValueError):
        raise StopRefused("Impossible d'identifier notre processus dans /proc.")
    while pid > 0 and pid not in excluded:
        excluded.add(pid)
        item = table.get(pid)
        if item is None:
            break
        pid = item["ppid"]
    return excluded


def _names_run(argument, run):
    """Match an argv path, not text embedded in a shell command or script."""
    if argument.startswith("--") and "=" in argument:
        argument = argument.split("=", 1)[1]
    prefix = str(run)
    return argument == prefix or argument.startswith(prefix + os.sep)


def _is_observer(args, root):
    if not args:
        return False
    # A following terminal does not own a solver and must not block migration.
    if Path(args[0]).name in ("tail", "head", "cat", "less", "more"):
        return True
    for index, argument in enumerate(args):
        candidate = Path(argument)
        if candidate.is_absolute() and candidate.parent in (root, root / "campaign") and candidate.suffix == ".py":
            if "suivre" in args[index + 1:]:
                return True
    return False


def _owned_processes(root, run, previous=None):
    table = _process_table()
    excluded = _ancestors(table)
    seeds = {
        pid for pid, item in table.items()
        if pid not in excluded and not _is_observer(item["args"], root)
        and any(_names_run(arg, run) for arg in item["args"])
    }
    # Retain an observed descendant through reparenting, but never a reused PID.
    for pid, started in (previous or {}).items():
        item = table.get(pid)
        if item and pid not in excluded and item["start_ticks"] == started:
            seeds.add(pid)
    changed = True
    while changed:
        children = {pid for pid, item in table.items()
                    if pid not in excluded and item["ppid"] in seeds}
        changed = not children.issubset(seeds)
        seeds.update(children)
    return [table[pid] for pid in sorted(seeds)]


@contextmanager
def _launch_lock(run):
    path = run / "LAUNCH.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StopRefused("Verrou de lancement inaccessible : " + str(path)) from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StopRefused("Un lancement ou une reprise de cette campagne est en cours.") from exc
        yield
    finally:
        os.close(fd)


def _running_locked(run):
    path = run / "RUNNING.lock"
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise StopRefused("Verrou d'exécution inaccessible : " + str(path)) from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(fd)


def _option(args, flag):
    values = []
    for index, argument in enumerate(args):
        if argument == flag and index + 1 < len(args):
            values.append(args[index + 1])
        elif argument.startswith(flag + "="):
            values.append(argument.split("=", 1)[1])
    return values[0] if len(values) == 1 else None


def _controller_token(root, run):
    info = _read_object(run / "CONTROLLER.json")
    owner = _read_object(run / "CONTROLLER_OWNER.json")
    token = info.get("token")
    if not isinstance(token, str) or not token or len(token) > 256 or owner.get("token") != token:
        raise StopRefused("Le token du contrôleur actif ne correspond pas au reçu.")
    script = info.get("script")
    if not isinstance(script, str) or not Path(script).is_absolute():
        raise StopRefused("Chemin absolu du contrôleur absent du reçu.")
    script_path = Path(script).resolve()
    if info.get("out") != str(run) or root not in script_path.parents or not script_path.is_file():
        raise StopRefused("Le reçu du contrôleur désigne une autre campagne.")
    candidates = _owned_processes(root, run)
    matches = [item for item in candidates if script in item["args"] and "_run" in item["args"]
               and _option(item["args"], "--out") == str(run)
               and _option(item["args"], "--token") == token]
    if len(matches) != 1:
        raise StopRefused("Le contrôleur actif n'est pas identifié par sa commande et son token.")
    return token


def _request_stop(root, run):
    token = _controller_token(root, run)
    value = dict(requested=_stamp(), requester_pid=os.getpid(), token=token,
                 request_source="R8 cooperative migration")
    path = run / "STOP.json"
    temporary = run / ("STOP." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inactive_receipt(root, run):
    return dict(stopped=True, root=str(root), run=str(run), checked_at_utc=_stamp(),
                controller_locked=False, processes=[], process_signals_sent=0,
                results_deleted=False)


def _description(processes):
    return ", ".join(str(item["pid"]) for item in processes) or "aucun PID visible"


@contextmanager
def stopped_run_lock(root, run):
    """Prove inactivity and keep LAUNCH.lock until the caller leaves the context.

    Use this around a new controller launch so that its precondition and launch
    are protected by the same lock.  Do not nest it with _launch_lock or with
    another stopped_run_lock for the same run.
    """
    root, run = _validated_paths(root, run)
    with _launch_lock(run):
        processes = _owned_processes(root, run)
        locked = _running_locked(run)
        if locked or processes:
            raise StopRefused("Campagne encore active : verrou=%s ; PIDs=%s" %
                              (locked, _description(processes)))
        # The second scan closes the common exit/spawn observation race.
        processes = _owned_processes(root, run)
        if _running_locked(run) or processes:
            raise StopRefused("Activité détectée pendant la vérification : " + _description(processes))
        yield _inactive_receipt(root, run)


def assert_stopped(root, run):
    """Return an inactivity receipt, or raise StopRefused if activity remains.

    This is a point-in-time check.  Use stopped_run_lock for an atomic
    check-and-launch operation rather than acquiring LAUNCH.lock twice.
    """
    with stopped_run_lock(root, run) as receipt:
        return receipt


def stop_reference(root, run, timeout=90):
    """Request the controller's cooperative stop and wait for its whole tree.

    Existing outputs, PLAN.json and INSTALLATION.json remain untouched.  A
    surviving orphan or mismatched ownership receipt fails closed.  No signal
    is sent, including after a timeout.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Le délai d'arrêt doit être strictement positif et fini.")
    root, run = _validated_paths(root, run)
    deadline = time.monotonic() + timeout
    known = {}
    with _launch_lock(run):
        processes = _owned_processes(root, run)
        known.update({item["pid"]: item["start_ticks"] for item in processes})
        if _running_locked(run):
            _request_stop(root, run)
            print("Arrêt demandé au contrôleur de la campagne sélectionnée.", flush=True)
        elif processes:
            raise StopRefused("Contrôleur absent mais processus du run encore présents : " + _description(processes))
        stable = 0
        previous_ids = None
        while True:
            processes = _owned_processes(root, run, known)
            known.update({item["pid"]: item["start_ticks"] for item in processes})
            locked = _running_locked(run)
            if not locked and not processes:
                stable += 1
                if stable >= 2:
                    print("ARRÊT_CONFIRMÉ : contrôleur et travailleurs absents.", flush=True)
                    return _inactive_receipt(root, run)
            else:
                stable = 0
                ids = tuple(item["pid"] for item in processes)
                if ids != previous_ids:
                    print("Arrêt en cours ; processus restants : " + _description(processes), flush=True)
                    previous_ids = ids
            if time.monotonic() >= deadline:
                raise StopRefused("Arrêt non confirmé après %.1f s ; verrou=%s ; PIDs=%s. Aucun signal forcé envoyé." %
                                  (timeout, locked, _description(processes)))
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = assert_stopped(args.root, args.run) if args.check_only else stop_reference(args.root, args.run, args.timeout)
    except (StopRefused, OSError, ValueError) as exc:
        print("ARRÊT_REFUSÉ : " + str(exc), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
