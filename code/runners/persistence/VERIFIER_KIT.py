#!/usr/bin/env python3
"""Verify the delivered files; never compile or start an experiment."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/'MANIFESTE_SHA256.json').read_text())
    for name,expected in manifest['files'].items():
        path=HERE/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Fichier absent ou lien symbolique : '+name)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Empreinte différente : '+name)
    print('KIT VÉRIFIÉ : '+str(len(manifest['files']))+' fichiers ; aucun solveur lancé',flush=True)


if __name__=='__main__':main()
