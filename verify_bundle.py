#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST_SHA256.json').read_text())
bad=[name for name,digest in manifest.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
if bad:raise SystemExit('ÉCHEC : '+', '.join(bad))
print(f'VÉRIFIÉ : {len(manifest)} fichiers identiques au manifeste')
