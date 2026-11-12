#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ipopt_code=0
factoriel_code=0
bash "$KIT_DIR/REPRENDRE_IPOPT_ALIGNE.sh" || ipopt_code=$?
bash "$KIT_DIR/REPRENDRE_FACTORIEL.sh" || factoriel_code=$?
if [[ "$ipopt_code" -ne 0 || "$factoriel_code" -ne 0 ]]; then
  echo "Un dispositif n’a pas pu reprendre. Consulter SUIVRE_TOUT.sh ; l’autre reste indépendant." >&2
  exit 2
fi
