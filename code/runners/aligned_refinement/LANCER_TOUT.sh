#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--nouveau" ) ]]; then
  echo "Utiliser sans argument ou avec --nouveau ; les chemins personnalisés se passent aux lanceurs séparés." >&2
  exit 2
fi
bash "$KIT_DIR/LANCER_IPOPT_ALIGNE.sh" "$@"
bash "$KIT_DIR/LANCER_FACTORIEL.sh" "$@"
