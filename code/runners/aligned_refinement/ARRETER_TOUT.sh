#!/usr/bin/env bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$KIT_DIR/ARRETER_IPOPT_ALIGNE.sh"
bash "$KIT_DIR/ARRETER_FACTORIEL.sh"
