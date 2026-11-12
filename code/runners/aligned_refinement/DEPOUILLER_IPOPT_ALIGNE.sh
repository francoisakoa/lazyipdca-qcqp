#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${LAZYQCQP_PYTHON:-python3}" -u "$SCRIPT_DIR/PILOTER_IPOPT.py" depouiller "$@"
