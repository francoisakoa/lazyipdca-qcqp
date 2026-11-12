#!/usr/bin/env bash
set -euo pipefail
LAZYQCQP_KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 -u "$LAZYQCQP_KIT/mega_ma27/SUIVRE.py"
