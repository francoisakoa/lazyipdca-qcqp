#!/usr/bin/env bash
set -euo pipefail
LAZYQCQP_KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LAZYQCQP_RUN="${1:-$(cat "$LAZYQCQP_KIT/mega_ma27/LAST_RUN.txt")}"
LAZYQCQP_EXPORT="${2:-/extra/LAZYQCQP_DEPOUILLEMENT/$(date -u +%Y%m%dT%H%M%S_%NZ)}"
exec python3 -u "$LAZYQCQP_KIT/mega_ma27/DEPOUILLER.py" --run "$LAZYQCQP_RUN" --out "$LAZYQCQP_EXPORT" --archive
