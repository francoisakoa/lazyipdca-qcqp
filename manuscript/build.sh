#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for round in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error Akoa_LazyQCQP_20261111_R24.tex
  pdflatex -interaction=nonstopmode -halt-on-error Akoa_LazyQCQP_20261111_R24_Supplement.tex
done
