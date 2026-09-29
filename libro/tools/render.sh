#!/usr/bin/env bash
# Renderiza páginas a PNG para inspección visual: tools/render.sh 03 1 5
set -euo pipefail
cd "$(dirname "$0")/.."
N=$(printf "%02d" "$((10#$1))")
mkdir -p build/cap$N/png
pdftoppm -r 70 -png -f "${2:-1}" -l "${3:-4}" build/cap$N/cap$N.pdf build/cap$N/png/p
ls build/cap$N/png/*.png
