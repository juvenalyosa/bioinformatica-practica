#!/usr/bin/env bash
# Compila el libro completo en build/libro/main.pdf
#   tools/build_libro.sh            (todos los capítulos existentes)
#   tools/build_libro.sh 03 05      (solo esos capítulos, vía \includeonly)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/libro
mkdir -p "$OUT/capitulos" "$OUT/preliminares"
PRE=""
if [ $# -gt 0 ]; then
  L=$(for n in "$@"; do printf "capitulos/cap%02d," "$((10#$n))"; done); PRE="\\includeonly{${L%,}}"
fi
latexmk -lualatex -shell-escape -interaction=nonstopmode -halt-on-error \
  ${PRE:+-usepretex="$PRE"} -outdir="$OUT" -auxdir="$OUT" main.tex > "$OUT/latexmk.out" 2>&1 \
  || { grep -n -A5 '^!' "$OUT/main.log" | head -30; tail -20 "$OUT/latexmk.out"; exit 1; }
echo "OK -> $OUT/main.pdf ($(python3 -c "import fitz;print(fitz.open('$OUT/main.pdf').page_count)") páginas)"
grep -E "undefined|multiply defined" "$OUT/main.log" | sort -u | head -20 || true
