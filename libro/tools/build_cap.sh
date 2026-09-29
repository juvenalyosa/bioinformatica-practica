#!/usr/bin/env bash
# Compila UN capítulo de forma aislada:  tools/build_cap.sh 03
# Salida: build/cap03/cap03.pdf   (cada capítulo en su propio directorio,
# así varios agentes pueden compilar en paralelo sin pisarse).
set -euo pipefail
cd "$(dirname "$0")/.."
N=$(printf "%02d" "$((10#$1))")
OUT=build/cap$N
mkdir -p "$OUT"
cat > "$OUT/cap$N.tex" <<TEX
\documentclass[11pt,twoside,openany]{book}
\usepackage{estilo/bioinfo}
\addbibresource{bib/cap$N.bib}
\begin{document}
\setcounter{chapter}{$((10#$N - 1))}
\input{capitulos/cap$N}
\end{document}
TEX
latexmk -lualatex -shell-escape -interaction=nonstopmode -halt-on-error \
  -outdir="$OUT" -auxdir="$OUT" "$OUT/cap$N.tex" > "$OUT/latexmk.out" 2>&1 \
  || { tail -40 "$OUT/latexmk.out"; grep -n -A5 '^!' "$OUT/cap$N.log" | head -40; exit 1; }
echo "OK -> $OUT/cap$N.pdf  ($(python3 -c "import fitz;print(fitz.open('$OUT/cap$N.pdf').page_count)") páginas)"
grep -cE 'Overfull \\hbox \(([2-9][0-9]|[0-9]{3,})\.' "$OUT/cap$N.log" | xargs -I{} echo "Overfull >20pt: {}"
grep -E "undefined|Citation .* undefined|multiply defined" "$OUT/cap$N.log" | sort -u | head -20 || true
