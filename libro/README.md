# Bioinformática Práctica — el libro

Libro de texto para maestría, en español, compuesto en LaTeX. Acompaña al curso de notebooks de
Google Colab de este repositorio: cada capítulo corresponde a un módulo y cada sección a una lección.

👤 **Juvenal Yosa, PhD** · juvenal.yosa@gmail.com · 🤖 Copiloto: **Claude** (Anthropic) · Licencia MIT

## Contenido

19 capítulos (módulos 0–18) organizados en cinco partes, unas 800 páginas, unas 250 figuras
vectoriales (TikZ/PGFPlots) generadas a partir de cálculos reproducibles y 549 referencias en
formato APA 7, verificadas una a una contra Crossref, DataCite u OpenLibrary.

| Parte | Capítulos |
|---|---|
| I · Fundamentos | 0 Laboratorio digital · 1 Biología molecular · 2 Formatos y bases de datos |
| II · Comparar secuencias | 3 Alineamiento · 4 MSA, motivos y HMM · 5 Filogenética |
| III · Del secuenciador al genoma | 6 NGS · 7 Mapeo · 8 Ensamblaje y anotación · 9 Variantes |
| IV · Genomas en acción | 10 Poblaciones y GWAS · 11 RNA-seq · 12 Célula única · 13 Epigenómica · 14 Metagenómica |
| V · Estructura, sistemas e inteligencia | 15 Estructural · 16 Redes y sistemas · 17 Machine learning · 18 Flujos reproducibles |

## Compilar

Requiere TeX Live 2025 (LuaLaTeX, biber, biblatex-apa, minted) y Pygments.

```bash
cd libro
tools/build_libro.sh            # libro completo -> build/libro/main.pdf  (~10 min)
tools/build_libro.sh 03 05      # solo algunos capítulos (\includeonly)
tools/build_cap.sh 3            # un capítulo aislado -> build/cap03/cap03.pdf
tools/render.sh 3 1 10          # páginas a PNG para revisión visual
```

## Verificar las referencias

```bash
python3 tools/verify_bib.py all   # o un capítulo: python3 tools/verify_bib.py 7
```

Cada entrada se contrasta con el registro oficial de su DOI (título, año y primer autor) o de su
ISBN; una sola discrepancia hace fallar la verificación.

## Estructura

```
main.tex              documento maestro (partes, preliminares, \include de capítulos)
estilo/bioinfo.sty    estilo del libro: tipografía, paleta del curso, cajas, títulos, APA
estilo/libro.sty      portada y páginas de parte
preliminares/         portada, créditos, prólogo, cómo leer este libro, colofón
capitulos/capNN.tex   un archivo por capítulo
bib/capNN.bib         referencias de cada capítulo (bibliografía al final de cada capítulo)
figuras/capNN/        generar.py + datos/TikZ de cada figura (reproducibles)
GUIA_AUTORES.md       normas de redacción, estilo y verificación
```
