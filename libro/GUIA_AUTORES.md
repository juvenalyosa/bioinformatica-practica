# Guía para autores de capítulos — «Bioinformática Práctica» (libro LaTeX)

Autor del libro: **Juvenal Yosa, PhD** (juvenal.yosa@gmail.com). Copiloto: Claude.
Público: estudiantes de **maestría**. Idioma: **español** (prosa); código, identificadores y
términos técnicos en inglés cuando así se usan en la literatura (márquelos con `\ingles{...}`).

## 1. Archivos que usted crea (y SOLO estos)

| Archivo | Contenido |
|---|---|
| `libro/capitulos/capNN.tex` | El capítulo completo (empieza con `\chapter{...}\label{cap:NN}`) |
| `libro/bib/capNN.bib` | Referencias del capítulo (biblatex, campos `journaltitle`, `date`, `doi`) |
| `libro/figuras/capNN/` | Scripts `generar.py` y archivos `.tex`/`.dat`/`.pdf` de figuras |

**Prohibido** editar `estilo/bioinfo.sty`, otros capítulos, `main.tex` o cualquier archivo fuera de
`libro/`. Si necesita un macro, defínalo al inicio de su capítulo con prefijo `\capNN...`
(p. ej. `\newcommand{\capcincoarbol}{...}`; los nombres de macro no admiten dígitos).
No haga `git commit`.

## 2. Modelo a imitar

Lea completo **`libro/capitulos/cap03.tex`** (capítulo piloto) y mire su PDF
`libro/build/cap03/cap03.pdf`. Imite su tono, densidad, estructura y uso de cajas.
Su capítulo debe ser **más extenso** que el piloto: apunte a **7–10 páginas por lección**
(sección), es decir, 25–40 páginas por capítulo.

## 3. Estructura obligatoria del capítulo

```
\chapter{Título}\label{cap:NN}
\epigrafe{Cita en español (traducida si hace falta)}{\textcite{cNN-clave}}   % cita real y verificada
\begin{objetivos} \begin{itemize} ... \end{itemize} \end{objetivos}
Párrafos de apertura (motivación + mapa del capítulo con \cref a las secciones)
\section{Lección N.1 ...}\label{sec:NN-...}
\enlacecolab{ruta/notebook.ipynb}{N.1 · Título corto}   % si el notebook existe
\enlacerepo{N.1 · Título corto}                          % si aún no existe
   ... contenido ...
\section{Lección N.2 ...}  ...
\begin{resumen} \begin{itemize} ... \end{itemize} \end{resumen}
\begin{ejercicios} \ej{1} ... \ej{2} ... \ej{3} ... \end{ejercicios}   % 7–10 ejercicios
\referenciascapitulo
```

## 4. Arco pedagógico de CADA sección (lección)

1. **Apertura intuitiva**: una situación cotidiana o comparación dentro de `\begin{intuicion}[Título evocador]`.
   **Nunca** escriba la palabra «Analogía» como rótulo o título. La comparación sirve de puerta
   de entrada; no convierta toda la sección en una analogía.
2. **Conceptos básicos** en prosa clara, con definiciones formales (`definicion`).
3. **Teoría formal extensa**: ecuaciones numeradas con `\label{eq:NN-...}`, cada una seguida de
   `\begin{simbolos}\simbolo{$x$}{significado}...\end{simbolos}` que explique **todos** los símbolos.
   Derivaciones paso a paso; resultados principales en `teorema`. La técnica sube de nivel a lo largo
   de la sección. Explique el *porqué*, no solo el *qué*. Prosa entretenida pero rigurosa.
4. **Ejemplo resuelto numérico** (`ejemplo`) con cifras verificadas (calcúlelas con Python).
5. **Código** breve y correcto en `\begin{codigo}[nombre.py] ... \end{codigo}` (Python) o
   `\begin{consola}[titulo] ... \end{consola}` (bash). Líneas ≤ 72 caracteres. APIs actuales
   (Biopython ≥ 1.83: `Bio.Blast.Applications` ya NO existe; use `subprocess`).
6. **Figura(s)** profesionales (ver §6) con pie extenso e interpretativo.
7. Cajas de apoyo cuando aporten: `historia` (nota histórica con cita), `cuidado` (errores
   frecuentes), `profundizacion` (material avanzado), `ideaclave` (1–2 líneas memorables).

## 5. Entornos y macros disponibles (conjunto CERRADO)

- Cajas: `objetivos`, `intuicion[título]`, `definicion{título}{label}`, `teorema{título}{label}`,
  `ejemplo{título}{label}`, `historia[título]`, `cuidado[título]`, `profundizacion[título]`,
  `ideaclave`, `resumen`, `ejercicios` (ítems con `\ej{1|2|3}`), `simbolos` + `\simbolo{}{}`.
  Nota: `definicion`, `teorema`, `ejemplo` son `tcbtheorem`: `\begin{ejemplo}{Título}{etiqueta}`;
  se referencian con `\cref{ej:etiqueta}`, `\cref{def:...}`, `\cref{teo:...}`.
- Código: `codigo[archivo]`, `consola[título]`, `\py{expr}`, `\cmd{herramienta}`, `\archivo{ruta}`.
- Secuencias: `\secuencia{ACGT...}` (colorea bases), `\nuc{A}`.
- Texto: `\ingles{}`, `\especie{Homo sapiens}`, `\gen{TP53}`, `\epigrafe{}{}`.
- Matemáticas: `\Prob`, `\E`, `\Var`, `\R`, `\argmax`, `\argmin`; `\num{}`, `\SI{}{}` (siunitx, coma decimal).
- Enlaces: `\enlacecolab{ruta}{texto}`, `\enlacerepo{texto}`, `\cref{...}`.
- Colores (paleta del curso): `azul naranja aqua amarillo magenta verde violeta rojo`,
  `tinta tinta2 gris rejilla base papel azulnoche azulprofundo azulmedio azulclaro azulpalido
  rojooscuro rojoclaro nocturno`, nucleótidos `nucA nucC nucG nucT`.
- TikZ: estilos `caja=<color>`, `flecha=<color>`, `etiqueta`, `nt=<A|C|G|T>` (base coloreada).
  pgfplots: use siempre `\begin{axis}[curso, ...]` (ciclo de colores y tipografía del libro);
  mapas de color `colormap name=curso` / `cursodiv`.
- En español los decimales usan **coma**: escriba `0{,}05` en modo matemático o `\num{0.05}`.
- Etiquetas con prefijo del capítulo: `sec:NN-`, `fig:NN-`, `tab:NN-`, `eq:NN-`.
  Puede referenciar otros capítulos con `\cref{cap:MM}` (se resuelven en el libro completo).

## 6. Figuras: calidad de libro, estilo «cinematográfico»

- Preferir **TikZ/pgfplots** vectoriales (fuentes y colores idénticos al texto). Mínimo **3–4 figuras
  por lección**: diagramas conceptuales (flujos, mecanismos, esquemas moleculares), gráficas de datos
  reales o simulados, matrices, árboles, etc.
- Datos de gráficas: calcúlelos con un script `figuras/capNN/generar.py` (numpy/scipy/Biopython) que
  escriba `.tex` o `.dat` que luego se cargan con `\input{figuras/capNN/x.tex}` o
  `\addplot table {figuras/capNN/x.dat};`. Así las figuras son reproducibles y exactas.
- Si un gráfico es muy denso (miles de puntos), use matplotlib guardando **PDF** vectorial con
  la paleta del curso (valores hex arriba) y fuente sans, e inclúyalo con `\includegraphics`.
- Anchura ≤ `\linewidth` (13 cm). Pies de figura largos e interpretativos (qué mirar, qué concluir).
- Sin imágenes descargadas de internet (derechos de autor).

## 7. Referencias — REQUISITO CRÍTICO

- **15–30 referencias por capítulo**, todas reales, citadas en el texto con `\textcite{}` / `\parencite{}`
  (APA 7, generado por biblatex-apa). Incluya los artículos originales de cada método, revisiones
  de autoridad y 1–2 libros de texto. **Cada entrada del .bib debe citarse** en el texto.
- Claves con prefijo del capítulo: `cNN-apellidoAAAA` (p. ej. `c05-felsenstein1981`).
- Cada entrada DEBE tener un **`doi`** verificable (preferido), o `isbn` (libros sin DOI), o `url`
  (solo `@software`/`@online`, p. ej. documentación de herramientas).
- Obtenga los metadatos de **Crossref**, nunca de memoria:
  `curl -s "https://api.crossref.org/works?query.bibliographic=TITULO+AUTOR&rows=3&mailto=juvenal.yosa@gmail.com"`
  o `curl -s https://api.crossref.org/works/DOI`. Copie título, año, revista, volumen, número, páginas y
  autores exactamente como los devuelve Crossref.
- Ejecute **`python3 tools/verify_bib.py NN`** (desde `libro/`) hasta obtener `NN/NN verificadas`.
  Si una referencia no se puede verificar, **elimínela** y reescriba el pasaje; nunca invente.
- Afirmaciones que atribuya a un artículo deben ser lo que el artículo realmente dice.

## 8. Compilar e inspeccionar

```
cd libro
tools/build_cap.sh NN          # compila SOLO su capítulo en build/capNN/ (seguro en paralelo)
tools/render.sh NN 1 40        # PNG de páginas en build/capNN/png/ para revisarlas visualmente
```
Requisitos para terminar: compila sin errores; `Overfull >20pt: 0`; sin citas indefinidas
(las referencias `cap:MM` a otros capítulos sí pueden quedar indefinidas en la compilación aislada);
`verify_bib.py` 100 % verificado; revisó visualmente con Read las PNG de TODAS sus figuras y de
varias páginas (nada se sale del margen, nada se solapa, cajas bien cortadas).

## 9. Informe final (breve)

Devuelva solo: nº de páginas, nº de figuras, nº de referencias verificadas, lista de secciones,
y cualquier problema pendiente. No pegue el contenido del capítulo.

## 10. Trampas de LaTeX ya conocidas

- Un `\input` dentro de las opciones de `\begin{axis}[...]` cuelga LuaLaTeX sin error: defina un
  estilo con `\pgfplotsset{capNN estilo/.style={...}}` y úselo.
- Escape `<` y `>` dentro de nodos TikZ (use `$<$`, `$>$`).
- Si mata una compilación colgada, borre `build/capNN/*.aux` y `*.bcf` antes de recompilar.
- `\py{}` (minted inline) dentro de `simbolos`, tablas tabularx o `\caption` rompe la caché de minted:
  use `\texttt{}` en esos lugares.
- Un `\input` dentro de `tabular` rompe `\bottomrule` en LuaLaTeX: escriba las filas directamente.
