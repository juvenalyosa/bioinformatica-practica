# Guía para construir una lección del curso "Bioinformática Práctica"

Repo: /Users/juvenalyosa/bioinformatics (GitHub juvenalyosa/bioinformatica-practica). Autor: Juvenal Yosa, PhD. Copiloto: Claude.
Scratchpad (SP): /private/tmp/claude-501/-Users-juvenalyosa-bioinformatics/87737d69-f260-4504-a19d-e50e273ee5bb/scratchpad

## Reglas duras
- NO hacer git commit/push. NO editar README.md, utils/, tools/nbbuild.py, tools/runnb.py (reportar si hace falta).
- No tocar archivos de otras lecciones (otros agentes trabajan en paralelo).
- Escribir el builder en VARIOS pasos (Write de la primera parte y luego añadir secciones con Edit o `cat >>`): las llamadas gigantes fallan por conexión.

## Estilo (leer primero como referencia: tools/builders/build_13.py, build_32.py, build_34.py del repo)
- Builder en SP/build_XY.py; `from nbbuild import NB, SETUP, header, gif` (SP/nbbuild.py == tools/nbbuild.py). Guardar el notebook en el repo en `<modulo>/<archivo>.ipynb` (usar ruta absoluta del repo; aceptar env NB_ROOT opcional).
- Empieza con `nb.md(header(PATH, "Lección X.Y · Título", "Módulo N — Nombre", duración, nivel, requisitos))` → objetivos → mapa de la clase → `nb.code(SETUP + "\nimport plotly.express as px\nimport plotly.graph_objects as go")`. El pie de página se añade solo en `nb.save`.
- Prosa en español con acentos correctos, MUY pedagógica y extensa. NUNCA escribir la palabra "analogía": las comparaciones cotidianas van integradas en la explicación.
- Por concepto: intuición en palabras simples → ejemplo pequeño resuelto a mano con números → ecuación en LaTeX bonita + tabla de símbolos → código → figura profesional → bloque "> 🔎 **Qué observamos.**". Añadir "> 🤔 **Antes de ejecutar, prediga…**" y "> ✅ **Compruebe su comprensión**".
- ≥2 figuras interactivas Plotly con hover didáctico (plantilla "curso" ya registrada por ec.set_style; título con `<br><sup>subtítulo</sup>`, leyenda `yanchor="bottom", y=1.02`, márgenes suficientes).
- ≥1–2 animaciones: `ec.animate(fig, update, frames, interval, name="X.Y_corto")` (≤60 cuadros), y JUSTO ANTES una celda `nb.md(gif("<modulo-dir>", "X.Y_corto", "pie de figura"))`.
- Muchas figuras matplotlib profesionales: título = conclusión, subtítulo con contexto (`ec.title(ax, main, sub)` / `ec.fig_title(fig, main, sub)`), etiquetas directas, paleta fija `ec.CATEGORICAL`/`ec.BLUE`…, `ec.NUC_COLORS` para nucleótidos (siempre con la letra), `ec.CMAP_SEQ` magnitud, `ec.CMAP_DIV` con signo, nunca doble eje Y.
- Identificadores en inglés; comentarios en español. Ejercicios con soluciones ocultas: celda que empieza `#@title 🔑 Solución — Ejercicio N { display-mode: "form" }`. Resumen 📌 y lecturas 📚 SOLO con referencias reales de las que esté seguro.
- Pasos sólo-Colab protegidos (`IN_COLAB`, `shutil.which`; `try: import X except ImportError: %pip install -q X`). apt-get sólo si IN_COLAB.
- Datos reales: descargar (NCBI/UniProt/…) con copia de respaldo en el repo `data/` (o `data/api_cache/`), cargada primero desde `../data`, luego red, luego URL raw de GitHub `https://raw.githubusercontent.com/juvenalyosa/bioinformatica-practica/main/data/...`. Reutilizar lo existente: data/NC_045512.2.gb, NC_004718.3.gb, NC_000908.2.gb, NC_000913.3.fasta.gz, data/api_cache/ (TP53, calmodulina, colágeno…).
- Tiempo total del notebook < ~3 min en Colab.

## Prueba (aislada, para no interferir)
```
ISO=$SP/iso_XY; mkdir -p $ISO/<modulo-dir>; ln -sfn /Users/juvenalyosa/bioinformatics/utils $ISO/utils; ln -sfn /Users/juvenalyosa/bioinformatics/data $ISO/data
cp <notebook del repo> $ISO/<modulo-dir>/
JUPYTER_PATH=$SP/venv/share/jupyter RUN_DIR=$SP/run_XY $SP/venv/bin/python /Users/juvenalyosa/bioinformatics/tools/runnb.py $ISO/<modulo-dir>/<nb>.ipynb
```
Imprime `ok ... images N` o `ERROR cell ...`. Las PNG (incluye Plotly vía kaleido) quedan en $SP/run_XY/<nb>/. Leer CADA PNG con Read y corregir superposiciones/recortes; mirar algunos cuadros de los GIF en $ISO/assets/. Instalar paquetes extra en el venv con `VIRTUAL_ENV=$SP/venv uv pip install ...` sólo si Colab los trae o el notebook los instala.

## Informe final (breve)
Rutas (notebook, builder), datos añadidos, lista de figuras/animaciones/interactivos, celdas sólo-Colab no verificadas, cambios necesarios en archivos compartidos.

## Coherencia con el libro (obligatorio desde el Módulo 7)
- EL LIBRO MANDA: cada lección acompaña una sección de `libro/capitulos/capNN.tex`. Léala COMPLETA antes de escribir y use exactamente
  sus símbolos, convenciones (índices 0/1, intervalos semiabiertos, normalizaciones), ecuaciones y los MISMOS ejemplos resueltos con las
  mismas cifras (`libro/figuras/capNN/cifras.txt`). La notación propia del notebook no debe chocar con la del libro.
- Progresión: cada sección empieza muy básica (situación cotidiana, palabras simples) y sube hasta nivel de maestría.
- Ejemplos siempre con contexto práctico real (brote, clínica, el clon LTEE de *E. coli* SRR2584863 que recorre los Módulos 6–8…).
- Antes de comitear un módulo: auditoría de solo lectura de los builders contra el capítulo (símbolos, cifras, referencias, afirmaciones
  dudosas); corregir los notebooks, y anotar las erratas del libro que aparezcan para corregirlas en `capNN.tex` (compilar el capítulo aislado).
- Herramientas que sólo existen en Colab: instalar protegido (`IN_COLAB`/`shutil.which`) y dejar SIEMPRE resultados precomputados en `data/`
  como respaldo; verificar con `curl -I` las URLs de descarga.
