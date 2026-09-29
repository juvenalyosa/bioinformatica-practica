# 🧬 Bioinformática Práctica

[![Copiloto: Claude](https://img.shields.io/badge/Copiloto-Claude-D97757?logo=claude&logoColor=white)](https://claude.ai) [![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-2a78d6.svg)](LICENSE)

👤 **Juvenal Yosa, PhD** · ✉️ [juvenal.yosa@gmail.com](mailto:juvenal.yosa@gmail.com) · 🤖 Copiloto: **Claude** (Anthropic)

Curso de **bioinformática práctica para maestría**, en español. Cada lección es un notebook de Google Colab que funciona como una **clase completa**: primero una explicación intuitiva con ejemplos cotidianos, luego la teoría formal (ecuaciones explicadas término a término) y los **experimentos computacionales ahí mismo**, con figuras, animaciones y gráficos interactivos de calidad de publicación.

* **Idioma:** explicaciones en español; código y términos técnicos en inglés (como en la literatura).
* **Requisitos:** un navegador y una cuenta de Google. Todo corre en Colab gratuito.
* **Cómo usarlo:** haga clic en el botón *Open in Colab* de cada lección y ejecute las celdas en orden (`Shift + Enter`).

## 📚 Temario

### Módulo 0 · Preparación del laboratorio digital

| Lección | Tema | Abrir |
|---|---|---|
| 0.1 | Colab como laboratorio: Python, numpy, pandas y el estilo gráfico del curso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-00-preparacion/0.1_colab_laboratorio.ipynb) |
| 0.2 | Biopython y herramientas de línea de comandos (SARS-CoV-2, BLAST) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-00-preparacion/0.2_biopython_herramientas.ipynb) |
| 0.3 | Reproducibilidad: semillas, versiones, hashes y Git | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-00-preparacion/0.3_reproducibilidad.ipynb) |

### Módulo 1 · Biología molecular para bioinformáticos

| Lección | Tema | Abrir |
|---|---|---|
| 1.1 | El dogma central como sistema de información: entropía, compresión y robustez del código genético | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-01-biologia-molecular/1.1_dogma_central.ipynb) |
| 1.2 | Genes, genomas y marcos abiertos de lectura: un buscador de ORFs en *Mycoplasma genitalium* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-01-biologia-molecular/1.2_genes_orfs.ipynb) |
| 1.3 | Composición de secuencias: GC, GC skew y el origen de replicación de *E. coli* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-01-biologia-molecular/1.3_composicion_gc_skew.ipynb) |

### Módulo 2 · Secuencias, formatos y bases de datos

| Lección | Tema | Abrir |
|---|---|---|
| 2.1 | El idioma de los archivos: FASTA, FASTQ (Phred), GenBank, GFF/BED, SAM/BAM (CIGAR, FLAG) y VCF | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-02-formatos-bases-datos/2.1_formatos_archivos.ipynb) |
| 2.2 | Bases de datos biológicas desde Python: NCBI, Ensembl, UniProt y PDB siguiendo al gen TP53 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-02-formatos-bases-datos/2.2_bases_de_datos.ipynb) |

### Módulo 3 · Alineamiento de secuencias

| Lección | Tema | Abrir |
|---|---|---|
| 3.1 | Dot plots: ver la similitud antes de medirla (SARS-CoV-2 vs SARS-CoV-1, calmodulina) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-03-alineamiento/3.1_dot_plots.ipynb) |
| 3.2 | Programación dinámica: Needleman-Wunsch y Smith-Waterman con la matriz animada paso a paso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-03-alineamiento/3.2_programacion_dinamica.ipynb) |
| 3.3 | Matrices de sustitución (PAM, BLOSUM) y penalizaciones de huecos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-03-alineamiento/3.3_matrices_sustitucion.ipynb) |
| 3.4 | BLAST y la estadística de Karlin-Altschul: semillas, X-drop, Gumbel y E-value | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-03-alineamiento/3.4_blast_estadistica.ipynb) |

### Módulo 4 · Alineamiento múltiple, motivos y perfiles

- 4.1 MSA: alineamiento progresivo · *próximamente*
- 4.2 Motivos, PWM, entropía y sequence logos · *próximamente*
- 4.3 Modelos ocultos de Márkov y perfiles (HMMER) · *próximamente*

### Módulo 5 · Filogenética y evolución molecular

- 5.1 Distancias evolutivas y modelos de sustitución · *próximamente*
- 5.2 UPGMA y Neighbor-Joining · *próximamente*
- 5.3 Máxima verosimilitud y bootstrap · *próximamente*

### Módulo 6 · Secuenciación de nueva generación (NGS)

- 6.1 Tecnologías de secuenciación · *próximamente*
- 6.2 Calidad Phred, control de calidad y trimming · *próximamente*
- 6.3 Cobertura y la teoría de Lander-Waterman · *próximamente*

### Módulo 7 · Mapeo de lecturas

- 7.1 Transformada de Burrows-Wheeler y FM-index · *próximamente*
- 7.2 BWA, minimap2 y samtools · *próximamente*
- 7.3 Visualización de alineamientos y cobertura · *próximamente*

### Módulo 8 · Ensamblaje y anotación de genomas

- 8.1 k-mers y espectros de k-mers · *próximamente*
- 8.2 Grafos de De Bruijn · *próximamente*
- 8.3 Ensamblaje con SPAdes y evaluación con QUAST · *próximamente*
- 8.4 Anotación genómica · *próximamente*

### Módulo 9 · Detección de variantes

- 9.1 Verosimilitud de genotipos · *próximamente*
- 9.2 Pipeline con bcftools · *próximamente*
- 9.3 Anotación funcional de variantes · *próximamente*

### Módulo 10 · Genómica de poblaciones y GWAS

- 10.1 Hardy-Weinberg y deriva génica (Wright-Fisher) · *próximamente*
- 10.2 Desequilibrio de ligamiento y PCA · *próximamente*
- 10.3 GWAS: Manhattan, QQ plots y corrección múltiple · *próximamente*

### Módulo 11 · Transcriptómica (RNA-seq)

- 11.1 Cuantificación (Salmon) · *próximamente*
- 11.2 Normalización y binomial negativa · *próximamente*
- 11.3 Expresión diferencial y FDR · *próximamente*
- 11.4 Enriquecimiento funcional (GO, KEGG, GSEA) · *próximamente*

### Módulo 12 · Transcriptómica de célula única

- 12.1 Control de calidad y normalización (Scanpy) · *próximamente*
- 12.2 PCA, t-SNE y UMAP · *próximamente*
- 12.3 Clustering y genes marcadores · *próximamente*
- 12.4 Trayectorias y pseudotiempo · *próximamente*

### Módulo 13 · Epigenómica y regulación

- 13.1 ChIP-seq y ATAC-seq · *próximamente*
- 13.2 Metilación del ADN · *próximamente*
- 13.3 Descubrimiento de motivos regulatorios · *próximamente*

### Módulo 14 · Metagenómica y microbioma

- 14.1 16S rRNA y ASVs · *próximamente*
- 14.2 Diversidad alfa y beta · *próximamente*
- 14.3 Metagenómica shotgun (Kraken2) · *próximamente*

### Módulo 15 · Bioinformática estructural

- 15.1 Estructura de proteínas y PDB · *próximamente*
- 15.2 Predicción de estructura: AlphaFold/ESMFold · *próximamente*
- 15.3 Docking molecular básico · *próximamente*

### Módulo 16 · Biología de sistemas y redes

- 16.1 Redes de interacción proteína-proteína · *próximamente*
- 16.2 Redes de regulación génica y modelos ODE · *próximamente*

### Módulo 17 · Machine Learning e IA en bioinformática

- 17.1 Clasificación de secuencias con ML clásico · *próximamente*
- 17.2 Deep learning para ADN · *próximamente*
- 17.3 Modelos de lenguaje de proteínas (ESM) · *próximamente*

### Módulo 18 · Flujos reproducibles y proyecto final

- 18.1 Snakemake/Nextflow · *próximamente*
- 18.2 Proyecto integrador · *próximamente*

## 🗂️ Estructura del repositorio

```
bioinformatica-practica/
├── modulo-00-preparacion/   # notebooks de cada módulo (con resultados ya ejecutados)
├── modulo-01-biologia-molecular/
├── modulo-02-formatos-bases-datos/
├── modulo-03-alineamiento/
├── assets/                  # vistas previas GIF de las animaciones
├── utils/estilo_curso.py    # estilo gráfico común (paleta validada para daltonismo)
├── tools/                   # scripts para construir (builders/) y probar los notebooks
└── data/                    # copias de respaldo de datos pequeños
```

## 🎨 Estilo gráfico

Todas las figuras usan `utils/estilo_curso.py`: paleta categórica en orden fijo validada para daltonismo, colores de nucleótidos convencionales (siempre acompañados de su letra), títulos que enuncian la conclusión y animaciones HTML que se reproducen dentro del notebook.

## 📄 Licencia y autoría

© 2026 **Juvenal Yosa, PhD** ([juvenal.yosa@gmail.com](mailto:juvenal.yosa@gmail.com)). Código bajo [licencia MIT](LICENSE).
Material desarrollado con **Claude** (Anthropic) como copiloto.
