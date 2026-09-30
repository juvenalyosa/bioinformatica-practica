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

| Lección | Tema | Abrir |
|---|---|---|
| 4.1 | Alineamiento múltiple: árbol guía, alineamiento progresivo y conservación del citocromo c en 14 especies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-04-msa-motivos-hmm/4.1_alineamiento_multiple.ipynb) |
| 4.2 | Motivos, PWM, entropía y sequence logos: redescubriendo Shine-Dalgarno en *E. coli* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-04-msa-motivos-hmm/4.2_motivos_pwm_logos.ipynb) |
| 4.3 | Modelos ocultos de Márkov: Viterbi, Forward-Backward, islas CpG en TP53 y perfiles HMMER | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-04-msa-motivos-hmm/4.3_modelos_ocultos_markov.ipynb) |

### Módulo 5 · Filogenética y evolución molecular

| Lección | Tema | Abrir |
|---|---|---|
| 5.1 | Distancias evolutivas y modelos de sustitución (JC69, K80, GTR, Gamma) en la polimerasa de 12 coronavirus | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-05-filogenetica/5.1_distancias_modelos.ipynb) |
| 5.2 | Árboles a partir de distancias: UPGMA y Neighbor-Joining animados, enraizado y Robinson-Foulds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-05-filogenetica/5.2_upgma_neighbor_joining.ipynb) |
| 5.3 | Máxima verosimilitud (poda de Felsenstein), bootstrap e IQ-TREE con 12 primates | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-05-filogenetica/5.3_maxima_verosimilitud_bootstrap.ipynb) |

### Módulo 6 · Secuenciación de nueva generación (NGS)

| Lección | Tema | Abrir |
|---|---|---|
| 6.1 | Tecnologías de secuenciación: Sanger, Illumina (SBS animada), Nanopore (squiggle) y PacBio HiFi | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-06-ngs/6.1_tecnologias_secuenciacion.ipynb) |
| 6.2 | Control de calidad y limpieza de lecturas reales de *E. coli*: módulos FastQC desde cero, trimming y fastp | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-06-ngs/6.2_control_calidad_trimming.ipynb) |
| 6.3 | Cobertura y la teoría de Lander-Waterman: Poisson, sesgo GC y planificación de experimentos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-06-ngs/6.3_cobertura_lander_waterman.ipynb) |

### Módulo 7 · Mapeo de lecturas

| Lección | Tema | Abrir |
|---|---|---|
| 7.1 | Transformada de Burrows-Wheeler, arreglo de sufijos y FM-index: búsqueda hacia atrás animada | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-07-mapeo/7.1_bwt_fm_index.ipynb) |
| 7.2 | BWA, minimap2 y samtools con lecturas reales del clon LTEE de *E. coli*: minimizadores, encadenamiento y MAPQ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-07-mapeo/7.2_bwa_minimap2_samtools.ipynb) |
| 7.3 | Ver para creer: pileup, vista tipo IGV, cobertura, variantes estructurales y deleciones detectables | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-07-mapeo/7.3_visualizacion_alineamientos.ipynb) |

### Módulo 8 · Ensamblaje y anotación de genomas

| Lección | Tema | Abrir |
|---|---|---|
| 8.1 | k-mers y espectros: tamaño del genoma, heterocigosidad y errores sin ensamblar (modelo tipo GenomeScope) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-08-ensamblaje/8.1_kmers_espectros.ipynb) |
| 8.2 | Grafos de De Bruijn: Euler, Hierholzer animado, unitigs, puntas, burbujas y el grafo real de SPAdes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-08-ensamblaje/8.2_grafos_de_bruijn.ipynb) |
| 8.3 | Ensamblaje con SPAdes y evaluación con QUAST: N50/NG50/auN, errores de ensamblaje y completitud | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-08-ensamblaje/8.3_spades_quast.ipynb) |
| 8.4 | Anotación genómica: ORFs, modelos de Márkov, HMM de genes, Prodigal y la anotación del ensamblaje LTEE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-08-ensamblaje/8.4_anotacion_genomica.ipynb) |

### Módulo 9 · Detección de variantes

| Lección | Tema | Abrir |
|---|---|---|
| 9.1 | Verosimilitud de genotipos: modelo de error, PL, prior, posterior, QUAL/GQ y llamada conjunta | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-09-variantes/9.1_verosimilitud_genotipos.ipynb) |
| 9.2 | Pipeline con bcftools sobre el clon LTEE: VCF, normalización, filtrado, ti/tv y evaluación tipo GIAB | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-09-variantes/9.2_pipeline_bcftools.ipynb) |
| 9.3 | Anotación funcional: consecuencias en el codón, VEP, gnomAD, SIFT/CADD, ClinVar y ACMG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-09-variantes/9.3_anotacion_variantes.ipynb) |

### Módulo 10 · Genómica de poblaciones y GWAS

| Lección | Tema | Abrir |
|---|---|---|
| 10.1 | Hardy-Weinberg y deriva génica: pruebas χ² y exacta, Wright-Fisher, Kimura y 26 poblaciones de 1000 Genomas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-10-poblaciones-gwas/10.1_hardy_weinberg_deriva.ipynb) |
| 10.2 | Desequilibrio de ligamiento y PCA: D, D′, r², el barrido de la lactasa (LCT), F_ST y ancestría | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-10-poblaciones-gwas/10.2_ligamiento_pca.ipynb) |
| 10.3 | GWAS: modelo aditivo, estratificación (el caso LCT–estatura), Bonferroni/BH, QQ y Manhattan | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-10-poblaciones-gwas/10.3_gwas.ipynb) |

### Módulo 11 · Transcriptómica (RNA-seq)

| Lección | Tema | Abrir |
|---|---|---|
| 11.1 | Cuantificación de transcritos: EM de isoformas, TPM y Salmon con lecturas reales del experimento airway | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-11-rnaseq/11.1_cuantificacion_salmon.ipynb) |
| 11.2 | Normalización y binomial negativa: factores de tamaño, dispersión y QC de 16 muestras (un probable intercambio de etiquetas) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-11-rnaseq/11.2_normalizacion_binomial_negativa.ipynb) |
| 11.3 | Expresión diferencial y FDR: GLM binomial negativo, contracción, filtrado independiente y la respuesta a dexametasona | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-11-rnaseq/11.3_expresion_diferencial_fdr.ipynb) |
| 11.4 | Enriquecimiento funcional: hipergeométrica, sesgo de longitud y GSEA con GO, KEGG y Hallmark | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-11-rnaseq/11.4_enriquecimiento_funcional.ipynb) |

### Módulo 12 · Transcriptómica de célula única

| Lección | Tema | Abrir |
|---|---|---|
| 12.1 | Control de calidad y normalización de célula única: matrices dispersas, rodilla, MAD, dobletes y HVG (PBMC 3k) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-12-celula-unica/12.1_qc_normalizacion.ipynb) |
| 12.2 | PCA, t-SNE (desde cero) y UMAP: perplejidad, vecinos y las distorsiones de los mapas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-12-celula-unica/12.2_pca_tsne_umap.ipynb) |
| 12.3 | Clustering (Louvain/Leiden) y genes marcadores (Wilcoxon): inmunofenotipo de la sangre periférica | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-12-celula-unica/12.3_clustering_marcadores.ipynb) |
| 12.4 | Trayectorias y pseudotiempo: mapas de difusión, DPT, PAGA y velocidad de ARN en la hematopoyesis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-12-celula-unica/12.4_trayectorias_pseudotiempo.ipynb) |

### Módulo 13 · Epigenómica y regulación

| Lección | Tema | Abrir |
|---|---|---|
| 13.1 | ChIP-seq y ATAC-seq con datos de ENCODE: correlación cruzada, llamado de picos tipo MACS, IDR, FRiP y patrón nucleosomal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-13-epigenomica/13.1_chipseq_atacseq.ipynb) |
| 13.2 | Metilación del ADN: islas CpG, bisulfito, valores β/M, prueba de Wald y una DMR real en el promotor de GSTP1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-13-epigenomica/13.2_metilacion_adn.ipynb) |
| 13.3 | Descubrimiento de motivos: EM (MEME), Gibbs, enriquecimiento y centralidad redescubriendo CTCF frente a JASPAR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/bioinformatica-practica/blob/main/modulo-13-epigenomica/13.3_descubrimiento_motivos.ipynb) |

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
├── modulo-04-msa-motivos-hmm/
├── modulo-05-filogenetica/
├── modulo-06-ngs/
├── modulo-07-mapeo/
├── modulo-08-ensamblaje/
├── modulo-09-variantes/
├── modulo-10-poblaciones-gwas/
├── modulo-11-rnaseq/
├── modulo-12-celula-unica/
├── modulo-13-epigenomica/
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
