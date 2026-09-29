"""Genera datos y figuras del capítulo 6 (NGS) a partir de cálculos reales.

Ejecutar desde libro/:  python3 figuras/cap06/generar.py
Escribe .dat/.tex en figuras/cap06/ e imprime las cifras que se citan
en los ejemplos del texto (única fuente de verdad).
"""
from pathlib import Path
import math
import numpy as np
from scipy import stats

OUT = Path(__file__).parent
rng = np.random.default_rng(2024)


def w(name, text):
    (OUT / name).write_text(text)


# =====================================================================
# 6.1  Costo por genoma (NHGRI, datos descargados en nhgri_costos.csv)
# =====================================================================
d = np.genfromtxt(OUT / "nhgri_costos.csv", delimiter=",", names=True)
w("costo.dat", "anio costo\n" + "\n".join(
    f"{a:.3f} {c:.2f}" for a, c in zip(d["anio"], d["costo_genoma"])))
a0, c0 = d["anio"][0], d["costo_genoma"][0]
moore = [(a, c0 * 2 ** (-(a - a0) / 2)) for a in np.linspace(a0, 2022.5, 30)]
w("moore.dat", "anio costo\n" + "\n".join(f"{a:.3f} {c:.2f}" for a, c in moore))
print(f"[costo] {a0:.1f}: {c0:,.0f} USD   final: {d['costo_genoma'][-1]:,.0f} USD"
      f"   ley de Moore en 2022: {moore[-1][1]:,.0f} USD"
      f"   factor real: {c0 / d['costo_genoma'][-1]:,.0f}x")
i08 = np.argmin(abs(d["anio"] - 2008.0)); i07 = np.argmin(abs(d["anio"] - 2007.5))
print(f"[costo] 2007.5 -> {d['costo_genoma'][i07]:,.0f};  2008 -> {d['costo_genoma'][i08]:,.0f}")

# =====================================================================
# 6.2  Phred
# =====================================================================
for Q in [10, 20, 30, 40]:
    print(f"[phred] Q={Q}: p={10 ** (-Q / 10):g}  exactitud={1 - 10 ** (-Q / 10):.4%}")

# Registro FASTQ de ejemplo (Phred+33)
seq = "".join(rng.choice(list("ACGT"), 24))
qs0 = [34, 35, 37, 37, 38, 38, 37, 36, 38, 37, 36, 35,
       36, 33, 32, 30, 31, 27, 25, 22, 20, 14, 11, 2]
qual = "".join(chr(q + 33) for q in qs0)
print("[fastq] seq =", seq, " qual =", qual)
w("fastq_ej.tex", "\\def\\capseisseq{" + seq + "}\n")
qs = [ord(c) - 33 for c in qual]
print("[fastq] Q =", qs)
ps = [10 ** (-q / 10) for q in qs]
print(f"[fastq] errores esperados = {sum(ps):.2f} en {len(qs)} bases")

# Errores esperados en una lectura de 150 pb, todo Q30 y todo Q20
for Q in (20, 30):
    print(f"[phred] 150 pb a Q{Q}: E[errores]={150 * 10 ** (-Q / 10):.3f};"
          f" P(sin error)={(1 - 10 ** (-Q / 10)) ** 150:.3f}")

# ---------- Perfil de calidad por posición tipo FastQC (simulado) ----------
L, n = 150, 20000
pos = np.arange(1, L + 1)
mu = 36.5 - 0.5 * np.exp(-(pos - 1) / 2.0) - 6.0 * (pos / L) ** 3   # caída 3'
sd = 1.2 + 4.0 * (pos / L) ** 2
Qsim = np.clip(np.round(rng.normal(mu[None, :], sd[None, :], (n, L))
                        - rng.exponential(1.0, (n, L)) * (pos / L) ** 2 * 4), 2, 41)
# agrupar como FastQC: 1..9 individuales, luego bloques de 5
bins = [(i, i) for i in range(1, 10)] + [(i, min(i + 4, L)) for i in range(10, L + 1, 5)]
lines = []
for k, (a, b) in enumerate(bins):
    v = Qsim[:, a - 1:b].ravel()
    p10, q1, med, q3, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    lines.append((k + 1, f"{a}" if a == b else f"{a}-{b}", p10, q1, med, q3, p90, v.mean()))
tex = []
for k, lab, p10, q1, med, q3, p90, m in lines:
    tex.append(
        rf"\addplot[fill=amarillo!70,draw=tinta2,thin,boxplot prepared={{draw position={k},"
        rf"lower whisker={p10:.1f},lower quartile={q1:.1f},median={med:.1f},"
        rf"upper quartile={q3:.1f},upper whisker={p90:.1f},box extend=0.7}},"
        rf"boxplot/every median/.style={{draw=rojooscuro,thick}}] coordinates {{}};")
w("fastqc_cajas.tex", "\n".join(tex))
w("fastqc_media.dat", "x media\n" + "\n".join(f"{k} {m:.2f}" for k, *_, m in lines))
w("fastqc_ticks.tex",
  "\\pgfplotsset{capseis/ticks/.style={xtick={" + ",".join(str(k) for k, *_ in lines[::3]) + "},\nxticklabels={"
  + ",".join(lab.split("-")[0] for _, lab, *_ in lines[::3]) + "}}}\n")
print(f"[fastqc] {len(bins)} cajas; mediana última={lines[-1][4]:.1f};"
      f" p10 última={lines[-1][2]:.1f}")

# ---------- Ventana deslizante (Trimmomatic SLIDINGWINDOW:4:20) ----------
qread = [38, 38, 37, 38, 36, 37, 35, 36, 34, 35, 33, 34, 32, 30, 31, 28,
         29, 26, 24, 27, 22, 19, 21, 17, 15, 18, 12, 10, 8, 6]
wlen, thr = 4, 20
cut = None
means = []
for i in range(len(qread) - wlen + 1):
    mwin = sum(qread[i:i + wlen]) / wlen
    means.append(mwin)
    if cut is None and mwin < thr:
        cut = i
# Trimmomatic conserva hasta el inicio de la ventana fallida y además
# recorta bases finales de baja calidad dentro de ella (aquí: simple).
print(f"[ventana] primera ventana con media<{thr}: inicio en posición {cut + 1}"
      f" (media={means[cut]:.2f}); se conservan {cut} bases")
print("[ventana] medias:", [round(x, 2) for x in means])
w("ventana_q.dat", "pos q\n" + "\n".join(f"{i + 1} {q}" for i, q in enumerate(qread)))
w("ventana_m.dat", "pos m\n" + "\n".join(
    f"{i + 2.5} {m:.2f}" for i, m in enumerate(means)))
w("ventana_corte.tex", rf"\def\capseiscorte{{{cut + 0.5}}}\def\capseisventana{{{cut + 1}}}")

# Recorte por calidad de BWA / cutadapt (-q 20): suma acumulada desde 3'
def cutadapt_trim(q, thr):
    s, best, idx = 0, 0, len(q)
    for i in range(len(q) - 1, -1, -1):
        s += thr - q[i]
        if s < 0:
            break
        if s > best:
            best, idx = s, i
    return idx
print(f"[cutadapt -q 20] conserva {cutadapt_trim(qread, 20)} bases")

# =====================================================================
# 6.3  Lander-Waterman
# =====================================================================
def lw(c, sigma=1.0, N=None):
    return dict(cubierta=1 - math.exp(-c), islas_por_GL=c * math.exp(-c * sigma))

cs = np.linspace(0, 8, 161)
w("lw_frac.dat", "c f\n" + "\n".join(f"{c:.3f} {1 - math.exp(-c):.5f}" for c in cs))
w("lw_islas.dat", "c s1 s08 s05\n" + "\n".join(
    f"{c:.3f} {c * math.exp(-c):.5f} {c * math.exp(-0.8 * c):.5f}"
    f" {c * math.exp(-0.5 * c):.5f}" for c in cs))

# Simulación: G=1e6, L=500, posiciones de inicio uniformes (genoma lineal)
G, Lr = 1_000_000, 500
sim = []
for c in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]:
    fr, isl = [], []
    for rep in range(5):
        N = int(round(c * G / Lr))
        st = np.sort(rng.integers(0, G - Lr + 1, N))
        cov = np.zeros(G + 1, dtype=np.int32)
        np.add.at(cov, st, 1)
        np.add.at(cov, st + Lr, -1)
        depth = np.cumsum(cov)[:G]
        fr.append((depth > 0).mean())
        # islas (sigma=1): nueva isla si el siguiente inicio > fin del bloque actual
        ends = np.maximum.accumulate(st + Lr)
        isl.append(1 + np.sum(st[1:] >= ends[:-1]))
    sim.append((c, np.mean(fr), np.mean(isl) / (G / Lr)))
w("lw_sim.dat", "c f islas\n" + "\n".join(f"{c} {f:.5f} {i:.5f}" for c, f, i in sim))
for c, f, i in sim:
    print(f"[LW sim] c={c}: cubierta={f:.4f} (teo {1 - math.exp(-c):.4f});"
          f" islas/(G/L)={i:.4f} (teo {c * math.exp(-c):.4f})")

# Ejemplo resuelto: bacteria de 5 Mb, lecturas de 150 pb, T=30 -> sigma=0.8
G, Lr, T = 5_000_000, 150, 30
sigma = 1 - T / Lr
for c in (1, 5, 10):
    N = c * G / Lr
    unc = G * math.exp(-c)
    isl = N * math.exp(-c * sigma)
    lon = Lr * ((math.exp(c * sigma) - 1) / c + 1 - sigma)
    print(f"[LW ej] c={c}: N={N:,.0f} no_cubiertas={unc:,.1f} islas={isl:,.1f}"
          f" long_media_isla={lon:,.0f} pb")

# Profundidad necesaria: P(D>=k) >= 0.99
def cmin(k, prob=0.99, r=None):
    c = k * 0.5
    while True:
        if r is None:
            ok = stats.poisson.sf(k - 1, c) >= prob
        else:
            ok = stats.nbinom.sf(k - 1, r, r / (r + c)) >= prob
        if ok:
            return c
        c += 0.01
for k in (10, 20):
    print(f"[profundidad] P(D>={k})>=0.99  Poisson: c>={cmin(k):.2f}"
          f"   NB(r=10): c>={cmin(k, r=10):.2f}   NB(r=5): c>={cmin(k, r=5):.2f}")
print(f"[profundidad] c=30: P(D<10) Poisson={stats.poisson.cdf(9, 30):.2e}"
      f"  NB r=10: {stats.nbinom.cdf(9, 10, 10 / 40):.4f}")
print(f"[profundidad] c=30 Poisson: fracción con D=0: {math.exp(-30):.2e}")

# Histograma de profundidad: Poisson(30) vs NB con la misma media
c, r = 30, 10
ks = np.arange(0, 81)
w("prof.dat", "k pois nb\n" + "\n".join(
    f"{k} {stats.poisson.pmf(k, c):.6f} {stats.nbinom.pmf(k, r, r / (r + c)):.6f}"
    for k in ks))
# simulación con sesgo GC: tasa gamma -> mezcla Poisson-gamma
lam = rng.gamma(r, c / r, 200_000)
dsim = rng.poisson(lam)
h = np.bincount(dsim, minlength=81)[:81] / len(dsim)
w("prof_sim.dat", "k f\n" + "\n".join(f"{k} {h[k]:.6f}" for k in ks[::2]))
print(f"[NB] var teórica={c + c * c / r:.1f}  var sim={dsim.var():.1f}")

# Sesgo GC: curva unimodal simulada (cobertura relativa vs GC del fragmento)
gc = np.linspace(0.2, 0.8, 61)
rel = np.exp(-((gc - 0.48) / 0.14) ** 2 / 2)
rel /= rel[np.argmin(abs(gc - 0.48))]
gcs = rng.uniform(0.22, 0.78, 700)
obs = np.exp(-((gcs - 0.48) / 0.14) ** 2 / 2) * rng.gamma(20, 1 / 20, 700)
w("gc_curva.dat", "gc rel\n" + "\n".join(f"{g:.3f} {v:.4f}" for g, v in zip(gc, rel)))
w("gc_puntos.dat", "gc rel\n" + "\n".join(f"{g:.3f} {v:.4f}" for g, v in zip(gcs, obs)))

# Esquema de lecturas sobre un genoma (para la figura de cobertura)
Gs, Ls, Ns = 120, 14, 22
st = np.sort(rng.integers(0, Gs - Ls + 1, Ns))
rows_end = []
tk = [r"\begin{tikzpicture}[x=0.95mm,y=1mm]"]
for s in st:
    for ri, e in enumerate(rows_end):
        if s > e + 1:
            rows_end[ri] = s + Ls; break
    else:
        rows_end.append(s + Ls); ri = len(rows_end) - 1
    col = "azul" if ri % 2 == 0 else "aqua"
    tk.append(rf"\draw[{col},line width=2.2pt,-{{Stealth[length=1.6mm]}}] "
              rf"({s},{-4 - 3 * ri}) -- ({s + Ls},{-4 - 3 * ri});")
depth = np.zeros(Gs, int)
for s in st:
    depth[s:s + Ls] += 1
nrows = len(rows_end)
base = -4 - 3 * nrows - 14
tk.append(rf"\fill[tinta2] (0,-1.2) rectangle ({Gs},0);")
tk.append(r"\node[etiqueta,anchor=east] at (-1,-0.6) {genoma};")
path = " -- ".join(f"({i},{base + 2.2 * d_}) -- ({i + 1},{base + 2.2 * d_})"
                   for i, d_ in enumerate(depth))
tk.append(rf"\fill[azul!25] (0,{base}) -- {path} -- ({Gs},{base}) -- cycle;")
tk.append(rf"\draw[azulprofundo,thick] {path};")
tk.append(rf"\draw[base] (0,{base}) -- ({Gs},{base});")
mean = depth.mean()
ml = f"{mean:.2f}".replace(".", "{,}")
tk.append(rf"\draw[naranja,dashed,thick] (0,{base + 2.2 * mean:.2f}) -- "
          rf"({Gs},{base + 2.2 * mean:.2f}) node[above left,etiqueta,"
          rf"text=naranja!80!black] {{$\bar c={ml}$}};")
for k in range(0, depth.max() + 1, 2):
    tk.append(rf"\draw[base] (-0.8,{base + 2.2 * k}) -- (0,{base + 2.2 * k});"
              rf"\node[etiqueta,anchor=east] at (-1,{base + 2.2 * k}) {{{k}}};")
tk.append(rf"\node[etiqueta,rotate=90] at (-6,{base + 6}) {{profundidad}};")
tk.append(rf"\node[etiqueta,anchor=east] at (-1,{-4 - 1.5 * (nrows - 1)}) {{lecturas}};")
# huecos (profundidad 0)
zeros = np.where(depth == 0)[0]
if len(zeros):
    runs = np.split(zeros, np.where(np.diff(zeros) > 1)[0] + 1)
    for rr in runs:
        tk.append(rf"\fill[rojo!25] ({rr[0]},{base - 2}) rectangle ({rr[-1] + 1},0.8);")
        tk.append(rf"\node[etiqueta,text=rojooscuro] at ({(rr[0] + rr[-1] + 1) / 2},{base - 4.5}) {{hueco}};")
tk.append(r"\end{tikzpicture}")
w("lecturas.tex", "\n".join(tk))
print(f"[esquema] N={Ns} L={Ls} G={Gs}: c={Ns * Ls / Gs:.2f}, media={mean:.2f},"
      f" huecos={int((depth == 0).sum())} pb, filas={nrows}")

# Plataformas (órdenes de magnitud; ver texto para las fuentes)
print("[ok] figuras generadas en", OUT)

# =====================================================================
# Anatomía de un registro FASTQ (figura 6.2.x)
# =====================================================================
ESC = {"#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&", "_": r"\_",
       "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
       "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
       "<": r"\textless{}", ">": r"\textgreater{}", '"': r"\textquotedbl{}"}
qs = [ord(ch) - 33 for ch in qual]
dx = 4.45
f = [r"\begin{tikzpicture}[x=1mm,y=1mm,font=\sffamily\scriptsize]"]
f.append(r"\node[anchor=west,font=\ttfamily\small,text=tinta,align=left] at (-14,34)"
         r" {@lectura\_0001 1:N:0:ACGTAC};")
f.append(r"\node[anchor=east,etiqueta,text=gris] at (-15,34) {1};")
f.append(r"\node[anchor=east,etiqueta,text=gris] at (-15,26) {2};")
f.append(r"\node[anchor=east,etiqueta,text=gris] at (-15,20) {3};")
f.append(r"\node[anchor=east,etiqueta,text=gris] at (-15,14) {4};")
f.append(r"\node[anchor=west,font=\ttfamily\small,text=tinta] at (-14,20) {+};")
for i, (b, ch, q) in enumerate(zip(seq, qual, qs)):
    x = i * dx
    col = "verde" if q >= 30 else ("amarillo" if q >= 20 else "rojo")
    f.append(rf"\node[nt={b},minimum size=4.2mm] at ({x},26) {{{b}}};")
    f.append(rf"\node[draw=rejilla,fill=papel!60,minimum size=4.2mm,inner sep=0pt,"
             rf"font=\ttfamily\bfseries\small,text=tinta] (q{i}) at ({x},14) {{{ESC.get(ch, ch)}}};")
    f.append(rf"\node[text=tinta2] at ({x},7) {{{ord(ch)}}};")
    f.append(rf"\node[text={col}!70!black,font=\sffamily\scriptsize\bfseries] at ({x},1) {{{q}}};")
    f.append(rf"\fill[{col}!75] ({x - 1.7},-26) rectangle ({x + 1.7},{-26 + 0.55 * q});")
f.append(r"\node[anchor=east,etiqueta] at (-4,7) {ASCII};")
f.append(r"\node[anchor=east,etiqueta] at (-4,1) {$Q$};")
f.append(r"\node[anchor=east,etiqueta] at (-4,-15) {perfil de $Q$};")
f.append(rf"\draw[base,thin] (-3,-26) -- ({(len(qs) - 1) * dx + 3},-26);")
for qq in (10, 20, 30):
    f.append(rf"\draw[tinta2!60,dashed] (-3,{-26 + 0.55 * qq}) -- ({(len(qs) - 1) * dx + 3},{-26 + 0.55 * qq});"
             rf"\node[etiqueta,anchor=west] at ({(len(qs) - 1) * dx + 3.5},{-26 + 0.55 * qq}) {{Q{qq}}};")
f.append(r"\end{tikzpicture}")
w("fastq_anatomia.tex", "\n".join(f))
