"""Genera los datos y figuras TikZ del capítulo 0 a partir de cálculos reales.

Ejecutar desde cualquier lugar:  python3 figuras/cap00/generar.py
Requiere numpy, matplotlib (solo para leer sus mapas de color) y Biopython.
"""
import hashlib
import time
from math import comb, sqrt
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
ROOT = OUT.parents[2]                       # .../bioinformatics
GB = ROOT / "data" / "NC_045512.2.gb"       # genoma de referencia de SARS-CoV-2


def w(name, text):
    (OUT / name).write_text(text)
    print("escrito", name)


# ------------------------------------------------------------------
# 1. Tiempos: bucle de Python vs numpy vs str.count (contenido GC)
# ------------------------------------------------------------------
def gc_loop(seq):
    n = 0
    for b in seq:
        if b == "G" or b == "C":
            n += 1
    return n / len(seq)


def gc_numpy(seq):
    a = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return np.count_nonzero((a == 71) | (a == 67)) / a.size


def gc_count(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


def best(f, x, r=3):
    t = []
    for _ in range(r):
        t0 = time.perf_counter(); f(x); t.append(time.perf_counter() - t0)
    return min(t)


rng = np.random.default_rng(2024)
rows = ["n loop numpy count"]
for e in range(3, 8):
    n = 10 ** e
    s = "".join(rng.choice(list("ACGT"), size=n))
    rows.append(f"{n} {best(gc_loop, s):.4e} {best(gc_numpy, s):.4e} "
                f"{best(gc_count, s):.4e}")
w("tiempos.dat", "\n".join(rows) + "\n")

# ------------------------------------------------------------------
# 2. GC en ventana deslizante sobre SARS-CoV-2 + banda binomial
# ------------------------------------------------------------------
from Bio import SeqIO

g = SeqIO.read(GB, "genbank")
seq = str(g.seq)
L = len(seq)
arr = np.frombuffer(seq.encode(), dtype=np.uint8)
isgc = ((arr == 71) | (arr == 67)).astype(float)
p = isgc.mean()
W, STEP = 500, 50
cs = np.concatenate([[0], np.cumsum(isgc)])
starts = np.arange(0, L - W + 1, STEP)
gcw = (cs[starts + W] - cs[starts]) / W
centers = (starts + W / 2) / 1000
sd = sqrt(p * (1 - p) / W)
outside = np.mean(np.abs(gcw - p) > 2 * sd)
w("gc_ventana.dat", "kb gc\n" + "\n".join(f"{c:.3f} {v:.4f}"
                                          for c, v in zip(centers, gcw)) + "\n")
genes = []
for f in g.features:
    if f.type == "gene":
        genes.append((f.qualifiers["gene"][0], int(f.location.start) + 1,
                      int(f.location.end)))
w("gc_resumen.txt", f"L={L}\np={p:.5f}\nW={W}\nsd={sd:.5f}\n"
  f"fuera_2sd={outside:.4f}\nmin={gcw.min():.4f}\nmax={gcw.max():.4f}\n"
  f"genes={genes}\n")

# ------------------------------------------------------------------
# 3. Luminosidad perceptual (CIELAB L*) de jet frente a viridis
# ------------------------------------------------------------------
from matplotlib import colormaps


def lstar(rgb):
    c = np.asarray(rgb)[:, :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(Y > (6 / 29) ** 3, np.cbrt(Y), Y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


t = np.linspace(0, 1, 101)
jet, vir = colormaps["jet"](t), colormaps["viridis"](t)
Lj, Lv = lstar(jet), lstar(vir)
w("luminosidad.dat", "t jet viridis\n" + "\n".join(
    f"{a:.2f} {b:.2f} {c:.2f}" for a, b, c in zip(t, Lj, Lv)) + "\n")
# franjas de color (para dibujar debajo de la gráfica)
strip = []
for name, cm_, y0 in (("jet", jet, 0), ("viridis", vir, 1)):
    for i in range(len(t) - 1):
        r, gg, b = cm_[i][:3]
        strip.append(rf"\fill[color={{rgb,1:red,{r:.3f};green,{gg:.3f};blue,{b:.3f}}}]"
                     rf" ({t[i]:.3f},{y0}) rectangle ({t[i+1]+0.002:.3f},{y0+0.8});")
w("franjas.tex", "\n".join(strip) + "\n")

# ------------------------------------------------------------------
# 4. Tabla del código genético (TikZ)
# ------------------------------------------------------------------
from Bio.Data import CodonTable

tab = CodonTable.unambiguous_dna_by_id[1]
three = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
         "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
         "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
         "Y": "Tyr", "V": "Val", "*": "Stop"}
cls = {**{a: "amarillo" for a in "AVILMFWY"}, **{a: "aqua" for a in "STNQ"},
       **{a: "azul" for a in "KRH"}, **{a: "rojo" for a in "DE"},
       **{a: "violeta" for a in "GPC"}, "*": "tinta2"}
B = "UCAG"
STAR = r"$\ast$"
cw, rh = 2.55, 0.42                       # ancho de columna, alto de fila (cm)
c = [r"\begin{tikzpicture}[font=\sffamily\scriptsize]"]
for j, b2 in enumerate(B):
    c.append(rf"\node[nt={b2.replace('U','T')},minimum width=2.3cm] at ({j*cw+cw/2+0.55},0.45)"
             rf" {{{b2}}};")
c.append(r"\node[etiqueta,anchor=south] at (5.65,0.75) {2.ª base};")
c.append(r"\node[etiqueta,rotate=90] at (-0.75,-3.4) {1.ª base};")
c.append(r"\node[etiqueta,rotate=-90] at (11.35,-3.4) {3.ª base};")
for i, b1 in enumerate(B):
    y0 = -i * 4 * rh - 0.05
    c.append(rf"\node[nt={b1.replace('U','T')},minimum height={4*rh-0.08}cm] at (0,{y0-2*rh+0.03})"
             rf" {{{b1}}};")
    for k, b3 in enumerate(B):
        y = y0 - k * rh - rh / 2
        c.append(rf"\node[text=gris] at (10.85,{y:.3f}) {{\ttfamily {b3}}};")
        for j, b2 in enumerate(B):
            cod = (b1 + b2 + b3)
            dna = cod.replace("U", "T")
            aa = "*" if dna in tab.stop_codons else tab.forward_table[dna]
            col = cls[aa]
            x = j * cw + 0.55
            extra = r",draw=verde,line width=0.9pt" if dna == "ATG" else ""
            c.append(rf"\node[anchor=west,fill={col}!18,minimum width={cw-0.12}cm,"
                     rf"minimum height={rh-0.05}cm,inner sep=2pt,rounded corners=1pt{extra}]"
                     rf" at ({x+0.06:.2f},{y:.3f}) {{\ttfamily {cod}\hspace{{4pt}}"
                     rf"\sffamily\bfseries\color{{{col}!70!black}}{three[aa]}"
                     rf"\hfill\mdseries {aa if aa != '*' else STAR}}};")
c.append(r"\end{tikzpicture}")
w("codigo_genetico.tex", "\n".join(c) + "\n")
deg = {}
for cod, aa in tab.forward_table.items():
    deg[aa] = deg.get(aa, 0) + 1

# ------------------------------------------------------------------
# 5. Seis marcos de lectura de SARS-CoV-2 (TikZ)
# ------------------------------------------------------------------
from Bio.Seq import Seq

stops = {"TAA", "TAG", "TGA"}
MINC = 100                                  # ORF mínimo (codones)
S = 11.2 / L                                # cm por nucleótido
rc = str(Seq(seq).reverse_complement())
c = [r"\begin{tikzpicture}[font=\sffamily\scriptsize]"]
frames = [("+1", seq, 0), ("+2", seq, 1), ("+3", seq, 2),
          ("$-$1", rc, 0), ("$-$2", rc, 1), ("$-$3", rc, 2)]
nlong = {}
for r, (lab, s, off) in enumerate(frames):
    y = -r * 0.55 - (0.35 if r >= 3 else 0)
    rev = s is rc
    c.append(rf"\node[anchor=east,text=tinta2] at (-0.1,{y}) {{{lab}}};")
    c.append(rf"\fill[papel] (0,{y-0.16}) rectangle (11.2,{y+0.16});")
    last, cnt = off, 0
    stop_pos = []
    for i in range(off, len(s) - 2, 3):
        if s[i:i+3] in stops:
            stop_pos.append(i)
            if (i - last) // 3 >= MINC:
                a, b = last, i + 3
                if rev:
                    a, b = L - b, L - a
                col = "azul" if not rev else "naranja"
                c.append(rf"\fill[{col}] ({a*S:.3f},{y-0.16}) rectangle ({b*S:.3f},{y+0.16});")
                cnt += 1
            last = i + 3
    nlong[lab] = (cnt, len(stop_pos))
    for i in stop_pos:
        xx = (L - i - 1.5) * S if rev else (i + 1.5) * S
        c.append(rf"\draw[tinta2!55,line width=0.15pt] ({xx:.3f},{y-0.16}) -- ({xx:.3f},{y+0.16});")
# pista de genes anotados
yg = 0.75
c.append(rf"\node[anchor=east,text=tinta2] at (-0.1,{yg}) {{genes}};")
for name, a, b in genes:
    col = "violeta" if name != "S" else "rojo"
    c.append(rf"\fill[{col}!75,rounded corners=0.5pt] ({(a-1)*S:.3f},{yg-0.14}) rectangle ({b*S:.3f},{yg+0.14});")
for name, a, b in genes:
    if name in ("ORF1ab", "S", "N", "M"):
        c.append(rf"\node[anchor=south,text=tinta] at ({(a+b)/2*S:.3f},{yg+0.14}) {{{name}}};")
for kb in range(0, 31, 5):
    x = kb * 1000 * S
    c.append(rf"\draw[base] ({x:.3f},-3.35) -- ({x:.3f},-3.45) node[below,text=gris] {{{kb}}};")
c.append(r"\draw[base] (0,-3.35) -- (11.2,-3.35);")
c.append(r"\node[text=tinta2] at (5.6,-4.0) {posición en el genoma (kb)};")
c.append(r"\end{tikzpicture}")
w("orfs.tex", "\n".join(c) + "\n")
w("orfs_resumen.txt", f"{nlong}\n")

# ------------------------------------------------------------------
# 6. RANDU frente a PCG64: proyección que revela los planos
# ------------------------------------------------------------------
def randu(seed, n):
    x, out = seed, np.empty(n)
    for i in range(n):
        x = (65539 * x) % 2**31
        out[i] = x / 2**31
    return out


N3 = 2500
ur = randu(1, N3 + 2)
up = np.random.default_rng(1).random(N3 + 2)
nh = np.array([9, -6, 1]) / np.linalg.norm([9, -6, 1])
e1 = np.cross(nh, [0, 0, 1]); e1 /= np.linalg.norm(e1)
for name, u in (("randu", ur), ("pcg", up)):
    T = np.column_stack([u[:-2], u[1:-1], u[2:]])
    xy = np.column_stack([T @ e1, T @ nh * np.linalg.norm([9, -6, 1])])
    w(f"{name}.dat", "a b\n" + "\n".join(f"{a:.4f} {b:.4f}" for a, b in xy) + "\n")
T = np.column_stack([ur[:-2], ur[1:-1], ur[2:]])
k = np.round(T @ np.array([9, -6, 1]), 6)
print("RANDU: valores distintos de 9u-6v+w:", len(np.unique(np.round(k))))

# ------------------------------------------------------------------
# 7. Monte Carlo: lecturas de 150 nt sin errores (e = 0,01)
# ------------------------------------------------------------------
Lr, err = 150, 0.01
pt = (1 - err) ** Lr
Ns = np.unique(np.round(np.logspace(1, 5, 60)).astype(int))
cols = []
for seed in (1, 2, 3, 4):
    g_ = np.random.default_rng(seed)
    ok = (g_.random((Ns[-1], Lr)) >= err).all(axis=1)
    run = np.cumsum(ok) / np.arange(1, Ns[-1] + 1)
    cols.append(run[Ns - 1])
w("montecarlo.dat", "N s1 s2 s3 s4\n" + "\n".join(
    f"{n} " + " ".join(f"{c_[i]:.5f}" for c_ in cols) for i, n in enumerate(Ns)) + "\n")
w("mc_resumen.txt", f"p={pt:.6f}\nse1e4={sqrt(pt*(1-pt)/1e4):.6f}\n"
  f"final={[round(float(c_[-1]),5) for c_ in cols]}\n")

# ------------------------------------------------------------------
# 8. Efecto avalancha de SHA-256
# ------------------------------------------------------------------
ga = np.random.default_rng(256)
base = ga.choice(list("ACGT"), size=1000)
dist = []
for _ in range(10000):
    s2 = base.copy()
    i = ga.integers(1000)
    s2[i] = ga.choice([b for b in "ACGT" if b != s2[i]])
    h1 = int.from_bytes(hashlib.sha256("".join(base).encode()).digest(), "big")
    h2 = int.from_bytes(hashlib.sha256("".join(s2).encode()).digest(), "big")
    dist.append(bin(h1 ^ h2).count("1"))
dist = np.array(dist)
ks = np.arange(100, 157)
freq = np.array([(dist == k_).mean() for k_ in ks])
binom = np.array([comb(256, int(k_)) / 2**256 for k_ in ks])
w("avalancha.dat", "k freq binom\n" + "\n".join(
    f"{k_} {f_:.5f} {b_:.5f}" for k_, b_, f_ in zip(ks, binom, freq)) + "\n")
ex = {s_: hashlib.sha256(s_.encode()).hexdigest() for s_ in ("GATTACA", "GATTACC")}
hx = [int(v, 16) for v in ex.values()]
w("avalancha_resumen.txt", f"media={dist.mean():.3f} sd={dist.std():.3f} "
  f"min={dist.min()} max={dist.max()}\n{ex}\nbits_distintos="
  f"{bin(hx[0]^hx[1]).count('1')}\n")
