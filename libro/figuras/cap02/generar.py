"""Genera figuras y datos del capítulo 2 (formatos y bases de datos).

Usa el genoma de SARS-CoV-2 y las respuestas de API guardadas en ../../../data.
"""
import json, math, collections
from pathlib import Path
import numpy as np
from Bio import SeqIO

OUT = Path(__file__).parent
DATA = OUT.parents[2] / "data"
API = DATA / "api_cache"
ref = str(SeqIO.read(DATA / "NC_045512.2.gb", "genbank").seq)
S0 = 21563                                   # inicio (1-based) del gen S


def w(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")


def tex_char(c):
    esc = {"#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
    return esc.get(c, c)


# ------------------------------------------------------------------
# 1. Simulación de calidades (mismo modelo que el notebook 2.1)
# ------------------------------------------------------------------
rng = np.random.default_rng(21)
L, N = 150, 4000


def simulate_quality(n, length, rng):
    pos = np.arange(length)
    mean_q = 37 - 12 * (pos / length) ** 2
    cl = rng.normal(0, 2.5, size=(n, 1))
    noise = rng.normal(0, 2.0, size=(n, length))
    q = mean_q + cl + noise
    bad = rng.random(n) < 0.06
    q[bad] -= np.clip((pos - 100) * 0.4, 0, None)
    return np.clip(np.round(q), 2, 41).astype(int)


Q = simulate_quality(N, L, rng)
P = 10 ** (-Q / 10)
err = rng.random(Q.shape) < P
EE = P.sum(1)
pct = np.percentile(Q, [10, 25, 50, 75, 90], axis=0)
with open(OUT / "calidad_ciclo.dat", "w") as fh:
    fh.write("ciclo p10 p25 p50 p75 p90 media\n")
    for i in range(L):
        fh.write(f"{i+1} " + " ".join(f"{v:.1f}" for v in pct[:, i])
                 + f" {Q[:, i].mean():.2f}\n")
# calibración: EE predicho vs errores observados, por grupos
edges = np.array([0, .1, .15, .2, .3, .5, .75, 1, 1.5, 2, 3, 5, 20])
idx = np.digitize(EE, edges) - 1
obs = err.sum(1)
with open(OUT / "ee_calib.dat", "w") as fh:
    fh.write("ee obs n\n")
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() >= 10:
            fh.write(f"{EE[m].mean():.3f} {obs[m].mean():.3f} {m.sum()}\n")
stats = {
    "err_rate": float(err.mean()), "err_per_read": float(err.sum(1).mean()),
    "ee_mean": float(EE.mean()), "frac_ee_le1": float((EE <= 1).mean()),
    "q_first": float(Q[:, 0].mean()), "q_last": float(Q[:, -1].mean()),
}

# ------------------------------------------------------------------
# 2. Registro FASTQ con barras de calidad
# ------------------------------------------------------------------
n = 30
seq = list(ref[S0 - 1:S0 - 1 + n])
qr = np.random.default_rng(5)
qv = np.clip(np.round(38 - 14 * (np.arange(n) / n) ** 2
                      + qr.normal(0, 1.8, n)), 2, 41).astype(int)
qv[22] = 8                                    # una base dudosa ...
seq[22] = "G" if seq[22] != "G" else "C"      # ... que además es un error
qs = "".join(chr(q + 33) for q in qv)
dx = 0.27
c = [r"\begin{tikzpicture}[x=1cm,y=1cm]"]
lab = [(0, "1", r"@ + identificador"), (-0.55, "2", "secuencia"),
       (-1.1, "3", r"separador +"), (-1.65, "4", "calidades")]
for y, k, t in lab:
    c.append(rf"\node[etiqueta,anchor=east] at (-0.25,{y}) "
             rf"{{\textbf{{{k}}}\enspace {t}}};")
c.append(r"\node[anchor=west,font=\ttfamily\small,text=azulprofundo,inner sep=0pt] "
         r"at (-0.12,0) {@SIM:1:FC7:1:1101:1234:5678 1:N:0:ATCACG};")
for i, b in enumerate(seq):
    c.append(rf"\node[font=\ttfamily\bfseries\small,text=nuc{b}] at ({i*dx},-0.55) {{{b}}};")
c.append(r"\node[anchor=west,font=\ttfamily\small,inner sep=0pt] at (-0.12,-1.1) {+};")
for i, ch in enumerate(qs):
    c.append(rf"\node[font=\ttfamily\small,text=tinta] at ({i*dx},-1.65) {{{tex_char(ch)}}};")
# barras
y0, sc = -4.6, 0.055
c.append(rf"\fill[aqua!8] ({-0.2},{y0+30*sc}) rectangle ({(n-1)*dx+0.2},{y0+42*sc});")
c.append(rf"\fill[amarillo!10] ({-0.2},{y0+20*sc}) rectangle ({(n-1)*dx+0.2},{y0+30*sc});")
c.append(rf"\fill[rojo!8] ({-0.2},{y0}) rectangle ({(n-1)*dx+0.2},{y0+20*sc});")
for i, q in enumerate(qv):
    col = "azul" if q >= 30 else ("amarillo" if q >= 20 else "rojo")
    c.append(rf"\fill[{col}] ({i*dx-0.1},{y0}) rectangle ({i*dx+0.1},{y0+q*sc});")
    c.append(rf"\draw[rejilla,densely dotted] ({i*dx},-1.85) -- ({i*dx},{y0+q*sc+0.05});")
for qq in (10, 20, 30, 40):
    c.append(rf"\draw[base,thin] ({-0.2},{y0+qq*sc}) -- ({(n-1)*dx+0.2},{y0+qq*sc});")
    c.append(rf"\node[etiqueta,anchor=east] at (-0.2,{y0+qq*sc}) {{$Q={qq}$}};")
c.append(rf"\draw[base] ({-0.2},{y0}) -- ({(n-1)*dx+0.2},{y0});")
i = 22
c.append(rf"\draw[rojooscuro,thick,rounded corners=1pt] ({i*dx-0.14},-0.78) rectangle ({i*dx+0.14},-0.32);")
c.append(rf"\draw[rojooscuro,thick,rounded corners=1pt] ({i*dx-0.14},-1.88) rectangle ({i*dx+0.14},-1.42);")
c.append(rf"\node[etiqueta,text=rojooscuro,anchor=west,align=left] at ({(n-1)*dx+0.35},{y0+0.35}) "
         rf"{{base 23: \texttt{{{tex_char(qs[22])}}}\\$Q=8$, $p\approx0{{,}}16$\\(y es un error)}};")
c.append(rf"\node[etiqueta,anchor=west,align=left] at ({(n-1)*dx+0.35},{y0+1.75}) "
         rf"{{\textcolor{{azul}}{{$\blacksquare$}} $Q\ge30$\\\textcolor{{amarillo}}{{$\blacksquare$}} $20\le Q<30$\\"
         rf"\textcolor{{rojo}}{{$\blacksquare$}} $Q<20$}};")
c.append(r"\end{tikzpicture}")
w("fastq_registro.tex", c)
stats["fastq_seq"] = "".join(seq)
stats["fastq_qual"] = qs
stats["fastq_q"] = qv.tolist()
stats["fastq_EE"] = float((10 ** (-qv / 10)).sum())
stats["fastq_P0"] = float(np.prod(1 - 10 ** (-qv / 10)))

# ------------------------------------------------------------------
# 3. Tabla ASCII Phred+33 y rangos de codificación
# ------------------------------------------------------------------
cm = [(0xCD, 0xE2, 0xFB), (0x86, 0xB6, 0xEF), (0x2A, 0x78, 0xD6),
      (0x1C, 0x5C, 0xAB), (0x0D, 0x36, 0x6B)]


def cmap(t):
    t = min(max(t, 0), 1) * (len(cm) - 1)
    k = min(int(t), len(cm) - 2)
    f = t - k
    return "".join(f"{round(cm[k][j] + f*(cm[k+1][j]-cm[k][j])):02X}" for j in range(3))


c = [r"\begin{tikzpicture}[x=1cm,y=1cm]"]
ncol, s = 14, 0.74
for q in range(42):
    r, k = divmod(q, ncol)
    col = cmap(q / 41)
    txt = "white" if q > 18 else "tinta"
    c.append(rf"\definecolor{{capdosQ{q}}}{{HTML}}{{{col}}}")
    c.append(rf"\fill[capdosQ{q},rounded corners=1.5pt] ({k*s},{-r*s}) rectangle ({k*s+s*0.93},{-r*s-s*0.93});")
    c.append(rf"\node[font=\ttfamily\bfseries,text={txt}] at ({k*s+s*0.465},{-r*s-s*0.38}) {{{tex_char(chr(q+33))}}};")
    c.append(rf"\node[font=\sffamily\tiny,text={txt}] at ({k*s+s*0.465},{-r*s-s*0.75}) {{{q}\,/\,{q+33}}};")
c.append(rf"\node[etiqueta,anchor=west] at (0,0.3) {{carácter \texttt{{chr(Q+33)}}; abajo: $Q$\,/\,código ASCII}};")
# rangos
base_y = -4.3
xa = lambda a: (a - 33) * (ncol * s - 0.1) / (126 - 33)
c.append(rf"\draw[base] ({xa(33)},{base_y}) -- ({xa(126)},{base_y});")
for a in (33, 59, 64, 74, 104, 126):
    c.append(rf"\draw[base] ({xa(a)},{base_y}) -- ++(0,-0.1) node[below,etiqueta] {{{a}}};")
c.append(rf"\node[etiqueta] at ({xa(80)},{base_y-0.55}) {{código ASCII}};")
rows = [("Sanger / Illumina 1.8+ (Phred+33)", 33, 126, 33, 74, "azul"),
        ("Solexa (Solexa+64)", 59, 126, 59, 104, "naranja"),
        ("Illumina 1.3--1.7 (Phred+64)", 64, 126, 64, 104, "violeta")]
for j, (name, a, b, ta, tb, col) in enumerate(rows):
    y = base_y + 0.35 + 0.45 * (2 - j)
    c.append(rf"\fill[{col}!18] ({xa(a)},{y-0.13}) rectangle ({xa(b)},{y+0.13});")
    c.append(rf"\fill[{col}] ({xa(ta)},{y-0.13}) rectangle ({xa(tb)},{y+0.13});")
    c.append(rf"\node[etiqueta,anchor=west,text=white,font=\sffamily\tiny\bfseries] at ({xa(ta)+0.05},{y}) {{{name}}};")
c.append(r"\end{tikzpicture}")
w("ascii_phred.tex", c)

# ------------------------------------------------------------------
# 4. CIGAR sobre la referencia
# ------------------------------------------------------------------
W0 = S0 + 40                                  # ventana (1-based)
Wn = 30
win = ref[W0 - 1:W0 - 1 + Wn]
CONS = {"M": (1, 1), "I": (1, 0), "D": (0, 1), "N": (0, 1), "S": (1, 0),
        "H": (0, 0), "=": (1, 1), "X": (1, 1)}
import re
pc = lambda s: [(int(a), b) for a, b in re.findall(r"(\d+)([MIDNSHX=])", s)]
reads = [("r1", 3, "12M", {7}),
         ("r2", 6, "3S9M2I8M", set()),
         ("r3", 2, "10M3D12M", set()),
         ("r4", 4, "5M14N6M", set())]
dx = 0.3
c = [r"\begin{tikzpicture}[x=1cm,y=1cm]"]
for j, b in enumerate(win):
    c.append(rf"\node[nt={b},minimum size=4.2mm,font=\ttfamily\bfseries\scriptsize] at ({j*dx},0) {{{b}}};")
    if (j + 1) % 5 == 0 or j == 0:
        c.append(rf"\node[etiqueta,font=\sffamily\tiny] at ({j*dx},0.42) {{{W0 + j}}};")
c.append(r"\node[etiqueta,anchor=east] at (-0.3,0) {referencia};")
mr = np.random.default_rng(4)
summ = []
for k, (nm, p, cig, mism) in enumerate(reads):
    y = -0.75 - 0.72 * k
    r = p - 1               # índice 0-based en la ventana
    qpos = 0
    ops = pc(cig)
    x_first = None
    for nn, op in ops:
        if op == "S":
            for t in range(nn):
                xx = (r - nn + t) * dx
                bb = "ACGT"[mr.integers(4)]
                c.append(rf"\node[fill=papel,text=gris,minimum size=4.0mm,inner sep=0pt,rounded corners=1pt,font=\ttfamily\scriptsize] at ({xx},{y}) {{{bb.lower()}}};")
            qpos += nn
        elif op in "M=X":
            for t in range(nn):
                bb = win[r]
                if qpos in mism:
                    bb = "T" if bb != "T" else "C"
                    c.append(rf"\node[nt={bb},minimum size=4.0mm,font=\ttfamily\bfseries\scriptsize,draw=tinta,line width=0.9pt] at ({r*dx},{y}) {{{bb}}};")
                else:
                    c.append(rf"\node[fill={('nuc'+bb)}!22,text=nuc{bb}!70!black,minimum size=4.0mm,inner sep=0pt,rounded corners=1pt,font=\ttfamily\bfseries\scriptsize] at ({r*dx},{y}) {{{bb}}};")
                if x_first is None: x_first = r
                r += 1; qpos += 1
        elif op == "I":
            xx = (r - 0.5) * dx
            c.append(rf"\draw[violeta,line width=1.6pt] ({xx},{y-0.22}) -- ({xx},{y+0.22});")
            c.append(rf"\node[etiqueta,text=violeta,font=\sffamily\tiny\bfseries] at ({xx},{y+0.38}) {{+{nn}}};")
            qpos += nn
        elif op == "D":
            for t in range(nn):
                c.append(rf"\node[font=\ttfamily\bfseries\scriptsize,text=naranja] at ({r*dx},{y}) {{-}};")
                r += 1
        elif op == "N":
            c.append(rf"\draw[aqua,thick] ({(r-0.5)*dx},{y}) .. controls ({(r+nn/2)*dx},{y+0.35}) .. ({(r+nn-0.5)*dx},{y});")
            c.append(rf"\node[etiqueta,text=aqua!60!black,font=\sffamily\tiny] at ({(r+nn/2-0.5)*dx},{y+0.08}) {{intrón ({nn}N)}};")
            r += nn
    qlen = sum(nn for nn, op in ops if CONS[op][0])
    span = sum(nn for nn, op in ops if CONS[op][1])
    pos1 = W0 + p - 1
    summ.append((nm, pos1, cig, qlen, span, pos1 + span - 1))
    c.append(rf"\node[etiqueta,anchor=east] at (-0.3,{y}) {{\textbf{{{nm}}}}};")
    c.append(rf"\node[etiqueta,anchor=west,align=left,font=\sffamily\tiny] at ({Wn*dx+0.05},{y}) "
             rf"{{\texttt{{{cig}}}\\$\ell_q={qlen}$, $\ell_r={span}$}};")
c.append(r"\end{tikzpicture}")
w("cigar.tex", c)
stats["cigar"] = summ

# ------------------------------------------------------------------
# 5. FLAG bit a bit
# ------------------------------------------------------------------
FB = [(1, "pareada"), (2, "par correcto"), (4, "no alineada"), (8, "pareja no al."),
      (16, "hebra $-$"), (32, "pareja $-$"), (64, "R1"), (128, "R2"),
      (256, "secundaria"), (512, "falla QC"), (1024, "duplicado"), (2048, "suplement.")]
ex = [(99, "R1 de un par típico"), (147, "R2 de ese par"), (83, "R1 en hebra $-$"),
      (163, "R2 en hebra $+$"), (4, "no alineada"), (1171, "R2 duplicado"),
      (2064, "suplementaria, hebra $-$")]
c = [r"\begin{tikzpicture}[x=1cm,y=1cm]"]
s = 0.62
for k, (b, name) in enumerate(reversed(FB)):
    c.append(rf"\node[etiqueta,rotate=60,anchor=west,font=\sffamily\tiny] at ({k*s},0.12) {{{name}}};")
    c.append(rf"\node[etiqueta,font=\sffamily\tiny\bfseries] at ({k*s},-0.18) {{{b}}};")
for r, (f, d) in enumerate(ex):
    y = -0.6 - r * 0.6
    c.append(rf"\node[anchor=east,font=\sffamily\small\bfseries,text=tinta] at (-0.45,{y}) {{{f}}};")
    for k, (b, _) in enumerate(reversed(FB)):
        on = bool(f & b)
        st = "fill=azul,text=white" if on else "fill=rejilla!70,text=gris"
        c.append(rf"\node[{st},minimum width={s*0.9}cm,minimum height=0.4cm,rounded corners=1.5pt,"
                 rf"font=\ttfamily\scriptsize,inner sep=0pt] at ({k*s},{y}) {{{int(on)}}};")
    terms = " + ".join(str(b) for b, _ in FB if f & b) or "0"
    c.append(rf"\node[etiqueta,anchor=west,align=left,font=\sffamily\tiny] at ({11*s+0.4},{y}) {{{terms}\\\textcolor{{gris}}{{{d}}}}};")
c.append(r"\end{tikzpicture}")
w("flag.tex", c)

# ------------------------------------------------------------------
# 6. Binning UCSC/BAI: ejemplo
# ------------------------------------------------------------------
def reg2bin(beg, end):
    end -= 1
    if beg >> 14 == end >> 14: return ((1 << 15) - 1) // 7 + (beg >> 14)
    if beg >> 17 == end >> 17: return ((1 << 12) - 1) // 7 + (beg >> 17)
    if beg >> 20 == end >> 20: return ((1 << 9) - 1) // 7 + (beg >> 20)
    if beg >> 23 == end >> 23: return ((1 << 6) - 1) // 7 + (beg >> 23)
    if beg >> 26 == end >> 26: return ((1 << 3) - 1) // 7 + (beg >> 26)
    return 0
stats["bin_tp53"] = reg2bin(7668420, 7687490)
stats["bin_read"] = reg2bin(7674219, 7674369)
stats["nbins"] = sum(8 ** l for l in range(6))

# ------------------------------------------------------------------
# 7. Verosimilitudes de genotipo (VCF PL)
# ------------------------------------------------------------------
e, nr, na = 0.01, 6, 4
lik = {"0/0": (1-e)**nr * e**na, "0/1": 0.5**(nr+na), "1/1": e**nr * (1-e)**na}
m = max(lik.values())
stats["PL"] = {g: round(-10*math.log10(v/m)) for g, v in lik.items()}
stats["lik"] = {g: v for g, v in lik.items()}

# ------------------------------------------------------------------
# 10. Modelo génico de TP53 (Ensembl, GRCh38)
# ------------------------------------------------------------------
ens = json.load(open(API / "ensembl_lookup_ENSG00000141510.json"))
can = next(t for t in ens["Transcript"] if t.get("is_canonical"))
ex = sorted(can["Exon"], key=lambda e: -e["start"])
cs, ce = can["Translation"]["start"], can["Translation"]["end"]
g0, g1 = can["start"], can["end"]
X = lambda v: (v - g0) / (g1 - g0) * 11.6
c = [r"\begin{tikzpicture}[x=1cm,y=1cm]"]
c.append(rf"\draw[gris,thick] ({X(g0)},0) -- ({X(g1)},0);")
for k in range(1, 16):
    xx = k * 11.6 / 16
    c.append(rf"\draw[-{{Stealth[length=1.4mm]}},gris] ({xx+0.1},0) -- ({xx-0.1},0);")
for i, e in enumerate(ex, 1):
    s, t = e["start"], e["end"]
    c.append(rf"\fill[azulclaro] ({X(s)},-0.13) rectangle ({max(X(t), X(s)+0.03)},0.13);")
    a, b = max(s, cs), min(t, ce)
    if a < b:
        c.append(rf"\fill[azul] ({X(a)},-0.25) rectangle ({max(X(b), X(a)+0.03)},0.25);")
    xm = (X(s) + X(t)) / 2
    yy = 0.45 if i % 2 else 0.75
    if i >= 2 and i <= 11:
        c.append(rf"\draw[rejilla] ({xm},0.27) -- ({xm},{yy-0.1});")
    c.append(rf"\node[etiqueta,font=\sffamily\tiny\bfseries] at ({xm},{yy}) {{{i}}};")
# eje
for v in range(7670000, 7690000, 5000):
    if g0 <= v <= g1:
        c.append(rf"\draw[base] ({X(v)},-0.55) -- ++(0,-0.08) node[below,etiqueta,font=\sffamily\tiny] {{\num{{{v}}}}};")
c.append(rf"\draw[base] ({X(g0)},-0.55) -- ({X(g1)},-0.55);")
c.append(rf"\node[etiqueta] at (5.8,-1.2) {{cromosoma 17 (GRCh38), coordenadas 1-based}};")
c.append(rf"\node[etiqueta,anchor=west,align=left] at (0,1.25) {{\texttt{{{can['id']}.{can['version']}}} · hebra $-$ · "
         rf"{len(ex)} exones · 17:\num{{{g0}}}--\num{{{g1}}} (\num{{{g1-g0+1}}} pb) · CDS de {can['Translation']['length']} aa}};")
c.append(r"\end{tikzpicture}")
w("tp53_gen.tex", c)
exon_len = sum(e["end"] - e["start"] + 1 for e in can["Exon"])
stats["tp53"] = dict(gene=(ens["start"], ens["end"], ens["version"]), tx=(can["id"], can["version"], g0, g1),
                     nex=len(ex), exon_len=exon_len, span=g1 - g0 + 1, cds=(cs, ce),
                     ntx=len(ens["Transcript"]), assembly=ens.get("assembly_name"))

# ------------------------------------------------------------------
# 11. Variantes de p53 (UniProt P04637)
# ------------------------------------------------------------------
up = json.load(open(API / "uniprot_P04637.json"))
var = [f for f in up["features"] if f["type"] == "Natural variant"]
pos = collections.Counter(f["location"]["start"]["value"] for f in var)
Lp = up["sequence"]["length"]
with open(OUT / "p53_variantes.dat", "w") as fh:
    fh.write("pos n\n")
    for p in range(1, Lp + 1):
        fh.write(f"{p} {pos.get(p, 0)}\n")
K = sum(1 for f in var if 102 <= f["location"]["start"]["value"] <= 292)
Nv = len(var)
pd = 191 / Lp
z = (K - Nv * pd) / math.sqrt(Nv * pd * (1 - pd))
from scipy import stats as st
pval = st.binom.sf(K - 1, Nv, pd)
stats["p53"] = dict(N=Nv, K=K, p=pd, exp=Nv * pd, z=z, pval=float(pval), L=Lp,
                    top=pos.most_common(8), npos=len(pos), audit=up["entryAudit"],
                    seq_hotspots={k: up["sequence"]["value"][k-1] for k in (175, 245, 248, 249, 273, 282)})

# ------------------------------------------------------------------
# 12. PDB: entradas de p53 por año y método
# ------------------------------------------------------------------
pdb = json.load(open(API / "rcsb_graphql_P04637_entries.json"))["data"]["entries"]
by = collections.Counter()
res = []
for e in pdb:
    ai = e.get("rcsb_accession_info") or {}
    d = ai.get("initial_release_date")
    if not d: continue
    y = int(d[:4])
    m = (e.get("exptl") or [{}])[0].get("method", "")
    mm = {"X-RAY DIFFRACTION": "xray", "SOLUTION NMR": "nmr",
          "ELECTRON MICROSCOPY": "em"}.get(m, "otro")
    by[(y, mm)] += 1
    r = (e.get("rcsb_entry_info") or {}).get("resolution_combined")
    if r and mm in ("xray", "em"):
        res.append((y, r[0], mm))
yrs = sorted({y for y, _ in by})
with open(OUT / "pdb_anual.dat", "w") as fh:
    fh.write("anio xray nmr em otro\n")
    for y in range(min(yrs), max(yrs) + 1):
        fh.write(f"{y} " + " ".join(str(by[(y, k)]) for k in ("xray", "nmr", "em", "otro")) + "\n")
stats["pdb"] = dict(n=len(pdb), dated=sum(by.values()),
                    methods={k: sum(v for (y, kk), v in by.items() if kk == k) for k in ("xray", "nmr", "em", "otro")},
                    first=min(yrs), last=max(yrs),
                    best=min(res, key=lambda t: t[1]) if res else None,
                    med_xray=float(np.median([r for _, r, m in res if m == "xray"])),
                    med_em=float(np.median([r for _, r, m in res if m == "em"])) if any(m == "em" for *_, m in res) else None)

# Las cifras citadas en el texto se imprimen abajo (no se guardan en disco).
print(json.dumps({k: v for k, v in stats.items()}, indent=1, default=str)[:4000])
