"""Genera datos y figuras TikZ del capítulo 1 a partir de datos reales.

Ejecutar desde libro/:  python3 figuras/cap01/generar.py
Imprime todas las cifras que se citan en el texto.
"""
import gzip
import math
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

OUT = Path(__file__).parent
DATA = Path(__file__).resolve().parents[3] / "data"
COMP = str.maketrans("ACGTN", "TGCAN")


def wdat(name, header, rows, fmt="{:.6g}"):
    with open(OUT / name, "w") as fh:
        fh.write(" ".join(header) + "\n")
        for r in rows:
            fh.write(" ".join(fmt.format(v) if not isinstance(v, str) else v
                              for v in r) + "\n")


# =====================================================================
# 1.1  Entropía, compresión y robustez del código
# =====================================================================
BASES = "TCAG"
AA64 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = {a + b + c: AA64[16 * i + 4 * j + k]
        for i, a in enumerate(BASES) for j, b in enumerate(BASES)
        for k, c in enumerate(BASES)}


def entropy(seq, k=1):
    from collections import Counter
    cnt = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    p = np.array(list(cnt.values()), float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


sars = SeqIO.read(DATA / "NC_045512.2.gb", "genbank")
real = str(sars.seq).upper()
rng = np.random.default_rng(11)
seqs = {"aleatoria": "".join(rng.choice(list("ACGT"), len(real))),
        "sars": real,
        "cag": ("CAG" * (len(real) // 3 + 1))[:len(real)]}
rows = []
print("== 1.1 Entropía (bits/base) y gzip")
for name, s in seqs.items():
    hk = [entropy(s, k) / k for k in (1, 2, 3)]
    gz = 8 * len(gzip.compress(s.encode(), 9)) / len(s)
    rows.append([name] + hk + [gz])
    print(name, [round(h, 3) for h in hk], "gzip", round(gz, 3))
wdat("entropia.dat", ["sec", "k1", "k2", "k3", "gzip"], rows, "{:.3f}")
comp = {b: real.count(b) / len(real) for b in "ACGT"}
print("composición SARS-CoV-2", {b: round(v, 4) for b, v in comp.items()})

spike = next(f for f in sars.features
             if f.type == "CDS" and f.qualifiers.get("gene") == ["S"])
spike_dna = str(spike.extract(sars.seq))
print("Spike nt", len(spike_dna), "inicio", spike_dna[:45],
      str(Seq(spike_dna[:45]).translate()))

# Efecto de mutaciones puntuales por posición del codón (Spike)
codons = [spike_dna[i:i + 3] for i in range(0, len(spike_dna) - 3, 3)]
eff = np.zeros((3, 3))  # pos x (sin, mis, stop)
for c in codons:
    for p in range(3):
        for b in "ACGT":
            if b == c[p]:
                continue
            m = c[:p] + b + c[p + 1:]
            a0, a1 = CODE[c], CODE[m]
            eff[p, 0 if a1 == a0 else (2 if a1 == "*" else 1)] += 1
effp = 100 * eff / eff.sum(1, keepdims=True)
print("Spike codones", len(codons), "mutaciones", int(eff.sum()))
for p in range(3):
    print(f"  pos {p+1}: sin {effp[p,0]:.1f} mis {effp[p,1]:.1f} "
          f"stop {effp[p,2]:.1f}")
wdat("mut_pos.dat", ["pos", "sin", "mis", "stop"],
     [[p + 1, *effp[p]] for p in range(3)], "{:.2f}")

# ---------------- Transcripción y traducción de Spike (TikZ) ---------
seg = spike_dna[:36]
tmpl = seg.translate(COMP)
mrna = seg.replace("T", "U")
prot = str(Seq(seg).translate())
AACLS = {**{a: "amarillo" for a in "AVILMFWP"}, **{a: "aqua" for a in "STNQCYG"},
         **{a: "azul" for a in "KRH"}, **{a: "rojo" for a in "DE"}}
Wt = 0.3
X = [r"\begin{tikzpicture}[font=\sffamily\scriptsize]"]
rowsy = {"cod": 0.0, "tmp": -0.42, "rna": -1.55, "aa": -2.55}
labs = {"cod": "codificante", "tmp": "molde", "rna": "ARNm", "aa": "proteína"}
ends = {"cod": ("5'", "3'"), "tmp": ("3'", "5'"), "rna": ("5'", "3'"),
        "aa": ("N", "C")}
for key, s_ in (("cod", seg), ("tmp", tmpl), ("rna", mrna)):
    y = rowsy[key]
    for i, b in enumerate(s_):
        bb = "T" if b == "U" else b
        X.append(rf"\node[font=\ttfamily\bfseries\scriptsize,text=nuc{bb}] at ({i*Wt+Wt/2:.3f},{y}) {{{b}}};")
for key, y in rowsy.items():
    X.append(rf"\node[anchor=east,text=tinta2] at (-0.35,{y}) {{{labs[key]}}};")
    X.append(rf"\node[text=gris] at (-0.15,{y}) {{{ends[key][0]}}};")
    X.append(rf"\node[text=gris] at ({len(seg)*Wt+0.15:.3f},{y}) {{{ends[key][1]}}};")
# pares de bases
for i in range(len(seg)):
    X.append(rf"\draw[rejilla] ({i*Wt+Wt/2:.3f},-0.1) -- ({i*Wt+Wt/2:.3f},-0.32);")
X.append(rf"\draw[flecha=tinta2] ({len(seg)*Wt/2:.3f},-0.62) -- node[right,etiqueta]{{transcripción (T$\to$U)}} ({len(seg)*Wt/2:.3f},-1.35);")
for k in range(len(seg) // 3):
    x0, x1 = 3 * k * Wt + 0.03, 3 * (k + 1) * Wt - 0.03
    X.append(rf"\draw[decorate,decoration={{brace,mirror,amplitude=2pt}},gris] ({x0:.3f},-1.72) -- ({x1:.3f},-1.72);")
    a = prot[k]
    X.append(rf"\draw[gris] ({(x0+x1)/2:.3f},-1.85) -- ({(x0+x1)/2:.3f},-2.35);")
    X.append(rf"\node[circle,fill={AACLS[a]},minimum size=5mm,inner sep=0pt,text=white,font=\sffamily\bfseries\scriptsize] (aa{k}) at ({(x0+x1)/2:.3f},-2.55) {{{a}}};")
    if k:
        X.append(rf"\draw[gris,thick] (aa{k-1}) -- (aa{k});")
X.append(rf"\node[anchor=west,etiqueta] at ({len(seg)*Wt/2+0.1:.3f},-2.1) {{}};")
X.append(r"\end{tikzpicture}")
(OUT / "spike_flujo.tex").write_text("\n".join(X))
print("Spike 36 nt:", seg, prot)

# Costo de códigos (Kyte-Doolittle)
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}
sense = [c for c, a in CODE.items() if a != "*"]
pairs = [(c, c[:p] + b + c[p + 1:]) for c in sense for p in range(3)
         for b in "ACGT" if b != c[p] and CODE[c[:p] + b + c[p + 1:]] != "*"]
aa = sorted(KD)
idx = {a: i for i, a in enumerate(aa)}
A = np.array([idx[CODE[x]] for x, _ in pairs])
B = np.array([idx[CODE[y]] for _, y in pairs])
h = np.array([KD[a] for a in aa])


def cost(perm):
    hp = h[perm]
    return float(np.mean((hp[A] - hp[B]) ** 2))


std = cost(np.arange(20))
rng = np.random.default_rng(1998)
NR = 100_000
rc = np.array([cost(rng.permutation(20)) for _ in range(NR)])
frac = (rc <= std).mean()
print(f"== Código: pares missense {len(pairs)}; Phi estándar {std:.3f}; "
      f"media azar {rc.mean():.3f} sd {rc.std():.3f}; min {rc.min():.3f}; "
      f"fracción <= {frac:.5f} ({(rc <= std).sum()} de {NR})")
hist, edges = np.histogram(rc, bins=60, range=(6, 21))
wdat("costos_hist.dat", ["x", "n"],
     [[edges[i], hist[i]] for i in range(len(hist))] + [[edges[-1], 0]],
     "{:.4f}")
# ejemplo resuelto: número de mutaciones sinónimas de cada tipo de codón
nsyn = sum(1 for c in sense for p in range(3) for b in "ACGT"
           if b != c[p] and CODE[c[:p] + b + c[p + 1:]] == CODE[c])
ntot = len(sense) * 9
nstop = sum(1 for c in sense for p in range(3) for b in "ACGT"
            if b != c[p] and CODE[c[:p] + b + c[p + 1:]] == "*")
print(f"Código: {ntot} mutaciones desde 61 codones: sin {nsyn} "
      f"({nsyn/ntot:.3f}), stop {nstop}, missense {ntot-nsyn-nstop}")

# ---------------- Rueda del código genético (TikZ) -------------------
CLS = {**{a: "amarillo" for a in "AVILMFWP"}, **{a: "aqua" for a in "STNQCYG"},
       **{a: "azul" for a in "KRH"}, **{a: "rojo" for a in "DE"}, "*": "gris"}
THREE = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
         "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
         "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
         "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
         "*": "Stop"}
RB = "UCAG"
r0, r1, r2, r3, r4 = 0.0, 1.0, 1.9, 2.75, 3.95
L = [r"\begin{tikzpicture}[scale=1.0,font=\sffamily]"]


def wedge(ri, ro, a0, a1, fill, draw="white"):
    return (rf"\fill[{fill}] ({a0}:{ri}) arc[start angle={a0},end angle={a1},"
            rf"radius={ri}] -- ({a1}:{ro}) arc[start angle={a1},"
            rf"end angle={a0},radius={ro}] -- cycle;"
            rf"\draw[{draw},line width=0.5pt] ({a0}:{ri}) -- ({a0}:{ro});")


def lab(r, a, t, extra=""):
    return rf"\node[rotate={a-90 if 0<=(a%360)<=180 else a+90},{extra}] at ({a}:{r}) {{{t}}};"


# ángulos: empezamos en 90° y avanzamos en sentido horario
def ang(i, n):  # i-ésimo de n sectores
    w = 360 / n
    return 90 - i * w, 90 - (i + 1) * w


nucfill = {"U": "nucT", "C": "nucC", "A": "nucA", "G": "nucG"}
for i, a in enumerate(RB):
    a0, a1 = ang(i, 4)
    L.append(wedge(0.001, r1, a1, a0, nucfill[a]))
    L.append(rf"\node[text=white,font=\sffamily\bfseries\Large] at ({(a0+a1)/2}:{0.55}) {{{a}}};")
    for j, b in enumerate(RB):
        b0, b1 = ang(4 * i + j, 16)
        L.append(wedge(r1, r2, b1, b0, nucfill[b] + "!55"))
        L.append(rf"\node[text=white,font=\sffamily\bfseries] at ({(b0+b1)/2}:{(r1+r2)/2}) {{{b}}};")
        for k, c in enumerate(RB):
            c0, c1 = ang(16 * i + 4 * j + k, 64)
            L.append(wedge(r2, r3, c1, c0, nucfill[c] + "!25"))
            L.append(rf"\node[text=tinta,font=\sffamily\scriptsize] at ({(c0+c1)/2}:{(r2+r3)/2}) {{{c}}};")
# anillo de aminoácidos: agrupar codones consecutivos con el mismo aa
cod_order = [a + b + c for a in RB for b in RB for c in RB]
aas = [CODE[x.replace("U", "T")] for x in cod_order]
i = 0
while i < 64:
    j = i
    while j + 1 < 64 and aas[j + 1] == aas[i] and (j + 1) % 4 != 0:
        j += 1
    a0, _ = ang(i, 64)
    _, a1 = ang(j, 64)
    col = CLS[aas[i]]
    L.append(wedge(r3, r4, a1, a0, col + "!35", "white"))
    mid = (a0 + a1) / 2
    t = THREE[aas[i]]
    L.append(rf"\node[text=tinta,font=\sffamily\scriptsize\bfseries,rotate={mid if -90<=mid<=90 else mid+180}] at ({mid}:{(r3+r4)/2}) {{{t}}};")
    i = j + 1
L.append(rf"\draw[white,line width=1.2pt] (0,0) circle ({r1});")
L.append(r"\end{tikzpicture}")
(OUT / "rueda_codigo.tex").write_text("\n".join(L))

# =====================================================================
# 1.2  ORFs en Mycoplasma genitalium
# =====================================================================
IDX = np.full(256, -1, dtype=np.int64)
for i, b in enumerate("TCAG"):
    IDX[ord(b)] = i


def cid(c):
    return 16 * "TCAG".index(c[0]) + 4 * "TCAG".index(c[1]) + "TCAG".index(c[2])


def cids(s):
    a = IDX[np.frombuffer(s.encode(), dtype=np.uint8)]
    n = len(a) // 3
    a = a[:3 * n].reshape(n, 3)
    ids = 16 * a[:, 0] + 4 * a[:, 1] + a[:, 2]
    ids[(a < 0).any(1)] = -1
    return ids


def find_orfs(seq, min_codons=100, starts=("ATG",), stops=("TAA", "TAG", "TGA")):
    Ls = len(seq)
    rcs = seq.translate(COMP)[::-1]
    sid, aid = [cid(c) for c in stops], [cid(c) for c in starts]
    out = []
    for strand, s in ((1, seq), (-1, rcs)):
        for f in range(3):
            c = cids(s[f:])
            sp = np.flatnonzero(np.isin(c, sid))
            st = np.flatnonzero(np.isin(c, aid))
            prev = np.concatenate([[-1], sp[:-1]])
            k = np.searchsorted(st, prev + 1)
            has = k < len(st)
            fs = np.full(len(sp), np.iinfo(np.int64).max)
            fs[has] = st[k[has]]
            n = sp - fs
            keep = (fs < sp) & (n >= min_codons)
            a = f + 3 * fs[keep]
            b = f + 3 * sp[keep] + 3
            if strand == -1:
                a, b = Ls - b, Ls - a
            for x, y, z in zip(a, b, n[keep]):
                out.append((int(x), int(y), strand, f + 1, int(z)))
    return out


def stop_lengths(seq, stops):
    sid = [cid(c) for c in stops]
    rcs = seq.translate(COMP)[::-1]
    out = []
    for s in (seq, rcs):
        for f in range(3):
            pos = np.flatnonzero(np.isin(cids(s[f:]), sid))
            out.append(np.diff(pos) - 1)
    return np.concatenate(out)


def surv(lens, grid):
    srt = np.sort(lens)
    return 1 - np.searchsorted(srt, grid, side="left") / len(srt)


rec = SeqIO.read(DATA / "NC_000908.2.gb", "genbank")
g = str(rec.seq).upper()
Lg = len(g)
gc = (g.count("G") + g.count("C")) / Lg
cds = [f for f in rec.features if f.type == "CDS" and "pseudo" not in f.qualifiers]
pseudo = [f for f in rec.features if f.type == "CDS" and "pseudo" in f.qualifiers]
print(f"== 1.2 M. genitalium: {Lg} pb, GC {gc:.4f}, CDS {len(cds)}, "
      f"pseudo {len(pseudo)}, tablas {set(f.qualifiers.get('transl_table',['?'])[0] for f in cds)}")
clen = np.array([len(f.location) // 3 - 1 for f in cds])
print(f"   longitud CDS (aa): mediana {np.median(clen):.0f}, min {clen.min()}, "
      f"max {clen.max()}, <100 aa: {(clen<100).sum()}, frac cod "
      f"{sum(len(f.location) for f in cds)/Lg:.3f}")
full4 = full11 = tga = 0
for f in cds:
    nt = f.extract(rec.seq)
    ann = len(f.qualifiers["translation"][0])
    full4 += len(nt.translate(table=4, to_stop=True)) == ann
    full11 += len(nt.translate(table=11, to_stop=True)) == ann
    cc = [str(nt[i:i + 3]) for i in range(0, len(nt) - 3, 3)]
    tga += "TGA" in cc
print(f"   completas tabla4 {full4}/{len(cds)}  tabla11 {full11} "
      f"({full11/len(cds):.3f})  con TGA interno {tga}")
fr = {b: g.count(b) / Lg for b in "ACGT"}
p4 = sum(fr[c[0]] * fr[c[1]] * fr[c[2]] for c in ("TAA", "TAG"))
p11 = sum(fr[c[0]] * fr[c[1]] * fr[c[2]] for c in ("TAA", "TAG", "TGA"))
print(f"   composición {({b: round(v,4) for b,v in fr.items()})}; p_stop4 "
      f"{p4:.4f} media {(1-p4)/p4:.2f}; p_stop11 {p11:.4f}")
rng = np.random.default_rng(2024)
shuf = "".join(rng.permutation(np.array(list(g))))
S4 = ("TAA", "TAG")
lr, ls = stop_lengths(g, S4), stop_lengths(shuf, S4)
grid = np.arange(0, 601, 5)
sr, ss = surv(lr, grid), surv(ls, grid)
th = (1 - p4) ** grid
rows = [[x, a if a > 0 else float("nan"), b if b > 0 else float("nan"), c] for x, a, b, c in zip(grid, sr, ss, th)]
wdat("supervivencia.dat", ["l", "real", "shuf", "teo"], rows, "{:.6g}")
print(f"   tramos real {len(lr)} barajado {len(ls)}; media real {lr.mean():.2f} "
      f"barajado {ls.mean():.2f}")
for ell in (50, 100, 200, 300):
    print(f"   P(L>={ell}) real {surv(lr,[ell])[0]:.2e} shuf {surv(ls,[ell])[0]:.2e} "
          f"teo {(1-p4)**ell:.2e}")
# azar uniforme
p = 3 / 64
print(f"   uniforme: p {p:.4f}, media {(1-p)/p:.2f}, l_1% {math.log(0.01)/math.log(1-p):.1f}, "
      f"P(L>=100) {(1-p)**100:.4f}, P(L>=300) {(1-p)**300:.2e}")

ST = ("ATG", "GTG", "TTG")
annot_keys = set()
for f in cds:
    s0, e0, sd = int(f.location.start), int(f.location.end), f.location.strand
    annot_keys.add((e0 if sd == 1 else s0, sd))
ann_iv = np.array([(int(f.location.start), int(f.location.end)) for f in cds])


def keys(o):
    return {(b if sd == 1 else a, sd) for a, b, sd, _, _ in o}


def evaluate(o):
    k = keys(o)
    tp = len(k & annot_keys)
    return tp / len(annot_keys), tp / max(len(k), 1), len(o), tp


def remove_shadows(o, mx=0.5):
    order = sorted(o, key=lambda r: r[1] - r[0], reverse=True)
    ks, ke, keep = np.array([], int), np.array([], int), []
    for r in order:
        s0, e0 = r[0], r[1]
        ov = np.minimum(e0, ke) - np.maximum(s0, ks)
        if len(ov) == 0 or ov.max() <= mx * (e0 - s0):
            keep.append(r)
            ks, ke = np.append(ks, s0), np.append(ke, e0)
    return keep


allo = find_orfs(g, 0, ST, S4)
allsh = find_orfs(shuf, 0, ST, S4)
rows = []
for t in range(40, 251, 10):
    p_ = [r for r in allo if r[4] >= t]
    se, pr, n, tp = evaluate(p_)
    se2, pr2, n2, tp2 = evaluate(remove_shadows(p_))
    nsh = sum(1 for r in allsh if r[4] >= t)
    rows.append([t, se, pr, se2, pr2, n, nsh, nsh / max(n, 1)])
    if t in (40, 60, 100, 150, 200, 250):
        print(f"   umbral {t}: n {n} VP {tp} sens {se:.3f} prec {pr:.3f} | filtro n {n2} "
              f"sens {se2:.3f} prec {pr2:.3f} | barajado {nsh} FDR {nsh/max(n,1):.3f}")
wdat("sens_prec.dat", ["t", "se", "pr", "sef", "prf", "n", "nsh", "fdr"], rows,
     "{:.4f}")
# falsos positivos sombra a 100 codones
p100 = [r for r in allo if r[4] >= 100]
fp = [r for r in p100 if ((r[1] if r[2] == 1 else r[0]), r[2]) not in annot_keys]
psi = np.array([(int(f.location.start), int(f.location.end)) for f in pseudo])


def ovl(r, iv):
    ov = np.minimum(r[1], iv[:, 1]) - np.maximum(r[0], iv[:, 0])
    return bool((ov > 0.5 * (r[1] - r[0])).any())


print(f"   FP a 100: {len(fp)}; sombra {sum(ovl(r, ann_iv) for r in fp)}; "
      f"pseudo {sum(ovl(r, psi) for r in fp)}")
# genes perdidos (ORFs>=100 con filtro)
kf = keys(remove_shadows(p100))
missed = [len(f.location)//3-1 for f in cds
          if ((int(f.location.end) if f.location.strand == 1 else int(f.location.start)),
              f.location.strand) not in kf]
print(f"   genes no encontrados {len(missed)}; <100 aa: {sum(m<100 for m in missed)}")

# ---------------- Escáner de seis marcos (TikZ) ----------------------
demo = "GCATGGCTAAAGAACTGTTTGCATAAGGATGCGTCATGACCTGAAATTCA"
demo = demo[:45]
rcd = demo.translate(COMP)[::-1]
STOP = {"TAA", "TAG", "TGA"}


def states(s, f):
    cods = [s[f + 3 * k:f + 3 * k + 3] for k in range((len(s) - f) // 3)]
    st, op = ["n"] * len(cods), None
    for k, c in enumerate(cods):
        if op is None and c == "ATG":
            op = k
            st[k] = "start"
        elif op is not None and c in STOP:
            for j in range(op + 1, k):
                st[j] = "orf"
            st[k] = "stop"
            op = None
        elif c in STOP:
            st[k] = "stop0"
    if op is not None:
        for j in range(op + 1, len(cods)):
            st[j] = "open"
    return cods, st


W = 0.25  # cm por nucleótido
T = [r"\begin{tikzpicture}[font=\sffamily\scriptsize]"]
for i, b in enumerate(demo):
    T.append(rf"\node[font=\ttfamily\bfseries\scriptsize,text=nuc{b}] at ({i*W+W/2:.3f},0.55) {{{b}}};")
    if i % 10 == 0:
        T.append(rf"\node[text=gris,font=\sffamily\tiny] at ({i*W+W/2:.3f},0.9) {{{i+1}}};")
T.append(r"\node[anchor=east,text=tinta2] at (-0.15,0.55) {5'\,$\to$\,3'};")
sty = {"n": "fill=papel,draw=rejilla,text=tinta2",
       "start": "fill=verde,draw=verde,text=white",
       "stop": "fill=rojo,draw=rojo,text=white",
       "stop0": "fill=rojoclaro,draw=rojoclaro,text=tinta",
       "orf": "fill=azul!80,draw=azul!80,text=white",
       "open": "fill=azulpalido,draw=azulclaro,text=tinta2"}
ypos = []
orf_summary = []
for r, (s, f, lab) in enumerate([(demo, 0, "+1"), (demo, 1, "+2"), (demo, 2, "+3"),
                                  (rcd, 0, "$-$1"), (rcd, 1, "$-$2"), (rcd, 2, "$-$3")]):
    y = -0.2 - 0.52 * r - (0.35 if r >= 3 else 0)
    T.append(rf"\node[anchor=east,text=tinta,font=\sffamily\scriptsize\bfseries] at (-0.15,{y:.3f}) {{marco {lab}}};")
    cods, st = states(s, f)
    for k, (c, z) in enumerate(zip(cods, st)):
        x0 = (f + 3 * k) * W + 0.02
        T.append(rf"\node[{sty[z]},rounded corners=1pt,minimum height=3.6mm,minimum width={3*W-0.04:.3f}cm,inner sep=0pt,font=\ttfamily\tiny] at ({x0+1.5*W-0.02:.3f},{y:.3f}) {{{c}}};")
    orf_summary.append((lab, "".join(z[0] for z in st)))
T.append(r"\draw[rejilla,dashed] (-1.6,-1.575) -- (" + f"{len(demo)*W:.2f}" + ",-1.575);")

T.append(r"\end{tikzpicture}")
(OUT / "seis_marcos.tex").write_text("\n".join(T))
print("   demo", demo, "rc", rcd)
for lab, s in orf_summary:
    print("   ", lab, s)

# =====================================================================
# 1.3  GC skew en E. coli
# =====================================================================
with gzip.open(DATA / "NC_000913.3.fasta.gz", "rt") as fh:
    ec = str(SeqIO.read(fh, "fasta").seq).upper()
Le = len(ec)
ORI = (3_925_744, 3_925_975)
TUS = (1_684_259, 1_685_188)
arr = np.frombuffer(ec.encode(), dtype=np.uint8)
step = (arr == ord("G")).astype(np.int32) - (arr == ord("C")).astype(np.int32)
cs = np.cumsum(step)
pori, pter = int(np.argmin(cs)) + 1, int(np.argmax(cs)) + 1
omid, tmid = np.mean(ORI), np.mean(TUS)


def cdist(a, b):
    d = abs(a - b) % Le
    return min(d, Le - d)


gce = (ec.count("G") + ec.count("C")) / Le
print(f"== 1.3 E. coli {Le} pb GC {gce:.4f}; G {ec.count('G')} C {ec.count('C')}; "
      f"min cs {pori} (val {cs[pori-1]}) dist ori {cdist(pori, omid):.0f}; "
      f"max cs {pter} (val {cs[pter-1]}) dist tus {cdist(pter, tmid)/1000:.1f} kb; "
      f"arcos {((pori-pter)%Le)/1e6:.3f} / {((pter-pori)%Le)/1e6:.3f} Mb")
wdat("skew_acum.dat", ["mb", "k"],
     [[i / 1e6, cs[i] / 1000] for i in range(0, Le, 5000)] + [[Le / 1e6, cs[-1] / 1000]],
     "{:.5f}")
win = 10_000
nw = Le // win
a2 = arr[:nw * win].reshape(nw, win)
gw, cw = (a2 == ord("G")).sum(1), (a2 == ord("C")).sum(1)
sk = (gw - cw) / (gw + cw)
wdat("skew_ventanas.dat", ["mb", "pos", "neg"],
     [[(i + 0.5) * win / 1e6, max(s, 0), min(s, 0)] for i, s in enumerate(sk)],
     "{:.5f}")
right = ((np.arange(nw) + 0.5) * win > pori) | ((np.arange(nw) + 0.5) * win < pter)
print(f"   skew medio replicor ori->ter(horario) {sk[right].mean():+.4f}, "
      f"otro {sk[~right].mean():+.4f}; ventanas + {int((sk>0).sum())} de {nw}")
# ejemplo: saldo en oriC ± 200 kb
# Ruido vs tamaño de ventana
sizes = np.unique(np.logspace(np.log10(200), np.log10(200_000), 16).astype(int))
rows = []
for w in sizes:
    n = Le // w
    a3 = arr[:n * w].reshape(n, w)
    gg, cc = (a3 == ord("G")).sum(1), (a3 == ord("C")).sum(1)
    s = (gg - cc) / (gg + cc)
    rows.append([w, s.std(), 1 / np.sqrt(w * gce)])
sig = [r for r in rows if r[0] >= 90_000][0][1]
wdat("ruido_ventana.dat", ["w", "sd", "teo"], rows, "{:.6g}")
print("   ruido:", [(int(r[0]), round(r[1], 4), round(r[2], 4)) for r in rows])
s_sig = abs(sk).mean()
print(f"   |skew| medio a 10 kb {s_sig:.4f}")

# AT skew
at = []
w = 100_000
n = Le // w
a4 = arr[:n * w].reshape(n, w)
A_, T_ = (a4 == ord("A")).sum(1), (a4 == ord("T")).sum(1)
G_, C_ = (a4 == ord("G")).sum(1), (a4 == ord("C")).sum(1)
print(f"   |AT skew| medio {np.abs((A_-T_)/(A_+T_)).mean():.4f} "
      f"|GC skew| {np.abs((G_-C_)/(G_+C_)).mean():.4f}")

# Mapa circular (TikZ)
wc = 20_000
nc = Le // wc
a5 = arr[:nc * wc].reshape(nc, wc)
g5, c5 = (a5 == ord("G")).sum(1), (a5 == ord("C")).sum(1)
gcw = (g5 + c5) / wc
skw = (g5 - c5) / (g5 + c5)
dev = gcw - gcw.mean()
print(f"   mapa: GC ventana 20 kb min {gcw.min():.3f} max {gcw.max():.3f}; "
      f"skew min {skw.min():.3f} max {skw.max():.3f}")
M = [r"\begin{tikzpicture}[font=\sffamily\scriptsize]"]
R0, R1 = 2.2, 3.25   # base del anillo interno (skew) y externo (GC)
M.append(rf"\draw[base,line width=0.4pt] (0,0) circle ({R0});")
M.append(rf"\draw[base,line width=0.4pt] (0,0) circle ({R1});")
for i in range(nc):
    a0 = 90 - 360 * i * wc / Le
    a1 = 90 - 360 * (i + 1) * wc / Le
    v = skw[i]
    col = "azul" if v >= 0 else "rojo"
    rr = R0 + 7 * v
    M.append(rf"\fill[{col}] ({a0:.2f}:{R0}) arc[start angle={a0:.2f},end angle={a1:.2f},radius={R0}] -- ({a1:.2f}:{rr:.3f}) arc[start angle={a1:.2f},end angle={a0:.2f},radius={rr:.3f}] -- cycle;")
    d = dev[i]
    col = "azulprofundo" if d >= 0 else "azulclaro"
    rr = R1 + 4 * d
    M.append(rf"\fill[{col}] ({a0:.2f}:{R1}) arc[start angle={a0:.2f},end angle={a1:.2f},radius={R1}] -- ({a1:.2f}:{rr:.3f}) arc[start angle={a1:.2f},end angle={a0:.2f},radius={rr:.3f}] -- cycle;")
for mb in range(5):
    a = 90 - 360 * mb * 1e6 / Le
    M.append(rf"\draw[gris] ({a:.2f}:1.38) -- ({a:.2f}:1.5); \node[text=gris] at ({a:.2f}:1.12) {{{mb} Mb}};")
for pos, lab_ in [(omid, r"\textbf{\emph{oriC}}"), (tmid, r"\textbf{\emph{tus}}\,/\,\emph{ter}")]:
    a = 90 - 360 * pos / Le
    M.append(rf"\draw[tinta,thick] ({a:.2f}:1.55) -- ({a:.2f}:3.6);")
    M.append(rf"\node[text=tinta,font=\sffamily\small] at ({a:.2f}:3.95) {{{lab_}}};")
M.append(r"\node[align=center,text=tinta,font=\sffamily\scriptsize] at (0,-0.1) {\emph{E.\,coli}\\K-12\\" + f"{Le/1e6:.2f}".replace(".", "{,}") + r"~Mb};")
M.append(r"\end{tikzpicture}")
(OUT / "mapa_circular.tex").write_text("\n".join(M))
