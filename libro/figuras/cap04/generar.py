"""Genera datos y figuras TikZ del capítulo 4 a partir de cálculos reales.

Uso:  python3 figuras/cap04/generar.py   (desde libro/)
Escribe .tex/.dat en figuras/cap04/ e imprime los números que se citan
en los ejemplos resueltos del texto.
"""
from pathlib import Path
import math
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

OUT = Path(__file__).parent
rng = np.random.default_rng(2024)
B62 = substitution_matrices.load("BLOSUM62")

# =====================================================================
# 4.1  MSA progresivo de cinco globinas (UniProt/Swiss-Prot)
# =====================================================================
GLOB = {
 "HBB_HUMAN": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
 "HBA_HUMAN": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
 "MYG_HUMAN": "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKHPGDFGADAQGAMNKALELFRKDMASNYKELGFQG",
 "HBB1_MOUSE": "MVHLTDAEKAAVSCLWGKVNSDEVGGEALGRLLVVYPWTQRYFDSFGDLSSASAIMGNAKVKAHGKKVITAFNDGLNHLDSLKGTFASLSELHCDKLHVDPENFRLLGNMIVIVLGHHLGKDFTPAAQAAFQKVVAGVATALAHKYH",
 "MYG_PHYMC": "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG",
}
NAMES = list(GLOB)
GO, GE = 10.0, 1.0   # apertura y extensión (unidades BLOSUM62)

al = PairwiseAligner(mode="global", substitution_matrix=B62,
                     open_gap_score=-GO, extend_gap_score=-GE)
k = len(NAMES)
D = np.zeros((k, k))
for a in range(k):
    for b in range(a + 1, k):
        aln = al.align(GLOB[NAMES[a]], GLOB[NAMES[b]])[0]
        s1, s2 = aln[0], aln[1]
        pares = [(u, v) for u, v in zip(s1, s2) if u != "-" and v != "-"]
        ident = sum(u == v for u, v in pares) / len(pares)
        D[a, b] = D[b, a] = 1 - ident
print("Distancias (1 - identidad):")
for a in range(k):
    print(f"{NAMES[a]:11s}", " ".join(f"{D[a,b]:.3f}" for b in range(k)))

# --- UPGMA -----------------------------------------------------------
clusters = {i: ([i], 0.0) for i in range(k)}   # id -> (hojas, altura)
Dm = {(i, j): D[i, j] for i in range(k) for j in range(k) if i != j}
merges, nid = [], k
while len(clusters) > 1:
    ids = list(clusters)
    best = min(((i, j) for i in ids for j in ids if i < j), key=lambda p: Dm[p])
    i, j = best
    h = Dm[best] / 2
    li, lj = clusters[i][0], clusters[j][0]
    merges.append((i, j, nid, h, Dm[best]))
    for o in ids:
        if o in (i, j):
            continue
        d = (len(li) * Dm[(min(i, o), max(i, o))] + len(lj) * Dm[(min(j, o), max(j, o))]) / (len(li) + len(lj))
        Dm[(min(nid, o), max(nid, o))] = Dm[(max(nid, o), min(nid, o))] = d
    del clusters[i], clusters[j]
    clusters[nid] = (li + lj, h)
    nid += 1
print("UPGMA:", [(a, b, c, round(h, 3), round(d, 3)) for a, b, c, h, d in merges])


# --- alineamiento perfil-perfil (Gotoh sobre perfiles, puntuación SP media)
def col_score(A, B):
    tot, npar = 0.0, 0
    for a in A:
        for b in B:
            if a != "-" and b != "-":
                tot += B62[a][b]
            npar += 1
    return tot / npar


def gapfrac(col):
    return sum(c != "-" for c in col) / len(col)


def align_profiles(P, Q):
    """P, Q: listas de filas (cadenas de igual longitud)."""
    Pc = ["".join(r[c] for r in P) for c in range(len(P[0]))]
    Qc = ["".join(r[c] for r in Q) for c in range(len(Q[0]))]
    n, m = len(Pc), len(Qc)
    NEG = -1e9
    M = np.full((n + 1, m + 1), NEG); X = M.copy(); Y = M.copy()
    M[0, 0] = 0
    for i in range(1, n + 1):
        X[i, 0] = -GO - (i - 1) * GE
    for j in range(1, m + 1):
        Y[0, j] = -GO - (j - 1) * GE
    S = np.array([[col_score(a, b) for b in Qc] for a in Pc])
    gp = np.array([gapfrac(a) for a in Pc]); gq = np.array([gapfrac(b) for b in Qc])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            M[i, j] = S[i-1, j-1] + max(M[i-1, j-1], X[i-1, j-1], Y[i-1, j-1])
            X[i, j] = max(M[i-1, j] - GO * gp[i-1], X[i-1, j] - GE * gp[i-1])
            Y[i, j] = max(M[i, j-1] - GO * gq[j-1], Y[i, j-1] - GE * gq[j-1])
    # traceback
    i, j = n, m
    st = int(np.argmax([M[n, m], X[n, m], Y[n, m]]))
    cols = []
    while i > 0 or j > 0:
        if st == 0:
            cols.append((i-1, j-1))
            prev = [M[i-1, j-1], X[i-1, j-1], Y[i-1, j-1]]
            st = int(np.argmax(prev)); i, j = i-1, j-1
        elif st == 1:
            cols.append((i-1, None))
            st = 0 if (j > 0 or i > 1) and M[i-1, j] - GO * gp[i-1] >= X[i-1, j] - GE * gp[i-1] else 1
            if i - 1 == 0 and j == 0: st = 0
            i -= 1
        else:
            cols.append((None, j-1))
            st = 0 if M[i, j-1] - GO * gq[j-1] >= Y[i, j-1] - GE * gq[j-1] else 2
            if j - 1 == 0 and i == 0: st = 0
            j -= 1
        if i == 0 and j > 0: st = 2
        if j == 0 and i > 0: st = 1
    cols.reverse()
    newP = ["".join(r[c] if c is not None else "-" for c, _ in cols) for r in P]
    newQ = ["".join(r[c] if c is not None else "-" for _, c in cols) for r in Q]
    return newP + newQ


prof = {i: ([NAMES[i]], [GLOB[NAMES[i]]]) for i in range(k)}
for i, j, nid_, h, d in merges:
    ni, ri = prof.pop(i); nj, rj = prof.pop(j)
    prof[nid_] = (ni + nj, align_profiles(ri, rj))
names_msa, rows = list(prof.values())[0]
msa = dict(zip(names_msa, rows))
L = len(rows[0])
print("Longitud MSA:", L)
for nm in names_msa:
    print(f"{nm:11s} {msa[nm]}")


def sp_score(rows, go=GO, ge=GE):
    """Suma de pares con huecos afines por par (huecos '-' vs '-' ignorados)."""
    tot = 0.0
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            gap_open = False
            for u, v in zip(rows[a], rows[b]):
                if u == "-" and v == "-":
                    continue
                if u == "-" or v == "-":
                    tot -= ge if gap_open else go
                    gap_open = True
                else:
                    tot += B62[u][v]; gap_open = False
    return tot
print("SP del MSA:", sp_score(rows))

# ejemplo de columna para el texto
col = "".join(msa[nm][15] for nm in names_msa)
print("Columna 16:", col, "SP=", sum(B62[col[a]][col[b]] for a in range(5) for b in range(a+1, 5)))

# --- figura: dendrograma + matriz de distancias ------------------------
order = merges[-1]  # raíz
def leaves(c):
    if c < k: return [c]
    for i, j, n_, h, d in merges:
        if n_ == c: return leaves(i) + leaves(j)
leaf_order = leaves(merges[-1][2])
ypos = {c: idx for idx, c in enumerate(leaf_order)}
height = {c: 0.0 for c in range(k)}
for i, j, n_, h, d in merges:
    ypos[n_] = (ypos[i] + ypos[j]) / 2; height[n_] = h
H = max(height.values())
sx = 3.6 / H     # cm por unidad de distancia
t = [r"\begin{tikzpicture}[y=-0.95cm]"]
# matriz de distancias a la izquierda
t.append(r"\begin{scope}[shift={(-4.3,0)}]")
for a, ca in enumerate(leaf_order):
    t.append(rf"\node[anchor=east,font=\ttfamily\scriptsize,text=tinta2] at (-0.05,{a}) {{{NAMES[ca].replace('_', r'\_')}}};")
    for b, cb in enumerate(leaf_order):
        v = D[ca, cb]
        shade = int(round(100 * (1 - v / D.max()) * 0.85)) if a != b else 0
        txt = "white" if shade > 55 else "tinta"
        lab = "" if a == b else f"{v:.2f}".replace(".", "{,}")
        t.append(rf"\node[draw=white,fill=azul!{shade},minimum width=0.66cm,minimum height=0.9cm,inner sep=0pt,font=\sffamily\scriptsize,text={txt}] at ({0.36+b*0.68},{a}) {{{lab}}};")
t.append(r"\node[etiqueta,anchor=south] at (1.72,-0.75) {distancia $d_{ij}=1-\text{identidad}$};")
t.append(r"\end{scope}")
# dendrograma
X0 = 0.6
def xof(c): return X0 + (H - height[c]) * sx
for i, j, n_, h, d in merges:
    for c in (i, j):
        t.append(rf"\draw[azulprofundo,very thick,line cap=round] ({xof(c):.3f},{ypos[c]:.3f}) -- ({xof(n_):.3f},{ypos[c]:.3f});")
    t.append(rf"\draw[azulprofundo,very thick,line cap=round] ({xof(n_):.3f},{ypos[i]:.3f}) -- ({xof(n_):.3f},{ypos[j]:.3f});")
for step, (i, j, n_, h, d) in enumerate(merges, 1):
    t.append(rf"\node[circle,fill=naranja,text=white,font=\sffamily\bfseries\scriptsize,inner sep=1.2pt,minimum size=4.2mm] at ({xof(n_):.3f},{ypos[n_]:.3f}) {{{step}}};")
for c in leaf_order:
    t.append(rf"\fill[azulprofundo] ({xof(c):.3f},{ypos[c]:.3f}) circle (1.6pt);")
    t.append(rf"\node[anchor=west,font=\ttfamily\scriptsize,text=tinta] at ({xof(c)+0.1:.3f},{ypos[c]:.3f}) {{{NAMES[c].replace('_', r'\_')}}};")
# escala
ymax = len(leaf_order) - 0.4
t.append(rf"\draw[base] ({X0},{ymax}) -- ({X0+H*sx:.3f},{ymax});")
for v in np.arange(0, H + 1e-9, 0.1):
    xx = X0 + (H - v) * sx
    t.append(rf"\draw[base] ({xx:.3f},{ymax}) -- ({xx:.3f},{ymax+0.1}); \node[font=\sffamily\tiny,text=gris,anchor=north] at ({xx:.3f},{ymax+0.1}) {{{v:.1f}}};".replace(".", "{,}", 1) if False else
             rf"\draw[base] ({xx:.3f},{ymax}) -- ({xx:.3f},{ymax+0.1}); \node[font=\sffamily\tiny,text=gris,anchor=north] at ({xx:.3f},{ymax+0.1}) {{{str(round(v,1)).replace('.', '{,}')}}};")
t.append(rf"\node[etiqueta,anchor=north] at ({X0+H*sx/2:.3f},{ymax+0.45}) {{altura UPGMA ($d/2$)}};")
t.append(r"\end{tikzpicture}")
(OUT / "arbol_guia.tex").write_text("\n".join(t))

# --- figura: fragmento del MSA coloreado + conservación ---------------
GRUPOS = {**{c: "azul" for c in "AILMFWV"}, **{c: "rojo" for c in "KR"},
          **{c: "magenta" for c in "DE"}, **{c: "verde" for c in "NQST"},
          **{c: "naranja" for c in "G"}, **{c: "amarillo" for c in "P"},
          **{c: "aqua" for c in "HY"}, **{c: "violeta" for c in "C"}}
C0, C1 = 0, 62
t = [r"\begin{tikzpicture}[x=1.75mm,y=-4.6mm]"]
for r, nm in enumerate(leaf_order):
    nmn = NAMES[nm]
    t.append(rf"\node[anchor=east,font=\ttfamily\scriptsize,text=tinta2] at (-0.2,{r}) {{{nmn.replace('_', r'\_')}}};")
    for c in range(C0, C1):
        ch = msa[nmn][c]
        x = c - C0
        if ch == "-":
            t.append(rf"\node[font=\ttfamily\tiny,text=base,inner sep=0pt] at ({x+0.5},{r}) {{-}};")
            continue
        colv = [msa[n2][c] for n2 in names_msa]
        frac = colv.count(ch) / len(colv)
        colr = GRUPOS.get(ch, "gris")
        fillp = 70 if frac >= 0.8 else (35 if frac >= 0.6 else 0)
        txt = "white" if fillp >= 70 else "tinta"
        if fillp:
            t.append(rf"\fill[{colr}!{fillp}] ({x+0.04},{r-0.45}) rectangle ({x+0.96},{r+0.45});")
        t.append(rf"\node[font=\ttfamily\scriptsize,text={txt},inner sep=0pt] at ({x+0.5},{r}) {{{ch}}};")
# barras de conservación (1 - entropía normalizada)
base_y = len(leaf_order) + 2.3
for c in range(C0, C1):
    colv = [msa[n2][c] for n2 in names_msa if msa[n2][c] != "-"]
    cnt = {a: colv.count(a) for a in set(colv)}
    Hc = -sum(v/5 * math.log2(v/5) for v in cnt.values())
    gaps = 5 - len(colv)
    cons = max(0.0, (math.log2(5) - Hc) / math.log2(5)) * (len(colv) / 5)
    x = c - C0
    t.append(rf"\fill[amarillo!80!black] ({x+0.1},{base_y}) rectangle ({x+0.9},{base_y - 1.8*cons:.3f});")
t.append(rf"\draw[base] (0,{base_y}) -- ({C1-C0},{base_y});")
t.append(rf"\node[anchor=east,etiqueta] at (-0.2,{base_y-0.9}) {{conservación}};")
for c in range(C0, C1, 10):
    t.append(rf"\node[font=\sffamily\tiny,text=gris] at ({c-C0+0.5},-0.9) {{{c+1}}};")
t.append(r"\end{tikzpicture}")
(OUT / "msa_globinas.tex").write_text("\n".join(t))
(OUT / "msa_globinas.aln").write_text("\n".join(f"{nm:11s} {msa[nm]}" for nm in names_msa) + "\n")

# =====================================================================
# 4.2  Motivo de unión de TBP (JASPAR MA0108.2, conteos)
# =====================================================================
BASES = "ACGT"
TBP = {  # https://jaspar.elixir.no/api/v1/matrix/MA0108.2/
 "A": [61, 16, 352, 3, 354, 268, 360, 222, 155, 56, 83, 82, 82, 68, 77],
 "C": [145, 46, 0, 10, 0, 0, 3, 2, 44, 135, 147, 127, 118, 107, 101],
 "G": [152, 18, 2, 2, 5, 0, 20, 44, 157, 150, 128, 128, 128, 139, 140],
 "T": [31, 309, 35, 374, 30, 121, 6, 121, 33, 48, 31, 52, 61, 75, 71],
}
N = np.array([TBP[b] for b in BASES], dtype=float)          # 4 x W
W = N.shape[1]
alpha = 0.5                                                   # pseudoconteo por base
F = (N + alpha) / (N.sum(0) + 4 * alpha)
q = np.full(4, 0.25)
PWM = np.log2(F / q[:, None])
Hcol = -(F * np.log2(F)).sum(0)
R = 2 - Hcol
print("\nTBP: N por columna =", N.sum(0).astype(int))
print("Columna 3 conteos", N[:, 2], "frecuencias", np.round(F[:, 2], 4),
      "PWM", np.round(PWM[:, 2], 2), "R=", round(R[2], 3))
print("Columna 1 frecuencias", np.round(F[:, 0], 3), "PWM", np.round(PWM[:, 0], 2), "R=", round(R[0], 3))
print("R por columna:", np.round(R, 2), " total", round(R.sum(), 2))
cons = "".join(BASES[i] for i in F.argmax(0))
print("Consenso:", cons, " puntuación máx", round(PWM.max(0).sum(), 2), " mín", round(PWM.min(0).sum(), 2))

# matriz PWM como tabla coloreada
t = [r"\begin{tikzpicture}[x=0.74cm,y=-0.62cm]"]
vmax = 3.0
for bi, b in enumerate(BASES):
    t.append(rf"\node[nt={b}] at (-0.95,{bi}) {{{b}}};")
    for c in range(W):
        v = PWM[bi, c]
        if v >= 0:
            fill = f"azul!{min(90, int(90 * v / 2.0))}"
        else:
            fill = f"rojo!{min(80, int(80 * -v / vmax))}"
        txt = "white" if (v > 1.2 or v < -4) else "tinta"
        lab = f"{v:.1f}".replace(".", "{,}").replace("-", "$-$")
        t.append(rf"\node[draw=white,fill={fill},minimum width=0.74cm,minimum height=0.62cm,inner sep=0pt,font=\sffamily\scriptsize,text={txt}] at ({c},{bi}) {{{lab}}};")
for c in range(W):
    t.append(rf"\node[font=\sffamily\scriptsize,text=gris] at ({c},-0.8) {{{c+1}}};")
    rl = f"{R[c]:.2f}".replace(".", "{,}")
    t.append(rf"\node[font=\sffamily\scriptsize,text=tinta2] at ({c},4.05) {{{rl}}};")
t.append(r"\node[anchor=east,etiqueta] at (-0.5,4.05) {$R_i$ (bits)};")
t.append(r"\end{tikzpicture}")
(OUT / "pwm_tbp.tex").write_text("\n".join(t))

# logo TikZ: letras escaladas por f*R
t = [r"\begin{tikzpicture}[x=0.76cm,y=1.9cm]"]
t.append(r"\draw[base] (0.4,0) -- (%.2f,0);" % (W + 0.6))
t.append(r"\draw[base] (0.4,0) -- (0.4,2);")
for v in (0, 1, 2):
    t.append(rf"\draw[base] (0.4,{v}) -- (0.3,{v}); \node[font=\sffamily\scriptsize,text=gris,anchor=east] at (0.3,{v}) {{{v}}};")
t.append(r"\node[etiqueta,rotate=90] at (-0.35,1) {bits};")
for c in range(W):
    hs = F[:, c] * R[c]
    order = np.argsort(hs)          # pequeñas abajo, la mayor arriba
    y = 0.0
    for bi in order:
        h = hs[bi]
        if h > 0.004:
            t.append(rf"\node[anchor=south west,inner sep=0pt,text=nuc{BASES[bi]}] at ({c+0.54:.3f},{y:.4f}) {{\resizebox{{0.7cm}}{{{h*1.9:.4f}cm}}{{\sffamily\bfseries {BASES[bi]}}}}};")
        y += h
    assert abs(y - R[c]) < 1e-9 and y <= 2
    t.append(rf"\node[font=\sffamily\scriptsize,text=gris] at ({c+1},-0.14) {{{c+1}}};")
t.append(r"\end{tikzpicture}")
(OUT / "logo_tbp.tex").write_text("\n".join(t))

# distribución de puntuaciones: sitios simulados vs fondo
nsim = 20000
def sample_sites(Fm, n):
    idx = np.stack([rng.choice(4, size=n, p=Fm[:, c]) for c in range(W)], 1)
    return idx
def scores(idx):
    return PWM[idx, np.arange(W)].sum(1)
s_sit = scores(sample_sites(F, nsim))
s_bg = scores(rng.integers(0, 4, size=(nsim, W)))
DKL = (F * np.log2(F / q[:, None])).sum()
print("media sitios", round(s_sit.mean(), 3), " D_KL total", round(DKL, 3),
      " media fondo", round(s_bg.mean(), 3), " -sum D(q||f)", round(-(q[:, None] * np.log2(q[:, None] / F)).sum(), 3))
thr = 6.0
print("umbral", thr, " sensibilidad", round((s_sit >= thr).mean(), 3), " FPR", (s_bg >= thr).mean())
bins = np.arange(-40, 22, 1.0)
hs, _ = np.histogram(s_sit, bins, density=True)
hb, _ = np.histogram(s_bg, bins, density=True)
with open(OUT / "hist_pwm.dat", "w") as fh:
    fh.write("x sitios fondo\n")
    for x, a, b in zip(bins[:-1], hs, hb):
        fh.write(f"{x+0.5} {a:.5f} {b:.5f}\n")
# barrido de una secuencia con dos sitios implantados
Lseq = 400
seq = rng.integers(0, 4, size=Lseq)
plant = [120, 290]
for p in plant:
    seq[p:p+W] = sample_sites(F, 1)[0]
sc = np.array([PWM[seq[i:i+W], np.arange(W)].sum() for i in range(Lseq - W + 1)])
with open(OUT / "barrido.dat", "w") as fh:
    fh.write("pos score\n")
    for i, v in enumerate(sc):
        fh.write(f"{i+1} {v:.3f}\n")
print("barrido: máximos", [(int(i+1), round(sc[i], 2)) for i in np.argsort(sc)[-4:]],
      " sitios plantados", [p+1 for p in plant], "->", "".join(BASES[b] for b in seq[plant[0]:plant[0]+W]),
      "".join(BASES[b] for b in seq[plant[1]:plant[1]+W]))
# curva de entropía binaria/4-letras para figura conceptual: la genera pgfplots

# =====================================================================
# 4.3  HMM de dos estados (H = rico en GC, L = pobre en GC)
# =====================================================================
S = ["H", "L"]
A = np.array([[0.8, 0.2], [0.3, 0.7]])
E = np.array([[0.15, 0.35, 0.35, 0.15], [0.35, 0.15, 0.15, 0.35]])
pi = np.array([0.5, 0.5])
x = "GCGCATTAT"
xi = [BASES.index(c) for c in x]
Lx = len(xi)
# forward
f = np.zeros((2, Lx)); f[:, 0] = pi * E[:, xi[0]]
for i in range(1, Lx):
    f[:, i] = E[:, xi[i]] * (f[:, i-1] @ A)
Px = f[:, -1].sum()
b = np.ones((2, Lx))
for i in range(Lx - 2, -1, -1):
    b[:, i] = A @ (E[:, xi[i+1]] * b[:, i+1])
post = f * b / Px
# viterbi en log2
lv = np.zeros((2, Lx)); ptr = np.zeros((2, Lx), int)
lv[:, 0] = np.log2(pi) + np.log2(E[:, xi[0]])
for i in range(1, Lx):
    for s in range(2):
        cand = lv[:, i-1] + np.log2(A[:, s])
        ptr[s, i] = int(np.argmax(cand)); lv[s, i] = cand.max() + np.log2(E[s, xi[i]])
path = [int(np.argmax(lv[:, -1]))]
for i in range(Lx - 1, 0, -1):
    path.append(ptr[path[-1], i])
path = path[::-1]
print("\nHMM ejemplo x =", x)
print("forward f1..f3:", np.round(f[:, :3], 5).tolist())
print("P(x) =", f"{Px:.4e}", " log2 =", round(math.log2(Px), 3))
print("Viterbi log2:", np.round(lv, 2).tolist())
print("camino:", "".join(S[s] for s in path), " log2 P(x,pi*) =", round(lv[:, -1].max(), 3),
      " P(pi*|x)=", round(2 ** lv[:, -1].max() / Px, 4))
print("posterior H:", np.round(post[0], 3).tolist())
# trellis TikZ
t = [r"\begin{tikzpicture}[x=1.28cm,y=-1.55cm]"]
for i, c in enumerate(x):
    t.append(rf"\node[nt={c}] at ({i},-0.8) {{{c}}};")
    t.append(rf"\node[font=\sffamily\tiny,text=gris] at ({i},-1.25) {{$i={i+1}$}};")
for s in range(2):
    t.append(rf"\node[anchor=east,font=\sffamily\bfseries\small,text={'naranja' if s == 0 else 'azul'}] at (-0.55,{s}) {{{S[s]}}};")
for i in range(1, Lx):
    for s in range(2):
        for r in range(2):
            onp = path[i-1] == r and path[i] == s
            chosen = ptr[s, i] == r
            if onp:
                sty = "naranja,very thick,-{Stealth[length=2mm]}"
            elif chosen:
                sty = "tinta2,thin,-{Stealth[length=1.5mm]}"
            else:
                sty = "rejilla,thin,dashed"
            t.append(rf"\draw[{sty},shorten >=0.43cm,shorten <=0.43cm] ({i-1},{r}) -- ({i},{s});")
for i in range(Lx):
    for s in range(2):
        onp = path[i] == s
        sty = "fill=amarillo!40,draw=amarillo!80!black,thick" if onp else "fill=papel!70,draw=rejilla"
        lab = f"{lv[s, i]:.1f}".replace(".", "{,}").replace("-", "$-$")
        t.append(rf"\node[circle,{sty},minimum size=0.84cm,inner sep=0pt,font=\sffamily\scriptsize{'\\bfseries' if onp else ''}] at ({i},{s}) {{{lab}}};")
t.append(r"\end{tikzpicture}")
(OUT / "trellis.tex").write_text("\n".join(t))

# simulación larga + decodificación posterior
A2 = np.array([[0.98, 0.02], [0.02, 0.98]])
def simulate(A_, E_, L_, pi_=pi):
    st = np.zeros(L_, int); ob = np.zeros(L_, int)
    st[0] = rng.choice(2, p=pi_)
    for i in range(L_):
        if i: st[i] = rng.choice(2, p=A_[st[i-1]])
        ob[i] = rng.choice(4, p=E_[st[i]])
    return st, ob
def fb_scaled(A_, E_, pi_, ob):
    L_ = len(ob); f_ = np.zeros((2, L_)); c = np.zeros(L_)
    f_[:, 0] = pi_ * E_[:, ob[0]]; c[0] = f_[:, 0].sum(); f_[:, 0] /= c[0]
    for i in range(1, L_):
        f_[:, i] = E_[:, ob[i]] * (f_[:, i-1] @ A_); c[i] = f_[:, i].sum(); f_[:, i] /= c[i]
    b_ = np.ones((2, L_))
    for i in range(L_ - 2, -1, -1):
        b_[:, i] = A_ @ (E_[:, ob[i+1]] * b_[:, i+1]) / c[i+1]
    return f_, b_, c
def viterbi(A_, E_, pi_, ob):
    L_ = len(ob); v = np.zeros((2, L_)); p_ = np.zeros((2, L_), int)
    v[:, 0] = np.log(pi_) + np.log(E_[:, ob[0]])
    for i in range(1, L_):
        cand = v[:, i-1][:, None] + np.log(A_)
        p_[:, i] = cand.argmax(0); v[:, i] = cand.max(0) + np.log(E_[:, ob[i]])
    pth = [int(v[:, -1].argmax())]
    for i in range(L_ - 1, 0, -1): pth.append(p_[pth[-1], i])
    return np.array(pth[::-1])
st, ob = simulate(A2, E, 400)
f_, b_, c = fb_scaled(A2, E, pi, ob)
pH = (f_ * b_)[0]
vp = viterbi(A2, E, pi, ob)
print("posterior: exactitud", round(((pH > 0.5) == (st == 0)).mean(), 3),
      " viterbi exactitud", round((vp == st).mean(), 3))
with open(OUT / "posterior.dat", "w") as fh:
    fh.write("pos real post vit\n")
    for i in range(400):
        fh.write(f"{i+1} {int(st[i] == 0)} {pH[i]:.4f} {int(vp[i] == 0)}\n")

# Baum-Welch
st, ob = simulate(A2, E, 5000)
Ahat = np.array([[0.6, 0.4], [0.4, 0.6]])
Ehat = np.array([[0.2, 0.3, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]])
pihat = pi.copy()
hist = []
for it in range(60):
    f_, b_, c = fb_scaled(Ahat, Ehat, pihat, ob)
    ll = np.log(c).sum()
    gamma = f_ * b_
    xi_ = np.zeros((2, 2))
    for i in range(len(ob) - 1):
        m = f_[:, i][:, None] * Ahat * (Ehat[:, ob[i+1]] * b_[:, i+1])[None, :] / c[i+1]
        xi_ += m
    hist.append((it, ll / math.log(2), Ahat[0, 0], Ehat[0, 2]))
    Ahat = xi_ / xi_.sum(1, keepdims=True)
    Ehat = np.stack([np.array([gamma[s, ob == a].sum() for a in range(4)]) for s in range(2)])
    Ehat /= Ehat.sum(1, keepdims=True)
    pihat = gamma[:, 0]
f_, b_, c = fb_scaled(A2, E, pi, ob)
ll_true = np.log(c).sum() / math.log(2)
print("Baum-Welch: logL(bits) inicio", round(hist[0][1], 1), " final", round(hist[-1][1], 1),
      " verdadero", round(ll_true, 1))
print("  a_HH final", round(Ahat[0, 0], 4), " a_LL", round(Ahat[1, 1], 4), " e_H(G)", round(Ehat[0, 2], 4),
      " e_L(G)", round(Ehat[1, 2], 4))
print("  iteraciones a 0.1 bits del final:", next(h[0] for h in hist if hist[-1][1] - h[1] < 0.1))
with open(OUT / "baumwelch.dat", "w") as fh:
    fh.write("it ll aHH eHG\n")
    for it, ll, a, e in hist:
        fh.write(f"{it} {ll:.3f} {a:.5f} {e:.5f}\n")
with open(OUT / "baumwelch_ref.tex", "w") as fh:
    fh.write(f"\\def\\capcuatrollverdadero{{{ll_true:.2f}}}\n")

# =====================================================================
# 4.3  Construcción de un profile HMM a partir de un MSA de juguete
# =====================================================================
toy = ["CA--TG", "CAA-TG", "CG--TG", "CAACT-", "TA--TG"]
ncol = len(toy[0])
match = [sum(r[c] != "-" for r in toy) / len(toy) >= 0.5 for c in range(ncol)]
print("\nProfile HMM: columnas match", [c+1 for c in range(ncol) if match[c]])
mcols = [c for c in range(ncol) if match[c]]
for j, c in enumerate(mcols, 1):
    cnt = {bb: sum(r[c] == bb for r in toy) for bb in BASES}
    tot = sum(cnt.values())
    print(f"  M{j}: conteos {cnt}  e(+1) =", {bb: f"{(cnt[bb]+1)/(tot+4):.3f}" for bb in BASES})
# transiciones: recorrer cada secuencia por estados
from collections import Counter
trans = Counter()
for r in toy:
    states, j = ["B"], 0
    for c in range(ncol):
        if match[c]:
            j += 1
            states.append(f"M{j}" if r[c] != "-" else f"D{j}")
        elif r[c] != "-":
            states.append(f"I{j}")
    states.append("E")
    for a_, b__ in zip(states, states[1:]):
        trans[(a_, b__)] += 1
    print("  ", r, "->", " ".join(states))
print("  transiciones:", dict(sorted(trans.items())))

# columnas de TBP: frecuencia máxima frente a contenido de información
with open(OUT / "tbp_ic.dat", "w") as fh:
    fh.write("pmax R col\n")
    for c in range(W):
        fh.write(f"{F[:, c].max():.4f} {R[c]:.4f} {c+1}\n")
with open(OUT / "tbp_ic_etiq.dat", "w") as fh:
    fh.write("pmax R col\n")
    for c in (0, 1, 2, 3, 5, 7):
        fh.write(f"{F[:, c].max():.4f} {R[c]:.4f} {c+1}\n")
