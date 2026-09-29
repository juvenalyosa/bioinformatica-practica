"""Genera las figuras TikZ del capítulo 3 a partir de cálculos reales."""
from pathlib import Path
from Bio.Align import substitution_matrices
OUT = Path(__file__).parent

# ---------------- Matriz de Needleman-Wunsch con traceback ----------------
x, y = "GATTACA", "GTTAGCA"
M, MM, G = 1, -1, -2
n, m = len(x), len(y)
F = [[0]*(m+1) for _ in range(n+1)]
for i in range(1, n+1): F[i][0] = i*G
for j in range(1, m+1): F[0][j] = j*G
for i in range(1, n+1):
    for j in range(1, m+1):
        s = M if x[i-1] == y[j-1] else MM
        F[i][j] = max(F[i-1][j-1]+s, F[i-1][j]+G, F[i][j-1]+G)
# traceback (prioridad diagonal)
path, i, j, ax, ay = [(n, m)], n, m, "", ""
while i > 0 or j > 0:
    if i > 0 and j > 0 and F[i][j] == F[i-1][j-1] + (M if x[i-1] == y[j-1] else MM):
        ax, ay, i, j = x[i-1]+ax, y[j-1]+ay, i-1, j-1
    elif i > 0 and F[i][j] == F[i-1][j] + G:
        ax, ay, i = x[i-1]+ax, "-"+ay, i-1
    else:
        ax, ay, j = "-"+ax, y[j-1]+ay, j-1
    path.append((i, j))
onpath = set(path)
c = [r"\begin{tikzpicture}[x=8.2mm,y=-8.2mm,every node/.style={font=\sffamily\small}]"]
# encabezados
for j, b in enumerate(y, start=1):
    c.append(rf"\node[nt={b}] at ({j+1},0) {{{b}}};")
for i, b in enumerate(x, start=1):
    c.append(rf"\node[nt={b}] at (0,{i+1}) {{{b}}};")
c.append(r"\node[text=gris] at (1,0) {$\varepsilon$}; \node[text=gris] at (0,1) {$\varepsilon$};")
for i in range(n+1):
    for j in range(m+1):
        style = r"fill=amarillo!35,draw=amarillo!80!black" if (i, j) in onpath else r"fill=papel!60,draw=rejilla"
        col = "tinta" if F[i][j] >= 0 else "rojooscuro"
        bold = r"\bfseries" if (i, j) in onpath else ""
        c.append(rf"\node[{style},minimum size=7.6mm,rounded corners=1pt,text={col},font=\sffamily\small{bold}] (c{i}{j}) at ({j+1},{i+1}) {{{F[i][j]}}};")
for (a, b), (p, q) in zip(path[:-1], path[1:]):
    c.append(rf"\draw[-{{Stealth[length=1.6mm]}},thick,naranja] (c{a}{b}) -- (c{p}{q});")
c.append(r"\end{tikzpicture}")
(OUT / "nw_matriz.tex").write_text("\n".join(c))
(OUT / "nw_alineamiento.txt").write_text(f"{ax}\n{ay}\nscore={F[n][m]}\n")
print(ax, ay, F[n][m], sep="\n")

# ---------------- Dot plot: repetición + inversión ----------------------
import random
random.seed(7)
core = "".join(random.choice("ACGT") for _ in range(14))
rep = "".join(random.choice("ACGT") for _ in range(6))
comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
seq1 = core[:5] + rep + core[5:10] + rep + core[10:]
rc = "".join(comp[b] for b in reversed(core[5:10]))
seq2 = core[:5] + rep + rc + rep + core[10:]
w, t = 3, 3
L1, L2 = len(seq1), len(seq2)
c = [r"\begin{tikzpicture}[x=3.1mm,y=-3.1mm]"]
c.append(rf"\fill[papel!50] (0.5,0.5) rectangle ({L2+0.5},{L1+0.5});")
for k in range(0, L1+1, 1):
    pass
for i in range(L1):
    for j in range(L2):
        fw = sum(1 for k in range(-(w//2), w//2+1)
                 if 0 <= i+k < L1 and 0 <= j+k < L2 and seq1[i+k] == seq2[j+k])
        rv = sum(1 for k in range(-(w//2), w//2+1)
                 if 0 <= i+k < L1 and 0 <= j-k < L2 and seq1[i+k] == comp[seq2[j-k]])
        if fw >= t:
            c.append(rf"\fill[azulprofundo] ({j+1},{i+1}) circle (1.25mm);")
        elif rv >= t:
            c.append(rf"\fill[naranja] ({j+1},{i+1}) circle (1.25mm);")
for j, b in enumerate(seq2, 1):
    c.append(rf"\node[font=\ttfamily\bfseries\tiny,text=nuc{b}] at ({j},-0.3) {{{b}}};")
for i, b in enumerate(seq1, 1):
    c.append(rf"\node[font=\ttfamily\bfseries\tiny,text=nuc{b}] at (-0.3,{i}) {{{b}}};")
c.append(rf"\draw[base] (0.5,0.5) rectangle ({L2+0.5},{L1+0.5});")
c.append(rf"\node[etiqueta,anchor=west] at ({L2+1},3) {{diagonal\\principal:\\identidad}};")
c.append(rf"\node[etiqueta,anchor=west] at ({L2+1},{L1-4}) {{diagonales\\paralelas:\\repeticiones}};")
c.append(rf"\node[etiqueta,anchor=west,text=naranja] at ({L2+1},{L1//2+1}) {{antidiagonal:\\inversión}};")
c.append(r"\end{tikzpicture}")
(OUT / "dotplot.tex").write_text("\n".join(c))

# ---------------- BLOSUM62 como mapa de calor ---------------------------
B = substitution_matrices.load("BLOSUM62")
order = "CSTAGPDEQNHRKMILVWYF"
lines = [r"\begin{tikzpicture}[x=4.4mm,y=-4.4mm]"]
def colr(v):
    if v > 0:  return f"azul!{min(100,int(12+v*9))}"
    if v < 0:  return f"rojo!{min(100,int(10-v*14))}"
    return "papel"
for i, a in enumerate(order):
    for j, b in enumerate(order):
        v = int(B[a][b])
        txt = "white" if abs(v) >= 5 else "tinta"
        lines.append(rf"\node[fill={colr(v)},minimum size=4.3mm,inner sep=0pt,font=\sffamily\fontsize{{5}}{{5}}\selectfont,text={txt}] at ({j},{i}) {{{v}}};")
    lines.append(rf"\node[font=\ttfamily\bfseries\scriptsize] at (-1,{i}) {{{a}}};")
    lines.append(rf"\node[font=\ttfamily\bfseries\scriptsize] at ({i},-1) {{{a}}};")
groups = [(0,1,"C"),(1,4,"pequeños"),(4,5,""),(5,9,"polares/ácidos"),(9,12,"básicos"),(12,16,"hidrofóbicos"),(16,20,"aromáticos")]
for a0, a1, name in [(1,5,"pequeños"),(5,9,"ácidos/amidas"),(9,12,"básicos"),(12,16,"alifáticos"),(16,20,"aromáticos")]:
    lines.append(rf"\draw[tinta,thick] ({a0-0.5},{a0-0.5}) rectangle ({a1-0.5},{a1-0.5});")
    lines.append(rf"\node[etiqueta,anchor=west] at (20.2,{(a0+a1)/2-0.5}) {{{name}}};")
lines.append(r"\end{tikzpicture}")
(OUT / "blosum62.tex").write_text("\n".join(lines))
print(seq1, seq2)


# =====================================================================
#  Ampliación del capítulo: figuras de datos adicionales
# =====================================================================
import math
import numpy as np
from scipy.optimize import brentq
from scipy import stats
from scipy.linalg import expm

cifras = []                      # números citados en el texto
def anota(k, v):
    cifras.append(f"{k} = {v}")

AA = "ARNDCQEGHILKMFPSTWYV"
# Frecuencias de fondo de Robinson y Robinson (las que usa BLAST).
BG = np.array([.07805, .05129, .04487, .05364, .01925, .04264, .06295,
               .07377, .02199, .05142, .09019, .05744, .02243, .03856,
               .05203, .07120, .05841, .01330, .03216, .06441])
BG = BG / BG.sum()
IDX = {a: i for i, a in enumerate(AA)}

def matriz(nombre):
    Bm = substitution_matrices.load(nombre)
    return np.array([[Bm[a][b] for b in AA] for a in AA], float)

def lam_de(S, p=BG):
    return brentq(lambda l: (np.outer(p, p)*np.exp(l*S)).sum() - 1, 1e-4, 10)

def H_bits(S, p=BG):
    l = lam_de(S, p)
    q = np.outer(p, p)*np.exp(l*S)
    return (q*l*S).sum()/math.log(2)

B62 = matriz("BLOSUM62")
LAM62 = lam_de(B62)
anota("lambda BLOSUM62 (nats)", round(LAM62, 4))
anota("H BLOSUM62 (bits, fondo Robinson)", round(H_bits(B62), 3))
anota("E[s] BLOSUM62", round((np.outer(BG, BG)*B62).sum(), 3))

def dat(nombre, cabecera, filas):
    with open(OUT / nombre, "w") as fh:
        fh.write(cabecera + "\n")
        for f in filas:
            fh.write(" ".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                              for v in f) + "\n")

# ---------------- 1. Dot plot real: hemoglobinas y calmodulina ------------
HBA = ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"
       "KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP"
       "AVHASLDKFLASVSTVLTSKYR")                         # UniProt P69905
HBB = ("MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
       "VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG"
       "KEFTPPVQAAYQKVVAGVANALAHKYH")                      # UniProt P68871
CALM = ("MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADG"
        "NGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDE"
        "EVDEMIREADIDGDGQVNYEEFVQMMTAK")                  # UniProt P0DP23
W_P = 11
rng = np.random.default_rng(3)
# umbral: cuantil 0,999 de la puntuación de ventanas aleatorias
ra = rng.choice(20, size=(200000, W_P), p=BG)
rb = rng.choice(20, size=(200000, W_P), p=BG)
nul = B62[ra, rb].sum(1)
T_P = int(np.quantile(nul, 0.999))
anota("dotplot proteinas: w", W_P)
anota("dotplot proteinas: umbral (cuantil 0.999)", T_P)

def dot_blosum(x, y, w=W_P, t=T_P):
    X = np.array([IDX[c] for c in x]); Y = np.array([IDX[c] for c in y])
    S = B62[X[:, None], Y[None, :]]
    h = w // 2
    acc = np.full(S.shape, -99.0)
    n, m = S.shape
    for i in range(h, n - h):
        for j in range(h, m - h):
            acc[i, j] = sum(S[i+k, j+k] for k in range(-h, h+1))
    return np.argwhere(acc >= t)

for nombre, (a, b) in {"dot_hb.dat": (HBA, HBB),
                       "dot_calm.dat": (CALM, CALM)}.items():
    pts = dot_blosum(a, b)
    dat(nombre, "x y", [(int(j)+1, int(i)+1) for i, j in pts])
    anota(f"{nombre}: puntos", len(pts))
anota("long HBA HBB CALM", (len(HBA), len(HBB), len(CALM)))

# identidad global HBA-HBB (Needleman-Wunsch con BLOSUM62, 11/1)
from Bio import Align
al = Align.PairwiseAligner(mode="global", substitution_matrix=
                           substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-11, extend_gap_score=-1)
aln = al.align(HBA, HBB)[0]
ident = sum(1 for u, v in zip(aln[0], aln[1]) if u == v and u != "-")
anota("HBA-HBB global: score, identidades, columnas",
      (aln.score, ident, aln.shape[1]))

# ---------------- 2. Efecto de la ventana (ADN simulado) + ruido ----------
rng = np.random.default_rng(11)
x = "".join(rng.choice(list("ACGT"), 100))
y = list(x)
for k in range(len(y)):                           # 15 % de sustituciones
    if rng.random() < 0.15:
        y[k] = rng.choice([c for c in "ACGT" if c != y[k]])
y = "".join(y)
y = y[:30] + y[38:70] + "".join(rng.choice(list("ACGT"), 6)) + y[70:]
for w, t in [(1, 1), (7, 5), (13, 9)]:
    X = np.frombuffer(x.encode(), np.uint8); Y = np.frombuffer(y.encode(), np.uint8)
    eq = (X[:, None] == Y[None, :]).astype(int)
    h = w // 2
    acc = np.zeros_like(eq)
    n, m = eq.shape
    for k in range(-h, h+1):
        sh = np.zeros_like(eq)
        i0, i1 = max(0, -k), min(n, n-k); j0, j1 = max(0, -k), min(m, m-k)
        sh[i0:i1, j0:j1] = eq[i0+k:i1+k, j0+k:j1+k]
        acc += sh
    pts = np.argwhere(acc >= t)
    dat(f"ventana_w{w}.dat", "x y", [(int(j)+1, int(i)+1) for i, j in pts])
    anota(f"ventana w={w} t={t}: puntos / celdas",
          (len(pts), n*m, round(len(pts)/(n*m), 4)))
filas = []
for t in range(1, 12):
    fila = [t]
    for w, a in [(11, .25), (11, .05)]:
        fila.append(float(stats.binom.sf(t-1, w, a)))
    filas.append(tuple(fila))
dat("ruido.dat", "t adn prot", filas)
anota("p_ruido w=11 t=7 ADN", float(stats.binom.sf(6, 11, .25)))
anota("p_ruido w=15 t=10 ADN / prot",
      (float(stats.binom.sf(9, 15, .25)), float(stats.binom.sf(9, 15, .05))))

# ---------------- 3. Conteo de alineamientos -----------------------------
def delannoy(n, m):
    return sum(math.comb(n, k)*math.comb(m, k)*2**k for k in range(min(n, m)+1))
filas = []
for n in list(range(1, 10)) + list(range(10, 301, 10)):
    filas.append((n, math.log10(math.comb(2*n, n)), math.log10(delannoy(n, n))))
dat("conteo.dat", "n binom delannoy", filas)
anota("conteo n=3: binom, delannoy", (math.comb(6, 3), delannoy(3, 3)))
anota("conteo n=300 log10: binom, delannoy",
      (round(math.log10(math.comb(600, 300)), 1), round(math.log10(delannoy(300, 300)), 1)))
anota("delannoy(2,2), (2,3)", (delannoy(2, 2), delannoy(2, 3)))

# ---------------- 4. Matriz de Smith-Waterman ---------------------------
xs, ys = "TGTTACGG", "GGTTGACTA"
MS, MMS, GS = 3, -3, 2
n, m = len(xs), len(ys)
H = [[0]*(m+1) for _ in range(n+1)]
for i in range(1, n+1):
    for j in range(1, m+1):
        s = MS if xs[i-1] == ys[j-1] else MMS
        H[i][j] = max(0, H[i-1][j-1]+s, H[i-1][j]-GS, H[i][j-1]-GS)
best = max((H[i][j], i, j) for i in range(n+1) for j in range(m+1))
_, i, j = best
path, ax, ay = [(i, j)], "", ""
while H[i][j] > 0:
    s = MS if xs[i-1] == ys[j-1] else MMS
    if H[i][j] == H[i-1][j-1] + s:
        ax, ay, i, j = xs[i-1]+ax, ys[j-1]+ay, i-1, j-1
    elif H[i][j] == H[i-1][j] - GS:
        ax, ay, i = xs[i-1]+ax, "-"+ay, i-1
    else:
        ax, ay, j = "-"+ax, ys[j-1]+ay, j-1
    path.append((i, j))
onpath = set(path)
c = [r"\begin{tikzpicture}[x=8.2mm,y=-8.2mm,every node/.style={font=\sffamily\small}]"]
for j, b in enumerate(ys, start=1):
    c.append(rf"\node[nt={b}] at ({j+1},0) {{{b}}};")
for i2, b in enumerate(xs, start=1):
    c.append(rf"\node[nt={b}] at (0,{i2+1}) {{{b}}};")
c.append(r"\node[text=gris] at (1,0) {$\varepsilon$}; \node[text=gris] at (0,1) {$\varepsilon$};")
for i2 in range(n+1):
    for j2 in range(m+1):
        v = H[i2][j2]
        if (i2, j2) in onpath:
            st = r"fill=amarillo!35,draw=amarillo!80!black"
        elif v == 0:
            st = r"fill=white,draw=rejilla"
        else:
            st = rf"fill=azul!{min(45, 4+3*v)},draw=rejilla"
        col = "gris" if v == 0 else "tinta"
        bold = r"\bfseries" if (i2, j2) in onpath else ""
        c.append(rf"\node[{st},minimum size=7.6mm,rounded corners=1pt,text={col},font=\sffamily\small{bold}] (s{i2}x{j2}) at ({j2+1},{i2+1}) {{{v}}};")
for (a, b), (p_, q_) in zip(path[:-1], path[1:]):
    c.append(rf"\draw[-{{Stealth[length=1.6mm]}},thick,naranja] (s{a}x{b}) -- (s{p_}x{q_});")
bi, bj = path[0]
c.append(rf"\draw[rojo,very thick,rounded corners=2pt] ($(s{bi}x{bj}.north west)+(-0.04,0.04)$) rectangle ($(s{bi}x{bj}.south east)+(0.04,-0.04)$);")
c.append(r"\end{tikzpicture}")
(OUT / "sw_matriz.tex").write_text("\n".join(c))
anota("SW TGTTACGG/GGTTGACTA +3/-3/d=2", (best[0], ax, ay, best[1], best[2]))

# ---------------- 5. PAM reconstruida: H(k), identidad(k), cruces --------
P250 = matriz("PAM250")
l250 = lam_de(P250)
q = np.outer(BG, BG)*np.exp(l250*P250)
q = (q + q.T)/2; q /= q.sum(); pq = q.sum(1)
Pc = q/pq[:, None]                          # P(b | a) a 250 PAM
Dh = np.sqrt(pq)
A = Dh[:, None]*Pc/Dh[None, :]; A = (A + A.T)/2
ev, U = np.linalg.eigh(A)
Lg = U @ np.diag(np.log(ev)) @ U.T          # log simetrizado
Qr = (Lg/Dh[:, None])*Dh[None, :]           # generador (rate matrix)
def cambio(tt):                             # fracción esperada de cambios
    Pt = expm(Qr*tt)
    return 1 - (pq*np.diag(Pt)).sum()
t1 = 1/250                                  # PAM1 = raíz 250-ésima
anota("PAM1 reconstruida: fracción de cambio", round(cambio(t1), 5))
anota("PAM1 cruda: elemento mínimo (antes de recortar)", float(expm(Qr*t1).min()))
anota("t para 1 % exacto (x250)", round(250*brentq(lambda tt: cambio(tt) - 0.01, 1e-6, 1), 3))
def Pk(k):
    Pt = expm(Qr*t1*k); Pt = np.clip(Pt, 1e-12, None)
    return Pt/Pt.sum(1, keepdims=True)
P1 = Pk(1)
anota("PAM1 min elemento", float(P1.min()))
anota("PAM1 diag W, C, S, N (prob. conservar)",
      tuple(round(P1[IDX[a], IDX[a]], 4) for a in "WCSN"))
def sk(k):                                  # puntuación en bits
    return np.log2(Pk(k)/pq[None, :])
def Hk(k):
    Q = pq[:, None]*Pk(k)
    return (Q*sk(k)).sum()
filas = []
for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140,
          160, 180, 200, 250, 300, 350, 400]:
    filas.append((k, Hk(k), 100*(pq*np.diag(Pk(k))).sum()))
dat("pam_curva.dat", "k H ident", filas)
for k in [1, 30, 70, 120, 250]:
    anota(f"PAM{k}: H bits, identidad %", (round(Hk(k), 3),
          round(100*(pq*np.diag(Pk(k))).sum(), 1)))
# matrices reales (fondo de Robinson)
pts = []
for nombre in ["PAM30", "PAM70", "PAM250", "BLOSUM45", "BLOSUM50",
               "BLOSUM62", "BLOSUM80", "BLOSUM90"]:
    S = matriz(nombre)
    anota(f"{nombre}: lambda, H bits, E[s] (unid.)",
          (round(lam_de(S), 4), round(H_bits(S), 3),
           round((np.outer(BG, BG)*S).sum(), 3)))
dat("pam_reales.dat", "k H", [(30, H_bits(matriz("PAM30"))),
                              (70, H_bits(matriz("PAM70"))),
                              (250, H_bits(matriz("PAM250")))])
# cruces: puntuación esperada por columna (bits) de la matriz PAM j
# sobre pares separados por k PAM
filas = []
ks = [1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200,
      220, 250, 280, 320, 360, 400]
for k in ks:
    Q = pq[:, None]*Pk(k)
    fila = [k]
    for j in [30, 120, 250]:
        fila.append(float((Q*sk(j)).sum()))
    fila.append(Hk(k))
    filas.append(tuple(fila))
dat("cruce.dat", "k pam30 pam120 pam250 opt", filas)
for k in [30, 120, 250]:
    Q = pq[:, None]*Pk(k)
    anota(f"divergencia {k}: bits/col con PAM30/120/250",
          tuple(round(float((Q*sk(j)).sum()), 3) for j in [30, 120, 250]))

# ---------------- 6. BLOSUM paso a paso (bloque de juguete) -------------
bloque = ["ASLT", "ASLT", "ATLS", "SSIT", "ATLT", "SAIT"]
anota("bloque BLOSUM juguete", bloque)
from collections import Counter
f = Counter()
for col in zip(*bloque):
    for a in range(len(col)):
        for b in range(a+1, len(col)):
            f[tuple(sorted((col[a], col[b])))] += 1
tot = sum(f.values())
letras = sorted(set("".join(bloque)))
qx = {k: v/tot for k, v in f.items()}
pl = {a: qx.get((a, a), 0) + sum(v for k, v in qx.items()
      if a in k and k[0] != k[1])/2 for a in letras}
anota("BLOSUM juguete: pares f", dict(f))
anota("BLOSUM juguete: total pares", tot)
anota("BLOSUM juguete: p", {a: round(v, 4) for a, v in pl.items()})
for k in sorted(qx):
    a, b = k
    e = pl[a]*pl[b]*(1 if a == b else 2)
    anota(f"  par {a}{b}: q={qx[k]:.4f} e={e:.4f} s=2log2={2*math.log2(qx[k]/e):.3f}", "")

# agrupamiento al 62 % (enlace simple) y pesos 1/|cluster|
def ident(u, v):
    return sum(a == b for a, b in zip(u, v))/len(u)
grupos = [[k] for k in range(len(bloque))]
cambio_ = True
while cambio_:
    cambio_ = False
    for g1 in range(len(grupos)):
        for g2 in range(g1+1, len(grupos)):
            if any(ident(bloque[a], bloque[b]) >= 0.62
                   for a in grupos[g1] for b in grupos[g2]):
                grupos[g1] += grupos.pop(g2); cambio_ = True; break
        if cambio_: break
anota("BLOSUM62 juguete: grupos", [[bloque[k] for k in g] for g in grupos])
fw = Counter()
for col in zip(*bloque):
    for g1 in range(len(grupos)):
        for g2 in range(g1+1, len(grupos)):
            for a in grupos[g1]:
                for b in grupos[g2]:
                    fw[tuple(sorted((col[a], col[b])))] += 1/(len(grupos[g1])*len(grupos[g2]))
anota("BLOSUM62 juguete: pares ponderados", {k: round(v, 3) for k, v in fw.items()})

# ---------------- 7. Vecindad de palabras vs T ---------------------------
import itertools
todas = np.array(list(itertools.product(range(20), repeat=3)))
def vecindad(pal):
    idx = [IDX[c] for c in pal]
    return B62[idx[0], todas[:, 0]] + B62[idx[1], todas[:, 1]] + B62[idx[2], todas[:, 2]]
sc = vecindad("PQG")
rng = np.random.default_rng(5)
muestras = rng.choice(20, size=(3000, 3), p=BG)
medias = {}
allsc = [B62[a, todas[:, 0]] + B62[b, todas[:, 1]] + B62[c3, todas[:, 2]]
         for a, b, c3 in muestras]
allsc = np.array(allsc)
wdb = BG[todas[:, 0]]*BG[todas[:, 1]]*BG[todas[:, 2]]
filas = []
for T in range(8, 21):
    n_pqg = int((sc >= T).sum())
    n_med = float((allsc >= T).sum(1).mean())
    # prob. de que una palabra aleatoria de la base sea vecina (promedio)
    ph = float(((allsc >= T)*wdb[None, :]).sum(1).mean())
    filas.append((T, n_pqg if n_pqg > 0 else "nan", n_med, ph, ph*1e8))
    anota(f"T={T}: vecinos PQG, medio, prob hit por posicion",
          (n_pqg, round(n_med, 1), f"{ph:.2e}"))
dat("vecindad.dat", "T pqg media phit hits", filas)
orden = np.argsort(-sc)
anota("PQG vecinos T>=13",
      [("".join(AA[k] for k in todas[o]), int(sc[o])) for o in orden if sc[o] >= 13])

# ---------------- 8. lambda para ADN -------------------------------------
filas = []
for L in np.linspace(0, 2.0, 81):
    fila = [float(L)]
    for ma, mi in [(1, -1), (1, -2), (1, -3), (2, -3)]:
        fila.append(0.25*math.exp(L*ma) + 0.75*math.exp(L*mi) - 1)
    filas.append(tuple(fila))
dat("lambda_adn.dat", "l s11 s12 s13 s23", filas)
for ma, mi in [(1, -1), (1, -2), (1, -3), (2, -3), (5, -4)]:
    lam = brentq(lambda L: 0.25*math.exp(L*ma) + 0.75*math.exp(L*mi) - 1, 1e-6, 5)
    q_id = 0.25*math.exp(lam*ma)
    Hn = lam*(q_id*ma + (1-q_id)*mi)
    anota(f"ADN +{ma}/{mi}: lambda, q_ident (frac. identidad objetivo), H bits",
          (round(lam, 4), round(q_id, 3), round(Hn/math.log(2), 3)))

# ---------------- 9. Distribución de Gumbel por simulación ---------------
rng = np.random.default_rng(2024)
NS, LEN = 10000, 200
maxs = np.empty(NS)
for b0 in range(0, NS, 500):
    X = rng.choice(20, size=(500, LEN), p=BG)
    Y = rng.choice(20, size=(500, LEN), p=BG)
    Hprev = np.zeros((500, LEN)); best = np.zeros(500)
    for i in range(LEN):
        s_row = B62[X[:, i][:, None], Y]           # (500, LEN)
        Hn = np.zeros((500, LEN))
        Hn[:, 0] = np.maximum(0, s_row[:, 0])
        Hn[:, 1:] = np.maximum(0, Hprev[:, :-1] + s_row[:, 1:])
        best = np.maximum(best, Hn.max(1))
        Hprev = Hn
    maxs[b0:b0+500] = best
mu, beta = stats.gumbel_r.fit(maxs)
lam_hat = 1/beta
K0, H0 = 0.134, 0.401          # valores que imprime BLAST+ (BLOSUM62, sin huecos)
ell = math.log(K0*LEN*LEN)/H0
mef = LEN - ell
mu_teo = math.log(K0*LEN*LEN)/LAM62
mu_ef = math.log(K0*mef*mef)/LAM62
K_hat = math.exp(lam_hat*mu)/(mef*mef)
anota("Gumbel sim: N, L", (NS, LEN))
anota("Gumbel sim: media, desv, min, max",
      (round(maxs.mean(), 2), round(maxs.std(), 2), maxs.min(), maxs.max()))
anota("Gumbel ajuste: mu, beta, lambda_hat", (round(mu, 2), round(beta, 3), round(lam_hat, 4)))
anota("Gumbel teoria: mu sin corr, ell, m', mu con L efectiva",
      (round(mu_teo, 2), round(ell, 1), round(mef, 1), round(mu_ef, 2)))
anota("K estimado con lambda_hat y m'", round(K_hat, 3))
anota("P(S>=40) empirica vs teorica (Lef)",
      (float((maxs >= 40).mean()),
       float(1-math.exp(-K0*mef*mef*math.exp(-LAM62*40)))))
edges = np.arange(maxs.min()-0.5, maxs.max()+1.5, 1)
hist, _ = np.histogram(maxs, bins=edges, density=True)
dat("gumbel_hist.dat", "x d", [(float(e+0.5), float(h)) for e, h in zip(edges[:-1], hist)])
xs_ = np.linspace(10, 60, 201)
dat("gumbel_fit.dat", "x ajuste teoria",
    [(float(v), float(stats.gumbel_r.pdf(v, mu, beta)),
      float(stats.gumbel_r.pdf(v, mu_ef, 1/LAM62))) for v in xs_])

(OUT / "cifras.txt").write_text("\n".join(map(str, cifras)) + "\n")
print("\n".join(map(str, cifras)))
