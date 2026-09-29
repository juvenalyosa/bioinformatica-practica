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
