"""Genera datos y figuras del capítulo 7 (mapeo de lecturas).

Ejecutar desde libro/:  python3 figuras/cap07/generar.py
Escribe .tex/.dat en figuras/cap07/ e imprime (y guarda en cifras.txt)
todas las cifras que se citan en el texto: es la única fuente de verdad.
"""
from pathlib import Path
import math
import numpy as np

OUT = Path(__file__).parent
rng = np.random.default_rng(7)
LOG = []


def w(name, text):
    (OUT / name).write_text(text)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def tt(c):
    """Carácter para TikZ (escapa el centinela)."""
    return r"\$" if c == "$" else c


# =====================================================================
# 7.1  El problema de escala: ocurrencias esperadas de un k-mero
# =====================================================================
G = 3.1e9
for k in (10, 12, 16, 20, 25, 32):
    say(f"[kmer] G=3.1e9 k={k}: E[ocurrencias al azar]={G / 4 ** k:.3g}")
say(f"[kmer] k mínimo con G/4^k<0.01: "
    f"{next(k for k in range(1, 40) if G / 4 ** k < 0.01)}")
say(f"[kmer] tabla directa de 4^k enteros de 32 bits, k=12: "
    f"{4 ** 12 * 4 / 2 ** 20:.0f} MiB;  k=16: {4 ** 16 * 4 / 2 ** 30:.0f} GiB")

# =====================================================================
# 7.1  BWT, arreglo de sufijos, C, Occ, LF y búsqueda hacia atrás
# =====================================================================
T = "TACATACAG$"
n = len(T)
rot = [T[i:] + T[:i] for i in range(n)]
SA = sorted(range(n), key=lambda i: T[i:])
M = [rot[i] for i in SA]
F = "".join(r[0] for r in M)
L = "".join(r[-1] for r in M)
say(f"[bwt] T={T}  SA={SA}  F={F}  L=BWT={L}")
ALF = "$ACGT"
C = {c: sum(1 for x in T if x < c) for c in ALF}
say("[bwt] C =", C)


def occ(c, i):
    """Número de c en L[0:i]."""
    return L[:i].count(c)


LF = [C[L[i]] + occ(L[i], i) for i in range(n)]
say("[bwt] LF =", LF)
occtab = {c: [occ(c, i) for i in range(n + 1)] for c in "ACGT"}
for c in "ACGT":
    say(f"[bwt] Occ({c}, i) i=0..{n}:", occtab[c])

# Reconstrucción de T desde L (recorrido LF desde la fila del centinela)
i, rec = 0, []
for _ in range(n - 1):
    rec.append(L[i])
    i = LF[i]
say("[bwt] reconstruido:", "".join(reversed(rec)) + "$")


def backward(P):
    sp, ep = 0, n
    steps = [("", sp, ep)]
    for j in range(len(P) - 1, -1, -1):
        c = P[j]
        sp, ep = C[c] + occ(c, sp), C[c] + occ(c, ep)
        steps.append((P[j:], sp, ep))
        if sp >= ep:
            break
    return steps


for P in ("ACA", "GAC", "TAC"):
    st = backward(P)
    say(f"[busqueda] P={P}:", " -> ".join(f"'{s}'[{a},{b})" for s, a, b in st),
        " posiciones:", sorted(SA[k] for k in range(st[-1][1], st[-1][2]))
        if st[-1][1] < st[-1][2] else "ninguna")
# detalle numérico para el texto
for s, a, b in backward("ACA"):
    say(f"[busqueda] ACA paso '{s}': sp={a} ep={b}")

# Figura: rotaciones y matriz ordenada
cw, rh = 0.34, 0.40


def grid_rows(rows, x0, hlF=False, hlL=False, bold=None, idx=None,
              idxlabel="", band=None, dim=False):
    out = []
    if band is not None:
        a, b = band
        if a < b:
            out.append(rf"\fill[amarillo!40,rounded corners=1.5pt] "
                       rf"({x0 - 0.2:.2f},{-a * rh + 0.2:.2f}) rectangle "
                       rf"({x0 + (n - 1) * cw + 0.2:.2f},{-(b - 1) * rh - 0.2:.2f});")
    for r, s in enumerate(rows):
        y = -r * rh
        for c, ch in enumerate(s):
            x = x0 + c * cw
            fill = ""
            if hlF and c == 0:
                fill = "azul!18"
            if hlL and c == n - 1:
                fill = "naranja!25"
            if fill:
                out.append(rf"\fill[{fill}] ({x - cw / 2:.2f},{y - rh / 2:.2f}) "
                           rf"rectangle ({x + cw / 2:.2f},{y + rh / 2:.2f});")
            style = ""
            if bold is not None and c < bold and band and band[0] <= r < band[1]:
                style = r"\bfseries\color{rojooscuro}"
            elif dim and not (band and band[0] <= r < band[1]):
                style = r"\color{gris!70}"
            out.append(rf"\node[font=\ttfamily\small] at ({x:.2f},{y:.2f}) "
                       rf"{{{style}{tt(ch)}}};")
        if idx is not None:
            out.append(rf"\node[etiqueta,anchor=east] at ({x0 - 0.25:.2f},{y:.2f}) "
                       rf"{{{idx[r]}}};")
    if idxlabel:
        out.append(rf"\node[etiqueta,anchor=east] at ({x0 - 0.25:.2f},{rh:.2f}) "
                   rf"{{{idxlabel}}};")
    return out


lines = [r"\begin{tikzpicture}"]
lines += grid_rows(rot, 0.0, idx=list(range(n)), idxlabel="$i$")
lines.append(rf"\node[font=\sffamily\bfseries\small,text=tinta] at "
             rf"({(n - 1) * cw / 2:.2f},{1.0:.2f}) {{Rotaciones de $T$}};")
xs = (n - 1) * cw + 2.3
lines.append(rf"\draw[flecha=tinta2] ({(n - 1) * cw + 0.35:.2f},{-(n - 1) * rh / 2:.2f})"
             rf" -- node[above,etiqueta]{{ordenar}} ({xs - 1.15:.2f},{-(n - 1) * rh / 2:.2f});")
lines += grid_rows(M, xs, hlF=True, hlL=True, idx=[f"{r}\\,({SA[r]})" for r in range(n)],
                   idxlabel=r"$r\,(\mathrm{SA})$")
lines.append(rf"\node[font=\sffamily\bfseries\small,text=tinta] at "
             rf"({xs + (n - 1) * cw / 2:.2f},{1.0:.2f}) {{Matriz ordenada $M$}};")
lines.append(rf"\node[etiqueta,text=azulprofundo] at ({xs:.2f},{-n * rh + 0.05:.2f}) {{$F$}};")
lines.append(rf"\node[etiqueta,text=naranja!70!black] at ({xs + (n - 1) * cw:.2f},"
             rf"{-n * rh + 0.05:.2f}) {{$L$}};")
lines.append(rf"\node[etiqueta,anchor=west,align=left] at ({xs + (n - 1) * cw + 0.45:.2f},"
             rf"{-(n - 1) * rh / 2:.2f}) {{$\mathrm{{BWT}}(T)=L$\\[2pt]{{\ttfamily\small {L.replace('$', chr(92) + '$')}}}}};")
lines.append(r"\end{tikzpicture}")
w("rotaciones.tex", "\n".join(lines) + "\n")

# Figura: columnas F y L con LF-mapping
rk_L, seen = [], {}
for ch in L:
    rk_L.append(seen.get(ch, 0))
    seen[ch] = seen.get(ch, 0) + 1
rk_F, seen = [], {}
for ch in F:
    rk_F.append(seen.get(ch, 0))
    seen[ch] = seen.get(ch, 0) + 1
lines = [r"\begin{tikzpicture}[y=0.46cm]"]
xL, xF2, xF = 3.2, 7.0, 0.0
for r in range(n):
    y = -r
    lines.append(rf"\node[etiqueta,anchor=east] at (-0.45,{y}) {{{r}}};")
    lines.append(rf"\node[draw=rejilla,fill=azul!15,minimum width=7mm,minimum height=3.8mm,"
                 rf"inner sep=0pt,font=\ttfamily\small] (F{r}) at ({xF},{y}) "
                 rf"{{{tt(F[r])}$_{{{rk_F[r]}}}$}};")
    mid = M[r][1:-1]
    lines.append(rf"\node[font=\ttfamily\scriptsize,text=gris!80] at ({(xF + xL) / 2:.2f},{y}) "
                 rf"{{{mid.replace('$', chr(92) + '$')}}};")
    lines.append(rf"\node[draw=rejilla,fill=naranja!20,minimum width=7mm,minimum height=3.8mm,"
                 rf"inner sep=0pt,font=\ttfamily\small] (L{r}) at ({xL},{y}) "
                 rf"{{{tt(L[r])}$_{{{rk_L[r]}}}$}};")
    lines.append(rf"\node[draw=rejilla,fill=azul!15,minimum width=7mm,minimum height=3.8mm,"
                 rf"inner sep=0pt,font=\ttfamily\small] (G{r}) at ({xF2},{y}) "
                 rf"{{{tt(F[r])}$_{{{rk_F[r]}}}$}};")
    lines.append(rf"\node[etiqueta,anchor=west] at ({xF2 + 0.45},{y}) {{{r}}};")
colL = {"$": "gris", "A": "nucA", "C": "nucC", "G": "nucG!85!black", "T": "nucT"}
for r in range(n):
    lines.append(rf"\draw[-{{Stealth[length=1.8mm]}},thick,{colL[L[r]]}] (L{r}.east) -- (G{LF[r]}.west);")
lines.append(rf"\node[etiqueta,text=azulprofundo] at ({xF},1.1) {{$F$}};")
lines.append(rf"\node[etiqueta,text=naranja!70!black] at ({xL},1.1) {{$L$}};")
lines.append(rf"\node[etiqueta,text=azulprofundo] at ({xF2},1.1) {{$F$}};")
lines.append(rf"\node[etiqueta] at ({(xL + xF2) / 2:.2f},1.1) {{$\mathrm{{LF}}(r)=C[L_r]+\mathrm{{Occ}}(L_r,r)$}};")
lines.append(r"\end{tikzpicture}")
w("lfmap.tex", "\n".join(lines) + "\n")

# Tabla C / Occ
rows = []
for i in range(n + 1):
    rows.append(f"{i} & " + (f"\\texttt{{{tt(L[i])}}}" if i < n else "--") + " & "
                + " & ".join(str(occtab[c][i]) for c in "ACGT") + r"\\")
w("occ_tabla.tex", "\n".join(rows) + "\n")
say("[bwt] C tabla:", " ".join(f"{c}:{C[c]}" for c in "ACGT"))

# Figura: búsqueda hacia atrás en pasos (ACA)
steps = backward("ACA")
cw = 0.255
lines = [r"\begin{tikzpicture}"]
pw = (n - 1) * cw + 0.55
for p, (suf, a, b) in enumerate(steps):
    x0 = p * (pw + 0.35) + 0.35
    lines += grid_rows(M, x0, bold=len(suf), band=(a, b), dim=True)
    tit = "inicio" if not suf else rf"\texttt{{{suf}}}"
    lines.append(rf"\node[font=\sffamily\bfseries\small,text=tinta] at "
                 rf"({x0 + (n - 1) * cw / 2:.2f},0.95) {{Paso {p}: {tit}}};")
    lines.append(rf"\node[etiqueta] at ({x0 + (n - 1) * cw / 2:.2f},{-n * rh + 0.02:.2f}) "
                 rf"{{$[\,{a},{b})$\quad {b - a} fila{'s' if b - a != 1 else ''}}};")
    if p == 0:
        for r in range(n):
            lines.append(rf"\node[etiqueta,anchor=east] at ({x0 - 0.22:.2f},{-r * rh:.2f}) {{{r}}};")
    if p > 0:
        lines.append(rf"\draw[flecha=tinta2] ({x0 - 0.62:.2f},{-(n - 1) * rh / 2:.2f}) -- ({x0 - 0.3:.2f},{-(n - 1) * rh / 2:.2f});")
lines.append(r"\end{tikzpicture}")
w("backward.tex", "\n".join(lines) + "\n")
cw = 0.34

# Memoria del índice FM humano frente al intervalo de muestreo
nG = 3.1e9
bitsw = math.ceil(math.log2(nG))
dat = ["s bwt occ sa total"]
for e in range(0, 9):
    s = 2 ** e
    bwt = 2 * nG / 8 / 1e9
    occm = 4 * 32 * nG / s / 8 / 1e9
    sam = 32 * nG / s / 8 / 1e9
    dat.append(f"{s} {bwt:.4f} {occm:.4f} {sam:.4f} {bwt + occm + sam:.4f}")
w("memoria.dat", "\n".join(dat) + "\n")
say(f"[mem] log2 n = {bitsw} bits;  SA completo 32 bit: {32 * nG / 8 / 1e9:.1f} GB;"
    f"  BWT 2 bit: {2 * nG / 8 / 1e9:.3f} GB")
for s in (1, 32, 128):
    say(f"[mem] s={s}: Occ(4x32 bit c/s) {4 * 32 * nG / s / 8 / 1e9:.3f} GB;"
        f" SA muestreado {32 * nG / s / 8 / 1e9:.3f} GB")
say(f"[mem] BWA (Li y Durbin): 2n + n*ceil(log2 n)/32 bits ="
    f" {(2 * nG + nG * bitsw / 32) / 8 / 1e9:.3f} GB por hebra")
say(f"[mem] ejemplo occ d=128, SA s=32: "
    f"{(2 * nG + 4 * 32 * nG / 128 + 32 * nG / 32) / 8 / 1e9:.3f} GB")

# =====================================================================
# 7.2  Minimizadores
# =====================================================================
ENC = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_code(s):
    v = 0
    for ch in s:
        v = v * 4 + ENC[ch]
    return v


def hash64(key, mask):
    """Hash invertible de minimap2 (hash64 de Thomas Wang)."""
    key = (~key + (key << 21)) & mask
    key = key ^ (key >> 24)
    key = ((key + (key << 3)) + (key << 8)) & mask
    key = key ^ (key >> 14)
    key = ((key + (key << 2)) + (key << 4)) & mask
    key = key ^ (key >> 28)
    key = (key + (key << 31)) & mask
    return key


def minimizers(s, k, wdw, order="hash"):
    mask = (1 << (2 * k)) - 1
    codes = [kmer_code(s[i:i + k]) for i in range(len(s) - k + 1)]
    val = [hash64(c, mask) if order == "hash" else c for c in codes]
    sel = set()
    for j in range(len(val) - wdw + 1):
        win = val[j:j + wdw]
        m = min(win)
        sel.add(j + win.index(m))
    return sorted(sel), val


seq_demo = "GATTACAGGTCCATGACATTGCAA"
kd, wd = 3, 4
sel, val = minimizers(seq_demo, kd, wd, order="lex")
say(f"[minim] demo seq={seq_demo} k={kd} w={wd} (orden lexicográfico) seleccionados={sel}",
    [seq_demo[i:i + kd] for i in sel])
nk = len(seq_demo) - kd + 1
say(f"[minim] demo: {len(sel)} de {nk} k-meros; densidad={len(sel) / nk:.3f};"
    f" 2/(w+1)={2 / (wd + 1):.3f}")
lines = [r"\begin{tikzpicture}[x=0.5cm,y=0.5cm]"]
for i, ch in enumerate(seq_demo):
    lines.append(rf"\node[nt={ch},minimum size=4.2mm,font=\ttfamily\bfseries\scriptsize] at ({i},0) {{{ch}}};")
    lines.append(rf"\node[etiqueta,font=\sffamily\tiny] at ({i},0.75) {{{i}}};")
NWIN = 6
for j in range(NWIN):
    y = -1.15 - j * 0.9
    win = val[j:j + wd]
    mi = j + win.index(min(win))
    lines.append(rf"\draw[rejilla,rounded corners=2pt] ({j - 0.5},{y + 0.38}) rectangle ({j + wd - 1 + 0.5},{y - 0.38});")
    for q in range(j, j + wd):
        if q == mi:
            lines.append(rf"\node[fill=rojo!30,rounded corners=1pt,inner sep=0.6pt,font=\ttfamily\bfseries\tiny,text=rojooscuro] at ({q},{y}) {{{seq_demo[q:q + kd]}}};")
        else:
            lines.append(rf"\node[inner sep=0.6pt,font=\ttfamily\tiny,text=gris] at ({q},{y}) {{{seq_demo[q:q + kd]}}};")
    lines.append(rf"\node[etiqueta,anchor=east] at (-0.8,{y}) {{ventana {j + 1}}};")
ysel = -1.15 - NWIN * 0.9 - 0.25
lines.append(rf"\node[etiqueta,anchor=east] at (-0.8,{ysel}) {{minimizadores}};")
lines.append(rf"\draw[rejilla] (-0.5,{ysel}) -- ({len(seq_demo) - 0.5},{ysel});")
for i in sel:
    lines.append(rf"\draw[rojooscuro,line width=1.6pt] ({i - 0.3},{ysel}) -- ({i + kd - 1 + 0.3},{ysel + 0.0});")
    lines.append(rf"\fill[rojooscuro] ({i},{ysel}) circle (0.14);")
    lines.append(rf"\node[font=\ttfamily\tiny,text=rojooscuro,anchor=north] at ({i},{ysel - 0.2}) {{{seq_demo[i:i + kd]}}};")
lines.append(r"\end{tikzpicture}")
w("minimizadores.tex", "\n".join(lines) + "\n")

# densidad en una secuencia aleatoria larga
Ls = 200_000
rs = "".join(rng.choice(list("ACGT"), Ls))
for k_, w_ in ((15, 10), (21, 11), (19, 19)):
    sh, _ = minimizers(rs, k_, w_, "hash")
    sl, _ = minimizers(rs, k_, w_, "lex")
    say(f"[minim] aleatoria {Ls} pb k={k_} w={w_}: densidad hash={len(sh) / (Ls - k_ + 1):.4f}"
        f"  lex={len(sl) / (Ls - k_ + 1):.4f}  2/(w+1)={2 / (w_ + 1):.4f}")

# =====================================================================
# 7.2  Encadenamiento de anclas (fórmula de minimap2)
# =====================================================================
Lref = 6000
ref = list(rng.choice(list("ACGT"), Lref))
ref[4200:4600] = ref[1500:1900]          # repetición: copia de 400 pb
ref = "".join(ref)
q0, q1 = 1000, 3000
qs, i = [], q0
err = 0.04
while i < q1:
    u = rng.random()
    if u < err * 0.7:
        qs.append(rng.choice([b for b in "ACGT" if b != ref[i]])); i += 1
    elif u < err * 0.85:
        qs.append(rng.choice(list("ACGT")))           # inserción
    elif u < err:
        i += 1                                         # deleción
    else:
        qs.append(ref[i]); i += 1
query = "".join(qs)
kc, wc = 15, 10
sref, vref = minimizers(ref, kc, wc)
sq, vq = minimizers(query, kc, wc)
idx = {}
for p in sref:
    idx.setdefault(vref[p], []).append(p)
anch = []
for p in sq:
    for r_ in idx.get(vq[p], []):
        if ref[r_:r_ + kc] == query[p:p + kc]:
            anch.append((r_ + kc - 1, p + kc - 1, kc))
anch.sort()
wbar = kc
Gmax = 5000


def gamma(l):
    return 0.0 if l == 0 else 0.01 * wbar * abs(l) + 0.5 * math.log2(abs(l))


f = [0.0] * len(anch)
P = [-1] * len(anch)
for a_, (x, y, wi) in enumerate(anch):
    best, bp = wi, -1
    for b_ in range(a_ - 1, max(-1, a_ - 51), -1):
        xj, yj, _ = anch[b_]
        if yj >= y or xj >= x or max(y - yj, x - xj) > Gmax:
            continue
        al = min(min(y - yj, x - xj), wi)
        sc = f[b_] + al - gamma((y - yj) - (x - xj))
        if sc > best:
            best, bp = sc, b_
    f[a_], P[a_] = best, bp
end = int(np.argmax(f))
chain = []
while end != -1:
    chain.append(end)
    end = P[end]
chain = chain[::-1]
cs = set(chain)
w("anclas.dat", "x y\n" + "\n".join(f"{x} {y}" for i_, (x, y, _) in enumerate(anch) if i_ not in cs) + "\n")
w("cadena.dat", "x y\n" + "\n".join(f"{anch[i_][0]} {anch[i_][1]}" for i_ in chain) + "\n")
say(f"[cadena] consulta {len(query)} pb, error {err}; minimizadores ref={len(sref)} consulta={len(sq)};"
    f" anclas={len(anch)}; en la cadena={len(chain)}; fuera={len(anch) - len(chain)};"
    f" puntuación f={max(f):.1f}")
say(f"[cadena] cadena ref {anch[chain[0]][0]}..{anch[chain[-1]][0]}; "
    f"consulta {anch[chain[0]][1]}..{anch[chain[-1]][1]}")

# =====================================================================
# 7.2  MAPQ
# =====================================================================


def mapq(ps):
    tot = sum(ps)
    pw = 1 - max(ps) / tot
    return -10 * math.log10(pw) if pw > 0 else float("inf"), pw


p1 = 10 ** (-35 / 10)
p2 = 10 ** (-25 / 10) * 10 ** (-30 / 10)
q, pw = mapq([p1, p2])
say(f"[mapq] p1={p1:.3g} p2={p2:.3g}  P(error)={pw:.4g}  MAPQ={q:.1f}")
q, pw = mapq([p1, p1])
say(f"[mapq] dos idénticos: P(error)={pw}  MAPQ={q:.2f}")
q, pw = mapq([p1, p1, p1])
say(f"[mapq] tres idénticos: P(error)={pw:.4f}  MAPQ={q:.2f}")
p3 = 10 ** (-20 / 10)
q, pw = mapq([p1, p3 * 10 ** (-30 / 10)])
say(f"[mapq] p1 vs (Q20,Q30): MAPQ={q:.1f}")
for f1, f2, m in ((200, 0, 40), (200, 150, 40), (200, 195, 40)):
    mq = 40 * (1 - f2 / f1) * min(1, m / 10) * math.log(f1)
    say(f"[mapq-mm2] f1={f1} f2={f2} m={m}: mapQ={mq:.1f} (tope 60 -> {min(60, mq):.0f})")

# =====================================================================
# 7.2  Tamaño de inserto
# =====================================================================
mu, sd = 350, 40
ins = np.concatenate([rng.normal(mu, sd, 20000),
                      rng.normal(mu + 1200, sd, 150)])
h, edges = np.histogram(ins, bins=np.arange(100, 1800, 20))
w("inserto.dat", "x y\n" + "\n".join(f"{(edges[i] + 10):.0f} {h[i] / len(ins) / 20:.6g}"
                                        for i in range(len(h))) + "\n")
say(f"[inserto] media={mu} sd={sd}; mu+4sd={mu + 4 * sd}; mu-4sd={mu - 4 * sd};"
    f" P(>mu+4sd | normal)={0.5 * math.erfc(4 / math.sqrt(2)):.2e}")

# =====================================================================
# 7.3  Algoritmo de mosdepth (ejemplo pequeño)
# =====================================================================
Lm = 14
bloques = [[(1, 7)], [(3, 6), (8, 12)], [(5, 11)], [(9, 14)]]   # semiabiertos
d = np.zeros(Lm + 1, dtype=int)
for bl in bloques:
    for a, b in bl:
        d[a] += 1
        d[b] -= 1
cov = np.cumsum(d[:-1])
say("[mosdepth] delta =", d[:-1].tolist(), " cobertura =", cov.tolist())
lines = [r"\begin{tikzpicture}[x=0.72cm,y=0.5cm]"]
cols = ["azul", "aqua", "magenta", "violeta"]
for r, bl in enumerate(bloques):
    y = -r * 0.9
    for t_, (a, b) in enumerate(bl):
        lines.append(rf"\fill[{cols[r]}!70,rounded corners=1pt] ({a},{y - 0.25}) rectangle ({b},{y + 0.25});")
    if len(bl) > 1:
        lines.append(rf"\draw[tinta,thick] ({bl[0][1]},{y}) -- ({bl[1][0]},{y}) "
                     rf"node[midway,above,etiqueta,font=\sffamily\tiny] {{D}};")
    lines.append(rf"\node[etiqueta,anchor=east] at (-0.2,{y}) {{lectura {r + 1}}};")
yd = -len(bloques) * 0.9 - 0.6
yc = yd - 1.2
for i in range(Lm):
    lines.append(rf"\draw[rejilla] ({i},{yd - 0.4}) rectangle ({i + 1},{yd + 0.4});")
    v = d[i]
    col = "verde!50!black" if v > 0 else ("rojooscuro" if v < 0 else "gris")
    txt = f"+{v}" if v > 0 else str(v)
    lines.append(rf"\node[font=\sffamily\scriptsize,text={col}] at ({i + 0.5},{yd}) {{{txt}}};")
    lines.append(rf"\fill[azul!{15 + 20 * cov[i]}] ({i},{yc - 0.4}) rectangle ({i + 1},{yc + 0.4});")
    lines.append(rf"\draw[rejilla] ({i},{yc - 0.4}) rectangle ({i + 1},{yc + 0.4});")
    lines.append(rf"\node[font=\sffamily\scriptsize\bfseries,text={'white' if cov[i] >= 3 else 'tinta'}] at ({i + 0.5},{yc}) {{{cov[i]}}};")
    lines.append(rf"\node[etiqueta,font=\sffamily\tiny] at ({i + 0.5},{yc - 0.75}) {{{i}}};")
lines.append(rf"\node[etiqueta,anchor=east] at (-0.2,{yd}) {{arreglo $\delta$}};")
lines.append(rf"\node[etiqueta,anchor=east] at (-0.2,{yc}) {{$c=\sum\delta$}};")
lines.append(rf"\node[etiqueta,anchor=east] at (-0.2,{yc - 0.75}) {{posición}};")
lines.append(r"\end{tikzpicture}")
w("mosdepth.tex", "\n".join(lines) + "\n")

# =====================================================================
# 7.3  Pileup estilo IGV
# =====================================================================
refp = "".join(rng.choice(list("ACGT"), 44))
snv = 21
alt = [b for b in "ACGT" if b != refp[snv]][1]
# (inicio, longitud, hebra, lleva_alt, errores{pos:base}, deleción(inicio,long), softclip_izq)
reads = [
    (0, 24, "+", False, {}, None, 0),
    (2, 26, "-", True, {}, None, 0),
    (5, 24, "+", False, {9: None}, None, 0),
    (8, 26, "+", True, {}, (28, 2), 0),
    (11, 24, "-", False, {}, None, 0),
    (13, 26, "+", True, {}, None, 0),
    (15, 24, "-", True, {}, None, 4),
    (17, 26, "+", False, {}, None, 0),
    (19, 24, "-", True, {33: None}, None, 0),
    (26, 18, "+", False, {}, None, 0),
    (1, 16, "-", False, {}, None, 0),
    (28, 16, "-", False, {}, None, 0),
]
lanes = []
placed = []
for rd in reads:
    a, ln = rd[0] - rd[6], rd[1]
    b = rd[0] + ln + (rd[5][1] if rd[5] else 0)
    for li, ends in enumerate(lanes):
        if a > ends + 1:
            lanes[li] = b
            placed.append((li, rd))
            break
    else:
        lanes.append(b)
        placed.append((len(lanes) - 1, rd))
depth = np.zeros(len(refp), dtype=int)
altc = 0
for li, (s0, ln, st, isalt, errs, dele, sc) in placed:
    for p in range(s0, min(s0 + ln + (dele[1] if dele else 0), len(refp))):
        if dele and dele[0] <= p < dele[0] + dele[1]:
            continue
        depth[p] += 1
        if p == snv and isalt:
            altc += 1
say(f"[pileup] SNV en posición {snv}: ref={refp[snv]} alt={alt}; profundidad={depth[snv]};"
    f" alt={altc}; frecuencia alélica={altc / depth[snv]:.2f}")
lines = [r"\begin{tikzpicture}[x=0.25cm,y=0.38cm]"]
dmax = depth.max()
for p in range(len(refp)):
    hgt = 2.2 * depth[p] / dmax
    if p == snv:
        fa = altc / depth[p]
        lines.append(rf"\fill[nuc{alt}] ({p + 0.05},0) rectangle ({p + 0.95},{hgt * fa:.2f});")
        lines.append(rf"\fill[nuc{refp[snv]}] ({p + 0.05},{hgt * fa:.2f}) rectangle ({p + 0.95},{hgt:.2f});")
    else:
        lines.append(rf"\fill[gris!45] ({p + 0.05},0) rectangle ({p + 0.95},{hgt:.2f});")
lines.append(r"\node[etiqueta,anchor=east] at (-0.4,1.1) {cobertura};")
yref = -1.0
lines.append(rf"\node[etiqueta,anchor=east] at (-0.4,{yref}) {{referencia}};")
for p, ch in enumerate(refp):
    lines.append(rf"\node[font=\ttfamily\bfseries\tiny,text=nuc{ch}] at ({p + 0.5},{yref}) {{{ch}}};")
for li, (s0, ln, st, isalt, errs, dele, sc) in placed:
    y = -2.3 - li * 0.95
    e_ = s0 + ln + (dele[1] if dele else 0)
    e_ = min(e_, len(refp))
    col = "azul!22" if st == "+" else "rojo!18"
    if dele:
        lines.append(rf"\fill[{col}] ({s0},{y - 0.32}) rectangle ({dele[0]},{y + 0.32});")
        lines.append(rf"\draw[tinta,thick] ({dele[0]},{y}) -- ({dele[0] + dele[1]},{y});")
        lines.append(rf"\fill[{col}] ({dele[0] + dele[1]},{y - 0.32}) rectangle ({e_},{y + 0.32});")
        lines.append(rf"\node[font=\sffamily\tiny\bfseries,text=tinta] at ({dele[0] + dele[1] / 2},{y + 0.62}) {{{dele[1]}}};")
    else:
        lines.append(rf"\fill[{col}] ({s0},{y - 0.32}) rectangle ({e_},{y + 0.32});")
    # punta de flecha para la hebra
    if st == "+":
        lines.append(rf"\fill[{col}] ({e_},{y - 0.32}) -- ({e_ + 0.45},{y}) -- ({e_},{y + 0.32}) -- cycle;")
    else:
        lines.append(rf"\fill[{col}] ({s0},{y - 0.32}) -- ({s0 - 0.45},{y}) -- ({s0},{y + 0.32}) -- cycle;")
    if sc:
        for q_ in range(s0 - sc, s0):
            bb = rng.choice(list("ACGT"))
            lines.append(rf"\node[font=\ttfamily\bfseries\tiny,text=nuc{bb}!45] at ({q_ + 0.5},{y}) {{{bb}}};")
        lines.append(rf"\draw[gris,densely dotted] ({s0 - sc},{y - 0.32}) rectangle ({s0},{y + 0.32});")
    if isalt and s0 <= snv < e_:
        lines.append(rf"\node[font=\ttfamily\bfseries\tiny,text=nuc{alt}] at ({snv + 0.5},{y}) {{{alt}}};")
    for pe in errs:
        bb = [b for b in "ACGT" if b != refp[pe]][0]
        lines.append(rf"\node[font=\ttfamily\bfseries\tiny,text=nuc{bb}] at ({pe + 0.5},{y}) {{{bb}}};")
ybot = -2.3 - (len(lanes) - 1) * 0.95 - 0.8
lines.append(rf"\draw[amarillo!80!black,thick,dashed,rounded corners=1pt] ({snv},{ybot}) rectangle ({snv + 1},2.5);")
lines.append(rf"\node[etiqueta,anchor=south,text=amarillo!40!black] at ({snv + 0.5},2.5) {{SNV heterocigoto}};")
lines.append(r"\end{tikzpicture}")
w("pileup.tex", "\n".join(lines) + "\n")
say(f"[pileup] carriles={len(lanes)}; ref={refp}")

# =====================================================================
# 7.3  Perfil de cobertura simulado con CNV y repetición
# =====================================================================
Lg = 100_000
Lr = 150
c_obj = 30.0
cn = np.full(Lg, 2.0)
cn[30_000:34_000] = 1          # deleción heterocigota 4 kb
cn[55_000:56_500] = 0          # deleción homocigota 1,5 kb
cn[72_000:78_000] = 3          # duplicación heterocigota 6 kb
rep = (slice(88_000, 90_000))  # repetición: lecturas con MAPQ 0
# sesgo suave (tipo GC) multiplicativo
xg = np.arange(Lg)
bias = np.exp(0.12 * np.sin(2 * np.pi * xg / 17_000) + 0.08 * np.sin(2 * np.pi * xg / 5_300 + 1))
rate = c_obj / Lr * cn / 2 * bias
rate_starts = np.convolve(rate, np.ones(Lr) / Lr, mode="same")
starts = rng.poisson(rate_starts)
dall = np.zeros(Lg + Lr + 1, dtype=np.int64)
dq = np.zeros(Lg + Lr + 1, dtype=np.int64)
pos = np.repeat(np.arange(Lg), starts)
for s0 in pos:
    dall[s0] += 1
    dall[s0 + Lr] -= 1
    inrep = (s0 + Lr > rep.start) and (s0 < rep.stop)
    if not inrep:
        dq[s0] += 1
        dq[s0 + Lr] -= 1
    else:
        # parte fuera de la repetición conserva MAPQ si solapa < 50 %
        ov = min(s0 + Lr, rep.stop) - max(s0, rep.start)
        if ov < Lr / 2:
            dq[s0] += 1
            dq[s0 + Lr] -= 1
call = np.cumsum(dall)[:Lg]
cq = np.cumsum(dq)[:Lg]
say(f"[perfil] lecturas={len(pos)}; cobertura media todas={call.mean():.1f};"
    f" MAPQ>=20={cq.mean():.1f}")
B = 200
nb = Lg // B
xb = (np.arange(nb) * B + B / 2) / 1000
ball = call[:nb * B].reshape(nb, B).mean(1)
bq = cq[:nb * B].reshape(nb, B).mean(1)
w("perfil.dat", "x all q\n" + "\n".join(f"{xb[i]:.2f} {ball[i]:.2f} {bq[i]:.2f}" for i in range(nb)) + "\n")
Wn = 1000
nw = Lg // Wn
wm = cq[:nw * Wn].reshape(nw, Wn).mean(1)
med = np.median(wm)
lr = np.log2(np.maximum(wm, 0.25) / med)
xw = (np.arange(nw) * Wn + Wn / 2) / 1000
w("log2.dat", "x lr\n" + "\n".join(f"{xw[i]:.2f} {lr[i]:.3f}" for i in range(nw)) + "\n")
for nm, sl in (("normal", slice(0, 30)), ("het", slice(30, 34)), ("hom", slice(55, 56)),
               ("dup", slice(72, 78)), ("rep", slice(88, 90))):
    say(f"[perfil] ventanas 1 kb {nm}: media={wm[sl].mean():.1f} log2={lr[sl].mean():.2f}")
say(f"[perfil] mediana ventanas={med:.1f}; log2 teórico het={math.log2(0.5):.2f}"
    f" dup={math.log2(1.5):.2f}")
say(f"[perfil] cobertura MAPQ>=20 dentro de la repetición: {cq[88_000:90_000].mean():.1f};"
    f" todas: {call[88_000:90_000].mean():.1f}")

# =====================================================================
# 7.3  Tamaño mínimo detectable por profundidad (Poisson)
# =====================================================================
zst = 5.0
for c_ in (10, 30, 60):
    Whet = 4 * zst ** 2 * Lr / c_
    Whom = zst ** 2 * Lr / c_
    say(f"[deteccion] c={c_} L={Lr} z*={zst}: W_het={Whet:.0f} pb  W_hom={Whom:.0f} pb")
dat = ["c het hom het3"]
for c_ in np.linspace(5, 80, 31):
    dat.append(f"{c_:.2f} {4 * zst ** 2 * Lr / c_:.1f} {zst ** 2 * Lr / c_:.1f} "
               f"{4 * 3 ** 2 * Lr / c_:.1f}")
w("deteccion.dat", "\n".join(dat) + "\n")

(OUT / "cifras.txt").write_text("\n".join(LOG) + "\n")
