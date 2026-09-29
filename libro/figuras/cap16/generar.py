"""Genera datos y figuras del capítulo 16 (biología de sistemas y redes).

Ejecutar desde libro/:  python3 figuras/cap16/generar.py  > figuras/cap16/cifras.txt
Escribe .dat/.tex en figuras/cap16/ e imprime TODAS las cifras que se citan
en el texto (única fuente de verdad).

Datos reales: red física de levadura de STRING v12.0 (taxón 4932, archivo
protein.physical.links.v12.0.txt.gz), aristas con combined_score >= 700,
nombres de protein.info.v12.0.txt.gz, cada par una sola vez. El archivo
filtrado es levadura_string700.tsv.gz (se regenera con descargar_string()).
"""
import os
import sys
if os.environ.get("PYTHONHASHSEED") != "0":      # Louvain depende del orden de
    os.environ["PYTHONHASHSEED"] = "0"           # iteración de conjuntos: fijarlo
    os.execv(sys.executable, [sys.executable] + sys.argv)
from pathlib import Path
import gzip
import math
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

OUT = Path(__file__).parent
rng = np.random.default_rng(16)
COLS = ["azul", "naranja", "aqua", "violeta", "magenta", "amarillo", "verde", "rojo"]


def w(name, text):
    (OUT / name).write_text(text)


def descargar_string():
    """Reconstruye levadura_string700.tsv.gz desde STRING (no se usa si existe)."""
    import urllib.request
    base = "https://stringdb-downloads.org/download/"
    fl = "protein.physical.links.v12.0/4932.protein.physical.links.v12.0.txt.gz"
    fi = "protein.info.v12.0/4932.protein.info.v12.0.txt.gz"
    for f in (fl, fi):
        urllib.request.urlretrieve(base + f, OUT / Path(f).name)
    names = {}
    for l in gzip.open(OUT / Path(fi).name, "rt"):
        if not l.startswith("#"):
            p = l.split("\t")
            names[p[0]] = p[1]
    with gzip.open(OUT / "levadura_string700.tsv.gz", "wt") as o:
        o.write("# STRING v12.0, 4932, physical links, score >= 700\n")
        o.write("prot1\tprot2\tscore\n")
        for l in gzip.open(OUT / Path(fl).name, "rt"):
            if l.startswith("protein1"):
                continue
            a, b, s = l.split()
            if int(s) >= 700 and a < b:
                o.write(f"{names[a]}\t{names[b]}\t{s}\n")


def tex_name(s):
    return s.replace("_", r"\_")


# =====================================================================
# 16.1 (a)  Grafo de juguete: grado, clustering, caminos, intermediación
# =====================================================================
toy = nx.Graph([("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"),
                ("D", "E"), ("D", "F"), ("E", "F"), ("F", "G")])
order = sorted(toy)
A = nx.to_numpy_array(toy, nodelist=order, dtype=int)
print("[toy] nodos", order)
print("[toy] A =\n", A)
print("[toy] grados", dict(toy.degree()))
print("[toy] clustering", {k: round(v, 3) for k, v in nx.clustering(toy).items()})
print("[toy] C medio", round(nx.average_clustering(toy), 3),
      " transitividad", round(nx.transitivity(toy), 3))
print("[toy] L medio", round(nx.average_shortest_path_length(toy), 3),
      " diámetro", nx.diameter(toy))
bc = nx.betweenness_centrality(toy, normalized=False)
print("[toy] intermediación (no normalizada)", bc)
print("[toy] A^2 diag", np.diag(A @ A), " A^3 diag/2 (triángulos x2)",
      np.diag(A @ A @ A))
print("[toy] (A^2)[A,D] =", (A @ A)[0, 3], " (A^3)[A,E] =", (A @ A @ A)[0, 4])
part = [{"A", "B", "C"}, {"D", "E", "F", "G"}]
print("[toy] Q(ABC|DEFG) =", round(nx.community.modularity(toy, part), 4))
part2 = [{"A", "B", "C", "D"}, {"E", "F", "G"}]
print("[toy] Q(ABCD|EFG) =", round(nx.community.modularity(toy, part2), 4))
# Descomposición de Q a mano
m = toy.number_of_edges()
for c in part:
    lc = toy.subgraph(c).number_of_edges()
    dc = sum(d for _, d in toy.degree(c))
    print(f"[toy] comunidad {sorted(c)}: l_c={lc}, d_c={dc},"
          f" término={lc / m - (dc / (2 * m)) ** 2:.4f}")

# =====================================================================
# 16.1 (b)  Red de levadura STRING >= 700
# =====================================================================
if not (OUT / "levadura_string700.tsv.gz").exists():
    descargar_string()
G = nx.Graph()
for l in gzip.open(OUT / "levadura_string700.tsv.gz", "rt"):
    if l.startswith("#") or l.startswith("prot1"):
        continue
    a, b, s = l.split()
    G.add_edge(a, b, score=int(s))
n, m = G.number_of_nodes(), G.number_of_edges()
deg = np.array([d for _, d in G.degree()])
gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print(f"[levadura] n={n} m={m} <k>={deg.mean():.2f} kmax={deg.max()}"
      f" componentes={nx.number_connected_components(G)} gigante={len(gc)}")
top = sorted(G.degree(), key=lambda x: -x[1])[:8]
print("[levadura] top grado", top)
C_y = nx.average_clustering(G)
print(f"[levadura] C={C_y:.3f} transitividad={nx.transitivity(G):.3f}")
src = rng.choice(list(gc), 400, replace=False)
Ls = []
for s0 in src:
    dd = nx.single_source_shortest_path_length(gc, s0)
    Ls.extend(v for v in dd.values() if v > 0)
L_y = np.mean(Ls)
print(f"[levadura] L (400 orígenes, gigante) = {L_y:.2f}  diámetro>= {max(Ls)}")
frac_ribo = np.mean([u.startswith(("RPS", "RPL")) for u, _ in
                     sorted(G.degree(), key=lambda x: -x[1])[:50]])
print(f"[levadura] fracción ribosomales entre top-50 grado: {frac_ribo:.2f}")

# Modelos nulos con igual n y m
p_er = 2 * m / (n * (n - 1))
ER = nx.gnm_random_graph(n, m, seed=1)
mba = int(round(m / n))
BA = nx.barabasi_albert_graph(n, mba, seed=1)
gcer = ER.subgraph(max(nx.connected_components(ER), key=len))
Ler = np.mean([v for s0 in rng.choice(list(gcer), 200, replace=False)
               for v in nx.single_source_shortest_path_length(gcer, s0).values() if v])
print(f"[ER] p={p_er:.5f} C={nx.average_clustering(ER):.4f} L={Ler:.2f}"
      f" teoria C=p={p_er:.4f}, L~ln n/ln<k>={math.log(n) / math.log(2 * m / n):.2f}")
print(f"[BA] m={mba} aristas={BA.number_of_edges()} C={nx.average_clustering(BA):.4f}"
      f" kmax={max(d for _, d in BA.degree())}")


def logbin(d, nb=18):
    d = np.asarray(d)
    d = d[d > 0]
    edges = np.unique(np.round(np.logspace(0, np.log10(d.max() + 1), nb)).astype(int))
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        c = ((d >= a) & (d < b)).sum()
        if c:
            rows.append((math.sqrt(a * (b - 1)) if b - 1 > a else a, c / (len(d) * (b - a))))
    return rows


for nm, gg in (("grado_levadura.dat", G), ("grado_er.dat", ER), ("grado_ba.dat", BA)):
    rows = logbin([d for _, d in gg.degree()])
    w(nm, "k pk\n" + "\n".join(f"{k:.3f} {p:.4e}" for k, p in rows))
# pendiente por mínimos cuadrados log-log (solo ilustrativo) en la cola k>=10
for nm, gg in (("levadura", G), ("BA", BA)):
    rows = np.array(logbin([d for _, d in gg.degree()]))
    sel = rows[:, 0] >= 10
    sl = np.polyfit(np.log10(rows[sel, 0]), np.log10(rows[sel, 1]), 1)[0]
    print(f"[grado] pendiente log-log k>=10 {nm}: {sl:.2f}")
# Estimador de máxima verosimilitud discreto aproximado (Clauset) con kmin=10
for nm, gg in (("levadura", G), ("BA", BA)):
    d = np.array([x for _, x in gg.degree()])
    kmin = 2 * mba
    t = d[d >= kmin]
    alpha = 1 + len(t) / np.sum(np.log(t / (kmin - 0.5)))
    print(f"[grado] alfa MV (kmin={kmin}) {nm}: {alpha:.2f}  (n_cola={len(t)})")

# =====================================================================
# 16.1 (c)  Mundo pequeño de Watts-Strogatz
# =====================================================================
ps = np.logspace(-4, 0, 17)
nws, kws = 1000, 10
C0 = nx.average_clustering(nx.watts_strogatz_graph(nws, kws, 0))
L0 = nx.average_shortest_path_length(nx.watts_strogatz_graph(nws, kws, 0))
rows = []
for p in ps:
    cs, ls = [], []
    for r in range(4):
        g = nx.connected_watts_strogatz_graph(nws, kws, p, seed=100 + r)
        cs.append(nx.average_clustering(g))
        srcs = rng.choice(nws, 150, replace=False)
        ls.append(np.mean([v for s0 in srcs for v in
                           nx.single_source_shortest_path_length(g, s0).values() if v]))
    rows.append((p, np.mean(cs) / C0, np.mean(ls) / L0))
w("ws.dat", "p C L\n" + "\n".join(f"{p:.5f} {c:.4f} {l:.4f}" for p, c, l in rows))
print(f"[WS] n={nws} k={kws} C0={C0:.3f} (teoria 3(k-2)/(4(k-1))={3 * (kws - 2) / (4 * (kws - 1)):.3f})"
      f" L0={L0:.2f} (teoria n/2k={nws / (2 * kws):.1f})")
for p, c, l in rows:
    if p in (ps[4], ps[8], ps[12]):
        print(f"[WS] p={p:.4f}: C/C0={c:.3f}  L/L0={l:.3f}")

# =====================================================================
# 16.1 (d)  Subred de la vía de feromonas (ego-red de FUS3, radio 2)
# =====================================================================
E = nx.ego_graph(G, "FUS3", radius=2)
comms = sorted(nx.community.louvain_communities(E, seed=1), key=len, reverse=True)
Q = nx.community.modularity(E, comms)
print(f"[FUS3] n={E.number_of_nodes()} m={E.number_of_edges()} comunidades="
      f"{[len(c) for c in comms]} Q={Q:.3f}")
for i, c in enumerate(comms):
    print(f"[FUS3] comunidad {i} ({COLS[i]}):", sorted(c))
btw = nx.betweenness_centrality(E)
print("[FUS3] top intermediación", sorted(btw.items(), key=lambda x: -x[1])[:5])
print("[FUS3] top grado", sorted(E.degree(), key=lambda x: -x[1])[:5])
# comparación Girvan-Newman (primer nivel de división óptimo en Q)
best = (None, -1)
for k, part in zip(range(12), nx.community.girvan_newman(E)):
    q = nx.community.modularity(E, part)
    if q > best[1]:
        best = (part, q)
print(f"[FUS3] Girvan-Newman mejor Q={best[1]:.3f} con {len(best[0])} comunidades")
# modularidad esperada en redes aleatorizadas con igual grado
qs = []
for r in range(20):
    R = nx.double_edge_swap(nx.Graph(E), nswap=4 * E.number_of_edges(),
                            max_tries=10 ** 5, seed=r)
    qs.append(nx.community.modularity(R, nx.community.louvain_communities(R, seed=1)))
print(f"[FUS3] Q en 20 aleatorizaciones: {np.mean(qs):.3f} +- {np.std(qs):.3f}")

pos = nx.kamada_kawai_layout(E)
P = np.array(list(pos.values()))
mn, mx = P.min(0), P.max(0)
sc = np.array([11.4, 7.6]) / (mx - mn)
cid = {u: i for i, c in enumerate(comms) for u in c}
dE = dict(E.degree())
lines = []
for u, v in E.edges():
    same = cid[u] == cid[v]
    lines.append(rf"\draw[{'rejilla!70!gris' if same else 'tinta2'},"
                 rf"{'line width=0.35pt' if same else 'line width=0.7pt,densely dashed'}]"
                 rf" (n-{u}) -- (n-{v});")
nodes = []
for u in E:
    x, y = (np.array(pos[u]) - mn) * sc
    r = 1.2 + 0.28 * math.sqrt(dE[u])
    ang = math.degrees(math.atan2(y - 3.8, x - 5.7))
    nodes.append(rf"\node[circle,draw=white,line width=0.4pt,fill={COLS[cid[u]]},"
                 rf"minimum size={2 * r:.1f}mm,inner sep=0pt] (n-{u}) at ({x:.2f},{y:.2f}) {{}};")
    nodes.append(rf"\node[font=\sffamily\tiny,text=tinta,inner sep=1pt,anchor={(ang + 180) % 360:.0f}]"
                 rf" at (n-{u}.{ang:.0f}) {{{tex_name(u)}}};")
w("red_fus3.tex", "\n".join(nodes) + "\n\\begin{scope}[on background layer]\n"
  + "\n".join(lines) + "\n\\end{scope}\n")

# =====================================================================
# 16.2 (a)  Tiempo de respuesta: regulación simple vs autorregulación negativa
# =====================================================================
alpha = 1.0                       # tasa de eliminación (1/tiempo de generación)
beta = 1.0
xst = beta / alpha
t = np.linspace(0, 5, 251)
x_simple = xst * (1 - np.exp(-alpha * t))
K, nH = 0.3, 2                     # represión cooperativa
# producción máxima ajustada para que el estado estacionario sea el mismo
beta_nar = alpha * xst * (1 + (xst / K) ** nH)
sol = solve_ivp(lambda _t, x: [beta_nar / (1 + (x[0] / K) ** nH) - alpha * x[0]],
                (0, 5), [0], t_eval=t, rtol=1e-9, atol=1e-12, method="LSODA")
x_nar = sol.y[0]
t12s = math.log(2) / alpha
t12n = t[np.argmax(x_nar >= 0.5 * xst)]
fine = solve_ivp(lambda _t, x: [beta_nar / (1 + (x[0] / K) ** nH) - alpha * x[0]],
                 (0, 1), [0], dense_output=True, rtol=1e-10, atol=1e-12, method="LSODA")
t12n = brentq(lambda s: fine.sol(s)[0] - 0.5 * xst, 1e-6, 1)
print(f"[NAR] beta_nar/beta = {beta_nar / beta:.0f}; t1/2 simple = {t12s:.3f};"
      f" t1/2 NAR = {t12n:.4f}; aceleración = {t12s / t12n:.1f}x")
print(f"[NAR] aprox. crecimiento lineal: 0.5 xst / beta_nar = {0.5 * xst / beta_nar:.4f}")
for K2, n2 in ((0.25, 1), (0.1, 1), (0.2, 2)):
    b2 = alpha * xst * (1 + (xst / K2) ** n2)
    f2 = solve_ivp(lambda _t, x: [b2 / (1 + (x[0] / K2) ** n2) - alpha * x[0]],
                   (0, 1), [0], dense_output=True, rtol=1e-10, atol=1e-12, method="LSODA")
    t2 = brentq(lambda s_: f2.sol(s_)[0] - 0.5 * xst, 1e-6, 1)
    print(f"[NAR] variante K={K2} n={n2}: beta/beta_simple={b2:.1f} t1/2={t2:.4f}"
          f" aceleración={t12s / t2:.1f}x")
# Rosenfeld: logarítmico, tiempo en generaciones con alpha = ln2/tau
w("nar.dat", "t simple nar\n" + "\n".join(f"{a:.3f} {b:.4f} {c:.4f}"
                                         for a, b, c in zip(t, x_simple, x_nar)))
# Ejemplo numérico: tiempo de generación 30 min
for tau in (30.0,):
    a = math.log(2) / tau
    print(f"[NAR] tau={tau} min -> alpha={a:.4f}/min; t1/2 = {math.log(2) / a:.1f} min")
    for half in (5, 60):
        print(f"[NAR] vida media activa {half} min + dilución: alpha_total="
              f"{a + math.log(2) / half:.4f}/min; t1/2 = {math.log(2) / (a + math.log(2) / half):.1f} min")

# =====================================================================
# 16.2 (b)  FFL coherente tipo 1 con compuerta AND: retardo sensible al signo
# =====================================================================
def ffl(tt, s_on, s_off, Ky=0.5, Kz=0.5, Kxz=0.5, h=4):
    def f(_t, y):
        Y, Z = y
        X = 1.0 if s_on <= _t < s_off else 0.0
        fy = (X / 0.1) ** h / (1 + (X / 0.1) ** h)
        gz = ((X / Kxz) ** h / (1 + (X / Kxz) ** h)) * ((Y / Kz) ** h / (1 + (Y / Kz) ** h))
        return [fy - Y, gz - Z]
    s = solve_ivp(f, (tt[0], tt[-1]), [0, 0], t_eval=tt, max_step=0.01, rtol=1e-8)
    return s.y


tt = np.linspace(0, 8, 401)
Yc, Zc = ffl(tt, 1.0, 1.5)        # pulso corto
Yl, Zl = ffl(tt, 1.0, 5.0)        # pulso largo
# control: regulación simple de Z por X (sin Y)
Zs = np.where(tt < 1, 0, np.where(tt < 5, 1 - np.exp(-(tt - 1)),
                                  (1 - np.exp(-4)) * np.exp(-(tt - 5))))
w("ffl.dat", "t Xc Yc Zc Xl Yl Zl Zs\n" + "\n".join(
    f"{a:.3f} {1.0 if 1 <= a < 1.5 else 0.0} {b:.4f} {c:.4f} {1.0 if 1 <= a < 5 else 0.0}"
    f" {d:.4f} {e:.4f} {g:.4f}" for a, b, c, d, e, g in zip(tt, Yc, Zc, Yl, Zl, Zs)))
print(f"[FFL] pulso corto: max Z = {Zc.max():.4f}; pulso largo: max Z = {Zl.max():.3f}")
tZ = tt[np.argmax(Zl >= 0.5 * Zl.max())] - 1
print(f"[FFL] retardo ON: Y cruza Kz=0.5 en t-1 = {tt[np.argmax(Yl >= 0.5)] - 1:.2f};"
      f" Z alcanza la mitad de su máximo en t-1 = {tZ:.2f}; control simple: {math.log(2):.2f}")
tZoff = tt[np.argmax((tt > 5) & (Zl <= 0.5 * Zl.max()))] - 5
print(f"[FFL] apagado: Z cae a la mitad en t-5 = {tZoff:.2f} (sin retardo, ln2={math.log(2):.2f})")

# =====================================================================
# 16.2 (c)  Interruptor biestable (Gardner et al.): plano de fases
# =====================================================================
a1 = a2 = 10.0
bb = gg_ = 2.0


def toggle(_t, y):
    u, v = y
    return [a1 / (1 + v ** bb) - u, a2 / (1 + u ** gg_) - v]


def jac(u, v):
    return np.array([[-1, -a1 * bb * v ** (bb - 1) / (1 + v ** bb) ** 2],
                     [-a2 * gg_ * u ** (gg_ - 1) / (1 + u ** gg_) ** 2, -1]])


# puntos fijos: u = a1/(1+v^2), v = a2/(1+u^2) -> una ecuación en u
F = lambda u: u - a1 / (1 + (a2 / (1 + u ** gg_)) ** bb)
grid = np.linspace(1e-4, 11, 200001)
vals = F(grid)
roots = [brentq(F, grid[i], grid[i + 1]) for i in range(len(grid) - 1)
         if vals[i] * vals[i + 1] < 0]
fps = []
for u in roots:
    v = a2 / (1 + u ** gg_)
    ev = np.linalg.eigvals(jac(u, v))
    fps.append((u, v, ev))
    print(f"[toggle] punto fijo u={u:.4f} v={v:.4f} autovalores={np.round(ev, 4)}"
          f" -> {'estable' if max(ev.real) < 0 else 'silla (inestable)'}")
    print(f"[toggle]   jacobiano={np.round(jac(u, v), 4).tolist()}")
w("toggle_fp.dat", "u v estable\n" + "\n".join(
    f"{u:.4f} {v:.4f} {int(max(e.real) < 0)}" for u, v, e in fps))
vv = np.linspace(0, 11, 221)
w("toggle_nu.dat", "v u\n" + "\n".join(f"{v:.4f} {a1 / (1 + v ** bb):.4f}" for v in vv))
uu = np.linspace(0, 11, 221)
w("toggle_nv.dat", "u v\n" + "\n".join(f"{u:.4f} {a2 / (1 + u ** gg_):.4f}" for u in uu))
# campo vectorial normalizado
qrows = []
for u in np.linspace(0.25, 10.75, 15):
    for v in np.linspace(0.25, 10.75, 15):
        du, dv = toggle(0, [u, v])
        nrm = math.hypot(du, dv)
        qrows.append(f"{u:.3f} {v:.3f} {0.45 * du / nrm:.4f} {0.45 * dv / nrm:.4f}")
w("toggle_q.dat", "u v du dv\n" + "\n".join(qrows))
inits = [(0.3, 2.0), (2.0, 0.3), (9.0, 10.5), (10.5, 9.0), (6.0, 10.8), (10.8, 5.0),
         (0.2, 0.25), (4.0, 3.9)]
trs = []
for i, y0 in enumerate(inits):
    s = solve_ivp(toggle, (0, 12), y0, t_eval=np.linspace(0, 12, 300), rtol=1e-8)
    w(f"toggle_tr{i}.dat", "u v\n" + "\n".join(f"{a:.4f} {b:.4f}" for a, b in s.y.T))
    trs.append((y0, s.y[:, -1]))
for y0, yf in trs:
    print(f"[toggle] trayectoria desde {y0} -> ({yf[0]:.2f}, {yf[1]:.2f})")
# sin cooperatividad (n=1) solo hay un punto fijo
F1 = lambda u: u - a1 / (1 + a2 / (1 + u))
r1 = [brentq(F1, grid[i], grid[i + 1]) for i in range(len(grid) - 1)
      if F1(grid[i]) * F1(grid[i + 1]) < 0]
print(f"[toggle] n=1: puntos fijos u={np.round(r1, 4)}")
# rango biestable en alfa (simétrico, n=2)
def nfp(a, n=2):
    Fa = lambda u: u - a / (1 + (a / (1 + u ** n)) ** n)
    g = np.linspace(1e-5, a + 1, 40001)
    fv = Fa(g)
    return int(np.sum(fv[:-1] * fv[1:] < 0))
acrit = brentq(lambda a: nfp(a) - 2, 1.5, 3) if False else None
for a in (1.5, 1.9, 2.0, 2.1, 3.0, 10.0):
    print(f"[toggle] alfa={a}: {nfp(a)} puntos fijos")

# =====================================================================
# 16.2 (d)  Repressilator (modelo adimensional de Elowitz y Leibler)
# =====================================================================
ra, ra0, rb, rn = 216.0, 0.216, 5.0, 2.0


def repr_(_t, y):
    m1, m2, m3, p1, p2, p3 = y
    return [-m1 + ra / (1 + p3 ** rn) + ra0,
            -m2 + ra / (1 + p1 ** rn) + ra0,
            -m3 + ra / (1 + p2 ** rn) + ra0,
            -rb * (p1 - m1), -rb * (p2 - m2), -rb * (p3 - m3)]


tr = np.linspace(0, 60, 1201)
s = solve_ivp(repr_, (0, 60), [1, 0, 0, 2, 1, 3], t_eval=tr, rtol=1e-9, atol=1e-9)
P1, P2, P3 = s.y[3], s.y[4], s.y[5]
w("repressilator.dat", "t p1 p2 p3\n" + "\n".join(
    f"{a:.3f} {b:.3f} {c:.3f} {d:.3f}" for a, b, c, d in zip(tr, P1, P2, P3)))
# periodo: máximos de p1 después de t=20
from scipy.signal import find_peaks
pk, _ = find_peaks(P1)
pk = pk[tr[pk] > 20]
per = np.diff(tr[pk]).mean()
print(f"[repr] periodo = {per:.2f} (unidades de vida media del ARNm / ln2);"
      f" amplitud p1: {P1[tr > 20].min():.2f} - {P1[tr > 20].max():.1f}")
# punto fijo simétrico y criterio de inestabilidad
pst = brentq(lambda p: p - ra / (1 + p ** rn) - ra0, 0, ra + 1)
X = ra * rn * pst ** (rn - 1) / (1 + pst ** rn) ** 2
lhs = (rb + 1) ** 2 / rb
rhs = 3 * X ** 2 / (4 - 2 * X) if X < 2 else math.inf
print(f"[repr] p*={pst:.3f}; X=-f'(p*)={X:.3f}; (b+1)^2/b={lhs:.3f}; 3X^2/(4-2X)={rhs:.3f}"
      f" -> {'oscila' if lhs < rhs else 'estable'}")
# comprobación del criterio contra autovalores en una rejilla
def maxre(Xv, bv):
    J = np.zeros((6, 6))
    for i in range(3):
        J[i, i], J[i, 3 + (i - 1) % 3], J[3 + i, 3 + i], J[3 + i, i] = -1, -Xv, -bv, bv
    return np.linalg.eigvals(J).real.max()
mis = sum((Xv >= 2 or (bv + 1) ** 2 / bv < 3 * Xv ** 2 / (4 - 2 * Xv)) != (maxre(Xv, bv) > 0)
          for Xv in np.linspace(0.1, 5, 40) for bv in np.logspace(-2, 2, 40))
print(f"[repr] criterio vs autovalores: {mis} discrepancias en 1600 casos")
Jr = np.zeros((6, 6))
for i in range(3):
    Jr[i, i] = -1
    Jr[i, 3 + (i - 1) % 3] = -X
    Jr[3 + i, 3 + i] = -rb
    Jr[3 + i, i] = rb
evr = np.linalg.eigvals(Jr)
print(f"[repr] autovalores del jacobiano en p*: max Re = {evr.real.max():.3f};"
      f" par dominante = {np.round(evr[np.argmax(evr.real)], 3)}")
# con n=1 (sin cooperatividad)
pst1 = brentq(lambda p: p - ra / (1 + p) - ra0, 0, ra + 1)
X1 = ra / (1 + pst1) ** 2
print(f"[repr] n=1: X={X1:.3f}; 3X^2/(4-2X)={3 * X1 ** 2 / (4 - 2 * X1):.3f} vs {lhs:.2f}")
# diagrama de estabilidad en (beta, alfa) para n=2: frontera
fr_rows = []
for lb in np.linspace(-1, 3, 81):
    b = 10 ** lb
    L = (b + 1) ** 2 / b
    # buscar alfa mínimo con 3X^2/(4+2X) = L  (X crece con alfa)
    def g(la):
        a = 10 ** la
        p = brentq(lambda q: q - a / (1 + q * q), 0, a + 1)
        Xa = a * 2 * p / (1 + p * p) ** 2
        return (3 * Xa ** 2 / (4 - 2 * Xa) if Xa < 2 else 1e9) - L
    try:
        la = brentq(g, -1, 8)
        fr_rows.append((b, 10 ** la))
    except ValueError:
        pass
w("repr_frontera.dat", "beta alfa\n" + "\n".join(f"{b:.4f} {a:.4f}" for b, a in fr_rows))
print(f"[repr] frontera: alfa mínimo (en beta=1) = {min(a for b, a in fr_rows):.2f}")

# =====================================================================
# 16.2 (e)  Gillespie: nacimiento y muerte de una proteína
# =====================================================================
def ssa_bd(k, g, x0, T, rs):
    t, x, ts, xs = 0.0, x0, [0.0], [x0]
    while t < T:
        a1_, a2_ = k, g * x
        a0 = a1_ + a2_
        t += rs.exponential(1 / a0)
        x += 1 if rs.random() * a0 < a1_ else -1
        ts.append(t)
        xs.append(x)
    return np.array(ts), np.array(xs)


kb, gb = 1.0, 0.1                 # media 10 moléculas
for j in range(3):
    ts, xs = ssa_bd(kb, gb, 0, 100, np.random.default_rng(50 + j))
    # formato escalera
    keep = ts <= 100
    ts, xs = ts[keep], xs[keep]
    w(f"ssa{j}.dat", "t x\n" + "\n".join(f"{a:.3f} {b}" for a, b in zip(ts, xs)))
tode = np.linspace(0, 100, 201)
w("ssa_ode.dat", "t x\n" + "\n".join(f"{a:.2f} {kb / gb * (1 - math.exp(-gb * a)):.4f}"
                                      for a in tode))
ts, xs = ssa_bd(kb, gb, 10, 20000, np.random.default_rng(7))
# distribución temporal ponderada por tiempo de permanencia (t>50)
dt = np.diff(ts)
xsx = xs[:-1]
sel = ts[:-1] > 50
hist = np.bincount(xsx[sel], weights=dt[sel], minlength=31)[:31]
hist = hist / dt[sel].sum()
mu = kb / gb
pois = [math.exp(-mu) * mu ** x / math.factorial(x) for x in range(31)]
w("ssa_hist.dat", "x ssa pois\n" + "\n".join(f"{x} {h:.5f} {p:.5f}"
                                             for x, (h, p) in enumerate(zip(hist, pois))))
mean_t = np.sum(xsx[sel] * dt[sel]) / dt[sel].sum()
var_t = np.sum((xsx[sel] - mean_t) ** 2 * dt[sel]) / dt[sel].sum()
print(f"[SSA] media={mean_t:.2f} var={var_t:.2f} Fano={var_t / mean_t:.3f}"
      f" CV={math.sqrt(var_t) / mean_t:.3f} (teoria 1/sqrt(10)={1 / math.sqrt(10):.3f})")
print(f"[SSA] eventos simulados en T=20000: {len(ts) - 1}")

# =====================================================================
# 16.2 (f)  Inferencia de redes: GENIE3 (bosques aleatorios) vs correlación
# =====================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

def red_sintetica(seed, ng=20, ntf=6, ns=300, ruido=0.05):
    """Red al azar (ntf reguladores) y ns estados estacionarios perturbados."""
    r2 = np.random.default_rng(seed)
    TF = list(range(ntf))
    Wt = np.zeros((ng, ng))           # Wt[i, j] != 0 : i regula j
    for j in range(ng):
        regs = r2.choice([t_ for t_ in TF if t_ != j], size=r2.integers(1, 3),
                         replace=False)
        for i in regs:
            Wt[i, j] = r2.choice([-1, 1]) * r2.uniform(1, 2)

    def estado(basal):
        x = np.ones(ng)
        for _ in range(400):           # iteración de punto fijo amortiguada
            xn = basal + 2 / (1 + np.exp(-(Wt.T @ x - 1.5)))
            x = 0.5 * x + 0.5 * xn
        return x

    Xe = np.array([estado(r2.lognormal(0, 0.6, ng)) for _ in range(ns)])
    Xe += r2.normal(0, ruido, Xe.shape)
    return TF, Wt, (Xe - Xe.mean(0)) / Xe.std(0)


def puntuar(TF, Xs, ng=20):
    S_gen, S_cor = np.zeros((ng, ng)), np.zeros((ng, ng))
    Cc = np.abs(np.corrcoef(Xs.T))
    for j in range(ng):
        inp = [i for i in TF if i != j]
        rf = RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=0)
        rf.fit(Xs[:, inp], Xs[:, j])
        S_gen[inp, j] = rf.feature_importances_
        S_cor[inp, j] = Cc[inp, j]
    mask = np.zeros((ng, ng), bool)
    for j in range(ng):
        mask[[i for i in TF if i != j], j] = True
    return S_gen, S_cor, mask


resumen = {"genie3": [], "cor": [], "azar": []}
for seed in range(3, 8):
    TF, Wt, Xs = red_sintetica(seed)
    S_gen, S_cor, mask = puntuar(TF, Xs)
    ytrue = (Wt[mask] != 0).astype(int)
    for nm, S in (("genie3", S_gen), ("cor", S_cor)):
        ap = average_precision_score(ytrue, S[mask])
        resumen[nm].append(ap)
        if seed == 3:
            pr, rc, _ = precision_recall_curve(ytrue, S[mask])
            w(f"pr_{nm}.dat", "r p\n" + "\n".join(f"{a:.4f} {b:.4f}"
                                                   for a, b in zip(rc[::-1], pr[::-1])))
            print(f"[GRN] red 3: {nm} AUPR={ap:.3f} AUROC={roc_auc_score(ytrue, S[mask]):.3f}")
    resumen["azar"].append(ytrue.mean())
    if seed == 3:
        print(f"[GRN] red 3: {int((Wt != 0).sum())} aristas verdaderas;"
              f" candidatos={mask.sum()} positivos={ytrue.sum()} azar={ytrue.mean():.3f}")
for nm, v in resumen.items():
    print(f"[GRN] 5 redes: {nm} AUPR = {np.mean(v):.3f} +- {np.std(v):.3f}  {np.round(v, 3)}")
