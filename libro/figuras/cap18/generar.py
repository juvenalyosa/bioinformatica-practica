"""Genera datos y figuras del capítulo 18 (flujos reproducibles).

Ejecutar desde libro/:  python3 figuras/cap18/generar.py
Escribe .tex/.dat en figuras/cap18/ e imprime (y guarda en cifras.txt)
las cifras que se citan en el texto (única fuente de verdad).
Todo es determinista: no hay números aleatorios.
"""
from pathlib import Path
from collections import deque
import heapq
import math
import networkx as nx

OUT = Path(__file__).parent
LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def w(name, text):
    (OUT / name).write_text(text)


# ---------------------------------------------------------------------
# 1. El DAG de trabajos del pipeline (mismas reglas que el Snakefile)
# ---------------------------------------------------------------------
COLOR = {"bwa_index": "gris", "fastp": "azul", "mapear": "aqua",
         "dedup": "verde", "llamar": "naranja", "filtrar": "amarillo",
         "anotar": "magenta", "consenso": "violeta", "filogenia": "rojo",
         "multiqc": "azulprofundo", "all": "tinta"}


def duracion(regla, n):
    """Minutos (ilustrativos) de cada trabajo con n muestras."""
    return {"bwa_index": 2, "fastp": 3, "mapear": 12, "dedup": 4,
            "consenso": 1, "llamar": 6 + 3 * n, "filtrar": 1,
            "anotar": 2, "filogenia": 4 + 2 * n, "multiqc": 1,
            "all": 0, "region": (6 + 3 * n), "concat": 1}[regla]


def dag(muestras, regiones=0):
    """regiones>0: llamado dividido por regiones (scatter-gather)."""
    G = nx.DiGraph()
    n = len(muestras)
    for m in muestras:
        G.add_edge(f"fastp_{m}", f"mapear_{m}")
        G.add_edge("bwa_index", f"mapear_{m}")
        G.add_edge(f"mapear_{m}", f"dedup_{m}")
        G.add_edge(f"fastp_{m}", "multiqc")
        G.add_edge("filtrar", f"consenso_{m}")
        G.add_edge(f"consenso_{m}", "filogenia")
        if regiones:
            for r in range(regiones):
                G.add_edge(f"dedup_{m}", f"region_{r}")
        else:
            G.add_edge(f"dedup_{m}", "llamar")
    if regiones:
        for r in range(regiones):
            G.add_edge(f"region_{r}", "concat")
        G.add_edge("concat", "filtrar")
    else:
        G.add_edge("llamar", "filtrar")
    G.add_edge("filtrar", "anotar")
    for v in ("anotar", "filogenia", "multiqc"):
        G.add_edge(v, "all")
    for v in G:
        r = v.split("_")[0] if v not in ("bwa_index",) else v
        d = duracion(r, n)
        if r == "region":
            d = duracion("llamar", n) / regiones
        G.nodes[v]["regla"] = r
        G.nodes[v]["t"] = d
    return G


M3 = ["A", "B", "C"]
G = dag(M3)
p(f"[dag] n=3: {G.number_of_nodes()} trabajos, {G.number_of_edges()} aristas")


def orden_nodos(G):
    """Orden canónico para desempatar (el del Snakefile)."""
    reglas = list(COLOR)
    return sorted(G, key=lambda v: (reglas.index(G.nodes[v]["regla"]), v))


def kahn(G):
    """Kahn con cola FIFO; devuelve el orden y la traza paso a paso."""
    indeg = {v: G.in_degree(v) for v in G}
    cola = deque(v for v in orden_nodos(G) if indeg[v] == 0)
    orden, traza = [], []
    while cola:
        v = cola.popleft()
        orden.append(v)
        liberados = []
        for u in sorted(G.successors(v), key=orden_nodos(G).index):
            indeg[u] -= 1
            if indeg[u] == 0:
                cola.append(u)
                liberados.append(u)
        traza.append((v, liberados, list(cola)))
    assert len(orden) == len(G), "hay un ciclo"
    return orden, traza


orden, traza = kahn(G)
indeg0 = [v for v in orden_nodos(G) if G.in_degree(v) == 0]
p("[kahn] fuentes iniciales:", indeg0)
p("[kahn] grados de entrada:", {v: G.in_degree(v) for v in orden_nodos(G)})
for i, (v, lib, cola) in enumerate(traza, 1):
    p(f"[kahn] paso {i:2d}: saca {v:12s} libera {lib}  cola={cola}")

# niveles (rondas de Kahn = frentes paralelos)
nivel = {}
for v in nx.topological_sort(G):
    nivel[v] = 1 + max((nivel[u] for u in G.predecessors(v)), default=0)
L = max(nivel.values())
p(f"[kahn] rondas (niveles) = {L}; ancho por ronda =",
  [sum(1 for v in G if nivel[v] == k) for k in range(1, L + 1)])


def etiqueta(v):
    r = G.nodes[v]["regla"]
    s = v[len(r) + 1:] if v != r else ""
    r = r.replace("_", r"\_")
    return r + (r"\,\textsubscript{" + s + "}" if s else "")


def chip(v):
    return (r"\capdieciochochip{" + COLOR[G.nodes[v]["regla"]] + "}{"
            + etiqueta(v) + "}")


# Tabla de Kahn (figura 18.2)
filas = []
for i, (v, lib, cola) in enumerate(traza, 1):
    filas.append(f"{i} & {chip(v)} & "
                 + (" ".join(chip(u) for u in lib) if lib else "---")
                 + " & " + (" ".join(chip(u) for u in cola) if cola
                            else r"\emph{vacía}") + r"\\")
cab = (r"\begin{tabular}{@{}r l l l@{}}" "\n" r"\toprule" "\n"
       r"\textbf{paso} & \textbf{sale de $Q$} & \textbf{grado llega a 0}"
       r" & \textbf{cola $Q$ tras el paso}\\" "\n" r"\midrule" "\n"
       r"0 & & & " + " ".join(chip(v) for v in indeg0) + r"\\")
w("kahn_tabla.tex", cab + "\n" + "\n".join(filas)
  + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")

# DAG en TikZ (figura 18.1): x = ronda de Kahn, y = fila fija por muestra
FILA = {"A": 0.5, "B": -0.5, "C": -1.5}
pos = {"bwa_index": 1.5, "multiqc": -2.5, "anotar": 1.6}
for v in G:
    r = G.nodes[v]["regla"]
    s = v[len(r) + 1:]
    pos.setdefault(v, FILA.get(s, -0.5))
X = lambda v: 1.62 * (nivel[v] - 1)
ID = lambda v: v.replace("_", "")
lines = []
for v in orden_nodos(G):
    lines.append(rf"\node[capdieciochonodo={COLOR[G.nodes[v]['regla']]}]"
                 rf" ({ID(v)}) at ({X(v):.2f},{pos[v]:.2f})"
                 rf" {{{etiqueta(v)}}};")
especiales = set()
for a, b in G.edges():
    ra = G.nodes[a]["regla"]
    if ra == "bwa_index":           # bus gris a la derecha de la ronda 1
        lines.append(rf"\draw[capdieciochobus=gris] ({ID(a)}.east) -| "
                     rf"(0.81,{pos[b]:.2f}) -- ({ID(b)}.west);")
    elif b == "multiqc":            # bus por la izquierda y por debajo
        lines.append(rf"\draw[capdieciochobus=azulprofundo] ({ID(a)}.west)"
                     rf" -- ++(-0.22,0) |- ({ID(b)}.west);")
    elif a == "multiqc":
        lines.append(rf"\draw[capdieciochoarista] ({ID(a)}.east) -| "
                     rf"({ID(b)}.south);")
    elif b == "anotar":
        lines.append(rf"\draw[capdieciochoarista] ({ID(a)}.north) |- "
                     rf"({ID(b)}.west);")
    elif a == "anotar":
        lines.append(rf"\draw[capdieciochoarista] ({ID(a)}.east) -| "
                     rf"({ID(b)}.north);")
    else:
        lines.append(rf"\draw[capdieciochoarista] ({ID(a)}) -- ({ID(b)});")
for k in range(1, L + 1):
    lines.append(rf"\node[etiqueta] at ({1.62 * (k - 1):.2f},-3.05)"
                 rf" {{ronda {k}}};")
w("dag.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------------
# 2. Invalidación: qué se rehace si cambian los parámetros de fastp_A
# ---------------------------------------------------------------------
for v in ("fastp_A", "filtrar", "bwa_index", "multiqc"):
    d = nx.descendants(G, v) - {"all"}
    p(f"[cache] cambiar {v}: rehace {1 + len(d)} de "
      f"{G.number_of_nodes() - 1} trabajos -> {sorted(d)}")

# ---------------------------------------------------------------------
# 3. Trabajo, camino crítico y planificación voraz
# ---------------------------------------------------------------------


def trabajo_span(G):
    W = sum(G.nodes[v]["t"] for v in G)
    fin = {}
    for v in nx.topological_sort(G):
        fin[v] = G.nodes[v]["t"] + max((fin[u] for u in G.predecessors(v)),
                                       default=0)
    S = max(fin.values())
    # reconstruir camino crítico
    v = max(fin, key=fin.get)
    cam = [v]
    while list(G.predecessors(v)):
        v = max(G.predecessors(v), key=lambda u: fin[u])
        cam.append(v)
    return W, S, cam[::-1]


def planificar(G, P):
    """Lista voraz: prioridad = camino más largo hasta el final."""
    cola_larga = {}
    for v in reversed(list(nx.topological_sort(G))):
        cola_larga[v] = G.nodes[v]["t"] + max(
            (cola_larga[u] for u in G.successors(v)), default=0)
    indeg = {v: G.in_degree(v) for v in G}
    listos = [(-cola_larga[v], v) for v in G if indeg[v] == 0]
    heapq.heapify(listos)
    t, libres, corriendo = 0.0, P, []   # (fin, v)
    hecho = 0
    while hecho < len(G):
        while listos and libres:
            _, v = heapq.heappop(listos)
            heapq.heappush(corriendo, (t + G.nodes[v]["t"], v))
            libres -= 1
        t, v = heapq.heappop(corriendo)
        libres += 1
        hecho += 1
        for u in G.successors(v):
            indeg[u] -= 1
            if indeg[u] == 0:
                heapq.heappush(listos, (-cola_larga[u], u))
    return t


W3, S3, cam3 = trabajo_span(G)
p(f"[span] n=3: W={W3} min, S={S3} min, W/S={W3 / S3:.2f}; camino={cam3}")
for P in (1, 2, 3, 4, 8):
    T = planificar(G, P)
    p(f"[plan] n=3 P={P}: T_P={T:.1f}  cota W/P+S={W3 / P + S3:.1f}"
      f"  max(W/P,S)={max(W3 / P, S3):.1f}  speedup={W3 / T:.2f}")

N = 24
G24 = dag([f"m{i:02d}" for i in range(N)])
G24s = dag([f"m{i:02d}" for i in range(N)], regiones=24)
W24, S24, cam24 = trabajo_span(G24)
W24s, S24s, _ = trabajo_span(G24s)
p(f"[span] n=24: W={W24}, S={S24}, W/S={W24 / S24:.2f}")
p(f"[span] n=24 scatter(24): W={W24s:.1f}, S={S24s:.2f}, "
  f"W/S={W24s / S24s:.2f}")
Ps = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
rows = []
for P in Ps:
    a = W24 / planificar(G24, P)
    b = W24s / planificar(G24s, P)
    rows.append(f"{P} {a:.3f} {b:.3f}")
    p(f"[plan] n=24 P={P:2d}: speedup={a:.2f}  scatter={b:.2f}")
w("speedup.dat", "P base scatter\n" + "\n".join(rows) + "\n")


def karp_flatt(psi, P):
    return (1 / psi - 1 / P) / (1 - 1 / P)


for P in (8, 16, 32):
    a = W24 / planificar(G24, P)
    b = W24s / planificar(G24s, P)
    p(f"[karp-flatt] P={P}: e_base={karp_flatt(a, P):.3f} "
      f"e_scatter={karp_flatt(b, P):.3f}")
# fracción serial 'de Amdahl' que mejor ajusta cada curva (mín. cuadrados)
def ajusta(col):
    best = None
    for k in range(1, 1000):
        s = k / 1000
        err = sum((1 / (s + (1 - s) / P) - v) ** 2
                  for P, v in zip(Ps, col))
        if best is None or err < best[0]:
            best = (err, s)
    return best[1]


base = [W24 / planificar(G24, P) for P in Ps]
scat = [W24s / planificar(G24s, P) for P in Ps]
sb, ss = ajusta(base), ajusta(scat)
p(f"[amdahl] s ajustada: base={sb:.3f} (lim {1 / sb:.1f}),"
  f" scatter={ss:.3f} (lim {1 / ss:.1f})")
w("amdahl.tex", rf"\def\capdieciochosb{{{sb:.3f}}}"
  rf"\def\capdieciochoss{{{ss:.3f}}}" + "\n")
for s in (0.05, 0.10, 0.25):
    p(f"[amdahl] s={s}: S(8)={1 / (s + (1 - s) / 8):.2f} "
      f"S(64)={1 / (s + (1 - s) / 64):.2f} lim={1 / s:.0f} "
      f"gustafson(64)={64 - s * 63:.2f}")

# ---------------------------------------------------------------------
# 4. Sumas de verificación: cota del cumpleaños
# ---------------------------------------------------------------------
for bits in (32, 128, 256):
    for n in (1e5, 1e6, 1e9):
        pr = -math.expm1(-n * (n - 1) / 2 ** (bits + 1))
        p(f"[hash] b={bits} n={n:.0e}: P(colision)~{pr:.3g}")

# ---------------------------------------------------------------------
# 5. Presupuesto del proyecto integrador
# ---------------------------------------------------------------------
G_MB, prof, Lr, n_m = 4.41, 100, 150, 24
pares = prof * G_MB * 1e6 / (2 * Lr)
bases = 2 * Lr * pares
bytes_fq = pares * 2 * (2 * Lr + 2 + 2 * 30)  # seq+qual+saltos+cabeceras
gz = bytes_fq / 4
p(f"[presupuesto] pares/muestra={pares:.3g}, bases={bases:.3g}, "
  f"FASTQ sin comprimir={bytes_fq / 1e9:.2f} GB, gz~{gz / 1e9:.2f} GB, "
  f"total gz n={n_m}: {n_m * gz / 1e9:.1f} GB")
mapeo_cpu_h = n_m * duracion("mapear", n_m) / 60
p(f"[presupuesto] horas-CPU de mapeo (1 núcleo/trabajo): {mapeo_cpu_h:.1f};"
  f" W total={W24 / 60:.1f} h; T_16={planificar(G24, 16) / 60:.2f} h;"
  f" T_16 scatter={planificar(G24s, 16) / 60:.2f} h")

# ---------------------------------------------------------------------
# 6. Rúbrica: nota ponderada de ejemplo
# ---------------------------------------------------------------------
pesos = [0.20, 0.20, 0.15, 0.15, 0.15, 0.15]
notas = [4, 3, 3, 4, 2, 3]
tot = sum(a * b for a, b in zip(pesos, notas))
p(f"[rubrica] sum w={sum(pesos):.2f}; nota={tot:.2f}/4 -> "
  f"{tot / 4 * 5:.2f}/5 -> {tot / 4 * 100:.1f}/100")

(OUT / "cifras.txt").write_text("\n".join(LOG) + "\n")
