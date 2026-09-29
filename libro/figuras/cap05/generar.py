"""Genera datos y figuras del capítulo 5 (filogenética) a partir de cálculos reales.

Todo lo que aparece en los ejemplos resueltos se imprime aquí, de modo que
las cifras del texto se copian de esta salida y no de memoria.
Ejecutar desde libro/:  python3 figuras/cap05/generar.py
"""
from pathlib import Path
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import gamma as gammadist

OUT = Path(__file__).parent
rng = np.random.default_rng(2024)
NT = "ACGT"


def dfact(k):
    r = 1
    while k > 1:
        r, k = r * k, k - 2
    return r


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace(".", "{,}")


# =====================================================================
# 1. Número de topologías
# =====================================================================
print("== Topologías ==")
rows = []
for n in [3, 4, 5, 6, 7, 8, 10, 15, 20, 50]:
    u, r = dfact(2 * n - 5), dfact(2 * n - 3)
    rows.append((n, u, r))
    print(n, u, r, f"{u:.3e}")


def latex_big(v):
    if v < 10**6:
        return rf"\num{{{v}}}"
    e = int(np.floor(np.log10(float(v))))
    m = float(v) / 10**e
    return rf"${m:.2f}\times10^{{{e}}}$".replace(".", "{,}")


lines = [rf"{n} & {latex_big(u)} & {latex_big(r)}\\" for n, u, r in rows]
(OUT / "topologias.tex").write_text(
    "\\begin{tabular}{@{}rrr@{}}\n\\toprule\n"
    "$n$ & no enraizados $(2n-5)!!$ & enraizados $(2n-3)!!$\\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# =====================================================================
# 2. Modelos de sustitución
# =====================================================================
def Q_gtr(rates, pi):
    """rates = (a,b,c,d,e,f) para AC, AG, AT, CG, CT, GT."""
    a, b, c, d, e, f = rates
    R = np.array([[0, a, b, c], [a, 0, d, e], [b, d, 0, f], [c, e, f, 0]], float)
    Q = R * pi[None, :]
    np.fill_diagonal(Q, -Q.sum(1))
    mu = -np.dot(pi, np.diag(Q))
    return Q / mu            # normalizado: 1 sustitución esperada por unidad


def Q_hky(kappa, pi):
    return Q_gtr((1, kappa, 1, 1, kappa, 1), pi)


PI_EQ = np.full(4, 0.25)
# comprobación: JC con expm frente a la fórmula cerrada
Qjc = Q_hky(1.0, PI_EQ)
for t in (0.1, 0.5, 2.0):
    P = expm(Qjc * t)
    closed = 0.25 + 0.75 * np.exp(-4 * t / 3)
    assert abs(P[0, 0] - closed) < 1e-12
print("JC expm == fórmula cerrada: OK")
print("Q JC (normalizada):\n", np.round(Qjc, 4))
Qk = Q_hky(4.0, PI_EQ)
print("Q K80 kappa=4:\n", np.round(Qk, 4))
print("P(0.3) K80 kappa=4:\n", np.round(expm(Qk * 0.3), 4))


# ---- curva de saturación ---------------------------------------------
def p_jc(d):
    return 0.75 * (1 - np.exp(-4 * d / 3))


def p_k80(d, kappa):
    # d = (alpha + 2 beta) t ; alpha/beta = kappa
    beta_t = d / (kappa + 2)
    alpha_t = kappa * beta_t
    Ptr = 0.25 + 0.25 * np.exp(-4 * beta_t) - 0.5 * np.exp(-2 * (alpha_t + beta_t))
    Qtv = 0.5 - 0.5 * np.exp(-4 * beta_t)
    return Ptr + Qtv, Ptr, Qtv


def p_jcg(d, a):
    return 0.75 * (1 - (1 + 4 * d / (3 * a)) ** (-a))


ds = np.linspace(0, 3, 121)
with open(OUT / "saturacion.dat", "w") as fh:
    fh.write("d pjc pk80 ptr ptv pjcg\n")
    for d in ds:
        pk, ptr, ptv = p_k80(d, 10.0)
        fh.write(f"{d:.4f} {p_jc(d):.5f} {pk:.5f} {ptr:.5f} {ptv:.5f} {p_jcg(d, 0.5):.5f}\n")


def simulate_pair(d, L, Q, pi, alpha=None):
    x = rng.choice(4, size=L, p=pi)
    r = np.ones(L) if alpha is None else rng.gamma(alpha, 1 / alpha, size=L)
    y = np.empty(L, int)
    for i in range(L):
        P = expm(Q * d * r[i]) if alpha is not None else None
        if alpha is None:
            break
        y[i] = rng.choice(4, p=P[x[i]])
    if alpha is None:
        P = expm(Q * d)
        cum = P.cumsum(1)
        u = rng.random(L)
        y = (u[:, None] > cum[x]).sum(1)
    return x, y


with open(OUT / "saturacion_sim.dat", "w") as fh:
    fh.write("d p\n")
    for d in np.arange(0.1, 3.01, 0.2):
        x, y = simulate_pair(d, 500, Qjc, PI_EQ)
        fh.write(f"{d:.2f} {(x != y).mean():.4f}\n")

# ---- densidades gamma --------------------------------------------------
xs = np.linspace(0.005, 3, 200)
with open(OUT / "gamma.dat", "w") as fh:
    fh.write("r a025 a05 a1 a4\n")
    for r in xs:
        vals = [gammadist.pdf(r, a, scale=1 / a) for a in (0.25, 0.5, 1, 4)]
        fh.write(f"{r:.4f} " + " ".join(f"{min(v, 4.0):.5f}" for v in vals) + "\n")


def gamma_cats(alpha, k=4):
    """Tasas medias por categoría (método de la media, Yang 1994)."""
    qs = gammadist.ppf(np.linspace(0, 1, k + 1), alpha, scale=1 / alpha)
    from scipy.special import gammainc
    cdf1 = gammainc(alpha + 1, qs * alpha)
    rates = (cdf1[1:] - cdf1[:-1]) * k
    return rates


print("Categorías Γ α=0.5:", np.round(gamma_cats(0.5), 4))

# ---- ejemplo resuelto de distancias ------------------------------------
print("\n== Ejemplo de distancias ==")
s1 = "ATGGCTAAGCTTACGGATCCAGTACTGAAACGTTGCAAGT"
s2 = list(s1)
changes = {3: "A", 8: "A", 14: "T", 20: "T", 27: "G", 33: "C", 11: "T", 36: "C"}
for pos, b in changes.items():
    s2[pos] = b
s2 = "".join(s2)
ts_set = {frozenset("AG"), frozenset("CT")}
L = len(s1)
nd = sum(a != b for a, b in zip(s1, s2))
nts = sum(a != b and frozenset(a + b) in ts_set for a, b in zip(s1, s2))
ntv = nd - nts
p = nd / L
P_, Q_ = nts / L, ntv / L
djc = -0.75 * np.log(1 - 4 * p / 3)
dk80 = -0.5 * np.log(1 - 2 * P_ - Q_) - 0.25 * np.log(1 - 2 * Q_)
var_jc = p * (1 - p) / (L * (1 - 4 * p / 3) ** 2)
kappa_hat = (-0.5 * np.log(1 - 2 * P_ - Q_) + 0.25 * np.log(1 - 2 * Q_)) / (-0.25 * np.log(1 - 2 * Q_)) * 2 / 2
print(s1)
print(s2)
print("".join("|" if a == b else ("i" if frozenset(a + b) in ts_set else "v") for a, b in zip(s1, s2)))
print(f"L={L} dif={nd} ts={nts} tv={ntv} p={p:.4f} P={P_:.4f} Q={Q_:.4f}")
print(f"dJC={djc:.4f} sd={np.sqrt(var_jc):.4f} dK80={dk80:.4f}")
for a in (0.5, 1.0):
    print(f"dJC+G(a={a})={0.75 * a * ((1 - 4 * p / 3) ** (-1 / a) - 1):.4f}")
(OUT / "par_secuencias.txt").write_text(f"{s1}\n{s2}\n")

# =====================================================================
# 3. UPGMA y Neighbor-Joining sobre un árbol aditivo sin reloj
# =====================================================================
print("\n== NJ / UPGMA ==")
taxa = ["A", "B", "C", "D", "E"]
# árbol verdadero no enraizado: ((A:.1,B:.4):.1, C:.2, (D:.3,E:.2):.1)
edges = {("A", "u"): .1, ("B", "u"): .4, ("u", "v"): .1, ("C", "v"): .2,
         ("v", "w"): .1, ("D", "w"): .3, ("E", "w"): .2}
import collections
adj = collections.defaultdict(dict)
for (a, b), l in edges.items():
    adj[a][b] = l
    adj[b][a] = l


def path_len(a, b):
    stack = [(a, None, 0.0)]
    while stack:
        n, par, acc = stack.pop()
        if n == b:
            return acc
        for m, l in adj[n].items():
            if m != par:
                stack.append((m, n, acc + l))


D = np.array([[path_len(a, b) for b in taxa] for a in taxa])
print(np.round(D, 3))
with open(OUT / "nj_D.tex", "w") as fh:
    fh.write("\\begin{tabular}{@{}c|ccccc@{}}\n$D$ & A & B & C & D & E\\\\\\midrule\n")
    for i, a in enumerate(taxa):
        fh.write(a + " & " + " & ".join(fmt(D[i, j], 1) if j != i else "0" for j in range(5)) + r"\\" + "\n")
    fh.write("\\end{tabular}\n")


def four_point_ok(D):
    n = len(D)
    for i, j, k, l in itertools.combinations(range(n), 4):
        s = sorted([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]])
        if abs(s[2] - s[1]) > 1e-9:
            return False
    return True


print("¿aditiva? (4 puntos):", four_point_ok(D))
i, j, k, l = 0, 1, 2, 3
print("sumas 4 puntos ABCD:", D[0, 1] + D[2, 3], D[0, 2] + D[1, 3], D[0, 3] + D[1, 2])


def upgma(D, names):
    D = D.copy().astype(float)
    clusters = [(nm, 1, 0.0) for nm in names]     # (newick, tamaño, altura)
    steps = []
    while len(clusters) > 1:
        n = len(clusters)
        best = min(((D[a, b], a, b) for a in range(n) for b in range(a + 1, n)))
        d, a, b = best
        h = d / 2
        ca, cb = clusters[a], clusters[b]
        nw = f"({ca[0]}:{h - ca[2]:.3f},{cb[0]}:{h - cb[2]:.3f})"
        steps.append((ca[0], cb[0], d, h))
        sa, sb = ca[1], cb[1]
        newrow = (sa * D[a] + sb * D[b]) / (sa + sb)
        keep = [x for x in range(n) if x not in (a, b)]
        Dn = np.zeros((len(keep) + 1, len(keep) + 1))
        Dn[:-1, :-1] = D[np.ix_(keep, keep)]
        Dn[-1, :-1] = Dn[:-1, -1] = newrow[keep]
        clusters = [clusters[x] for x in keep] + [(nw, sa + sb, h)]
        D = Dn
    return clusters[0][0], steps


nw_up, st_up = upgma(D, taxa)
print("UPGMA:", nw_up)
for s in st_up:
    print("  une", s)


def nj(D, names, verbose=True, record=None):
    D = D.copy().astype(float)
    names = list(names)
    nodes = list(names)
    k = 0
    out_edges = []
    while len(nodes) > 3:
        n = len(nodes)
        r = D.sum(1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        a, b = np.unravel_index(np.argmin(Q), Q.shape)
        a, b = min(a, b), max(a, b)
        la = 0.5 * D[a, b] + (r[a] - r[b]) / (2 * (n - 2))
        lb = D[a, b] - la
        if record is not None:
            record.append(dict(nodes=list(nodes), D=D.copy(), Q=Q.copy(), r=r.copy(),
                               a=a, b=b, la=la, lb=lb))
        k += 1
        u = f"u{k}"
        out_edges += [(nodes[a], u, la), (nodes[b], u, lb)]
        du = 0.5 * (D[a] + D[b] - D[a, b])
        keep = [x for x in range(n) if x not in (a, b)]
        Dn = np.zeros((len(keep) + 1, len(keep) + 1))
        Dn[:-1, :-1] = D[np.ix_(keep, keep)]
        Dn[-1, :-1] = Dn[:-1, -1] = du[keep]
        nodes = [nodes[x] for x in keep] + [u]
        D = Dn
    # tres nodos restantes: estrella
    a, b, c = 0, 1, 2
    la = 0.5 * (D[a, b] + D[a, c] - D[b, c])
    lb = 0.5 * (D[a, b] + D[b, c] - D[a, c])
    lc = 0.5 * (D[a, c] + D[b, c] - D[a, b])
    out_edges += [(nodes[0], "c", la), (nodes[1], "c", lb), (nodes[2], "c", lc)]
    if record is not None:
        record.append(dict(nodes=list(nodes), D=D.copy(), final=(la, lb, lc)))
    return out_edges


rec = []
E = nj(D, taxa, record=rec)
for e in E:
    print("  arista", e[0], e[1], round(e[2], 6))
for st in rec[:-1]:
    print("paso: nodos", st["nodes"], "r=", np.round(st["r"], 3), "une",
          st["nodes"][st["a"]], st["nodes"][st["b"]], "la=%.3f lb=%.3f" % (st["la"], st["lb"]))
    print(np.round(st["Q"], 3))
print("final", rec[-1]["nodes"], np.round(rec[-1]["final"], 4))
# tabla Q del primer paso
Q0 = rec[0]["Q"]
with open(OUT / "nj_Q.tex", "w") as fh:
    fh.write("\\begin{tabular}{@{}c|ccccc|c@{}}\n$Q$ & A & B & C & D & E & $r_i$\\\\\\midrule\n")
    for i, a in enumerate(taxa):
        cells = []
        for j in range(5):
            if i == j:
                cells.append("--")
            else:
                v = Q0[i, j]
                s = fmt(v, 1).replace("-", "$-$")
                if (i, j) in ((rec[0]["a"], rec[0]["b"]), (rec[0]["b"], rec[0]["a"])):
                    s = r"\cellcolor{amarillo!40}\textbf{" + s + "}"
                cells.append(s)
        fh.write(a + " & " + " & ".join(cells) + r" & " + fmt(rec[0]["r"][i], 1) + r"\\" + "\n")
    fh.write("\\end{tabular}\n")

# =====================================================================
# 4. Verosimilitud: algoritmo de poda
# =====================================================================
print("\n== Poda de Felsenstein ==")


class Node:
    def __init__(self, name=None, children=(), bl=0.0):
        self.name, self.children, self.bl = name, list(children), bl


def P_jc(t):
    e = np.exp(-4 * t / 3)
    return np.full((4, 4), 0.25 - 0.25 * e) + np.eye(4) * e


def prune(node, obs, Pfun):
    """Devuelve vector L(node) (4,) para un sitio; obs: dict nombre->índice."""
    if not node.children:
        v = np.zeros(4)
        v[obs[node.name]] = 1
        return v
    v = np.ones(4)
    for ch in node.children:
        v *= Pfun(ch.bl) @ prune(ch, obs, Pfun)
    return v


# árbol de 4 hojas enraizado en el nodo interno x: ((1,2)x? ) -> raíz=nodo 5
t1, t2, t3, t4, t5 = 0.1, 0.2, 0.1, 0.2, 0.3
n6 = Node("6", [Node("3", bl=t3), Node("4", bl=t4)], bl=t5)
root = Node("5", [Node("1", bl=t1), Node("2", bl=t2), n6])
pattern = {"1": 0, "2": 1, "3": 2, "4": 2}   # A C G G
vec6 = P_jc(t3)[:, 2] * P_jc(t4)[:, 2]
vec5 = prune(root, pattern, P_jc)
Lsite = 0.25 * vec5.sum()
# fuerza bruta
brute = 0
for x5, x6 in itertools.product(range(4), repeat=2):
    brute += (0.25 * P_jc(t1)[x5, 0] * P_jc(t2)[x5, 1] * P_jc(t5)[x5, x6]
              * P_jc(t3)[x6, 2] * P_jc(t4)[x6, 2])
assert abs(brute - Lsite) < 1e-15
print("P(0.1):", np.round(P_jc(0.1)[0], 5), "P(0.2):", np.round(P_jc(0.2)[0], 5),
      "P(0.3):", np.round(P_jc(0.3)[0], 5))
print("L6 =", np.round(vec6, 5))
print("L5 =", np.round(vec5, 6))
print(f"L(sitio) = {Lsite:.6e}  ln = {np.log(Lsite):.4f}   (fuerza bruta {brute:.6e})")
with open(OUT / "poda_vals.tex", "w") as fh:
    fh.write(r"\def\capcincoLseis{(" + ",\\,".join(fmt(v, 4) for v in vec6) + ")}\n")
    fh.write(r"\def\capcincoLcinco{(" + ",\\,".join(fmt(v * 1000, 3) for v in vec5) + r")\times10^{-3}}" + "\n")
    fh.write(r"\def\capcincoLsitio{" + fmt(Lsite * 1e4, 3) + r"\times10^{-4}}" + "\n")

# =====================================================================
# 5. Motor de verosimilitud vectorizado (patrones de sitios)
# =====================================================================


def parse_newick(s):
    s = s.strip().rstrip(";")
    pos = 0

    def node():
        nonlocal pos
        if s[pos] == "(":
            pos += 1
            ch = [node()]
            while s[pos] == ",":
                pos += 1
                ch.append(node())
            pos += 1  # ')'
            nm = ""
        else:
            ch = []
            nm = ""
        while pos < len(s) and s[pos] not in ",():":
            nm += s[pos]
            pos += 1
        bl = 0.0
        if pos < len(s) and s[pos] == ":":
            pos += 1
            st = pos
            while pos < len(s) and s[pos] not in ",()":
                pos += 1
            bl = float(s[st:pos])
        return Node(nm or None, ch, bl)
    return node()


def postorder(n):
    for c in n.children:
        yield from postorder(c)
    yield n


def leaves(n):
    return [x.name for x in postorder(n) if not x.children]


def simulate(tree, L, Q, pi, alpha=None):
    rates = np.ones(L) if alpha is None else rng.gamma(alpha, 1 / alpha, L)
    seqs = {}
    evals, evecs = np.linalg.eig(Q)
    inv = np.linalg.inv(evecs)

    def Pt(t):
        return np.real(evecs @ np.diag(np.exp(evals * t)) @ inv)

    def rec(n, states):
        if not n.children:
            seqs[n.name] = states
            return
        for c in n.children:
            new = np.empty(L, int)
            u = rng.random(L)
            for i in range(L):
                P = Pt(c.bl * rates[i])
                new[i] = min(np.searchsorted(np.cumsum(P[states[i]]), u[i]), 3)
            rec(c, new)
    rec(tree, rng.choice(4, L, p=pi))
    return seqs


def loglik(tree, patt, w, Q, pi, rates=(1.0,), names=None):
    """patt: dict nombre -> array de estados por patrón; w: pesos."""
    evals, evecs = np.linalg.eig(Q)
    inv = np.linalg.inv(evecs)
    total = 0
    for r in rates:
        part = {}
        for n in postorder(tree):
            if not n.children:
                v = np.zeros((len(w), 4))
                v[np.arange(len(w)), patt[n.name]] = 1
                part[id(n)] = v
            else:
                v = np.ones((len(w), 4))
                for c in n.children:
                    P = np.real(evecs @ np.diag(np.exp(evals * c.bl * r)) @ inv)
                    v *= part[id(c)] @ P.T
                part[id(n)] = v
        total = total + (part[id(tree)] @ pi) / len(rates)
    return float(np.sum(w * np.log(total)))


def compress(seqs, names):
    M = np.array([seqs[n] for n in names]).T
    u, inv, cnt = np.unique(M, axis=0, return_inverse=True, return_counts=True)
    return {n: u[:, i] for i, n in enumerate(names)}, cnt.astype(float)


# ---- árbol verdadero de 6 taxones y alineamiento simulado ----------------
true_nwk = ("((((Humano:0.035,Chimpance:0.035):0.008,Gorila:0.045):0.03,"
            "Orangutan:0.08):0.06,Macaco:0.14,Titi:0.20);")
true_tree = parse_newick(true_nwk)
names = leaves(true_tree)
pi_true = np.array([0.30, 0.20, 0.20, 0.30])
Q_true = Q_hky(4.0, pi_true)
Lsim = 600
sim = simulate(true_tree, Lsim, Q_true, pi_true, alpha=0.5)
with open(OUT / "primates_sim.fasta", "w") as fh:
    for n in names:
        fh.write(f">{n}\n" + "".join(NT[i] for i in sim[n]) + "\n")
patt, w = compress(sim, names)
print("\n== Alineamiento simulado ==", Lsim, "sitios,", len(w), "patrones")


def all_edges(tree):
    return [n for n in postorder(tree) if n is not tree]


def fit(tree, model):
    """Optimiza longitudes de rama + parámetros del modelo."""
    br = all_edges(tree)
    emp = np.array([np.sum(w * (patt[n] == k)) for k in range(4) for n in names]).reshape(4, -1).sum(1)
    emp = emp / emp.sum()
    nb = len(br)

    def unpack(x):
        bls = np.exp(x[:nb])
        rest = x[nb:]
        kappa, alpha, rates6 = 1.0, None, None
        pi = PI_EQ
        if model in ("F81", "HKY", "HKY+G", "GTR+G"):
            pi = emp
        i = 0
        if model in ("K80", "HKY", "HKY+G"):
            kappa = np.exp(rest[i]); i += 1
        if model == "GTR+G":
            rates6 = np.r_[np.exp(rest[i:i + 5]), 1.0]; i += 5
        if model.endswith("+G"):
            alpha = np.exp(rest[i]); i += 1
        return bls, kappa, alpha, pi, rates6

    def nll(x):
        bls, kappa, alpha, pi, rates6 = unpack(x)
        for n, b in zip(br, bls):
            n.bl = b
        Q = Q_gtr(rates6, pi) if rates6 is not None else Q_hky(kappa, pi)
        rates = (1.0,) if alpha is None else gamma_cats(alpha)
        return -loglik(tree, patt, w, Q, pi, rates)

    nextra = {"JC": 0, "K80": 1, "F81": 0, "HKY": 1, "HKY+G": 2, "GTR+G": 6}[model]
    x0 = np.r_[np.log(np.full(nb, 0.1)), np.zeros(nextra)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(-9, 2)] * nb + [(-4, 4)] * nextra)
    res = minimize(nll, res.x, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-5, fatol=1e-6))
    bls, kappa, alpha, pi, rates6 = unpack(res.x)
    nfree = nb + nextra + (3 if model in ("F81", "HKY", "HKY+G", "GTR+G") else 0)
    nll(res.x)
    return -res.fun, nfree, kappa, alpha


# topología inicial: NJ con distancias JC
def pdist(a, b):
    return np.mean(a != b)


Dsim = np.array([[pdist(sim[a], sim[b]) for b in names] for a in names])
Djc = -0.75 * np.log(1 - 4 * np.clip(Dsim, 0, 0.74) / 3)


def edges_to_newick(edges, names, root_at):
    g = collections.defaultdict(list)
    for a, b, l in edges:
        g[a].append((b, l))
        g[b].append((a, l))

    def rec(n, par):
        ch = [(m, l) for m, l in g[n] if m != par]
        if not ch:
            return n
        return "(" + ",".join(f"{rec(m, n)}:{max(l, 1e-6):.5f}" for m, l in ch) + ")"
    # enraizar en el nodo interno vecino de root_at
    (nb_, l0), = g[root_at]
    return rec(nb_, None) + ";"


E_nj = nj(Djc, names)
nj_nwk = edges_to_newick(E_nj, names, "Titi")
print("NJ (JC):", nj_nwk)


def splits_of_edges(edges, names):
    g = collections.defaultdict(list)
    for a, b, l in edges:
        g[a].append(b)
        g[b].append(a)
    S = set()
    for a, b, l in edges:
        if a in names or b in names:
            continue
        # lado de a sin pasar por b
        seen, st = {b}, [a]
        side = set()
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x)
            if x in names:
                side.add(x)
            st += g[x]
        full = frozenset(names)
        sd = frozenset(side)
        S.add(min(sd, full - sd, key=lambda z: sorted(z)))
    return S


nj_splits = splits_of_edges(E_nj, names)
print("splits NJ:", [sorted(s) for s in nj_splits])

# ---- bootstrap NJ ---------------------------------------------------
B = 1000
counts = collections.Counter()
M = np.array([sim[n] for n in names])
for bb in range(B):
    cols = rng.integers(0, Lsim, Lsim)
    Mb = M[:, cols]
    Pd = (Mb[:, None, :] != Mb[None, :, :]).mean(2)
    Db = -0.75 * np.log(1 - 4 * np.clip(Pd, 0, 0.74) / 3)
    for s in splits_of_edges(nj(Db, names), names):
        counts[s] += 1
for s in nj_splits:
    print("soporte", sorted(s), counts[s] / B)
print("otras biparticiones frecuentes:", [(sorted(s), c / B) for s, c in counts.most_common(8) if s not in nj_splits])


# =====================================================================
# 5b. Esquema del bootstrap: alineamiento original y una réplica
# =====================================================================
filas = ["ACGTTAGCAT", "ACGTCAGCAT", "ATGTCAGCGT", "GTGCCAACGT", "GTACCGATGC"]
etq = ["Hum", "Chi", "Gor", "Ora", "Mac"]
cols_rep = [3, 3, 7, 1, 10, 5, 5, 2, 9, 7]
cc = [r"\begin{tikzpicture}[font=\sffamily\small]"]
w_ = 0.36


def bloque(x0, cols, titulo, marcar):
    for j, col in enumerate(cols):
        cc.append(rf"\node[etiqueta] at ({x0 + j * w_:.2f},0.42) {{{col}}};")
        for i, fila in enumerate(filas):
            b = fila[col - 1]
            cc.append(rf"\node[nt={b}, minimum size=3.2mm, font=\ttfamily\bfseries\tiny] "
                      rf"at ({x0 + j * w_:.2f},{-i * w_:.2f}) {{{b}}};")
    cc.append(rf"\node[font=\sffamily\small\bfseries, text=tinta, anchor=south] at "
              rf"({x0 + 4.5 * w_:.2f},0.62) {{{titulo}}};")
    if marcar:
        for j, col in enumerate(cols):
            if cols.count(col) > 1:
                cc.append(rf"\draw[naranja, thick, rounded corners=1pt] ({x0 + j * w_ - 0.17:.2f},0.6) "
                          rf"rectangle ({x0 + j * w_ + 0.17:.2f},{-4 * w_ - 0.19:.2f});")


for i, e_ in enumerate(etq):
    cc.append(rf"\node[etiqueta, anchor=east] at (-0.25,{-i * w_:.2f}) {{{e_}}};")
bloque(0, list(range(1, 11)), "alineamiento original", False)
bloque(5.4, cols_rep, "réplica bootstrap", True)
cc.append(r"\draw[flecha=naranja] (3.6,-0.72) -- node[above, etiqueta, text=naranja!80!black] "
          r"{remuestrear} node[below, etiqueta, text=naranja!80!black] {columnas} (5.05,-0.72);")
cc.append(r"\end{tikzpicture}")
(OUT / "bootstrap_esquema.tex").write_text("\n".join(cc) + "\n")

# ---- ML en la topología NJ: selección de modelos --------------------
ml_tree = parse_newick(nj_nwk)
print("\n== Selección de modelos (topología NJ fija) ==")
res_models = []
for mdl in ["JC", "K80", "F81", "HKY", "HKY+G", "GTR+G"]:
    lnL, k, kappa, alpha = fit(ml_tree, mdl)
    aic = 2 * k - 2 * lnL
    bic = k * np.log(Lsim) - 2 * lnL
    res_models.append((mdl, lnL, k, aic, bic, kappa, alpha))
    print(f"{mdl:6s} lnL={lnL:.2f} k={k} AIC={aic:.2f} BIC={bic:.2f} kappa={kappa:.2f} alpha={alpha}")
amin = min(r[3] for r in res_models)
bmin = min(r[4] for r in res_models)
with open(OUT / "modelos.tex", "w") as fh:
    fh.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n\\textbf{Modelo} & $k$ & $\\hat\\ell$ & AIC & "
             "$\\Delta$AIC & BIC & $\\Delta$BIC\\\\\n\\midrule\n")
    for mdl, lnL, k, aic, bic, kappa, alpha in res_models:
        dA, dB = aic - amin, bic - bmin
        ss = lambda v: r"\textbf{" + fmt(v, 1) + "}" if abs(v) < 1e-9 else fmt(v, 1)
        fh.write(mdl.replace("+G", r"+$\Gamma$") + f" & {k} & $" + fmt(lnL, 2).replace("-", "-") + "$ & " + fmt(aic, 1) + " & " + ss(dA)
                 + " & " + fmt(bic, 1) + " & " + ss(dB) + r"\\" + "\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
# refit best model to get branch lengths for the figure
best = min(res_models, key=lambda r: r[4])[0]
lnL, k, kappa, alpha = fit(ml_tree, "HKY+G")
print("HKY+G kappa=%.3f alpha=%.3f" % (kappa, alpha))


# ---- dibujo TikZ del árbol con soporte ------------------------------
def tikz_tree(tree, support, xscale, fname, true_bl=None):
    lv = leaves(tree)
    y = {}
    cnt = [0]

    def lay(n, x0):
        x = x0 + n.bl
        n._x = x
        if not n.children:
            n._y = -cnt[0] * 0.7
            cnt[0] += 1
        else:
            for c in n.children:
                lay(c, x)
            n._y = (n.children[0]._y + n.children[-1]._y) / 2
    tree.bl = 0
    lay(tree, 0)
    c = [r"\begin{tikzpicture}[font=\sffamily\small]"]
    for n in postorder(tree):
        if n is tree:
            continue
    def draw(n, px):
        X = n._x * xscale
        if n is not tree:
            c.append(rf"\draw[tinta, thick] ({px:.3f},{n._y:.3f}) -- ({X:.3f},{n._y:.3f});")
        if n.children:
            c.append(rf"\draw[tinta, thick] ({X:.3f},{n.children[0]._y:.3f}) -- ({X:.3f},{n.children[-1]._y:.3f});")
            if n is not tree:
                s = frozenset(leaves(n))
                full = frozenset(lv)
                key = min(s, full - s, key=lambda z: sorted(z))
                if key in support:
                    val = int(round(100 * support[key]))
                    col = "verde!70!black" if val >= 95 else ("naranja!85!black" if val >= 70 else "rojooscuro")
                    c.append(rf"\node[font=\sffamily\scriptsize\bfseries, text={col}, anchor=west, inner sep=1pt, fill=white, fill opacity=.85, text opacity=1] at ({X + 0.06:.3f},{n._y:.3f}) {{{val}}};")
                    c.append(rf"\fill[{col}] ({X:.3f},{n._y:.3f}) circle (1.6pt);")
            for ch in n.children:
                draw(ch, X)
        else:
            nm = {"Chimpance": "Chimpancé", "Orangutan": "Orangután", "Titi": "Tití"}.get(n.name, n.name)
            c.append(rf"\node[anchor=west, font=\sffamily\small\itshape, text=tinta] at ({X + 0.08:.3f},{n._y:.3f}) {{{nm}}};")
    draw(tree, 0)
    ymin = min(n._y for n in postorder(tree))
    c.append(rf"\draw[tinta2] (0,{ymin - 0.55:.3f}) -- ({0.05 * xscale:.3f},{ymin - 0.55:.3f});")
    c.append(rf"\node[etiqueta, anchor=north] at ({0.025 * xscale:.3f},{ymin - 0.6:.3f}) {{0,05 sust./sitio}};")
    c.append(r"\end{tikzpicture}")
    (OUT / fname).write_text("\n".join(c))


support = {s: counts[s] / B for s in nj_splits}
tikz_tree(ml_tree, support, 40.0, "arbol_bootstrap.tex")
tikz_tree(parse_newick(true_nwk), {}, 40.0, "arbol_verdadero.tex")

# =====================================================================
# 6. Atracción de ramas largas: parsimonia vs ML (4 taxones)
# =====================================================================
print("\n== Atracción de ramas largas ==")
long_, short_, inner = 1.0, 0.05, 0.05
# árbol verdadero ((1,2),(3,4)) con ramas largas en 1 y 3
tops = {"12|34": ((0, 1), (2, 3)), "13|24": ((0, 2), (1, 3)), "14|23": ((0, 3), (1, 2))}


def sim4(L):
    t = parse_newick(f"((a:{long_},b:{short_}):{inner},(c:{long_},d:{short_}):0.0);")
    s = simulate_fast(t, L)
    return np.array([s["a"], s["b"], s["c"], s["d"]])


def simulate_fast(tree, L):
    seqs = {}

    def rec(n, states):
        if not n.children:
            seqs[n.name] = states
            return
        for c in n.children:
            P = P_jc(c.bl)
            cum = P.cumsum(1)
            u = rng.random(L)
            new = np.minimum((u[:, None] > cum[states]).sum(1), 3)
            rec(c, new)
    rec(tree, rng.integers(0, 4, L))
    return seqs


def parsimony_choice(M):
    a, b, c, d = M
    inf = [(a == b) & (c == d) & (a != c), (a == c) & (b == d) & (a != b), (a == d) & (b == c) & (a != b)]
    sc = [x.sum() for x in inf]
    best = [i for i, v in enumerate(sc) if v == max(sc)]
    return int(rng.choice(best))          # empates: al azar


def ml4(M, top):
    (i, j), (k, l) = tops[top]
    pat, wt = compress({str(x): M[x] for x in range(4)}, [str(x) for x in range(4)])

    def nll(x):
        b = np.exp(x)
        t = Node(None, [Node(None, [Node(str(i), bl=b[0]), Node(str(j), bl=b[1])], bl=b[4]),
                        Node(str(k), bl=b[2]), Node(str(l), bl=b[3])])
        return -loglik(t, pat, wt, Qjc, PI_EQ)
    r = minimize(nll, np.log([0.3] * 5), method="L-BFGS-B", bounds=[(-9, 2)] * 5)
    return -r.fun


lens = [50, 100, 200, 500, 1000, 2000, 5000]
if (OUT / "lba.dat").exists():            # simulación lenta (~5 min)
    print("lba.dat ya existe; bórrelo para repetir la simulación")
    raise SystemExit
reps = 60
with open(OUT / "lba.dat", "w") as fh:
    fh.write("L pars ml\n")
    for Ls in lens:
        okp = okm = 0
        for _ in range(reps):
            M = sim4(Ls)
            okp += parsimony_choice(M) == 0
            lk = [ml4(M, tp) for tp in tops]
            okm += int(np.argmax(lk)) == 0
        fh.write(f"{Ls} {okp / reps:.3f} {okm / reps:.3f}\n")
        print(Ls, okp / reps, okm / reps)
print("listo")
