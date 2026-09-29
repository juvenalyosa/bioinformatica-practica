"""Genera datos de figuras y cifras del capítulo 14 (metagenómica y microbioma).

Ejecutar desde libro/:  python3 figuras/cap14/generar.py
Escribe .dat/.tex en figuras/cap14/ e imprime las cifras que se citan en
los ejemplos del texto (única fuente de verdad). Todo es simulado con
semilla fija, por lo que es reproducible.
"""
from pathlib import Path
import math
import numpy as np
from scipy import stats
from scipy.special import gammaln

OUT = Path(__file__).parent
rng = np.random.default_rng(1977)          # año de Woese y Fox


def w(name, text):
    (OUT / name).write_text(text)


# =====================================================================
# 14.1  DADA2: abundancia frente a distancia de Hamming (simulado)
# =====================================================================
L = 250
BASES = np.array(list("ACGT"))
ref = rng.choice(4, L)


def mutar(seq, k):
    s = seq.copy()
    pos = rng.choice(L, k, replace=False)
    for p in pos:
        s[p] = (s[p] + rng.integers(1, 4)) % 4
    return s


# seis variantes reales: la 2 difiere de la 1 en un único nucleótido
centros = [ref, mutar(ref, 1), mutar(ref, 4), mutar(ref, 9), mutar(ref, 15),
           mutar(ref, 22)]
abund = np.array([5000, 900, 2000, 400, 150, 60])
e = 0.002                                   # tasa de sustitución por base
lecturas = {}
origen = {}
for c, (sq, n) in enumerate(zip(centros, abund)):
    for _ in range(n):
        s = sq.copy()
        m = rng.random(L) < e
        if m.any():
            s[m] = (s[m] + rng.integers(1, 4, m.sum())) % 4
        key = s.tobytes()
        lecturas[key] = lecturas.get(key, 0) + 1
        origen.setdefault(key, c)
claves_centros = {sq.tobytes(): i for i, sq in enumerate(centros)}
C = np.array(centros)
filas_v, filas_e = [], []
for key, a in lecturas.items():
    s = np.frombuffer(key, dtype=ref.dtype)
    if key in claves_centros:
        i = claves_centros[key]
        # distancia al centro real más abundante distinto de sí mismo
        otros = [j for j in range(len(C)) if abund[j] > abund[i]]
        d = min((C[j] != s).sum() for j in otros) if otros else 0
        filas_v.append((d, a, i))
    else:
        dist = (C != s).sum(axis=1)
        d = dist.min()
        filas_e.append((d + rng.uniform(-0.28, 0.28), a))
n_unicas = len(lecturas)
n_err = sum(a for k, a in lecturas.items() if k not in claves_centros)
print(f"[dada2] lecturas={abund.sum()}  únicas={n_unicas}  "
      f"lecturas con error={n_err} ({n_err/abund.sum():.1%})")
print(f"[dada2] variantes reales (d, a): {sorted(filas_v)}")
print(f"[dada2] error más abundante: {max(a for _, a in filas_e)}")
w("dada2_err.dat", "d a\n" + "\n".join(f"{d:.3f} {a}" for d, a in filas_e))
w("dada2_real.dat", "d a\n" + "\n".join(f"{d} {a}" for d, a, _ in filas_v))
# umbral: abundancia mínima con p < 1e-40 frente al centro de 5000 lecturas
Omega = 1e-40
filas_u = []
for d in range(1, 26):
    lam = (e / 3) ** d * (1 - e) ** (L - d)
    mu = abund[0] * lam
    lden = math.log(-math.expm1(-mu))          # log P(A > 0)

    def lcola(a):                                # log P(A >= a), estable
        ks = np.arange(a, a + 400)
        lt = ks * math.log(mu) - mu - gammaln(ks + 1)
        mx = lt.max()
        return mx + math.log(np.exp(lt - mx).sum())
    a = 1
    while lcola(a) - lden >= math.log(Omega):
        a += 1
    filas_u.append((d, a, mu))
w("dada2_umbral.dat", "d a\n" + "\n".join(f"{d} {a}" for d, a, _ in filas_u))
print("[dada2] umbral (d, a*, mu):",
      [(d, a, f"{mu:.3g}") for d, a, mu in filas_u[:4]])
# ejemplo del texto: centro 5000, variante a 1 nt con 900 lecturas
lam1 = (e / 3) * (1 - e) ** (L - 1)
mu1 = 5000 * lam1
lp557 = (557 * math.log(mu1) - mu1 - gammaln(558)
         - math.log(-math.expm1(-mu1))) / math.log(10)
print(f"[dada2] lambda(d=1)={lam1:.3e}  esperado={mu1:.3f}  "
      f"log10 p(a>=557) aprox (término dominante)={lp557:.0f}")
print(f"[dada2] esperado con a>=10: log10 p = "
      f"{stats.poisson.logsf(9, mu1)/math.log(10):.2f}")

# errores esperados en una lectura
for Q in (30, 25, 20):
    print(f"[ee] 250 pb a Q{Q}: EE={250*10**(-Q/10):.2f}")
qprof = np.concatenate([np.full(150, 36), np.linspace(36, 18, 100)])
print(f"[ee] perfil 36->18: EE={np.sum(10**(-qprof/10)):.2f}")

# =====================================================================
# 14.1  Clasificador bayesiano ingenuo (ejemplo numérico de juguete)
# =====================================================================
# 3 géneros con M secuencias de referencia; conteos m(w,G) de 4 palabras
M = {"Bacteroides": 40, "Prevotella": 25, "Escherichia": 60}
m = {"Bacteroides": [38, 30, 2, 35], "Prevotella": [20, 3, 1, 24],
     "Escherichia": [1, 4, 58, 2]}
N = 125
n_w = [59, 37, 61, 61]
Pi = [(nw + 0.5) / (N + 1) for nw in n_w]
logp = {}
for g in M:
    lp = sum(math.log((m[g][i] + Pi[i]) / (M[g] + 1)) for i in range(4))
    logp[g] = lp
    print(f"[bayes] {g}: log P(S|G) = {lp:.3f}   "
          f"P(w|G)={[round((m[g][i]+Pi[i])/(M[g]+1),3) for i in range(4)]}")
mx = max(logp.values())
Z = sum(math.exp(v - mx) for v in logp.values())
for g in M:
    print(f"[bayes] posterior (priori uniforme) {g}: {math.exp(logp[g]-mx)/Z:.3f}")

# =====================================================================
# 14.2  Curvas de rarefacción y Chao1
# =====================================================================
def comunidad(S, alpha):
    p = rng.dirichlet(np.full(S, alpha))
    return np.sort(p)[::-1]


com = {"A": (comunidad(400, 0.15), 30000),
       "B": (comunidad(180, 0.6), 18000),
       "C": (comunidad(90, 2.0), 8000)}
conteos = {}
for k, (p, n) in com.items():
    conteos[k] = rng.multinomial(n, p)


def rarefaccion(x, ns):
    x = x[x > 0]
    Ntot = x.sum()
    out = []
    for n in ns:
        # E[S_n] = sum_i 1 - C(N - N_i, n)/C(N, n)
        lg = (gammaln(Ntot - x + 1) - gammaln(Ntot - x - n + 1)
              - gammaln(Ntot + 1) + gammaln(Ntot - n + 1))
        term = np.where(Ntot - x >= n, np.exp(lg), 0.0)
        out.append((1 - term).sum())
    return np.array(out)


def chao1(x):
    x = x[x > 0]
    f1, f2 = (x == 1).sum(), (x == 2).sum()
    S = len(x)
    bc = S + f1 * (f1 - 1) / (2 * (f2 + 1))
    return S, f1, f2, bc


lin = []
for k, x in conteos.items():
    Ntot = x.sum()
    ns = np.unique(np.concatenate([np.geomspace(1, Ntot, 60).astype(int), [Ntot]]))
    r = rarefaccion(x, ns)
    w(f"rare_{k}.dat", "n s\n" + "\n".join(f"{a} {b:.3f}" for a, b in zip(ns, r)))
    S, f1, f2, c1 = chao1(x)
    r8 = rarefaccion(x, [8000])[0]
    lin.append(rf"\def\capcatorceChao{k}{{{c1:.1f}}}\def\capcatorceSobs{k}{{{S}}}")
    print(f"[rare] {k}: S_real={len(com[k][0])} N={Ntot} S_obs={S} f1={f1} f2={f2} "
          f"Chao1(bc)={c1:.1f}  E[S_8000]={r8:.1f}  "
          f"H={-(com[k][0]*np.log(com[k][0])).sum():.2f}")
w("rare_macros.tex", "\n".join(lin) + "\n")

# Ejemplo numérico de diversidad alfa: comunidad de juguete
toy = np.array([50, 20, 10, 10, 5, 2, 1, 1, 1])
p = toy / toy.sum()
H = -(p * np.log(p)).sum()
D = (p ** 2).sum()
f1, f2 = (toy == 1).sum(), (toy == 2).sum()
print(f"[alfa] N={toy.sum()} S={len(toy)} H={H:.4f} expH={math.exp(H):.3f} "
      f"Simpson={D:.4f} invSimpson={1/D:.3f} GiniSimpson={1-D:.4f} "
      f"Pielou={H/math.log(len(toy)):.3f} f1={f1} f2={f2} "
      f"Chao1={len(toy)+f1**2/(2*f2):.2f} Chao1bc={len(toy)+f1*(f1-1)/(2*(f2+1)):.2f}")

# =====================================================================
# 14.2  Perfiles de diversidad de Hill
# =====================================================================
def hill(p, q):
    p = p[p > 0]
    if abs(q - 1) < 1e-9:
        return math.exp(-(p * np.log(p)).sum())
    return ((p ** q).sum()) ** (1 / (1 - q))


pA = 0.5 ** np.arange(1, 61); pA[-1] += 1 - pA.sum(); pA = pA / pA.sum()
pA = np.array([0.40] + list(0.60 * np.ones(59) / 59))           # 1 dominante + 59 raras
pB = np.ones(20) / 20                                            # 20 equitativas
x = np.arange(1, 41); pC = x ** -1.2; pC = pC / pC.sum()        # 40, Zipf
qs = np.round(np.linspace(0, 4, 81), 3)
w("hill.dat", "q A B C\n" + "\n".join(
    f"{q} {hill(pA, q):.4f} {hill(pB, q):.4f} {hill(pC, q):.4f}" for q in qs))
for q in (0, 1, 2):
    print(f"[hill] q={q}: A={hill(pA, q):.2f} B={hill(pB, q):.2f} C={hill(pC, q):.2f}")
print(f"[hill] q=inf: A={1/pA.max():.2f} B={1/pB.max():.2f} C={1/pC.max():.2f}")

# =====================================================================
# 14.2  Datos composicionales: el cierre crea correlaciones espurias
# =====================================================================
absA = np.array([100, 80, 60, 40])       # antes (células x 1e6 / g)
absB = np.array([100, 80, 60, 400])      # después: solo el taxón 4 crece
relA, relB = absA / absA.sum(), absB / absB.sum()
w("compos.tex", "\n".join(
    [rf"\def\capcatorceAbs{s}{{{','.join(str(v) for v in a)}}}" for s, a in (("A", absA), ("B", absB))]
    + [rf"\def\capcatorceRel{s}{{{','.join(f'{v:.3f}' for v in r)}}}" for s, r in (("A", relA), ("B", relB))]) + "\n")
print(f"[compos] rel antes={np.round(relA,3)} después={np.round(relB,3)}")
g = lambda v: math.exp(np.log(v).mean())
clrA, clrB = np.log(relA / g(relA)), np.log(relB / g(relB))
print(f"[compos] clr antes={np.round(clrA,3)} después={np.round(clrB,3)}")
print(f"[compos] log-razón t1/t2 antes={math.log(relA[0]/relA[1]):.3f} "
      f"después={math.log(relB[0]/relB[1]):.3f}")

# =====================================================================
# 14.2  Beta: Bray-Curtis, PCoA y PERMANOVA (simulado)
# =====================================================================
T = 60
base = rng.dirichlet(np.full(T, 0.4))
grupos = {"sano": base,
          "dieta": rng.dirichlet(80 * base + 0.05),
          "antibiotico": None}
ab = base.copy(); ab[np.argsort(base)[-8:]] *= 0.08; ab = ab / ab.sum()
grupos["antibiotico"] = ab
X, lab = [], []
for gname, pb in grupos.items():
    for _ in range(15):
        pi = rng.dirichlet(60 * pb + 0.01)
        X.append(rng.multinomial(10000, pi) / 10000)
        lab.append(gname)
X = np.array(X); lab = np.array(lab)


def bray(a, b):
    return np.abs(a - b).sum() / (a + b).sum()


n = len(X)
Dm = np.array([[bray(X[i], X[j]) for j in range(n)] for i in range(n)])
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ (Dm ** 2) @ J
ev, V = np.linalg.eigh(B)
idx = np.argsort(ev)[::-1]; ev, V = ev[idx], V[:, idx]
coords = V[:, :2] * np.sqrt(ev[:2])
pos_sum = ev[ev > 0].sum()
var = ev[:2] / pos_sum
neg = ev[ev < -1e-10]
for gname in grupos:
    s = lab == gname
    w(f"pcoa_{gname}.dat", "x y\n" + "\n".join(
        f"{a:.4f} {b:.4f}" for a, b in coords[s]))
w("pcoa_macros.tex",
  rf"\def\capcatorceEjeUno{{{100*var[0]:.1f}}}\def\capcatorceEjeDos{{{100*var[1]:.1f}}}" + "\n")


def permanova(D, g, perms=999):
    N = len(g); lv = np.unique(g); a = len(lv)
    D2 = D ** 2
    SST = D2[np.triu_indices(N, 1)].sum() / N

    def ssw(gg):
        s = 0
        for l in lv:
            k = np.where(gg == l)[0]
            sub = D2[np.ix_(k, k)]
            s += sub[np.triu_indices(len(k), 1)].sum() / len(k)
        return s
    W = ssw(g)
    F = ((SST - W) / (a - 1)) / (W / (N - a))
    cnt = 0
    for _ in range(perms):
        gp = rng.permutation(g)
        Wp = ssw(gp)
        Fp = ((SST - Wp) / (a - 1)) / (Wp / (N - a))
        cnt += Fp >= F
    return F, (cnt + 1) / (perms + 1), 1 - W / SST, SST, W


F, pv, R2, SST, SSW = permanova(Dm, lab)
print(f"[pcoa] var ejes: {var[0]:.3f} {var[1]:.3f}  autovalores negativos: {len(neg)} "
      f"(mín {neg.min() if len(neg) else 0:.4f})")
print(f"[permanova] SST={SST:.4f} SSW={SSW:.4f} F={F:.2f} R2={R2:.3f} p={pv:.3f}")
for gname in grupos:
    s = lab == gname
    sub = Dm[np.ix_(s, s)][np.triu_indices(s.sum(), 1)]
    print(f"[beta] BC medio dentro de {gname}: {sub.mean():.3f}")
print(f"[beta] BC medio sano-antibiótico: "
      f"{Dm[np.ix_(lab=='sano', lab=='antibiotico')].mean():.3f}")

# ejemplo de Bray-Curtis / Jaccard a mano
u = np.array([10, 0, 5, 3, 2]); v = np.array([4, 6, 5, 0, 5])
bc = np.abs(u - v).sum() / (u + v).sum()
jac = 1 - ((u > 0) & (v > 0)).sum() / ((u > 0) | (v > 0)).sum()
print(f"[beta] ejemplo u={u} v={v}: sum|u-v|={np.abs(u-v).sum()} "
      f"sum={(u+v).sum()} BC={bc:.4f}  sum min={np.minimum(u,v).sum()}  Jaccard={jac:.3f}")

# =====================================================================
# 14.3  Bracken: redistribución bayesiana (ejemplo)
# =====================================================================
verd = np.array([6000, 3000, 1000])
pSS = np.array([0.5, 0.8, 0.3])            # P(asignada a especie | especie)
K = verd * pSS
G = (verd * (1 - pSS)).sum()
PS = K / K.sum()
wgt = (1 - pSS) * PS
post = wgt / wgt.sum()
brk = K + G * post
print(f"[bracken] Kraken especie={K}  género={G}  P(S|G)={np.round(post,3)} "
      f"Bracken={np.round(brk,0)}  verdad={verd}")
w("bracken.tex", "\n".join([
    r"\def\capcatorceBrkVerd{" + ",".join(f"{v:.0f}" for v in verd) + "}",
    r"\def\capcatorceBrkKr{" + ",".join(f"{v:.0f}" for v in K) + "}",
    r"\def\capcatorceBrkBr{" + ",".join(f"{v:.0f}" for v in brk) + "}"]) + "\n")
w("bracken.dat", "i verdad kraken bracken\n" + "\n".join(
    f"{i} {verd[i]} {K[i]:.0f} {brk[i]:.0f}" for i in range(3)))

# probabilidad de coincidencia espuria de un k-mer
for k in (15, 21, 31, 35):
    print(f"[kmer] k={k}: 4^-k={4.0**-k:.2e}  espurias esperadas en 1e11 pb "
          f"de referencia: {1e11*4.0**-k:.2e}")

# cobertura de un genoma raro
for a in (0.01, 0.001):
    Nr = 2e7 * 2; Lr = 150; Gs = 4e6
    cov = Nr * Lr * a / Gs
    print(f"[cov] abundancia {a}: cobertura={cov:.1f}x   fracción no cubierta"
          f" e^-c={math.exp(-cov):.2e}")

# =====================================================================
# 14.3  Binning: GC frente a cobertura (simulado)
# =====================================================================
genomas = [(0.33, 8), (0.42, 22), (0.51, 60), (0.58, 15), (0.67, 140)]
filas = []
for gi, (gc, cov) in enumerate(genomas):
    ncont = rng.integers(35, 70)
    lens = np.clip(rng.lognormal(math.log(15000), 0.8, ncont), 2500, 300000)
    for Lc in lens:
        gcc = rng.normal(gc, 0.6 * math.sqrt(0.25 / Lc) * 20 + 0.004)
        cc = rng.gamma(shape=cov * Lc / 20000 + 5, scale=1) * cov / (cov * Lc / 20000 + 5)
        filas.append((gi, gcc, cc, Lc))
for gi in range(len(genomas)):
    w(f"bin_{gi}.dat", "gc cov len\n" + "\n".join(
        f"{100*g:.2f} {c:.2f} {l/1000:.1f}" for k, g, c, l in filas if k == gi))
print(f"[bin] contigs={len(filas)}  por genoma={[sum(1 for f in filas if f[0]==i) for i in range(5)]}")

# CheckM: ejemplo de completitud y contaminación
marc = 104
halladas_1 = 95; dup = 6
print(f"[checkm] completitud={halladas_1/marc:.1%} contaminación={dup/marc:.1%}")
