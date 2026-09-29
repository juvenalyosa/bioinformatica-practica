"""Genera datos y figuras del capítulo 10 (genómica de poblaciones y GWAS).

Ejecutar desde libro/:  python3 figuras/cap10/generar.py
Escribe .dat/.tex/.pdf en figuras/cap10/ e imprime (y guarda en cifras.txt)
las cifras que se citan en el texto: única fuente de verdad.
"""
from pathlib import Path
import math
import numpy as np
from scipy import stats

import matplotlib
import matplotlib.ticker
matplotlib.use("pdf")
from matplotlib import font_manager, pyplot as plt, rcParams
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).parent
rng = np.random.default_rng(1908)
CIFRAS = []


def w(name, text):
    (OUT / name).write_text(text)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    CIFRAS.append(s)


# ---------------------------------------------------------------- estilo mpl
FD = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/adobe/sourcesanspro/"
for f in ("SourceSansPro-Regular.otf", "SourceSansPro-Bold.otf",
          "SourceSansPro-RegularIt.otf"):
    try:
        font_manager.fontManager.addfont(FD + f)
    except Exception as e:  # pragma: no cover
        print("fuente:", e)
rcParams.update({
    "font.family": "Source Sans Pro", "font.size": 8.5,
    "pdf.fonttype": 3, "mathtext.fontset": "custom",
    "mathtext.rm": "Source Sans Pro", "mathtext.it": "Source Sans Pro:italic",
    "axes.edgecolor": "#C3C2B7", "axes.linewidth": 0.6,
    "axes.labelcolor": "#52514E", "xtick.color": "#898781",
    "ytick.color": "#898781", "xtick.labelcolor": "#898781",
    "ytick.labelcolor": "#898781", "axes.spines.top": False,
    "axes.spines.right": False, "savefig.transparent": True,
})
C = dict(azul="#2A78D6", naranja="#EB6834", aqua="#1BAF7A",
         amarillo="#EDA100", magenta="#E87BA4", verde="#008300",
         violeta="#4A3AA7", rojo="#E34948", tinta="#0B0B0B",
         tinta2="#52514E", gris="#898781", rejilla="#E1E0D9",
         base="#C3C2B7", papel="#F0EFEC", azulnoche="#0D366B",
         azulprofundo="#104281", azulclaro="#9EC5F4", rojooscuro="#B8302F")

# =====================================================================
# 10.1  Hardy-Weinberg
# =====================================================================
p = np.linspace(0, 1, 101)
w("hwe_curvas.dat", "p AA Aa aa\n" + "\n".join(
    f"{x:.2f} {x*x:.4f} {2*x*(1-x):.4f} {(1-x)**2:.4f}" for x in p))

# Diagrama de De Finetti: coordenadas (x,y) en un triángulo de lado 1
S3 = math.sqrt(3) / 2


def tern(fAA, fAa, faa):
    return faa + fAa / 2, fAa * S3


par = [tern(x * x, 2 * x * (1 - x), (1 - x) ** 2) for x in np.linspace(0, 1, 61)]
par_F = [tern(x * x + 0.3 * x * (1 - x), 2 * x * (1 - x) * 0.7,
              (1 - x) ** 2 + 0.3 * x * (1 - x)) for x in np.linspace(0, 1, 61)]
pts_hwe, pts_F = [], []
for _ in range(25):   # muestras de n=200 de poblaciones en HWE
    q = rng.uniform(0.1, 0.9)
    c = rng.multinomial(200, [q * q, 2 * q * (1 - q), (1 - q) ** 2]) / 200
    pts_hwe.append(tern(*c))
for _ in range(12):   # poblaciones endogámicas F=0.3
    q = rng.uniform(0.15, 0.85)
    F = 0.3
    pr = [q * q + F * q * (1 - q), 2 * q * (1 - q) * (1 - F),
          (1 - q) ** 2 + F * q * (1 - q)]
    c = rng.multinomial(200, pr) / 200
    pts_F.append(tern(*c))


def coords(L, s=5.2):
    return " ".join(f"({x*s:.3f},{y*s:.3f})" for x, y in L)


w("definetti.tex", "\n".join([
    r"\def\capdiezparabola{" + coords(par) + "}",
    r"\def\capdiezparabolaF{" + coords(par_F) + "}",
    r"\def\capdiezpuntosH{" + coords(pts_hwe) + "}",
    r"\def\capdiezpuntosF{" + coords(pts_F) + "}",
]) + "\n")

# Ejemplos numéricos: chi-cuadrado y F


def hwe_chi2(nAA, nAa, naa):
    n = nAA + nAa + naa
    pA = (2 * nAA + nAa) / (2 * n)
    e = np.array([pA**2, 2*pA*(1-pA), (1-pA)**2]) * n
    o = np.array([nAA, nAa, naa])
    x2 = ((o - e) ** 2 / e).sum()
    return pA, e, x2, stats.chi2.sf(x2, 1), 1 - nAa / e[1]


for lab, obs in (("equilibrio", (298, 489, 213)), ("deficit", (350, 380, 270))):
    pA, e, x2, pv, F = hwe_chi2(*obs)
    say(f"[hwe-{lab}] obs={obs} pA={pA:.4f} esperados="
        f"{e[0]:.1f},{e[1]:.1f},{e[2]:.1f} chi2={x2:.3f} p={pv:.3g} F={F:.3f}")


def hwe_exact(n_het, n_hom1, n_hom2):
    """Prueba exacta de Wigginton et al. (2005): valor p bilateral."""
    n_rare = 2 * min(n_hom1, n_hom2) + n_het
    n = n_het + n_hom1 + n_hom2
    probs = np.zeros(n_rare + 1)
    mid = n_rare * (2 * n - n_rare) // (2 * n)
    if (mid % 2) != (n_rare % 2):
        mid += 1
    probs[mid] = 1.0
    hr = (n_rare - mid) // 2
    hc = n - mid - hr
    h = mid
    while h >= 2:
        probs[h - 2] = probs[h] * h * (h - 1) / (4 * (hr + 1) * (hc + 1))
        h -= 2; hr += 1; hc += 1
    hr = (n_rare - mid) // 2
    hc = n - mid - hr
    h = mid
    while h <= n_rare - 2:
        probs[h + 2] = probs[h] * 4 * hr * hc / ((h + 2) * (h + 1))
        h += 2; hr -= 1; hc -= 1
    probs /= probs.sum()
    return probs[probs <= probs[n_het] + 1e-12].sum(), probs


# muestra pequeña con alelo raro: 100 individuos, 2 aa, 7 Aa, 91 AA
pex, probs = hwe_exact(7, 91, 2)
pA, e, x2, pv, F = hwe_chi2(91, 7, 2)
say(f"[hwe-exacta] n=100 AA=91 Aa=7 aa=2: q={1-pA:.3f} esperados="
    f"{e[0]:.2f},{e[1]:.2f},{e[2]:.2f} chi2={x2:.3f} p_chi2={pv:.3g} "
    f"p_exacta={pex:.4f}")
nz = [(k, probs[k]) for k in range(len(probs)) if probs[k] > 0]
say("[hwe-exacta] P(nAB) =", ", ".join(f"{k}:{v:.4f}" for k, v in nz))
w("hwe_exacta.dat", "het prob obs\n" + "\n".join(
    f"{k} {v:.5f} {1 if k == 7 else 0}" for k, v in nz))

# =====================================================================
# 10.1  Wright-Fisher: genealogía didáctica (2N = 8 copias, 7 generaciones)
# =====================================================================


def genealogia(seed):
    r = np.random.default_rng(seed)
    G, K = 10, 8
    padres = r.integers(0, K, size=(G - 1, K))  # padres[g][i]: padre en g
    alelo = np.zeros((G, K), int)
    alelo[0, :4] = 1
    for g in range(1, G):
        alelo[g] = alelo[g - 1][padres[g - 1]]
    # linajes ancestrales de la generación actual
    anc = [set() for _ in range(G)]
    anc[G - 1] = set(range(K))
    for g in range(G - 1, 0, -1):
        anc[g - 1] = {padres[g - 1][i] for i in anc[g]}
    return G, K, padres, alelo, anc


best = None
for seed in range(400):
    G, K, pad, al, anc = genealogia(seed)
    fr = al.mean(1)
    nanc = [len(a) for a in anc]
    tfix = next((g for g in range(G) if fr[g] in (0.0, 1.0)), G)
    if nanc[0] == 1 and 6 <= tfix <= 8 and len(set(fr[:6])) >= 4:
        best = seed
        break
G, K, pad, al, anc = genealogia(best)
dx, dy = 0.78, 0.66
lines = []
for g in range(G - 1):
    for i in range(K):
        j = pad[g][i]
        on = i in anc[g + 1]
        st = "azulnoche, line width=1.3pt" if on else "base, thin"
        lines.append(rf"\draw[{st}] ({j*dx:.2f},{-g*dy:.2f}) -- "
                     rf"({i*dx:.2f},{-(g+1)*dy:.2f});")
for g in range(G):
    for i in range(K):
        col = "azul" if al[g, i] else "naranja"
        ring = ",draw=azulnoche,line width=1pt" if i in anc[g] else ",draw=white"
        lines.append(rf"\fill[{col}{ring}] ({i*dx:.2f},{-g*dy:.2f}) circle (1.3mm);")
    val = f"{al[g].mean():.3f}".replace(".", "{,}")
    lines.append(rf"\node[etiqueta,anchor=west] at ({K*dx-0.25:.2f},{-g*dy:.2f})"
                 rf" {{$\hat p_{{{g}}}={val}$}};")
    lines.append(rf"\node[etiqueta,anchor=east] at (-0.35,{-g*dy:.2f}) {{$t={g}$}};")
w("genealogia.tex", "\n".join(lines) + "\n")
say(f"[genealogia] semilla={best} p_t=" + ",".join(f"{x:.3f}" for x in al.mean(1))
    + " linajes=" + ",".join(str(len(a)) for a in anc))

# =====================================================================
# 10.1  Trayectorias Wright-Fisher
# =====================================================================


def wf(N, p0, T, reps, s=0.0, r=rng):
    X = np.zeros((T + 1, reps))
    X[0] = p0
    for t in range(T):
        x = X[t]
        xs = x * (1 + s) / (1 + s * x)
        X[t + 1] = r.binomial(2 * N, xs) / (2 * N)
    return X


T = 200
for N in (10, 50, 250):
    X = wf(N, 0.5, T, 12)
    w(f"wf_N{N}.dat", "t " + " ".join(f"r{k}" for k in range(12)) + "\n" +
      "\n".join(f"{t} " + " ".join(f"{v:.3f}" for v in X[t]) for t in range(T + 1)))
    Xb = wf(N, 0.5, T, 5000)
    fijo = ((Xb[-1] == 0) | (Xb[-1] == 1)).mean()
    say(f"[wf] N={N}: fracción de réplicas absorbidas a t={T}: {fijo:.3f}")

# Heterocigosidad
Tm = 100
lines = ["t " + " ".join(f"N{N}" for N in (10, 25, 100))]
Hs = {}
for N in (10, 25, 100):
    X = wf(N, 0.5, Tm, 4000)
    Hs[N] = (2 * X * (1 - X)).mean(1)
for t in range(0, Tm + 1, 2):
    lines.append(f"{t} " + " ".join(f"{Hs[N][t]:.4f}" for N in (10, 25, 100)))
w("het.dat", "\n".join(lines))
for N in (10, 25, 100):
    say(f"[het] N={N}: H_50 sim={Hs[N][50]:.4f} teoria="
        f"{0.5*(1-1/(2*N))**50:.4f}")
say(f"[het] ejemplo N=50, t=100: H/H0={(1-1/100)**100:.4f}, "
    f"H={0.5*(1-1/100)**100:.4f}; vida media t1/2={math.log(2)*2*50:.1f} "
    f"(exacta {math.log(0.5)/math.log(1-1/100):.1f})")

# Tiempos de fijación (diploide, N individuos): Kimura-Ohta
for N in (50,):
    p0 = 0.5
    tbar = -4 * N * (p0 * math.log(p0) + (1 - p0) * math.log(1 - p0))
    Xb = wf(N, p0, 3000, 4000)
    absorbed = (Xb == 0) | (Xb == 1)
    tabs = absorbed.argmax(0)
    say(f"[fijacion] N={N} p0=0.5: E[T] teoría={tbar:.1f}, "
        f"sim={tabs.mean():.1f}; nueva mutación fijada: 4N={4*N}")

# Distribución de p_t (difusión) y probabilidad de fijación con selección
N = 50
X = wf(N, 0.5, 100, 40000)
edges = np.linspace(0, 1, 21)
lines = ["x " + " ".join(f"t{t}" for t in (5, 20, 50, 100))]
dens = {}
for t in (5, 20, 50, 100):
    v = X[t]
    inner = v[(v > 0) & (v < 1)]
    h, _ = np.histogram(inner, bins=edges)
    dens[t] = h / len(v) / 0.05
    say(f"[difusion] N=50 t={t}: P(perdido)={np.mean(v == 0):.3f} "
        f"P(fijado)={np.mean(v == 1):.3f} Var={v.var():.4f} "
        f"teoria Var={0.25*(1-(1-1/100)**t):.4f}")
for k in range(20):
    lines.append(f"{edges[k]:.2f} " + " ".join(f"{dens[t][k]:.4f}"
                                              for t in (5, 20, 50, 100)))
lines.append("1.00 " + " ".join(f"{dens[t][-1]:.4f}" for t in (5, 20, 50, 100)))
w("difusion.dat", "\n".join(lines))

N, p0 = 50, 0.1
S_vals = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6, 8]) / (4 * N)
lines = ["S u"]
for s in S_vals:
    Xs = wf(N, p0, 2500, 3000, s=s)
    u = np.mean(Xs[-1] == 1)
    lines.append(f"{4*N*s:.2f} {u:.4f}")
w("fijacion_sim.dat", "\n".join(lines))
say("[fijacion-sel] N=50 p0=0.1 (4Ns, u_sim): " + "; ".join(lines[1:]))
for S4 in (0, 2, 8):
    s = S4 / (4 * N)
    u = p0 if s == 0 else (1 - math.exp(-4 * N * s * p0)) / (1 - math.exp(-4 * N * s))
    say(f"[fijacion-sel] teoria 4Ns={S4}: u={u:.4f}")
# nueva mutación con s = 0.01, N = 1e4
Nn, s = 10000, 0.01
u = (1 - math.exp(-2 * s)) / (1 - math.exp(-4 * Nn * s))
say(f"[kimura] N=1e4 s=0.01 p=1/2N: u={u:.5f} (aprox 2s={2*s}); neutral 1/2N={1/(2*Nn):.1e}")

# Coalescente
for n in (2, 10, 100):
    say(f"[coalescente] n={n}: E[T_MRCA]=4N(1-1/n) = {4*(1-1/n):.3f} N; "
        f"E[L]=4N*sum 1/i = {4*sum(1/i for i in range(1, n)):.3f} N")

# =====================================================================
# 10.2  Desequilibrio de ligamiento
# =====================================================================
# ejemplo numérico de D, D', r2
pAB, pAb, paB, pab = 0.50, 0.10, 0.05, 0.35
pA_, pB_ = pAB + pAb, pAB + paB
D = pAB - pA_ * pB_
Dmax = min(pA_ * (1 - pB_), (1 - pA_) * pB_) if D > 0 else \
    min(pA_ * pB_, (1 - pA_) * (1 - pB_))
r2 = D * D / (pA_ * (1 - pA_) * pB_ * (1 - pB_))
say(f"[ld-ejemplo] pA={pA_:.2f} pB={pB_:.2f} D={D:.4f} Dmax={Dmax:.4f} "
    f"D'={D/Dmax:.4f} r2={r2:.4f} check={pAB*pab-pAb*paB:.4f}")
for c, t in ((0.5, 5), (0.01, 50), (0.001, 100)):
    say(f"[ld-decay] c={c} t={t}: (1-c)^t={(1-c)**t:.4f}")
for Ne in (1e3, 1e4):
    for d in (1e4, 1e5):
        c = 1e-8 * d
        say(f"[ld-sved] Ne={Ne:.0e} d={d:.0e}: E[r2]={1/(1+4*Ne*c):.4f}")

t = np.arange(0, 201)
w("ld_decay_t.dat", "t c05 c01 c001 c0001\n" + "\n".join(
    f"{k} {(0.5)**k:.5f} {(0.9)**k:.5f} {(0.99)**k:.5f} {(0.999)**k:.5f}"
    for k in t))
dkb = np.logspace(-1, 3, 81)
w("ld_decay_d.dat", "kb Ne1e3 Ne1e4 Ne1e5\n" + "\n".join(
    f"{d:.4f} " + " ".join(f"{1/(1+4*Ne*1e-8*d*1e3):.5f}" for Ne in (1e3, 1e4, 1e5))
    for d in dkb))

# Haplotipos por mosaico con puntos calientes de recombinación
nS, nH, Kf = 70, 600, 7
pos = np.sort(rng.uniform(0, 200, nS))           # kb
hot = np.array([48.0, 105.0, 160.0])
# bloques delimitados por los puntos calientes; en cada bloque, Kf haplotipos
# fundadores relacionados por un árbol (las mutaciones caen en sus ramas)
bloque = np.searchsorted(hot, pos)
founders = np.zeros((Kf, nS), int)
for bq in range(len(hot) + 1):
    clados = [{i} for i in range(Kf)]
    vivos = [{i} for i in range(Kf)]
    while len(vivos) > 1:
        i, j = sorted(rng.choice(len(vivos), 2, replace=False))
        nuevo = vivos[i] | vivos[j]
        vivos = [v for k, v in enumerate(vivos) if k not in (i, j)] + [nuevo]
        if len(nuevo) < Kf:
            clados.append(nuevo)
    for jx in np.where(bloque == bq)[0]:
        cl = clados[rng.integers(len(clados))]
        founders[list(cl), jx] = 1
peso = rng.dirichlet(np.full(Kf, 1.5))
gap = np.diff(pos)
pswitch = 1 - np.exp(-0.0015 * gap)              # fondo: 0,0015 por kb
for h in hot:
    inside = (pos[:-1] <= h) & (pos[1:] > h)
    pswitch[inside] = 0.7                        # punto caliente
H = np.zeros((nH, nS), int)
for i in range(nH):
    f = rng.choice(Kf, p=peso)
    H[i, 0] = founders[f, 0]
    for j in range(1, nS):
        if rng.random() < pswitch[j - 1]:
            f = rng.choice(Kf, p=peso)
        H[i, j] = founders[f, j]
H ^= (rng.random(H.shape) < 0.002).astype(int)    # mutaciones recurrentes
fr = H.mean(0)
keep = (fr > 0.08) & (fr < 0.92)
H, pos = H[:, keep], pos[keep]
nS = H.shape[1]


def ld(h1, h2):
    pa, pb = h1.mean(), h2.mean()
    pab_ = (h1 & h2).mean()
    D = pab_ - pa * pb
    if D >= 0:
        dm = min(pa * (1 - pb), (1 - pa) * pb)
    else:
        dm = min(pa * pb, (1 - pa) * (1 - pb))
    return abs(D) / dm, D * D / (pa * (1 - pa) * pb * (1 - pb))


Dp = np.zeros((nS, nS)); R2 = np.zeros((nS, nS))
for i in range(nS):
    for j in range(i + 1, nS):
        Dp[i, j], R2[i, j] = ld(H[:, i], H[:, j])
say(f"[ld-sim] SNPs={nS} haplotipos={nH} r2 medio={R2[np.triu_indices(nS,1)].mean():.3f} "
    f"D' medio={Dp[np.triu_indices(nS,1)].mean():.3f}")


def cmap(colors):
    return LinearSegmentedColormap.from_list("c", colors)


cm_r2 = cmap(["#FCFCFB", "#CDE2FB", "#86B6EF", "#2A78D6", "#1C5CAB", "#0D366B"])
cm_dp = cmap(["#FCFCFB", "#F3B0AE", "#E66767", "#B8302F"])
fig, axes = plt.subplots(2, 1, figsize=(5.1, 4.6),
                         gridspec_kw=dict(hspace=0.12))
for ax, M, cm, lab in ((axes[0], Dp, cm_dp, "$|D'|$"),
                       (axes[1], R2, cm_r2, "$r^2$")):
    polys, vals = [], []
    for i in range(nS):
        for j in range(i + 1, nS):
            cx, cy = (i + j) / 2, -(j - i) / 2
            polys.append([(cx, cy + 0.5), (cx + 0.5, cy), (cx, cy - 0.5),
                          (cx - 0.5, cy)])
            vals.append(M[i, j])
    pc = PolyCollection(polys, array=np.array(vals), cmap=cm,
                        edgecolors="none", clim=(0, 1))
    ax.add_collection(pc)
    # regla de posiciones físicas
    L = nS - 1
    for k in range(nS):
        xg = pos[k] / 200 * L
        ax.plot([xg, k], [2.6, 0.55], color="#C3C2B7", lw=0.35)
    ax.plot([0, L], [2.6, 2.6], color="#52514E", lw=0.8)
    for h in hot:
        ax.plot(h / 200 * L, 2.6, marker="v", color=C["naranja"], ms=5,
                clip_on=False)
    ax.set_xlim(-1, nS)
    ax.set_ylim(-nS / 2 - 0.5, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.0, shrink=0.55,
                      anchor=(0.0, 0.2))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.2f}".replace(".", ",")))
    cb.set_label(lab, fontsize=9, color="#52514E")
axes[0].text(0, 3.3, "0 kb", fontsize=7, color="#898781", ha="left")
axes[0].text(nS - 1, 3.3, "200 kb", fontsize=7, color="#898781", ha="right")
axes[0].text(-1, -nS / 4, "a", fontsize=11, fontweight="bold", color="#0B0B0B")
axes[1].text(-1, -nS / 4, "b", fontsize=11, fontweight="bold", color="#0B0B0B")
fig.savefig(OUT / "ld_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# =====================================================================
# 10.2  PCA de tres poblaciones (Balding-Nichols) + mezclados
# =====================================================================


def balding_nichols(pa, F, r=rng):
    a = pa * (1 - F) / F
    b = (1 - pa) * (1 - F) / F
    return r.beta(a, b)


Mloc = 6000
panc = rng.uniform(0.05, 0.95, Mloc)
pP = {"A": balding_nichols(panc, 0.02), "B": balding_nichols(panc, 0.02),
      "C": balding_nichols(panc, 0.06)}
nper = 130
Gs, grp, alfa = [], [], []
for k in "ABC":
    Gs.append(rng.binomial(2, pP[k], size=(nper, Mloc)))
    grp += [k] * nper
    alfa += [np.nan] * nper
nmix = 60
a_mix = rng.uniform(0.05, 0.95, nmix)
for a in a_mix:
    pm = a * pP["A"] + (1 - a) * pP["B"]
    Gs.append(rng.binomial(2, pm, size=(1, Mloc)))
    grp.append("M")
    alfa.append(a)
Gm = np.vstack(Gs).astype(float)
grp = np.array(grp)
ph = Gm.mean(0) / 2
ok = (ph > 0.01) & (ph < 0.99)
Z = (Gm[:, ok] - 2 * ph[ok]) / np.sqrt(2 * ph[ok] * (1 - ph[ok]))
U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
ev = Sv ** 2 / Z.shape[1]
pcs = U[:, :2] * Sv[:2] / math.sqrt(Z.shape[1])
if pcs[grp == "C", 0].mean() < 0:
    pcs[:, 0] *= -1
if pcs[grp == "A", 1].mean() < 0:
    pcs[:, 1] *= -1
for k in "ABCM":
    sel = grp == k
    w(f"pca_{k}.dat", "pc1 pc2\n" + "\n".join(
        f"{x:.3f} {y:.3f}" for x, y in pcs[sel]))
w("pca_ev.dat", "k ev\n" + "\n".join(f"{i+1} {e:.3f}" for i, e in enumerate(ev[:10])))
say(f"[pca] n={Gm.shape[0]} M={ok.sum()} autovalores top5=" +
    ",".join(f"{e:.2f}" for e in ev[:5]) +
    f" var.expl PC1={ev[0]/ev.sum():.4f} PC2={ev[1]/ev.sum():.4f}")
r_mix = np.corrcoef(a_mix, pcs[grp == "M", 1])[0, 1]
say(f"[pca] correlación entre fracción de ancestría A y PC2 en mezclados: {r_mix:.3f}")


def wc_fst(G1, G2):
    """Weir & Cockerham (1984), dos poblaciones, genotipos 0/1/2."""
    r = 2
    n = np.array([G1.shape[0], G2.shape[0]], float)
    ps = np.vstack([G1.mean(0) / 2, G2.mean(0) / 2])
    hs = np.vstack([(G1 == 1).mean(0), (G2 == 1).mean(0)])
    nbar = n.mean()
    nc = (r * nbar - (n ** 2).sum() / (r * nbar)) / (r - 1)
    pbar = (n[:, None] * ps).sum(0) / (r * nbar)
    s2 = (n[:, None] * (ps - pbar) ** 2).sum(0) / ((r - 1) * nbar)
    hbar = (n[:, None] * hs).sum(0) / (r * nbar)
    a = nbar / nc * (s2 - (pbar * (1 - pbar) - (r - 1) / r * s2 - hbar / 4)
                     / (nbar - 1))
    b = nbar / (nbar - 1) * (pbar * (1 - pbar) - (r - 1) / r * s2
                             - (2 * nbar - 1) / (4 * nbar) * hbar)
    c = hbar / 2
    return a.sum() / (a + b + c).sum()


GA, GB, GC = (Gm[grp == k].astype(int) for k in "ABC")
say(f"[fst] WC84: A-B={wc_fst(GA, GB):.4f} A-C={wc_fst(GA, GC):.4f} "
    f"B-C={wc_fst(GB, GC):.4f}")

# =====================================================================
# 10.3  GWAS: estratificación con genotipos reales simulados
# =====================================================================
n1 = n2 = 1000
Mg = 10000
pa = rng.uniform(0.05, 0.95, Mg)
p1, p2 = balding_nichols(pa, 0.02), balding_nichols(pa, 0.02)
G = np.vstack([rng.binomial(2, p1, (n1, Mg)), rng.binomial(2, p2, (n2, Mg))]
              ).astype(float)
pop = np.r_[np.zeros(n1), np.ones(n2)]
causal = rng.choice(Mg, 10, replace=False)
beta = rng.normal(0, 0.12, 10)
Gc = G - G.mean(0)
y = Gc[:, causal] @ beta + 0.3 * pop + rng.normal(0, 1, n1 + n2)


def assoc(y, G):
    """Regresión lineal marginal vectorizada; devuelve t y p."""
    n = len(y)
    Gc = G - G.mean(0)
    yc = y - y.mean()
    sxx = (Gc ** 2).sum(0)
    b = Gc.T @ yc / sxx
    res = (yc ** 2).sum() - b ** 2 * sxx
    se = np.sqrt(res / (n - 2) / sxx)
    tt = b / se
    return tt, 2 * stats.t.sf(np.abs(tt), n - 2)


def residualize(A, Q):
    return A - Q @ (Q.T @ A)


t_naive, p_naive = assoc(y, G)
Zs = (G - G.mean(0)) / (G.std(0) + 1e-12)
Uq, Sq, _ = np.linalg.svd(Zs, full_matrices=False)
Q = np.column_stack([np.ones(len(y)) / math.sqrt(len(y)), Uq[:, :2]])
Q, _ = np.linalg.qr(Q)
yr = residualize(y[:, None], Q)[:, 0]
Gr = residualize(G, Q)
t_pc, p_pc = assoc(yr, Gr)


def lam(pv):
    return np.median(stats.chi2.isf(pv, 1)) / stats.chi2.ppf(0.5, 1)


say(f"[estrat] n={len(y)} M={Mg}: lambda ingenuo={lam(p_naive):.3f} "
    f"lambda con 2 PCs={lam(p_pc):.3f}; chi2 mediana teórica="
    f"{stats.chi2.ppf(0.5,1):.4f}")
say(f"[estrat] significativos p<5e-8: ingenuo={np.sum(p_naive<5e-8)} "
    f"(causales {np.sum(p_naive[causal]<5e-8)}), PCs={np.sum(p_pc<5e-8)} "
    f"(causales {np.sum(p_pc[causal]<5e-8)}); p<1e-3 ingenuo={np.sum(p_naive<1e-3)} "
    f"PCs={np.sum(p_pc<1e-3)} (esperado {Mg*1e-3:.0f})")
say(f"[estrat] corr(PC1, población)={abs(np.corrcoef(Uq[:,0], pop)[0,1]):.3f}")


def qq_rows(pv, keep_top=300, n_rest=500):
    pv = np.sort(pv)
    m = len(pv)
    e = -np.log10((np.arange(1, m + 1) - 0.5) / m)
    o = -np.log10(pv)
    idx = list(range(min(keep_top, m)))
    rest = np.unique(np.round(np.geomspace(keep_top + 1, m, n_rest)).astype(int) - 1)
    idx = sorted(set(idx) | set(rest.tolist()))
    return e[idx], o[idx]


for name, pv in (("naive", p_naive), ("pc", p_pc)):
    e, o = qq_rows(pv)
    w(f"qq_{name}.dat", "e o\n" + "\n".join(f"{a:.4f} {b:.4f}" for a, b in zip(e, o)))

# banda de confianza del 95 % (estadísticos de orden beta)


def banda(m, npts=120):
    ks = np.unique(np.round(np.geomspace(1, m, npts)).astype(int))
    lo = stats.beta.ppf(0.025, ks, m - ks + 1)
    hi = stats.beta.ppf(0.975, ks, m - ks + 1)
    e = -np.log10((ks - 0.5) / m)
    return "e lo hi\n" + "\n".join(
        f"{a:.4f} {-math.log10(h):.4f} {-math.log10(l):.4f}"
        for a, l, h in zip(e, lo, hi))


w("qq_banda10k.dat", banda(Mg))

# =====================================================================
# 10.3  Manhattan simulado (estadísticos z con LD por bloques)
# =====================================================================
chrlen = np.array([248, 242, 198, 190, 181, 171, 159, 145, 138, 134, 135,
                   133, 114, 107, 102, 90, 83, 80, 59, 64, 47, 51], float)
Mtot = 90000
nper_chr = np.round(chrlen / chrlen.sum() * Mtot).astype(int)
señales = {2: (0.35, 8.8), 6: (0.18, 12.5), 9: (0.62, 7.1), 11: (0.40, 6.2),
           12: (0.55, 5.1), 16: (0.30, 7.8), 19: (0.45, 5.5), 3: (0.7, 4.8)}
z_all, chr_all, pos_all = [], [], []
for c in range(1, 23):
    m = nper_chr[c - 1]
    rho = np.where(rng.random(m) < 0.12, rng.uniform(0, 0.3, m),
                   rng.uniform(0.85, 0.995, m))
    z = np.empty(m)
    z[0] = rng.normal()
    eps = rng.normal(size=m)
    for j in range(1, m):
        z[j] = rho[j] * z[j - 1] + math.sqrt(1 - rho[j] ** 2) * eps[j]
    if c in señales:
        fpos, mu = señales[c]
        k = int(fpos * m)
        lr = np.cumsum(np.log(np.maximum(rho, 1e-12)))
        corr = np.exp(-np.abs(lr - lr[k]))
        z += mu * corr
    z_all.append(z)
    chr_all.append(np.full(m, c))
    pos_all.append(np.sort(rng.uniform(0, chrlen[c - 1], m)))
z_all = np.concatenate(z_all)
chr_all = np.concatenate(chr_all)
pos_all = np.concatenate(pos_all)
pv_m = 2 * stats.norm.sf(np.abs(z_all))
lp = -np.log10(pv_m)
say(f"[manhattan] M={len(pv_m)} lambda={lam(pv_m):.3f} p<5e-8: {np.sum(pv_m<5e-8)} "
    f"SNPs en {len(set(chr_all[pv_m<5e-8]))} loci-cromosomas; "
    f"p<1e-5: {np.sum(pv_m<1e-5)}; min p={pv_m.min():.2e}; "
    f"Bonferroni(M)={0.05/len(pv_m):.2e}")
loci = sorted(set(chr_all[pv_m < 5e-8]))
say(f"[manhattan] cromosomas con señal significativa: {loci}")
for c in sorted(señales):
    sel = chr_all == c
    say(f"[manhattan] chr{c}: min p={pv_m[sel].min():.2e} mu={señales[c][1]}")

off = np.r_[0, np.cumsum(chrlen)[:-1]] + np.arange(22) * 12
xg = pos_all + off[chr_all - 1]
fig, ax = plt.subplots(figsize=(5.1, 2.35))
cols = np.where(chr_all % 2 == 1, C["azul"], C["azulclaro"])
sig = pv_m < 5e-8
ax.scatter(xg[~sig], lp[~sig], s=1.2, c=cols[~sig], lw=0, rasterized=True)
ax.scatter(xg[sig], lp[sig], s=2.4, c=C["naranja"], lw=0, rasterized=True)
ax.axhline(-math.log10(5e-8), color=C["rojo"], lw=0.7, ls=(0, (4, 2)))
ax.axhline(5, color=C["gris"], lw=0.6, ls=(0, (1, 2)))
ax.text(xg.max() + 20, -math.log10(5e-8), r"$5\times10^{-8}$", fontsize=7,
        color=C["rojooscuro"], va="center")
ax.text(xg.max() + 20, 5, r"$10^{-5}$", fontsize=7, color=C["gris"], va="center")
mids = off + chrlen / 2
ax.set_xticks(mids)
ax.set_xticklabels([str(i) if i <= 12 or i % 2 == 0 else "" for i in range(1, 23)],
                   fontsize=7)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-20, xg.max() + 20)
ax.set_ylim(0, lp.max() * 1.06)
ax.set_xlabel("cromosoma")
ax.set_ylabel(r"$-\log_{10} p$")
ax.grid(axis="y", color=C["rejilla"], lw=0.5)
ax.set_axisbelow(True)
fig.savefig(OUT / "manhattan.pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.close(fig)

e, o = qq_rows(pv_m, keep_top=400, n_rest=600)
w("qq_manhattan.dat", "e o\n" + "\n".join(f"{a:.4f} {b:.4f}" for a, b in zip(e, o)))
w("qq_banda90k.dat", banda(len(pv_m)))

# =====================================================================
# 10.3  Bonferroni frente a Benjamini-Hochberg y potencia
# =====================================================================
m, m1, alpha = 200, 20, 0.05
zt = np.r_[rng.normal(3.3, 1, m1), rng.normal(0, 1, m - m1)]
pt = 2 * stats.norm.sf(np.abs(zt))
es_real = np.r_[np.ones(m1, bool), np.zeros(m - m1, bool)]
o = np.argsort(pt)
ps, reales = pt[o], es_real[o]
k = np.arange(1, m + 1)
bh_ok = np.where(ps <= alpha * k / m)[0]
kbh = bh_ok.max() + 1 if len(bh_ok) else 0
nbon = np.sum(ps <= alpha / m)
say(f"[bh] m={m} verdaderos={m1}: Bonferroni={nbon} (falsos {np.sum(~reales[:nbon])}); "
    f"BH={kbh} (falsos {np.sum(~reales[:kbh])}); p<0.05 sin corregir="
    f"{np.sum(ps<0.05)} (falsos {np.sum(~reales[ps<0.05])})")
w("bh.dat", "k p real\n" + "\n".join(
    f"{i+1} {ps[i]:.3e} {int(reales[i])}" for i in range(60)))
# ejemplo pequeño de BH (10 valores p)
ej = np.array([0.0001, 0.0008, 0.0021, 0.0042, 0.0110, 0.0290, 0.0480,
               0.0730, 0.2100, 0.6200])
thr = 0.05 * np.arange(1, 11) / 10
say("[bh-ejemplo] umbrales: " + ", ".join(f"{t:.3f}" for t in thr) +
    f"; kmax={np.where(ej <= thr)[0].max()+1}; bonferroni={np.sum(ej <= 0.005)}")
say(f"[bonferroni] 0.05/1e6={0.05/1e6:.1e}; P(ningún FP) 1e6 tests a 0.05:"
    f" esperados FP={1e6*0.05:.0f}")

thr_chi = stats.chi2.isf(5e-8, 1)
nn = np.logspace(3, 6, 61)
lines = ["n q0005 q001 q002 q005"]
for n in nn:
    row = [n]
    for q2 in (0.0005, 0.001, 0.002, 0.005):
        ncp = n * q2 / (1 - q2)
        row.append(stats.ncx2.sf(thr_chi, 1, ncp))
    lines.append(" ".join(f"{v:.5g}" for v in row))
w("potencia.dat", "\n".join(lines))
say(f"[potencia] umbral chi2(5e-8)={thr_chi:.2f}")
for n, q2 in ((10000, 0.001), (50000, 0.001), (100000, 0.0005), (500000, 0.0005)):
    ncp = n * q2 / (1 - q2)
    say(f"[potencia] n={n} q2={q2}: ncp={ncp:.1f} potencia="
        f"{stats.ncx2.sf(thr_chi, 1, ncp):.3f}")
for q2 in (0.0005, 0.001):
    f_ = lambda n: stats.ncx2.sf(thr_chi, 1, n * q2 / (1 - q2)) - 0.8
    lo, hi = 1e3, 1e7
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if f_(mid) < 0 else (lo, mid)
    say(f"[potencia] n para 80% con q2={q2}: {hi:,.0f}")
# varianza explicada por un SNP aditivo
for pp, b in ((0.3, 0.05), (0.1, 0.1)):
    say(f"[q2] p={pp} beta={b} DE: q2=2p(1-p)b^2={2*pp*(1-pp)*b*b:.5f}")

(OUT / "cifras.txt").write_text("\n".join(CIFRAS) + "\n")
